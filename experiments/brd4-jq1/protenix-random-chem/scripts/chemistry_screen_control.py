"""Allocated focused checks, finite native probes and measured dispatch."""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import statistics
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from chemistry_screen_ops import capacity, read, require, submit, write


def controls(plan, plan_path):
    root = Path(plan['root']) / 'controls'; root.mkdir(exist_ok=False)
    source = Path(plan['source_root'])
    names = plan.get('control_tests', ['test_protenix_chemistry_screen.py', 'test_chemistry_severity.py'])
    require(names and all(Path(name).name == name and name.endswith('.py') for name in names),
            'Controls must be named tests inside this source snapshot')
    argv = [sys.executable, '-m', 'pytest', *[str(source / 'tests' / x) for x in names],
            '-q', '-o', 'addopts=', '-p', 'no:cacheprovider', '--junitxml=' + str(root / 'tests.xml')]
    result = subprocess.run(argv, capture_output=True, text=True)
    write(root / 'tests-command.json', dict(argv=argv, returncode=result.returncode))
    (root / 'tests-stdout.log').write_text(result.stdout); (root / 'tests-stderr.log').write_text(result.stderr)
    require(result.returncode == 0, 'Focused allocated controls failed')
    xml = ET.fromstring((root / 'tests.xml').read_bytes())
    require(not any(list(xml.iter(tag)) for tag in ('failure', 'error', 'skipped')), 'Incomplete allocated controls')
    startups = []
    for arm in plan['arms']:
        argv = [sys.executable, str(source / 'protenix_chemistry_screen.py'), '--plan', str(plan_path), '--arm', arm, '--startup-only']
        result = subprocess.run(argv, capture_output=True, text=True)
        startups.append(dict(arm=arm, argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        write(root / 'startups.json', startups)
        require(result.returncode == 0, 'Actual metadata startup failed: ' + arm)
    write(root / 'assessment.json', dict(status='passed', cases=len(list(xml.iter('testcase'))),
          startups=len(startups), job_id=os.environ['LSB_JOBID'], numerical_scope='small allocated synthetic controls; no native models'))


def probe(plan, plan_path, internal_seconds):
    root = Path(plan['root']); output = root / 'probe'; output.mkdir(exist_ok=False)
    controls_path = root / 'controls' / 'assessment.json'
    if plan.get('controls_assessment'):
        from protenix_chemistry_screen import bound
        controls_path = bound(plan['controls_assessment'])
    require(read(controls_path)['status'] == 'passed', 'Allocated controls incomplete')
    started = time.monotonic(); results = []
    for arm in plan['probe_arms']:
        argv = [sys.executable, str(Path(plan['source_root']) / 'protenix_chemistry_screen.py'),
                '--plan', str(plan_path), '--arm', arm, '--smoke-updates', '25',
                '--internal-seconds', str(int(internal_seconds - (time.monotonic() - started)))]
        from protenix_chemistry_screen import module
        routing = module(Path(plan['routing_source'])/'scripts/cuda_ownership.py', '_probe_cuda_metadata')
        visible = routing.resolve_visibility(os.environ)['selected_cuda_visible_devices']
        monitor = None
        with (output/(arm+'.gpu-memory.csv')).open('x') as memory, (output/(arm+'.gpu-memory.stderr')).open('x') as memory_error:
            try:
                monitor = subprocess.Popen(['nvidia-smi','--id='+visible,
                    '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','-lms','250'],
                    stdout=memory, stderr=memory_error)
                with (output / (arm + '.stdout.log')).open('x') as out, (output / (arm + '.stderr.log')).open('x') as err:
                    result = subprocess.run(argv, stdout=out, stderr=err, timeout=max(1, internal_seconds - (time.monotonic() - started)))
            finally:
                if monitor is not None:
                    monitor.terminate()
                    try: monitor.wait(timeout=10)
                    except subprocess.TimeoutExpired: monitor.kill(); monitor.wait(timeout=5)
        require(result.returncode == 0, 'Native probe failed: ' + arm)
        parent = output / arm
        summary = read(parent / 'summary.json')
        require(summary['status'] == 'completed_exploratory_updates' and summary['applied_updates'] == 25
                and summary['probe_not_adopted'] is True and summary['frozen_before'] == summary['frozen_after'], 'Incomplete finite native probe')
        updates = [json.loads(line) for line in (parent / 'study' / 'updates.jsonl').read_text().splitlines()]
        require(len(updates) == 25 and [r['step'] for r in updates] == list(range(25)), 'Probe update prefix differs')
        warm = [r['optimization_seconds'] for r in updates[5:]]
        require(all(math.isfinite(v) and v > 0 for v in warm), 'Invalid measured timing')
        memory_values = [int(line.split(',')[0]) for line in (output/(arm+'.gpu-memory.csv')).read_text().splitlines()
                         if line.split(',')[0].strip().isdigit()]
        results.append(dict(arm=arm, status='finite_native_probe_completed', warm_seconds_max=max(warm),
                            warm_seconds_median=statistics.median(warm),
                            sampled_peak_memory_MiB=max(memory_values) if memory_values else None,
                            memory_observation='250ms telemetry; not an absolute peak guarantee',
                            setup_and_probe_seconds=summary['elapsed_seconds']+summary['setup_elapsed_seconds'],
                            job_id=os.environ['LSB_JOBID'], derivative_qualified=False, states_adopted=False))
        write(output / 'results.json', results)
    write(output / 'assessment.json', dict(status='passed', job_id=os.environ['LSB_JOBID'], results=results,
          scope='finite attached implementation paths and timing; not precise derivative, binding or chemical efficacy certification'))


def dispatch(plan, plan_path):
    root = Path(plan['root'])
    write(root / 'dispatch-started.json', dict(job_id=os.environ['LSB_JOBID']), exclusive=True)
    assessment = read(root / 'probe' / 'assessment.json')
    require(assessment['status'] == 'passed', 'No complete native implementation probe')
    rows = {r['arm']: r for r in assessment['results']}
    budgets = {}; oversized = {}
    for arm, row in plan['arms'].items():
        measurement = rows[row['timing_probe']]
        # A conservative warm upper sample plus25%, setup and cleanup. No old
        # full-reference budgets or arbitrary inflated23-hour request.
        seconds = measurement['warm_seconds_max'] * row['screen']['updates'] * 1.25 + 1800
        minutes = max(45, math.ceil(seconds / 300) * 5)
        if minutes > 1440:
            oversized[arm] = dict(required_wall_minutes=minutes, reason='Measured full arm exceeds V10024h; needs device/runtime review')
            continue
        budgets[arm] = dict(wall_minutes=minutes, measurement=measurement, margin=1.25, setup_cleanup_seconds=1800)
    write(root / 'measured-budgets.json', budgets)
    write(root / 'runtime-blocked-arms.json', oversized)
    for arm in budgets:
        submit(plan_path, 'run', arm, budgets[arm]['wall_minutes'])
    write(root / 'dispatch-complete.json', dict(status='all_requested_arms_submitted' if not oversized else 'fitting_arms_submitted_oversized_arms_need_review',
          jobs=read(root / 'jobs.json'), runtime_blocked=oversized))


def observed_job(job):
    from chemistry_screen_ops import command
    value = command(['bjobs', '-json', '-o', 'jobid user stat queue job_name command', str(job)])
    require(value['returncode'] == 0, 'Cannot authenticate job state')
    rows = json.loads(value['stdout'])['RECORDS']
    require(len(rows) == 1 and rows[0].get('JOBID') == str(job) and 'STAT' in rows[0], 'Unknown job outcome')
    return rows[0]


def unstarted_original(old):
    from chemistry_screen_ops import command
    from protenix_chemistry_screen import bound
    sub = read(bound(old['submission']))
    row = observed_job(sub['job_id'])
    argv = sub['argv']; expected = argv[argv.index('bash'):]
    require(row['USER'] == 's234818' and row['QUEUE'] == 'gpuv100'
            and row['JOB_NAME'] == sub['job_name'] and shlex.split(row['COMMAND']) == expected,
            'Original writer identity differs')
    history = command(['bhist', '-l', sub['job_id']])
    require(history['returncode'] == 0 and not any(token in history['stdout'] for token in
            ('Dispatched', 'Started on', 'Running with', 'Done successfully', 'Exited with')),
            'Original has executed; preserve it')
    require(not Path(old['payload']).exists(), 'Original saved output exists')
    return row, history


def measured_wall(measurement, updates):
    seconds = measurement['warm_seconds_max'] * updates * 1.25 + 1800
    require(math.isfinite(seconds) and seconds > 1800, 'Invalid measured device timing')
    return max(45, math.ceil(seconds / 300) * 5)


def migrate(plan, plan_path):
    """Only this fresh device's verified probe can replace its unstarted arms."""
    from chemistry_screen_ops import command
    from protenix_chemistry_screen import bound
    root = Path(plan['root'])
    write(root / 'migration-started.json', {'job_id': os.environ['LSB_JOBID']}, exclusive=True)
    sub = read(root / 'jobs/probe/submission.json')
    probe_job = sub['job_id']; observations = []
    for _ in range(12):
        row = observed_job(probe_job); observations.append(row)
        require(row['USER'] == 's234818' and row['JOB_NAME'] == sub['job_name']
                and row['QUEUE'] == plan['arms'][plan['probe_arms'][0]]['queue'], 'Probe identity differs')
        if row['STAT'] not in ('RUN', 'DONE'):
            write(root / 'migration-summary.json', {'status': 'probe_failed_originals_preserved', 'scheduler': row})
            return
        log = root / 'jobs/probe' / ('stdout.' + probe_job + '.log')
        if row['STAT'] == 'DONE' and log.is_file() and 'Successfully completed.' in log.read_text():
            require(('Subject: Job ' + probe_job + ': <' + sub['job_name'] + '>') in log.read_text(), 'Wrong probe LSF')
            break
        time.sleep(5)
    else:
        raise RuntimeError('Probe terminal evidence did not settle; original jobs preserved')
    write(root / 'probe-terminal-proof.json', observations)
    assessment = read(root / 'probe/assessment.json')
    require(assessment['status'] == 'passed' and assessment['job_id'] == probe_job, 'Incomplete device probe')
    samples = {r['arm']: r for r in assessment['results']}
    decisions = {}
    for arm in plan['production_arms']:
        parent = root / ('migration-' + arm); parent.mkdir(exist_ok=False)
        old = plan['migration'][arm]; job = read(bound(old['submission']))['job_id']
        decision = {'original_job': job}; held = retired = False
        try:
            current = observed_job(job)
            if current['STAT'] != 'PEND':
                decision.update(status='original_kept', original_state=current['STAT']); continue
            current, history = unstarted_original(old)
            write(parent / 'unstarted-before.json', {'scheduler': current, 'history': history})
            minutes = measured_wall(samples[plan['arms'][arm]['timing_probe']], plan['arms'][arm]['screen']['updates'])
            if minutes > 1440:
                decision.update(status='original_kept_device_too_slow', required_minutes=minutes); continue
            startup = command([sys.executable, str(Path(plan['source_root'])/'protenix_chemistry_screen.py'),
                               '--plan', str(plan_path), '--arm', arm, '--startup-only'], 120)
            write(parent / 'actual-startup.json', startup)
            require(startup['returncode'] == 0, 'Replacement startup failed')
            write(parent / 'capacity-before-retirement.json', capacity(plan, 'migration-' + arm))
            require(not list((root/'production'/arm).glob('submission*.json')), 'Existing replacement outcome')
            write(parent / 'retirement-intent.json', {'job_id': job, 'only_if_unstarted': True}, exclusive=True)
            stopped = command(['bstop', job]); write(parent / 'stop-response.json', stopped)
            require(stopped['returncode'] == 0, 'Original hold needs reconciliation'); held = True
            for _ in range(12):
                state = observed_job(job)['STAT']
                if state != 'PEND': break
                time.sleep(2)
            if state != 'PSUSP':
                decision.update(status='original_kept', original_state=state); continue
            current, history = unstarted_original(old)
            write(parent / 'held-unstarted.json', {'scheduler': current, 'history': history})
            killed = command(['bkill', job]); write(parent / 'retirement-response.json', killed)
            require(killed['returncode'] == 0, 'Retirement outcome unknown')
            for _ in range(12):
                if observed_job(job)['STAT'] == 'EXIT': break
                time.sleep(2)
            require(observed_job(job)['STAT'] == 'EXIT', 'Retirement not established'); retired = True
            write(parent / 'retired-unstarted.json', {'job_id': job, 'applied_updates': 0})
            successor = submit(plan_path, 'run', arm, minutes)
            decision.update(status='migrated', job_id=successor, wall_minutes=minutes)
        except BaseException as error:
            decision.update(status='needs_reconciliation', error=repr(error), original_retired=retired)
        finally:
            if held and not retired:
                write(parent/'restore-original.json', command(['bresume', job]))
            decisions[arm] = decision; write(parent/'decision.json', decision)
    write(root/'migration-summary.json', {'status': 'conditional_migration_finished', 'decisions': decisions,
                                         'scientific_settings_changed': False, 'saved_results_changed': False})


def read_committed_prefix(path, expected_count):
    """A terminal prefix must have every row, including its final newline."""
    last, count = None, 0
    with Path(path).open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n'), 'Incomplete terminal JSONL row')
            row = json.loads(line)
            require(row['global_step'] == count, 'Noncontiguous terminal JSONL prefix')
            count += 1
            require(count <= expected_count, 'Unexpected extra terminal updates/states')
            last = row
    require(count == expected_count, 'Requested terminal update count not reached')
    return last


def make_tail_plan(plan, final_state, snapshot, reference, parent_job):
    """One fresh child; only its chemistry coefficient changes at the endpoint."""
    rule = plan['conditional_tail']
    parent_name = rule['parent_arm']; original = plan['arms'][parent_name]
    offset = original['screen']['updates']
    require(type(offset) is int and offset == 50000, 'Conditional parent must complete50k')
    require(rule['target_updates'] == 70000 and rule['coefficient'] == 100., 'Unexpected conditional tail policy')
    require(type(rule['wall_minutes']) is int and 10 < rule['wall_minutes'] <= 1440,
            'Conditional tail requires a bounded measured walltime')
    require(final_state['global_step'] == offset, 'Conditional checkpoint offset differs')
    child = deepcopy(plan)
    name = rule.get('arm_name', parent_name+'-repair20k')
    require(isinstance(name, str) and name != parent_name and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name),
            'Fresh conditional arm required')
    row = deepcopy(original); row['name'] = name
    row['screen']['updates'] = rule['target_updates']
    row['screen']['chemistry'] = {'mode': 'constant', 'coefficient': rule['coefficient']}
    row['requested_wall_minutes'] = rule['wall_minutes']
    row['internal_seconds'] = rule['wall_minutes']*60-600
    row['predecessor_job_ids'] = list(dict.fromkeys(original.get('predecessor_job_ids', [])+[parent_job]))
    state = deepcopy(final_state); state['step'] = offset
    row['resume'] = {'offset': offset, 'state': state, 'snapshot': deepcopy(snapshot),
                     'reference': deepcopy(reference), 'predecessor_job_id': parent_job,
                     'policy_change': True}
    for key in ('output_cap_bytes', 'file_cap'):
        if key in rule:
            row[key] = rule[key]
    child['arms'] = {name: row}; child['production_arms'] = [name]; child['probe_arms'] = [name]
    child.pop('conditional_tail')
    child['conditional_parent'] = {'arm': parent_name, 'job_id': parent_job,
                                   'global_offset': offset, 'final_state': deepcopy(final_state)}
    return child, name


def inspect_endpoint_snapshot(point_path, target, expected_temperature, model_key):
    """Inspect saved FP32 arrays on allocated CPU without a scorer evaluation."""
    # No scorer/weights are loaded: all following operations run on this allocated CPU.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.state import RelaxedGraph, validate_graph
    from mosaic.small_molecule.io import hard_readout
    from protenix_chemistry_screen import logits_class
    require(jax.default_backend() == 'cpu', 'Conditional readout must run on allocated CPU')
    with np.load(point_path, allow_pickle=False) as point:
        require(int(point['step']) == target, 'Endpoint matrix step differs')
        require(np.array_equal(point['key_data'], np.asarray(jax.random.key_data(jax.random.key(model_key)))),
                'Endpoint model key differs')
        require(point['temperature'].dtype == np.float32
                and np.array_equal(point['temperature'], np.asarray(expected_temperature, dtype=np.float32)),
                'Endpoint represented temperature differs')
        expected_shapes = {'X': (25, 6), 'E_upper': (300, 5), 'atom_logits': (25, 6), 'bond_logits': (300, 5)}
        for name, shape in expected_shapes.items():
            require(point[name].shape == shape and point[name].dtype == np.float32
                    and np.isfinite(point[name]).all(), 'Nonfinite/incompatible endpoint '+name)
        graph = RelaxedGraph(jnp.asarray(point['X']), jnp.asarray(point['E_upper'])); validate_graph(graph)
        logits = logits_class()(jnp.asarray(point['atom_logits']), jnp.asarray(point['bond_logits']), expected_temperature)
        represented = logits.graph()
        for name in ('X', 'E_upper'):
            require(np.allclose(np.asarray(getattr(represented, name)), point[name],
                                rtol=32*np.finfo(np.float32).eps, atol=32*np.nextafter(np.float32(0), np.float32(1))),
                    'Endpoint logits/graph disagree beyond FP32 device rounding')
        readout = hard_readout(graph); readout.pop('pairs', None)
    return readout


def verify_completed_parent(plan, plan_path):
    """Allocated CPU only: authenticate completion, then inspect the saved graph."""
    from chemistry_screen_ops import command
    from protenix_chemistry_screen import binding, bound, temperature_at
    root = Path(plan['root']); rule = plan['conditional_tail']; arm = rule['parent_arm']
    settings = plan['arms'][arm]; target = settings['screen']['updates']
    require(target == 50000, 'Conditional parent target must be50k')
    parent = root/'production'/arm; sub = read(parent/'submission.json'); job = sub['job_id']
    require(sub['kind'] == 'run' and sub['arm'] == arm and sub['plan'] == binding(plan_path),
            'Parent submission/plan identity differs')
    require(read(root/'jobs.json')[arm] == job and read(parent/'release.json')['returncode'] == 0,
            'Parent accepted/released job differs')
    own = read(root/'jobs/conditional/submission.json')
    require(own['job_id'] == os.environ['LSB_JOBID'] and own['kind'] == 'conditional'
            and own['plan'] == binding(plan_path), 'Wrong conditional callback identity')
    require('-w' in own['argv'] and own['argv'][own['argv'].index('-w')+1] == 'done('+job+')',
            'Conditional callback must wait for parent success')
    observations = []
    log_path = parent/('stdout.'+job+'.log')
    for _ in range(12):
        result = command(['bjobs', '-json', '-o', 'jobid user stat queue job_name command', job])
        row = None
        if result['returncode'] == 0:
            rows = json.loads(result['stdout'])['RECORDS']
            require(isinstance(rows, list) and len(rows) <= 1, 'Ambiguous parent scheduler record')
            if rows:
                row = rows[0]
                require(row['JOBID'] == job and row['USER'] == 's234818'
                        and row['JOB_NAME'] == sub['job_name'] and row['QUEUE'] == settings['queue']
                        and shlex.split(row['COMMAND']) == sub['argv'][sub['argv'].index('bash'):],
                        'Parent scheduler ownership differs')
                require(row['STAT'] in ('RUN', 'DONE'), 'Parent did not finish successfully')
        observations.append(result)
        log = log_path.read_text() if log_path.is_file() else ''
        persisted_done = ('Subject: Job '+job+': <'+sub['job_name']+'>') in log and 'Successfully completed.' in log
        if persisted_done and (row is None or row['STAT'] == 'DONE'):
            break
        time.sleep(5)
    else:
        raise RuntimeError('Parent terminal evidence did not settle')
    summary_path = root/arm/'payload/summary.json'; summary = read(summary_path)
    require(summary['job_id'] == job and summary['arm'] == arm
            and summary['status'] == 'completed_exploratory_updates'
            and summary['applied_updates'] == target and summary['actual_steps'] == target
            and summary['full_budget_complete'] is True and summary['probe_not_adopted'] is False
            and summary['configuration'] == settings['screen'], 'Parent scientific update budget incomplete')
    require(summary['frozen_before'] == summary['frozen_after'] == settings['frozen_identity'],
            'Parent frozen model/context changed')
    for key in ('config', 'snapshot', 'reference', 'pocket', 'context_ready'):
        bound(settings[key])
    source = Path(plan['source_root'])
    for relative, digest in read(source/'checksums.json').items():
        require(binding(source/relative)['sha256'] == digest, 'Conditional source changed')
    study = root/arm/'payload/study'; state_path = study/'states.jsonl'; update_path = study/'updates.jsonl'
    final = read_committed_prefix(state_path, target+1)
    read_committed_prefix(update_path, target)
    report = read(study/'report.json')
    require(report['status'] == 'completed_exploratory_updates' and report['applied_updates'] == target
            and report['last_state'] == final, 'Parent report/prefix disagree')
    point_path = study/'snapshots'/f'step-{target:05d}.npz'; snapshot = binding(point_path)
    require(final['snapshot_retained'] is True and final['snapshot'] == str(point_path)
            and final['snapshot_sha256'] == snapshot['sha256'], 'Endpoint snapshot identity differs')
    require(final['latest_snapshot'] == binding(study/'latest-state.npz'), 'Latest checkpoint identity differs')
    reference = binding(study/'references'/f'step-{target:05d}.json.gz')
    expected_temperature = temperature_at(settings['screen'], target)
    require(math.isclose(final['temperature'], expected_temperature, rel_tol=4*2**-23, abs_tol=4*2**-149)
            and final['repair_weight'] == 30.,
            'Parent final schedule differs')
    require(final['noise_state']['next_update'] == target
            and final['noise_state']['policy'] == settings['screen']['logit_noise'], 'Parent RNG checkpoint differs')

    readout = inspect_endpoint_snapshot(point_path, target, expected_temperature, settings['model_key'])
    require(readout['element_indices'] == final['readout']['element_indices']
            and readout['bond_upper_indices'] == final['readout']['bond_upper_indices'], 'Endpoint argmax labels differ')
    valid = readout.get('sanitized') is True and readout.get('fragment_count') == 1
    require(valid == final['strict_valid_connected'], 'Endpoint validity does not reproduce')
    import gzip
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from protenix_chemistry_screen import restore_teacher
    with gzip.open(reference['path'], 'rt', encoding='utf8') as stream:
        teacher, environment = restore_teacher(json.load(stream), mosaic_spec(25))
    require(teacher.identity == final['preparation_identity']
            and (None if environment is None else environment.identity) == final['neighbor_preparation_identity'],
            'Endpoint fixed teacher differs')
    return {'parent_job_id': job, 'parent_arm': arm, 'global_step': target, 'state': final,
            'snapshot': snapshot, 'reference': reference, 'strict_valid_connected': valid, 'readout': readout,
            'parent_summary': binding(summary_path), 'states': binding(state_path), 'updates': binding(update_path),
            'same_job_lsf': binding(log_path), 'scheduler_observations': observations,
            'frozen_identity': summary['frozen_after'], 'scorer_evaluations': 0,
            'scope': 'Allocated CPU saved-endpoint validation; no derivative or binding certification'}


def conditional(plan, plan_path):
    root = Path(plan['root']); output = root/'conditional'; output.mkdir(exist_ok=False)
    write(output/'started.json', {'job_id': os.environ['LSB_JOBID']}, exclusive=True)
    try:
        proof = verify_completed_parent(plan, plan_path)
        write(output/'endpoint-verification.json', proof, exclusive=True)
        if proof['strict_valid_connected']:
            write(output/'assessment.json', {'status': 'skipped_endpoint_already_valid',
                  'parent_job_id': proof['parent_job_id'], 'global_step': proof['global_step'],
                  'optimizer_submitted': False, 'saved_results_changed': False}, exclusive=True)
            return
        child, arm = make_tail_plan(plan, proof['state'], proof['snapshot'], proof['reference'], proof['parent_job_id'])
        child_path = output/'child-plan.json'; write(child_path, child, exclusive=True); child_path.chmod(0o444)
        from chemistry_screen_ops import command
        startup = command([sys.executable, str(Path(plan['source_root'])/'protenix_chemistry_screen.py'),
                           '--plan', str(child_path), '--arm', arm, '--startup-only'], 120)
        write(output/'child-startup.json', startup, exclusive=True)
        require(startup['returncode'] == 0, 'Conditional child startup failed')
        write(output/'capacity-before-submit.json', capacity(child, 'conditional-tail'), exclusive=True)
        write(output/'submission-started.json', {'arm': arm, 'parent_job_id': proof['parent_job_id']}, exclusive=True)
        job = submit(child_path, 'run', arm, plan['conditional_tail']['wall_minutes'])
        write(output/'assessment.json', {'status': 'invalid_endpoint_tail_submitted', 'parent_job_id': proof['parent_job_id'],
              'job_id': job, 'global_start': proof['global_step'], 'global_target': child['arms'][arm]['screen']['updates'],
              'coefficient': 100., 'temperature': .4, 'noise_ratio': 0., 'saved_results_changed': False}, exclusive=True)
    except BaseException as error:
        write(output/'failure.json', {'status': 'failed_no_blind_retry', 'error': repr(error),
                                     'job_id': os.environ['LSB_JOBID']}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--kind', choices=('controls', 'probe', 'dispatch', 'migrate', 'conditional'), required=True)
    parser.add_argument('--internal-seconds', type=int, required=True)
    args = parser.parse_args(); plan = read(args.plan)
    require(os.environ.get('LSB_JOBID', '').isdigit(), 'Allocated job required')
    write(Path(plan['root']) / ('capacity-' + args.kind + '.json'), capacity(plan, args.kind))
    if args.kind == 'controls': controls(plan, args.plan)
    elif args.kind == 'probe': probe(plan, args.plan, args.internal_seconds)
    elif args.kind == 'dispatch': dispatch(plan, args.plan)
    elif args.kind == 'conditional': conditional(plan, args.plan)
    else: migrate(plan, args.plan)


if __name__ == '__main__':
    main()
