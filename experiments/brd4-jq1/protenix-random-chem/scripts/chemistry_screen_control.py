"""Allocated focused checks, finite native probes and measured dispatch."""
import argparse
import json
import math
import os
from pathlib import Path
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
    names = ['test_protenix_chemistry_screen.py', 'test_chemistry_severity.py']
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--kind', choices=('controls', 'probe', 'dispatch', 'migrate'), required=True)
    parser.add_argument('--internal-seconds', type=int, required=True)
    args = parser.parse_args(); plan = read(args.plan)
    require(os.environ.get('LSB_JOBID', '').isdigit(), 'Allocated job required')
    write(Path(plan['root']) / ('capacity-' + args.kind + '.json'), capacity(plan, args.kind))
    if args.kind == 'controls': controls(plan, args.plan)
    elif args.kind == 'probe': probe(plan, args.plan, args.internal_seconds)
    elif args.kind == 'dispatch': dispatch(plan, args.plan)
    else: migrate(plan, args.plan)


if __name__ == '__main__':
    main()
