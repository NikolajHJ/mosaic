"""Unsealed launcher-only adoption controller for two named frozen studies.

All mathematical work remains in the original authenticated scientific driver.
The allocated metadata CPU certificate never replaces its scientific CPU proof.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import adoption_contract as contract
from adoption_contract import (BASE, GIB, PROFILES, VERSION, atomic, exclusive,
                               file_record, read, require, sha)

PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
METADATA_SUITE = 'mosaic-peer-adoption-metadata-v1'
# Reviewed fixture scope; these cases remain unexecuted until allocated on HPC.
METADATA_CASES = 75
ACTIVE = None


def control_config(profile):
    require(profile in ('boltz30k', 'dbmol10k'), 'Unknown adoption consumer')
    return PROFILES[profile]


def resources(phase):
    if phase == 'cpu':
        return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']
    if phase in ('native', 'segment'):
        return ['-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R',
                'rusage[mem=8GB]', '-gpu', 'num=1:mode=exclusive_process',
                '-W', '02:00' if phase == 'native' else '23:00']
    require(phase in ('after-cpu', 'observe', 'dispatch', 'after-native', 'after-segment'),
            'Unknown metadata/native phase')
    return ['-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30']


def parent_path(profile, label, phase):
    p = control_config(profile)
    if phase in ('cpu', 'after-cpu'):
        require(label == phase, 'Wrong metadata CPU parent')
    elif phase in ('observe', 'dispatch'):
        require(re.fullmatch(phase + r'-\d{4}', label), 'Metadata observer needs unique parent')
    else:
        match = re.fullmatch(r'(boltz2|boltz2_chemistry|pilot)-(native|segment|after-native|after-segment)-(\d{4})', label)
        require(match is not None and match.group(2) == phase, 'Wrong scientific phase parent')
        arm = 'boltz2_dbmol' if match.group(1) == 'pilot' else match.group(1)
        require(arm in p['arms'] and int(match.group(3)) < p['arms'][arm][1], 'Undeclared boundary scope')
        require(phase not in ('native', 'after-native') or int(match.group(3)) > 0,
                'Original initial native jobs cannot be replayed')
    return p['control' if phase in ('cpu', 'after-cpu', 'observe', 'dispatch') else 'root'] / label


def command(profile, label, phase, launcher_manifest, dependency=None):
    p = control_config(profile)
    parent = parent_path(profile, label, phase)
    require(re.fullmatch(r'[0-9a-f]{64}', launcher_manifest), 'Missing immutable adoption launcher pin')
    if dependency is not None:
        require(re.fullmatch(r'ended\(\d+\)(?:&&ended\(\d+\))*', dependency),
                'Only actual ended dependencies are declared')
    args = ['bsub', *resources(phase), '-J', 'mosaic-peer-' + profile + '-' + label,
            '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if dependency is not None:
        args += ['-w', dependency]
    wrapper = p['launcher'] / 'scripts/hpc' / ('peer_adoption_cpu.lsf' if phase == 'cpu' else 'peer_adoption_stage.lsf')
    return args + ['bash', '-l', str(wrapper), str(p['launcher']), launcher_manifest,
                   profile, str(p['source']), str(p['root']), p['manifest'],
                   str(p['control']), phase, str(parent)]


def executing(profile, parent, phase, launcher_manifest):
    parent = Path(parent)
    require(parent == parent_path(profile, parent.name, phase), 'Own output/control parent escaped')
    path = parent / 'submission.json'
    for attempt in range(16):
        if path.exists():
            break
        require(attempt < 15, 'Missing actual own submission')
        time.sleep(2)
    value = read(path)
    require(value['outcome'] == 'submitted' and value['job_id'] == os.environ['LSB_JOBID']
            and value['phase'] == phase
            and value['args'] == command(profile, parent.name, phase, launcher_manifest, value['dependency']),
            'Actual allocated job/source/resource vector differs')
    return value


def final_job(parent, queue):
    """Real scheduler/LSF proof for newly allocated metadata or native jobs."""
    parent = Path(parent)
    record = read(parent / 'submission.json')
    job = record['job_id']
    for index in range(16):
        result = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
        raw = result.stdout + result.stderr
        path = parent / ('scheduler-final-' + str(index) + '.txt')
        with path.open('x') as stream:
            stream.write(raw)
        compact = re.sub(r'\s+', '', raw)
        require(result.returncode == 0 and re.findall(r'Job<([^>]+)>', compact) == [job]
                and re.findall(r'Queue<([^>]+)>', compact) == [queue], 'Actual same-job scheduler unavailable')
        status = re.findall(r'Status<([^>]+)>', compact)
        if status == ['DONE']:
            break
        require(status in (['RUN'], ['PEND']) and index < 15, 'Original allocated producer failed')
        time.sleep(2)
    lsf = contract.lsf_terminal(parent, job, expected='Done')
    return [file_record(path), file_record(parent / 'submission.json'), lsf['proof']]


def verify_sources(profile, launcher_manifest):
    p = control_config(profile)
    records = contract.verify_seal(p['source'], p['manifest'])
    records += contract.verify_seal(p['launcher'], launcher_manifest)
    if profile == 'dbmol10k':
        records += contract.verify_seal(p['original_launcher'], p['original_launcher_manifest'], metadata_only=True)
    return records


def metadata_certificate(profile, launcher_manifest):
    p = control_config(profile)
    path = p['control'] / 'metadata-cpu-certificate.json'
    value = read(path)
    require(METADATA_CASES > 0 and value['version'] == VERSION and value['suite'] == METADATA_SUITE
            and value['profile'] == profile and value['source_manifest_sha256'] == p['manifest']
            and value['launcher_manifest_sha256'] == launcher_manifest
            and value['cases'] == METADATA_CASES
            and value['scientific_readiness_passed'] is False,
            'Separate metadata CPU certificate scope differs')
    contract.verify_proofs(value)
    return value


def certify_metadata_cpu(profile, launcher_manifest):
    p = control_config(profile)
    require(METADATA_CASES > 0, 'Exact allocated metadata fixture count not declared; draft cannot certify')
    proofs = final_job(p['control'] / 'cpu', 'hpc')
    proofs += verify_sources(profile, launcher_manifest)
    proofs += contract.verify_metadata_xml(p['control'] / 'cpu/payload/adoption.xml', METADATA_CASES,
                                           {'test_peer_adoption'})
    value = {'status': 'passed', 'version': VERSION, 'suite': METADATA_SUITE,
             'profile': profile, 'cases': METADATA_CASES, 'source_manifest_sha256': p['manifest'],
             'launcher_manifest_sha256': launcher_manifest, 'proofs': unique_proofs(proofs),
             'scientific_readiness_passed': False, 'future_peer_results_required_for_this_cpu': False,
             'original_scientific_cpu_certificate_replaced': False}
    exclusive(p['control'] / 'metadata-cpu-certificate.json', value)
    return value


def unique_proofs(proofs):
    values = {}
    for proof in proofs:
        previous = values.setdefault(proof['path'], proof)
        require(previous == proof, 'Conflicting proof identity')
    return list(values.values())


def import_original(profile):
    p = control_config(profile)
    source = p['source']
    if profile == 'dbmol10k':
        path = p['original_launcher'] / 'scripts/dbmol_storage_ops.py'
        sys.path.insert(0, str(p['original_launcher'] / 'scripts'))
    else:
        path = source / 'scripts/boltz_serial_ops.py'
    sys.path.insert(0, str(source / 'scripts'))
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('peer_original_' + profile, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if profile == 'dbmol10k':
        module.configure_launcher(p['original_launcher_manifest'])
    return module


def peer_budget(profile, tree, *, full_required):
    """Actual pilot maxima, or explicit validation allowance before any longs."""
    p = PROFILES[profile]
    projection = p['root'] / 'projection.json'
    current = {arm: sum(tree(path) for path in p['root'].glob(
        ('pilot' if profile == 'dbmol10k' else arm) + '-*')) for arm in p['arms']}
    complete = projection.exists() and (profile == 'dbmol10k' or set(read(projection)) == set(p['arms']))
    if not complete:
        require(not full_required, 'Authentic full peer native projections are required')
        require(not list(p['root'].glob('*-segment-*')), 'Unknown peer production lacks full projection')
        full = {arm: GIB for arm in p['arms']}
        require(all(0 <= current[arm] <= full[arm] for arm in full), 'Peer validation allowance exceeded')
        return {'mode': 'native_allowances_only_not_full_long_capacity', 'full': full,
                'current': current, 'remaining': sum(full[arm] - current[arm] for arm in full)}
    evidence = contract.verify_native_units(profile)
    full = evidence['full']
    require(all(0 <= current[arm] <= full[arm] for arm in full), 'Peer output exceeds authenticated full budget')
    return {'mode': 'full_empirical_projections', 'full': full, 'current': current,
            'remaining': sum(full[arm] - current[arm] for arm in full), 'evidence': evidence}


def load_original(profile, launcher_manifest):
    """Authenticate before importing; only declared metadata routing changes."""
    verify_sources(profile, launcher_manifest)
    p = control_config(profile)
    op = import_original(profile)
    original_storage = op.storage_gate
    original_final = op.final_job
    # Numerical driver, configuration, scientific CPU/readiness/FD/parity/null/
    # terminal/frozen/snapshot functions remain original callable objects.
    if profile == 'boltz30k':
        op.protenix_budgets = lambda full_required: peer_budget('protenix50k40k', op.tree, full_required=full_required)
    else:
        def external(full):
            peers, missing = {}, []
            for key, peer in [('extended', 'protenix50k40k'), ('baseline', 'boltz30k')]:
                value = peer_budget(peer, op.tree, full_required=full)
                peers[key] = value
                if value['mode'] != 'full_empirical_projections':
                    missing.append(key)
            return peers, missing
        op.external_budgets = external

    def storage(parent, *args, **kwargs):
        # First retain the original exact admission. A fresh stronger second
        # observation additionally charges BOTH adoption CPU/control allowances.
        stage = args[0]
        # This gate precedes every inherited quota observation. Unknown peers
        # persist a raw inventory and fail without assuming their future growth.
        inventory_path = Path(parent) / ('adoption-noisy-inventory-' + stage + '.json')
        inventory = {'status': 'failed', 'inventory': None}
        try:
            inventory['inventory'] = contract.unknown_noisy_capsules()
            require(inventory['inventory']['status'] == 'declared_capsules_only',
                    'Additional external noisy capsules need authenticated capacity adoption')
            inventory['status'] = 'passed'
        except Exception as error:
            inventory['error'] = repr(error)
            raise
        finally:
            exclusive(inventory_path, inventory)
        original = original_storage(parent, *args, **kwargs)
        import fixed25_lrreset_storage as helper
        future = 0
        controls = {}
        for consumer in ('boltz30k', 'dbmol10k'):
            root = PROFILES[consumer]['control']
            used = op.tree(root)
            require(0 <= used <= GIB, 'Adoption metadata exceeded explicit1GiB cap')
            controls[consumer] = {'full': GIB, 'current': used, 'remaining': GIB - used}
            future += GIB - used
        record = {'status': 'failed', 'version': VERSION, 'original_gate': original,
                  'work3': None, 'metadata_controls': controls,
                  'projected_bytes': original['projected_bytes'] + future,
                  'reserve_bytes': 4 * GIB, 'file_reserve': 100000,
                  'stale_scope': 'entire BASE includes old/new outputs, controls, sources and uploads; noise additionally charged',
                  'no_gate_waiver': True}
        path = Path(parent) / ('storage-' + stage) / 'adoption-capacity.json'
        try:
            raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
            noise = (op.NOISE_PATHS if profile == 'dbmol10k' else
                     (op.NOISY_ROOT, op.NOISY_RELEASE, op.NOISY_UPLOAD))
            gate = helper.gate_record('work3', raw, record['projected_bytes'],
                                      helper.stale_work_bytes() + sum(op.tree(x) for x in noise),
                                      shutil.disk_usage(p['root']).free,
                                      helper.inode_observation(os.statvfs(p['root'])))
            record.update(status=gate['status'], work3=gate)
        except Exception as error:
            record['error'] = repr(error)
            raise
        finally:
            exclusive(path, record)
        require(record['status'] == 'passed', 'Full unchanged budgets plus adoption allowance failed')
        return record
    op.storage_gate = storage
    op.command = lambda label, phase, manifest, dependency=None: command(profile, label, phase, launcher_manifest, dependency)
    op.submit = lambda label, phase, manifest, dependency=None: submit(profile, label, phase, launcher_manifest, dependency, op=op)
    op.final_job = lambda parent, queue: original_final(parent, queue) + adoption_proofs(profile, launcher_manifest)
    if profile == 'boltz30k':
        op.dispatch = lambda manifest, parent: dispatch(profile, parent, launcher_manifest, op=op)
        op.ensure_startup = lambda arm, phase, index, manifest, owner: ensure_startup(profile, arm, phase, index, launcher_manifest, owner, op)
        op.verify_startup = lambda arm, phase, index, manifest: verify_startup(profile, arm, phase, index, launcher_manifest, op)
    else:
        op.dispatch = lambda parent, manifest: dispatch(profile, parent, launcher_manifest, op=op)
        op.ensure_startup = lambda phase, index, manifest: ensure_startup(profile, 'boltz2_dbmol', phase, index, launcher_manifest, current_owner(profile), op)
        op.verify_startup_entry = lambda phase, index, manifest: verify_startup(profile, 'boltz2_dbmol', phase, index, launcher_manifest, op)
    return op


def current_owner(profile):
    p = control_config(profile)
    candidates = []
    for root in (p['control'], p['root']):
        for parent in root.iterdir():
            path = parent / 'submission.json'
            if parent.is_dir() and path.is_file() and read(path).get('job_id') == os.environ.get('LSB_JOBID'):
                candidates.append(parent)
    require(len(candidates) == 1, 'Actual allocated adoption callback owner missing/ambiguous')
    return candidates[0]


def startup_path(profile, arm, phase, index):
    p = control_config(profile)
    require(arm in p['arms'] and phase in ('native', 'segment')
            and type(index) is int and 0 <= index < p['arms'][arm][1], 'Unknown startup scope')
    return p['control'] / ('startup-' + arm + '-' + phase + '-' + format(index, '04d') + '.json')


def ensure_startup(profile, arm, phase, index, launcher_manifest, owner, op):
    from premodel_startup import run_startup_smoke
    path = startup_path(profile, arm, phase, index)
    if path.exists():
        return verify_startup(profile, arm, phase, index, launcher_manifest, op)
    exclusive(path.with_name(path.stem + '-started.json'),
              {'status': 'unknown', 'phase': phase, 'index': index, 'owner': str(owner)})
    kwargs = {'arm': arm, 'index': index, 'phase': phase} if profile == 'boltz30k' else {'index': index, 'phase': phase}
    result = run_startup_smoke(op, PROFILES[profile]['manifest'],
        Path(os.environ['TMPDIR']) / ('adoption-startup-' + arm + '-' + phase + '-' + str(index)), **kwargs)
    value = {'status': 'startup_only_passed', 'version': VERSION, 'profile': profile,
             'arm': arm, 'phase': phase, 'index': index, 'source_manifest_sha256': PROFILES[profile]['manifest'],
             'launcher_manifest_sha256': launcher_manifest, 'owner': file_record(Path(owner) / 'submission.json'),
             'result': result, 'metadata_cpu_certificate': file_record(PROFILES[profile]['control'] / 'metadata-cpu-certificate.json'),
             'adoption_certificate': file_record(PROFILES[profile]['control'] / 'adoption-certificate.json')}
    exclusive(path, value)
    return verify_startup(profile, arm, phase, index, launcher_manifest, op)


def expected_args(profile, arm, phase, index, output, op):
    p = control_config(profile)
    if profile == 'boltz30k':
        inputs = op.startup_inputs(arm, phase, index, p['manifest'])
        return op.startup_expected_args(arm, phase, index, p['manifest'], inputs, output), inputs
    from dbmol_protocol import CONTEXT_FILE, POCKET
    cpu = op.cpu_certificate(p['manifest'])
    snapshot = op.input_snapshot(index)
    ready = op.readiness(index, p['manifest'])[0] if phase == 'segment' else None
    paths = {'config': op.config(), 'cpu-certificate': cpu,
             'reference': p['root'] / op.CPU_PARENT / 'payload/composition_reference.json',
             'pocket': p['root'] / op.CPU_PARENT / 'payload/pocket_definition.json',
             'prepared_context': Path(op.PREPARED) / 'context.pt'}
    args = [op.PYTHON, str(p['source'] / 'scripts/dbmol_driver.py'), '--mode', 'validate' if phase == 'native' else 'run',
            '--config', str(op.config()), '--output', str(output), '--manifest', p['manifest'],
            '--cpu-certificate', str(cpu), '--cpu-certificate-sha256', sha(cpu),
            '--reference', str(paths['reference']), '--pocket', str(paths['pocket']), '--pocket-sha256', POCKET,
            '--prepared', op.PREPARED, '--context-sha256', CONTEXT_FILE, '--worker-python', op.WORKER,
            '--offset', str(index * op.SEGMENT), '--segment-steps', str(op.SEGMENT)]
    if snapshot:
        args += ['--resume-snapshot', snapshot['path'], '--resume-sha256', snapshot['sha256']]
        paths['resume-snapshot'] = Path(snapshot['path'])
    if ready:
        args += ['--ready', str(ready), '--ready-sha256', sha(ready)]
        paths['ready'] = ready
    return args, {key: file_record(path) for key, path in paths.items()}


def verify_startup(profile, arm, phase, index, launcher_manifest, op):
    p = control_config(profile)
    path = startup_path(profile, arm, phase, index)
    value = read(path)
    result = value['result']
    require(value['status'] == 'startup_only_passed' and value['version'] == VERSION
            and value['profile'] == profile and value['arm'] == arm and value['phase'] == phase
            and value['index'] == index and value['source_manifest_sha256'] == p['manifest']
            and value['launcher_manifest_sha256'] == launcher_manifest, 'Startup adoption scope differs')
    require(result['version'] == 'mosaic-actual-generated-command-premodel-v1'
            and result['status'] == 'startup_only_passed' and result['model_loaded'] is False
            and result['scientific_readiness_passed'] is False and result['gradient_passed'] is False
            and result['optimizer_updates'] == 0 and result['saved_results_changed'] is False
            and result['source'] == str(p['source']) and result['source_manifest_sha256'] == p['manifest']
            and result['git_revision_reference'] == contract.REVISION
            and result['arm'] == arm and result['phase'] == phase and result['index'] == index,
            'Startup-only boundary is not the required real driver path')
    require(result['summary']['status'] == 'failed'
            and result['summary']['error'] == 'PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded'
            and not result['summary'].get('cases') and not result['summary'].get('native_gradient_checks'),
            'Intentional startup summary cannot claim readiness or updates')
    owner = value['owner']
    require(file_record(owner['path']) == owner, 'Startup actual owner bytes changed')
    actual = read(owner['path'])
    parent = Path(owner['path']).parent
    require(actual['phase'] in ('after-cpu', 'observe', 'dispatch', 'after-native', 'after-segment')
            and actual['outcome'] == 'submitted' and actual['job_id'] == result['job_id']
            and actual['args'] == command(profile, parent.name, actual['phase'], launcher_manifest, actual['dependency']),
            'Startup owner actual job/source/resources differ')
    args = result['args']
    require(args.count('--output') == 1, 'Startup lacks exact disposable output')
    output = args[args.index('--output') + 1]
    expected, inputs = expected_args(profile, arm, phase, index, output, op)
    require(args == expected and result['inputs'] == inputs, 'Complete actual startup arguments/input hashes differ')
    require(file_record(value['metadata_cpu_certificate']['path']) == value['metadata_cpu_certificate']
            and file_record(value['adoption_certificate']['path']) == value['adoption_certificate'],
            'Startup metadata/adoption proof changed')
    return file_record(path)


def adoption_proofs(profile, launcher_manifest):
    p = control_config(profile)
    metadata_certificate(profile, launcher_manifest)
    path = p['control'] / 'adoption-certificate.json'
    value = read(path)
    require(value['status'] == 'passed' and value['version'] == VERSION and value['profile'] == profile
            and value['source_manifest_sha256'] == p['manifest']
            and value['launcher_manifest_sha256'] == launcher_manifest,
            'Adoption certificate source/profile differs')
    contract.verify_proofs(value)
    owner = read(p['root'] / 'protenix02-adoption-owner.json')
    require(owner == read(p['control'] / 'ownership.json'), 'Exclusive ownership proof differs')
    contract.verify_inventory(owner['immutable_original_proofs'])
    return [file_record(path), file_record(p['control'] / 'metadata-cpu-certificate.json'),
            file_record(p['root'] / 'protenix02-adoption-owner.json')]


def submit(profile, label, phase, launcher_manifest, dependency=None, *, op=None):
    p = control_config(profile)
    parent = parent_path(profile, label, phase)
    startup = None
    if phase in ('native', 'segment'):
        require(op is not None, 'Native submission lacks authenticated original controller')
        _, arm, index = stage_decode(profile, label)
        startup = verify_startup(profile, arm, phase, index, launcher_manifest, op)
        adoption_proofs(profile, launcher_manifest)
    args = command(profile, label, phase, launcher_manifest, dependency)
    parent.mkdir(exist_ok=False)
    value = {'phase': phase, 'args': args, 'dependency': dependency, 'outcome': 'unknown',
             'utc': datetime.now(timezone.utc).isoformat(), 'version': VERSION,
             'source_manifest_sha256': p['manifest'], 'launcher_manifest_sha256': launcher_manifest}
    if startup is not None:
        value['startup_smoke'] = startup
    exclusive(parent / 'submission-started.json', value)
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=60)
        value.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        ids = re.findall(r'Job <(\d+)>', result.stdout)
        if result.returncode == 0 and len(ids) == 1:
            value.update(outcome='submitted', job_id=ids[0])
    except Exception as error:
        value['error'] = repr(error)
    finally:
        exclusive(parent / 'submission.json', value)
    require(value['outcome'] == 'submitted', 'Unknown outcome blocks duplicate submission')
    jobs_path = parent.parent / 'jobs.json'
    jobs = read(jobs_path)
    require(label not in jobs, 'Actual jobs identity would be overwritten')
    jobs[label] = value['job_id']
    atomic(jobs_path, jobs)
    result = subprocess.run(['bjobs', '-l', value['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(result.stdout + result.stderr)
    return value


def stage_decode(profile, label):
    p = control_config(profile)
    match = re.fullmatch(r'(boltz2|boltz2_chemistry|pilot)-(native|segment|after-native|after-segment)-(\d{4})', label)
    require(match is not None, 'Unknown additive scientific stage')
    arm = 'boltz2_dbmol' if match.group(1) == 'pilot' else match.group(1)
    index = int(match.group(3))
    require(arm in p['arms'] and index < p['arms'][arm][1], 'Stage arm/index differs')
    return match.group(2), arm, index


def maybe_queue_observer(profile, parent, launcher_manifest):
    p = control_config(profile)
    state = contract.pending_dependencies(('boltz30k', 'dbmol10k', 'protenix50k40k'))
    exclusive(Path(parent) / 'peer-observation.json', state)
    require(state['status'] != 'failed_peer', 'An actual peer native callback failed; adoption is blocked')
    if state['status'] != 'actual_callbacks_known':
        return state
    # Never queue an observer when no dependency is needed and all evidence is
    # already complete; dispatch in this allocated callback handles that case.
    if state['all_callbacks_terminal']:
        return state
    for label in sorted(read(p['control'] / 'jobs.json')):
        if label.startswith('observe-'):
            record = read(p['control'] / label / 'submission.json')
            require(record['outcome'] == 'submitted', 'Unknown observer outcome blocks duplication')
            return record
    return submit(profile, 'observe-0001', 'observe', launcher_manifest, state['dependency'])


def adopt(profile, parent, launcher_manifest, op):
    p = control_config(profile)
    metadata = metadata_certificate(profile, launcher_manifest)
    dependencies = contract.pending_dependencies(('boltz30k', 'dbmol10k', 'protenix50k40k'))
    if dependencies['status'] == 'failed_peer':
        exclusive(Path(parent) / 'failure.json', {**dependencies, 'production_submitted': False})
        raise ValueError('An actual peer native callback failed; no adoption or production')
    if dependencies['status'] != 'actual_callbacks_known':
        exclusive(Path(parent) / 'waiting.json', {**dependencies, 'production_submitted': False})
        return False
    if not dependencies['all_callbacks_terminal']:
        maybe_queue_observer(profile, parent, launcher_manifest)
        return False
    # A terminal-Done callback must already have persisted its complete proof.
    # Missing/corrupt proof after actual terminal success is a real failure,
    # rather than an impossible observer retry on the same ended dependency.
    peers = {key: contract.verify_native_units(key) for key in ('boltz30k', 'dbmol10k', 'protenix50k40k')}
    # Semantic scientific failures are never hidden as pending.
    for arm in p['arms']:
        if profile == 'boltz30k':
            op.native_assessment(arm, 0, p['manifest'])
        else:
            op.readiness(0, p['manifest'])
    history = contract.verify_failed_protenix_history()
    owner = contract.claim_ownership(profile, metadata)
    proofs = unique_proofs(verify_sources(profile, launcher_manifest)
                           + [file_record(p['control'] / 'metadata-cpu-certificate.json'),
                              file_record(p['control'] / 'ownership.json')]
                           + owner['snapshots'] + owner['immutable_original_proofs'] + history['proofs'])
    snapshots = {}
    for key, peer in peers.items():
        source = PROFILES[key]['root'] / 'projection.json'
        target = p['control'] / 'adoption-snapshot' / ('peer-' + key + '-projection.json')
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        snapshots[key] = file_record(target)
        proofs += peer['proofs'] + [snapshots[key]]
    value = {'status': 'passed', 'version': VERSION, 'profile': profile,
             'source_manifest_sha256': p['manifest'], 'launcher_manifest_sha256': launcher_manifest,
             'source': str(p['source']), 'output_root': str(p['root']), 'control_root': str(p['control']),
             'peer_units': {key: peer['units'] for key, peer in peers.items()},
             'initial_projection_snapshots': snapshots, 'proofs': unique_proofs(proofs),
             'original_scientific_cpu_certificate_replaced': False, 'scientific_results_changed': False,
             'original_failures_preserved': True}
    exclusive(p['control'] / 'adoption-certificate.json', value)
    return True


def dispatch(profile, parent, launcher_manifest, *, op):
    p = control_config(profile)
    cert = p['control'] / 'adoption-certificate.json'
    if not cert.exists() and not adopt(profile, parent, launcher_manifest, op):
        return
    adoption_proofs(profile, launcher_manifest)
    # Full, fresh per-peer projections are mandatory at every production action.
    for peer in ('boltz30k', 'dbmol10k', 'protenix50k40k'):
        contract.verify_native_units(peer)
    for arm in p['arms']:
        for index in range(p['arms'][arm][1]):
            label = ('pilot' if profile == 'dbmol10k' else arm) + '-segment-' + format(index, '04d')
            native = p['root'] / (contract.native_name(profile, arm, index) + '-certificate.json')
            if not native.exists():
                continue
            target = p['root'] / label
            if target.exists():
                existing = read(target / 'submission.json')
                require(existing['outcome'] == 'submitted'
                        and existing['args'] == command(profile, label, 'segment', launcher_manifest, None),
                        'Existing/unknown production outcome blocks duplicate')
                continue
            if profile == 'boltz30k':
                op.native_assessment(arm, index, p['manifest'])
                op.storage_gate(parent, 'dispatch-' + arm + '-' + str(index), op.all_budgets(), existing_full=True)
            else:
                op.readiness(index, p['manifest'])
                op.storage_gate(parent, 'dispatch-' + str(index), True)
            ensure_startup(profile, arm, 'segment', index, launcher_manifest, parent, op)
            value = submit(profile, label, 'segment', launcher_manifest, op=op)
            callback = label.replace('-segment-', '-after-segment-')
            submit(profile, callback, 'after-segment', launcher_manifest, 'ended(' + value['job_id'] + ')', op=op)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('boltz30k', 'dbmol10k'), required=True)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--launcher-manifest', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    p = control_config(args.profile)
    require(os.environ.get('LSB_JOBID') and args.launcher == p['launcher'] and args.source == p['source']
            and args.root == p['root'] and args.control == p['control'] and args.manifest == p['manifest'],
            'Unallocated or wrong sealed profile/source/root')
    verify_sources(args.profile, args.launcher_manifest)
    executing(args.profile, args.parent, args.phase, args.launcher_manifest)
    op = load_original(args.profile, args.launcher_manifest)
    if args.phase == 'cpu':
        # No future success is required for this independent metadata CPU entry.
        if args.profile == 'boltz30k':
            complete = (p['root'] / 'projection.json').exists() and set(read(p['root'] / 'projection.json')) == set(p['arms'])
            op.storage_gate(args.parent, 'metadata-cpu-entry', op.all_budgets() if complete else None)
        else:
            op.storage_gate(args.parent, 'metadata-cpu-entry', False)
        return
    if args.phase in ('native', 'segment'):
        # GPU work must not retain the metadata dispatch lock; independent arms
        # remain independent while each exact entry repeats its full admission.
        metadata_certificate(args.profile, args.launcher_manifest)
        adoption_proofs(args.profile, args.launcher_manifest)
        phase, arm, index = stage_decode(args.profile, args.parent.name)
        startup = verify_startup(args.profile, arm, phase, index, args.launcher_manifest, op)
        require(read(args.parent / 'submission.json')['startup_smoke'] == startup,
                'Actual entry startup proof is not submission bound')
        if args.profile == 'boltz30k':
            op.produce(args.parent, phase, arm, index, args.manifest)
        else:
            op.produce(args.parent, phase, index, args.manifest)
        return
    with (p['root'] / 'dispatch.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        with (p['control'] / 'dispatch.lock').open('a') as control_stream:
            fcntl.flock(control_stream, fcntl.LOCK_EX)
            if args.phase == 'after-cpu':
                certify_metadata_cpu(args.profile, args.launcher_manifest)
                dispatch(args.profile, args.parent, args.launcher_manifest, op=op)
                return
            metadata_certificate(args.profile, args.launcher_manifest)
            if args.phase in ('observe', 'dispatch'):
                dispatch(args.profile, args.parent, args.launcher_manifest, op=op)
                return
            adoption_proofs(args.profile, args.launcher_manifest)
            phase, arm, index = stage_decode(args.profile, args.parent.name)
            if phase == 'after-native':
                if args.profile == 'boltz30k':
                    op.after_native(arm, index, args.manifest, args.parent)
                else:
                    op.after_native(args.parent, index, args.manifest)
            else:
                if args.profile == 'boltz30k':
                    op.certify_segment(arm, index, args.manifest, args.parent)
                else:
                    op.certify_segment(args.parent, index, args.manifest)


if __name__ == '__main__':
    main()
