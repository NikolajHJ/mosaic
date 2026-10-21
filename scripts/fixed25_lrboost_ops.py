"""Fresh high-rate fixed-step JQ1 campaign after new allocated CPU/native gates.

All three arms are new scientific runs.  Historical readiness is never adopted
as a pass at the new rates, and no predecessor is resumed or overwritten.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from campaign25_contract import (
    ARMS, FIXED25_LRBOOST_VERSION, REFERENCE_SHA, config25, sha, verify_readiness,
)
from campaign25_pocket_ops import final_job, read
import fixed25_lrboost_storage as storage


TAG = '20261021-brd4-jq1-25-fixed-lrboost-01'
VERSION = 'jq1-three-arm-fixed-lrboost-launcher-v1'
SOURCE = Path('/work3/s234818/mosaic/releases') / TAG / 'source'
ROOT = storage.ROOT
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
PREPARED = Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared')
BOLTZ_CONTEXT = 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
CONTEXTS = {'protenix': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4',
            'boltz2': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
            'boltz2_chemistry': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'}
RATES = {'protenix': 3.0, 'boltz2': 0.05, 'boltz2_chemistry': 0.05}
STOP_ROOT = Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-lrboost-stop-01')
STOP_ASSESSMENT_SHA256 = '8fc3ce7a6fffd0df8bb30a23ad1f37c9954beb5a572ee2b3ed8746206778800c'
AUTHORIZATION = {'version': VERSION, 'fresh_updates_per_arm': 10000, 'rates': RATES,
    'acceptance_policy': 'fixed_step', 'one_candidate': True, 'diffusion_gradient': False,
    'predecessor_boltz_jobs_intentionally_stopped': ['29778203', '29778204'],
    'completed_protenix_preserved': '29777709'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    exclusive(temporary, value)
    os.replace(temporary, path)


def hashed(path, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.resolve() == path.absolute(),
            'Unsafe scientific proof path')
    digest = sha(path)
    require(expected is None or digest == expected, 'Scientific proof changed: ' + str(path))
    return {'path': str(path), 'sha256': digest}


def config_path(arm):
    require(arm in ARMS, 'Unknown high-rate arm')
    return SOURCE / f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json'


def validate_configs():
    for arm in ARMS:
        config = config25(read(config_path(arm)))
        require(config['version'] == FIXED25_LRBOOST_VERSION and config['arm'] == arm
                and config['initial_step_size'] == RATES[arm]
                and config['max_backtracks'] == 1 and config['acceptance_policy'] == 'fixed_step',
                'Launcher cannot execute historical or changed scientific policies')


def native_parent(arm, phase):
    require(arm in ARMS and phase in ('validate', 'after-ready', 'run'), 'Unknown native/callback phase')
    return storage.output_parent(arm) if phase == 'run' else ROOT / (phase + '-' + arm)


def submission_command(arm, phase, manifest, cpu_sha, *, producer=None):
    parent = native_parent(arm, phase)
    gpu = phase != 'after-ready'
    require((producer is None and gpu) or (phase == 'after-ready' and str(producer).isdigit()),
            'Only own readiness callback may depend on a producer')
    command = ['bsub', '-q', 'gpul40s' if gpu else 'hpc', '-n', '4' if gpu else '1',
        '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]',
        '-W', '23:00' if phase == 'run' else '02:00' if gpu else '00:30',
        '-J', 'mosaic-jq1-lrboost-' + phase + '-' + arm,
        '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if gpu:
        command += ['-gpu', 'num=1:mode=exclusive_process']
    if producer is not None:
        command += ['-w', 'ended(' + str(producer) + ')']
    return command + ['bash', '-l', str(SOURCE / 'scripts/hpc/fixed25_lrboost_stage.lsf'),
        str(SOURCE), str(ROOT), manifest, phase, arm, str(parent), str(producer or '0'), cpu_sha]


def submit(arm, phase, manifest, cpu_sha, *, producer=None):
    parent = native_parent(arm, phase)
    command = submission_command(arm, phase, manifest, cpu_sha, producer=producer)
    # Parent and marker are exclusive even when bsub times out or loses a reply.
    require(not parent.exists(), 'Submission already exists or its outcome is unknown; no duplicate')
    parent.mkdir(parents=True, exist_ok=False)
    marker = parent / 'submission-started.json'
    exclusive(marker, {'version': VERSION, 'args': command, 'phase': phase, 'arm': arm})
    record = {'version': VERSION, 'outcome': 'unknown', 'utc': datetime.now(timezone.utc).isoformat(),
        'args': command, 'phase': phase, 'arm': arm, 'output_parent': str(parent),
        'source_manifest_sha256': manifest, 'cpu_certificate_sha256': cpu_sha,
        'config_sha256': sha(config_path(arm)), 'learning_rate': RATES[arm]}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
        jobs = re.findall(r'Job <(\d+)>', result.stdout)
        if result.returncode == 0 and len(jobs) == 1:
            record.update(outcome='submitted', job_id=jobs[0])
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        exclusive(parent / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Unknown/failed bsub result preserved; no blind duplicate')
    actual = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(actual.stdout + actual.stderr, encoding='utf-8')
    return record


def verify_cpu(manifest, expected_sha=None):
    path = ROOT / 'cpu-certificate.json'
    hashed(path, expected_sha)
    record = read(path)
    require(record['status'] == 'passed' and record['source_manifest_sha256'] == manifest
            and record['tests'] >= 299 and record['proofs'], 'New full CPU certificate failed or differs')
    for proof in record['proofs']:
        hashed(proof['path'], proof['sha256'])
    require(sha(ROOT / 'cpu/payload/pocket_definition.json') == POCKET
            and sha(ROOT / 'cpu/payload/composition_reference.json') == REFERENCE_SHA,
            'Original scientific pocket/reference identity changed')
    return record


def certify_cpu(manifest, producer):
    status, queue, proofs = final_job(producer, ROOT / 'cpu')
    require(status == 'DONE' and queue == 'hpc', 'Actual new full CPU allocation failed')
    from check_boltz_cpu_proof import verify
    assessment = verify(ROOT / 'cpu/payload', manifest)
    xml = ROOT / 'cpu/payload/hpc-contract.xml'
    cases = ET.parse(xml).findall('.//testcase')
    modules = {part for c in cases for part in c.get('classname', '').split('.')}
    require({'test_fixed25_lrboost_contract', 'test_fixed25_lrboost_ops', 'test_fixed25_lrboost_storage'} <= modules,
            'New high-rate contract/controller/storage fixtures did not all run')
    proofs += assessment['proofs'] + [hashed(ROOT / 'cpu/payload' / name) for name in
        ('assessment.json', 'composition_reference.json', 'pocket_definition.json',
         'hpc-contract.log', 'legacy.log', 'native-mapping.log', 'chemistry.log')]
    exclusive(ROOT / 'cpu-certificate.json', {'status': 'passed', 'source_manifest_sha256': manifest,
        'producer_job_id': str(producer), 'tests': assessment['passed'], 'proofs': proofs})
    verify_cpu(manifest)
    cpu_sha = sha(ROOT / 'cpu-certificate.json')
    storage.check_work_storage(ROOT / 'native-validation-dispatch-quota', phase='validate')
    exclusive(ROOT / 'native-dispatch-started.json', {'version': VERSION,
        'source_manifest_sha256': manifest, 'cpu_certificate_sha256': cpu_sha,
        'authorization': AUTHORIZATION})
    jobs = []
    for arm in ARMS:
        native = submit(arm, 'validate', manifest, cpu_sha)
        callback = submit(arm, 'after-ready', manifest, cpu_sha, producer=native['job_id'])
        jobs.append({'arm': arm, 'native': native, 'callback': callback})
        atomic(ROOT / 'native-submissions.json', {'version': VERSION, 'jobs': jobs,
            'all_three_submitted': len(jobs) == 3, 'long_runs_submitted': False})
    return jobs


def validate_projection(assessment):
    record = assessment['storage_projection']
    maximum, full = record['maximum_pilot_update_bytes'], record['projected_bytes']
    require(type(maximum) is int and maximum > 0 and type(full) is int
            and full == 2 * maximum * 10000 + 1024**3,
            'New-rate pilot storage projection does not match original conservative formula')
    return full


def authenticate_native_terminal(job, proofs):
    require(proofs, 'Actual native terminal proofs are missing')
    scheduler, stdout = False, False
    for proof in proofs:
        path = Path(proof['path'])
        hashed(path, proof['sha256'])
        if path.suffix not in ('.txt', '.log'):
            continue
        raw = path.read_text(encoding='utf-8')
        compact = re.sub(r'\s+', '', raw)
        if (re.findall(r'Job<([^>]+)>', compact) == [job]
                and re.findall(r'Status<([^>]+)>', compact) == ['DONE']
                and re.findall(r'Queue<([^>]+)>', compact) == ['gpul40s']):
            scheduler = True
        if (f'Subject: Job {job}:' in raw and 'Successfully completed.' in raw
                and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I)):
            stdout = True
    require(scheduler and stdout, 'Actual same-job native DONE/LSF success missing')


def authenticate_ready(arm, manifest, decision):
    path = ROOT / ('validate-' + arm) / 'payload/study/assessment.json'
    own = read(native_parent(arm, 'validate') / 'submission.json')
    require(decision['status'] == 'passed' and decision['native_job'] == own['job_id']
            and own['outcome'] == 'submitted' and own['source_manifest_sha256'] == manifest
            and own['learning_rate'] == RATES[arm]
            and own['config_sha256'] == sha(config_path(arm))
            and own['args'] == submission_command(arm, 'validate', manifest, sha(ROOT / 'cpu-certificate.json')),
            'Readiness decision does not match actual new native job')
    for proof in decision['proofs']:
        hashed(proof['path'], proof['sha256'])
    authenticate_native_terminal(own['job_id'], decision['proofs'])
    assessment = verify_readiness(path, decision['assessment_sha256'], config_path(arm), manifest)
    summary = read(path.with_name('summary.json'))
    require(summary['status'] == 'passed' and summary['job_id'] == own['job_id']
            and summary['mode'] == 'validate' and summary['source_manifest_sha256'] == manifest
            and summary['config_sha256'] == sha(config_path(arm)) and summary['config'] == read(config_path(arm))
            and summary['native_hard_parity']['status'] == 'passed'
            and summary['native_gradient_checks']['status'] == 'passed'
            and summary['pocket_validation_status'] == 'passed'
            and summary['optimization_parameters'] == {'step_size': RATES[arm], 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
            and summary['frozen_identity_after']['model_digest'] == summary['model_digest'] == assessment['model_digest']
            and summary['frozen_identity_after']['context_digest'] == summary['context_digest'] == CONTEXTS[arm],
            'Actual new-rate native summary/parity/frozen/gradient identity failed')
    if arm != 'protenix':
        require(summary['context_file_sha256'] == BOLTZ_CONTEXT
                and summary['frozen_identity_after']['status'] == 'passed', 'Fixed detached Boltz context/frozen response failed')
    require(assessment['context_digest'] == CONTEXTS[arm] and assessment['pocket_definition_sha256'] == POCKET
            and assessment['diffusion_gradient'] is False and assessment['runtime_gate']['status'] == 'passed',
            'New-rate frozen context/pocket/runtime readiness failed')
    require(decision['projected_bytes'] == validate_projection(assessment), 'New pilot projection changed')
    return assessment


def verify_stopped_predecessors():
    """New long runs cannot overlap the predecessor Boltz trajectories."""
    require(isinstance(STOP_ASSESSMENT_SHA256, str) and re.fullmatch(r'[0-9a-f]{64}', STOP_ASSESSMENT_SHA256),
            'Actual predecessor stop proof must be sealed before dispatch')
    path = STOP_ROOT / 'assessment.json'
    hashed(path, STOP_ASSESSMENT_SHA256)
    record = read(path)
    require(record['version'] == 'jq1-lrboost-predecessor-stop-v1' and record['status'] == 'passed'
            and record['user_requested'] is True
            and set(record['jobs']) == {'boltz2', 'boltz2_chemistry'}
            and {j['job_id'] for j in record['jobs'].values()} == {'29778203', '29778204'}
            and record['preserved_completed_protenix'] == '29777709',
            'Predecessor stop proof is incomplete or changed')
    for proof in record['proofs']:
        hashed(proof['path'], proof['sha256'])
    for job in record['jobs'].values():
        require(job['status'] == 'EXIT', 'Predecessor scientific allocation is still active')
        for field in ('scheduler', 'stdout'):
            proof = job[field]
            require(proof in record['proofs'], 'Actual stop scheduler/LSF proof omitted')
            hashed(proof['path'], proof['sha256'])
        scheduler = Path(job['scheduler']['path']).read_text(encoding='utf-8')
        stdout = Path(job['stdout']['path']).read_text(encoding='utf-8')
        compact = re.sub(r'\s+', '', scheduler)
        require(re.findall(r'Job<([^>]+)>', compact) == [job['job_id']]
                and re.findall(r'Status<([^>]+)>', compact) == ['EXIT']
                and re.findall(r'Queue<([^>]+)>', compact) == ['gpul40s']
                and f'Subject: Job {job["job_id"]}:' in stdout
                and 'TERM_OWNER' in scheduler + stdout, 'Actual intentional same-job termination not authenticated')
    return hashed(path)


def dispatch(manifest):
    paths = {a: ROOT / ('decision-' + a + '.json') for a in ARMS}
    if not all(p.exists() for p in paths.values()):
        return None
    require(not (ROOT / 'long-dispatch-started.json').exists(), 'Long dispatch already exists/unknown; no duplicate')
    verify_cpu(manifest)
    decisions = {a: read(p) for a, p in paths.items()}
    projections = {a: validate_projection(authenticate_ready(a, manifest, decisions[a])) for a in ARMS}
    stop = verify_stopped_predecessors()
    receipt = ROOT / ('quota-dispatch-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    gate = storage.check_campaign_storage(projections, receipt, phase='dispatch')
    cpu_sha = sha(ROOT / 'cpu-certificate.json')
    exclusive(ROOT / 'long-dispatch-started.json', {'version': VERSION,
        'source_manifest_sha256': manifest, 'cpu_certificate_sha256': cpu_sha,
        'authorization': AUTHORIZATION, 'projections': projections,
        'readiness': {a: {'path': str(paths[a]), 'sha256': sha(paths[a])} for a in ARMS},
        'predecessors_stopped': stop, 'storage_gate': gate})
    jobs = []
    for arm in ARMS:
        jobs.append(submit(arm, 'run', manifest, cpu_sha))
        atomic(ROOT / 'long-submissions.json', {'version': VERSION, 'jobs': jobs,
            'all_three_submitted': len(jobs) == 3, 'source_manifest_sha256': manifest})
    return jobs


def after_ready(arm, manifest, producer):
    verify_cpu(manifest)
    callback = read(native_parent(arm, 'after-ready') / 'submission.json')
    require(callback['outcome'] == 'submitted' and callback['job_id'] == os.environ.get('LSB_JOBID')
            and callback['args'] == submission_command(arm, 'after-ready', manifest,
                sha(ROOT / 'cpu-certificate.json'), producer=producer),
            'Readiness callback differs from its actual own submission')
    own = read(native_parent(arm, 'validate') / 'submission.json')
    require(own['job_id'] == producer and own['outcome'] == 'submitted', 'Readiness callback has wrong producer')
    status, queue, proofs = final_job(producer, native_parent(arm, 'validate'))
    path = ROOT / ('decision-' + arm + '.json')
    if status != 'DONE' or queue != 'gpul40s':
        exclusive(path, {'status': 'failed', 'native_job': producer, 'proofs': proofs})
        raise ValueError('Actual new-rate native validation failed; no long dispatch')
    assessment_path = native_parent(arm, 'validate') / 'payload/study/assessment.json'
    assessment = verify_readiness(assessment_path, sha(assessment_path), config_path(arm), manifest)
    decision = {'status': 'passed', 'native_job': producer,
        'assessment_sha256': sha(assessment_path), 'projected_bytes': validate_projection(assessment),
        'proofs': proofs + [hashed(assessment_path)]}
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        exclusive(path, decision)
        authenticate_ready(arm, manifest, decision)
        return dispatch(manifest)


def driver_command(arm, phase, manifest, cpu_sha):
    require(arm in ARMS and phase in ('validate', 'run'), 'Unknown new scientific execution phase')
    command = [PYTHON, str(SOURCE / 'examples/brd4_25_atoms.py'), '--mode', phase,
        '--config', str(config_path(arm)), '--output', str(native_parent(arm, phase) / 'payload/study'),
        '--manifest', manifest, '--cpu-certificate', str(ROOT / 'cpu-certificate.json'),
        '--cpu-certificate-sha256', cpu_sha, '--reference', str(ROOT / 'cpu/payload/composition_reference.json'),
        '--pocket', str(ROOT / 'cpu/payload/pocket_definition.json'), '--pocket-sha256', POCKET]
    if arm != 'protenix':
        command += ['--prepared', str(PREPARED), '--context-sha256', BOLTZ_CONTEXT]
    if phase == 'run':
        ready = ROOT / ('validate-' + arm) / 'payload/study/assessment.json'
        command += ['--ready', str(ready), '--ready-sha256', sha(ready)]
    return command


def execute(arm, phase, manifest, cpu_sha):
    verify_cpu(manifest, cpu_sha)
    parent = native_parent(arm, phase)
    receipt_path = parent / 'submission.json'
    for _ in range(15):
        if receipt_path.exists():
            break
        time.sleep(1)
    receipt = read(receipt_path)
    require(receipt['outcome'] == 'submitted' and receipt['job_id'] == os.environ.get('LSB_JOBID')
            and receipt['args'] == submission_command(arm, phase, manifest, cpu_sha)
            and receipt['source_manifest_sha256'] == manifest and receipt['cpu_certificate_sha256'] == cpu_sha
            and receipt['config_sha256'] == sha(config_path(arm)) and receipt['learning_rate'] == RATES[arm],
            'Actual allocated scientific job/configuration/source differs')
    if phase == 'run':
        marker = read(ROOT / 'long-dispatch-started.json')
        require(marker['source_manifest_sha256'] == manifest and marker['cpu_certificate_sha256'] == cpu_sha
                and marker['authorization'] == AUTHORIZATION, 'Whole dispatch scientific authorization changed')
        for a in ARMS:
            proof = marker['readiness'][a]
            hashed(proof['path'], proof['sha256'])
            authenticate_ready(a, manifest, read(proof['path']))
        verify_stopped_predecessors()
        storage.check_campaign_storage(marker['projections'], ROOT / ('entry-quota-' + arm), phase='entry')
    else:
        storage.check_work_storage(ROOT / ('validation-quota-' + arm), phase='validate')
    payload = parent / 'payload'
    payload.mkdir(exist_ok=False)
    command = driver_command(arm, phase, manifest, cpu_sha)
    exclusive(payload / 'driver-invocation.json', {'args': command, 'cwd': str(SOURCE),
        'source_manifest_sha256': manifest, 'cpu_certificate_sha256': cpu_sha, 'job_id': os.environ['LSB_JOBID']})
    with (payload / 'gpu.txt').open('x') as stream:
        subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], stdout=stream, check=True)
    with (payload / 'native.log').open('x') as stream:
        subprocess.run(command, cwd=SOURCE, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--phase', choices=('preflight', 'after-cpu', 'after-ready', 'validate', 'run'), required=True)
    parser.add_argument('--arm', choices=('none', *ARMS), default='none')
    parser.add_argument('--producer', default='0')
    parser.add_argument('--cpu-certificate-sha256', default='0')
    args = parser.parse_args()
    require(args.source.resolve() == SOURCE and args.root.resolve() == ROOT, 'Wrong immutable high-rate source/control root')
    subprocess.run([PYTHON, str(SOURCE / 'scripts/verify_stage0_release.py'), str(SOURCE), args.manifest], check=True)
    validate_configs()
    if args.phase == 'preflight':
        print(json.dumps(storage.check_work_storage(ROOT / 'preflight-quota', phase='preflight')))
        return
    require(bool(os.environ.get('LSB_JOBID')), 'New scientific validation/execution requires allocated DTU HPC')
    if args.phase == 'after-cpu':
        certify_cpu(args.manifest, args.producer)
    elif args.phase == 'after-ready':
        after_ready(args.arm, args.manifest, args.producer)
    else:
        execute(args.arm, args.phase, args.manifest, args.cpu_certificate_sha256)


if __name__ == '__main__':
    main()
