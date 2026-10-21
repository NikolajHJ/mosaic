"""Evidence-only launcher adoption of the passed original fixed-step native gates.

This tool never changes scientific source02, configurations, readiness records,
model/context bytes, optimizer settings, or historical failure records. It only
places NEW long-run output on work3 after authenticated archival cleanup and a fresh whole-campaign gate.
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

from campaign25_contract import ARMS, FIXED25_VERSION, config25, sha, verify_cpu_certificate, verify_readiness
from campaign25_pocket_ops import parse_final, read, write
from fixed25_ops import CONTEXTS, POCKET, config_path
import fixed25_split_storage as storage

SCIENCE = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-02/source')
SCIENCE_ROOT = storage.SCIENCE_ROOT
SCIENCE_MANIFEST = '1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64'
TOOL_SOURCE = Path('/work3/s234818/mosaic/releases') / storage.TAG / 'source'
ROOT = storage.ROOT
CPU_CERT_SHA = 'a595c100b39816d0f87f5d7d44c7a94c35d9138e6f510a0d284a79f20d5317f2'
CPU_ASSESSMENT_SHA = '0e821878a92575c3a02370fc22c2b66df3307accf9d7d1c17917af806eabbad9'
NATIVE_PINS = {
    'protenix': {'job': '29776117', 'assessment': '3e1a61215ecb5dd5b7999a05030c3de96b1e811a5f2c1b606af53649e2ca3953',
                 'scheduler': '5088f861cfaee776a98b490e8dbf1c78c1883a8e57cb1bef05e68aba4d8923f3',
                 'stdout': '57f73edeea99ec9492c3110bcb4e8a0aca4d7d16cd5abb6373faaea9869a0e00',
                 'config': '69257fd9700d8ca0f42e13ae95ec4c3a15ca623064f657acf4fb8c85b65f1d00'},
    'boltz2': {'job': '29776119', 'assessment': '903a7facf258e29c868160d325287d6c332b339d7a68bf367182b20fca6ff0f0',
               'scheduler': '1ef7da257522129c6f5d946b5aa7c025318d91e4dfe4f7f50e667694da70b059',
               'stdout': '7a08a32a7ed9ea0e1a76aef1424aa64f954dc7e4ccb2bb1839dc678b1a115cb8',
               'config': 'b6217159b0180b8c2f4b3e57764b2101b79e20e1e8cd021b357ae6563e5b161b'},
    'boltz2_chemistry': {'job': '29776121', 'assessment': '5602e2c80dd37c6951356eba04762ceb9799b24ccd6126db7fd8c175ec662e88',
                         'scheduler': 'bd46f85a918fa43f9b29739a8802482dfa422264c4fc21e55c3c024d707873f5',
                         'stdout': 'b98bd96fc7ff054a91ba8925d0f3d16b507ce3c085f7a1cf95c55398e93cbf58',
                         'config': 'dd6347a0756320bc79dea5fbbf7da9aeca8e8bf93dc38f30cd141fd240059049'},
}
SUITE = 'fixed25-split-launcher-contract-v1'
VERSION = 'fixed25-work3-storage-adoption-v2'


def check_release(source, manifest):
    if not re.fullmatch('[a-f0-9]{64}', manifest):
        raise ValueError('A source manifest SHA256 is required')
    subprocess.run([sys.executable, str(SCIENCE / 'scripts/verify_stage0_release.py'),
                    str(source), manifest], check=True, timeout=180)


def hashed(path, expected):
    path = Path(path)
    if not re.fullmatch('[a-f0-9]{64}', expected) or sha(path) != expected:
        raise ValueError('Persisted evidence bytes changed: ' + str(path))
    return {'path': str(path), 'sha256': expected}


def persisted_success(job, queue, scheduler, stdout):
    """Authenticate captured actual terminal evidence; never synthesize bjobs DONE."""
    status, actual_queue = parse_final(Path(scheduler).read_text(), job)
    log = Path(stdout).read_text()
    if (status != 'DONE' or actual_queue != queue or
            f'Subject: Job {job}:' not in log or 'Successfully completed.' not in log):
        raise ValueError('Missing actual same-job DONE/queue/LSF success')


def assert_original_unsubmitted():
    if (SCIENCE_ROOT / 'long-dispatch-started.json').exists() or (SCIENCE_ROOT / 'long-submissions.json').exists():
        raise ValueError('Original production dispatch exists/unknown; no competing split submission')
    for arm in ARMS:
        if (SCIENCE_ROOT / ('run-' + arm)).exists():
            raise ValueError('Original long output exists/unknown; no competing split submission')


def authenticate_original():
    """Read and rehash every original CPU/native proof under unchanged source02."""
    check_release(SCIENCE, SCIENCE_MANIFEST)
    assert_original_unsubmitted()
    cpu_path = SCIENCE_ROOT / 'cpu-certificate.json'
    cpu = verify_cpu_certificate(cpu_path, CPU_CERT_SHA, SCIENCE_MANIFEST)
    if cpu['producer_job_id'] != '29776103' or cpu['tests'] != 482:
        raise ValueError('Original allocated CPU identity/count changed')
    payload = SCIENCE_ROOT / 'cpu/payload'
    hashed(payload / 'assessment.json', CPU_ASSESSMENT_SHA)
    from check_boltz_cpu_proof import verify
    actual = verify(payload, SCIENCE_MANIFEST)
    if actual['passed'] != 482:
        raise ValueError('Original CPU full gate did not pass all 482 cases')
    expected_cpu_scheduler = SCIENCE_ROOT / 'cpu/scheduler-final-0.txt'
    expected_cpu_stdout = SCIENCE_ROOT / 'cpu/stdout.29776103.log'
    proof_map = {p['path']: p['sha256'] for p in cpu['proofs']}
    for p in (expected_cpu_scheduler, expected_cpu_stdout):
        if str(p) not in proof_map:
            raise ValueError('Original CPU certificate omits actual scheduler/LSF proofs')
    persisted_success('29776103', 'hpc', expected_cpu_scheduler, expected_cpu_stdout)
    records = {}
    for arm in ARMS:
        pin = NATIVE_PINS[arm]
        parent = SCIENCE_ROOT / ('validate-' + arm)
        decision_path = SCIENCE_ROOT / ('decision-' + arm + '.json')
        decision = read(decision_path)
        scheduler = parent / 'scheduler-final-1.txt'
        stdout = parent / ('stdout.' + pin['job'] + '.log')
        required = [hashed(scheduler, pin['scheduler']), hashed(stdout, pin['stdout'])]
        if (decision['status'] != 'passed' or decision['native_job'] != pin['job'] or
                decision['assessment_sha256'] != pin['assessment'] or decision['scheduler_proofs'] != required):
            raise ValueError('Original native decision/job/proof binding changed')
        persisted_success(pin['job'], 'gpul40s', scheduler, stdout)
        config = config_path(SCIENCE, arm)
        hashed(config, pin['config'])
        if config25(read(config))['version'] != FIXED25_VERSION:
            raise ValueError('Only original declared fixed-step configs are eligible')
        assessment_path = parent / 'payload/study/assessment.json'
        assessment = verify_readiness(assessment_path, pin['assessment'], config, SCIENCE_MANIFEST)
        if (assessment['context_digest'] != CONTEXTS[arm] or assessment['pocket_definition_sha256'] != POCKET or
                assessment['diffusion_gradient'] is not False or
                decision['projected_bytes'] != assessment['storage_projection']['projected_bytes']):
            raise ValueError('Original frozen context/pocket/projection/no-diffusion changed')
        records[arm] = {'native_job': pin['job'], 'assessment': hashed(assessment_path, pin['assessment']),
                        'config': hashed(config, pin['config']), 'scheduler_proofs': required,
                        'decision': {'path': str(decision_path), 'sha256': sha(decision_path)},
                        'projected_bytes': decision['projected_bytes'], 'context_digest': assessment['context_digest'],
                        'model_digest': assessment['model_digest'], 'pocket_definition_sha256': POCKET}
    failed = SCIENCE_ROOT / 'shared-dispatch-quota/work3-storage-gate.json'
    gate = read(failed)
    projected = sum(x['projected_bytes'] for x in records.values())
    if (gate['status'] != 'failed' or gate['projected_bytes'] != projected or
            gate['reserve_bytes'] != storage.RESERVE or gate['required_bytes'] != projected + storage.RESERVE):
        raise ValueError('Historical combined work3 storage failure must remain failed and unchanged')
    return {'science_source': str(SCIENCE), 'source_manifest_sha256': SCIENCE_MANIFEST,
            'original_root': str(SCIENCE_ROOT), 'cpu_certificate': hashed(cpu_path, CPU_CERT_SHA),
            'cpu_assessment': hashed(payload / 'assessment.json', CPU_ASSESSMENT_SHA),
            'arms': records, 'historical_failed_storage': {'path': str(failed), 'sha256': sha(failed)}}


def capture_adoption():
    original = authenticate_original()
    record = {'version': VERSION, 'status': 'passed', 'original': original,
              'placement': storage.PLACEMENT, 'output_parents': {a: str(storage.output_parent(a)) for a in ARMS},
              'scope': 'Evidence-only original readiness adoption, not a new native test or completed trajectory'}
    path = ROOT / 'adoption.json'
    if path.exists():
        if read(path) != record:
            raise ValueError('Existing immutable split adoption differs')
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        with path.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
    return record


def verify_adoption(expected):
    path = ROOT / 'adoption.json'
    hashed(path, expected)
    record = read(path)
    if (record.get('version') != VERSION or record.get('status') != 'passed' or
            record.get('original') != authenticate_original() or record.get('placement') != storage.PLACEMENT or
            record.get('output_parents') != {a: str(storage.output_parent(a)) for a in ARMS}):
        raise ValueError('Split adoption differs from freshly authenticated original evidence')
    return record


def verify_tool_cpu(tool_manifest, expected_sha):
    path = ROOT / 'cpu-certificate.json'
    hashed(path, expected_sha)
    cert = read(path)
    job = cert['producer_job_id']
    if (cert['status'] != 'passed' or cert['source_manifest_sha256'] != tool_manifest or
            cert['suite'] != SUITE or not isinstance(job, str) or not job.isdigit() or
            type(cert['tests']) is not int or cert['tests'] <= 0):
        raise ValueError('New immutable split launcher CPU gate is not certified')
    proofs = cert['proofs']
    if len(proofs) != len({p['path'] for p in proofs}):
        raise ValueError('Duplicated launcher CPU proofs')
    for proof in proofs:
        p = Path(proof['path'])
        if not p.resolve().is_relative_to(ROOT / 'cpu'):
            raise ValueError('Launcher CPU proof outside its dedicated root')
        hashed(p, proof['sha256'])
    scheduler = [p for p in proofs if re.fullmatch(r'scheduler-final-\d+\.txt', Path(p['path']).name)]
    logs = [p for p in proofs if Path(p['path']).name == 'stdout.' + job + '.log']
    xml = [p for p in proofs if Path(p['path']).suffix == '.xml']
    if len(scheduler) != 1 or len(logs) != 1 or len(xml) != 1:
        raise ValueError('Launcher CPU needs exactly one actual final/LSF/raw XML proof')
    persisted_success(job, 'hpc', scheduler[0]['path'], logs[0]['path'])
    result = ET.parse(xml[0]['path']).getroot()
    cases = result.findall('.//testcase')
    allowed = {'test_fixed25_split_storage', 'test_fixed25_split_ops'}
    if (len(cases) != cert['tests'] or {p.get('classname', '').split('.')[-1] for p in cases} != allowed or
            any(result.findall('.//' + tag) for tag in ('failure', 'error', 'skipped'))):
        raise ValueError('Raw split launcher contracts lack both complete passing modules')
    for suite in result.iter('testsuite'):
        if (any(int(suite.get(k, '0')) != 0 for k in ('failures', 'errors', 'skipped')) or
                int(suite.get('tests', '-1')) != len(suite.findall('testcase'))):
            raise ValueError('Raw launcher CPU XML counts differ')
    return cert


def submit(arm, tool_manifest, cpu_sha, adoption_sha):
    if arm not in ARMS:
        raise ValueError('Unknown split long arm')
    parent = storage.output_parent(arm)
    parent.mkdir(parents=True, exist_ok=False)
    command = ['bsub', '-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
               '-W', '23:00', '-J', 'mosaic-jq1-fixed-split-' + arm,
               '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
               '-gpu', 'num=1:mode=exclusive_process', 'bash', '-l',
               str(TOOL_SOURCE / 'scripts/hpc/fixed25_split_stage.lsf'), str(TOOL_SOURCE), tool_manifest,
               str(ROOT), arm, cpu_sha, adoption_sha]
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'arm': arm, 'phase': 'run',
               'args': command, 'output_parent': str(parent), 'tool_manifest_sha256': tool_manifest,
               'science_manifest_sha256': SCIENCE_MANIFEST, 'outcome': 'unknown'}
    try:
        reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
        receipt.update(stdout=reply.stdout, stderr=reply.stderr, returncode=reply.returncode)
        matches = re.findall(r'Job <(\d+)>', reply.stdout)
        if reply.returncode == 0 and len(matches) == 1:
            receipt.update(job_id=matches[0], outcome='submitted')
    except Exception as error:
        receipt.update(error=repr(error))
    finally:
        with (parent / 'submission.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    if receipt['outcome'] != 'submitted':
        raise RuntimeError('Unknown/failed split submission; exclusive dispatch marker forbids blind retry')
    actual = subprocess.run(['bjobs', '-l', receipt['job_id']], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(actual.stdout + actual.stderr)
    return receipt


def dispatch(tool_manifest, cpu_sha, adoption_sha):
    check_release(TOOL_SOURCE, tool_manifest)
    adoption = verify_adoption(adoption_sha)
    verify_tool_cpu(tool_manifest, cpu_sha)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = ROOT / 'long-dispatch-started.json'
        if marker.exists() or (ROOT / 'long-submissions.json').exists():
            raise RuntimeError('Previous split dispatch exists/unknown; do not duplicate any submission')
        if any(storage.output_parent(a).exists() for a in ARMS):
            raise ValueError('A declared split long output already exists/unknown; no partial competing dispatch')
        projections = {a: adoption['original']['arms'][a]['projected_bytes'] for a in ARMS}
        gate = storage.check_campaign_storage(projections, ROOT / 'quota-dispatch', phase='dispatch')
        marker_data = {'version': VERSION, 'tool_manifest_sha256': tool_manifest,
                       'science_manifest_sha256': SCIENCE_MANIFEST, 'cpu_certificate_sha256': cpu_sha,
                       'adoption_sha256': adoption_sha, 'placement': storage.PLACEMENT, 'storage_gate': gate,
                       'scope': 'Three independent new 10000-update submissions, not completed trajectories'}
        with marker.open('x') as stream:
            json.dump(marker_data, stream, indent=2)
        jobs = []
        for arm in ARMS:
            jobs.append(submit(arm, tool_manifest, cpu_sha, adoption_sha))
            write(ROOT / 'long-submissions.json', {'jobs': jobs, 'all_three_submitted': len(jobs) == 3,
                  'tool_manifest_sha256': tool_manifest, 'science_manifest_sha256': SCIENCE_MANIFEST})
        return jobs


def driver_command(arm, python=sys.executable):
    if arm not in ARMS:
        raise ValueError('Unknown fixed-step long arm')
    ready = SCIENCE_ROOT / f'validate-{arm}/payload/study/assessment.json'
    command = [python, str(SCIENCE / 'examples/brd4_25_atoms.py'), '--mode', 'run',
               '--config', str(config_path(SCIENCE, arm)), '--output', str(storage.output_parent(arm) / 'payload/study'),
               '--manifest', SCIENCE_MANIFEST, '--cpu-certificate', str(SCIENCE_ROOT / 'cpu-certificate.json'),
               '--cpu-certificate-sha256', CPU_CERT_SHA, '--reference', str(SCIENCE_ROOT / 'cpu/payload/composition_reference.json'),
               '--pocket', str(SCIENCE_ROOT / 'cpu/payload/pocket_definition.json'), '--pocket-sha256', POCKET,
               '--ready', str(ready), '--ready-sha256', NATIVE_PINS[arm]['assessment']]
    if arm != 'protenix':
        command += ['--prepared', '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared',
                    '--context-sha256', 'a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093']
    return command


def run_arm(arm, tool_manifest, cpu_sha, adoption_sha):
    if arm not in ARMS:
        raise ValueError('Unknown split arm')
    check_release(TOOL_SOURCE, tool_manifest)
    adoption = verify_adoption(adoption_sha)
    verify_tool_cpu(tool_manifest, cpu_sha)
    marker = read(ROOT / 'long-dispatch-started.json')
    if any(marker.get(k) != value for k, value in (
            ('tool_manifest_sha256', tool_manifest), ('science_manifest_sha256', SCIENCE_MANIFEST),
            ('cpu_certificate_sha256', cpu_sha), ('adoption_sha256', adoption_sha), ('placement', storage.PLACEMENT))):
        raise ValueError('GPU entry does not match authenticated split dispatch')
    parent = storage.output_parent(arm)
    receipt_path = parent / 'submission.json'
    # A job can be scheduled immediately after bsub returns, while its actual
    # reply is still being persisted. Wait briefly for that one receipt only.
    for _ in range(15):
        if receipt_path.exists():
            break
        time.sleep(1)
    receipt = read(receipt_path)
    if (receipt.get('outcome') != 'submitted' or receipt.get('arm') != arm or
            receipt.get('job_id') != os.environ.get('LSB_JOBID') or
            receipt.get('output_parent') != str(parent) or
            receipt.get('tool_manifest_sha256') != tool_manifest or
            receipt.get('science_manifest_sha256') != SCIENCE_MANIFEST or
            receipt.get('args', [])[-6:] != [str(TOOL_SOURCE), tool_manifest, str(ROOT), arm, cpu_sha, adoption_sha]):
        raise ValueError('GPU entry does not match its actual submission job/args/output/source pins')
    payload = parent / 'payload'
    payload.mkdir(exist_ok=False)
    projections = {a: adoption['original']['arms'][a]['projected_bytes'] for a in ARMS}
    # All outstanding assigned arms share each filesystem's budget, even if
    # their scheduler jobs enter at different times. No per-arm double pledge.
    storage.check_campaign_storage(projections, ROOT / ('entry-quota-' + arm), phase='entry')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                         capture_output=True, text=True, check=True, timeout=30)
    (payload / 'gpu.txt').write_text(gpu.stdout)
    command = driver_command(arm)
    write(payload / 'driver-invocation.json', {'args': command, 'cwd': str(SCIENCE),
          'tool_manifest_sha256': tool_manifest, 'science_manifest_sha256': SCIENCE_MANIFEST})
    with (payload / 'native.log').open('x') as log:
        subprocess.run(command, cwd=SCIENCE, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('capture', 'dispatch', 'run'))
    parser.add_argument('--tool-source', type=Path, required=True)
    parser.add_argument('--tool-manifest', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--cpu-certificate-sha256')
    parser.add_argument('--adoption-sha256')
    args = parser.parse_args()
    if args.root.resolve() != ROOT or args.tool_source.resolve() != TOOL_SOURCE:
        raise ValueError('Wrong declared immutable split tool/control roots')
    check_release(TOOL_SOURCE, args.tool_manifest)
    if args.phase == 'capture':
        print(json.dumps(capture_adoption()))
    else:
        if not args.cpu_certificate_sha256 or not args.adoption_sha256:
            raise ValueError('Actual new CPU and original-native adoption SHA pins are required')
        if args.phase == 'dispatch':
            print(json.dumps(dispatch(args.tool_manifest, args.cpu_certificate_sha256, args.adoption_sha256)))
        else:
            run_arm(args.arm, args.tool_manifest, args.cpu_certificate_sha256, args.adoption_sha256)


if __name__ == '__main__':
    main()


