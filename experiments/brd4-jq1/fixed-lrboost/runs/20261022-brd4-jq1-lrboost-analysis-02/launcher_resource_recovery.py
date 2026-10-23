"""Protenix-only resource adoption; unchanged CPU282-certified diagnostic core."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

VERSION = 'protenix-diagnostic-shared-budget-adoption-v1'
LOWER = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-03')
LOWER_SOURCE = LOWER.parent / 'restart-lrreset-release-03/source'
LOWER_MANIFEST = 'ff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567'
CORE_ROOT = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-02')
CORE_SOURCE = Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-02/source')
CORE_MANIFEST = '6d547929ce3227302729b022628cc26612d9829e72c0a70861870023463dac3a'
ROOT = CORE_ROOT / 'launcher-recovery-03'
SOURCE = Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-02-launcher-03/source')
UPLOAD = SOURCE.parent.parent.parent / '20261022-brd4-jq1-lrboost-analysis-02-launcher-03.tar.gz'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def resources(phase):
    require(phase in ('cpu', 'dispatch', 'diagnostic', 'diagnostic-capture'), 'Undeclared diagnostic-only phase')
    gpu = phase == 'diagnostic'
    result = ['-q', 'gpul40s' if gpu else 'hpc', '-n', '4' if gpu else '1' if phase == 'dispatch' else '2',
              '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]' if phase == 'dispatch' else 'rusage[mem=4GB]',
              '-W', '02:00' if gpu else '00:30' if phase == 'dispatch' else '00:45']
    if gpu:
        result += ['-gpu', 'num=1:mode=exclusive_process']
    return result


def parent(phase):
    resources(phase)
    return CORE_ROOT / ('diagnostic-protenix' if phase == 'diagnostic' else 'diagnostic-capture-protenix') if phase in ('diagnostic', 'diagnostic-capture') else ROOT / ('cpu' if phase == 'cpu' else 'after-cpu')


def command(phase, manifest, dependency=None):
    p = parent(phase)
    args = ['bsub', *resources(phase), '-J', 'mosaic-prodiag-shared03-' + phase,
            '-oo', str(p / 'stdout.%J.log'), '-eo', str(p / 'stderr.%J.log')]
    if dependency:
        require(phase in ('dispatch', 'diagnostic-capture') and re.fullmatch(r'ended\([0-9]+\)', dependency), 'Wrong own-ended dependency')
        args += ['-w', dependency]
    else:
        require(phase in ('cpu', 'diagnostic'), 'Own callback dependency is mandatory')
    return args + ['bash', '-l', str(SOURCE / 'launcher_resource_recovery.lsf'), manifest, phase]


def raw_command():
    return [PYTHON, str(CORE_SOURCE / 'scripts/lrboost_endpoint_diagnostics.py'), '--arm', 'protenix',
            '--output', str(CORE_ROOT / 'diagnostic-protenix/payload'), '--manifest', CORE_MANIFEST,
            '--nearby-step', '9900', '--max-output-bytes', str(256 * 1024**2), '--max-seconds', '5400']


def authenticate(manifest):
    for p, expected in ((SOURCE, manifest), (LOWER_SOURCE, LOWER_MANIFEST), (CORE_SOURCE, CORE_MANIFEST)):
        require(sha(p / 'source_manifest.json') == expected, 'Immutable source identity changed')
        subprocess.run([PYTHON, str(LOWER_SOURCE / 'scripts/verify_stage0_release.py'), str(p), expected], check=True)
    sys.path[:0] = [str(CORE_SOURCE / 'scripts'), str(LOWER_SOURCE / 'scripts')]
    import fixed25_lrreset_ops as lower
    import jq1_lrboost_report_ops as core
    lc = lower.verify_cpu(LOWER_MANIFEST); dc = core.verify_cpu(CORE_MANIFEST)
    require(lc['producer_job_id'] == '29792104' and lc['tests'] == 724
            and dc['producer_job_id'] == '29792111' and dc['tests'] == 282, 'Original CPU724/282 proof chains changed')
    lower.verify_finished_predecessors()
    return lower, core


def shared_gate(manifest, phase):
    lower, core = authenticate(manifest)
    receipt = LOWER / ('diagnostic-shared-' + phase + '-' + os.environ.get('LSB_JOBID', 'preflight'))
    marker = LOWER / 'long-dispatch-started.json'
    if marker.exists():
        gate = lower.storage.check_campaign_storage(read(marker)['projections'], receipt, phase='entry')
    else:
        gate = lower.storage.check_work_storage(receipt, phase='validate')
    # The exact shared gate already charges CORE_ROOT and CORE_SOURCE. Add this
    # small independent launcher release/upload too, retaining every reserve.
    extra = sum(p.stat().st_size for p in SOURCE.parent.rglob('*') if p.is_file())
    if UPLOAD.is_file():
        extra += UPLOAD.stat().st_size
    work = gate['filesystems']['work3']
    require(work['available_bytes_conservative'] - extra >= work['required_bytes'], 'Shared gate plus launcher bytes does not fit')
    result = {'status': 'passed', 'version': VERSION, 'shared_gate': str(receipt / 'lrreset-storage-gate.json'),
              'shared_gate_sha256': sha(receipt / 'lrreset-storage-gate.json'), 'extra_launcher_bytes': extra,
              'required_bytes': work['required_bytes'], 'available_after_extra_bytes': work['available_bytes_conservative'] - extra,
              'reserve_bytes': 4 * 1024**3, 'file_reserve': 100000,
              'scope': 'All current lower-rate outputs plus original 2GiB report/512MiB diagnostic allowance; completed predecessors authenticated, no further old scientific execution'}
    exclusive(receipt / 'diagnostic-resource-adoption.json', result)
    return result


def submit(phase, manifest, dependency=None):
    p = parent(phase); p.mkdir(exist_ok=False)
    args = command(phase, manifest, dependency)
    record = {'version': VERSION, 'phase': phase, 'arm': 'protenix' if phase.startswith('diagnostic') else 'none',
              'source_manifest_sha256': manifest, 'core_manifest_sha256': CORE_MANIFEST,
              'args': args, 'dependency': dependency, 'outcome': 'unknown', 'utc': datetime.now(timezone.utc).isoformat()}
    exclusive(p / 'submission-started.json', record)
    try:
        r = subprocess.run(args, text=True, capture_output=True, timeout=60)
        record.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        ids = re.findall(r'Job <(\d+)>', r.stdout)
        if r.returncode == 0 and len(ids) == 1:
            record.update(outcome='submitted', job_id=ids[0])
    finally:
        exclusive(p / 'submission.json', record)
    require(record['outcome'] == 'submitted', 'Failed/unknown outcome preserved; no duplicate')
    r = subprocess.run(['bjobs', '-l', record['job_id']], capture_output=True, text=True, timeout=30)
    (p / 'scheduler-at-submission.txt').write_text(r.stdout + r.stderr)
    return record


def validate(phase, manifest):
    item = read(parent(phase) / 'submission.json')
    require(item['outcome'] == 'submitted' and item['job_id'] == os.environ.get('LSB_JOBID')
            and item['source_manifest_sha256'] == manifest
            and item['args'] == command(phase, manifest, item.get('dependency')), 'Actual allocation differs from exact submitted resource/source/phase vector')
    return item


def cpu_certificate(manifest, create=False):
    from campaign25_pocket_ops import final_job
    lower, core = authenticate(manifest)
    cert = ROOT / 'cpu-certificate.json'
    if create:
        sub = read(parent('cpu') / 'submission.json'); status, queue, proofs = final_job(sub['job_id'], parent('cpu'))
        require(status == 'DONE' and queue == 'hpc', 'Actual launcher CPU allocation failed')
        core.actual_lsf((parent('cpu') / ('stdout.' + sub['job_id'] + '.log')).read_text(), sub['job_id'])
        xml = parent('cpu') / 'payload/tests.xml'; tree = ET.parse(xml)
        cases = list(tree.iter('testcase'))
        require(len(cases) == 12 and not any(list(tree.iter(t)) for t in ('failure', 'error', 'skipped')), 'All 12 resource-adoption fixtures must pass')
        proofs += [{'path': str(xml), 'sha256': sha(xml)}, {'path': str(parent('cpu') / 'payload/tests.log'), 'sha256': sha(parent('cpu') / 'payload/tests.log')}]
        exclusive(cert, {'status': 'passed', 'source_manifest_sha256': manifest, 'tests': len(cases), 'producer_job_id': sub['job_id'], 'proofs': proofs})
    record = read(cert)
    require(record['status'] == 'passed' and record['source_manifest_sha256'] == manifest and record['tests'] == 12, 'Wrong allocated launcher CPU certificate')
    for proof in record['proofs']:
        require(sha(proof['path']) == proof['sha256'], 'Launcher CPU proof changed')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('manifest'); parser.add_argument('phase')
    args = parser.parse_args(); require(bool(os.environ.get('LSB_JOBID')), 'Allocated DTU execution only')
    authenticate(args.manifest); validate(args.phase, args.manifest)
    if args.phase == 'cpu':
        shared_gate(args.manifest, 'cpu'); run = parent('cpu') / 'payload'; run.mkdir()
        with (run / 'tests.log').open('x') as log:
            subprocess.run([PYTHON, '-m', 'pytest', str(SOURCE / 'test_launcher_resource_recovery.py'), '-o', 'addopts=',
                            '--rootdir', str(SOURCE), '--basetemp', str(Path(os.environ['TMPDIR']) / 'resource-tests'),
                            '--junitxml', str(run / 'tests.xml')], cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT, check=True)
        return
    cpu_certificate(args.manifest, create=args.phase == 'dispatch')
    gate = shared_gate(args.manifest, args.phase)
    if args.phase == 'dispatch':
        exclusive(ROOT / 'dispatch-started.json', {'manifest': args.manifest, 'gate': gate})
        native = submit('diagnostic', args.manifest)
        capture = submit('diagnostic-capture', args.manifest, 'ended(' + native['job_id'] + ')')
        exclusive(ROOT / 'diagnostic-submissions.json', {'native': native, 'capture': capture})
    elif args.phase == 'diagnostic':
        exclusive(parent('diagnostic') / 'resource-adoption.json', {'launcher_manifest': args.manifest, 'core_manifest': CORE_MANIFEST,
                  'cpu_certificate_sha256': sha(ROOT / 'cpu-certificate.json'), 'gate': gate, 'raw_command': raw_command()})
        _, core = authenticate(args.manifest); core.run_diagnostic(raw_command())
    elif args.phase == 'diagnostic-capture':
        _, core = authenticate(args.manifest); core.diagnostic_capture(parent('diagnostic-capture'), 'protenix', CORE_MANIFEST)


if __name__ == '__main__':
    main()
