"""Immutable CPU -> separate DBMol affinity reference qualification.

All original studies and failures stay unchanged. Native-component evidence is
not a complete continuation admission. There is no optimizer dispatch branch.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'dbmol-endpoint-recovery-10'
SOURCE = BASE / 'dbmol-endpoint-recovery-release-10/source'
PARENT = BASE / 'protenix-endpoint-fix-release-08/source'
PARENT_PIN = '7082a76a2da1849cfdf377735ebd7ccbc77e298395c7f2f154f18904a1db6856'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
TORCH = '/work3/s234818/latentmol/env/boltz311/bin/python'
COUNTS = {'tests.test_dbmol_reference': 12, 'tests.test_dbmol_endpoint_precision': 10,
          'tests.test_recovery_capacity': 10}


def helpers():
    import hashlib
    assert hashlib.sha256((PARENT / 'source_manifest.json').read_bytes()).hexdigest() == PARENT_PIN
    spec = importlib.util.spec_from_file_location('_dbmol_metadata_parent', PARENT / 'scripts/protenix_precision_ops.py')
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    assert len(m.verify(PARENT, PARENT_PIN)['files']) == 20
    return m


def external(op, pin):
    m = op.verify(SOURCE, pin); plan = op.read(SOURCE / 'plan.json')
    op.require(plan['version'] == 'dbmol-separate-affinity-reference-v1' and plan['root'] == str(ROOT)
               and plan['original_results_changed'] is False and plan['optimizer_updates'] == 0,
               'New DBMol reference scope differs')
    op.require(m['cpu_counts'] == COUNTS, 'Closed exact CPU counts differ')
    for row in m['parent_sources']:
        op.require(len(op.verify(row['path'], row['manifest_sha256'])['files']) == row['files'], 'Original source scope changed')
    for row in plan['protected_inputs']: op.check_binding(row)
    return m, plan


def argv(phase, pin, dependency=None):
    op = helpers(); op.require(phase in ('cpu', 'after-cpu', 'gpu'), 'No continuation/optimizer phase exists')
    if phase == 'after-cpu': op.require(bool(re.fullmatch(r'ended\([1-9][0-9]*\)', dependency or '')), 'Exact own-ended dependency required')
    else: op.require(dependency is None, 'CPU/native phase must be independent')
    return ['bsub', *op.resources(phase), '-J', 'mosaic-dbmol-reference10-' + phase,
            '-oo', str(ROOT / phase / 'stdout.%J.log'), '-eo', str(ROOT / phase / 'stderr.%J.log'),
            *(['-w', dependency] if dependency else []), 'bash', '-l', str(SOURCE / 'scripts/dbmol_recovery_stage.lsf'),
            str(SOURCE), pin, phase]


def submit(phase, pin, dependency=None):
    op = helpers(); command = argv(phase, pin, dependency); parent = ROOT / phase; parent.mkdir(exist_ok=False)
    record = {'status': 'unknown', 'phase': phase, 'argv': command, 'dependency': dependency,
              'source_manifest_sha256': pin, 'submitted_utc': op.datetime.now(op.timezone.utc).isoformat()}
    op.write(parent / 'submission-started.json', record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        record.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        jobs = re.findall(r'Job <([1-9][0-9]*)> is submitted', result.stdout)
        if result.returncode == 0 and len(jobs) == 1: record.update(status='submitted', job_id=jobs[0])
    finally: op.write(parent / 'submission.json', record)
    op.require(record['status'] == 'submitted', 'Unknown outcome; inspect before recovery, never duplicate')
    import fcntl
    with (ROOT / 'jobs.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        saved = op.read(ROOT / 'jobs.json') if (ROOT / 'jobs.json').exists() else {}
        op.require(phase not in saved, 'Existing actual job cannot be replaced')
        saved[phase] = record['job_id']; op.atomic(ROOT / 'jobs.json', saved)
    return record


def cpu_cases(op):
    cases = []
    for file in ('tests-torch.xml', 'tests-policy.xml'):
        cases += list(ET.parse(ROOT / 'cpu/payload' / file).getroot().iter('testcase'))
    op.require(dict(Counter(c.get('classname') for c in cases)) == COUNTS
               and all(not any(n.tag in ('failure', 'error', 'skipped') for n in c) for c in cases),
               'Actual allocated CPU cases failed or attribution differs')
    op.require(len({(c.get('classname'), c.get('name')) for c in cases}) == len(cases), 'Duplicate fixture records')
    return len(cases)


def execute(phase, pin):
    op = helpers(); manifest, plan = external(op, pin); parent = ROOT / phase
    for number in range(21):
        if (parent / 'submission.json').exists(): break
        op.require(number < 20, 'Own submission was not committed'); time.sleep(1)
    own = op.read(parent / 'submission.json')
    op.require(own.get('status') == 'submitted' and own.get('job_id') == os.environ.get('LSB_JOBID')
        and own.get('argv') == argv(phase, pin, own.get('dependency')), 'Own job/source/command/resources differ')
    op.require(os.environ.get('LSB_QUEUE') == ('gpul40s' if phase == 'gpu' else 'hpc')
               and os.environ.get('LSB_DJOB_NUMPROC') == ('1' if phase == 'after-cpu' else '4'), 'Real queue/slots differ')
    import recovery_capacity
    recovery_capacity.gate(parent, 'entry')
    payload = parent / 'payload'; payload.mkdir(exist_ok=False)
    command = [PYTHON, str(SOURCE / 'scripts/dbmol_endpoint_precision.py'), '--plan', str(SOURCE / 'plan.json'),
               '--plan-sha256', op.sha(SOURCE / 'plan.json')]
    if phase == 'cpu':
        startup = command + ['--output', str(payload / 'startup'), '--startup-only']
        op.write(payload / 'startup-command.json', {'argv': startup}); subprocess.run(startup, check=True, timeout=180)
        startup = plan['reference_argv'] + ['--startup-only', '--startup-output', str(payload / 'reference-startup.json')]
        op.write(payload / 'reference-startup-command.json', {'argv': startup}); subprocess.run(startup, check=True, timeout=180)
        for python, names, label in ((TORCH, ['test_dbmol_reference.py'], 'torch'),
                (PYTHON, ['test_dbmol_endpoint_precision.py', 'test_recovery_capacity.py'], 'policy')):
            args = [python, '-m', 'pytest', *[str(SOURCE / 'tests' / n) for n in names],
                '--rootdir', str(SOURCE), '--import-mode=importlib', '-q', '-o', 'addopts=', '-p', 'no:cacheprovider',
                '--junitxml=' + str(payload / ('tests-' + label + '.xml'))]
            op.write(payload / ('command-' + label + '.json'), {'argv': args})
            with (payload / ('tests-' + label + '.log')).open('x') as f:
                subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1500)
        op.write(payload / 'assessment.json', {'status': 'passed', 'cases': cpu_cases(op),
                  'source_manifest_sha256': pin, 'optimizer_updates': 0})
    elif phase == 'after-cpu':
        producer = op.read(ROOT / 'cpu/submission.json'); job = producer['job_id']
        op.require(own['dependency'] == 'ended(' + job + ')', 'Callback bound to wrong predecessor')
        q = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=40)
        op.write(payload / 'cpu-final-scheduler.json', {'returncode': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr})
        compact = ''.join(q.stdout.split())
        op.require(q.returncode == 0 and compact.count('Job<' + job + '>,') == 1
                   and re.findall(r'Status<([^>]+)>', compact) == ['DONE'], 'Actual CPU scheduler success missing')
        op.lsf_success((ROOT / 'cpu' / ('stdout.' + job + '.log')).read_text(), job,
                       'mosaic-dbmol-reference10-cpu', 'hpc')
        op.require(op.read(ROOT / 'cpu/payload/assessment.json')['status'] == 'passed', 'CPU assessment failed')
        proofs = [op.binding(p) for p in (ROOT / 'cpu').rglob('*') if p.is_file() and p.name.endswith(('.json', '.xml', '.log'))]
        op.write(ROOT / 'cpu-certificate.json', {'status': 'passed', 'cases': cpu_cases(op),
                  'source_manifest_sha256': pin, 'proofs': proofs})
        recovery_capacity.gate(parent, 'dispatch'); submit('gpu', pin)
        op.write(payload / 'assessment.json', {'status': 'passed_CPU_and_native_reference_submitted', 'optimizer_jobs_submitted': False})
    else:
        cert = op.read(ROOT / 'cpu-certificate.json')
        op.require(cert['status'] == 'passed' and cert['source_manifest_sha256'] == pin, 'Own full CPU proof missing')
        for row in cert['proofs']: op.check_binding(row)
        command += ['--output', str(payload / 'native')]
        op.write(payload / 'native-command.json', {'argv': command})
        try:
            with (payload / 'native.log').open('x') as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=5600)
            result = op.read(payload / 'native/assessment.json')
            op.require(result['status'] == 'passed_affinity_reference_only' and result['can_submit_continuation'] is False,
                       'Full native reference evidence incomplete')
            op.write(payload / 'assessment.json', {'status': 'passed_affinity_reference_only',
                      'optimizer_jobs_submitted': False, 'native_assessment': op.binding(payload / 'native/assessment.json')})
        finally:
            external(op, pin)
            op.require(op.owned_bytes(ROOT) + op.owned_bytes(SOURCE.parent) <= 512 * 1024**2,
                       'Whole declared512MiB recovery output budget exceeded')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=('cpu', 'after-cpu', 'gpu'), required=True)
    p.add_argument('--pin', required=True); a = p.parse_args(); execute(a.phase, a.pin)


if __name__ == '__main__': main()
