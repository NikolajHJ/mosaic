"""Launcher-only all-work3 adoption; scientific source625 remains unchanged."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'restart-lrreset-04'
CONTROL = ROOT / 'all-work3-launch-02'
SOURCE = BASE / 'restart-lrreset-allwork3-launch-02/source'
SCIENCE = BASE / 'restart-lrreset-release-04/source'
SCI_MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
SCI_CPU = '6a87093f3da0d1ca0694c7152de84fdeed308251f709c685b526d978b1455e23'
VERSION = 'jq1-lrreset-all-work3-launch-adoption-v1'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
ARMS = ('protenix', 'boltz2', 'boltz2_chemistry')
EXPECTED_TESTS = 26
sys.path[:0] = [str(SCIENCE / 'scripts'), str(SCIENCE / 'src')]
import fixed25_lrreset_ops as science


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def proof(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.resolve() == p.absolute(), 'Unsafe proof path')
    return {'path': str(p), 'sha256': sha(p)}


def exclusive(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def atomic(path, value):
    path = Path(path); temporary = path.with_name(path.name + '.tmp')
    exclusive(temporary, value); os.replace(temporary, path)


def output_parent(arm):
    require(arm in ARMS, 'Unknown arm')
    return ROOT / ('run-' + arm)


def remaining(projections, current):
    require(set(projections) == set(current) == set(ARMS), 'All three projections/current outputs required')
    require(all(type(v) is int and v >= 1024**3 for v in projections.values()), 'Invalid full projection')
    require(all(type(current[a]) is int and 0 <= current[a] <= projections[a] for a in ARMS), 'Invalid/exceeded output projection')
    return sum(projections[a] - current[a] for a in ARMS)


def storage_gate(projections, path, *, phase):
    require(phase in ('dispatch', 'entry'), 'Unknown full budget phase')
    path = science.storage.receipt_dir(path)
    record = {'version': VERSION, 'status': 'failed', 'phase': phase,
              'utc': datetime.now(timezone.utc).isoformat(), 'placement': {a: 'work3' for a in ARMS},
              'projections': projections, 'home_output_writes': False,
              'scope': 'All three NEW output projections on work3; failed home inventory is preserved, never relabelled',
              'filesystems': {}}
    try:
        current = {a: science.storage.tree_bytes(output_parent(a)) for a in ARMS}
        outstanding = remaining(projections, current)
        raw = subprocess.run(['getquota_work3.sh'], capture_output=True, text=True, check=True, timeout=60).stdout
        observed = science.storage.inode_observation(os.statvfs(ROOT))
        gate = science.storage.gate_record('work3', raw,
            outstanding + science.storage.CONCURRENT_ANALYSIS_ALLOWANCE,
            science.storage.stale_work_bytes(), shutil.disk_usage(ROOT).free, observed)
        record.update(current_outputs=current, outstanding_native_bytes=outstanding,
                      concurrent_analysis_allowance_bytes=science.storage.CONCURRENT_ANALYSIS_ALLOWANCE,
                      status=gate['status'], filesystems={'work3': gate})
    except Exception as error:
        record['error'] = repr(error)
        raise
    finally:
        exclusive(path / 'lrreset-storage-gate.json', record)
    require(record['status'] == 'passed', 'Full all-work3 projection/reserve/inode gate failed')
    return record


def verify_failure():
    pin = read(CONTROL / 'inherited-failure.json')
    require(pin['job'] == '29792418' and pin['status'] == 'EXIT' and pin['queue'] == 'hpc', 'Wrong failed dispatch predecessor')
    for row in pin['proofs']:
        require(sha(row['path']) == row['sha256'], 'Failed dispatch evidence changed')
    require(pin['scheduler'] in pin['proofs'] and pin['gate'] in pin['proofs'], 'Incomplete failure proof binding')
    gate = read(pin['gate']['path'])
    require(sha(pin['gate']['path']) == pin['gate']['sha256'] and gate['status'] == 'failed'
            and 'TimeoutExpired' in gate['error'], 'Original failed home gate must remain failed')
    scheduler = Path(pin['scheduler']['path']).read_text(); compact = re.sub(r'\s+', '', scheduler)
    require(re.findall(r'Job<([^>]+)>', compact) == ['29792418']
            and re.findall(r'Status<([^>]+)>', compact) == ['EXIT']
            and re.findall(r'Queue<([^>]+)>', compact) == ['hpc'], 'Actual failed callback scheduler proof missing')
    lsf = (ROOT / 'after-ready-boltz2_chemistry/stdout.29792418.log').read_text()
    require('Subject: Job 29792418:' in lsf and 'Exited with exit code 1.' in lsf, 'Actual failed callback LSF missing')
    return pin


def inherited():
    science.verify_cpu(SCI_MANIFEST, SCI_CPU)
    science.validate_configs()
    failure = verify_failure()
    prior = read(CONTROL / 'previous-launcher-failure.json')
    require(prior['cpu'] == '29792659' and prior['callback'] == '29792660' and prior['callback_status'] == 'EXIT', 'Wrong failed launcher predecessor')
    for row in prior['proofs']:
        require(sha(row['path']) == row['sha256'], 'Failed launcher predecessor proof changed')
    values = {}; readiness = {}
    for arm in ARMS:
        path = ROOT / ('decision-' + arm + '.json')
        decision = read(path)
        assessment = science.authenticate_ready(arm, SCI_MANIFEST, decision)
        values[arm] = science.validate_projection(assessment)
        readiness[arm] = proof(path)
    original = science.verify_finished_predecessors()
    return {'scientific_source_manifest_sha256': SCI_MANIFEST, 'scientific_cpu_sha256': SCI_CPU,
            'projections': values, 'readiness': readiness, 'finished_predecessors': original,
            'failed_home_dispatch': failure, 'previous_launcher_failure': prior}


def verify_cpu(manifest):
    path = CONTROL / 'cpu-certificate.json'; c = read(path)
    require(c['status'] == 'passed' and c['manifest'] == manifest and c['tests'] == EXPECTED_TESTS,
            'Own launcher CPU certificate missing/changed')
    for p in c['proofs']:
        require(sha(p['path']) == p['sha256'], 'Own CPU proof changed')
    return proof(path)


def long_command(arm, manifest, cpu_sha):
    parent = output_parent(arm)
    return ['bsub', '-q', 'gpul40s', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]',
            '-W', '23:00', '-gpu', 'num=1:mode=exclusive_process', '-J', 'mosaic-lrreset04-work3-' + arm,
            '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log'),
            'bash', '-l', str(SOURCE / 'all_work3_launch.lsf'), str(SOURCE), manifest, 'run', arm, cpu_sha]


def cpu_command(name, manifest, producer='0'):
    require(name in ('cpu', 'after-cpu'), 'Unknown CPU phase')
    parent = CONTROL / name
    command = ['bsub', '-q', 'hpc', '-n', '2' if name == 'cpu' else '1', '-R', 'span[hosts=1]',
               '-R', 'rusage[mem=4GB]' if name == 'cpu' else 'rusage[mem=2GB]',
               '-W', '00:20' if name == 'cpu' else '00:30', '-J', 'mosaic-lrreset04-work3-' + name,
               '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if name == 'after-cpu':
        require(producer.isdigit() and producer != '0', 'CPU producer required')
        command += ['-w', 'ended(' + producer + ')']
    return command + ['bash', '-l', str(SOURCE / 'all_work3_launch.lsf'), str(SOURCE), manifest,
                      'cpu' if name == 'cpu' else 'certify', 'none', '0']


def driver_command(arm):
    command = science.driver_command(arm, 'run', SCI_MANIFEST, SCI_CPU)
    command[command.index('--output') + 1] = str(output_parent(arm) / 'payload/study')
    return command


def submit_long(arm, manifest, cpu_sha):
    parent = output_parent(arm)
    require(not parent.exists(), 'Existing/unknown long outcome; no duplicate')
    parent.mkdir(parents=True, exist_ok=False)
    command = long_command(arm, manifest, cpu_sha)
    exclusive(parent / 'submission-started.json', {'args': command, 'outcome': 'unknown'})
    row = {'version': VERSION, 'outcome': 'unknown', 'arm': arm, 'args': command,
           'output_parent': str(parent), 'learning_rate': science.RATES[arm],
           'source_manifest_sha256': SCI_MANIFEST, 'launch_manifest_sha256': manifest,
           'cpu_certificate_sha256': SCI_CPU, 'launch_cpu_certificate_sha256': cpu_sha,
           'config_sha256': science.sha(science.config_path(arm))}
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=60)
        row.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
        ids = re.findall(r'Job <(\d+)>', r.stdout)
        if r.returncode == 0 and len(ids) == 1:
            row.update(outcome='submitted', job_id=ids[0])
    except Exception as error:
        row['error'] = repr(error)
        raise
    finally:
        exclusive(parent / 'submission.json', row)
    require(row['outcome'] == 'submitted', 'Unknown bsub outcome preserved; no blind duplicate')
    return row


def dispatch(manifest):
    native = inherited(); cpu = verify_cpu(manifest)
    with (ROOT / 'dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not (ROOT / 'long-dispatch-started.json').exists()
                and not (ROOT / 'long-submissions.json').exists()
                and not any(output_parent(a).exists() for a in ARMS)
                and not science.storage.HOME_ROOT.exists(), 'Existing/unknown dispatch; no duplicate')
        gate = storage_gate(native['projections'], CONTROL / 'dispatch-quota', phase='dispatch')
        exclusive(CONTROL / 'adoption.json', {'version': VERSION, 'launcher_manifest': manifest,
            'launcher_cpu': cpu, **native})
        marker = {'version': VERSION, 'source_manifest_sha256': SCI_MANIFEST,
                  'launch_manifest_sha256': manifest, 'launcher_cpu': cpu,
                  'adoption': proof(CONTROL / 'adoption.json'),
                  'placement': {a: 'work3' for a in ARMS}, **native, 'storage_gate': gate}
        exclusive(ROOT / 'long-dispatch-started.json', marker)
        jobs = []
        for arm in ARMS:
            jobs.append(submit_long(arm, manifest, cpu['sha256']))
            atomic(ROOT / 'long-submissions.json', {'version': VERSION, 'jobs': jobs,
                'all_three_submitted': len(jobs) == 3, 'source_manifest_sha256': SCI_MANIFEST,
                'launch_manifest_sha256': manifest})
        return jobs


def actual_cpu_done(job, parent):
    for attempt in range(12):
        result = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True, timeout=30)
        raw = result.stdout
        with (parent / ('scheduler-observation-%02d.txt' % attempt)).open('x') as stream:
            stream.write(raw)
        with (parent / ('scheduler-observation-%02d-stderr.txt' % attempt)).open('x') as stream:
            stream.write(result.stderr)
        compact = re.sub(r'\s+', '', raw)
        require(result.returncode == 0 and re.findall(r'Job<([^>]+)>', compact) == [job]
                and re.findall(r'Queue<([^>]+)>', compact) == ['hpc'], 'Actual own CPU job/queue missing or malformed')
        statuses = re.findall(r'Status<([^>]+)>', compact)
        if statuses == ['DONE']:
            return raw
        require(statuses in (['RUN'], ['PEND']), 'Actual own CPU failed or status unknown')
        time.sleep(2)
    raise ValueError('Actual own CPU DONE/hpc not observed within bounded wait')


def certify(manifest):
    own = read(CONTROL / 'cpu/submission.json'); job = own['job_id']
    after = read(CONTROL / 'after-cpu/submission.json')
    require(own['outcome'] == after['outcome'] == 'submitted'
            and own['args'] == cpu_command('cpu', manifest)
            and after['args'] == cpu_command('after-cpu', manifest, job)
            and after['job_id'] == os.environ.get('LSB_JOBID')
            and os.environ.get('LSB_QUEUE') == 'hpc', 'CPU/callback actual submission differs')
    raw = actual_cpu_done(job, CONTROL / 'cpu')
    lsf = CONTROL / ('cpu/stdout.' + job + '.log')
    require('Subject: Job ' + job + ':' in lsf.read_text() and 'Successfully completed.' in lsf.read_text(), 'Own CPU LSF success required')
    scheduler = CONTROL / 'cpu/scheduler-verified-final.txt'
    with scheduler.open('x') as stream:
        stream.write(raw)
    xml = CONTROL / 'cpu/payload/tests.xml'; tree = ET.parse(xml)
    require(len(list(tree.iter('testcase'))) == EXPECTED_TESTS and not any(list(tree.iter(t)) for t in ['failure', 'error', 'skipped']), 'Own full launcher contracts failed')
    exclusive(CONTROL / 'cpu-certificate.json', {'status': 'passed', 'manifest': manifest,
        'tests': EXPECTED_TESTS, 'job': job, 'proofs': [proof(p) for p in [xml, lsf, scheduler, CONTROL / 'cpu/payload/tests.log']]})
    return dispatch(manifest)


def entry(arm, manifest, cpu_sha):
    own = read(output_parent(arm) / 'submission.json')
    require(os.environ.get('LSB_QUEUE') == 'gpul40s'
            and own['outcome'] == 'submitted' and own['job_id'] == os.environ.get('LSB_JOBID')
            and own['args'] == long_command(arm, manifest, cpu_sha), 'Actual long submission differs')
    cpu = verify_cpu(manifest); require(cpu['sha256'] == cpu_sha, 'Own CPU pin differs')
    marker = read(ROOT / 'long-dispatch-started.json')
    require(marker['launch_manifest_sha256'] == manifest and marker['source_manifest_sha256'] == SCI_MANIFEST
            and sha(marker['adoption']['path']) == marker['adoption']['sha256'], 'Adoption/dispatch pin differs')
    native = inherited()
    require(native == {k: marker[k] for k in native}, 'Original scientific/native bindings changed')
    parent = output_parent(arm)
    storage_gate(native['projections'], parent / 'entry-quota', phase='entry')
    payload = parent / 'payload'; payload.mkdir()
    command = driver_command(arm)
    exclusive(payload / 'driver-invocation.json', {'args': command, 'source_manifest_sha256': SCI_MANIFEST,
        'launch_manifest_sha256': manifest, 'launch_cpu_sha256': cpu_sha, 'scientific_cpu_sha256': SCI_CPU})
    with (payload / 'native.log').open('x') as stream:
        subprocess.run(command, cwd=SCIENCE, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True); p.add_argument('--phase', choices=['certify', 'run'], required=True)
    p.add_argument('--arm', default='none'); p.add_argument('--cpu-sha', default='0')
    a = p.parse_args(); require(os.environ.get('LSB_JOBID'), 'Allocated launcher phase required')
    if a.phase == 'certify':
        print(json.dumps(certify(a.manifest)))
    else:
        entry(a.arm, a.manifest, a.cpu_sha)


if __name__ == '__main__':
    main()
