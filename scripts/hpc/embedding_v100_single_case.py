"""Continue only the most advanced frozen-VAE trajectory on V100."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

WORK = Path('/work3/s234818/latentmol')
NAME = 'source154-single-case-v100-attempt001'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def check_state(asset):
    p = Path(asset['path']).resolve()
    if not p.is_relative_to(WORK / 'runs') or sha(p) != asset['sha256']:
        raise ValueError('Saved state changed or lies outside the run directory')

def submit(args, command, release, control, label):
    intent = control / (label + '-intent.json')
    receipt = control / (label + '-receipt.json')
    if intent.exists() or receipt.exists():
        raise FileExistsError('Submission already attempted')
    script = ('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(release))
              + '\nexec ' + shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1',
                    'LATENTMOL_SOURCE_ARCHIVE=' + str(release) + '.tar.gz',
                    'bash', 'scripts/hpc/boltz_run.sh', *command]) + '\n')
    write(intent, {'bsub': args, 'command': command})
    try:
        r = subprocess.run(args, input=script, text=True, capture_output=True, timeout=60)
    except Exception as e:
        write(receipt, {'status': 'ambiguous_no_retry', 'error': str(e)})
        raise
    match = re.search(r'Job <([0-9]+)>', r.stdout)
    result = {'status': 'submitted' if r.returncode == 0 and match else 'failed_no_retry',
              'job_id': match.group(1) if match else None, 'returncode': r.returncode,
              'stdout': r.stdout, 'stderr': r.stderr}
    write(receipt, result)
    if result['status'] != 'submitted':
        raise RuntimeError('Submission failed; inspect receipt')
    return result

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--plan-sha256', required=True)
    p.add_argument('--after', required=True)
    p.add_argument('--job')
    a = p.parse_args()
    release = Path(__file__).resolve().parents[2]
    cfg_path = release / 'configs/embedding_v100_single_case.json'
    if (not os.environ.get('LSB_JOBID') or release != WORK / 'releases/source-154-single-case-v100-attempt001'
            or os.environ.get('LATENTMOL_SOURCE_SHA256') != a.source_sha256 or sha(cfg_path) != a.plan_sha256):
        raise ValueError('Wrong allocated release')
    spec = importlib.util.spec_from_file_location('_operator', release / 'scripts/hpc/embedding_operator.py')
    op = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(op)
    op.authenticate(release, a.source_sha256, a.plan_sha256, 'configs/embedding_v100_single_case.json')
    cfg = read(cfg_path)
    worker_release = Path(cfg['worker_release'])
    op.authenticate(worker_release, cfg['worker_source_sha256'], cfg['worker_plan_sha256'],
                    'configs/embedding_v100_serial_continuation.json')
    control = WORK / 'submissions' / NAME
    control.mkdir(parents=True, exist_ok=True)
    evidence = control / ('after-' + a.after + '.json')
    if evidence.exists():
        raise FileExistsError('Callback already attempted')
    if a.after == 'initial':
        state, update, sequence = cfg['initial_state'], cfg['initial_update'], 0
    else:
        previous = read(control / (a.after + '-worker-receipt.json'))
        request = read(control / (a.after + '-request.json'))
        if previous['status'] != 'submitted' or previous['job_id'] != a.job:
            raise ValueError('Wrong prior worker')
        r = subprocess.run(['bjobs', '-a', '-json', '-o', 'jobid stat job_name', a.job],
                           text=True, capture_output=True, timeout=30)
        rows = json.loads(r.stdout).get('RECORDS', []) if r.returncode == 0 and not r.stderr.strip() else []
        label = NAME + '-' + a.after
        if len(rows) != 1 or rows[0].get('JOBID') != a.job:
            raise ValueError('Scheduler could not identify prior worker')
        row = rows[0]
        if row.get('ERROR') == 'Job <' + a.job + '> is not found':
            log = (WORK / 'logs' / (label + '.' + a.job + '.out')).read_text(errors='replace')
            if ('Subject: Job ' + a.job + ': <' + label + '> in cluster <dcc> Done' not in log
                    or 'Successfully completed.' not in log or ' SOURCE=' + str(worker_release) not in log):
                raise ValueError('Missing terminal evidence for prior worker')
        elif row.get('STAT') != 'DONE' or row.get('JOB_NAME') != label:
            raise ValueError('Prior worker is not exact-name DONE')
        result = read(WORK / 'runs/source152-v100-serial-continuation-attempt001/jobs' / ('single-' + a.after) / 'result.json')
        core = result['result']
        if (result.get('status') != 'completed_v100_segment' or result.get('job_id') != a.job
                or result.get('source_sha256') != cfg['worker_source_sha256']
                or result.get('plan_sha256') != cfg['worker_plan_sha256']
                or core.get('case_id') != cfg['case'] or core.get('starting_update') != request['start_update']
                or core.get('completed_update', 0) <= request['start_update'] or core.get('scientific_stall') is not False
                or core['frozen_after'].get('frozen_state_unchanged') is not True
                or core['frozen_after'].get('context_unchanged') is not True):
            raise ValueError('Prior segment did not pass its original checks')
        state, update, sequence = core['current_state'], core['completed_update'], request['sequence']
    check_state(state)
    if update >= cfg['target_updates']:
        write(evidence, {'status': 'trajectory_complete_native_followup_pending', 'update': update, 'state': state})
        return
    sequence += 1
    node = 'segment-%06d' % sequence
    req = {'sequence': sequence, 'case': cfg['case'], 'start_update': update,
           'max_updates': min(cfg['max_updates_per_segment'], cfg['target_updates'] - update), 'state': state}
    request_path = control / (node + '-request.json')
    write(request_path, req)
    worker_config = worker_release / 'configs/embedding_v100_serial_continuation.json'
    output = WORK / 'runs/source152-v100-serial-continuation-attempt001/jobs' / ('single-' + node)
    output.parent.mkdir(parents=True, exist_ok=True)
    name = NAME + '-' + node
    worker_cmd = [str(WORK / 'env/boltz311/bin/python'), str(worker_release / 'scripts/hpc/embedding_v100_serial_worker.py'),
                  '--config', str(worker_config), '--request', str(request_path), '--request-sha256', sha(request_path),
                  '--source-sha256', cfg['worker_source_sha256'], '--plan-sha256', cfg['worker_plan_sha256'], '--output', str(output)]
    worker_args = ['bsub', '-J', name, '-q', 'gpuv100', '-n', '4', '-R', 'span[hosts=1]',
                   '-R', 'rusage[mem=16GB]', '-R', 'select[gpu32gb]', '-W', '24:00',
                   '-oo', str(WORK / 'logs' / (name + '.%J.out')), '-eo', str(WORK / 'logs' / (name + '.%J.err')),
                   '-gpu', 'num=1:mode=exclusive_process']
    worker = submit(worker_args, worker_cmd, worker_release, control, node + '-worker')
    cbname = name + '-after'
    cb_cmd = [str(WORK / 'env/boltz311/bin/python'), str(release / 'scripts/hpc/embedding_v100_single_case.py'),
              '--source-sha256', a.source_sha256, '--plan-sha256', a.plan_sha256, '--after', node, '--job', worker['job_id']]
    cb_args = ['bsub', '-J', cbname, '-q', 'hpc', '-n', '1', '-W', '0:30', '-R', 'rusage[mem=2GB]',
               '-R', 'span[hosts=1]', '-w', 'ended(' + worker['job_id'] + ')',
               '-oo', str(WORK / 'logs' / (cbname + '.%J.out')), '-eo', str(WORK / 'logs' / (cbname + '.%J.err'))]
    callback = submit(cb_args, cb_cmd, release, control, node + '-callback')
    write(evidence, {'status': 'next_segment_submitted', 'case': cfg['case'], 'starting_update': update,
                     'state': state, 'worker': worker, 'callback': callback})

if __name__ == '__main__':
    main()
