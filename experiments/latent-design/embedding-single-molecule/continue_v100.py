"""Resume the epoch-36 embedding trajectory in short, restartable V100 segments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import torch

WORK = Path('/work3/s234818/latentmol')
WORKER_RELEASE = WORK / 'releases/source-152-v100-serial-continuation-attempt001'
WORKER_SOURCE = 'b67ea113b011f508aba6814ed3fece6dcb54c64a32f377d22616337f350516b3'
WORKER_PLAN = '2da5f1e4be1ea6720a41c290a65147ef3a28187437d0058c69afa2b8cdcbcd1a'
JOB_ROOT = WORK / 'runs/source152-v100-serial-continuation-attempt001/jobs'
INITIAL = JOB_ROOT / 'single-segment-000001/trajectory'
INITIAL_UPDATE = 2902
INITIAL_SHA = 'd9d6acdfd8363536113151c16802f2eae24cd69884d5ea799cd571933904e9fa'
TARGET = 10000
SEGMENT_UPDATES = 500

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write_new(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())

def committed(trajectory, start):
    """Use only complete journal records whose checkpoint bytes match."""
    log = trajectory / 'segments' / ('from-%06d' % start) / 'commits.jsonl'
    rows = []
    if log.exists():
        with log.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                p = Path(row['current_state']['path'])
                if p.parent != trajectory / 'states' or not p.is_file() or sha(p) != row['current_state']['sha256']:
                    break
                rows.append(row)
    return rows[-1] if rows else None

def build_input(source, update, expected_sha, dest):
    """Clone the committed state and its authenticated cache, never editing source."""
    src_state = source / 'states' / ('state-%06d.pt' % update)
    if sha(src_state) != expected_sha:
        raise ValueError('Checkpoint bytes do not match the committed journal')
    state = torch.load(src_state, map_location='cpu', weights_only=True)
    if (state['case_id'] != 'case-001' or state['update'] != update or not state['gate_binding_verified']
            or state['counts']['iterations'] != update or not torch.isfinite(state['z']).all()
            or not torch.isfinite(state['z0']).all() or state['counts']['consecutive_noop'] >= 25):
        raise ValueError('Checkpoint state is incompatible or nonfinite')
    manifest = read(source / 'manifest.json')
    if manifest['sha256'] != state['signature_sha256'] or digest(manifest['signature']) != manifest['sha256']:
        raise ValueError('Checkpoint signature differs from its trajectory')
    if (state['z'] - state['z0']).double().norm().item() > 3.00001:
        raise ValueError('Checkpoint is outside the original trust radius')
    for item in [state['best_valid']['tensors'], state['last_observation']['record'], state['last_observation']['tensors']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Referenced checkpoint artifact changed')
    chain = hashlib.sha256(b'').hexdigest()
    entries = []
    index = source / 'hard-cache/index.jsonl'
    with index.open() as f:
        for _ in range(state['hard_cache_entries']):
            row = json.loads(next(f))
            if row['previous'] != chain or sha(source / 'hard-cache' / (row['key'] + '.json')) != row['sha256']:
                raise ValueError('Hard-cache chain is corrupt')
            chain = digest(row)
            entries.append(row)
    if chain != state['hard_cache_chain'] or state['last_hard_graph_sha256'] not in {r['key'] for r in entries}:
        raise ValueError('Hard-cache chain does not match the checkpoint')
    dest.mkdir(parents=True, exist_ok=False)
    for directory in ('states', 'segments', 'observations', 'hard-cache', 'best'):
        (dest / directory).mkdir()
    shutil.copy2(source / 'manifest.json', dest / 'manifest.json')
    shutil.copy2(src_state, dest / 'states' / src_state.name)
    with (dest / 'hard-cache/index.jsonl').open('x') as f:
        for row in entries:
            f.write(json.dumps(row, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())
    for row in entries:
        shutil.copy2(source / 'hard-cache' / (row['key'] + '.json'), dest / 'hard-cache' / (row['key'] + '.json'))
    return {'path': str(dest / 'states' / src_state.name), 'sha256': expected_sha}

def submit(args, command, release, intent, receipt):
    if intent.exists() or receipt.exists():
        raise FileExistsError('Submission already attempted')
    script = ('#!/bin/bash\nset -euo pipefail\ncd ' + shlex.quote(str(release)) + '\nexec '
              + shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1', 'LATENTMOL_SOURCE_ARCHIVE=' + str(WORKER_RELEASE) + '.tar.gz',
                            'bash', 'scripts/hpc/boltz_run.sh', *command]) + '\n')
    write_new(intent, {'bsub': args, 'command': command})
    try:
        r = subprocess.run(args, input=script, capture_output=True, text=True, timeout=60)
    except Exception as e:
        write_new(receipt, {'status': 'ambiguous_no_retry', 'error': str(e)})
        raise
    import re
    m = re.search(r'Job <([0-9]+)>', r.stdout)
    value = {'status': 'submitted' if r.returncode == 0 and m else 'failed_no_retry',
             'job_id': m.group(1) if m else None, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    write_new(receipt, value)
    if value['status'] != 'submitted':
        raise RuntimeError('Submission failed; inspect receipt')
    return value

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--after', type=int, default=0)
    p.add_argument('--job')
    a = p.parse_args()
    if not os.environ.get('LSB_JOBID') or not a.run.isalnum():
        raise ValueError('Run inside an allocated job with an alphanumeric run ID')
    root = JOB_ROOT / ('recovery-' + a.run)
    control = root / 'control'
    control.mkdir(parents=True, exist_ok=True)
    event = control / ('after-%06d.json' % a.after)
    if event.exists():
        raise FileExistsError('Continuation has already been attempted')
    lock = control / ('after-%06d.lock' % a.after)
    write_new(lock, {'controller_job_id': os.environ['LSB_JOBID']})
    if a.after == 0:
        queue = subprocess.run(['bjobs', '-u', 's234818', '-o', 'jobid stat job_name'], capture_output=True, text=True, timeout=30)
        if queue.returncode or queue.stderr.strip():
            raise RuntimeError('Cannot check for another active writer')
        for line in queue.stdout.splitlines()[1:]:
            parts = line.split(maxsplit=2)
            if len(parts) == 3 and parts[0] != os.environ['LSB_JOBID'] and parts[1] in ('RUN', 'PEND', 'PSUSP', 'USUSP', 'SSUSP'):
                if parts[2].startswith(('embedding-one-', 'source154-single-case-v100-', 'source151-v100-continuation-')):
                    raise RuntimeError('Another single-molecule writer or callback is active: ' + line)
        source, update, state_sha = INITIAL, INITIAL_UPDATE, INITIAL_SHA
        if not any(r['update'] == update and r['current_state']['sha256'] == state_sha
                   for r in [committed(source, 1899)] if r is not None):
            raise ValueError('Initial checkpoint is not the last committed update')
    else:
        previous = read(control / ('segment-%06d-worker.json' % a.after))
        request = read(control / ('segment-%06d-request.json' % a.after))
        if not a.job or previous['job_id'] != a.job:
            raise ValueError('Previous job identity mismatch')
        source = root / ('segment-%06d' % a.after) / 'trajectory'
        result_path = source.parent / 'result.json'
        result = read(result_path) if result_path.exists() and result_path.stat().st_size else None
        if result and result.get('status') == 'failed_not_passed':
            problem = str(result.get('error', '')).lower()
            if any(x in problem for x in ('nonfinite', 'nan', 'frozen', 'shape', 'incompatible', 'detached')):
                raise ValueError('Scientific or input failure requires inspection: ' + problem)
        row = committed(source, request['start_update'])
        if not row or row['update'] <= request['start_update']:
            raise ValueError('No new committed update; automatic retry stopped')
        update, state_sha = row['update'], row['current_state']['sha256']
        if result and result.get('status') == 'completed_v100_segment':
            if result['job_id'] != a.job or result['result']['completed_update'] != update or result['result']['scientific_stall']:
                raise ValueError('Successful segment receipt conflicts with checkpoint')
            reason = 'completed'
        else:
            name = 'embedding-one-' + a.run + '-%06d' % a.after
            logs = []
            for ext in ('out', 'err'):
                f = WORK / 'logs' / (name + '.' + a.job + '.' + ext)
                if f.exists():
                    logs.append(f.read_text(errors='replace')[-20000:])
            text = '\n'.join(logs)
            if not any(token in text for token in ('Communication error on send', 'TERM_RUNLIMIT', 'Stale file handle', 'Input/output error')):
                raise ValueError('Unexpected worker failure; automatic retry stopped')
            reason = 'recoverable_io_or_walltime'
    if update == TARGET:
        write_new(event, {'status': 'target_reached_native_followup_pending', 'update': update, 'checkpoint_sha256': state_sha})
        return
    if update > TARGET:
        raise ValueError('Checkpoint exceeds target')
    next_segment = a.after + 1
    dest = root / 'inputs' / ('segment-%06d' % next_segment)
    state = build_input(source, update, state_sha, dest)
    request_path = control / ('segment-%06d-request.json' % next_segment)
    request = {'case': 'case-001', 'start_update': update, 'max_updates': min(SEGMENT_UPDATES, TARGET - update), 'state': state}
    write_new(request_path, request)
    name = 'embedding-one-' + a.run + '-%06d' % next_segment
    output = root / ('segment-%06d' % next_segment)
    cmd = [str(WORK / 'env/boltz311/bin/python'), str(WORKER_RELEASE / 'scripts/hpc/embedding_v100_serial_worker.py'),
           '--config', str(WORKER_RELEASE / 'configs/embedding_v100_serial_continuation.json'), '--request', str(request_path),
           '--request-sha256', sha(request_path), '--source-sha256', WORKER_SOURCE, '--plan-sha256', WORKER_PLAN, '--output', str(output)]
    worker_args = ['bsub', '-J', name, '-q', 'gpuv100', '-n', '4', '-R', 'span[hosts=1]', '-R', 'rusage[mem=16GB]',
                   '-R', 'select[gpu32gb]', '-W', '24:00', '-oo', str(WORK / 'logs' / (name + '.%J.out')),
                   '-eo', str(WORK / 'logs' / (name + '.%J.err')), '-gpu', 'num=1:mode=exclusive_process']
    worker = submit(worker_args, cmd, WORKER_RELEASE, control / ('segment-%06d-worker-intent.json' % next_segment),
                    control / ('segment-%06d-worker.json' % next_segment))
    cbname = name + '-after'
    cb_cmd = [str(WORK / 'env/boltz311/bin/python'), str(Path(__file__).resolve()), '--run', a.run,
              '--after', str(next_segment), '--job', worker['job_id']]
    cb_args = ['bsub', '-J', cbname, '-q', 'hpc', '-n', '1', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]',
               '-W', '0:30', '-w', 'ended(' + worker['job_id'] + ')', '-oo', str(WORK / 'logs' / (cbname + '.%J.out')),
               '-eo', str(WORK / 'logs' / (cbname + '.%J.err'))]
    callback = submit(cb_args, cb_cmd, WORKER_RELEASE, control / ('segment-%06d-callback-intent.json' % next_segment),
                      control / ('segment-%06d-callback.json' % next_segment))
    write_new(event, {'status': 'submitted', 'from_update': update, 'reason': reason if a.after else 'initial_recovery',
                      'worker': worker, 'callback': callback})

if __name__ == '__main__':
    main()
