"""Authenticate one job-exclusive GPU that permits this job's CUDA processes.

LSF job exclusivity and NVIDIA compute exclusivity are different policies. The
Boltz pilot has an unchanged native PyTorch process and a separate JAX chemistry
process, so it requires shared/default compute with j_exclusive=yes. This module
never changes GPU mode. Its bounded canary is engineering evidence only; native
model, raw-gradient, paired-chemistry and runtime gates remain mandatory.
"""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import uuid


VERSION = 'job-exclusive-multi-process-GPU-v1'
GPU_RESOURCE = 'num=1:mode=shared:j_exclusive=yes:mps=no'
TORCH_PYTHON = '/work3/s234818/latentmol/env/boltz311/bin/python'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, record):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require_allocation(environment):
    job = environment.get('LSB_JOBID', '')
    require(re.fullmatch(r'[1-9]\d*', job) is not None, 'Actual allocated job is required')
    require(environment.get('LSB_QUEUE') == 'gpul40s'
            and environment.get('LSB_DJOB_NUMPROC') == '8',
            'Boltz chemistry requires its declared eight-slot L40S allocation')
    visible = environment.get('CUDA_VISIBLE_DEVICES', '')
    require(re.fullmatch(r'(?:\d+|GPU-[a-fA-F0-9-]+)', visible) is not None,
            'Require exactly one LSF-selected physical CUDA device')
    require(not sys.flags.optimize, 'Assertions must remain enabled')
    return job, visible


def scheduler_gpu_contract(raw, job):
    compact = re.sub(r'\s+', '', raw)
    require(re.findall(r'Job<([1-9]\d*)>', compact) == [job],
            'Actual scheduler proof must identify only this job')
    require('Status<RUN>' in compact and 'Queue<gpul40s>' in compact,
            'Actual scheduler proof must show this running L40S job')
    require(compact.count('GPUREQUIREMENTDETAILS:') == 1,
            'Require one resolved GPU resource section')
    section = compact.split('GPUREQUIREMENTDETAILS:', 1)[1]
    require(section.count('Combined:') == section.count('Effective:') == 1,
            'Require exactly one combined and effective GPU resource declaration')
    match = re.fullmatch(r'Combined:(.*?)Effective:(.+)', section)
    require(match is not None, 'Resolved combined GPU resources are missing')
    declarations = {}
    for label, raw_values in zip(('combined', 'effective'), match.groups()):
        values = {}
        for item in raw_values.rstrip(';').split(':'):
            require('=' in item, 'Malformed resolved GPU resource')
            name, value = item.split('=', 1)
            require(name not in values and name and value, 'Duplicate/empty resolved GPU resource')
            values[name] = value
        require(all(values.get(key) == value for key, value in
                    {'num': '1', 'mode': 'shared', 'j_exclusive': 'yes', 'mps': 'no'}.items()),
                'Require one whole-job-exclusive GPU with shared compute and no MPS in ' + label)
        declarations[label] = values
    return declarations


def nvidia_gpu_contract(raw):
    rows = list(csv.reader(io.StringIO(raw)))
    require(len(rows) == 1 and len(rows[0]) == 4, 'Require exactly one selected GPU observation')
    identity, name, memory, compute_mode = (field.strip() for field in rows[0])
    require(re.fullmatch(r'GPU-[a-fA-F0-9-]+', identity) is not None,
            'Selected physical GPU UUID missing')
    require('L40S' in name and re.fullmatch(r'\d+', memory) is not None
            and int(memory) >= 45000, 'Require the original 48 GB L40S class')
    require(compute_mode == 'Default',
            'NVIDIA compute mode must permit the same job\'s separate CUDA processes')
    return {'uuid': identity, 'name': name, 'memory_total_MiB': int(memory),
            'compute_mode': compute_mode}


def capture_gpu_admission(directory):
    directory = Path(directory)
    record = {'version': VERSION, 'status': 'failed',
              'scope': 'Actual resource/compute-mode admission only; no native or scientific pass',
              'gpu_mode_modified': False, 'requested_gpu_resource': GPU_RESOURCE}
    try:
        job, visible = require_allocation(os.environ)
        record.update(job_id=job, cuda_visible_devices=visible)
        scheduler = subprocess.run(['bjobs', '-l', job], capture_output=True, text=True,
                                   check=True, timeout=30)
        record['scheduler'] = {'command': ['bjobs', '-l', job], 'returncode': scheduler.returncode,
                               'stdout': scheduler.stdout, 'stderr': scheduler.stderr}
        record['resolved_gpu_resources'] = scheduler_gpu_contract(scheduler.stdout, job)
        command = ['nvidia-smi', '--id=' + visible,
                   '--query-gpu=uuid,name,memory.total,compute_mode', '--format=csv,noheader,nounits']
        observation = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        record['nvidia_smi'] = {'command': command, 'returncode': observation.returncode,
                                'stdout': observation.stdout, 'stderr': observation.stderr}
        record['gpu'] = nvidia_gpu_contract(observation.stdout)
        record['status'] = 'passed_resource_admission'
        return record
    except BaseException as error:
        record['error'] = repr(error)
        if isinstance(error, subprocess.CalledProcessError):
            record['failed_command'] = {'args': error.cmd, 'returncode': error.returncode,
                                        'stdout': error.stdout, 'stderr': error.stderr}
        raise
    finally:
        record['observed_utc'] = datetime.now(timezone.utc).isoformat()
        _write(directory / 'gpu-process-admission.json', record)


def _read(process, timeout):
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    require(bool(ready), 'Second CUDA-process canary timed out')
    line = process.stdout.readline()
    require(bool(line), 'Second CUDA-process canary exited before its response')
    return json.loads(line)


def validate_torch_canary(row, nonce, event):
    require(row.get('version') == VERSION and row.get('nonce') == nonce
            and row.get('event') == event, 'Stale or malformed Torch CUDA canary response')
    require(row.get('status') == 'passed' and type(row.get('cuda_device_count')) is int
            and row['cuda_device_count'] == 1
            and 'L40S' in row.get('device_name', '') and row.get('cuda_forward_backward') is True,
            'Second process must prove fresh CUDA forward/backward on the allocated L40S')
    require(type(row.get('pid')) is int and row['pid'] > 0
            and row.get('native_model_loaded') is False and type(row.get('optimizer_updates')) is int
            and row['optimizer_updates'] == 0,
            'CUDA canary provenance or bounded scope differs')
    return row


def coexistence_canary(client, directory):
    """Both real CUDA contexts remain live while fresh arithmetic interleaves."""
    require_allocation(os.environ)
    directory = Path(directory)
    nonce = uuid.uuid4().hex
    record = {'version': VERSION, 'status': 'failed', 'nonce': nonce,
              'native_model_loaded': False, 'optimizer_updates': 0,
              'scope': 'Two-process CUDA arithmetic only; full native/chemistry checks still required'}
    process = None
    with (directory / 'torch-canary-stderr.log').open('xb') as error_stream:
        try:
            command = [TORCH_PYTHON, '-u', str(Path(__file__).resolve()), '--torch-canary', nonce]
            record['command'] = command
            environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=error_stream, text=True, bufsize=1, env=environment)
            before = validate_torch_canary(_read(process, 90), nonce, 'ready')
            require(before['pid'] == process.pid and before['pid'] != client.ready['pid'],
                    'Canary and chemistry must be distinct authenticated processes')
            record['torch_before'] = before
            require(process.poll() is None and client.process.poll() is None,
                    'Both CUDA processes must remain alive')
            import numpy as np
            bonds = np.full((1, 25, 25, 5), np.log(np.float32(.2)), dtype=np.float32)
            active = np.ones((1, 25), dtype=np.bool_)
            caps = np.full((1, 25), 4, dtype=np.int32)
            primal = client.evaluate('primal', bonds, active, caps, (0, 1, 2, 3, 4))
            value, gradient = client.evaluate('value_grad', bonds, active, caps, (0, 1, 2, 3, 4))
            require(np.isfinite(primal).all() and np.isfinite(value).all() and np.isfinite(gradient).all(),
                    'Interleaved GPU chemistry canary must remain finite')
            record['chemistry'] = {'pid': client.ready['pid'], 'device': client.ready['device'],
                                    'fresh_primal_and_VJP': True,
                                    'primal_sha256': hashlib.sha256(primal.tobytes()).hexdigest(),
                                    'value_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                                    'gradient_sha256': hashlib.sha256(gradient.tobytes()).hexdigest()}
            process.stdin.write(json.dumps({'nonce': nonce, 'op': 'repeat'}) + '\n')
            process.stdin.flush()
            after = validate_torch_canary(_read(process, 90), nonce, 'repeat')
            require(after['pid'] == process.pid and process.poll() is None,
                    'Same second CUDA context must survive the chemistry calls')
            record['torch_after'] = after
            process.stdin.write(json.dumps({'nonce': nonce, 'op': 'shutdown'}) + '\n')
            process.stdin.flush()
            require(process.wait(timeout=15) == 0, 'Second CUDA process did not close successfully')
            record.update(status='passed_two_CUDA_process_canary', torch_returncode=process.returncode)
            return record
        except BaseException as error:
            record['error'] = repr(error)
            raise
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            if process is not None:
                for pipe in (process.stdin, process.stdout):
                    if pipe is not None:
                        pipe.close()
            record['completed_utc'] = datetime.now(timezone.utc).isoformat()
            _write(directory / 'gpu-process-coexistence.json', record)


def torch_canary(nonce):
    require_allocation(os.environ)
    require(re.fullmatch('[a-f0-9]{32}', nonce) is not None, 'Canary request identity missing')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Real allocated CUDA device required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device_name = torch.cuda.get_device_name(0)
    require('L40S' in device_name, 'Original GPU class required')

    def report(event):
        value = torch.arange(64, dtype=torch.float32, device='cuda', requires_grad=True)
        total = (value * value).sum()
        total.backward()
        torch.cuda.synchronize()
        require(value.is_cuda and total.is_cuda and value.grad.is_cuda
                and total.item() == 85344.0 and torch.equal(value.grad, 2 * value.detach()),
                'Fresh Torch CUDA forward/backward canary failed')
        print(json.dumps({'version': VERSION, 'nonce': nonce, 'event': event, 'status': 'passed',
                          'pid': os.getpid(), 'cuda_device_count': 1, 'device_name': device_name,
                          'cuda_forward_backward': True, 'native_model_loaded': False,
                          'optimizer_updates': 0}), flush=True)

    report('ready')
    for line in sys.stdin:
        request = json.loads(line)
        require(request.get('nonce') == nonce, 'Canary request identity changed')
        if request.get('op') == 'repeat':
            report('repeat')
        else:
            require(request.get('op') == 'shutdown', 'Unknown canary operation')
            return
    raise RuntimeError('CUDA canary input closed before an explicit shutdown')


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] == '--torch-canary', 'Undeclared canary entrypoint')
    torch_canary(sys.argv[2])
