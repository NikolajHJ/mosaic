"""Allocated CPU metadata/fake-process contracts, never GPU/model evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import gpu_process_admission as admission
import gpu_local_bridge as bridge


ENVIRONMENT = {'LSB_JOBID': '12345678', 'LSB_QUEUE': 'gpul40s',
               'LSB_DJOB_NUMPROC': '8', 'CUDA_VISIBLE_DEVICES': 'GPU-ab12-34cd'}
SCHEDULER = '''Job <12345678>, Job Name <fixture>, Status <RUN>, Queue <gpul40s>
GPU REQUIREMENT DETAILS:
 Combined: num=1:mode=shared:mps=no:j_exclusive=yes:aff=no:gvendor=nvidia
 Effective: num=1:mode=shared:mps=no:j_exclusive=yes:aff=no:gvendor=nvidia
'''
GPU = 'GPU-ab12-34cd, NVIDIA L40S, 46068, Default\n'


def test_declared_job_exclusivity_is_separate_from_default_compute():
    values = admission.scheduler_gpu_contract(SCHEDULER, '12345678')
    assert set(values) == {'combined', 'effective'}
    assert all(v['mode'] == 'shared' and v['j_exclusive'] == 'yes' and v['mps'] == 'no'
               for v in values.values())
    assert admission.GPU_RESOURCE == 'num=1:mode=shared:j_exclusive=yes:mps=no'
    assert admission.nvidia_gpu_contract(GPU)['compute_mode'] == 'Default'
    assert admission.require_allocation(ENVIRONMENT) == ('12345678', 'GPU-ab12-34cd')


@pytest.mark.parametrize('fault', [
    'other_job', 'duplicate_job', 'pending', 'wrong_queue', 'missing_section',
    'duplicate_section', 'no_effective', 'multiple_gpu', 'exclusive_compute',
    'not_job_exclusive', 'mps', 'duplicate_mode', 'malformed',
])
def test_wrong_or_ambiguous_actual_scheduler_contract_blocks(fault):
    raw = SCHEDULER
    if fault == 'other_job': raw = raw.replace('12345678', '12345679')
    elif fault == 'duplicate_job': raw += '\nJob <12345678>'
    elif fault == 'pending': raw = raw.replace('Status <RUN>', 'Status <PEND>')
    elif fault == 'wrong_queue': raw = raw.replace('Queue <gpul40s>', 'Queue <gpua100>')
    elif fault == 'missing_section': raw = raw.replace('GPU REQUIREMENT DETAILS:', 'other')
    elif fault == 'duplicate_section': raw += '\nGPU REQUIREMENT DETAILS:'
    elif fault == 'no_effective': raw = raw.replace('Effective:', 'other:')
    elif fault == 'multiple_gpu': raw = raw.replace('num=1', 'num=2')
    elif fault == 'exclusive_compute': raw = raw.replace('mode=shared', 'mode=exclusive_process')
    elif fault == 'not_job_exclusive': raw = raw.replace('j_exclusive=yes', 'j_exclusive=no')
    elif fault == 'mps': raw = raw.replace('mps=no', 'mps=yes')
    elif fault == 'duplicate_mode': raw = raw.replace('mode=shared', 'mode=shared:mode=shared')
    elif fault == 'malformed': raw = raw.replace('mps=no', 'mps')
    with pytest.raises(ValueError):
        admission.scheduler_gpu_contract(raw, '12345678')


def test_wrapped_resolved_gpu_fields_are_authenticated():
    raw = SCHEDULER.replace('mode=shared', 'mode=sha\n                         red')
    assert admission.scheduler_gpu_contract(raw, '12345678')['effective']['mode'] == 'shared'


@pytest.mark.parametrize('fault', ['effective_not_exclusive', 'duplicate_effective'])
def test_started_effective_contract_cannot_override_or_duplicate_valid_combined(fault):
    raw = SCHEDULER
    if fault == 'effective_not_exclusive':
        raw = raw.replace('Effective: num=1:mode=shared:mps=no:j_exclusive=yes',
                          'Effective: num=1:mode=shared:mps=no:j_exclusive=no')
    else:
        raw += 'Effective: num=1:mode=shared:mps=no:j_exclusive=yes\n'
    with pytest.raises(ValueError):
        admission.scheduler_gpu_contract(raw, '12345678')


@pytest.mark.parametrize('replacement', [
    ('GPU-ab12-34cd', '0'), ('NVIDIA L40S', 'NVIDIA A100'), ('46068', '24000'),
    ('46068', 'N/A'), ('Default', 'Exclusive_Process'), ('Default', 'Prohibited'),
])
def test_wrong_gpu_class_memory_identity_or_compute_mode_blocks(replacement):
    with pytest.raises(ValueError):
        admission.nvidia_gpu_contract(GPU.replace(*replacement))


@pytest.mark.parametrize('raw', ['', GPU + GPU, 'GPU-ab, L40S, 46068\n'])
def test_nvidia_observation_must_be_exactly_one_complete_device(raw):
    with pytest.raises(ValueError):
        admission.nvidia_gpu_contract(raw)


@pytest.mark.parametrize('key,value', [
    ('LSB_JOBID', ''), ('LSB_JOBID', 'abc'), ('LSB_QUEUE', 'hpc'),
    ('LSB_DJOB_NUMPROC', '4'), ('CUDA_VISIBLE_DEVICES', ''),
    ('CUDA_VISIBLE_DEVICES', '0,1'), ('CUDA_VISIBLE_DEVICES', 'MIG-GPU-ab'),
])
def test_allocation_or_device_scope_mismatch_blocks(key, value):
    environment = {**ENVIRONMENT, key: value}
    with pytest.raises(ValueError):
        admission.require_allocation(environment)


def test_resource_capture_preserves_actual_commands_stdout_and_exact_device(monkeypatch, tmp_path):
    calls = []
    for name, value in ENVIRONMENT.items(): monkeypatch.setenv(name, value)
    def run(command, **options):
        calls.append(command)
        assert options == {'capture_output': True, 'text': True, 'check': True, 'timeout': 30}
        return SimpleNamespace(returncode=0, stdout=SCHEDULER if command[0] == 'bjobs' else GPU, stderr='')
    monkeypatch.setattr(admission.subprocess, 'run', run)
    record = admission.capture_gpu_admission(tmp_path)
    assert record['status'] == 'passed_resource_admission' and record['gpu_mode_modified'] is False
    assert calls[0] == ['bjobs', '-l', '12345678']
    assert calls[1] == ['nvidia-smi', '--id=GPU-ab12-34cd',
                        '--query-gpu=uuid,name,memory.total,compute_mode', '--format=csv,noheader,nounits']
    assert all('-c' not in command and '--compute-mode' not in command for command in calls)
    assert json.loads((tmp_path / 'gpu-process-admission.json').read_text()) == record


def test_resource_query_failure_retains_failed_receipt_and_original_error(monkeypatch, tmp_path):
    for name, value in ENVIRONMENT.items(): monkeypatch.setenv(name, value)
    error = subprocess.CalledProcessError(1, ['bjobs', '-l', '12345678'])
    def run(*args, **kwargs): raise error
    monkeypatch.setattr(admission.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        admission.capture_gpu_admission(tmp_path)
    assert caught.value is error
    receipt = json.loads((tmp_path / 'gpu-process-admission.json').read_text())
    assert receipt['status'] == 'failed' and 'CalledProcessError' in receipt['error']


def canary_row():
    return {'version': admission.VERSION, 'nonce': 'a' * 32, 'event': 'ready',
            'status': 'passed', 'cuda_device_count': 1, 'device_name': 'NVIDIA L40S',
            'cuda_forward_backward': True, 'pid': 654321,
            'native_model_loaded': False, 'optimizer_updates': 0}


def test_canary_metadata_is_explicitly_engineering_not_native_readiness():
    row = canary_row()
    assert admission.validate_torch_canary(row, 'a' * 32, 'ready') is row
    assert row['native_model_loaded'] is False and row['optimizer_updates'] == 0


@pytest.mark.parametrize('key,value', [
    ('version', 'other'), ('nonce', 'b' * 32), ('event', 'repeat'), ('status', 'failed'),
    ('cuda_device_count', 2), ('device_name', 'CPU'), ('cuda_forward_backward', False),
    ('pid', 0), ('pid', True), ('native_model_loaded', True), ('optimizer_updates', 1),
])
def test_invalid_or_out_of_scope_canary_response_blocks(key, value):
    with pytest.raises(ValueError):
        admission.validate_torch_canary({**canary_row(), key: value}, 'a' * 32, 'ready')


def test_bridge_rejects_resource_contract_before_starting_an_extra_cuda_process(monkeypatch, tmp_path):
    calls = []
    error = ValueError('fixture wrong compute exclusivity')
    def capture(path): raise error
    monkeypatch.setattr(bridge, 'capture_gpu_admission', capture)
    monkeypatch.setattr(bridge.subprocess, 'Popen', lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError) as caught:
        bridge.LocalGPUClient(tmp_path / 'wrong-resource')
    assert caught.value is error and not calls


def test_bridge_canary_failure_closes_its_own_gpu_process_and_preserves_error(monkeypatch, tmp_path):
    error = RuntimeError('fixture second CUDA process rejected')
    class FakeProcess:
        returncode = None
        def poll(self): return self.returncode
        def kill(self): self.returncode = -9
        def wait(self, timeout): return self.returncode
    process = FakeProcess()
    monkeypatch.setattr(bridge, 'capture_gpu_admission', lambda path: {'fixture': True})
    monkeypatch.setattr(bridge.subprocess, 'Popen', lambda *args, **kwargs: process)
    monkeypatch.setattr(bridge.LocalGPUClient, '_read', lambda self: {
        'version': bridge.VERSION, 'event': 'ready', 'pid': 54321,
        'device': {'platform': 'gpu', 'kind': 'NVIDIA L40S'}, 'model_or_search_loaded': False})
    def failed(*args): raise error
    monkeypatch.setattr(bridge, 'coexistence_canary', failed)
    with pytest.raises(RuntimeError) as caught:
        bridge.LocalGPUClient(tmp_path / 'failed-canary')
    assert caught.value is error and process.returncode == -9
