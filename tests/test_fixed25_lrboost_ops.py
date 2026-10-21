"""Allocated high-rate launcher fixtures; synthetic metadata certifies no model."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import fixed25_lrboost_ops as ops


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('High-rate launcher fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    return path


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    root, source = tmp_path / 'run', tmp_path / 'release/source'
    root.mkdir()
    monkeypatch.setattr(ops, 'ROOT', root)
    monkeypatch.setattr(ops, 'SOURCE', source)
    monkeypatch.setattr(ops.storage, 'ROOT', root)
    monkeypatch.setattr(ops.storage, 'HOME_ROOT', tmp_path / 'home-new')
    configs = Path(__file__).resolve().parents[1] / 'examples/configs'
    for arm in ops.ARMS:
        name = f'brd4_jq1_25_fixed_lrboost_{arm}_10000.json'
        put(source / 'examples/configs' / name, json.loads((configs / name).read_text(encoding='utf-8')))
    return root, source


@pytest.mark.parametrize('arm', ops.ARMS)
@pytest.mark.parametrize('phase,walltime', [('validate', '02:00'), ('run', '23:00')])
def test_all_three_new_native_and_long_routes_have_exact_resources_and_no_interarm_dependency(isolated, arm, phase, walltime):
    command = ops.submission_command(arm, phase, 'a' * 64, 'b' * 64)
    assert command[:5] == ['bsub', '-q', 'gpul40s', '-n', '4']
    assert command[command.index('-W') + 1] == walltime
    assert command[command.index('-gpu') + 1] == 'num=1:mode=exclusive_process'
    assert 'rusage[mem=8GB]' in command and '-w' not in command
    assert command[-8:] == [str(ops.SOURCE), str(ops.ROOT), 'a' * 64, phase,
                             arm, str(ops.native_parent(arm, phase)), '0', 'b' * 64]


@pytest.mark.parametrize('arm', ops.ARMS)
def test_each_readiness_callback_depends_only_on_its_own_actual_native(isolated, arm):
    command = ops.submission_command(arm, 'after-ready', 'a' * 64, 'b' * 64, producer='123456')
    assert command[command.index('-q') + 1] == 'hpc'
    assert command[command.index('-n') + 1] == '1'
    assert command[command.index('-w') + 1] == 'ended(123456)'
    assert '-gpu' not in command


def test_exact_new_config_version_rates_and_all_fresh_roots(isolated):
    ops.validate_configs()
    assert ops.RATES == {'protenix': 3.0, 'boltz2': .05, 'boltz2_chemistry': .05}
    assert ops.storage.output_parent('protenix') == ops.ROOT / 'run-protenix'
    assert ops.storage.output_parent('boltz2') == ops.ROOT / 'run-boltz2'
    assert ops.storage.output_parent('boltz2_chemistry') == ops.storage.HOME_ROOT / 'run-boltz2_chemistry'
    assert '29777709' not in str(ops.storage.output_parent('protenix'))
    value = ops.read(ops.config_path('protenix'))
    value['initial_step_size'] = .3
    put(ops.config_path('protenix'), value)
    with pytest.raises(ValueError):
        ops.validate_configs()


@pytest.mark.parametrize('damage', ['unknown_reply', 'timeout'])
def test_unknown_submission_outcomes_are_persisted_and_never_blindly_duplicated(isolated, monkeypatch, damage):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if damage == 'timeout':
            raise subprocess.TimeoutExpired(command, 60)
        return SimpleNamespace(returncode=1, stdout='Unrecognized scheduler reply', stderr='diagnostic')
    monkeypatch.setattr(ops.subprocess, 'run', fake)
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        ops.submit('protenix', 'validate', 'a' * 64, 'b' * 64)
    parent = ops.native_parent('protenix', 'validate')
    assert ops.read(parent / 'submission.json')['outcome'] == 'unknown'
    assert (parent / 'submission-started.json').exists()
    with pytest.raises(ValueError, match='no duplicate'):
        ops.submit('protenix', 'validate', 'a' * 64, 'b' * 64)
    assert len(calls) == 1


def test_actual_bsub_id_and_exact_argv_are_saved_before_scheduler_read(isolated, monkeypatch):
    def fake(command, **kwargs):
        if command[0] == 'bsub':
            return SimpleNamespace(returncode=0, stdout='Job <123456> is submitted to queue <gpul40s>.', stderr='')
        parent = ops.native_parent('boltz2', 'validate')
        assert ops.read(parent / 'submission.json')['job_id'] == '123456'
        return SimpleNamespace(returncode=0, stdout='Job <123456>, Status <PEND>, Queue <gpul40s>', stderr='')
    monkeypatch.setattr(ops.subprocess, 'run', fake)
    record = ops.submit('boltz2', 'validate', 'a' * 64, 'b' * 64)
    assert record['outcome'] == 'submitted' and record['job_id'] == '123456'
    assert record['args'] == ops.submission_command('boltz2', 'validate', 'a' * 64, 'b' * 64)
    assert record['learning_rate'] == .05


@pytest.mark.parametrize('arm', ops.ARMS)
def test_driver_commands_use_only_new_readiness_and_exact_new_config_paths(isolated, arm):
    root, _ = isolated
    ready = put(root / f'validate-{arm}/payload/study/assessment.json', {'synthetic': 'new native fixture'})
    validate = ops.driver_command(arm, 'validate', 'a' * 64, 'b' * 64)
    production = ops.driver_command(arm, 'run', 'a' * 64, 'b' * 64)
    assert validate[validate.index('--config') + 1] == str(ops.config_path(arm))
    assert '--ready' not in validate
    assert production[production.index('--ready') + 1] == str(ready)
    assert production[production.index('--ready-sha256') + 1] == ops.sha(ready)
    assert production[production.index('--output') + 1] == str(ops.storage.output_parent(arm) / 'payload/study')
    assert ('--prepared' in production) is (arm != 'protenix')


@pytest.mark.parametrize('damage', ['wrong_job', 'running', 'wrong_queue', 'missing_lsf'])
def test_saved_native_terminal_proof_requires_actual_same_job_done_and_lsf_success(tmp_path, damage):
    job, status, queue = ('999', 'DONE', 'gpul40s') if damage == 'wrong_job' else ('123', 'DONE', 'gpul40s')
    if damage == 'running': status = 'RUN'
    if damage == 'wrong_queue': queue = 'hpc'
    scheduler = tmp_path / 'scheduler-final.txt'
    scheduler.write_text(f'Job <{job}>, Status <{status}>, Queue <{queue}>', encoding='utf-8')
    stdout = tmp_path / 'stdout.log'
    stdout.write_text('Subject: Job 123: Done\n' + ('' if damage == 'missing_lsf' else 'Successfully completed.\n'), encoding='utf-8')
    with pytest.raises(ValueError, match='same-job'):
        ops.authenticate_native_terminal('123', [ops.hashed(scheduler), ops.hashed(stdout)])


def test_projection_is_derived_from_the_new_native_pilot_instead_of_old_rate_passes():
    value = {'storage_projection': {'maximum_pilot_update_bytes': 1234,
                                   'projected_bytes': 2 * 1234 * 10000 + 1024**3}}
    assert ops.validate_projection(value) == 2 * 1234 * 10000 + 1024**3
    value['storage_projection']['projected_bytes'] -= 1
    with pytest.raises(ValueError, match='projection'):
        ops.validate_projection(value)


def test_missing_readiness_decision_does_not_dispatch_a_subset(isolated):
    root, _ = isolated
    put(root / 'decision-protenix.json', {'status': 'passed'})
    assert ops.dispatch('a' * 64) is None
    assert not (root / 'long-dispatch-started.json').exists()


def test_whole_fresh_projection_gate_precedes_exactly_three_independent_long_submissions(isolated, monkeypatch):
    root, _ = isolated
    put(root / 'cpu-certificate.json', {'synthetic': 'unit metadata only'})
    for arm in ops.ARMS:
        put(root / ('decision-' + arm + '.json'), {'status': 'passed', 'unit_arm': arm})
    # Explicit unit boundaries: this fixture tests dispatch ordering, not model
    # readiness, scheduler authentication or storage capacity.
    monkeypatch.setattr(ops, 'verify_cpu', lambda manifest: None)
    monkeypatch.setattr(ops, 'authenticate_ready', lambda a, m, d: {
        'storage_projection': {'maximum_pilot_update_bytes': 100 + ops.ARMS.index(a),
                              'projected_bytes': 2 * (100 + ops.ARMS.index(a)) * 10000 + 1024**3}})
    monkeypatch.setattr(ops, 'verify_stopped_predecessors', lambda: {'path': 'unit-stop-proof', 'sha256': 'a' * 64})
    order = []
    def gate(projections, receipt, *, phase):
        order.append('whole-gate')
        assert set(projections) == set(ops.ARMS) and phase == 'dispatch'
        return {'status': 'passed', 'synthetic_unit_boundary': True}
    def submit(arm, phase, manifest, cpu_sha):
        assert order and order[0] == 'whole-gate'
        assert (root / 'long-dispatch-started.json').exists()
        order.append(arm)
        return {'job_id': str(100 + len(order)), 'arm': arm, 'phase': phase}
    monkeypatch.setattr(ops.storage, 'check_campaign_storage', gate)
    monkeypatch.setattr(ops, 'submit', submit)
    jobs = ops.dispatch('a' * 64)
    assert order == ['whole-gate', *ops.ARMS] and len(jobs) == 3
    assert ops.read(root / 'long-submissions.json')['all_three_submitted']
    with pytest.raises(ValueError, match='no duplicate'):
        ops.dispatch('a' * 64)


def test_predecessor_stop_guard_authenticates_both_exact_exit_owner_records(tmp_path, monkeypatch):
    root = tmp_path / 'stopped'
    root.mkdir()
    monkeypatch.setattr(ops, 'STOP_ROOT', root)
    record = {'version': 'jq1-lrboost-predecessor-stop-v1', 'status': 'passed',
              'user_requested': True, 'preserved_completed_protenix': '29777709', 'jobs': {}, 'proofs': []}
    for arm, job in (('boltz2', '29778203'), ('boltz2_chemistry', '29778204')):
        scheduler = root / (arm + '-scheduler.txt')
        stdout = root / (arm + '-stdout.log')
        scheduler.write_text(f'Job <{job}>, Status <EXIT>, Queue <gpul40s>\nTERM_OWNER', encoding='utf-8')
        stdout.write_text(f'Subject: Job {job}: Exited\nExited with exit code 130. TERM_OWNER', encoding='utf-8')
        a, b = ops.hashed(scheduler), ops.hashed(stdout)
        record['proofs'] += [a, b]
        record['jobs'][arm] = {'job_id': job, 'status': 'EXIT', 'scheduler': a, 'stdout': b}
    path = put(root / 'assessment.json', record)
    monkeypatch.setattr(ops, 'STOP_ASSESSMENT_SHA256', ops.sha(path))
    assert ops.verify_stopped_predecessors()['sha256'] == ops.sha(path)
    record['jobs']['boltz2']['status'] = 'RUN'
    put(path, record)
    monkeypatch.setattr(ops, 'STOP_ASSESSMENT_SHA256', ops.sha(path))
    with pytest.raises(ValueError, match='still active'):
        ops.verify_stopped_predecessors()
