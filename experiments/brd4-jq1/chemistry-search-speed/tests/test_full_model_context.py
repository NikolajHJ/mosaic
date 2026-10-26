"""Allocated-only compiler-context admission metadata; no native evaluation."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import full_model_ops as op
import full_model_context as context


def qualified(policy='hash0'):
    environment = {}
    if 'hash0' in policy: environment['PYTHONHASHSEED'] = '0'
    if 'autotune0' in policy: environment['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
    return {'qualification': {'selected_policy': policy, 'selected_environment': environment,
                              'native_gradient_passed': False}}


def test_hash_context_does_not_mutate_original_environment_or_scientific_settings():
    original = {'JAX_ENABLE_X64': '0', 'KEY': '17'}
    changed = context.apply_environment(original, qualified())
    assert changed == {**original, 'PYTHONHASHSEED': '0'}
    assert original == {'JAX_ENABLE_X64': '0', 'KEY': '17'}


def test_autotune_context_preserves_unrelated_flags():
    result = context.apply_environment({'XLA_FLAGS': '--other_flag=true'}, qualified('autotune0'))
    assert result['XLA_FLAGS'] == '--other_flag=true --xla_gpu_autotune_level=0'


def test_both_controls_are_exact_declared_values_only():
    assert context.apply_environment({}, qualified('hash0_autotune0')) == {
        'PYTHONHASHSEED': '0', 'XLA_FLAGS': '--xla_gpu_autotune_level=0'}


def test_missing_or_undeclared_context_never_becomes_native_admission():
    for policy in (None, 'match_one_old_probability', 'float64', 'lower_learning_rate'):
        with pytest.raises(ValueError): context.apply_environment({}, {'qualification': {'selected_policy': policy}})


def test_extra_loss_or_optimizer_environment_is_forbidden():
    value = qualified(); value['qualification']['selected_environment']['LEARNING_RATE'] = '0.03'
    with pytest.raises(ValueError): context.apply_environment({}, value)
    value = qualified(); value['qualification']['native_gradient_passed'] = True
    with pytest.raises(ValueError): context.apply_environment({}, value)


def test_ambiguous_compiler_override_rejected_and_identical_override_not_duplicated():
    value = qualified('autotune0')
    with pytest.raises(ValueError): context.apply_environment({'XLA_FLAGS': '--xla_gpu_autotune_level=4'}, value)
    assert context.apply_environment({'XLA_FLAGS': '--xla_gpu_autotune_level=0'}, value)['XLA_FLAGS'] == '--xla_gpu_autotune_level=0'


def test_cpu_context_rehashes_file_and_rechecks_original_producer(tmp_path, monkeypatch):
    payload = tmp_path/'cpu/payload'; payload.mkdir(parents=True)
    value = qualified(); path = payload/'qualified-context.json'; op.write(path, value)
    assessment = {'status': 'passed', 'qualified_context': op._file_binding(path)}
    op.write(payload/'assessment.json', assessment)
    fake = SimpleNamespace(ROOT=tmp_path, require=op.require, _binding=op._binding)
    monkeypatch.setattr(context, 'qualify_predecessor', lambda *args: deepcopy(value))
    assert context.adopt_cpu(fake, {}) == (value, assessment['qualified_context'])
    path.write_text('{}')
    with pytest.raises(ValueError): context.adopt_cpu(fake, {})


def test_cpu_failed_or_missing_context_cannot_be_adopted(tmp_path):
    payload = tmp_path/'cpu/payload'; payload.mkdir(parents=True)
    op.write(payload/'assessment.json', {'status': 'failed'})
    fake = SimpleNamespace(ROOT=tmp_path, require=op.require)
    with pytest.raises(ValueError): context.adopt_cpu(fake, {})


def test_only_actual_replay_source_cpu_and_job_identity_may_qualify():
    fake = SimpleNamespace(require=op.require)
    for key in ('gpu_job_id', 'manifest_sha256', 'cpu_certificate_sha256', 'root'):
        value = deepcopy(context.PRODUCER); value[key] = 'different'
        with pytest.raises(ValueError): context.producer(fake, {'replay_context_producer': value})


def test_actual_replay_ended_dependency_is_exact_and_never_attached_to_native_gpu():
    args = op.argv('cpu', 'a'*64, 'ended(29818513)')
    assert args[args.index('-w')+1] == 'ended(29818513)'
    for phase, dependency in (('cpu', 'ended(999)'), ('cpu', 'done(29818513)'), ('protenix', 'ended(29818513)')):
        with pytest.raises(ValueError): op.argv(phase, 'a'*64, dependency)
