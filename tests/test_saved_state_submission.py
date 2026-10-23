"""Engineering checks for observer selection and the existing LSF gates.

Run on DTU HPC. These fixtures construct commands only; they never submit jobs,
load model weights or certify native numerical behavior.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/hpc'))
from submit_failed_gate_attribution import DIAGNOSTICS, validate_submission_plan, workflow


def _rows(diagnostic='failed_gate_attribution'):
    return workflow(Path('/work3/user/releases/observer'), Path('/work3/user'),
        Path('/work3/user/plan.json'), 'a' * 64, 'b' * 64, 'c' * 64,
        diagnostic=diagnostic)


def test_default_workflow_preserves_original_source080_commands():
    cpu, native = _rows()
    assert cpu['job_command'][-7:] == ['python', '-m', 'pytest', '-q',
        'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
        'tests/test_saved_state_submission.py']
    assert native['job_command'][-7:] == ['python', '-m',
        'latentmol.boltz.failed_gate_attribution', '--plan', '/work3/user/plan.json',
        '--sha256', 'a' * 64]
    assert native['dependency'] == 'done:cpu'
    assert native['arguments'][native['arguments'].index('-W') + 1] == '4:00'


def test_causal_observer_is_explicit_and_reuses_source_and_cpu_gates():
    cpu, native = _rows('attention_causal_diagnostic')
    assert 'tests/test_attention_causal_diagnostic.py' in cpu['job_command']
    assert 'latentmol.boltz.attention_causal_diagnostic' in native['job_command']
    assert native['dependency'] == 'done:cpu'
    assert native['cwd'] == '/work3/user/releases/observer'
    assert '-gpu' in native['arguments'] and 'select[gpu80gb]' in native['arguments']
    assert '--archive-sha256' in native['job_command']
    assert '--manifest-sha256' in native['job_command']
    assert 'no production patch' in DIAGNOSTICS['attention_causal_diagnostic']['scope']


def test_unregistered_observer_fails_without_scheduler_access():
    with pytest.raises(ValueError, match='Unregistered'):
        _rows('arbitrary_module')


def test_decoder_observer_has_bounded_native_job_and_recovery_fixtures():
    cpu, native = _rows('decoder_causal_diagnostic')
    assert 'tests/test_decoder_causal_diagnostic.py' in cpu['job_command']
    assert 'tests/test_defog_single_recovery.py' in cpu['job_command']
    assert 'latentmol.boltz.decoder_causal_diagnostic' in native['job_command']
    assert native['arguments'][native['arguments'].index('-W') + 1] == '1:00'
    assert native['dependency'] == 'done:cpu'
    assert 'decoder-only' in DIAGNOSTICS['decoder_causal_diagnostic']['scope']


def test_head_observer_uses_same_authentication_and_bounded_native_allocation():
    cpu, native = _rows('head_causal_diagnostic')
    assert 'tests/test_head_causal_diagnostic.py' in cpu['job_command']
    assert 'latentmol.boltz.head_causal_diagnostic' in native['job_command']
    assert native['arguments'][native['arguments'].index('-W') + 1] == '2:00'
    assert native['dependency'] == 'done:cpu'
    assert 'select[gpu80gb]' in native['arguments']
    assert '--manifest-sha256' in native['job_command']
    assert 'no production patch' in DIAGNOSTICS['head_causal_diagnostic']['scope']


def test_precision_and_native_head_observers_keep_cpu_gate_and_source_identity():
    for name in ('native_head_gradient_diagnostic', 'precision_gradient_diagnostic'):
        cpu, native = _rows(name)
        assert 'tests/test_native_head_gradient_diagnostic.py' in cpu['job_command']
        assert 'latentmol.boltz.' + name in native['job_command']
        assert native['dependency'] == 'done:cpu'
        assert '--manifest-sha256' in native['job_command']
        assert '--archive-sha256' in native['job_command']
        assert native['arguments'][native['arguments'].index('-W') + 1] == '2:00'
    cpu, _ = _rows('precision_gradient_diagnostic')
    assert 'tests/test_dbmol_precision.py' in cpu['job_command']
    assert 'tests/test_precision_gradient_diagnostic.py' in cpu['job_command']
    assert 'no latent updates' in DIAGNOSTICS['precision_gradient_diagnostic']['scope']


def test_affinity_observer_retains_source_gate_and_declares_counterfactual_scope():
    cpu, native = _rows('affinity_causal_diagnostic')
    assert 'tests/test_affinity_causal_diagnostic.py' in cpu['job_command']
    assert 'tests/test_native_head_gradient_diagnostic.py' in cpu['job_command']
    assert 'latentmol.boltz.affinity_causal_diagnostic' in native['job_command']
    assert native['dependency'] == 'done:cpu'
    assert '--archive-sha256' in native['job_command']
    assert '--manifest-sha256' in native['job_command']
    assert native['arguments'][native['arguments'].index('-W') + 1] == '2:00'
    assert 'no production patch' in DIAGNOSTICS['affinity_causal_diagnostic']['scope']


def test_regression_decoder_observer_keeps_cpu_gate_and_original_native_identity():
    cpu, native = _rows('affinity_decoder_causal_diagnostic')
    assert 'tests/test_affinity_decoder_causal_diagnostic.py' in cpu['job_command']
    assert 'tests/test_decoder_causal_diagnostic.py' in cpu['job_command']
    assert 'latentmol.boltz.affinity_decoder_causal_diagnostic' in native['job_command']
    assert native['dependency'] == 'done:cpu'
    assert '--archive-sha256' in native['job_command']
    assert '--manifest-sha256' in native['job_command']
    assert native['arguments'][native['arguments'].index('-W') + 1] == '1:00'
    assert 'select[gpu80gb]' in native['arguments']
    assert 'no production patch' in DIAGNOSTICS['affinity_decoder_causal_diagnostic']['scope']


def test_exact_matched_failure_pair_is_allowed_without_widening_other_observers():
    work, release = Path('/work3/user'), Path('/work3/user/releases/observer')
    plan = {'version': 1, 'output_root': '/work3/user/runs/observer-replay',
            'cases': [{'id': name, 'arm': 'fixed_initial_mask', 'step': 15}
                      for name in ('boltz_only', 'boltz_defog')]}
    validate_submission_plan(plan, work, release, 'failed_gate_attribution')
    with pytest.raises(ValueError, match='exact matched'):
        validate_submission_plan(plan, work, release, 'affinity_causal_diagnostic')
    plan['cases'][1]['step'] = 20
    with pytest.raises(ValueError, match='exact matched'):
        validate_submission_plan(plan, work, release, 'failed_gate_attribution')
