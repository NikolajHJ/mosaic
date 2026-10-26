"""Allocated-only metadata regressions; fake fingerprints prove no native gate."""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import pytest
SPEC = importlib.util.spec_from_file_location('context_metadata', Path(__file__).resolve().parents[1] / 'scripts/replay_context_qualification.py')
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)

def fingerprint(label, name):
    return {'dtype': 'uint32' if name == 'key_data' else 'float32', 'shape': [2] if name == 'key_data' else [25, 5], 'sha256': hashlib.sha256((label + name).encode()).hexdigest()}

def records():
    result = []
    for control in q.CONTROLS:
        for repeat in (0, 1):
            points = []
            for variant, label in q.POINTS:
                fields = {name: fingerprint(label, name) for name in q.FIELDS}
                points.append({'variant': variant, 'label': label, 'input_fields': fields, 'evaluations': [{'placement_control': placement, 'cache_control': cache, 'repeat': index, 'original_class_graph': True, 'outputs': deepcopy(fields)} for placement, cache, index in q.EVALUATIONS]})
            result.append({'version': 'chemistry-softmax-replay-diagnostic-v2', 'control': control, 'repeat': repeat, 'status': 'diagnostic_control_completed_original_failure_preserved', 'optimizer_updates': 0, 'model_evaluations': 0, 'original_gate_reclassified': False, 'can_certify_or_restart_original_study': False, 'probability_substitution': False, 'integrity_hash_scope': 'exact_manifest_member_readonly_hashlib_only', 'original_class': {'sha256': 'a' * 64}, 'input_before': {'original': True}, 'input_after': {'original': True}, 'points': points, 'runtime_after_control': {'python_hash_seed': '0' if 'hash0' in control else None, 'environment': {'XLA_FLAGS': '--xla_gpu_autotune_level=0'} if 'autotune0' in control else {}, 'jax_config_repr': {'jax_enable_x64': 'False', 'jax_default_matmul_precision': "'highest'"}}})
    return result

def test_all_cohorts_select_predeclared_least_change_only_for_new_pilot():
    value = q.qualify(records())
    assert value['selected_policy'] == 'hash0'
    assert not value['native_gradient_passed']
    assert not value['can_certify_or_restart_original_study']
    assert value['fresh_full_model_validation_required']

def test_missing_or_reordered_original_control_cannot_qualify():
    value = records()
    for changed in (value[1:], value[::-1], value[:10]):
        with pytest.raises(ValueError):
            q.qualify(changed)

def test_failed_or_reclassified_control_is_rejected():
    for field, replacement in (('status', 'failed'), ('original_gate_reclassified', True), ('model_evaluations', 1), ('probability_substitution', True)):
        value = records()
        value[0][field] = replacement
        with pytest.raises(ValueError):
            q.qualify(value)

def test_one_mismatching_fresh_process_excludes_whole_candidate():
    value = records()
    row = next((r for r in value if r['control'] == 'combined_hash0' and r['repeat'] == 1))
    row['points'][0]['evaluations'][0]['outputs']['E_upper']['sha256'] = 'b' * 64
    assert q.qualify(value)['selected_policy'] == 'autotune0'

def test_equal_inputs_at_different_saved_labels_cannot_have_different_outputs():
    value = records()
    for record in value:
        if record['control'].endswith('_hash0'):
            for point in record['points']:
                if point['variant'] == 'combined':
                    for evaluation in point['evaluations']:
                        evaluation['outputs']['E_upper']['sha256'] = 'b' * 64
    assert not q.qualify(value)['candidates']['hash0']['all_eight_points_stable']

def test_no_candidate_is_an_honest_negative_not_a_fallback():
    value = records()
    for policy in q.POLICIES:
        row = next((r for r in value if r['control'] == 'combined_' + policy and r['repeat'] == 1))
        row['points'][0]['evaluations'][0]['outputs']['E_upper']['sha256'] = 'b' * 64
    result = q.qualify(value)
    assert result['selected_policy'] is None
    assert result['status'] == 'no_isolated_candidate'

def test_missing_cache_repeat_or_non_original_graph_is_rejected():
    value = records()
    value[0]['points'][0]['evaluations'].pop()
    with pytest.raises(ValueError):
        q.qualify(value)
    value = records()
    value[0]['points'][0]['evaluations'][0]['original_class_graph'] = False
    with pytest.raises(ValueError):
        q.qualify(value)

def test_raw_logit_transport_dtype_or_key_changes_are_rejected():
    for name, field, replacement in (('atom_logits', 'sha256', 'b' * 64), ('bond_logits', 'dtype', 'float64'), ('key_data', 'shape', [3])):
        value = records()
        value[0]['points'][0]['evaluations'][0]['outputs'][name][field] = replacement
        with pytest.raises(ValueError):
            q.qualify(value)

def test_source_inputs_and_compiler_policy_must_match_actual_receipts():
    value = records()
    value[0]['input_after'] = {'changed': True}
    with pytest.raises(ValueError):
        q.qualify(value)
    value = records()
    value[0]['original_class'] = {'sha256': 'b' * 64}
    with pytest.raises(ValueError):
        q.qualify(value)
    value = records()
    value[-1]['runtime_after_control']['environment']['XLA_FLAGS'] = ''
    with pytest.raises(ValueError):
        q.qualify(value)

def test_only_predeclared_environment_overrides_are_exposed():
    assert q.environment('hash0') == {'PYTHONHASHSEED': '0'}
    assert q.environment('autotune0') == {'XLA_FLAGS': '--xla_gpu_autotune_level=0'}
    with pytest.raises(ValueError):
        q.environment('match_one_old_output')
