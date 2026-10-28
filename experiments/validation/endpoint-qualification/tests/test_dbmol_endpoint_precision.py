"""Allocated original scalar transport and independent precision-policy controls."""
import importlib.util
from copy import deepcopy
import os
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('dbmol_policy_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/dbmol_reference.py')
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)
COORDINATOR_SPEC = importlib.util.spec_from_file_location('dbmol_finer_policy_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/dbmol_endpoint_precision.py')
c = importlib.util.module_from_spec(COORDINATOR_SPEC); COORDINATOR_SPEC.loader.exec_module(c)


@pytest.fixture(autouse=True)
def allocated(): assert str(os.environ.get('LSB_JOBID', '')).isdigit()


def callbacks(factor=1.):
    import numpy as np
    values = {f: np.tile(np.asarray([.2, .3, .5], dtype=np.float32), (4, 1)) for f in ('X', 'E_upper')}
    weights = {f: np.arange(1, 13, dtype=np.float64).reshape(4, 3) for f in values}
    def primal(v): return np.float64(sum(np.sum(v[f].astype(np.float64) * weights[f]) for f in v))
    def production(v): return primal(v), {f: (weights[f] * factor).astype(np.float32) for f in v}
    def reference(v): return primal(v), {f: weights[f].copy() for f in v}
    return values, primal, production, reference


def evaluate(factor=1., **changes):
    p = r.original_metadata(); values, primal, production, reference = callbacks(factor)
    precision = {'full_model_float64': True, 'parameters_float64': True,
                 'native_features_float64': True, 'native_operations_float64': True,
                 'final_reduction_only': False}
    precision.update(changes)
    return p.evaluate(values, primal, production, primal, reference, lambda: {'model': 'frozen'},
        original_directions={'X': [{'row': 1, 'source': 0, 'target': 2}], 'E_upper': []},
        reference_precision=precision, original_observations={'failed': 'preserved'})


def test_actual_dbmol_fp64_scalar_and_fp32_gradient_are_preserved():
    result = evaluate()
    assert result['status'] == 'passed'
    assert result['baselines']['production']['primals'][0]['dtype'] == 'float64'
    assert result['baseline_fp32']['X']['dtype'] == 'float32'
    assert result['can_certify_original_study'] is False
    assert result['optimization_updates'] == 0


@pytest.mark.parametrize('factor', [0., -1., 1.5])
def test_independent_reference_rejects_zero_reversed_and_scaled_production_gradients(factor):
    assert evaluate(factor)['can_admit_new_child'] is False


@pytest.mark.parametrize('key,value', [('full_model_float64', False), ('parameters_float64', False),
    ('native_features_float64', False), ('native_operations_float64', False), ('final_reduction_only', True)])
def test_final_reduction_cast_or_partial_precision_never_qualifies(key, value):
    assert evaluate(**{key: value})['can_admit_new_child'] is False


def test_independent_reference_keeps_original_scales_and_tolerance():
    result = evaluate()
    assert result['policy']['original_epsilons'] == [.01, .003, .001]
    assert result['policy']['relative_tolerance'] == result['policy']['locality_tolerance'] == .15
    assert all(len(d['selected_measurements']) == 2 for d in result['directions'])


def finer_plan():
    return {'precision_policy': c.finer_policy_spec()}


def finer_evaluate(policy=None, *, factor=1., seal=None, curved=False, mutate=None):
    import numpy as np
    policy = c.FinerPolicy(finer_plan(), r.original_metadata()) if policy is None else policy
    values, primal0, production0, reference0 = callbacks(factor)
    baseline = {f: v.astype(np.float64) for f, v in values.items()}
    state = {'sealed': False}
    def primal(v):
        if any(not np.array_equal(v[f], values[f]) for f in v):
            assert state['sealed'], 'A perturbed native point was evaluated before sealing the fixed pair'
        value = primal0(v)
        if curved:
            value += 1000. * sum(np.sum((v[f].astype(np.float64) - baseline[f]) ** 2) for f in v)
        return np.float64(value)
    def production(v):
        scalar, gradients = production0(v)
        return primal(v), gradients
    def reference(v):
        scalar, gradients = reference0(v)
        if curved:
            gradients = {f: g + 2000. * (v[f].astype(np.float64) - baseline[f]) for f, g in gradients.items()}
        return primal(v), gradients
    def sealed(plan):
        state['sealed'] = True
        if seal is not None: seal(deepcopy(plan))
        if mutate is not None: mutate()
    return policy.evaluate(values, primal, production, primal, reference, lambda: {'model': 'frozen'},
        original_directions={'X': [{'row': 1, 'source': 0, 'target': 2}], 'E_upper': []},
        reference_precision={'full_model_float64': True, 'parameters_float64': True,
            'native_features_float64': True, 'native_operations_float64': True, 'final_reduction_only': False},
        original_observations={'failed': 'preserved'}, seal_callback=sealed)


def test_predeclared_finer_policy_spec_is_separate_and_keeps_all_guards():
    spec = c.finer_policy_spec()
    assert spec['epsilons'] == [.003, .001, .0003, .0001, .00003, .00001, .000003, .000001]
    assert spec['original_epsilons'] == [.01, .003, .001]
    assert spec['relative_tolerance'] == spec['locality_tolerance'] == .15
    assert spec['required_adjacent_passes'] == 2 and not spec['after_result_fallback']
    assert spec['changed_engine_fields'] == ['VERSION', 'EPSILONS']
    c.validate_finer_policy(finer_plan())


@pytest.mark.parametrize('key,value', [
    ('epsilons', [.00003, .00001]), ('epsilons', [.003, .001, .0003, .0001, .00003]),
    ('relative_tolerance', .2), ('locality_tolerance', .2), ('required_adjacent_passes', 1),
    ('after_result_fallback', True), ('original_failed_outcomes_preserved', False),
    ('version', 'protenix-separate-precision-admission-v1'), ('extra_override', True)])
def test_plan_cannot_override_the_fixed_ladder_guards_or_closed_schema(key, value):
    plan = finer_plan(); plan['precision_policy'][key] = value
    with pytest.raises(ValueError, match='Closed predeclared'):
        c.validate_finer_policy(plan)


def test_private_policy_clone_changes_only_two_authenticated_assignments():
    original = r.original_metadata(); policy = c.FinerPolicy(finer_plan(), original)
    assert policy.derivation['AST_reversal_exact']
    assert policy.derivation['changed_assignments'] == ['VERSION', 'EPSILONS']
    assert not policy.derivation['parent_source_modified']
    assert original.VERSION == c.POLICY_PARENT_VERSION and original.EPSILONS == c.POLICY_PARENT_EPSILONS
    assert policy._module.VERSION == c.FINER_POLICY_VERSION and policy._module.EPSILONS == c.FINER_EPSILONS
    assert policy._module._selected_probes.__globals__ is policy._module.__dict__
    assert original._selected_probes.__globals__ is original.__dict__
    assert policy._module._selected_probes is not original._selected_probes


def test_new_fixed_ladder_is_sealed_before_perturbed_forwards_and_uses_two_adjacent_scales():
    seals = []; result = finer_evaluate(seal=seals.append)
    assert result['status'] == 'passed' and result['version'] == c.FINER_POLICY_VERSION
    assert len(seals) == 1 and not seals[0]['selection_uses_FD'] and not seals[0]['after_result_fallback']
    assert all(d['selected_pair'] == [.000003, .000001] for d in seals[0]['directions'])
    assert all(d['selected_pair'] == [.000003, .000001] for d in result['directions'])
    assert result['policy']['epsilons'] == list(c.FINER_EPSILONS)
    assert result['policy']['original_epsilons'] == [.01, .003, .001]
    assert result['baseline_fp32_unchanged'] and result['frozen_unchanged']
    assert result['original_failed_outcome_preserved'] and not result['can_certify_original_study']
    assert result['optimization_updates'] == 0


def test_finer_prospective_policy_preserves_old_correct_gradient_curvature_rejection():
    original = r.original_metadata()
    old = finer_evaluate(original, curved=True)
    assert old['status'] == 'failed' and not old['can_admit_new_child']
    assert all(d['selected_pair'] == [.0001, .00003] for d in old['directions'])
    assert any(not m['reference_endpoint_locality_passed'] for d in old['directions'] for m in d['selected_measurements'])
    new = finer_evaluate(c.FinerPolicy(finer_plan(), original), curved=True)
    assert new['status'] == 'passed' and new['can_admit_new_child']
    assert new['original_failed_outcome_preserved'] and not new['can_certify_original_study']
    assert original.VERSION == c.POLICY_PARENT_VERSION and original.EPSILONS == c.POLICY_PARENT_EPSILONS
    still_old = finer_evaluate(original, curved=True)
    assert still_old['status'] == 'failed' and not still_old['can_admit_new_child']


@pytest.mark.parametrize('factor', [0., -1., 1.5])
def test_finer_ladder_still_rejects_zero_reversed_scaled_production_gradients(factor):
    assert finer_evaluate(factor=factor)['can_admit_new_child'] is False


@pytest.mark.parametrize('name,value', [
    ('EPSILONS', (.000003, .000001)), ('LOCALITY_TOLERANCE', .5),
    ('VERSION', c.POLICY_PARENT_VERSION)])
def test_mutated_private_policy_is_rejected_before_evaluations(name, value):
    policy = c.FinerPolicy(finer_plan(), r.original_metadata())
    setattr(policy._module, name, value)
    with pytest.raises(ValueError, match='Private finer policy constant mutated'):
        finer_evaluate(policy)


def test_mutated_private_function_is_rejected_before_evaluations():
    policy = c.FinerPolicy(finer_plan(), r.original_metadata())
    policy._module._agreement = lambda *args: {'passed': True}
    with pytest.raises(ValueError, match='Private finer policy function mutated'):
        finer_evaluate(policy)


def test_plan_mutation_after_construction_cannot_change_the_fixed_policy():
    plan = finer_plan(); policy = c.FinerPolicy(plan, r.original_metadata())
    plan['precision_policy']['epsilons'].append(.0000001)
    with pytest.raises(ValueError, match='Closed predeclared'):
        finer_evaluate(policy)


def test_policy_mutation_by_callback_cannot_escape_post_evaluation_guard():
    policy = c.FinerPolicy(finer_plan(), r.original_metadata())
    with pytest.raises(ValueError, match='Private finer policy constant mutated'):
        finer_evaluate(policy, mutate=lambda: setattr(policy._module, 'LOCALITY_TOLERANCE', .5))


def test_original_policy_namespace_mutation_is_rejected_without_changing_the_parent_file():
    original = r.original_metadata(); policy = c.FinerPolicy(finer_plan(), original)
    original.EPSILONS = c.FINER_EPSILONS
    with pytest.raises(ValueError, match='Original policy namespace changed'):
        finer_evaluate(policy)


@pytest.mark.parametrize('corruption', [
    b'\n# extra unsealed bytes\n', b'\nEPSILONS = (0.000001,)\n', b'\nLOCALITY_TOLERANCE = 1.0\n'])
def test_changed_parent_source_never_becomes_an_admitted_private_policy(corruption):
    source = c.POLICY_SOURCE.read_bytes()
    with pytest.raises(ValueError, match='source bytes differ'):
        c._clone_policy_source(source + corruption, str(c.POLICY_SOURCE))
