"""Allocated original scalar transport and independent precision-policy controls."""
import importlib.util
import os
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('dbmol_policy_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/dbmol_reference.py')
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


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
