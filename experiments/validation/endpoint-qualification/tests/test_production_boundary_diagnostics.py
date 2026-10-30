"""Portable control-flow contracts plus allocated numerical capture controls."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

P = Path(__file__).resolve().parents[1] / 'scripts/production_boundary_diagnostics.py'
if not P.exists():
    P = P.with_name('continuation_failure_diagnostics.py')
S = importlib.util.spec_from_file_location('_production_boundary_diagnostic_tests', P)
d = importlib.util.module_from_spec(S)
S.loader.exec_module(d)


def test_metadata_one_native_boundary_stops_before_following_work():
    tree = d.stopped_tree(ast.parse("report['native_gradient_checks']=check()\nfollowed()"))
    calls = []
    with pytest.raises(RuntimeError, match=d.STOP):
        exec(compile(tree, '<toy>', 'exec'), {'report': {}, 'check': lambda: calls.append('check'),
                                            'followed': lambda: calls.append('unexpected')})
    assert calls == ['check']


@pytest.mark.parametrize('count', [0, 2])
def test_metadata_missing_or_duplicate_boundary_fails_closed(count):
    with pytest.raises(ValueError, match='Exactly one'):
        d.stopped_tree(ast.parse("report['native_gradient_checks']=check()\n" * count))


@pytest.mark.parametrize('name', ['run_chemical_updates', 'run_updates', 'update_parameters'])
def test_metadata_optimizer_calls_are_replaced_without_evaluating_arguments(name):
    tree = d.stopped_tree(ast.parse(name + "(side_effect())\nreport['native_gradient_checks']=check()"))
    with pytest.raises(RuntimeError, match='Optimizer execution is forbidden'):
        exec(compile(tree, '<toy>', 'exec'), {'_diagnostic_forbidden_update': d.forbidden_update})


def test_metadata_unexpected_new_errors_do_not_count_as_observed_original_failures():
    assert d.expected_observation(RuntimeError(d.STOP))
    assert d.expected_observation(ValueError(next(iter(d.EXPECTED))))
    assert not d.expected_observation(RuntimeError(next(iter(d.EXPECTED))))
    assert not d.expected_observation(ValueError('CUDA failed'))


def test_metadata_capture_is_not_a_continuation_dispatch():
    calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(P.read_bytes())) if isinstance(n, ast.Call)}
    assert not calls & {'dispatch', 'submit', 'run_updates', 'run_chemical_updates', 'update_parameters'}
    assert d.CAP == 128 * 1024**2


@pytest.mark.parametrize('change', [False, True])
def test_measurement_capture_retains_actual_callback_counts_and_exact_original_result(tmp_path, change):
    np = d.numpy()
    calls = []
    def measurement(callbacks, values):
        p, ad, f = callbacks
        scalar = p(values)
        value, gradient = ad(values)
        return {'scalar': float(scalar), 'ad_scalar': float(value), 'frozen': f(),
                'gradient': gradient['X'].tolist()}
    precision = SimpleNamespace(_measurement=measurement)
    values = {'X': np.asarray([[1., 0.]], np.float32), 'E_upper': np.asarray([[0., 1.]], np.float32)}
    current = [0]
    def primal(v):
        calls.append('p')
        return np.float32(1 + current[0])
    def autodiff(v):
        calls.append('ad')
        return np.float32(1), {k: np.ones_like(a) for k, a in v.items()}
    callbacks = (primal, autodiff, lambda: {'model': 'unchanged'})
    with d.captured_measurements(precision, tmp_path / 'measurements') as records:
        first = precision._measurement(callbacks, values)
        current[0] = int(change)
        second = precision._measurement(callbacks, values)
        assert records[1]['versus_first']['primal']['exact_equal'] is (not change)
        assert second['scalar'] == 1 + int(change)
        assert first['ad_scalar'] == second['ad_scalar'] == 1
    assert calls == ['p', 'ad', 'p', 'ad']
    assert precision._measurement is measurement
