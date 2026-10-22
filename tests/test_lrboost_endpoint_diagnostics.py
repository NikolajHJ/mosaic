"""Allocated diagnostic contract/toy fixtures; never a native endpoint pass."""

import copy
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import lrboost_endpoint_diagnostics as diag


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Endpoint diagnostic fixtures require allocated DTU HPC')


def probabilities():
    values = np.full((25, 6), .0625, np.float32)
    values[:, 0] = .6875
    return values


def test_only_initial_final_and_at_most_one_predeclared_nearby_saved_state():
    assert diag.selected_steps() == [0, 10000]
    assert diag.selected_steps(9900) == [0, 9900, 10000]
    with pytest.raises(ValueError):
        diag.selected_steps(1234)
    with pytest.raises(ValueError):
        diag.selected_steps([9500, 9900])


def test_diagnostic_outputs_are_bound_to_a_separate_analysis_release_not_a_saved_study():
    source = Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01/source')
    assert diag.diagnostic_output_path(source, 'protenix') == Path(
        '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/diagnostic-protenix/payload')
    with pytest.raises(ValueError):
        diag.diagnostic_output_path(Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source'), 'protenix')


@pytest.mark.parametrize('arm', ['protenix', 'boltz2'])
def test_original_native_direction_rule_and_reported_failed_direction_are_distinct_sources(arm):
    gradients = np.asarray([[0., 1., 2.], [-2., 7., 1.]], dtype=np.float32)
    assert diag.original_direction(gradients, arm=arm) == {'row': 1, 'source': 0, 'target': 1}
    saved = {'directions': [{'field': 'X', 'row': 3, 'source': 1, 'destination': 4, 'sweep': []}]}
    assert diag.original_report_direction(saved, 'X') == {'row': 3, 'source': 1, 'target': 4}
    assert diag.original_report_direction(saved, 'E_upper') is None
    boltz = {'rows': [{'field': 'E_upper', 'row': 183, 'source': 0, 'target': 1}]}
    assert diag.original_report_direction(boltz, 'E_upper') == {'row': 183, 'source': 0, 'target': 1}


def test_original_scale_boundary_failure_is_recorded_without_clipping_or_evaluating_invalid_points():
    values = probabilities()
    values[3, 1], values[3, 0] = np.float32(.0001), np.float32(.7499)
    original = values.copy()
    points, evidence = diag.raw_endpoints(values, {'row': 3, 'source': 1, 'target': 4}, .01)
    np.testing.assert_array_equal(values, original)
    assert not evidence['original_strict_interior_precheck']
    assert not evidence['usable_diagnostic_points']
    assert min(p.min() for p in points) < 0
    assert any(not e['simplex_feasible'] for e in evidence['endpoints'])
    assert all(p.dtype == np.float32 for p in points)


def test_feasible_finer_raw_probe_stays_fp32_and_representably_changed():
    values = probabilities()
    points, evidence = diag.raw_endpoints(values, {'row': 3, 'source': 1, 'target': 4}, .00001)
    assert evidence['usable_diagnostic_points']
    assert all(e['both_affected_entries_changed'] and e['simplex_feasible'] for e in evidence['endpoints'])
    assert all(p.min() >= 0 and p.max() <= 1 for p in points)
    for point in points:
        assert np.count_nonzero(point != values) == 2


def test_finer_scale_that_rounds_to_original_point_is_not_evaluated():
    _, evidence = diag.raw_endpoints(probabilities(), {'row': 0, 'source': 0, 'target': 1}, 1e-12)
    assert not evidence['usable_diagnostic_points']
    assert not all(e['both_affected_entries_changed'] for e in evidence['endpoints'])


def test_only_one_other_original_scale_feasible_contrast_is_selected_deterministically():
    values = probabilities()
    gradient = np.zeros_like(values)
    gradient[3, 4] = 8
    gradient[4, 2] = 5
    original = {'row': 3, 'source': 0, 'target': 4}
    alternative = diag.feasible_alternative(values, gradient, original)
    assert alternative is not None and alternative != original
    assert diag.raw_endpoints(values, alternative, .01)[1]['usable_diagnostic_points']
    assert diag.feasible_alternative(values, gradient, original) == alternative


def test_no_feasible_other_pair_returns_explicit_absence():
    values = np.zeros((1, 6), np.float32)
    values[0, 0] = 1
    assert diag.feasible_alternative(values, np.zeros_like(values), {'row': 0, 'source': 0, 'target': 1}) is None


def test_independent_primal_fd_curvature_repeat_noise_and_relative_error_are_saved_only_as_diagnostics():
    epsilon, x = .01, .5
    function = lambda value: value**3
    result = diag.fd_evidence([function(x - epsilon)] * 2, [function(x + epsilon)] * 2,
                              epsilon, 3 * x*x, function(x))
    assert result['finite_difference'] == pytest.approx(.7501)
    assert result['central_second_difference'] == pytest.approx(3.)
    assert result['maximum_repeat_spread'] == 0
    assert result['certifies_original_gradient_gate'] is False
    assert result['agrees_diagnostic_only']
    noisy = diag.fd_evidence([0., 1.], [0., 1.], epsilon, 1., 0.)
    assert noisy['endpoint_difference_signal'] == 0 and noisy['spread_over_signal'] is None
    assert not noisy['agrees_diagnostic_only']


@pytest.mark.parametrize('damage', ['nonfinite_primal', 'nonfinite_gradient', 'empty_endpoint'])
def test_unresolved_or_nonfinite_native_observations_never_become_diagnostic_agreement(damage):
    minus, plus, analytic = [1., 1.], [2., 2.], 1.
    if damage == 'nonfinite_primal': plus = [np.nan]
    elif damage == 'nonfinite_gradient': analytic = np.inf
    else: minus = []
    with pytest.raises(ValueError):
        diag.fd_evidence(minus, plus, .01, analytic, 1.)


def test_scale_dependence_classification_keeps_mechanism_uncertain_and_never_certifies():
    original = {'epsilon': .01, 'geometry': {'original_strict_interior_precheck': False},
                'evaluation': None}
    assert 'does not admit' in diag.classify_direction([original])[0]
    original['geometry']['original_strict_interior_precheck'] = True
    original['evaluation'] = {'agrees_diagnostic_only': False, 'endpoint_difference_signal': 1., 'maximum_repeat_spread': 0.}
    fine = copy.deepcopy(original)
    fine['epsilon'] = .0001
    fine['evaluation']['agrees_diagnostic_only'] = True
    interpretations = diag.classify_direction([original, fine])
    assert any('not a demonstrated root cause' in text for text in interpretations)


def test_persistent_output_limit_does_not_copy_or_remove_older_evidence(tmp_path):
    prior = tmp_path / 'prior'
    prior.mkdir()
    (prior / 'saved.bin').write_bytes(b'preserved')
    output = tmp_path / 'new'
    output.mkdir()
    (output / 'diagnostic.bin').write_bytes(b'12345')
    assert diag.output_budget(output, 5) == 5
    with pytest.raises(ValueError, match='exceeded'):
        diag.output_budget(output, 4)
    assert (prior / 'saved.bin').read_bytes() == b'preserved'
    with pytest.raises(ValueError, match='256 MiB'):
        diag.output_budget(output, diag.DEFAULT_MAX_OUTPUT_BYTES + 1)


def test_logit_coordinate_control_uses_original_saved_graph_and_independent_toy_primal(tmp_path):
    # Numerical toy only, allocated CPU.  This tests the representation/control
    # machinery; it cannot establish any frozen native model result.
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry.state import GraphLogits
    logits = GraphLogits(jnp.asarray([[1., .5, 0., 0., 0., 0.]], dtype=jnp.float32),
                        jnp.zeros((0, 5), dtype=jnp.float32))
    graph = logits.graph()
    def loss(value, *, key):
        return jnp.sum(value.X**2) + jnp.sum(value.E_upper**2), {}
    proofs = []
    result = diag.logit_control(loss, graph, logits, jax.random.key(17), 'X',
                               {'row': 0, 'source': 0, 'target': 1}, tmp_path, proofs)
    assert len(result['probes']) == 3 and len(proofs) == 13
    assert all(p['both_points_representably_changed'] and p['evaluation'] is not None for p in result['probes'])
    assert all(not p['certifies_original_raw_probability_gate'] for p in result['probes'])
    assert all(Path(p['path']).is_file() and diag.sha(p['path']) == p['sha256'] for p in proofs)
