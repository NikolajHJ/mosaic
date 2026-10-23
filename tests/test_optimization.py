"""Bounded projected-step fixtures; run on HPC, without pretrained weights."""

import pytest
import torch

from latentmol.optimization import NonfiniteObjective, StepControl, backtracking_step, projected_proposal


def invoke(point, gradient, score, *, baseline=None, control=None, **kwargs):
    return backtracking_step(point, torch.zeros_like(point), gradient,
        baseline_loss=score(point) if baseline is None else baseline,
        learning_rate=kwargs.pop("learning_rate", 1.0),
        max_gradient_norm=kwargs.pop("max_gradient_norm", 100.0),
        trust_radius=kwargs.pop("trust_radius", 10.0), control=control or StepControl(version="armijo_v1"), score=score, **kwargs)


def test_backtracking_rejects_overshoot_and_accepts_a_measured_smaller_step():
    point = torch.zeros(1, 1, dtype=torch.float64)
    result = invoke(point, torch.full_like(point, -2), lambda x: (x - 1).square().sum())
    assert result.accepted and result.point.item() == 1
    assert result.report["accepted_scale"] == .5
    assert result.report["trials"][0]["reason"] == "no_strict_decrease"
    accepted = result.report["trials"][1]
    assert accepted["original_gradient_dot_actual_displacement"] == -2
    assert accepted["actual_loss_improvement"] == 1
    assert accepted["armijo_satisfied"] and accepted["informative_decrease"]
    assert result.report["oracle_calls_attempted"] == 2


def test_wrong_direction_stalls_instead_of_advancing_with_clipping():
    point = torch.zeros(1, 1, dtype=torch.float64)
    result = invoke(point, torch.full_like(point, 2), lambda x: (x - 1).square().sum())
    assert not result.accepted and torch.equal(result.point, point)
    assert result.report["stalled"] and result.report["status"] == "stalled"
    assert len(result.report["trials"]) == 16 and result.report["oracle_calls_attempted"] == 16


def test_nonfinite_trial_can_be_rejected_without_hiding_a_finite_smaller_trial():
    point = torch.zeros(1, 1, dtype=torch.float64)
    def score(x):
        return torch.tensor(float("nan"), dtype=x.dtype) if x.item() > 1 else (x - 1).square().sum()
    result = invoke(point, torch.full_like(point, -2), score)
    assert result.accepted and result.point.item() == 1
    assert result.report["trials"][0]["reason"] == "nonfinite_objective"
    assert result.report["oracle_calls_nonfinite"] == 1


def test_declared_nonfinite_exception_is_a_trial_rejection_but_unexpected_errors_propagate():
    point = torch.zeros(1, 1, dtype=torch.float64)
    def reject(x):
        raise NonfiniteObjective("native scalar was nonfinite")
    result = invoke(point, torch.ones_like(point), reject, baseline=torch.tensor(1., dtype=point.dtype))
    assert not result.accepted and result.report["oracle_calls_nonfinite"] == 16
    report = {}
    def unexpected(x):
        raise RuntimeError("unexpected native error")
    with pytest.raises(RuntimeError, match="unexpected native error"):
        invoke(point, torch.ones_like(point), unexpected, baseline=torch.tensor(1., dtype=point.dtype), report=report)
    assert report["status"] == "failed" and report["oracle_calls_unexpected_errors"] == 1
    assert report["trials"][0]["reason"] == "unexpected_objective_error"


@pytest.mark.parametrize("bad_value", [None, [], {"other_head": 1.}])
def test_malformed_callback_is_an_error_not_a_recoverable_nonfinite_trial(bad_value):
    point = torch.zeros(1, 1, dtype=torch.float64)
    report = {}
    with pytest.raises(ValueError, match="scalar loss"):
        invoke(point, torch.ones_like(point), lambda x: bad_value,
               baseline=torch.tensor(1., dtype=point.dtype), report=report)
    assert report["oracle_calls_nonfinite"] == 0
    assert report["status"] == "failed"


def test_zero_or_nonrepresentable_moves_are_retained_without_oracle_calls():
    point = torch.full((1, 1), 1e20, dtype=torch.float32)
    def forbidden(x):
        raise AssertionError("A rounded-away move must not consume an oracle forward")
    result = backtracking_step(point, point.clone(), torch.ones_like(point),
        baseline_loss=torch.tensor(1.), learning_rate=1., max_gradient_norm=1., trust_radius=100.,
        control=StepControl(version="armijo_v1"), score=forbidden)
    assert not result.accepted and torch.equal(result.point, point)
    assert result.report["oracle_calls_attempted"] == 0
    assert len(result.report["trials"]) == 16
    assert all(row["reason"] == "zero_or_nonrepresentable_move" for row in result.report["trials"])


def test_armijo_uses_unclipped_gradient_and_actual_radius_projected_displacement():
    point = torch.zeros(1, 1, dtype=torch.float64)
    result = invoke(point, torch.full_like(point, -6), lambda x: (x - 3).square().sum(),
                    learning_rate=10., max_gradient_norm=1., trust_radius=1.)
    assert result.accepted and result.point.item() == 1
    trial = result.report["trials"][0]
    assert result.report["clipping_factor"] == 1 / 6
    assert trial["proposal_displacement_norm"] == 10
    assert trial["actual_displacement_norm"] == 1
    assert trial["predicted_loss_improvement"] == 6
    assert trial["actual_loss_improvement"] == 5


def test_non_descent_radius_projection_is_rejected_before_scoring():
    # An inconsistent external saved point outside the global ball may be pulled
    # against its proposed direction. The actual projected step must still descend.
    point = torch.full((1, 1), 2., dtype=torch.float64)
    def forbidden(x):
        raise AssertionError("Non-descent projected moves must not be scored")
    result = invoke(point, -torch.ones_like(point), forbidden,
                    baseline=torch.tensor(1., dtype=point.dtype), trust_radius=1.)
    assert not result.accepted and torch.equal(result.point, point)
    assert result.report["oracle_calls_attempted"] == 0
    assert all(row["reason"] == "non_descent_projected_move" for row in result.report["trials"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scale_one_exactly_preserves_legacy_clipping_and_projection_arithmetic(dtype):
    point, start, gradient = (torch.tensor([[.5, -.2]], dtype=dtype), torch.zeros((1, 2), dtype=dtype),
                              torch.tensor([[-9., 3.]], dtype=dtype))
    norm = float(gradient.double().norm())
    clipped = gradient * min(1., .1 / max(norm, 1e-12))
    proposal = point - 1000. * clipped
    delta = proposal - start
    expected = start + delta * min(1., 1. / max(float(delta.double().norm()), 1e-12))
    actual, metadata = projected_proposal(point, start, gradient, learning_rate=1000.,
                                         max_gradient_norm=.1, trust_radius=1.)
    assert torch.equal(actual, expected)
    assert metadata["radius_projection_factor"] < 1


def test_one_ulp_gain_is_not_a_meaningful_success_despite_armijo_roundoff():
    point = torch.zeros(1, 1)
    baseline = torch.tensor(1.)
    lower = torch.nextafter(baseline, torch.tensor(float("-inf")))
    result = invoke(point, torch.ones_like(point), lambda x: lower,
                    baseline=baseline, control=StepControl(version="armijo_v1", armijo=1e-12))
    assert not result.accepted
    assert result.report["policy"]["minimum_improvement_ulps"] == 2.
    assert all(row["armijo_satisfied"] and not row["informative_decrease"]
               for row in result.report["trials"])


def test_supplied_baseline_variation_is_disclosed_and_guards_apparent_improvement():
    point = torch.zeros(1, 1, dtype=torch.float64)
    result = invoke(point, torch.ones_like(point), lambda x: 1 + x.sum(),
                    learning_rate=.1, baseline_variation=.2)
    assert not result.accepted
    assert result.report["baseline_variation_measured"] and result.report["baseline_variation"] == .2
    assert result.report["numerical_guard_is_confidence_bound"] is False


@pytest.mark.parametrize("field,value", [("max_trials", True), ("max_trials", 17), ("armijo", True),
    ("armijo", 0), ("version", "unknown"), ("minimum_improvement_ulps", 1)])
def test_policy_is_explicit_and_bounded(field, value):
    with pytest.raises(ValueError):
        StepControl(**{field: value})
