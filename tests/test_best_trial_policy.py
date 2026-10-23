"""Allocated-HPC fixtures for future full-budget step selection; no model weights."""
import math
import json

import pytest
import torch

from latentmol.optimization import NonfiniteObjective, StepControl, backtracking_step


def choose(score, *, baseline=1., gradient=-1., control=None, point=None, report=None, **kwargs):
    point = torch.zeros(1, 1, dtype=torch.float64) if point is None else point
    return backtracking_step(point, point.clone(), torch.full_like(point, gradient),
        baseline_loss=torch.tensor(baseline, dtype=point.dtype), learning_rate=1.,
        max_gradient_norm=10., trust_radius=10., control=control or StepControl(),
        score=score, report=report, **kwargs)


def test_future_default_and_explicit_historical_policy_have_distinct_records():
    assert StepControl().version == "best_trial_v2"
    assert StepControl().record()["legacy_margin_parameters_applied"] is False
    old = StepControl(version="armijo_v1").record()
    assert old["gradient_for_armijo"] == "original_unclipped_gradient"
    assert "acceptance_rule" not in old


def test_full_size_improvement_returns_immediately_without_scoring_smaller_sizes():
    calls = []
    def score(point):
        calls.append(point.item())
        return torch.tensor(.5, dtype=point.dtype)
    result = choose(score)
    assert calls == [1.] and result.point.item() == 1.
    assert result.report["acceptance_kind"] == "first_improvement"
    assert result.report["monotone_descent"] and result.report["oracle_calls_attempted"] == 1


def test_backtracking_stops_at_first_strict_improvement_not_global_minimum():
    calls = []
    def score(point):
        calls.append(point.item())
        return (point - .375).square().sum()
    result = choose(score, baseline=.140625)
    assert calls == [1., .5]
    assert result.point.item() == .5 and result.report["accepted_trial"] == 1


def test_best_uphill_candidate_can_be_an_earlier_trial_and_is_applied():
    calls = []
    losses = [4., 3., 5., 2.] + [6.] * 12
    def score(point):
        loss = losses[len(calls)]
        calls.append(point.item())
        return torch.tensor(loss, dtype=point.dtype)
    result = choose(score)
    assert len(calls) == 16 and result.point.item() == .125
    assert result.accepted and not result.report["stalled"]
    assert result.report["accepted_trial"] == 3
    assert result.report["acceptance_kind"] == "best_finite_fallback"
    assert result.report["actual_loss_improvement"] == -1.
    assert not result.report["monotone_descent"]
    assert sum(trial["accepted"] for trial in result.report["trials"]) == 1


def test_equal_full_step_is_applied_immediately_without_smaller_trials():
    result = choose(lambda point: torch.ones((), dtype=point.dtype))
    assert result.accepted and result.point.item() == 1.
    assert result.report["accepted_trial"] == 0
    assert result.report["acceptance_kind"] == "full_step_nonincrease"
    assert result.report["actual_loss_improvement"] == 0.
    assert result.report["oracle_calls_attempted"] == 1


def test_all_uphill_tied_candidates_still_choose_the_earliest_trial():
    result = choose(lambda point: torch.full((), 2., dtype=point.dtype))
    assert result.report["accepted_trial"] == 0 and len(result.report["trials"]) == 16
    assert result.report["acceptance_kind"] == "best_finite_fallback"


def test_one_ulp_improvement_does_not_inherit_the_historical_armijo_margin():
    point = torch.zeros(1, 1, dtype=torch.float32)
    lower = torch.nextafter(torch.tensor(1.), torch.tensor(-math.inf))
    result = choose(lambda candidate: lower, point=point)
    assert result.accepted and result.report["accepted_trial"] == 0
    assert result.report["acceptance_kind"] == "first_improvement"


def test_finite_equal_noop_keeps_full_budget_semantics_without_claiming_movement():
    result = choose(lambda point: torch.ones((), dtype=point.dtype), gradient=0.)
    assert result.accepted and torch.equal(result.point, torch.zeros_like(result.point))
    assert result.report["no_op"] and not result.report["monotone_descent"]
    assert result.report["acceptance_kind"] == "full_step_nonincrease"
    assert len(result.report["trials"]) == 1
    assert result.report["actual_displacement_norm"] == 0.


def test_noop_replay_noise_does_not_claim_a_first_improvement():
    result = choose(lambda point: torch.tensor(.5, dtype=point.dtype), gradient=0.)
    assert result.report["acceptance_kind"] == "full_step_nonincrease"
    assert result.report["no_op"] and not result.report["monotone_descent"]
    assert len(result.report["trials"]) == 1


def test_non_descent_projected_direction_is_scored_under_v2():
    point = torch.tensor([[2.]], dtype=torch.float64)
    calls = []
    def score(candidate):
        calls.append(candidate.item())
        return torch.tensor(2., dtype=candidate.dtype)
    result = backtracking_step(point, torch.zeros_like(point), -torch.ones_like(point),
        baseline_loss=torch.tensor(1., dtype=point.dtype), learning_rate=1.,
        max_gradient_norm=1., trust_radius=1., control=StepControl(), score=score)
    assert calls and result.accepted and result.point.item() == 1.
    assert result.report["trials"][0]["original_gradient_dot_actual_displacement"] > 0


def test_mixed_nonfinite_and_finite_trials_select_only_a_finite_observation():
    def score(candidate):
        return torch.tensor(math.nan if candidate.item() > .5 else 2., dtype=candidate.dtype)
    result = choose(score)
    assert result.accepted and result.point.item() == .5
    assert result.report["oracle_calls_nonfinite"] == 1


@pytest.mark.parametrize("raised", [False, True])
def test_no_finite_trial_is_a_failed_run_not_a_stall_or_completed_budget(raised):
    report = {}
    def score(candidate):
        if raised:
            raise NonfiniteObjective("native scalar is nonfinite")
        return torch.tensor(math.nan, dtype=candidate.dtype)
    with pytest.raises(NonfiniteObjective, match="No finite candidate"):
        choose(score, report=report)
    assert report["status"] == "failed" and not report["accepted"] and not report["stalled"]
    assert report["oracle_calls_attempted"] == 16


def test_structural_exception_after_a_finite_trial_cannot_be_hidden_by_fallback():
    report, calls = {}, []
    def score(candidate):
        if calls:
            raise RuntimeError("native contract broken")
        calls.append(candidate.item())
        return torch.tensor(2., dtype=candidate.dtype)
    with pytest.raises(RuntimeError, match="native contract broken"):
        choose(score, report=report)
    assert report["status"] == "failed" and not report["accepted"]


def test_first_improvement_ignores_measured_variation_as_an_acceptance_margin():
    result = choose(lambda point: torch.tensor(.9, dtype=point.dtype), baseline_variation=100.)
    assert result.report["accepted_trial"] == 0 and result.report["monotone_descent"]
    assert result.report["baseline_variation"] == 100.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_finite_dtype_endpoint_keeps_unused_ulp_diagnostics_json_safe(dtype):
    point = torch.zeros(1, 1, dtype=dtype)
    result = choose(lambda candidate: torch.tensor(torch.finfo(dtype).max, dtype=dtype), point=point)
    assert result.accepted and result.report["acceptance_kind"] == "best_finite_fallback"
    assert all(math.isfinite(trial["output_ulp"]) for trial in result.report["trials"])
    json.dumps(result.report, allow_nan=False)
