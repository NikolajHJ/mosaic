"""Allocated-HPC scalar bookkeeping fixtures; do not execute locally."""

import json
import math

import pytest

from latentmol.protenix_occupancy.selection import endpoint_pair_summary, select_scalar_step


def scorer(values, actual_q=None):
    iterator = iter(values)
    return lambda q: dict(q=q if actual_q is None else actual_q,
        loss=next(iterator), artifact="saved-output.pt")


def test_full_step_improvement():
    result = select_scalar_step(.5, 1., -1, scorer([.9]))
    assert result["q"] == pytest.approx(.4)
    assert result["acceptance_kind"] == "first_improvement"
    assert result["improving"] and not result["no_op"]
    assert result["selected_index"] == 0 and len(result["trials"]) == 1


def test_full_equality_is_nonincrease():
    result = select_scalar_step(.5, 1., 1, scorer([1.]))
    assert result["acceptance_kind"] == "full_step_nonincrease"
    assert not result["improving"] and not result["no_op"]


def test_later_equality_does_not_end_halving():
    result = select_scalar_step(.5, 1., 1, scorer([2., 1., .9]))
    assert result["acceptance_kind"] == "first_improvement"
    assert result["selected_index"] == 2
    assert result["q"] == pytest.approx(.525)


def test_wrong_direction_uses_earliest_lowest_finite_fallback():
    losses = [3., 2., 2.] + [4.] * 13
    result = select_scalar_step(.5, 1., -1, scorer(losses))
    assert result["acceptance_kind"] == "best_finite_fallback"
    assert result["selected_index"] == 1 and result["loss"] == 2.
    assert not result["improving"]
    assert len(result["trials"]) == 16
    assert result["trials"][-1]["step_scale"] == 2. ** -15
    assert result["trials"][1]["payload"]["artifact"] == "saved-output.pt"


def test_later_equal_fallback_is_not_strict_improvement():
    result = select_scalar_step(.5, 1., 1, scorer([2.] + [1.] * 15))
    assert result["selected_index"] == 1
    assert result["acceptance_kind"] == "best_finite_fallback"
    assert not result["improving"]


@pytest.mark.parametrize("q,direction,expected", [(.01, -1, .01), (.99, 1, .99)])
def test_clipped_full_step_can_be_noop(q, direction, expected):
    result = select_scalar_step(q, 1., direction, scorer([1.]))
    assert result["q"] == expected and result["no_op"]
    assert result["acceptance_kind"] == "full_step_nonincrease"


def test_zero_direction_consumes_one_honest_noop():
    result = select_scalar_step(.2, 1., 0, scorer([1.]))
    assert len(result["trials"]) == 1 and result["no_op"]
    assert result["q"] == .2


def test_actual_quantized_q_controls_noop():
    # An unchanged input can fluctuate numerically; do not call that descent.
    result = select_scalar_step(.5, 1., 1, scorer([.9], actual_q=.5))
    assert result["trials"][0]["proposed_q"] == .6
    assert result["q"] == .5 and result["no_op"]
    assert result["acceptance_kind"] == "full_step_nonincrease"
    assert not result["improving"] and result["scalar_strictly_decreased"]


def test_negative_losses_are_valid():
    result = select_scalar_step(.5, -1., -1, scorer([-2.]))
    assert result["loss"] == -2. and result["improving"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_loss_is_recorded_then_skipped(value):
    result = select_scalar_step(.5, 1., 1, scorer([value, .5]))
    assert result["selected_index"] == 1
    assert not result["trials"][0]["finite"]
    assert result["trials"][0]["loss"] is None
    json.dumps(result, allow_nan=False)


def test_nonfinite_actual_q_is_recorded_and_skipped():
    count = 0
    def score(q):
        nonlocal count
        count += 1
        return dict(q=float("nan") if count == 1 else q, loss=.5)
    result = select_scalar_step(.5, 1., 1, score)
    assert result["selected_index"] == 1
    assert result["trials"][0]["q"] is None
    json.dumps(result, allow_nan=False)


def test_all_nonfinite_trials_fail_after_persisting_every_attempt():
    saved = []
    with pytest.raises(RuntimeError, match="All scalar-step trials"):
        select_scalar_step(.5, 1., 1, scorer([float("nan")] * 16), persist=saved.append)
    assert [len(rows) for rows in saved] == list(range(1, 17))
    assert saved[-1][-1]["trial_index"] == 15


def test_zero_direction_nonfinite_fails_after_one_trial():
    saved = []
    with pytest.raises(RuntimeError):
        select_scalar_step(.5, 1., 0, scorer([float("nan")]), persist=saved.append)
    assert len(saved) == len(saved[0]) == 1


def test_completed_and_exceptional_trials_are_persisted_then_error_propagates():
    class BudgetExhausted(RuntimeError):
        pass
    saved = []
    count = 0
    def score(q):
        nonlocal count
        count += 1
        if count == 2:
            raise BudgetExhausted("Forward budget exhausted")
        return dict(q=q, loss=2.)
    with pytest.raises(BudgetExhausted, match="Forward budget"):
        select_scalar_step(.5, 1., 1, score, persist=saved.append)
    assert [len(rows) for rows in saved] == [1, 2]
    assert saved[-1][-1]["status"] == "score_exception"
    assert saved[-1][-1]["error_type"] == "BudgetExhausted"


def test_receipt_writer_cannot_mutate_live_trials():
    def mutate(rows):
        rows.clear()
    result = select_scalar_step(.5, 1., 1, scorer([.5]), persist=mutate)
    assert len(result["trials"]) == 1


def test_persistence_error_propagates_before_next_forward():
    def failed(_rows):
        raise OSError("Receipt write failed")
    with pytest.raises(OSError, match="Receipt write failed"):
        select_scalar_step(.5, 1., 1, scorer([2.]), persist=failed)


@pytest.mark.parametrize("direction", [True, 2, -2, None])
def test_invalid_direction_rejected(direction):
    with pytest.raises(ValueError):
        select_scalar_step(.5, 1., direction, scorer([]))


@pytest.mark.parametrize("q,loss", [(float("nan"), 1.), (-.1, 1.), (1.1, 1.), (.5, float("inf"))])
def test_invalid_baseline_rejected(q, loss):
    with pytest.raises(ValueError):
        select_scalar_step(q, loss, 1, scorer([]))


def test_malformed_score_is_preserved_as_exception():
    saved = []
    with pytest.raises(TypeError, match="dictionary"):
        select_scalar_step(.5, 1., 1, lambda _q: None, persist=saved.append)
    assert saved[0][0]["status"] == "score_exception"


@pytest.mark.parametrize("full,deleted,preference,gap", [
    ([1.] * 3, [2.] * 3, "full", -1.),
    ([2.] * 3, [1.] * 3, "deleted", 1.),
    ([1.] * 3, [1.] * 3, "indeterminate", 0.),
])
def test_endpoint_direction_and_gap(full, deleted, preference, gap):
    result = endpoint_pair_summary(full, deleted)
    assert result["endpoint_gap"] == gap
    assert result["preference"] == preference
    assert result["calibration_passed"]
    assert result["uncertainty"] == 1e-12
    assert result["bernoulli_analysis_only"]["slope"] == gap
    assert not result["bernoulli_analysis_only"]["used_for_guidance"]


def test_endpoint_range_allowance_and_unresolved_gap():
    result = endpoint_pair_summary([1., 1. + 1e-8, 1.], [1. + 1e-7] * 3)
    assert result["uncertainty"] == pytest.approx(4e-8)
    assert result["calibration_passed"]
    assert not result["resolved"] and result["preference"] == "indeterminate"


def test_endpoint_excess_noise_does_not_establish_preference():
    result = endpoint_pair_summary([1., 1.001, 1.], [2.] * 3)
    assert not result["calibration_passed"]
    assert not result["resolved"] and result["preference"] == "indeterminate"


@pytest.mark.parametrize("full,deleted", [([1., 1.], [2.] * 3),
    ([1.] * 4, [2.] * 3), ([1.] * 3, [float("nan")] * 3)])
def test_endpoint_requires_three_finite_repeats(full, deleted):
    with pytest.raises(ValueError):
        endpoint_pair_summary(full, deleted)
