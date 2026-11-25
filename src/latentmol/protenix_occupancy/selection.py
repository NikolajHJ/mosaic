"""Scalar selection bookkeeping for the allocated Protenix deletion pilot.

The score callback owns every model evaluation and its artifacts. This module
neither loads a model nor turns a relaxed preference into chemical validation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence


SELECTION_POLICY = "protenix-q-best-trial-v2-v1"
ENDPOINT_POLICY = "protenix-native-endpoint-pair-v1"


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    return float(value)


def _safe_payload(value):
    """Keep finite callback metadata while making nonfinite receipts JSON safe."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item) for item in value]
    return value


def select_scalar_step(
    q: float,
    baseline_loss: float,
    direction: int,
    score: Callable[[float], dict],
    *,
    persist: Callable[[list[dict]], None] | None = None,
) -> dict:
    """Apply the declared full-step/halving/fallback scalar step policy.

    ``score`` returns ``{"q": actual_quantized_q, "loss": scalar, ...}``.
    Losses may have either sign. Each completed or exceptional score attempt is
    persisted before another evaluation starts. Persistence and scoring errors
    propagate, including resource-budget exceptions. Trial indices are zero
    based; ties select the earliest trial. Actual callback q values determine
    no-op reporting, so clipping or quantization is not disguised as movement.
    """
    q = _number(q, "q")
    baseline_loss = _number(baseline_loss, "baseline_loss")
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("Require finite probability q in [0, 1]")
    if not math.isfinite(baseline_loss):
        raise ValueError("Require finite baseline loss")
    if isinstance(direction, bool) or direction not in (-1, 0, 1):
        raise ValueError("Direction must be -1, 0 or +1")
    if not callable(score) or (persist is not None and not callable(persist)):
        raise TypeError("Scoring and persistence callbacks must be callable")

    trials: list[dict] = []

    def save():
        if persist is not None:
            # Receipt writers cannot mutate the live selection record.
            persist(copy.deepcopy(trials))

    def accepted(index: int, kind: str) -> dict:
        chosen = trials[index]
        return dict(version=SELECTION_POLICY, q=chosen["q"], loss=chosen["loss"],
            acceptance_kind=kind, no_op=chosen["q"] == q,
            improving=chosen["loss"] < baseline_loss and chosen["q"] != q,
            scalar_strictly_decreased=chosen["loss"] < baseline_loss,
            trials=trials, selected_index=index)

    limit = 1 if direction == 0 else 16
    for index in range(limit):
        scale = 2. ** -index
        proposal = min(.99, max(.01, q + .1 * direction * scale))
        trial = dict(trial_index=index, step_scale=scale,
            proposed_q=proposal, q=None, loss=None, finite=False)
        try:
            payload = score(proposal)
            if not isinstance(payload, dict):
                raise TypeError("Score callback must return a dictionary")
            actual_q = _number(payload["q"], "score q")
            loss = _number(payload["loss"], "score loss")
        except Exception as error:
            trial.update(status="score_exception", error_type=type(error).__name__,
                error=str(error))
            trials.append(trial)
            save()
            raise
        trial.update(status="evaluated", q=actual_q if math.isfinite(actual_q) else None,
            loss=loss if math.isfinite(loss) else None,
            finite=math.isfinite(actual_q) and 0 <= actual_q <= 1 and math.isfinite(loss),
            payload=_safe_payload(payload))
        if not math.isfinite(actual_q):
            trial["nonfinite_q_repr"] = repr(actual_q)
        elif not 0 <= actual_q <= 1:
            trial["invalid_probability"] = True
        if not math.isfinite(loss):
            trial["nonfinite_loss_repr"] = repr(loss)
        trials.append(trial)
        save()
        if not trial["finite"]:
            continue
        if loss < baseline_loss and actual_q != q:
            return accepted(index, "first_improvement")
        if index == 0 and loss <= baseline_loss:
            return accepted(index, "full_step_nonincrease")

    finite = [trial for trial in trials if trial["finite"]]
    if not finite:
        raise RuntimeError("All scalar-step trials were nonfinite or invalid")
    chosen = min(finite, key=lambda trial: (trial["loss"], trial["trial_index"]))
    return accepted(chosen["trial_index"], "best_finite_fallback")


def endpoint_pair_summary(full_losses: Sequence[float], deleted_losses: Sequence[float]) -> dict:
    """Summarize three independent repeats per native endpoint.

    The empirical repeat allowance is not a statistical confidence bound.
    Bernoulli interpolation of these endpoints is an analysis control only;
    the selector never receives it as its optimization objective.
    """
    groups = []
    for name, values in (("full_losses", full_losses), ("deleted_losses", deleted_losses)):
        samples = [_number(value, name) for value in values]
        if len(samples) != 3 or any(not math.isfinite(value) for value in samples):
            raise ValueError("Require exactly three finite losses per native endpoint")
        groups.append(samples)
    full, deleted = groups
    full_mean, deleted_mean = math.fsum(full) / 3, math.fsum(deleted) / 3
    full_range, deleted_range = max(full) - min(full), max(deleted) - min(deleted)
    uncertainty = max(1e-12, 4 * max(full_range, deleted_range))
    cap = 1e-6
    passed = uncertainty <= cap
    gap = full_mean - deleted_mean
    resolved = passed and abs(gap) > 10 * uncertainty
    preference = ("full" if gap < 0 else "deleted") if resolved else "indeterminate"
    return dict(version=ENDPOINT_POLICY, full_losses=full, deleted_losses=deleted,
        full_mean=full_mean, deleted_mean=deleted_mean,
        full_range=full_range, deleted_range=deleted_range,
        uncertainty=uncertainty, loss_difference_uncertainty=uncertainty,
        maximum_uncertainty=cap, calibration_passed=passed,
        endpoint_gap=gap, resolved=resolved, preference=preference,
        bernoulli_analysis_only=dict(intercept=deleted_mean, slope=gap,
            formula="(1-q)*deleted_mean + q*full_mean", used_for_guidance=False))
