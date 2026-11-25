"""Versioned bounded step selection for a frozen scalar molecular objective.

Gradient clipping controls a proposal's length; it does not prove that the
proposal improves the actual score. ``armijo_v1`` separately evaluates the
original scalar objective at a fixed molecular mask/context and accepts only a
measurable sufficient decrease. The future default ``best_trial_v2`` accepts
the full step if nonincreasing, then the first strict improvement or the best finite trial after exhausting the
declared sizes; it does not stop merely because that trial worsens the loss.
Chemistry, fixed repair targets and context changes stay with callers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class StepControl:
    """Explicit versioned opt-in; all trial scales are declared before scoring."""

    version: str = "best_trial_v2"
    max_trials: int = 16
    armijo: float = 1e-4
    minimum_improvement_ulps: float = 2.0

    def __post_init__(self):
        if self.version not in ("armijo_v1", "best_trial_v2"):
            raise ValueError("Unknown step-control version")
        if type(self.max_trials) is not int or not 1 <= self.max_trials <= 16:
            raise ValueError("Step control requires one to sixteen bounded trials")
        if (type(self.armijo) not in (int, float) or not math.isfinite(self.armijo)
                or not 0 < self.armijo < 1):
            raise ValueError("Armijo fraction must be finite and strictly between zero and one")
        if (type(self.minimum_improvement_ulps) not in (int, float)
                or not math.isfinite(self.minimum_improvement_ulps) or self.minimum_improvement_ulps < 2):
            raise ValueError("Require at least two scalar output ULPs of observed improvement")

    @classmethod
    def from_mapping(cls, value: dict) -> "StepControl":
        if (not isinstance(value, dict) or "version" not in value
                or set(value) - {"version", "max_trials", "armijo", "minimum_improvement_ulps"}):
            raise ValueError("step_control needs an explicit version and only supported settings")
        return cls(**value)

    def record(self) -> dict:
        if self.version == "best_trial_v2":
            return {**asdict(self), "trial_scales": [2.0 ** -index for index in range(self.max_trials)],
                    "radius_center": "initial_z", "gradient_for_armijo": "diagnostic_only",
                    "acceptance_rule": "full step if loss does not increase; otherwise first strict decrease, then lowest finite trial loss, earliest tie",
                    "legacy_margin_parameters_applied": False,
                    "no_finite_trial": "fail; never fabricate a successful update",
                    "scope": "same scalar objective, fixed mask/context/repair target and coefficient; loss may increase"}
        return {**asdict(self), "trial_scales": [2.0 ** -index for index in range(self.max_trials)],
                "radius_center": "initial_z", "gradient_for_armijo": "original_unclipped_gradient",
                "observed_improvement_guard": "at least minimum_improvement_ulps actual scalar output ULPs plus supplied baseline variation",
                "scope": "same scalar objective, fixed mask, fixed context; no monotonicity claim across context switches"}


class NonfiniteObjective(ValueError):
    """A caller-declared nonfinite scalar trial, rejected without hiding other bugs."""


@dataclass
class StepResult:
    point: torch.Tensor
    accepted: bool
    report: dict


def _loss_observation(value, declared_dtype: str | None = None, *, finite_ulp_only=False):
    """Accept tensor/scalar callbacks or an explicit loss/dtype metadata mapping."""
    metadata = {}
    if isinstance(value, dict):
        metadata = dict(value)
        declared_dtype = metadata.get("loss_dtype", declared_dtype)
        value = metadata.get("loss")
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not value.is_floating_point():
            raise ValueError("Step objective must be an explicitly reduced floating scalar")
        declared_dtype = str(value.dtype)
        value = float(value.detach())
    if value is None or type(value) not in (int, float):
        raise ValueError("Step objective callback must return a scalar loss, not a missing or malformed output")
    if not math.isfinite(value):
        raise NonfiniteObjective("Trial scalar objective is nonfinite")
    supported = {str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    if declared_dtype not in supported:
        raise ValueError("Declare the actual objective dtype for a Python scalar callback")
    scalar = torch.tensor(value, dtype=supported[declared_dtype])
    if not bool(scalar.isfinite()):
        raise NonfiniteObjective("Trial objective overflows its declared scalar dtype")
    up = torch.nextafter(scalar, torch.full_like(scalar, float("inf")))
    down = torch.nextafter(scalar, torch.full_like(scalar, float("-inf")))
    differences = [float(up.double() - scalar.double()), float(scalar.double() - down.double())]
    # v2 does not use ULPs as acceptance margins. At a finite dtype endpoint,
    # keep the finite adjacent spacing as JSON-safe diagnostic evidence; v1's
    # original max-neighbour arithmetic stays unchanged by the default flag.
    ulp = max(d for d in differences if math.isfinite(d)) if finite_ulp_only else max(differences)
    return float(value), declared_dtype, ulp, metadata


@torch.no_grad()
def projected_proposal(point, start, gradient, *, learning_rate, max_gradient_norm,
                       trust_radius, scale=1.0):
    """Form the legacy clipped step and initial-z radius projection, with evidence.

    Scale one uses the same operation order as the historical seven-line update.
    This function evaluates no objective and provides no acceptance claim. The
    returned pair is ``(candidate, metadata)``; ``candidate`` is detached.
    """
    if (point.shape != start.shape or point.shape != gradient.shape or point.numel() == 0
            or point.dtype != start.dtype or point.dtype != gradient.dtype
            or point.device != start.device or point.device != gradient.device
            or not point.is_floating_point()):
        raise ValueError("Point, initial point and gradient need the same floating shape/device")
    if not all(bool(value.isfinite().all()) for value in (point, start, gradient)):
        raise ValueError("Proposal inputs must be finite")
    for name, value in (("learning_rate", learning_rate), ("max_gradient_norm", max_gradient_norm),
                        ("trust_radius", trust_radius), ("scale", scale)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
    norm = float(gradient.double().norm())
    if not math.isfinite(norm):
        raise ValueError("Nonfinite analytic gradient norm")
    factor = min(1.0, max_gradient_norm / max(norm, 1e-12))
    clipped = gradient * factor
    proposal = point - (learning_rate * scale) * clipped
    delta = proposal - start
    delta_norm = float(delta.double().norm())
    projection_factor = min(1.0, trust_radius / max(delta_norm, 1e-12))
    candidate = start + delta * projection_factor
    displacement = candidate.double() - point.double()
    metadata = {"gradient_norm": norm, "clipping_factor": factor,
                "clipped_gradient_norm": float(clipped.double().norm()),
                "proposal_displacement_norm": float((proposal.double() - point.double()).norm()),
                "proposal_initial_displacement_norm": delta_norm,
                "radius_projection_factor": projection_factor,
                "actual_displacement_norm": float(displacement.norm()),
                "actual_initial_displacement_norm": float((candidate - start).double().norm()),
                "original_gradient_dot_actual_displacement": float((gradient.double() * displacement).sum()),
                "finite_proposal": bool(proposal.isfinite().all()),
                "finite_candidate": bool(candidate.isfinite().all())}
    return candidate.detach(), metadata


@torch.no_grad()
def backtracking_step(point: torch.Tensor, start: torch.Tensor, gradient: torch.Tensor, *,
                      baseline_loss, learning_rate: float, max_gradient_norm: float,
                      trust_radius: float, control: StepControl, score: Callable,
                      baseline_loss_dtype: str | None = None,
                      baseline_variation: float | None = None,
                      report: dict | None = None, persist: Callable = lambda: None) -> StepResult:
    """Select a step under the explicit version, keeping historical v1 intact.

    Scale one reproduces the legacy clipped proposal and global-radius projection.
    Armijo uses the original gradient dotted with the ACTUAL representable step,
    measured in widened arithmetic. It never uses the clipped gradient as the
    acceptance derivative. Every skipped/evaluated trial remains in the report.

    A declared scalar-ULP guard avoids approving apparent improvements caused solely by
    output rounding. This is not a calibrated bound on whole-network arithmetic.
    Supplied baseline variation is additional measured evidence; its absence is
    recorded as unmeasured, and no repeatability or confidence claim is inferred.
    Unexpected callback errors are persisted and propagated, not called stalls.
    """
    if control.version == "best_trial_v2":
        return _best_trial_step(point, start, gradient, baseline_loss=baseline_loss,
            learning_rate=learning_rate, max_gradient_norm=max_gradient_norm,
            trust_radius=trust_radius, control=control, score=score,
            baseline_loss_dtype=baseline_loss_dtype, baseline_variation=baseline_variation,
            report=report, persist=persist)
    if (point.shape != start.shape or point.shape != gradient.shape or point.numel() == 0
            or point.dtype != start.dtype or point.dtype != gradient.dtype
            or point.device != start.device or point.device != gradient.device
            or not point.is_floating_point()):
        raise ValueError("Point, initial point and gradient need the same floating shape/device")
    if not all(bool(value.isfinite().all()) for value in (point, start, gradient)):
        raise ValueError("Step control requires finite point, initial point and gradient")
    for name, value in (("learning_rate", learning_rate), ("max_gradient_norm", max_gradient_norm),
                        ("trust_radius", trust_radius)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {name}")
    if baseline_variation is not None and (type(baseline_variation) not in (int, float)
            or not math.isfinite(baseline_variation) or baseline_variation < 0):
        raise ValueError("Measured baseline variation must be finite and nonnegative")
    baseline, dtype, base_ulp, _ = _loss_observation(baseline_loss, baseline_loss_dtype)
    norm = float(gradient.double().norm())
    if not math.isfinite(norm):
        raise ValueError("Nonfinite analytic gradient norm")
    factor = min(1.0, max_gradient_norm / max(norm, 1e-12))
    clipped = gradient * factor
    report = report if report is not None else {}
    report.update(policy=control.record(), status="running", accepted=False, stalled=False,
                  gradient_norm=norm, clipping_factor=factor,
                  clipped_gradient_norm=float(clipped.double().norm()),
                  learning_rate=learning_rate, max_gradient_norm=max_gradient_norm, trust_radius=trust_radius,
                  baseline_loss=baseline, baseline_loss_dtype=dtype, baseline_output_ulp=base_ulp,
                  baseline_variation=baseline_variation,
                  baseline_variation_measured=baseline_variation is not None,
                  initial_displacement_norm=float((point - start).double().norm()),
                  oracle_calls_attempted=0, oracle_calls_completed=0, oracle_calls_nonfinite=0,
                  oracle_calls_unexpected_errors=0,
                  trials=[], numerical_guard_is_confidence_bound=False)
    persist()
    for index, scale in enumerate(report["policy"]["trial_scales"]):
        candidate, details = projected_proposal(point, start, gradient, learning_rate=learning_rate,
            max_gradient_norm=max_gradient_norm, trust_radius=trust_radius, scale=scale)
        predicted_change = details["original_gradient_dot_actual_displacement"]
        trial = {"index": index, "scale": scale, **details,
                 "predicted_loss_improvement": -predicted_change,
                 "required_armijo_improvement": -control.armijo * predicted_change,
                 "candidate_loss": None, "actual_loss_improvement": None,
                 "oracle_called": False, "accepted": False, "reason": None}
        report["trials"].append(trial)
        for key, value in list(trial.items()):
            if type(value) is float and not math.isfinite(value):
                trial[key] = None
        if not details["finite_proposal"] or not details["finite_candidate"]:
            trial["reason"] = "nonfinite_proposal"
        elif torch.equal(candidate, point):
            trial["reason"] = "zero_or_nonrepresentable_move"
        elif not math.isfinite(predicted_change) or predicted_change >= 0:
            trial["reason"] = "non_descent_projected_move"
        else:
            trial["oracle_called"] = True
            report["oracle_calls_attempted"] += 1
            persist()
            try:
                observed = score(candidate.detach())
                report["oracle_calls_completed"] += 1
                loss, observed_dtype, ulp, metadata = _loss_observation(observed, dtype)
            except NonfiniteObjective as exc:
                report["oracle_calls_nonfinite"] += 1
                trial.update(reason="nonfinite_objective", error=str(exc), error_type=type(exc).__name__)
            except Exception as exc:
                report["oracle_calls_unexpected_errors"] += 1
                trial.update(reason="unexpected_objective_error", error=str(exc), error_type=type(exc).__name__)
                report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                persist()
                raise
            else:
                improvement = baseline - loss
                guard = control.minimum_improvement_ulps * max(base_ulp, ulp) + (baseline_variation or 0.0)
                trial.update(candidate_loss=loss, candidate_loss_dtype=observed_dtype, output_ulp=ulp,
                             actual_loss_improvement=improvement, observed_improvement_guard=guard,
                             strictly_decreased=improvement > 0,
                             informative_decrease=improvement >= guard,
                             armijo_satisfied=improvement >= -control.armijo * predicted_change,
                             objective_metadata=metadata)
                accepted = (trial["strictly_decreased"] and trial["informative_decrease"]
                            and trial["armijo_satisfied"])
                trial.update(accepted=accepted, reason="accepted" if accepted else
                             ("no_strict_decrease" if not trial["strictly_decreased"] else
                              "decrease_below_numerical_guard" if not trial["informative_decrease"] else
                              "insufficient_armijo_decrease"))
                if accepted:
                    report.update(status="accepted", accepted=True, accepted_trial=index,
                                  accepted_scale=scale, stalled=False,
                                  actual_loss_improvement=improvement,
                                  actual_displacement_norm=trial["actual_displacement_norm"])
                    persist()
                    return StepResult(candidate.detach(), True, report)
        persist()
    report.update(status="stalled", accepted=False, stalled=True, stall_reason="trial_budget_exhausted",
                  actual_loss_improvement=0.0, actual_displacement_norm=0.0)
    persist()
    return StepResult(point.detach().clone(), False, report)


@torch.no_grad()
def _best_trial_step(point, start, gradient, *, baseline_loss, learning_rate,
                     max_gradient_norm, trust_radius, control, score,
                     baseline_loss_dtype, baseline_variation, report, persist):
    """Use a nonincreasing full step, then bounded search with uphill fallback.

    Baseline is deliberately NOT an additional candidate: adding it would turn
    every uphill fallback into an unchanged-point stall. Keep the actual point
    for the lowest-loss trial, since it need not be the last/smallest trial.
    Projection, scalar parsing and nonfinite classification reuse v1 helpers;
    v1's Armijo/ULP/directional acceptance gates do not apply to this version.
    """
    # Validate inputs with the same proposal arithmetic, without scoring a model.
    # A finite input may overflow a full proposal; smaller sizes can still work.
    projected_proposal(point, start, gradient, learning_rate=learning_rate,
        max_gradient_norm=max_gradient_norm, trust_radius=trust_radius)
    if baseline_variation is not None and (type(baseline_variation) not in (int, float)
            or not math.isfinite(baseline_variation) or baseline_variation < 0):
        raise ValueError("Measured baseline variation must be finite and nonnegative")
    baseline, dtype, base_ulp, _ = _loss_observation(baseline_loss, baseline_loss_dtype, finite_ulp_only=True)
    norm = float(gradient.double().norm())
    factor = min(1.0, max_gradient_norm / max(norm, 1e-12))
    report = report if report is not None else {}
    report.update(policy=control.record(), status="running", accepted=False, stalled=False,
        gradient_norm=norm, clipping_factor=factor,
        clipped_gradient_norm=float((gradient * factor).double().norm()),
        learning_rate=learning_rate, max_gradient_norm=max_gradient_norm, trust_radius=trust_radius,
        baseline_loss=baseline, baseline_loss_dtype=dtype, baseline_output_ulp=base_ulp,
        baseline_variation=baseline_variation, baseline_variation_measured=baseline_variation is not None,
        initial_displacement_norm=float((point - start).double().norm()),
        oracle_calls_attempted=0, oracle_calls_completed=0, oracle_calls_nonfinite=0,
        oracle_calls_unexpected_errors=0, trials=[], numerical_guard_is_confidence_bound=False)
    best = None

    def accept(index, candidate, kind):
        trial = report["trials"][index]
        trial.update(accepted=True, reason="accepted")
        report.update(status="accepted", accepted=True, stalled=False, accepted_trial=index,
            accepted_scale=trial["scale"], acceptance_kind=kind,
            monotone_descent=trial["strictly_decreased"] and not trial["zero_displacement"],
            no_op=trial["zero_displacement"], selected_loss=trial["candidate_loss"],
            actual_loss_improvement=trial["actual_loss_improvement"],
            actual_displacement_norm=trial["actual_displacement_norm"])
        persist()
        return StepResult(candidate.detach().clone(), True, report)

    persist()
    for index, scale in enumerate(report["policy"]["trial_scales"]):
        candidate, details = projected_proposal(point, start, gradient,
            learning_rate=learning_rate, max_gradient_norm=max_gradient_norm,
            trust_radius=trust_radius, scale=scale)
        trial = dict(index=index, scale=scale, **details,
            zero_displacement=torch.equal(candidate, point), candidate_loss=None,
            actual_loss_improvement=None, oracle_called=False, accepted=False, reason=None)
        # Keep invalid proposal evidence JSON-safe; an invalid proposal is never
        # eligible just because projecting an infinity happens to yield a number.
        for key, value in list(trial.items()):
            if type(value) is float and not math.isfinite(value):
                trial[key] = None
        report["trials"].append(trial)
        if not details["finite_proposal"] or not details["finite_candidate"]:
            trial["reason"] = "nonfinite_proposal"
            persist()
            continue
        trial["oracle_called"] = True
        report["oracle_calls_attempted"] += 1
        persist()
        try:
            observed = score(candidate.detach())
            report["oracle_calls_completed"] += 1
            loss, observed_dtype, ulp, metadata = _loss_observation(observed, dtype, finite_ulp_only=True)
        except NonfiniteObjective as exc:
            report["oracle_calls_nonfinite"] += 1
            trial.update(reason="nonfinite_objective", error=str(exc), error_type=type(exc).__name__)
        except Exception as exc:
            report["oracle_calls_unexpected_errors"] += 1
            trial.update(reason="unexpected_objective_error", error=str(exc), error_type=type(exc).__name__)
            report.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            persist()
            raise
        else:
            improvement = baseline - loss
            trial.update(candidate_loss=loss, candidate_loss_dtype=observed_dtype,
                output_ulp=ulp, actual_loss_improvement=improvement,
                strictly_decreased=improvement > 0, objective_metadata=metadata,
                reason="no_strict_decrease")
            # Strict '<' preserves the earliest trial on an exact loss tie.
            if best is None or loss < best[0]:
                best = (loss, index, candidate.detach().clone())
            if improvement > 0 and not trial["zero_displacement"]:
                return accept(index, candidate, "first_improvement")
            if index == 0 and loss <= baseline:
                # The user requested smaller sizes only when the full step
                # worsens the loss. Equality needs no extra model evaluations.
                return accept(index, candidate, "full_step_nonincrease")
        persist()
    if best is not None:
        return accept(best[1], best[2], "best_finite_fallback")
    report.update(status="failed", accepted=False, stalled=False,
                  failure_reason="no_finite_trial", error_type="NonfiniteObjective")
    persist()
    raise NonfiniteObjective("No finite candidate objective within the declared trial budget")
