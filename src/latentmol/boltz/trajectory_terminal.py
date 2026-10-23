"""Explicit trajectory termination without promoting partial runs to results.

The legacy contract remains a fully exhausted update budget. An opted-in
``armijo_v1`` trajectory can also terminate at an explicitly recorded stall.
``best_trial_v2`` instead records finite fallback selections and requires the
full requested update budget; an uphill fallback is never a descent proof.
Such a historical stall is a verified terminal point, not proof of convergence or a completed
update budget. Parent completion, frozen-state, gate and candidate identities
remain the calling reader's responsibility.

These readers inspect saved metadata only. They never load a model, change an
optimizer, choose a favorable checkpoint or perform chemistry conversion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from latentmol.optimization import StepControl


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Terminal {name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Terminal {name} must be finite")
    return value


def _armijo_step_report(report, control, *, accepted):
    """Check bounded attempt evidence; this does not rerun its scalar oracle."""
    if not isinstance(report, dict) or report.get("policy") != control.record():
        raise ValueError("Terminal step-control report differs from configured policy")
    status = "accepted" if accepted else "stalled"
    if (report.get("status") != status or report.get("accepted") is not accepted
            or report.get("stalled") is not (not accepted)):
        raise ValueError(f"Terminal step-control report does not prove {status}")
    trials = report.get("trials")
    if not isinstance(trials, list) or not trials or len(trials) > control.max_trials:
        raise ValueError("Terminal step-control report has an incomplete or unbounded trial list")
    scales = control.record()["trial_scales"]
    skipped = {"nonfinite_proposal", "zero_or_nonrepresentable_move", "non_descent_projected_move"}
    evaluated = {"nonfinite_objective", "no_strict_decrease", "decrease_below_numerical_guard",
                 "insufficient_armijo_decrease", "accepted"}
    for index, trial in enumerate(trials):
        if (not isinstance(trial, dict) or type(trial.get("index")) is not int
                or trial["index"] != index or trial.get("scale") != scales[index]
                or type(trial.get("scale")) not in (int, float)
                or type(trial.get("oracle_called")) is not bool
                or type(trial.get("accepted")) is not bool):
            raise ValueError("Terminal step-control trials differ from the declared ordered schedule")
        if (trial.get("reason") not in skipped | evaluated
                or trial["oracle_called"] is not (trial["reason"] in evaluated)
                or trial["accepted"] is not (trial["reason"] == "accepted")):
            raise ValueError("Terminal step-control trial lacks a finished outcome")
    calls = _integer(report.get("oracle_calls_attempted"), "oracle_calls_attempted")
    if calls != sum(trial["oracle_called"] for trial in trials):
        raise ValueError("Terminal step-control oracle count differs from trial evidence")
    completed = _integer(report.get("oracle_calls_completed"), "oracle_calls_completed")
    nonfinite = _integer(report.get("oracle_calls_nonfinite"), "oracle_calls_nonfinite")
    errors = _integer(report.get("oracle_calls_unexpected_errors"), "oracle_calls_unexpected_errors")
    if completed > calls or nonfinite > calls or errors != 0:
        raise ValueError("Terminal step-control report contains failed or inconsistent oracle evidence")
    if nonfinite != sum(trial["reason"] == "nonfinite_objective" for trial in trials):
        raise ValueError("Terminal nonfinite oracle count differs from trial evidence")
    # A nonfinite value may be classified inside the callback or after it
    # returns. Every other recorded oracle outcome requires a completed call.
    if completed < calls - nonfinite:
        raise ValueError("Terminal oracle outcomes lack their completed calls")
    if accepted:
        index = _integer(report.get("accepted_trial"), "accepted_trial")
        if (index != len(trials) - 1 or type(report.get("accepted_scale")) not in (int, float)
                or report.get("accepted_scale") != scales[index]):
            raise ValueError("Terminal accepted trial is not the final recorded attempt")
        if any(trial["accepted"] for trial in trials[:-1]):
            raise ValueError("Terminal step-control report contains multiple accepted attempts")
        trial = trials[index]
        gain = _finite(trial.get("actual_loss_improvement"), "actual_loss_improvement")
        guard = _finite(trial.get("observed_improvement_guard"), "observed_improvement_guard")
        required = _finite(trial.get("required_armijo_improvement"), "required_armijo_improvement")
        recorded_gain = _finite(report.get("actual_loss_improvement"), "recorded_actual_loss_improvement")
        if (trial.get("accepted") is not True or trial.get("oracle_called") is not True
                or trial.get("reason") != "accepted" or gain <= 0 or guard < 0 or required < 0
                or gain < guard or gain < required
                or any(trial.get(key) is not True for key in
                       ("strictly_decreased", "informative_decrease", "armijo_satisfied"))
                or recorded_gain != gain):
            raise ValueError("Terminal accepted update lacks measured sufficient-decrease evidence")
    else:
        recorded_gain = _finite(report.get("actual_loss_improvement"), "recorded_actual_loss_improvement")
        displacement = _finite(report.get("actual_displacement_norm"), "actual_displacement_norm")
        if (len(trials) != control.max_trials or any(trial["accepted"] for trial in trials)
                or report.get("stall_reason") != "trial_budget_exhausted"
                or recorded_gain != 0.0 or displacement != 0.0):
            raise ValueError("Terminal stall lacks its exhausted trial budget and unchanged-point receipt")


def _best_trial_step_report(report, control, *, accepted):
    """Recheck the declared finite-trial selection without asserting descent.

    A nonincreasing full-size trial terminates the search immediately. After
    a full-size increase, the first strict improving half-step terminates it.
    Otherwise all declared scales must have been attempted and
    the lowest observed finite loss is selected, with the earliest tie winning.
    An unchanged representable point remains a recorded, oracle-scored fallback.
    """
    if (not accepted or not isinstance(report, dict)
            or report.get("policy") != control.record()
            or report.get("status") != "accepted" or report.get("accepted") is not True
            or report.get("stalled") is not False):
        raise ValueError("Terminal best-trial report does not prove accepted full-budget progress")
    baseline = _finite(report.get("baseline_loss"), "baseline_loss")
    trials = report.get("trials")
    if not isinstance(trials, list) or not 1 <= len(trials) <= control.max_trials:
        raise ValueError("Terminal best-trial report has an incomplete or unbounded trial list")
    scales = control.record()["trial_scales"]
    finite = []
    calls = nonfinite = 0
    for index, trial in enumerate(trials):
        if (not isinstance(trial, dict) or type(trial.get("index")) is not int
                or trial["index"] != index or type(trial.get("scale")) not in (int, float)
                or trial["scale"] != scales[index] or type(trial.get("oracle_called")) is not bool
                or type(trial.get("accepted")) is not bool
                or type(trial.get("finite_proposal")) is not bool
                or type(trial.get("finite_candidate")) is not bool):
            raise ValueError("Terminal best-trial trials differ from the declared ordered schedule")
        reason = trial.get("reason")
        if reason == "nonfinite_proposal":
            if (trial["oracle_called"] or trial["accepted"]
                    or trial["finite_proposal"] and trial["finite_candidate"]):
                raise ValueError("Terminal nonfinite proposal has inconsistent oracle evidence")
            continue
        if reason not in ("nonfinite_objective", "no_strict_decrease", "accepted") or not trial["oracle_called"]:
            raise ValueError("Terminal best-trial attempt lacks a finished oracle outcome")
        if not trial["finite_proposal"] or not trial["finite_candidate"]:
            raise ValueError("Terminal best-trial oracle was called on a nonfinite proposal")
        calls += 1
        if reason == "nonfinite_objective":
            nonfinite += 1
            if trial["accepted"]:
                raise ValueError("Terminal best-trial selection is nonfinite")
            continue
        loss = _finite(trial.get("candidate_loss"), "candidate_loss")
        gain = _finite(trial.get("actual_loss_improvement"), "actual_loss_improvement")
        displacement = _finite(trial.get("actual_displacement_norm"), "actual_displacement_norm")
        zero = trial.get("zero_displacement")
        if (type(zero) is not bool or displacement < 0 or zero is not (displacement == 0)
                or gain != baseline - loss or trial.get("strictly_decreased") is not (gain > 0)
                or trial["accepted"] is not (reason == "accepted")):
            raise ValueError("Terminal best-trial finite outcome disagrees with measured loss/displacement")
        finite.append((index, loss, gain, zero))
    if not finite:
        raise ValueError("Terminal best-trial report lacks any finite candidate")
    attempted = _integer(report.get("oracle_calls_attempted"), "oracle_calls_attempted")
    completed = _integer(report.get("oracle_calls_completed"), "oracle_calls_completed")
    observed_nonfinite = _integer(report.get("oracle_calls_nonfinite"), "oracle_calls_nonfinite")
    errors = _integer(report.get("oracle_calls_unexpected_errors"), "oracle_calls_unexpected_errors")
    if (attempted != calls or observed_nonfinite != nonfinite or errors != 0
            or not calls - nonfinite <= completed <= calls):
        raise ValueError("Terminal best-trial oracle counts differ from completed trial evidence")
    selected = _integer(report.get("accepted_trial"), "accepted_trial")
    selected_rows = [row for row in finite if row[0] == selected]
    if (len(selected_rows) != 1 or type(report.get("accepted_scale")) not in (int, float)
            or report.get("accepted_scale") != scales[selected]
            or sum(trial["accepted"] for trial in trials) != 1
            or trials[selected]["accepted"] is not True):
        raise ValueError("Terminal best-trial selection does not identify exactly one finite trial")
    _, _, gain, zero = selected_rows[0]
    improving = [row for row in finite if row[2] > 0 and not row[3]]
    kind = report.get("acceptance_kind")
    full_nonincrease = finite[0][0] == 0 and finite[0][2] >= 0
    if full_nonincrease and (selected != 0 or len(trials) != 1):
        raise ValueError("Terminal search continued after a nonincreasing full-size trial")
    if kind == "full_step_nonincrease":
        if selected != 0 or len(trials) != 1 or gain < 0 or gain > 0 and not zero:
            raise ValueError("Terminal nonincrease selection is not the immediate full-size trial")
    elif kind == "first_improvement":
        if not improving or selected != improving[0][0] or selected != len(trials) - 1:
            raise ValueError("Terminal first improvement did not immediately stop the ordered search")
    elif kind == "best_finite_fallback":
        if (improving or len(trials) != control.max_trials
                or selected != min(finite, key=lambda row: (row[1], row[0]))[0]):
            raise ValueError("Terminal fallback is not the lowest finite loss after all declared trials")
    else:
        raise ValueError("Terminal best-trial selection lacks its explicit acceptance kind")
    if (report.get("monotone_descent") is not (gain > 0 and not zero)
            or report.get("no_op") is not zero
            or _finite(report.get("selected_loss"), "selected_loss") != trials[selected]["candidate_loss"]
            or _finite(report.get("actual_loss_improvement"), "recorded_actual_loss_improvement") != gain
            or _finite(report.get("actual_displacement_norm"), "recorded_actual_displacement_norm")
                != trials[selected]["actual_displacement_norm"]):
        raise ValueError("Terminal best-trial selected loss/descent/displacement receipt is inconsistent")


def validate_step_report(report, control, *, accepted=True):
    """Dispatch the recorded policy; never reinterpret a v1 stall as v2 success."""
    if control.version == "armijo_v1":
        _armijo_step_report(report, control, accepted=accepted)
    elif control.version == "best_trial_v2":
        _best_trial_step_report(report, control, accepted=accepted)
    else:
        raise ValueError("Unknown terminal step-control policy")


def validate_arm_terminal(config: dict, summary: dict, *, metric_rows=None,
                          step_reports=None) -> dict:
    """Return an explicit terminal receipt after checking the declared boundary.

    Legacy runs require their exact full budget, as before. Opted-in step
    control requires every metric and attempt report through the actual
    endpoint. No missing intermediate rows, unrecorded stall or failed attempt
    is treated as a terminal trajectory. This function does not authenticate
    parent files; callers must retain their existing integrity checks.
    """
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError("Terminal config and arm summary must be mappings")
    requested = _integer(config.get("optimizer", {}).get("steps"), "requested_updates", minimum=1)
    endpoint = summary.get("endpoint")
    if not isinstance(endpoint, dict):
        raise ValueError("Terminal arm lacks a saved endpoint")
    endpoint_step = _integer(endpoint.get("step"), "endpoint_step")
    steps = _integer(summary.get("steps"), "steps")
    policy = config.get("step_control")
    if policy is None:
        if (steps != requested or endpoint_step != requested or summary.get("stalled") is True
                or summary.get("optimization_status") == "stalled"):
            raise ValueError("Legacy trajectory must complete its full declared update budget")
        accepted, attempted, stalled, kind = requested, requested, False, "completed_budget"
    else:
        control = StepControl.from_mapping(policy)
        if summary.get("step_control") != control.record():
            raise ValueError("Terminal arm step-control policy differs from config")
        accepted = _integer(summary.get("accepted_updates"), "accepted_updates")
        attempted = _integer(summary.get("attempted_updates"), "attempted_updates")
        stalled = summary.get("stalled")
        recorded_requested = _integer(summary.get("requested_steps"), "requested_steps", minimum=1)
        if type(stalled) is not bool or recorded_requested != requested:
            raise ValueError("Terminal arm lacks its explicit requested budget and stall status")
        if (accepted != steps or endpoint_step != accepted or accepted > requested
                or summary.get("completed_requested_budget") is not (not stalled)):
            raise ValueError("Terminal endpoint and accepted updates disagree")
        kind = "stalled" if stalled else "completed_budget"
        if summary.get("optimization_status") != kind:
            raise ValueError("Terminal optimization status is inconsistent")
        if ((stalled and (accepted >= requested or attempted != accepted + 1))
                or (not stalled and (accepted != requested or attempted != accepted))):
            raise ValueError("Terminal accepted/attempted updates do not explain termination")
        if control.version == "best_trial_v2" and stalled:
            raise ValueError("Best-trial policy must complete its full requested update budget")
        if metric_rows is None or step_reports is None:
            raise ValueError("Opted-in terminal requires complete metrics and step-control evidence")
        if (not isinstance(step_reports, dict)
                or any(type(index) is not int for index in step_reports)
                or set(step_reports) != set(range(attempted))):
            raise ValueError("Terminal step-control attempt sequence is incomplete or has extra attempts")
        for index in range(attempted):
            validate_step_report(step_reports[index], control, accepted=index < accepted)
    if metric_rows is not None:
        if (not isinstance(metric_rows, list) or not metric_rows
                or any(not isinstance(row, dict) or type(row.get("step")) is not int for row in metric_rows)
                or [row["step"] for row in metric_rows] != list(range(endpoint_step + 1))
                or metric_rows[-1] != endpoint):
            raise ValueError("Terminal metrics lack the exact contiguous saved endpoint sequence")
        if policy is not None:
            for index, row in enumerate(metric_rows):
                expected = "accepted" if index < accepted else "stalled" if stalled else None
                if row.get("update_status") != expected:
                    raise ValueError("Terminal metric update status differs from step-control evidence")
    return {"version": "terminal_v1", "kind": kind, "requested_updates": requested,
            "accepted_updates": accepted, "attempted_updates": attempted,
            "endpoint_step": endpoint_step, "completed_budget": not stalled,
            "stalled": stalled, "step_control_version": policy.get("version") if policy else None,
            "claim_limit": ("Declared full-budget boundary; finite fallback and no-op updates do not establish descent or convergence."
                if policy is not None and policy.get("version") == "best_trial_v2" else
                "Declared terminal boundary; a stall is not convergence or a completed update budget.")}


def terminal_evidence_paths(folder: Path, config: dict) -> list[Path]:
    """Extra terminal evidence to seal for opted-in runs; legacy adds no files."""
    if config.get("step_control") is None:
        return []
    return sorted(folder.glob("metrics-*.json")) + sorted(folder.glob("step-control-*.json"))


def read_arm_terminal(folder: Path, config: dict, summary: dict) -> dict:
    """Load opt-in boundary evidence without changing historical reader inputs."""
    if config.get("step_control") is None:
        return validate_arm_terminal(config, summary)
    metrics, reports = [], {}
    for path in terminal_evidence_paths(folder, config):
        match = re.fullmatch(r"(metrics|step-control)-([0-9]{4,})\.json", path.name)
        if match is None:
            raise ValueError("Malformed terminal evidence filename")
        index = int(match[2])
        value = json.loads(path.read_text())
        if match[1] == "metrics":
            if not isinstance(value, dict) or value.get("step") != index:
                raise ValueError("Terminal metric filename differs from its saved step")
            metrics.append(value)
        else:
            if index in reports:
                raise ValueError("Duplicate terminal attempt report")
            reports[index] = value
    return validate_arm_terminal(config, summary, metric_rows=metrics, step_reports=reports)
