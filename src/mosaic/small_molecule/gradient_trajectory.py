"""Projected updates shared by native gradient checks and sealed continuations.

The default strict policy is unchanged. ``best_primal`` is an explicitly
exploratory policy: after exhausting the strict search it takes the finite
candidate with lowest independently measured primal loss, even if that loss
increases. Nonfinite evaluations still fail immediately.
"""

import numpy as np

from .io import write_json
from .optimizers import evaluate_graph, projected_step


def run_updates(
    loss,
    evaluation,
    *,
    key,
    directory,
    report,
    total_steps,
    initial_step_size,
    max_backtracks,
    current_primal,
    compact,
    record_readouts,
    primal_every_step,
    acceptance_policy="strict",
    start_step=0,
):
    # Imported at call time: gradient_checks owns the common artifact codec,
    # while this module owns only update selection and trajectory recording.
    from .gradient_checks import _save_evaluation, _snapshot, _value

    if acceptance_policy not in ("strict", "best_primal"):
        raise ValueError("Unknown acceptance policy")
    if acceptance_policy == "best_primal" and not primal_every_step:
        raise ValueError("Best-primal fallback requires independent primal evaluation")
    for step in range(start_step, total_steps):
        record = {
            "step": step,
            "before": float(evaluation.value),
            "primal_before": current_primal,
            "attempts": [],
        }
        report["updates"].append(record)
        best = None
        selected = None
        for backtrack in range(max_backtracks):
            rate = initial_step_size * 0.5**backtrack
            proposal = projected_step(evaluation, eta_X=rate, eta_E=rate)
            candidate = evaluate_graph(loss, proposal.next_graph, key)
            _save_evaluation(
                directory / f"step-{step}-trial-{backtrack}.npz",
                candidate,
                compact=compact,
                logits=not compact,
            )
            after = float(candidate.value)
            primal = None
            if primal_every_step:
                primal = _value(
                    loss,
                    candidate.graph,
                    key,
                    directory / f"primal-step-{step}-trial-{backtrack}.npz",
                    compact=compact,
                    logits=not compact,
                )
            if not np.isfinite(after) or (
                primal_every_step and not np.isfinite(primal)
            ):
                raise FloatingPointError(
                    "Nonfinite candidate cannot enter fallback selection"
                )
            strict = after < float(evaluation.value) - 1e-7 and (
                not primal_every_step or primal < current_primal - 1e-7
            )
            record["attempts"].append(
                {
                    "step_size": rate,
                    "loss": after,
                    "primal_loss": primal,
                    "projection_residuals": proposal.projection_residuals,
                    "passes_strict_threshold": bool(strict),
                }
            )
            trial = (candidate, primal, rate, backtrack)
            if acceptance_policy == "best_primal" and (
                best is None or primal < best[1]
            ):
                best = trial  # Deterministic ties retain the first candidate.
            if strict:
                selected = trial
                break
        fallback = selected is None
        if fallback:
            if acceptance_policy == "strict":
                raise AssertionError("No measured loss-decreasing projected step")
            selected = best
        candidate, primal, rate, trial_index = selected
        record.update(
            after=float(candidate.value),
            accepted_step_size=rate,
            selected_trial=trial_index,
            fallback_used=fallback,
            acceptance_policy=acceptance_policy,
            delta_X=float(
                np.linalg.norm(np.asarray(candidate.graph.X - evaluation.graph.X))
            ),
            delta_E_upper=float(
                np.linalg.norm(
                    np.asarray(candidate.graph.E_upper - evaluation.graph.E_upper)
                )
            ),
        )
        evaluation = candidate
        if primal_every_step:
            record.update(primal_after=primal, primal_increased=primal > current_primal)
            current_primal = primal
        elif record_readouts:
            current_primal = _value(
                loss,
                evaluation.graph,
                key,
                directory / f"primal-step-{step}-accepted.npz",
                compact=compact,
            )
        if record_readouts:
            report["states"].append(
                _snapshot(directory, step + 1, evaluation, current_primal)
            )
        # Snapshot arrays and scalar log lines are saved every update. Avoid
        # rewriting a growing multi-megabyte JSON on every exploratory step;
        # callers also flush the full record in finally on any normal failure.
        if (
            acceptance_policy == "strict"
            or (step + 1) % 25 == 0
            or step + 1 == total_steps
        ):
            write_json(directory / "report.json", report)
        print(
            f"accepted update {step + 1}/{total_steps}: loss={float(evaluation.value):.9g}; fallback={fallback}",
            flush=True,
        )
    return evaluation, current_primal
