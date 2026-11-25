"""Versioned local derivative validation, independent of frozen-model mathematics.

The opt-in ``local_v1`` policy addresses coarse secants and poorly conditioned
random projections. Directions are selected using the analytic gradient BEFORE
any finite-difference endpoint is scored. A failed direction is never replaced
after seeing its finite differences. This is a derivative gate, not an optimizer,
a chemistry metric, or permission to reclassify an earlier failed experiment.
"""

from __future__ import annotations

import math
from typing import Callable

import torch

from latentmol.multimodel.study import tensor_digest

POLICY = "local_v1"
EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.03, 0.1)
RANDOM_DIRECTIONS = 3
MAX_PROPOSALS = 64
PROJECTION_FRACTION = 0.5


def policy_record(*, seed: int, rtol: float, atol: float, minimum_passes: int) -> dict:
    """Declare the entire probe policy before any model endpoint evaluation."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Local FD seed must be a nonnegative integer")
    if (type(minimum_passes) is not int or not 2 <= minimum_passes <= len(EPSILONS)
            or not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0):
        raise ValueError("Local FD retains finite tolerances and at least two scales")
    return {"version": POLICY, "seed": seed, "epsilons": list(EPSILONS),
            "fd_rtol": rtol, "fd_atol": atol, "minimum_neighboring_passes": minimum_passes,
            "random_directions": RANDOM_DIRECTIONS, "maximum_gaussian_proposals": MAX_PROPOSALS,
            "projection_fraction": PROJECTION_FRACTION,
            "selection_rule": "abs(g dot d) >= 0.5 * norm(g) / sqrt(latent_numel), before FD scores",
            "legacy_projection_required_for_pass": False,
            "informative_row_rule": "abs(analytic)>fd_atol, expected signal>one output ULP, nonzero observed difference and realized chord",
            "finite_difference_target": "same scalar objective at fixed mask and fixed context",
            "occupancy_scope": "No derivative through the selector or through removed NO_ATOM logits"}


def direction_plan(point: torch.Tensor, gradient: torch.Tensor, *, seed: int,
                   allow_low_signal: bool = False):
    """Select exactly three sensitive Gaussian directions from a bounded pool.

    Proposal zero reproduces the legacy seeded projection (CPU Gaussian draw,
    production-dtype normalization). All draws examined, including rejected low
    projections, are recorded. Conditioning improves signal without changing
    finite-difference tolerances. Accepted directions come from independent
    Gaussian proposals conditioned on the declared analytic-projection event.
    """
    if point.shape != gradient.shape or point.numel() == 0 or not point.is_floating_point():
        raise ValueError("Point and gradient require the same nonempty floating shape")
    if not bool(point.isfinite().all()) or not bool(gradient.isfinite().all()):
        raise ValueError("Nonfinite local FD point or gradient")
    norm = float(gradient.double().norm())
    if math.isfinite(norm) and norm <= 1e-12 and allow_low_signal:
        # Only pointwise-v5 uses this path. No nonzero derivative is required
        # for correctness: probe flat/low-signal functions in four seeded,
        # unconditioned directions chosen before observing forward AD or chords.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        directions, proposals = {}, []
        for index, label in enumerate(("gradient", "random_1", "random_2", "random_3")):
            raw = torch.randn(point.shape, generator=generator).to(point)
            direction = raw / raw.norm()
            directions[label] = direction
            proposals.append(dict(proposal_index=index, generator_seed=seed, selected_label=label,
                direction_sha256=tensor_digest({"direction": direction}),
                analytic_projection_float64=float((gradient.double() * direction.double()).sum())))
        return directions, directions["gradient"], dict(
            selection_completed_before_endpoint_scores=True, latent_numel=point.numel(),
            gradient_norm=norm, selected_labels=list(directions), pool_exhausted=False,
            selection_rule="unconditioned_seeded_low_signal_v1",
            gradient_label_is_random_probe=True, proposals=proposals)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Local FD requires a nonzero analytic gradient")
    threshold = PROJECTION_FRACTION * norm / math.sqrt(point.numel())
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = {"gradient": gradient.detach() / norm}
    receipts = []
    legacy = None
    for index in range(MAX_PROPOSALS):
        raw = torch.randn(point.shape, generator=generator).to(point)
        direction = raw / raw.norm()
        # Widen only the diagnostic dot product; model, direction and endpoint
        # tensors retain their original dtype and ordinary production arithmetic.
        projection = float((gradient.double() * direction.double()).sum())
        accepted = math.isfinite(projection) and abs(projection) >= threshold
        label = f"random_{len(directions)}" if accepted else None
        receipt = {"proposal_index": index, "generator_seed": seed,
                   "direction_sha256": tensor_digest({"direction": direction}),
                   "analytic_projection_float64": projection,
                   "minimum_absolute_projection": threshold, "accepted": accepted,
                   "selected_label": label,
                   "rejection_reason": None if accepted else "analytic_projection_below_prespecified_threshold"}
        receipts.append(receipt)
        if index == 0:
            legacy = direction
        if accepted:
            directions[label] = direction
        if len(directions) == RANDOM_DIRECTIONS + 1:
            break
    metadata = {"selection_completed_before_endpoint_scores": True,
                "latent_numel": point.numel(), "gradient_norm": norm,
                "minimum_absolute_random_projection": threshold, "proposals": receipts,
                "selected_labels": list(directions),
                "pool_exhausted": len(directions) != RANDOM_DIRECTIONS + 1,
                "legacy_direction_sha256": tensor_digest({"direction": legacy})}
    return directions, legacy, metadata


def neighboring_passes(rows: list[dict], *, minimum: int) -> dict:
    """Find consecutive passing entries of the predefined grid, without tuning it."""
    by_epsilon = {row["epsilon"]: row for row in rows}
    runs, current = [], []
    for epsilon in EPSILONS:
        if by_epsilon.get(epsilon, {}).get("passed", False):
            current.append(epsilon)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)
    return {"passed": any(len(run) >= minimum for run in runs), "passing_neighbor_runs": runs,
            "minimum_neighboring_passes": minimum}


def _ulp(value: float, dtype: str) -> float:
    """Actual scalar spacing; this is not an error bound for a whole network."""
    types = {str(kind): kind for kind in (torch.float16, torch.bfloat16, torch.float32, torch.float64)}
    if dtype not in types:
        raise ValueError("Endpoint must report its actual floating loss dtype")
    x = torch.tensor(value, dtype=types[dtype])
    upper = torch.nextafter(x, torch.full_like(x, float("inf")))
    lower = torch.nextafter(x, torch.full_like(x, float("-inf")))
    return max(float(upper.double() - x.double()), float(x.double() - lower.double()))


def endpoint_row(point, direction, gradient, epsilon, plus, minus, baseline,
                 *, rtol, atol) -> dict:
    """Compare the nominal derivative while disclosing actual representable chords.

    Chord corrections and output ULPs are diagnostic evidence only. They do not
    manufacture a passing row when nominal FD and the analytic direction disagree.
    Low-signal disagreements therefore remain failures at the same tolerance.
    """
    fp, fm = plus["loss"], minus["loss"]
    if not all(math.isfinite(x) for x in (fp, fm, baseline)):
        raise ValueError("Nonfinite scalar in local FD validation")
    analytic = float((gradient * direction).sum())
    numerical = (fp - fm) / (2 * epsilon)
    error = abs(analytic - numerical)
    tolerance = atol + rtol * max(abs(analytic), abs(numerical))
    xp, xm = point + epsilon * direction, point - epsilon * direction
    dp, dm = xp.double() - point.double(), point.double() - xm.double()
    ulp = max(_ulp(fp, plus["loss_dtype"]), _ulp(fm, minus["loss_dtype"]))
    informative = (abs(analytic) > atol and abs(2 * epsilon * analytic) > ulp and fp != fm
                   and bool((dp + dm).norm() > 0))
    return {"epsilon": epsilon, "analytic": analytic, "central_difference": numerical,
            "absolute_error": error, "tolerance": tolerance,
            "passed_tolerance": error <= tolerance, "informative": informative,
            "passed": error <= tolerance and informative,
            "f_plus": fp, "f_minus": fm, "baseline_loss": baseline,
            "forward_slope": (fp - baseline) / epsilon,
            "backward_slope": (baseline - fm) / epsilon,
            "forward_backward_slope_difference": (fp - 2 * baseline + fm) / epsilon,
            "central_second_difference": (fp - 2 * baseline + fm) / epsilon ** 2,
            "loss_dtype_plus": plus["loss_dtype"], "loss_dtype_minus": minus["loss_dtype"],
            "output_ulp": ulp,
            "difference_in_output_ulps": abs(fp - fm) / ulp if ulp else None,
            "expected_signal_in_output_ulps": abs(2 * epsilon * analytic) / ulp if ulp else None,
            "one_output_ulp_derivative_scale": ulp / (2 * epsilon),
            "analytic_dot_float64": float((gradient.double() * direction.double()).sum()),
            "realized_plus_distance": float(dp.norm()), "realized_minus_distance": float(dm.norm()),
            "realized_plus_projection": float((dp * direction.double()).sum()),
            "realized_minus_projection": float((dm * direction.double()).sum()),
            "actual_secant_linear_prediction": float((gradient.double() * (dp + dm)).sum()) / (2 * epsilon),
            "natural_selector_crossed": bool(plus.get("natural_selector_crossed", False)
                                              or minus.get("natural_selector_crossed", False))}


def validate_local_fd(point, gradient, score: Callable, *, baseline: float,
                      seed: int, rtol: float, atol: float, minimum_passes: int,
                      report: dict | None = None, persist: Callable = lambda: None) -> dict:
    """Score a fully declared plan, retaining every local and coarse failure.

    ``score`` receives one perturbed point and returns loss/dtype plus an optional
    natural-selector warning. It must keep the same mask, graph policy, objective,
    native context and frozen model. The caller owns those invariants and parity.
    """
    report = report if report is not None else {}
    report.update(policy=policy_record(seed=seed, rtol=rtol, atol=atol, minimum_passes=minimum_passes),
                  passed=False, status="selecting_directions", finite_differences=[],
                  legacy_projection={"required_for_pass": False, "finite_differences": []})
    directions, legacy, plan = direction_plan(point, gradient, seed=seed)
    report["direction_selection"] = plan
    report["status"] = "scoring" if not plan["pool_exhausted"] else "failed_direction_pool"
    persist()
    if plan["pool_exhausted"]:
        return report
    # The legacy direction may be an accepted proposal. Reuse its endpoints
    # instead of scoring identical tensors twice; it remains separately reported.
    legacy_digest = plan["legacy_direction_sha256"]
    legacy_label = next((row["selected_label"] for row in plan["proposals"]
                         if row["accepted"] and row["direction_sha256"] == legacy_digest), None)
    all_directions = dict(directions)
    if legacy_label is None:
        all_directions["legacy_random"] = legacy
    for label, direction in all_directions.items():
        destination = (report["legacy_projection"]["finite_differences"] if label == "legacy_random"
                       else report["finite_differences"])
        for epsilon in EPSILONS:
            plus = score(point + epsilon * direction)
            minus = score(point - epsilon * direction)
            row = endpoint_row(point, direction, gradient, epsilon, plus, minus, baseline,
                               rtol=rtol, atol=atol)
            if "contact_selection" in plus or "contact_selection" in minus:
                row["contact_selection_plus"] = plus.get("contact_selection")
                row["contact_selection_minus"] = minus.get("contact_selection")
                # Compare winners, rather than floating tie margins.
                keys = ("best_ligand_token_per_pocket", "anchor_tokens")
                row["contact_winners_changed"] = any(
                    plus.get("contact_selection", {}).get(k) != minus.get("contact_selection", {}).get(k)
                    for k in keys)
            row["direction"] = label
            destination.append(row)
            persist()
    if legacy_label is not None:
        report["legacy_projection"]["reused_selected_direction"] = legacy_label
        report["legacy_projection"]["finite_differences"] = [
            {**row, "direction": "legacy_random"} for row in report["finite_differences"]
            if row["direction"] == legacy_label]
    report["direction_results"] = {
        label: neighboring_passes([row for row in report["finite_differences"]
                                   if row["direction"] == label], minimum=minimum_passes)
        for label in directions}
    report["legacy_projection"].update(neighboring_passes(
        report["legacy_projection"]["finite_differences"], minimum=minimum_passes))
    report["passed"] = all(row["passed"] for row in report["direction_results"].values())
    report["status"] = "passed" if report["passed"] else "failed"
    persist()
    return report
