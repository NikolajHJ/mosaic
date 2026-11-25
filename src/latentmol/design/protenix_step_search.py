"""Score a fixed-target normalized descent ray alongside bounded latent SGD."""

from dataclasses import replace
import math

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import evaluate_point, hard_from_graph


def matched_designs(design, search):
    """One center and each ray sample cost one native VJP call."""
    count, radii, floor = search["iterations"], search["radii"], search["minimum_improvement"]
    if type(count) is not int or count < 1 or not radii:
        raise ValueError("Require a positive bounded search budget")
    if (any(type(r) not in (int, float) or not math.isfinite(r) or not 0 < r <= 4 for r in radii)
            or sorted(set(radii)) != radii):
        raise ValueError("Search radii must be increasing, unique and in (0, 4]")
    if type(floor) not in (int, float) or not math.isfinite(floor) or floor <= 0:
        raise ValueError("Require a positive objective resolution floor")
    controls = count * (len(radii) + 1)
    if controls > design.schedule_horizon:
        raise ValueError("Matched scorer budget exceeds the base optimizer horizon")
    return {
        "current_sgd": replace(design, updates=controls, snapshot_interval=controls, checkpoint_interval=controls),
        "normalized_search": replace(design, updates=count, snapshot_interval=count, checkpoint_interval=count),
    }


def objective(result):
    loss = float(result["binding"]) + result["schedule"]["chemistry_weight"] * float(result["chemistry"])
    if not math.isfinite(loss):
        raise FloatingPointError("Nonfinite search objective")
    return loss


def _hard_record(graph, spec):
    values = {name: getattr(graph, name)[0].detach().cpu() for name in ("atoms", "charges", "bonds")}
    observation = inspect_graph(*values.values(), spec)
    return dict(valid=observation.smiles is not None, smiles=observation.smiles,
                failure=observation.failure_reason, atoms=int(values["atoms"].count_nonzero()),
                labels={name: value.tolist() for name, value in values.items()})


def normalized_update(model, scorer, center, active, design, step, *, radii, minimum_improvement,
                      keep_zero_gradient_budget=False):
    """Choose the lowest improving valid fixed-slot trial under the center target."""
    point = center["point"]
    baseline = objective(center)
    gradient = center["gradient"].double()
    norm = float(gradient.norm())
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite gradient norm in normalized search")
    rows, best = [], None
    if norm or keep_zero_gradient_budget:
        # On matched-start experiments, a zero direction consumes repeat calls
        # at the center, so a saturated arm cannot redistribute its scorer budget.
        direction = (gradient / norm).to(point) if norm else torch.zeros_like(point)
        for radius in radii:
            proposal = point - radius * direction
            trial = evaluate_point(model, scorer, proposal, active, design, step,
                                   target_bundle=center["target_bundle"])
            loss = objective(trial)
            fixed = _hard_record(hard_from_graph(trial["graph"]), model.spec)
            with torch.no_grad():
                raw = _hard_record(model.decode(proposal).harden(model.spec), model.spec)
            change = loss - baseline
            improving = bool(norm) and change <= -minimum_improvement
            row = dict(radius=radius, loss=loss, change=change,
                       binding_loss=float(trial["binding"]), chemistry_loss=float(trial["chemistry"]),
                       objective_resolution_floor=minimum_improvement, improving=improving,
                       acceptable=improving and fixed["valid"], selected=False,
                       displacement=float((proposal.double() - point.double()).norm()),
                       fixed=fixed, unconditioned=raw, latent=proposal.detach().cpu().flatten().tolist())
            rows.append(row)
            if row["acceptable"] and (best is None or loss < best[0]["loss"]):
                best = (row, proposal)
    if best is not None:
        best[0]["selected"] = True
    selected = best[1] if best else point
    return selected.detach(), dict(
        policy="normalized_combined_gradient_search", gradient_norm=norm,
        candidate_calls=len(rows), trials=rows, baseline_loss=baseline,
        target_sha256=center["repair_detail"]["target_sha256"], temperature=center["schedule"]["temperature"],
        chemistry_weight=center["schedule"]["chemistry_weight"], noise_norm=0.,
        accepted=best is not None, bound_factor=1.,
        disposition=("accepted" if best else "no_valid_resolved_decrease" if norm else "zero_gradient"),
        displacement_norm=float((selected.double() - point.double()).norm()),
    )


def scored_observations(centers, unconditioned, updates):
    """Keep every paid evaluation, identifying probes that became committed steps."""
    by_step = {row["step"]: row.get("trials", []) for row in updates}
    observations = []
    for fixed, raw in zip(centers, unconditioned, strict=True):
        step = fixed["step"]
        if step != raw["step"]:
            raise ValueError("Saved fixed and unconditioned steps differ")
        observations.append(dict(step=len(observations), trajectory_step=step, radius=None,
                                 selected=True, kind="center", fixed=fixed, unconditioned=raw))
        for trial in by_step.get(step, ()):
            observations.append(dict(step=len(observations), trajectory_step=step, radius=trial["radius"],
                selected=trial["selected"], kind="search_trial", fixed=trial["fixed"],
                unconditioned=trial["unconditioned"]))
    return observations
