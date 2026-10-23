"""Stage 2 fixed-pose Boltz pilot for selected AE/VAE latent spaces.

This is an engineering screen on one prepared complex. New hard molecules must
later receive native ligand featurization and fresh poses before binding claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from latentmol.chemistry import inspect_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.latent_priors import LatentReference
from latentmol.representation import HardGraphBatch
from .adapter import adapt_graph, fixed_occupancy_graph
from .contracts import PreparedComplex
from .model import FrozenBoltzAffinity, load_affinity_model
from .smoke import (context_digest, finite_difference_gate, parameter_digest,
                    persist_trajectory, score_discrete_projection)


class CountingScorer:
    """Count every fixed-context scorer call, including calibration and controls."""

    def __init__(self, scorer, path: Path):
        self.scorer = scorer
        self.path = path
        self.forward_calls = 0
        self.backward_calls = 0
        self.purpose = "unclassified"
        self.by_purpose = {}
        self._save()

    def _counts(self):
        return self.by_purpose.setdefault(self.purpose, {
            "forward_attempted": 0, "forward_completed": 0,
            "backward_attempted": 0, "backward_completed": 0})

    def _save(self):
        write_json(self.path, {"forward_completed": self.forward_calls,
                               "backward_completed": self.backward_calls,
                               "by_purpose": self.by_purpose})

    def __call__(self, *args, **kwargs):
        counts = self._counts()
        counts["forward_attempted"] += 1
        self._save()
        result = self.scorer(*args, **kwargs)
        self.forward_calls += 1
        counts["forward_completed"] += 1
        self._save()
        return result

    def gradient(self, objective, point):
        counts = self._counts()
        counts["backward_attempted"] += 1
        self._save()
        gradient, = torch.autograd.grad(objective, point)
        self.backward_calls += 1
        counts["backward_completed"] += 1
        self._save()
        return gradient


def _direction(dimension: int, seed: int, device: str):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    value = torch.randn(1, dimension, generator=generator).to(device)
    return value / value.norm().clamp_min(1e-12)


def _trust_value_gradient(point, anchor, radius: float):
    displacement = point - anchor
    distance = displacement.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    excess = (distance - radius).clamp_min(0)
    value = excess.square().mean()
    gradient = 2.0 * excess * displacement / distance
    return value, gradient


def _gradient_cosine(left, right):
    a, b = float(left.norm()), float(right.norm())
    return float((left * right).sum() / (a * b)) if a > 1e-12 and b > 1e-12 else None


def _hard_projection(graph, context):
    hard = HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1),
                          graph.bonds.argmax(-1))
    hard_cpu = hard.to("cpu")
    inspected = inspect_graph(hard_cpu.atoms[0], hard_cpu.charges[0],
                              hard_cpu.bonds[0], context.spec)
    return {"smiles": inspected.smiles, "failure": inspected.failure_reason,
            "heavy_atoms": int((hard_cpu.atoms[0] != 0).sum())}, hard_cpu


def _score_state(position, model, reference, context, scorer, *, require_gradient, purpose):
    point = position.detach().clone().requires_grad_(require_gradient)
    decoded = model.decode(reference.denormalize(point))
    graph = fixed_occupancy_graph(decoded, context.active, context.spec)
    features, types = adapt_graph(graph, context)
    scorer.purpose = purpose
    if require_gradient:
        scores = scorer(features, types, context.coordinates)
        gradient = scorer.gradient(-scores["affinity_probability_binary"].mean(), point)
    else:
        with torch.no_grad():
            scores = scorer(features, types, context.coordinates)
        gradient = None
    with torch.no_grad():
        hard_summary, hard = _hard_projection(graph, context)
        probabilities = {name: getattr(graph, name).detach().cpu().clone()
                         for name in ("atoms", "charges", "bonds")}
        ordinary_expected_size = float(decoded.probabilities().expected_size[0].detach())
        report = {"binary_probability": float(scores["affinity_probability_binary"].mean().detach()),
                  "affinity_pred_value": float(scores["affinity_pred_value"].mean().detach()),
                  "hard": hard_summary, "ordinary_expected_atoms": ordinary_expected_size}
    state = {"normalized_latent": position.detach().cpu().clone(),
             "soft_fixed_occupancy_graph": probabilities,
             "hard_fixed_occupancy_graph": {name: getattr(hard, name).clone()
                                            for name in ("atoms", "charges", "bonds")}}
    return report, state, gradient


def _hard_score(position, model, reference, context, scorer, *, purpose):
    with torch.no_grad():
        graph = fixed_occupancy_graph(model.decode(reference.denormalize(position)),
                                      context.active, context.spec)
        scorer.purpose = purpose
        return score_discrete_projection(graph, context, scorer)[0]


def _cohort_starts(anchor, *, repeat, count=8):
    if count != 8:
        raise ValueError("Pilot protocol fixes eight starts per repeat")
    starts = []
    for index in range(count):
        radius = 2.0 if index < 4 else 4.0
        direction = _direction(anchor.shape[-1], 23000 + 100 * repeat + index, str(anchor.device))
        starts.append({"repeat": repeat, "start_index": index, "radius": radius,
                       "point": (anchor + radius * direction).detach()})
    return starts


def _calibrate_weight(model, reference, context, scorer, anchor, *, trust_radius=3.5):
    rows = []
    for index in range(4):
        point = anchor + 4.0 * _direction(anchor.shape[-1], 19000 + index, str(anchor.device))
        report, _, bind_grad = _score_state(point, model, reference, context, scorer,
                                            require_gradient=True, purpose="weight_calibration")
        _, trust_grad = _trust_value_gradient(point, anchor, trust_radius)
        ratio = float(bind_grad.norm() / trust_grad.norm().clamp_min(1e-12))
        rows.append({"binary_probability": report["binary_probability"],
                     "binding_gradient_norm": float(bind_grad.norm()),
                     "trust_gradient_norm": float(trust_grad.norm()), "ratio": ratio})
    if any(not math.isfinite(row["binding_gradient_norm"]) or row["binding_gradient_norm"] <= 1e-12
           for row in rows):
        raise ValueError("Binary Boltz objective has no finite nonzero latent gradient")
    ratios = torch.tensor([row["ratio"] for row in rows])
    weight = float(ratios.median().clamp(1e-4, 1e4))
    return weight, rows


def _binary_gradient_gate(model, reference, context, scorer, anchor, run, *, at_anchor=False):
    """Check the exact binary objective used by the pilot at multiple FD scales."""
    point = anchor if at_anchor else anchor + 4.0 * _direction(anchor.shape[-1], 18888,
                                                             str(anchor.device))
    report, _, gradient = _score_state(point, model, reference, context, scorer,
                                        require_gradient=True, purpose="binary_fd_analytic")
    if not torch.isfinite(gradient).all() or float(gradient.norm()) <= 1e-12:
        raise ValueError("Binary objective has no finite nonzero analytic gradient")
    direction = gradient / gradient.norm()
    analytic = float((gradient * direction).sum())
    rows = []
    for epsilon in (0.0001, 0.0003, 0.001, 0.003, 0.01):
        values = []
        for sign in (+1, -1):
            measured, _, _ = _score_state(point + sign * epsilon * direction,
                                           model, reference, context, scorer,
                                           require_gradient=False, purpose="binary_fd_forward")
            values.append(-measured["binary_probability"])
        numerical = (values[0] - values[1]) / (2 * epsilon)
        relative = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-12)
        row = {"epsilon": epsilon, "analytic": analytic,
               "central_difference": numerical, "relative_error": relative}
        rows.append(row)
        run.log({"stage": "binary_objective_directional_fd", **row})
    gate = finite_difference_gate(rows, 0.02, 2)
    write_json(run.path / "binary_gradient_gate.json",
               {"point_binary_probability": report["binary_probability"],
                "at_encoded_anchor": at_anchor, "rows": rows, "gate": gate})
    if not gate["passed"]:
        raise AssertionError("Binary Boltz objective failed multi-scale finite differences")
    return gate


def _trajectory(start, anchor, arm, model, reference, context, scorer, *,
                weight, run, state_path, repeat, start_index, start_radius,
                steps=10, step_size=0.25, trust_radius=3.5, max_move=2.5):
    point = start.detach().clone()
    states, records = [], []
    before_forward, before_backward = scorer.forward_calls, scorer.backward_calls
    for step in range(steps + 1):
        report, state, bind_grad = _score_state(point, model, reference, context, scorer,
                                                require_gradient=arm in ("target", "joint") and step < steps,
                                                purpose=f"{arm}_relaxed")
        trust_value, trust_grad = _trust_value_gradient(point, anchor, trust_radius)
        gradient = (bind_grad if bind_grad is not None else torch.zeros_like(point))
        if arm in ("trust", "joint"):
            gradient = gradient + weight * trust_grad
        row = {"step": step, "arm": arm, **report,
               "normalized_distance_from_start": float((point - start).norm()),
               "normalized_distance_from_anchor": float((point - anchor).norm()),
               "trust_loss": float(trust_value),
               "binding_gradient_norm": float(bind_grad.norm()) if bind_grad is not None else None,
               "weighted_trust_gradient_norm": float(weight * trust_grad.norm()),
               "gradient_cosine": _gradient_cosine(bind_grad, trust_grad) if bind_grad is not None else None}
        if step in (0, steps):
            row["hard_projection_score"] = _hard_score(
                point, model, reference, context, scorer, purpose=f"{arm}_hard_projection")
        row["cumulative_scorer_forwards"] = scorer.forward_calls
        row["cumulative_scorer_backwards"] = scorer.backward_calls
        states.append(state)
        records.append(row)
        persist_trajectory(state_path, {"format_version": 1, "records": records, "states": states,
                                        "repeat": repeat, "start_index": start_index,
                                        "start_radius": start_radius, "arm": arm})
        run.log({"stage": "trajectory", "repeat": repeat, "start_index": start_index,
                 "start_radius": start_radius, **row})
        if step < steps:
            norm = gradient.norm()
            if not torch.isfinite(norm):
                raise ValueError("Combined Boltz/trust latent gradient is nonfinite")
            if norm > 1e-12:
                proposed = point - step_size * gradient / norm
                displacement = proposed - start
                scale = min(1.0, max_move / max(float(displacement.norm()), 1e-12))
                point = (start + displacement * scale).detach()
    calls = {"forward": scorer.forward_calls - before_forward,
             "backward": scorer.backward_calls - before_backward}
    return point, records, calls


def _random_control(start, target_end, anchor, model, reference, context, scorer, *, seed):
    displacement = float((target_end - start).norm())
    direction = _direction(start.shape[-1], seed, str(start.device))
    point = (start + displacement * direction).detach()
    report, state, _ = _score_state(point, model, reference, context, scorer,
                                   require_gradient=False, purpose="random_relaxed")
    report.update(step=10, arm="random_matched_displacement",
                  normalized_distance_from_start=displacement,
                  normalized_distance_from_anchor=float((point - anchor).norm()),
                  hard_projection_score=_hard_score(point, model, reference, context, scorer,
                                                    purpose="random_hard_projection"))
    return report, state


def _aggregate(rows, baseline):
    result = {}
    radii = sorted({row["start_radius"] for row in rows})
    for arm in ("target", "trust", "joint", "random_matched_displacement"):
        subset = [row for row in rows if row["arm"] == arm]
        if not subset:
            continue
        changed = [row for row in subset if row["hard"]["smiles"]
                   and baseline["hard"]["smiles"] and row["hard"]["smiles"] != baseline["hard"]["smiles"]]
        optimized_changed = [row for row in changed if row["start_hard"]["smiles"]
                             and row["hard"]["smiles"] != row["start_hard"]["smiles"]]
        improved = [row for row in optimized_changed if row["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
                    is not None and baseline["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
                    is not None and row["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
                    > baseline["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"] + 1e-4
                    and row["start_hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
                    is not None and row["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
                    > row["start_hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"] + 1e-4]
        result[arm] = {"attempts": len(subset), "valid_hard": sum(bool(row["hard"]["smiles"]) for row in subset),
                       "valid_changed_from_encoded_baseline": len(changed),
                       "valid_changed_from_start_and_baseline": len(optimized_changed),
                       "valid_changed_fixed_projection_improved_over_start_and_baseline": len(improved),
                       "unique_changed_hard": len({row["hard"]["smiles"] for row in optimized_changed}),
                       "best_relaxed_binary": max((row["binary_probability"] for row in subset), default=None),
                       "by_start_radius": {
                           str(radius): {"attempts": len(group),
                                         "valid_changed_from_start": sum(bool(row["hard"]["smiles"])
                                             and bool(row["start_hard"]["smiles"])
                                             and row["hard"]["smiles"] != row["start_hard"]["smiles"]
                                             for row in group)}
                           for radius in radii
                           for group in [[row for row in subset if row["start_radius"] == radius]]}}
    return result


def run_pilot(prepared: Path, affinity_checkpoint: Path, checkpoint: Path,
              reference_path: Path, data_manifest: Path, run_root: Path, *, mode: str,
              device: str = "cuda", start_policy: str = "perturbed_cohort"):
    if start_policy not in ("perturbed_cohort", "carbon_anchor"):
        raise ValueError("Unknown latent pilot start policy")
    context = PreparedComplex.load(prepared)
    model, training = load_model(checkpoint, "cpu", expected_spec=context.spec)
    if training["train_config"]["mode"] != mode:
        raise ValueError("AE/VAE mode and selected checkpoint disagree")
    manifest = json.loads(data_manifest.read_text())
    if manifest["dataset_sha256"] != training["dataset_sha256"]:
        raise ValueError("Checkpoint and data manifest differ")
    raw_reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    meta = raw_reference["metadata"]
    expected = {"checkpoint_sha256": file_sha256(checkpoint),
                "dataset_sha256": manifest["dataset_sha256"],
                "split_sha256": manifest["split_sha256"],
                "graph_spec": context.spec.fingerprint(), "mode": mode}
    if any(meta.get(key) != value for key, value in expected.items()) or meta.get("training_count") != 215000:
        raise ValueError("Training-code reference does not match selected model and split")
    reference = LatentReference.from_artifact(raw_reference, meta).to(device)
    if start_policy == "carbon_anchor":
        seed_result = inspect_graph(context.seed.atoms[0], context.seed.charges[0],
                                    context.seed.bonds[0], context.spec)
        if seed_result.smiles != "CCCCCCCCCCCCC":
            raise ValueError("Carbon-anchor pilot requires an exact native C13 seed")
    arms = ("target", "trust", "joint") if start_policy == "perturbed_cohort" else ("target",)
    starts_per_repeat = 8 if start_policy == "perturbed_cohort" else 1
    repeats = 2 if start_policy == "perturbed_cohort" else 1
    config = {"prepared": str(prepared), "prepared_sha256": file_sha256(prepared),
              "affinity_checkpoint_sha256": file_sha256(affinity_checkpoint),
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "reference": str(reference_path), "reference_sha256": file_sha256(reference_path),
              "dataset_sha256": manifest["dataset_sha256"], "split_sha256": manifest["split_sha256"],
              "mode": mode, "device": device, "context": "one fixed pose, occupancy and ligand map",
              "start_policy": start_policy, "starts_per_repeat": starts_per_repeat,
              "repeats": repeats, "steps": 10, "step_size": 0.25,
              "max_normalized_move": 2.5, "trust_radius": 3.5, "recycling_steps": 0,
              "objective": "maximize binary binder probability; trust arm only in perturbed cohort",
              "arms": [*arms, "random_matched_displacement"],
              "oracle_budget": f"{starts_per_repeat * repeats} start per arm, 10 updates; "
                               "all forward/backward attempts tracked; wall time measured by run"}
    if start_policy == "carbon_anchor":
        config["initial_relaxed_state"] = (
            "decoder probabilities at encoded C13 mean; exact hard argmax graph is C13; "
            "not a one-hot all-carbon graph")
    name = f"{mode}-boltz-carbon-anchor-pilot" if start_policy == "carbon_anchor" else f"{mode}-boltz-latent-pilot"
    with ExperimentRun(run_root, name, config, Path.cwd()) as run:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.to(device).eval().requires_grad_(False)
        context = context.to(device)
        boltz = load_affinity_model(affinity_checkpoint).to(device)
        scorer = CountingScorer(FrozenBoltzAffinity(boltz, 0), run.path / "oracle_cost.json")
        initial_parameters = parameter_digest(model, boltz)
        initial_context = context_digest(context)
        with torch.no_grad():
            mean, _ = model.encode(context.seed.to_relaxed(context.spec))
            anchor = reference.normalize(mean).detach()
            if start_policy == "carbon_anchor":
                decoded_seed = model.decode(mean).harden(context.spec)
                if any(not torch.equal(getattr(decoded_seed, field).cpu(),
                                       getattr(context.seed, field).cpu())
                       for field in ("atoms", "charges", "bonds")):
                    raise ValueError("Encoded C13 anchor does not decode to the exact native seed graph")
        baseline_report, baseline_state, _ = _score_state(anchor, model, reference, context,
                                                          scorer, require_gradient=False,
                                                          purpose="encoded_baseline")
        baseline_report["hard_projection_score"] = _hard_score(
            anchor, model, reference, context, scorer, purpose="encoded_baseline_hard_projection")
        write_json(run.path / "baseline.json", baseline_report)
        torch.save(baseline_state, run.path / "baseline_state.pt")
        binary_gradient_gate = _binary_gradient_gate(model, reference, context,
                                                     scorer, anchor, run,
                                                     at_anchor=start_policy == "carbon_anchor")
        if start_policy == "perturbed_cohort":
            weight, calibration = _calibrate_weight(model, reference, context, scorer, anchor)
        else:
            weight, calibration = 0.0, []
        shared_calls = {"forward": scorer.forward_calls, "backward": scorer.backward_calls}
        write_json(run.path / "weight_calibration.json", {"weight": weight, "rows": calibration})
        run.annotate(weight=weight, parameter_digest_before=initial_parameters,
                     context_digest_before=initial_context)
        final_rows = []
        for repeat in range(repeats):
            starts = _cohort_starts(anchor, repeat=repeat) if start_policy == "perturbed_cohort" else [
                {"repeat": repeat, "start_index": 0, "radius": 0.0, "point": anchor}]
            for item in starts:
                start = item["point"]
                target_endpoint = None
                start_hard, start_hard_projection_score = None, None
                for arm in arms:
                    state_path = run.path / f"trajectory-r{repeat}-i{item['start_index']}-{arm}.pt"
                    endpoint, records, calls = _trajectory(
                        start, anchor, arm, model, reference, context, scorer,
                        weight=weight, trust_radius=3.5, run=run, state_path=state_path,
                        repeat=repeat, start_index=item["start_index"], start_radius=item["radius"])
                    if arm == "target":
                        target_endpoint = endpoint
                        start_hard, start_hard_projection_score = records[0]["hard"], records[0]["hard_projection_score"]
                    final_rows.append({"repeat": repeat, "start_index": item["start_index"],
                                       "start_radius": item["radius"], "start_hard": records[0]["hard"],
                                       "start_hard_projection_score": records[0]["hard_projection_score"],
                                       "oracle_cost": calls,
                                       **records[-1]})
                    write_json(run.path / "final_attempts.json", final_rows)
                random_before = (scorer.forward_calls, scorer.backward_calls)
                random_row, random_state = _random_control(
                    start, target_endpoint, anchor, model, reference, context, scorer,
                    seed=29000 + 100 * repeat + item["start_index"])
                final_rows.append({"repeat": repeat, "start_index": item["start_index"],
                                   "start_radius": item["radius"],
                                   "start_hard": start_hard,
                                   "start_hard_projection_score": start_hard_projection_score,
                                   "oracle_cost": {"forward": scorer.forward_calls - random_before[0],
                                                   "backward": scorer.backward_calls - random_before[1]},
                                   **random_row})
                write_json(run.path / "final_attempts.json", final_rows)
                torch.save(random_state, run.path / f"random-r{repeat}-i{item['start_index']}.pt")
                run.log({"stage": "random_control", "repeat": repeat,
                         "start_index": item["start_index"], **random_row})
        frozen = parameter_digest(model, boltz) == initial_parameters
        context_frozen = context_digest(context) == initial_context
        summary = {"mode": mode, "baseline": baseline_report, "weight": weight,
                   "start_policy": start_policy,
                   "binary_gradient_gate": binary_gradient_gate,
                   "weight_calibration": calibration, "per_arm": _aggregate(final_rows, baseline_report),
                   "oracle_calls": {"all_fixed_context_forwards": scorer.forward_calls,
                                    "all_fixed_context_backwards": scorer.backward_calls,
                                    "shared_baseline_and_calibration": shared_calls,
                                    "by_purpose": scorer.by_purpose},
                   "parameters_unchanged": frozen,
                   "context_unchanged": context_frozen,
                   "fresh_pose_followup_required": True,
                   "limitation": "One fixed context, one pose, fixed atom count and ligand map; hard projection scores are not new Boltz predictions."}
        write_json(run.path / "final_attempts.json", final_rows)
        write_json(run.path / "summary.json", summary)
        if not frozen or not context_frozen:
            raise AssertionError("Frozen model or fixed context changed")
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--affinity-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    parser.add_argument("--start-policy", choices=("perturbed_cohort", "carbon_anchor"),
                        default="perturbed_cohort")
    args = parser.parse_args()
    print(run_pilot(args.prepared, args.affinity_checkpoint, args.checkpoint,
                    args.reference, args.data_manifest, args.run_root, mode=args.mode,
                    start_policy=args.start_policy))


if __name__ == "__main__":
    main()
