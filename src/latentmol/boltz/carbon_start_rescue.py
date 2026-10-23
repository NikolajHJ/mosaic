"""Matched AE/VAE feasibility screen for a nearby, exact-C13 latent start.

The native C13 encoder mean remains the reference anchor. A small, deterministic
decoder-only neighborhood search asks whether another latent point with the *same*
hard C13 graph has a usable frozen-Boltz derivative. This is a separate experiment
from the exact-anchor source031 pilot, not a replacement for its failed AE gate.
All Boltz calls retain the one fixed C13 pose, occupancy, and ligand map.
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
from .adapter import fixed_occupancy_graph
from .contracts import PreparedComplex
from .latent_pilot import (CountingScorer, _aggregate, _direction, _hard_score,
                           _random_control, _score_state, _trajectory)
from .model import FrozenBoltzAffinity, load_affinity_model
from .smoke import context_digest, finite_difference_gate, parameter_digest


CARBON_CHAIN = "CCCCCCCCCCCCC"
DIRECTION_SEEDS = tuple(33000 + index for index in range(8))
RADII = (0.25, 0.5, 1.0, 2.0)
FD_EPSILONS = (0.0001, 0.0003, 0.001, 0.003, 0.01)
MIN_GRADIENT_NORM = 1e-12
MAX_FD_CANDIDATES = 2


def _graph_labels_equal(actual: HardGraphBatch, expected: HardGraphBatch) -> dict[str, bool]:
    """Require slot-ordered atom, charge, and bond indices, including padding."""
    return {field: bool(torch.equal(getattr(actual, field), getattr(expected, field)))
            for field in ("atoms", "charges", "bonds")}


def exact_native_graphs(decoded, context: PreparedComplex) -> dict:
    """Audit both graph interpretations before considering a starting point.

    Ordinary hardening includes the learned NO_ATOM option. The fixed graph
    conditions occupancy before hardening for the Boltz adapter. Equality in only
    one interpretation is insufficient, even if its resulting SMILES is C13.
    """
    expected = context.seed
    ordinary = decoded.harden(context.spec)
    fixed = fixed_occupancy_graph(decoded, context.active, context.spec)
    fixed_hard = HardGraphBatch(fixed.atoms.argmax(-1), fixed.charges.argmax(-1),
                                fixed.bonds.argmax(-1))
    ordinary_equal = _graph_labels_equal(ordinary, expected)
    fixed_equal = _graph_labels_equal(fixed_hard, expected)
    return {"ordinary_fields_equal": ordinary_equal,
            "fixed_fields_equal": fixed_equal,
            "exact_native_c13": all(ordinary_equal.values()) and all(fixed_equal.values())}


def active_secant_response(candidate, anchor, active: torch.Tensor,
                           displacement: float) -> dict[str, float]:
    """Decoder-only active-channel secant norm per normalized-latent distance.

    This is a finite displacement in decoded *probabilities*, not a Jacobian.
    Atom elements and formal charges use active slots; bonds use each undirected
    active pair once. The distance is measured in standardized latent units.
    The three L2 norms are combined without fitting weights or consulting Boltz.
    """
    if displacement <= 0 or not math.isfinite(displacement):
        raise ValueError("A positive finite latent displacement is required")
    slots = active.nonzero().flatten()
    if slots.numel() < 2:
        raise ValueError("Need at least two native active slots for bond response")
    atom_delta = (candidate.atoms[:, slots, 1:] - anchor.atoms[:, slots, 1:]).float()
    charge_delta = (candidate.charges[:, slots] - anchor.charges[:, slots]).float()
    upper = torch.triu_indices(slots.numel(), slots.numel(), offset=1,
                               device=slots.device)
    bond_delta = (candidate.bonds[:, slots[:, None], slots[None, :]]
                  - anchor.bonds[:, slots[:, None], slots[None, :]])
    bond_delta = bond_delta[:, upper[0], upper[1]].float()
    element_norm = float(atom_delta.norm())
    charge_norm = float(charge_delta.norm())
    bond_norm = float(bond_delta.norm())
    combined = math.sqrt(element_norm ** 2 + charge_norm ** 2 + bond_norm ** 2)
    return {"element_l2": element_norm, "charge_l2": charge_norm,
            "bond_upper_triangle_l2": bond_norm,
            "combined_l2": combined, "per_latent_distance": combined / displacement}


def decoder_neighborhood(model, reference, context: PreparedComplex,
                         anchor: torch.Tensor, *, record_candidate=None
                         ) -> tuple[list[dict], dict[str, torch.Tensor]]:
    """Audit the anchor and eight shared directions at each of four radii.

    Returns small JSON rows and exact normalized latent tensors separately. No
    Boltz scorer is accepted by this function, preventing oracle-based selection.
    """
    if anchor.shape[0] != 1:
        raise ValueError("C13 local-start screen requires one encoded molecule")
    with torch.no_grad():
        anchor_decoded = model.decode(reference.denormalize(anchor))
        anchor_audit = exact_native_graphs(anchor_decoded, context)
        if not anchor_audit["exact_native_c13"]:
            raise ValueError("Encoded C13 anchor does not reproduce all native hard labels")
        anchor_prob = fixed_occupancy_graph(anchor_decoded, context.active, context.spec)
        anchor_ordinary = anchor_decoded.probabilities()
        rows = [{"candidate_id": "anchor", "radius": 0.0,
                 "direction_index": None, "direction_seed": None,
                 "normalized_latent_distance": 0.0,
                 "ordinary_expected_atoms": float(anchor_ordinary.expected_size[0]),
                 **anchor_audit, "active_secant": None}]
        latents = {"anchor": anchor.detach().cpu().clone()}
        if record_candidate is not None:
            record_candidate(rows[-1])
        directions = [_direction(anchor.shape[-1], seed, str(anchor.device))
                      for seed in DIRECTION_SEEDS]
        for radius in RADII:
            for index, direction in enumerate(directions):
                point = (anchor + radius * direction).detach()
                candidate_id = f"r{radius:g}-d{index}"
                decoded = model.decode(reference.denormalize(point))
                audit = exact_native_graphs(decoded, context)
                candidate_prob = fixed_occupancy_graph(decoded, context.active,
                                                       context.spec)
                measured_distance = float((point - anchor).norm())
                response = active_secant_response(candidate_prob, anchor_prob,
                                                  context.active, measured_distance)
                ordinary_size = float(decoded.probabilities().expected_size[0])
                rows.append({"candidate_id": candidate_id, "radius": radius,
                             "direction_index": index,
                             "direction_seed": DIRECTION_SEEDS[index],
                             "normalized_latent_distance": measured_distance,
                             "ordinary_expected_atoms": ordinary_size,
                             **audit, "active_secant": response})
                latents[candidate_id] = point.cpu().clone()
                if record_candidate is not None:
                    record_candidate(rows[-1])
    if len(rows) != 33 or len(latents) != 33:
        raise AssertionError("Local-start design requires 33 decoder evaluations")
    return rows, latents


def select_decoder_finalists(rows: list[dict]) -> list[dict]:
    """Pick at most one exact-hard finalist per radius by a fixed decoder metric."""
    finalists = []
    for radius in RADII:
        eligible = [row for row in rows if row["radius"] == radius
                    and row["exact_native_c13"]]
        if eligible:
            # Ties have a stable, seed-independent direction-index rule.
            selected = sorted(eligible,
                              key=lambda row: (-row["active_secant"]["per_latent_distance"],
                                               row["direction_index"]))[0]
            finalists.append(selected)
    return finalists


def _screen_gradient(point, model, reference, context, scorer, *, candidate_id: str):
    report, _, gradient = _score_state(point, model, reference, context, scorer,
                                        require_gradient=True,
                                        purpose="finalist_gradient_screen")
    finite = bool(torch.isfinite(gradient).all())
    norm = float(gradient.norm())
    viable = finite and math.isfinite(norm) and norm > MIN_GRADIENT_NORM
    return {"candidate_id": candidate_id,
            "normalized_gradient_norm": norm if math.isfinite(norm) else None,
            "gradient_finite": finite, "viable": viable,
            "binary_probability": report["binary_probability"],
            "affinity_pred_value": report["affinity_pred_value"]}, gradient.detach()


def _finite_difference_at(point, gradient, model, reference, context, scorer,
                          run, *, candidate_id: str) -> dict:
    """Use the screened exact objective gradient and source031's FD gate."""
    norm = float(gradient.norm())
    if (not bool(torch.isfinite(gradient).all()) or not math.isfinite(norm)
            or norm <= MIN_GRADIENT_NORM):
        raise ValueError("Finite differences require a finite nonzero analytic gradient norm")
    direction = gradient / norm
    analytic = float((gradient * direction).sum())
    rows = []
    for epsilon in FD_EPSILONS:
        values = []
        for sign in (+1, -1):
            report, _, _ = _score_state(point + sign * epsilon * direction,
                                         model, reference, context, scorer,
                                         require_gradient=False,
                                         purpose="candidate_binary_fd_forward")
            values.append(-report["binary_probability"])
        numerical = (values[0] - values[1]) / (2 * epsilon)
        relative = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-12)
        row = {"candidate_id": candidate_id, "epsilon": epsilon,
               "analytic": analytic, "central_difference": numerical,
               "relative_error": relative}
        rows.append(row)
        run.log({"stage": "candidate_binary_directional_fd", **row})
    gate = finite_difference_gate(rows, 0.02, 2)
    return {"candidate_id": candidate_id, "rows": rows, "gate": gate}


def _load_inputs(prepared: Path, checkpoint: Path, reference_path: Path,
                 data_manifest: Path, mode: str):
    """Share source031's checkpoint/reference compatibility requirements."""
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
    inspected = inspect_graph(context.seed.atoms[0], context.seed.charges[0],
                              context.seed.bonds[0], context.spec)
    if inspected.smiles != CARBON_CHAIN:
        raise ValueError("Local-start rescue requires exact native C13 seed")
    return context, model, LatentReference.from_artifact(raw_reference, meta), manifest


def run_rescue(prepared: Path, affinity_checkpoint: Path, checkpoint: Path,
               reference_path: Path, data_manifest: Path, run_root: Path, *,
               mode: str, device: str = "cuda") -> Path:
    """Run matched decoder search, four screens, at most two FD tests, one pilot."""
    context, model, reference, manifest = _load_inputs(
        prepared, checkpoint, reference_path, data_manifest, mode)
    config = {"prepared": str(prepared), "prepared_sha256": file_sha256(prepared),
              "affinity_checkpoint_sha256": file_sha256(affinity_checkpoint),
              "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "reference": str(reference_path), "reference_sha256": file_sha256(reference_path),
              "data_manifest": str(data_manifest),
              "data_manifest_sha256": file_sha256(data_manifest),
              "dataset_sha256": manifest["dataset_sha256"],
              "split_sha256": manifest["split_sha256"],
              "graph_spec_sha256": context.spec.fingerprint(),
              "seed_smiles": CARBON_CHAIN,
              "mode": mode, "device": device, "start_policy": "local_exact_c13_rescue",
              "anchor": "native C13 encoder mean in standardized latent coordinates",
              "direction_seeds": list(DIRECTION_SEEDS), "radii": list(RADII),
              "decoder_evaluations_planned": 33,
              "hard_filter": "ordinary and fixed-occupancy atom, charge, bond arrays exactly equal native C13",
              "decoder_selection": "per-radius maximum active-channel probability secant norm per standardized latent distance; stable direction-index ties",
              "boltz_selection": "first nearest-radius finalist with finite nonzero gradient and passing five-epsilon FD gate; at most two FD candidates; no affinity-value ranking",
              "gradient_screens_max": 4, "fd_candidates_max": MAX_FD_CANDIDATES,
              "fixed_context_scorer_upper_bound": {"forward": 41, "backward": 14},
              "fd_epsilons": list(FD_EPSILONS), "fd_relative_tolerance": 0.02,
              "fd_required_agreements": 2, "steps": 10, "step_size": 0.25,
              "max_normalized_move_from_selected_start": 2.5,
              "objective": "maximize frozen fixed-context binary binder probability",
              "controls": ["original_exact_C13_anchor", "random_matched_displacement"],
              "context": "one native C13 fixed pose, occupancy and ligand atom map",
              "recycling_steps": 0,
              "scope": "feasibility of an exact-hard-C13 nearby start; fresh native predictions required for binding comparison"}
    with ExperimentRun(run_root, f"{mode}-boltz-carbon-local-start-rescue", config,
                       Path.cwd()) as run:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = model.to(device).eval().requires_grad_(False)
        context = context.to(device)
        reference = reference.to(device)
        boltz = load_affinity_model(affinity_checkpoint).to(device)
        scorer = CountingScorer(FrozenBoltzAffinity(boltz, 0),
                                run.path / "oracle_cost.json")
        initial_parameters = parameter_digest(model, boltz)
        initial_context = context_digest(context)
        run.annotate(parameter_digest_before=initial_parameters,
                     context_digest_before=initial_context)
        with torch.no_grad():
            mean, _ = model.encode(context.seed.to_relaxed(context.spec))
            anchor = reference.normalize(mean).detach()
        baseline, baseline_state, _ = _score_state(
            anchor, model, reference, context, scorer,
            require_gradient=False, purpose="original_c13_baseline")
        baseline["hard_projection_score"] = _hard_score(
            anchor, model, reference, context, scorer,
            purpose="original_c13_baseline_hard_projection")
        if baseline["hard"]["smiles"] != CARBON_CHAIN:
            raise AssertionError("Original anchor fixed graph is not C13")
        write_json(run.path / "baseline.json", baseline)
        torch.save(baseline_state, run.path / "baseline_state.pt")

        partial_audit = []

        def record_candidate(row):
            partial_audit.append(row)
            write_json(run.path / "candidate_audit.json", partial_audit)

        rows, latents = decoder_neighborhood(model, reference, context, anchor,
                                             record_candidate=record_candidate)
        torch.save(latents, run.path / "candidate_latents.pt")
        finalists = select_decoder_finalists(rows)
        write_json(run.path / "decoder_finalists.json", finalists)
        by_radius = {item["radius"]: item for item in finalists}
        screen_slots = [{"radius": radius,
                         "candidate_id": by_radius[radius]["candidate_id"] if radius in by_radius else None,
                         "status": "pending" if radius in by_radius else "no_exact_native_c13_candidate"}
                        for radius in RADII]
        write_json(run.path / "screen_slots.json", screen_slots)
        run.log({"stage": "decoder_search", "evaluated": len(rows),
                 "exact_hard_candidates": sum(row["exact_native_c13"] for row in rows[1:]),
                 "finalist_count": len(finalists)})

        screens = []
        viable = []
        for finalist in finalists:
            candidate_id = finalist["candidate_id"]
            point = latents[candidate_id].to(device)
            screen, gradient = _screen_gradient(
                point, model, reference, context, scorer,
                candidate_id=candidate_id)
            screens.append({"radius": finalist["radius"], **screen})
            slot = next(item for item in screen_slots if item["radius"] == finalist["radius"])
            slot["status"] = "viable_gradient" if screen["viable"] else "gradient_gate_failed"
            write_json(run.path / "screen_slots.json", screen_slots)
            write_json(run.path / "gradient_screens.json", screens)
            run.log({"stage": "finalist_gradient_screen", "radius": finalist["radius"],
                     **screen})
            if screen["viable"]:
                viable.append((finalist, gradient))

        fd_results = []
        selected = None
        for finalist, gradient in viable[:MAX_FD_CANDIDATES]:
            candidate_id = finalist["candidate_id"]
            point = latents[candidate_id].to(device)
            fd = _finite_difference_at(point, gradient, model, reference, context,
                                       scorer, run, candidate_id=candidate_id)
            fd_results.append(fd)
            write_json(run.path / "fd_candidates.json", fd_results)
            if fd["gate"]["passed"]:
                selected = finalist
                break
        selection = {"selected_candidate_id": selected["candidate_id"] if selected else None,
                     "selected_radius": selected["radius"] if selected else None,
                     "decoder_evaluations": len(rows),
                     "exact_hard_candidates_excluding_anchor": sum(
                         row["exact_native_c13"] for row in rows[1:]),
                     "gradient_screens": len(screens),
                     "viable_gradient_screens": len(viable),
                     "fd_candidates_attempted": len(fd_results),
                     "fd_passes": sum(item["gate"]["passed"] for item in fd_results),
                     "selection_used_affinity_values": False,
                     "selection_used_native_predictions": False}
        write_json(run.path / "selection.json", selection)
        if selected is None:
            if not finalists:
                failure_reason = "no_exact_hard_c13_finalists"
            elif not viable:
                failure_reason = "no_finite_nonzero_finalist_gradients"
            else:
                failure_reason = "fd_no_candidate_passed"
            write_json(run.path / "summary.json", {
                "mode": mode, "start_policy": "local_exact_c13_rescue",
                "status": "failed_feasibility_gate",
                "failure_reason": failure_reason,
                "baseline": baseline, "selected": selection,
                "screen_slots": screen_slots,
                "screen_budget": {"planned": len(RADII),
                                  "available_finalists": len(finalists),
                                  "completed": len(screens)},
                "fd_budget": {"maximum_candidates": MAX_FD_CANDIDATES,
                              "attempted": len(fd_results)},
                "trajectory_attempted": False,
                "oracle_calls": {"all_fixed_context_forwards": scorer.forward_calls,
                                 "all_fixed_context_backwards": scorer.backward_calls,
                                 "by_purpose": scorer.by_purpose},
                "fresh_pose_followup_required": False,
                "limitation": "No local-start gradient-verified trajectory; fixed-context feasibility screen only."})
            raise AssertionError("No exact-C13 local-start finalist passed the binary gradient and finite-difference gates")

        start = latents[selected["candidate_id"]].to(device)
        end, records, target_calls = _trajectory(
            start, anchor, "target", model, reference, context, scorer,
            weight=0.0, run=run, state_path=run.path / "trajectory-r0-i0-target.pt",
            repeat=0, start_index=0, start_radius=selected["radius"],
            steps=10, step_size=0.25, trust_radius=3.5, max_move=2.5)
        target_row = {"repeat": 0, "start_index": 0,
                      "start_radius": selected["radius"],
                      "start_candidate_id": selected["candidate_id"],
                      "start_hard": records[0]["hard"],
                      "start_hard_projection_score": records[0]["hard_projection_score"],
                      "oracle_cost": target_calls, **records[-1]}
        write_json(run.path / "final_attempts.json", [target_row])
        if records[0]["hard"]["smiles"] != CARBON_CHAIN:
            raise AssertionError("Selected local start lost the hard native C13 graph")
        before = (scorer.forward_calls, scorer.backward_calls)
        random_row, random_state = _random_control(
            start, end, anchor, model, reference, context, scorer, seed=39000)
        random_row = {"repeat": 0, "start_index": 0,
                      "start_radius": selected["radius"],
                      "start_candidate_id": selected["candidate_id"],
                      "start_hard": records[0]["hard"],
                      "start_hard_projection_score": records[0]["hard_projection_score"],
                      "oracle_cost": {"forward": scorer.forward_calls - before[0],
                                      "backward": scorer.backward_calls - before[1]},
                      **random_row}
        torch.save(random_state, run.path / "random-r0-i0.pt")
        final_rows = [target_row, random_row]
        write_json(run.path / "final_attempts.json", final_rows)
        run.log({"stage": "random_control", **random_row})
        frozen = parameter_digest(model, boltz) == initial_parameters
        context_frozen = context_digest(context) == initial_context
        summary = {"mode": mode, "start_policy": "local_exact_c13_rescue",
                   "baseline": baseline, "selected": selection,
                   "per_arm": _aggregate(final_rows, baseline),
                   "parameters_unchanged": frozen,
                   "context_unchanged": context_frozen,
                   "oracle_calls": {"all_fixed_context_forwards": scorer.forward_calls,
                                    "all_fixed_context_backwards": scorer.backward_calls,
                                    "by_purpose": scorer.by_purpose},
                   "fresh_pose_followup_required": True,
                   "limitation": "One fixed C13 pose and atom count; hard projection scores are not fresh Boltz predictions. Local-start search and its Boltz screens consume extra budget versus source031."}
        write_json(run.path / "summary.json", summary)
        if not frozen or not context_frozen:
            raise AssertionError("Frozen model or fixed context changed")
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--affinity-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run_rescue(args.prepared, args.affinity_checkpoint, args.checkpoint,
                     args.reference, args.data_manifest, args.run_root, mode=args.mode,
                     device=args.device))


if __name__ == "__main__":
    main()
