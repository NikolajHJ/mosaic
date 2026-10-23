"""HPC real-weight gradient checks and a bounded latent-only engineering probe.

Checks hard-feature identity, native discrete fixed-pose forward agreement,
decoder-channel and latent derivatives, local finite differences, and unchanged
frozen parameters. Passing does not demonstrate chemical validity, useful ligand
optimization, variable occupancy, or derivatives through pose generation.
"""

import argparse
import hashlib
import math
from pathlib import Path

import torch

from latentmol.chemistry import decode_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import HardGraphBatch
from .adapter import adapt_graph, fixed_occupancy_graph
from .contracts import PreparedComplex
from .model import FrozenBoltzAffinity, load_affinity_model
from .prepare import installed_boltz_provenance
from .provenance import context_digest  # compatibility re-export


def parameter_digest(*models) -> str:
    """Hash every frozen parameter exactly; buffers and contexts are not optimized."""
    digest = hashlib.sha256()
    for model in models:
        for name, value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def finite_difference_gate(rows, tolerance: float = 0.02, required_agreements: int = 2) -> dict:
    """Require agreement with autograd at multiple distinct positive step sizes.

    One lucky finite-difference scale cannot satisfy this gate, nor can repeating
    the same epsilon. All individual errors remain in the report, including
    failures at larger steps where curvature or nonsmooth operations matter.
    """
    if tolerance <= 0 or required_agreements < 2:
        raise ValueError("Require a positive tolerance and at least two agreeing scales")
    agreeing = sorted({float(row["epsilon"]) for row in rows
                       if math.isfinite(float(row["epsilon"])) and row["epsilon"] > 0
                       and math.isfinite(float(row["relative_error"]))
                       and 0 <= row["relative_error"] <= tolerance})
    return {"passed": len(agreeing) >= required_agreements,
            "agreeing_epsilons": agreeing, "relative_tolerance": tolerance,
            "required_distinct_agreements": required_agreements}


@torch.no_grad()
def score_discrete_projection(graph, context: PreparedComplex, scorer):
    """Argmax then score valid hard graphs in the same frozen native context.

    This is a one-hot projection through the shared adapter. It deliberately
    retains the original atom names, reference conformer, bound pose, MSA, crop
    and slot maps. It is NOT ordinary ligand re-featurization or a new structure
    prediction. Invalid graphs remain in the trajectory with null hard scores.
    """
    hard = HardGraphBatch(graph.atoms.argmax(-1), graph.charges.argmax(-1), graph.bonds.argmax(-1))
    smiles = decode_graph(hard.atoms[0].cpu().numpy(), hard.charges[0].cpu().numpy(),
                          hard.bonds[0].cpu().numpy(), context.spec)
    row = {"discrete_valid": bool(smiles), "discrete_smiles": smiles,
           "fixed_context_discrete_affinity_pred_value": None,
           "fixed_context_discrete_affinity_probability_binary": None,
           "discrete_score_policy": "same-context one-hot projection; no pose or native feature rebuild"}
    if smiles:
        features, types = adapt_graph(hard.to_relaxed(context.spec), context)
        scores = scorer(features, types, context.coordinates)
        row["fixed_context_discrete_affinity_pred_value"] = float(scores["affinity_pred_value"].mean())
        row["fixed_context_discrete_affinity_probability_binary"] = float(scores["affinity_probability_binary"].mean())
    return row, hard


def persist_trajectory(path: Path, payload: dict) -> dict:
    """Atomically save tensor/plain-data states, retaining the latest complete point."""
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path),
            "recorded_points": len(payload["states"]), "format_version": payload["format_version"]}


def run_smoke(prepared_path: Path, affinity_checkpoint: Path, ae_checkpoint: Path,
              run_root: Path, recycling_steps: int = 0, device: str = "cuda",
              optimization_steps: int = 3) -> Path:
    """Exercise the exact saved context and both frozen neural models in FP32."""
    if not 0 <= optimization_steps <= 10:
        raise ValueError("Engineering probe is limited to zero through ten updates")
    config = {"prepared": str(prepared_path), "affinity_checkpoint": str(affinity_checkpoint),
              "ae_checkpoint": str(ae_checkpoint), "recycling_steps": recycling_steps,
              "device": device, "dtype": "float32", "epsilons": [0.0001, 0.0003, 0.001, 0.003, 0.01],
              "finite_difference_relative_tolerance": 0.02,
              "finite_difference_required_agreements": 2,
              "forward_atol": 2e-5, "forward_rtol": 2e-5,
              "optimization_steps": optimization_steps, "normalized_gradient_step": 0.05,
              "policy": "fixed occupancy; element/charge/bond gradients; fixed pose, MSA and all other native context"}
    with ExperimentRun(run_root, "boltz_fixed_pose_gradient_smoke", config, Path.cwd()) as run:
        torch.manual_seed(17)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        context = PreparedComplex.load(prepared_path)
        installed = installed_boltz_provenance()
        ae, checkpoint = load_model(ae_checkpoint, "cpu", expected_spec=context.spec)
        ae.requires_grad_(False).eval()
        boltz = load_affinity_model(affinity_checkpoint)
        original_digest = parameter_digest(ae, boltz)
        context, ae, boltz = context.to(device), ae.to(device), boltz.to(device)
        original_context_digest = context_digest(context)
        if torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        scorer = FrozenBoltzAffinity(boltz, recycling_steps)
        run.annotate(prepared_sha256=file_sha256(prepared_path), ae_checkpoint_sha256=file_sha256(ae_checkpoint),
                     boltz_checkpoint_sha256=file_sha256(affinity_checkpoint), boltz=installed,
                     context=context.metadata, ae_training_mode=checkpoint.get("train_config", {}).get("mode"),
                     bond_type_feature=bool(boltz.bond_type_feature),
                     affinity_ensemble=bool(boltz.affinity_ensemble), molecular_weight_correction=False,
                     activation_checkpointing=scorer.activation_checkpoint_policy,
                     atom_character_features_fixed=True, initial_parameter_sha256=original_digest,
                     initial_context_sha256=original_context_digest,
                     objective_direction="minimize raw predicted log10(IC50 in micromolar); engineering probe only",
                     objective_documentation="https://github.com/jwohlwend/boltz/blob/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc/docs/prediction.md#affinity-prediction")

        hard_features, hard_types = adapt_graph(context.seed.to_relaxed(context.spec), context)
        for name in ("ref_element", "ref_charge", "token_bonds"):
            torch.testing.assert_close(hard_features[name], context.features[name].to(hard_features[name]), atol=0, rtol=0)
        torch.testing.assert_close(hard_types, torch.nn.functional.one_hot(context.features["type_bonds"].long(), 7).to(hard_types), atol=0, rtol=0)
        with torch.no_grad():
            native = scorer.discrete_reference(context.features, context.coordinates)
            hard = scorer(hard_features, hard_types, context.coordinates)
        differences = {}
        for key in native:
            differences[key] = float((native[key] - hard[key]).abs().max())
            torch.testing.assert_close(hard[key], native[key], atol=config["forward_atol"], rtol=config["forward_rtol"])
        run.log({"stage": "native_discrete_fixed_pose_parity", "absolute_differences": differences,
                 "reference_scores": {key: float(value.mean()) for key, value in native.items()}})

        with torch.no_grad():
            mu, _ = ae.encode(context.seed.to_relaxed(context.spec))
        latent = mu.detach().clone().requires_grad_(True)
        decoded = ae.decode(latent)
        graph = fixed_occupancy_graph(decoded, context.active, context.spec)
        features, types = adapt_graph(graph, context)
        scores = scorer(features, types, context.coordinates)
        objective = scores["affinity_pred_value"].mean()
        grad_z, grad_atoms, grad_charges, grad_bonds = torch.autograd.grad(
            objective, (latent, decoded.atoms, decoded.charges, decoded.bonds))
        gradients = {"latent": grad_z, "element_logits": grad_atoms[..., 1:],
                     "charge_logits": grad_charges, "bond_logits": grad_bonds}
        norms = {name: float(value.norm()) for name, value in gradients.items()}
        finite = {name: bool(torch.isfinite(value).all()) for name, value in gradients.items()}
        run.log({"stage": "input_gradients", "objective": float(objective.detach()),
                 "gradient_norms": norms, "finite": finite,
                 "no_atom_logit_gradient_norm": float(grad_atoms[..., 0].norm())})
        if not all(finite.values()) or any(value <= 1e-12 for value in norms.values()):
            raise AssertionError("Every claimed chemical channel and the latent must have a finite nonzero gradient")
        if grad_atoms[..., 0].count_nonzero():
            raise AssertionError("Fixed occupancy unexpectedly depends on NO_ATOM logits")
        if any(p.requires_grad or p.grad is not None for model in (ae, boltz) for p in model.parameters()):
            raise AssertionError("Frozen model parameters acquired gradients")
        direction = grad_z.detach() / grad_z.norm()
        analytic = float((grad_z * direction).sum())
        trajectory_path = run.path / "latent_trajectory.pt"
        trajectory_payload = {
            "format_version": 1, "graph_spec": context.spec.to_dict(),
            "initial_latent": latent.detach().cpu().clone(),
            "initial_latent_gradient": grad_z.detach().cpu().clone(),
            "finite_difference_direction": direction.detach().cpu().clone(),
            "states": [], "records": [],
            "identity": {"run_id": run.run_id, "source_sha256": run.record["source"]["sha256"],
                         "source_archive_sha256": run.record["source_archive_sha256"],
                         "prepared_sha256": run.record["prepared_sha256"],
                         "ae_checkpoint_sha256": run.record["ae_checkpoint_sha256"],
                         "boltz_checkpoint_sha256": run.record["boltz_checkpoint_sha256"],
                         "fixed_context_sha256": original_context_digest,
                         "parameter_sha256": original_digest},
            "config": config,
        }
        trajectory_artifact = persist_trajectory(trajectory_path, trajectory_payload)
        run.annotate(latent_trajectory=trajectory_artifact)
        # Release graph intermediates before the finite-difference forwards.
        del scores, objective, features, types, graph, decoded

        @torch.no_grad()
        def value_at(point):
            fixed = fixed_occupancy_graph(ae.decode(point), context.active, context.spec)
            native_features, native_types = adapt_graph(fixed, context)
            return float(scorer(native_features, native_types, context.coordinates)["affinity_pred_value"].mean())

        differences = []
        for epsilon in config["epsilons"]:
            plus, minus = value_at(latent.detach() + epsilon*direction), value_at(latent.detach() - epsilon*direction)
            numerical = (plus - minus) / (2*epsilon)
            relative_error = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-12)
            row = {"epsilon": epsilon, "plus": plus, "minus": minus, "analytic": analytic,
                   "central_difference": numerical, "relative_error": relative_error}
            differences.append(row)
            run.log({"stage": "latent_directional_finite_difference", **row})
        fd_gate = finite_difference_gate(differences, config["finite_difference_relative_tolerance"],
                                         config["finite_difference_required_agreements"])
        fd_pass = fd_gate["passed"]
        run.log({"stage": "finite_difference_acceptance", **fd_gate})
        trajectory = []
        if fd_pass and optimization_steps:
            # This bounded engineering probe is intentionally not a design study:
            # a small normalized step limits movement, but adds no chemical prior
            # or validity guarantee. Only latent tensors are updated.
            probe = latent.detach().clone().requires_grad_(True)
            for step in range(optimization_steps + 1):
                decoded_probe = ae.decode(probe)
                fixed = fixed_occupancy_graph(decoded_probe, context.active, context.spec)
                probe_features, probe_types = adapt_graph(fixed, context)
                probe_scores = scorer(probe_features, probe_types, context.coordinates)
                score = probe_scores["affinity_pred_value"].mean()
                discrete_metrics, hard_projection = score_discrete_projection(fixed, context, scorer)
                row = {"step": step, "relaxed_affinity_pred_value": float(score.detach()),
                       "relaxed_affinity_probability_binary": float(probe_scores["affinity_probability_binary"].mean().detach()),
                       "latent_displacement": float((probe.detach() - latent.detach()).norm()),
                       "fixed_heavy_atom_count": int(context.active.sum()),
                       **discrete_metrics}
                trajectory.append(row)
                run.log({"stage": "engineering_latent_only_probe", **row})
                # CPU clones preserve the exact evaluated latent, raw logits and
                # slot-ordered hard graph, even if a later update or check fails.
                trajectory_payload["states"].append({
                    "step": step, "latent": probe.detach().cpu().clone(),
                    "decoder_logits": {name: getattr(decoded_probe, name).detach().cpu().clone()
                                       for name in ("atoms", "charges", "bonds")},
                    "hard_graph": {name: getattr(hard_projection, name).detach().cpu().clone()
                                   for name in ("atoms", "charges", "bonds")},
                })
                trajectory_payload["records"].append(row)
                trajectory_artifact = persist_trajectory(trajectory_path, trajectory_payload)
                run.annotate(latent_trajectory=trajectory_artifact)
                if step < optimization_steps:
                    gradient, = torch.autograd.grad(score, probe)
                    if not torch.isfinite(gradient).all() or gradient.norm() <= 1e-12:
                        raise AssertionError("Engineering probe lost a finite nonzero latent gradient")
                    probe = (probe - config["normalized_gradient_step"] * gradient / gradient.norm()).detach().requires_grad_(True)
                del fixed, decoded_probe, probe_features, probe_types, probe_scores, score, hard_projection
        final_digest = parameter_digest(ae, boltz)
        final_context_digest = context_digest(context)
        report = {"native_discrete_fixed_pose_parity": True, "gradient_norms": norms,
                  "finite_differences": differences, "parameters_unchanged": final_digest == original_digest,
                  "fixed_context_unchanged": final_context_digest == original_context_digest,
                  "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None,
                  "finite_difference_pass": fd_pass, "finite_difference_gate": fd_gate,
                  "engineering_probe": trajectory, "latent_trajectory": trajectory_artifact,
                  "claim": "partial derivative through fixed-occupancy chemical features, frozen trunk and affinity, and frozen decoder",
                  "unsupported": ["atom birth/death", "pose/diffusion derivatives", "dynamic native featurization", "chemical or binding improvement"]}
        write_json(run.path / "gradient_report.json", report)
        run.annotate(final_parameter_sha256=final_digest, final_context_sha256=final_context_digest, acceptance=report)
        if not all(report[key] for key in ("parameters_unchanged", "fixed_context_unchanged", "finite_difference_pass")):
            raise AssertionError("Frozen-parameter, fixed-context or local finite-difference gate failed; inspect saved report")
    return run.path


def main():
    """Command-line entrypoint; caller supplies an HPC GPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--affinity-checkpoint", required=True, type=Path)
    parser.add_argument("--ae-checkpoint", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--recycling-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--optimization-steps", type=int, default=3)
    args = parser.parse_args()
    run_smoke(args.prepared, args.affinity_checkpoint, args.ae_checkpoint,
              args.run_root, args.recycling_steps, args.device, args.optimization_steps)


if __name__ == "__main__":
    main()
