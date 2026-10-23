"""Bounded in-schema denoising feasibility pilot on the DTU HPC training split.

The pilot checks held-out corruption repair and a local soft-input derivative.
It does not measure a molecular prior's effect on Boltz optimization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry import decode_graph
from latentmol.data import GraphDataset, collate_graphs
from latentmol.denoiser import (DenoiserLossConfig, SchemaDenoiser,
                               corrupt_graph, denoising_reconstruction_losses)
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import RelaxedGraph
from latentmol.sampling import subset_indices
from latentmol.train import seed_everything


@dataclass(frozen=True)
class DenoiserPilotConfig:
    seed: int = 17
    train_limit: int = 5000
    validation_limit: int = 256
    epochs: int = 4
    batch_size: int = 64
    learning_rate: float = 0.0002
    weight_decay: float = 0.00001
    gradient_clip: float = 5.0
    train_time_min: float = 0.10
    train_time_max: float = 0.35
    validation_time: float = 0.25

    def __post_init__(self):
        if self.train_limit < 1 or self.validation_limit < 1 or self.epochs < 1 or self.batch_size < 1:
            raise ValueError("Pilot sizes, epochs and batch size must be positive")
        if not 0 < self.train_time_min < self.train_time_max <= 1:
            raise ValueError("Invalid train corruption range")
        if not 0 < self.validation_time <= 1 or self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid validation corruption or optimizer setting")


def _index_hash(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def _loader(data, indices: list[int], batch_size: int, *, shuffle: bool, seed: int):
    return DataLoader(Subset(data, indices), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_graphs, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def _evaluate(model, baseline, loader, reference_smiles: list[str], spec,
              device: str, time_value: float, noise_seed: int,
              loss_config: DenoiserLossConfig) -> dict:
    model.eval()
    baseline.eval()
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    loss_sum = baseline_loss_sum = 0.0
    count = valid = exact = baseline_valid = baseline_exact = 0
    size_error = baseline_size_error = 0
    predicted_edges = baseline_predicted_edges = true_edges = 0
    edge_hits = baseline_edge_hits = 0
    unique_smiles, baseline_unique_smiles = set(), set()
    cursor = 0
    for clean in loader:
        clean = clean.to(device)
        batch = clean.atoms.shape[0]
        times = clean.atoms.new_full((batch,), time_value, dtype=torch.float32)
        noisy = corrupt_graph(clean, spec, times, generator)
        graph = noisy.to_relaxed(spec)
        prediction = model(graph, times)
        comparison = baseline(graph).decoded
        loss_sum += float(denoising_reconstruction_losses(
            prediction, clean, loss_config)["reconstruction"]) * batch
        baseline_loss_sum += float(denoising_reconstruction_losses(
            comparison, clean, loss_config)["reconstruction"]) * batch
        candidate = prediction.harden(spec)
        control = comparison.harden(spec)
        expected_size = (clean.atoms != 0).sum(-1)
        size_error += int(((candidate.atoms != 0).sum(-1) - expected_size).abs().sum())
        baseline_size_error += int(((control.atoms != 0).sum(-1) - expected_size).abs().sum())
        upper = torch.ones((spec.max_nodes, spec.max_nodes),
                           dtype=torch.bool, device=device).triu(1)
        true_bond_mask = (clean.bonds != 0) & upper
        candidate_bond_mask = (candidate.bonds != 0) & upper
        control_bond_mask = (control.bonds != 0) & upper
        true_edges += int(true_bond_mask.sum())
        predicted_edges += int(candidate_bond_mask.sum())
        baseline_predicted_edges += int(control_bond_mask.sum())
        edge_hits += int((candidate_bond_mask & true_bond_mask).sum())
        baseline_edge_hits += int((control_bond_mask & true_bond_mask).sum())
        for i in range(batch):
            expected = reference_smiles[cursor + i]
            chosen = decode_graph(candidate.atoms[i].cpu(), candidate.charges[i].cpu(),
                                  candidate.bonds[i].cpu(), spec)
            original = decode_graph(control.atoms[i].cpu(), control.charges[i].cpu(),
                                    control.bonds[i].cpu(), spec)
            valid += bool(chosen)
            exact += chosen == expected
            baseline_valid += bool(original)
            baseline_exact += original == expected
            if chosen:
                unique_smiles.add(chosen)
            if original:
                baseline_unique_smiles.add(original)
        cursor += batch
        count += batch
    if cursor != len(reference_smiles):
        raise ValueError("Validation identity order disagrees with loader")
    return {"count": count, "reconstruction_loss": loss_sum / count,
            "frozen_ae_reconstruction_loss": baseline_loss_sum / count,
            "valid_hard": valid, "exact_hard": exact,
            "unique_valid_hard": len(unique_smiles),
            "size_mae": size_error / count,
            "predicted_edges": predicted_edges,
            "edge_precision": edge_hits / predicted_edges if predicted_edges else None,
            "edge_recall": edge_hits / true_edges if true_edges else None,
            "frozen_ae_valid_hard": baseline_valid,
            "frozen_ae_exact_hard": baseline_exact,
            "frozen_ae_unique_valid_hard": len(baseline_unique_smiles),
            "frozen_ae_size_mae": baseline_size_error / count,
            "frozen_ae_predicted_edges": baseline_predicted_edges,
            "frozen_ae_edge_precision": (baseline_edge_hits / baseline_predicted_edges
                                         if baseline_predicted_edges else None),
            "frozen_ae_edge_recall": (baseline_edge_hits / true_edges
                                      if true_edges else None)}


def _gradient_gate(model, baseline, clean, spec, device: str) -> dict:
    """Finite-difference check of one declared soft consistency scalar at fixed target."""
    model.eval()
    model.requires_grad_(False)
    baseline.eval()
    baseline.requires_grad_(False)
    clean = clean.to(device)
    with torch.no_grad():
        mu, _ = baseline.encode(clean.to_relaxed(spec))
        initial = mu[:1].clone()
        reference = baseline.decode(initial).probabilities()
        fixed = RelaxedGraph(reference.atoms.detach(), reference.charges.detach(),
                             reference.bonds.detach())
    time = torch.full((1,), 0.25, device=device)

    def score(latent):
        graph = baseline.decode(latent).probabilities()
        prediction = model(graph, time).probabilities()
        return ((prediction.atoms - fixed.atoms).square().mean()
                + (prediction.charges - fixed.charges).square().mean()
                + (prediction.bonds - fixed.bonds).square().mean())

    z = initial.detach().requires_grad_(True)
    analytic = torch.autograd.grad(score(z), z)[0]
    coordinate = int(analytic.abs().argmax())
    epsilon = 0.01
    plus, minus = initial.clone(), initial.clone()
    plus[0, coordinate] += epsilon
    minus[0, coordinate] -= epsilon
    with torch.no_grad():
        finite_difference = float((score(plus) - score(minus)) / (2 * epsilon))
    derivative = float(analytic.flatten()[coordinate])
    difference = abs(derivative - finite_difference)
    threshold = max(1e-5, 0.25 * abs(finite_difference))
    return {"soft_consistency_value": float(score(z).detach()),
            "latent_gradient_norm": float(analytic.norm()),
            "coordinate": coordinate, "analytic": derivative,
            "finite_difference": finite_difference, "absolute_error": difference,
            "threshold": threshold,
            "passed": bool(torch.isfinite(analytic).all() and analytic.norm() > 1e-8
                           and difference <= threshold),
            "scope": "Local derivative of a fixed-target soft consistency scalar only."}


def run_pilot(checkpoint_path: Path, dataset_dir: Path, cohorts_path: Path, run_root: Path,
              device: str, settings: DenoiserPilotConfig) -> Path:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    seed_everything(settings.seed)
    training = GraphDataset(dataset_dir, "train")
    validation = GraphDataset(dataset_dir, "val")
    spec = training.spec
    baseline, checkpoint = load_model(checkpoint_path, device, spec)
    if checkpoint["train_config"]["mode"] != "ae":
        raise ValueError("Pilot initializes only from the validation-selected AE")
    if checkpoint["dataset_sha256"] != training.manifest["dataset_sha256"]:
        raise ValueError("AE checkpoint and training cache differ")
    baseline.requires_grad_(False)
    train_indices = subset_indices(len(training), settings.train_limit, settings.seed + 401)
    cohorts = json.loads(cohorts_path.read_text())
    val_indices = cohorts["pilot"]
    reserved = set(cohorts["promotion_reserved"])
    if (len(val_indices) != settings.validation_limit or len(set(val_indices)) != len(val_indices)
            or any(index < 0 or index >= len(validation) for index in val_indices)
            or reserved.intersection(val_indices)):
        raise ValueError("Stage 1 pilot cohort is invalid or intersects its promotion reserve")
    train_loader = _loader(training, train_indices, settings.batch_size,
                           shuffle=True, seed=settings.seed + 403)
    val_loader = _loader(validation, val_indices, settings.batch_size,
                         shuffle=False, seed=settings.seed + 404)
    identities = [validation.smiles[i] for i in val_indices]
    config = {"settings": asdict(settings), "device": device,
              "dataset_sha256": training.manifest["dataset_sha256"],
              "split_sha256": training.manifest["split_sha256"],
              "graph_spec": spec.to_dict(), "ae_checkpoint": str(checkpoint_path),
              "ae_checkpoint_sha256": file_sha256(checkpoint_path),
              "ae_checkpoint_epoch": checkpoint["epoch"],
              "stage1_cohorts": str(cohorts_path),
              "stage1_cohorts_sha256": file_sha256(cohorts_path),
              "validation_role": "reused Stage 1 pilot development cohort; promotion reserve excluded",
              "train_indices_sha256": _index_hash(train_indices),
              "validation_indices_sha256": _index_hash(val_indices),
              "note": "Validation is development evaluation; the test split is unopened here."}
    loss_config = DenoiserLossConfig()
    config["loss_config"] = asdict(loss_config)
    config["intervention"] = "target-defined bond mask, no-bond and atom-padding reweighting, clean retention, hard-outcome checkpoint selection"
    with ExperimentRun(run_root, "stage3-schema-denoiser-repair", config, Path.cwd()) as run:
        write_json(run.path / "selected_indices.json", {"train": train_indices,
                                                         "validation": val_indices})
        model = SchemaDenoiser(spec, baseline.config).to(device)
        model.backbone.load_state_dict(baseline.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                      weight_decay=settings.weight_decay)
        train_generator = torch.Generator(device=device).manual_seed(settings.seed + 405)
        fixed_noise_seed = settings.seed + 406
        history = []
        best_epoch = None
        best_rank = None
        for epoch in range(1, settings.epochs + 1):
            model.train()
            weighted_loss = 0.0
            processed = 0
            for clean in train_loader:
                clean = clean.to(device)
                batch = clean.atoms.shape[0]
                times = (settings.train_time_min
                         + (settings.train_time_max - settings.train_time_min)
                         * torch.rand(batch, device=device, generator=train_generator))
                noisy = corrupt_graph(clean, spec, times, train_generator)
                predicted = model(noisy.to_relaxed(spec), times)
                noisy_loss = denoising_reconstruction_losses(
                    predicted, clean, loss_config)["reconstruction"]
                clean_time = torch.zeros(batch, device=device)
                clean_prediction = model(clean.to_relaxed(spec), clean_time)
                clean_loss = denoising_reconstruction_losses(
                    clean_prediction, clean, loss_config)["reconstruction"]
                loss = noisy_loss + loss_config.clean_retention_weight * clean_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
                optimizer.step()
                weighted_loss += float(loss.detach()) * batch
                processed += batch
            evaluation = _evaluate(model, baseline, val_loader, identities, spec,
                                   device, settings.validation_time, fixed_noise_seed,
                                   loss_config)
            clean_evaluation = _evaluate(model, baseline, val_loader, identities, spec,
                                         device, 0.0, fixed_noise_seed, loss_config)
            result = {"epoch": epoch, "train_reconstruction_loss": weighted_loss / processed,
                      "validation": evaluation,
                      "clean_validation": clean_evaluation}
            history.append(result)
            run.log({"stage": "epoch", **result})
            rank = (evaluation["exact_hard"], evaluation["valid_hard"],
                    -evaluation["reconstruction_loss"])
            if best_rank is None or rank > best_rank:
                best_rank, best_epoch = rank, epoch
                torch.save({"format_version": 2, "model_state": model.state_dict(),
                            "epoch": epoch, "config": config,
                            "model_config": asdict(baseline.config),
                            "loss_config": asdict(loss_config),
                            "graph_spec": spec.to_dict()}, run.path / "best.pt")
        write_json(run.path / "history.json", history)
        selected = torch.load(run.path / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(selected["model_state"])
        gate = _gradient_gate(model, baseline, next(iter(val_loader)), spec, device)
        write_json(run.path / "gradient_gate.json", gate)
        chosen = history[best_epoch - 1]
        selected_val = chosen["validation"]
        selected_clean = chosen["clean_validation"]
        repair_gate = (selected_val["valid_hard"] > selected_val["frozen_ae_valid_hard"]
                       and selected_val["exact_hard"] > selected_val["frozen_ae_exact_hard"]
                       and selected_val["unique_valid_hard"] >= 2
                       and selected_clean["valid_hard"] >= selected_clean["frozen_ae_valid_hard"] - 5
                       and selected_clean["exact_hard"] >= selected_clean["frozen_ae_exact_hard"] - 5)
        summary = {"selected_epoch_by_exact_valid_loss": best_epoch,
                   "selected_validation": history[best_epoch - 1]["validation"],
                   "selected_clean_validation": chosen["clean_validation"],
                   "repair_gate_passed": repair_gate,
                   "initial_frozen_ae_validation": {
                       k: v for k, v in history[0]["validation"].items()
                       if k.startswith("frozen_ae_")},
                   "gradient_gate": gate,
                   "best_checkpoint_sha256": file_sha256(run.path / "best.pt"),
                   "claim_limit": "Small AE-initialized denoising autoencoder; corruption repair and local derivatives do not prove useful chemical-prior guidance or native binding."}
        write_json(run.path / "summary.json", summary)
        run.log({"stage": "complete", "selected_epoch": best_epoch,
                 "gradient_gate_passed": gate["passed"]})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True,
                        help="Frozen Stage 1 cohorts.json with unused promotion reserve")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run_pilot(args.checkpoint, args.data, args.cohorts, args.run_root,
                    args.device, DenoiserPilotConfig()))


if __name__ == "__main__":
    main()
