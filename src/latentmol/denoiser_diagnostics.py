"""Post-hoc failure analysis of a frozen denoiser; no training or selection."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry import inspect_graph
from latentmol.data import GraphDataset, collate_graphs
from latentmol.denoiser import SchemaDenoiser, corrupt_graph
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.losses import LossConfig, reconstruction_losses
from latentmol.models import ModelConfig
from latentmol.representation import GraphSpec


@torch.no_grad()
def audit_condition(denoiser, baseline, loader, spec, loss_config,
                    device: str, *, time_value: float, noise_seed: int) -> dict:
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    methods = {name: {"failure_counts": Counter(), "hard_size_histogram": Counter(),
                      "valid": 0, "exact": 0, "active_atom_correct": 0,
                      "active_atom_total": 0, "true_edges": 0,
                      "predicted_edges": 0, "true_positive_edges": 0,
                      "sum_expected_size": 0.0, "sum_reconstruction_loss": 0.0,
                      "sum_atom_ce": 0.0, "sum_bond_ce": 0.0,
                      "examples": []}
               for name in ("frozen_ae", "denoiser")}
    attempted = 0
    for clean in loader:
        clean = clean.to(device)
        batch = clean.atoms.shape[0]
        time = torch.full((batch,), time_value, device=device)
        noisy = corrupt_graph(clean, spec, time, generator)
        graph = noisy.to_relaxed(spec)
        outputs = {"frozen_ae": baseline(graph).decoded,
                   "denoiser": denoiser(graph, time)}
        expected = [inspect_graph(clean.atoms[i].cpu(), clean.charges[i].cpu(),
                                  clean.bonds[i].cpu(), spec).smiles for i in range(batch)]
        upper = torch.ones((spec.max_nodes, spec.max_nodes),
                           device=device, dtype=torch.bool).triu(1)
        for name, output in outputs.items():
            record = methods[name]
            loss = reconstruction_losses(output, clean, loss_config)
            for field in ("reconstruction", "atom_ce", "bond_ce"):
                record["sum_" + ("reconstruction_loss" if field == "reconstruction" else field)] += float(loss[field]) * batch
            hard = output.harden(spec)
            active = clean.atoms != 0
            record["active_atom_correct"] += int(((hard.atoms == clean.atoms) & active).sum())
            record["active_atom_total"] += int(active.sum())
            true_edges = (clean.bonds != 0) & upper
            predicted_edges = (hard.bonds != 0) & upper
            record["true_edges"] += int(true_edges.sum())
            record["predicted_edges"] += int(predicted_edges.sum())
            record["true_positive_edges"] += int((true_edges & predicted_edges).sum())
            record["sum_expected_size"] += float((1.0 - output.atoms.softmax(-1)[..., 0]).sum())
            for i in range(batch):
                observed = inspect_graph(hard.atoms[i].cpu(), hard.charges[i].cpu(),
                                         hard.bonds[i].cpu(), spec)
                size = int((hard.atoms[i] != 0).sum())
                record["hard_size_histogram"][str(size)] += 1
                record["failure_counts"][observed.failure_reason or "valid"] += 1
                record["valid"] += bool(observed.smiles)
                record["exact"] += observed.smiles == expected[i]
                if len(record["examples"]) < 8:
                    record["examples"].append({"target": expected[i],
                                               "decoded": observed.smiles,
                                               "failure": observed.failure_reason,
                                               "target_size": int(active[i].sum()),
                                               "hard_size": size,
                                               "hard_edges": int(predicted_edges[i].sum())})
        attempted += batch
    if attempted == 0:
        raise ValueError("Diagnostic cohort is empty")
    for record in methods.values():
        for key in ("sum_reconstruction_loss", "sum_atom_ce", "sum_bond_ce", "sum_expected_size"):
            record[key.removeprefix("sum_") + "_mean"] = record.pop(key) / attempted
        record["failure_counts"] = dict(record["failure_counts"])
        record["hard_size_histogram"] = dict(record["hard_size_histogram"])
        record["active_atom_accuracy"] = record["active_atom_correct"] / record["active_atom_total"]
        record["bonded_pair_recall"] = (record["true_positive_edges"] / record["true_edges"]
                                         if record["true_edges"] else None)
    return {"time": time_value, "attempted": attempted, "methods": methods}


def run_diagnostic(pilot_dir: Path, dataset_dir: Path, run_root: Path,
                   device: str) -> Path:
    parent = json.loads((pilot_dir / "run.json").read_text())
    if parent.get("name") != "stage3-schema-denoiser-pilot" or parent.get("status") != "completed":
        raise ValueError("Denoiser pilot parent is not completed")
    selected = json.loads((pilot_dir / "selected_indices.json").read_text())
    validation = GraphDataset(dataset_dir, "val")
    if (parent["config"]["dataset_sha256"] != validation.manifest["dataset_sha256"]
            or parent["config"]["split_sha256"] != validation.manifest["split_sha256"]):
        raise ValueError("Diagnostic dataset or split differs from pilot")
    all_validation_indices = selected["validation"]
    index_hash = hashlib.sha256(json.dumps(
        all_validation_indices, separators=(",", ":")).encode()).hexdigest()
    if (index_hash != parent["config"]["validation_indices_sha256"]
            or len(set(all_validation_indices)) != len(all_validation_indices)
            or any(index < 0 or index >= len(validation) for index in all_validation_indices)):
        raise ValueError("Pilot validation indices differ or fall outside the split")
    val_indices = all_validation_indices[:128]
    spec = validation.spec
    best_path = pilot_dir / "best.pt"
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    if (checkpoint["config"]["dataset_sha256"] != validation.manifest["dataset_sha256"]
            or checkpoint["config"]["split_sha256"] != validation.manifest["split_sha256"]):
        raise ValueError("Selected denoiser checkpoint and diagnostic dataset differ")
    pilot_summary = json.loads((pilot_dir / "summary.json").read_text())
    if file_sha256(best_path) != pilot_summary["best_checkpoint_sha256"]:
        raise ValueError("Selected denoiser checkpoint differs from the pilot summary")
    if checkpoint.get("format_version") != 1 or GraphSpec.from_dict(
            checkpoint["graph_spec"]).fingerprint() != spec.fingerprint():
        raise ValueError("Denoiser checkpoint format or schema differs")
    ae_path = Path(parent["config"]["ae_checkpoint"])
    if file_sha256(ae_path) != parent["config"]["ae_checkpoint_sha256"]:
        raise ValueError("Frozen AE checkpoint differs from the pilot manifest")
    baseline, _ = load_model(ae_path, device, spec)
    model = SchemaDenoiser(spec, ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    baseline.eval().requires_grad_(False)
    loader = DataLoader(Subset(validation, val_indices), batch_size=64,
                        shuffle=False, collate_fn=collate_graphs, num_workers=0)
    config = {"parent_run": parent["run_id"],
              "parent_source_archive_sha256": parent["source_archive_sha256"],
              "best_checkpoint_sha256": file_sha256(best_path),
              "dataset_sha256": validation.manifest["dataset_sha256"],
              "validation_indices": val_indices,
              "conditions": ["clean_time0", "fixed_corruption_time0.25"],
              "corruption_note": "CPU seed 423 pairs models within this audit; it does not replay the pilot's CUDA draws with the same integer seed. Time zero lies outside training times [0.10,0.35].",
              "limit": "Failure analysis only; no optimization, retraining or checkpoint selection."}
    with ExperimentRun(run_root, "stage3-denoiser-failure-audit", config, Path.cwd()) as run:
        loss_config = LossConfig(**checkpoint["loss_config"])
        clean = audit_condition(model, baseline, loader, spec, loss_config,
                                device, time_value=0.0, noise_seed=423)
        corrupted = audit_condition(model, baseline, loader, spec, loss_config,
                                    device, time_value=0.25, noise_seed=423)
        write_json(run.path / "summary.json", {"clean": clean, "corrupted": corrupted,
                                               "claim_limit": config["limit"]})
        run.log({"stage": "failure_audit", "clean_valid": clean["methods"]["denoiser"]["valid"],
                 "corrupted_valid": corrupted["methods"]["denoiser"]["valid"],
                 "attempted_per_condition": clean["attempted"]})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(run_diagnostic(args.pilot, args.data, args.run_root, args.device))


if __name__ == "__main__":
    main()
