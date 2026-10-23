"""Stage 1: compare AE/VAE local geometry and inexpensive latent restraints.

All cohorts come from validation and are fixed before bandwidth/radius calibration.
The test split is never read here. Every attempted hard decode is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry import inspect_graph
from latentmol.data import GraphDataset, collate_graphs
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.latent_optimization import optimize_latents
from latentmol.latent_priors import LatentReference


def _encode(model, dataset, indices, device: str, batch_size: int = 64):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size,
                        shuffle=False, collate_fn=collate_graphs)
    codes = []
    with torch.no_grad():
        for hard in loader:
            hard = hard.to(device)
            mean, _ = model.encode(hard.to_relaxed(model.spec))
            codes.append(mean.cpu())
    return torch.cat(codes)


def _decode(model, codes: torch.Tensor, *, batch_size: int = 32):
    """Use the model's conditional-bond hardening, with no chemical repair."""
    outcomes = []
    with torch.no_grad():
        for chunk in codes.split(batch_size):
            decoded = model.decode(chunk)
            hard = decoded.harden(model.spec).to("cpu")
            expected = decoded.probabilities().expected_size.cpu().tolist()
            for row in range(len(chunk)):
                result = inspect_graph(hard.atoms[row], hard.charges[row], hard.bonds[row], model.spec)
                outcomes.append({"smiles": result.smiles, "failure": result.failure_reason,
                                 "heavy_atoms": int((hard.atoms[row] != 0).sum()),
                                 "expected_atoms": float(expected[row])})
    return outcomes


def _cohorts(count: int, seed: int):
    if count < 712:
        raise ValueError("Stage 1 needs at least 712 validation molecules")
    order = np.random.default_rng(seed).permutation(count)
    return {"calibration": order[:256].tolist(), "pilot": order[256:512].tolist(),
            "promotion_reserved": order[512:712].tolist()}


def _directions(count: int, dimensions: int, repeat: int):
    generator = torch.Generator(device="cpu").manual_seed(9107 + repeat)
    directions = torch.randn(count, dimensions, generator=generator)
    return directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _auc(pos: np.ndarray, neg: np.ndarray):
    if len(pos) < 10 or len(neg) < 10:
        return None
    return float(((pos[:, None] > neg[None, :]).mean()
                  + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def _objective(model, reference, kind, anchor, target_size, *, bandwidth,
               threshold, trust_radius):
    """Return a differentiable per-candidate cost in normalized coordinates."""
    def evaluate(position):
        codes = reference.denormalize(position)
        expected_size = model.decode(codes).probabilities().expected_size
        size = 0.2 * (expected_size - target_size).abs().sub(2.0).clamp_min(0).square()
        if kind == "size":
            return size
        if kind == "trust":
            return size + (position - anchor).norm(dim=-1).sub(trust_radius).clamp_min(0).square()
        if kind == "density":
            return size + reference.barrier(codes, bandwidth=bandwidth, threshold=threshold)
        raise ValueError(f"Unknown restraint: {kind}")
    return evaluate


def _attempt_metrics(rows):
    count = len(rows)
    valid = sum(bool(row["endpoint"]["smiles"]) for row in rows)
    changed = sum(bool(row["baseline"]["smiles"]) and bool(row["endpoint"]["smiles"])
                  and row["baseline"]["smiles"] != row["endpoint"]["smiles"] for row in rows)
    eligible = [row for row in rows if row["baseline"]["smiles"] and row["endpoint"]["smiles"]
                and row["baseline"]["smiles"] != row["endpoint"]["smiles"]
                and abs(row["endpoint"]["heavy_atoms"] - row["input_heavy_atoms"]) <= 2]
    rescued = sum(not row["baseline"]["smiles"] and bool(row["endpoint"]["smiles"]) for row in rows)
    unchanged = sum(bool(row["baseline"]["smiles"]) and row["baseline"]["smiles"] == row["endpoint"]["smiles"]
                    for row in rows)
    uniques = {row["endpoint"]["smiles"] for row in rows if row["endpoint"]["smiles"]}
    return {"attempts": count, "valid": valid, "valid_fraction": valid / count,
            "valid_to_different_valid": changed, "valid_to_different_valid_fraction": changed / count,
            "valid_changed_size_compliant": len(eligible),
            "valid_changed_size_compliant_fraction": len(eligible) / count,
            "unique_success_identities": len({row["endpoint"]["smiles"] for row in eligible}),
            "seed_molecules_with_success": len({row["validation_index"] for row in eligible}),
            "invalid_to_valid_rescues": rescued, "unchanged_valid": unchanged,
            "unique_valid": len(uniques)}


def _paired_interval(left, right, field: str, *, seed: int = 20260928, draws: int = 2000):
    """Resample whole validation molecules, keeping radii and repeats together."""
    by_arm = []
    for rows in (left, right):
        grouped = {}
        for row in rows:
            candidate = row["endpoint"]["smiles"]
            baseline = row["baseline"]["smiles"]
            if field == "valid":
                value = bool(candidate)
            elif field == "changed_size_compliant":
                value = bool(candidate and baseline and candidate != baseline
                             and abs(row["endpoint"]["heavy_atoms"] - row["input_heavy_atoms"]) <= 2)
            else:
                raise ValueError("Unsupported paired outcome")
            grouped.setdefault(row["validation_index"], []).append(int(value))
        by_arm.append(grouped)
    if by_arm[0].keys() != by_arm[1].keys():
        raise ValueError("Paired comparison has mismatched seed molecules")
    indices = sorted(by_arm[0])
    if any(len(by_arm[0][i]) != len(by_arm[1][i]) for i in indices):
        raise ValueError("Paired comparison has mismatched attempts per seed")
    differences = np.array([np.mean(by_arm[0][i]) - np.mean(by_arm[1][i]) for i in indices])
    rng = np.random.default_rng(seed)
    resampled = rng.integers(0, len(indices), size=(draws, len(indices)))
    interval = np.quantile(differences[resampled].mean(axis=1), [0.025, 0.975])
    return {"estimate": float(differences.mean()), "seed_clustered_95_interval": interval.tolist(),
            "seed_molecules": len(indices), "attempts_per_arm": len(left)}


def _run_arm(model, reference, cohort_codes, cohort_indices, cohort_smiles, baseline,
             *, radius, repeat, kind, bandwidth, threshold, trust_radius, device):
    direction = _directions(len(cohort_codes), cohort_codes.shape[1], repeat).to(device)
    anchor = reference.normalize(cohort_codes.to(device))
    starts = anchor + radius * direction
    start_decoded = _decode(model, reference.denormalize(starts))
    endpoints = []
    batch_stats = []
    target_size = torch.tensor([Chem.MolFromSmiles(value).GetNumHeavyAtoms()
                                for value in cohort_smiles], device=device, dtype=torch.float32)
    for offset in range(0, len(starts), 16):
        stop = offset + 16
        initial, center = starts[offset:stop], anchor[offset:stop]
        if kind == "perturbed":
            endpoint = initial
        else:
            objective = _objective(model, reference, kind, center, target_size[offset:stop],
                                   bandwidth=bandwidth, threshold=threshold,
                                   trust_radius=trust_radius)
            endpoint, diagnostics = optimize_latents(initial, center, objective,
                                                     steps=10, learning_rate=0.15, max_move=2.0)
            batch_stats.append(diagnostics)
        endpoints.append(endpoint)
    endpoint_codes = reference.denormalize(torch.cat(endpoints))
    endpoint_decoded = start_decoded if kind == "perturbed" else _decode(model, endpoint_codes)
    with torch.no_grad():
        starting_density = reference.log_density(reference.denormalize(starts), bandwidth).cpu().tolist()
        ending_density = reference.log_density(endpoint_codes, bandwidth).cpu().tolist()
        ending_residual = reference.residual(endpoint_codes).cpu().tolist()
    rows = [{"validation_index": int(index), "input": original, "baseline": base,
             "perturbed_start": started, "endpoint": finished,
             "input_heavy_atoms": int(Chem.MolFromSmiles(original).GetNumHeavyAtoms()),
             "radius": radius, "repeat": repeat, "arm": kind,
             "start_log_density": float(density0), "end_log_density": float(density1),
             "end_pca_residual": float(residual)}
            for index, original, base, started, finished, density0, density1, residual in
            zip(cohort_indices, cohort_smiles, baseline, start_decoded, endpoint_decoded,
                starting_density, ending_density, ending_residual)]
    return rows, batch_stats


def _fingerprint(smiles, generator):
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return generator.GetFingerprint(molecule) if molecule is not None else None


def _interpolation(model, reference, indices, smiles, codes, run):
    """Same near/far validation pairs and t values for AE and VAE."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [_fingerprint(value, generator) for value in smiles]
    pairs = []
    for anchor in range(128):
        scores = [(DataStructs.TanimotoSimilarity(fingerprints[anchor], fingerprints[other]), other)
                  for other in range(len(smiles)) if other != anchor]
        nearest = max(scores, key=lambda item: (item[0], -item[1]))
        furthest = min(scores, key=lambda item: (item[0], item[1]))
        pairs.extend((("near", anchor, nearest[1], nearest[0]),
                      ("far", anchor, furthest[1], furthest[0])))
    positions = (0.0, 0.25, 0.5, 0.75, 1.0)
    latent_paths = torch.stack([torch.stack([(1 - t) * codes[a] + t * codes[b] for t in positions])
                                for _, a, b, _ in pairs])
    decoded = _decode(model, latent_paths.flatten(0, 1))
    rows = []
    for p, (label, left, right, input_similarity) in enumerate(pairs):
        outputs = decoded[p * len(positions):(p + 1) * len(positions)]
        left_fp, right_fp = _fingerprint(outputs[0]["smiles"], generator), _fingerprint(outputs[-1]["smiles"], generator)
        point_rows = []
        for t, output in zip(positions, outputs):
            fingerprint = _fingerprint(output["smiles"], generator)
            similarities = ([float(DataStructs.TanimotoSimilarity(fingerprint, side))
                             if fingerprint is not None and side is not None else None
                             for side in (left_fp, right_fp)])
            point_rows.append({"t": t, "decoded": output, "similarity_to_decoded_endpoints": similarities})
        rows.append({"pair_class": label, "left_index": int(indices[left]),
                     "right_index": int(indices[right]), "input_similarity": float(input_similarity),
                     "left_input": smiles[left], "right_input": smiles[right], "points": point_rows})
    write_json(run.path / "interpolations.json", rows)
    summary = {}
    for label in ("near", "far"):
        group = [row for row in rows if row["pair_class"] == label]
        valid = sum(bool(point["decoded"]["smiles"]) for row in group for point in row["points"][1:-1])
        distinct = sum(bool(point["decoded"]["smiles"])
                       and point["decoded"]["smiles"] not in
                       (row["points"][0]["decoded"]["smiles"], row["points"][-1]["decoded"]["smiles"])
                       for row in group for point in row["points"][1:-1])
        bridge = sum(all(point["decoded"]["smiles"] for point in row["points"])
                     and any(point["decoded"]["smiles"] not in
                             (row["points"][0]["decoded"]["smiles"], row["points"][-1]["decoded"]["smiles"])
                             for point in row["points"][1:-1]) for row in group)
        summary[label] = {"paths": len(group), "intermediate_attempts": 3 * len(group),
                          "valid_intermediates": valid, "valid_distinct_intermediates": distinct,
                          "paths_with_all_valid_and_some_distinct": bridge,
                          "mean_input_tanimoto": float(np.mean([row["input_similarity"] for row in group]))}
    return summary


def run_study(checkpoint: Path, data_path: Path, run_root: Path, *, mode: str, device: str):
    train = GraphDataset(data_path, "train")
    validation = GraphDataset(data_path, "val")
    model, saved = load_model(checkpoint, device, train.spec)
    if saved["dataset_sha256"] != train.manifest["dataset_sha256"] or saved["train_config"]["mode"] != mode:
        raise ValueError("Checkpoint does not match dataset or declared training mode")
    model.requires_grad_(False)
    cohort = _cohorts(len(validation), 20260928)
    config = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
              "data_path": str(data_path), "dataset_sha256": train.manifest["dataset_sha256"],
              "split_sha256": train.manifest["split_sha256"], "graph_spec": train.spec.fingerprint(),
              "mode": mode, "device": device, "cohort_seed": 20260928, "cohort": cohort,
              "training_reference_count": 20000, "pca_components": 32,
              "radii": [2.0, 4.0, 6.0], "repeats": [0, 1, 2],
              "arms": ["perturbed", "size", "trust", "density"],
              "update_steps": 10, "update_lr": 0.15, "max_normalized_move": 2.0}
    with ExperimentRun(run_root, f"{mode}-latent-stage1", config, Path.cwd()) as run:
        train_codes = _encode(model, train, list(range(len(train))), device)
        torch.save(train_codes, run.path / "training_means.pt")
        metadata = {key: config[key] for key in
                    ("checkpoint_sha256", "dataset_sha256", "split_sha256", "graph_spec", "mode")}
        metadata.update(training_count=len(train), mean_codes_sha256=file_sha256(run.path / "training_means.pt"))
        reference = LatentReference.fit(train_codes, reference_count=20000, components=32, seed=17).to(device)
        artifact = reference.artifact(metadata)
        torch.save(artifact, run.path / "latent_reference.pt")
        LatentReference.from_artifact(torch.load(run.path / "latent_reference.pt", weights_only=True), metadata)
        run.annotate(reference_sha256=file_sha256(run.path / "latent_reference.pt"),
                     training_means_sha256=metadata["mean_codes_sha256"])
        write_json(run.path / "cohorts.json", cohort)
        encoded = {}
        baselines = {}
        for name in ("calibration", "pilot"):
            encoded[name] = _encode(model, validation, cohort[name], device).to(device)
            baselines[name] = _decode(model, encoded[name])
        calibration = encoded["calibration"]
        calibration_smiles = [validation.smiles[index] for index in cohort["calibration"]]
        # Fixed shared perturbation for calibration; pilot molecules are not read here.
        calibration_start = reference.denormalize(reference.normalize(calibration)
                            + 4.0 * _directions(256, calibration.shape[1], 0).to(device))
        calibration_outcomes = _decode(model, calibration_start)
        best_bandwidth, best_auc = 1.0, -1.0
        candidates = []
        for bandwidth in (0.7, 1.0, 1.4):
            with torch.no_grad():
                densities = reference.log_density(calibration_start, bandwidth).cpu().numpy()
            good = np.array([bool(item["smiles"]) for item in calibration_outcomes])
            auc = _auc(densities[good], densities[~good])
            candidates.append({"bandwidth": bandwidth, "validity_auc": auc})
            if auc is not None and auc > best_auc:
                best_bandwidth, best_auc = bandwidth, auc
        with torch.no_grad():
            baseline_density = reference.log_density(calibration, best_bandwidth).cpu().numpy()
        baseline_valid = np.array([bool(item["smiles"]) for item in baselines["calibration"]])
        if not baseline_valid.any():
            raise RuntimeError("No valid calibration baselines")
        threshold = float(np.quantile(baseline_density[baseline_valid], 0.1))
        trust_candidates = []
        for radius in (1.5, 2.5, 3.5):
            rows, _ = _run_arm(model, reference, calibration, cohort["calibration"],
                               calibration_smiles, baselines["calibration"], radius=4.0,
                               repeat=0, kind="trust", bandwidth=best_bandwidth,
                               threshold=threshold, trust_radius=radius, device=device)
            trust_candidates.append({"trust_radius": radius, **_attempt_metrics(rows)})
        winner = max(trust_candidates, key=lambda item:
                     (item["valid_to_different_valid"], item["valid"], -item["trust_radius"]))
        trust_radius = winner["trust_radius"]
        calibration_report = {"bandwidth_candidates": candidates, "chosen_bandwidth": best_bandwidth,
                              "density_threshold": threshold, "threshold_quantile": 0.1,
                              "trust_candidates": trust_candidates, "chosen_trust_radius": trust_radius,
                              "calibration_start_valid": int(sum(bool(x["smiles"]) for x in calibration_outcomes)),
                              "calibration_count": 256}
        write_json(run.path / "calibration.json", calibration_report)
        run.annotate(calibration=calibration_report)
        pilot_indices = cohort["pilot"]
        pilot_smiles = [validation.smiles[index] for index in pilot_indices]
        interpolation = _interpolation(model, reference, pilot_indices, pilot_smiles,
                                       encoded["pilot"], run)
        with (run.path / "attempts.jsonl").open("w", encoding="utf-8") as output:
            arm_results = []
            all_rows = {name: [] for name in ("perturbed", "size", "trust", "density")}
            for radius in (2.0, 4.0, 6.0):
                for repeat in (0, 1, 2):
                    for kind in ("perturbed", "size", "trust", "density"):
                        rows, diagnostics = _run_arm(
                            model, reference, encoded["pilot"], pilot_indices, pilot_smiles,
                            baselines["pilot"], radius=radius, repeat=repeat, kind=kind,
                            bandwidth=best_bandwidth, threshold=threshold,
                            trust_radius=trust_radius, device=device)
                        for row in rows:
                            output.write(json.dumps(row, allow_nan=False) + "\n")
                        all_rows[kind].extend(rows)
                        metrics = {"stage": "pilot_arm", "radius": radius, "repeat": repeat,
                                   "arm": kind, **_attempt_metrics(rows),
                                   "mean_gradient_norm": float(np.mean([x["mean_gradient_norm"] for x in diagnostics]))
                                                         if diagnostics else None,
                                   "mean_move": float(np.mean([x["mean_displacement_from_start"] for x in diagnostics]))
                                                if diagnostics else 0.0}
                        run.log(metrics)
                        arm_results.append(metrics)
        training_smiles = set(train.smiles)
        overall = {name: {**_attempt_metrics(rows),
                          "novel_valid_candidates": sum(bool(row["endpoint"]["smiles"])
                              and row["endpoint"]["smiles"] not in training_smiles
                              for row in rows)}
                   for name, rows in all_rows.items()}
        comparisons = {f"density_minus_{control}": {
            "valid_changed_size_compliant": _paired_interval(
                all_rows["density"], all_rows[control], "changed_size_compliant"),
            "validity": _paired_interval(all_rows["density"], all_rows[control], "valid")}
            for control in ("trust", "size", "perturbed")}
        density_count = overall["density"]["valid_to_different_valid"]
        density_diversity = (overall["density"]["unique_success_identities"] /
                             max(1, overall["density"]["seed_molecules_with_success"]))
        density_size_compliance = overall["density"]["valid_changed_size_compliant"] / max(1, density_count)
        density_promotion_gate = (all(
            comparisons[f"density_minus_{control}"]["valid_changed_size_compliant"]
            ["seed_clustered_95_interval"][0] > 0 and
            comparisons[f"density_minus_{control}"]["validity"]
            ["seed_clustered_95_interval"][0] > -0.02
            for control in ("trust", "size"))
            and density_diversity >= 0.8 and density_size_compliance >= 0.9)
        summary = {"mode": mode, "checkpoint_epoch": saved["epoch"],
                   "checkpoint_sha256": config["checkpoint_sha256"],
                   "dataset_sha256": config["dataset_sha256"], "cohorts": {k: len(v) for k,v in cohort.items()},
                   "reference": {"train_codes": len(train_codes), "kde_references": len(reference.references),
                                 "pca_dimensions": 32,
                                 "pca_variance_retained": float(reference.variance_retained.cpu()),
                                 "residual_95th": float(reference.residual_limit.cpu())},
                   "calibration": calibration_report, "interpolation": interpolation,
                   "pilot_arms": arm_results, "pilot_overall": overall,
                   "paired_comparisons": comparisons,
                   "density_promotion_gate": density_promotion_gate,
                   "density_promotion_rule": "seed-clustered 95% lower CI of valid, changed, hard-size-compliant gain over both trust and size > 0; validity lower bound > -2 percentage points for both; >=0.8 distinct successful identities per successful seed and >=0.9 hard-size compliance among changed valid outputs",
                   "density_success_identity_coverage": density_diversity,
                   "density_success_size_compliance": density_size_compliance,
                   "limitation": "Validation development cohorts, one model-training seed; density is not chemical validity."}
        write_json(run.path / "summary.json", summary)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run_study(args.checkpoint, args.data, args.run_root, mode=args.mode, device=args.device))


if __name__ == "__main__":
    main()
