"""Discrete reconstruction and latent-neighborhood evaluation without graph repair.

Validity always means a nonempty, connected, RDKit-sanitizable graph. It does not
claim synthesis feasibility or binding. Test data are evaluated only on explicit
request; training selects checkpoints using the validation split.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from latentmol.chemistry import decode_graph, inspect_graph
from latentmol.data import GraphDataset, collate_graphs
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.losses import LossConfig, reconstruction_losses
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.sampling import subset_indices
from latentmol.quality import GradedEvaluator, QualityScoreConfig, summarize_quality


def binomial_interval(successes: int, count: int):
    """Wilson 95% interval, with all attempted decodes in the denominator."""
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** .5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_model(path: Path, device, expected_spec: GraphSpec | None = None):
    """Load only this project's tensor/plain-data checkpoint format."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format version")
    spec = GraphSpec.from_dict(checkpoint["graph_spec"])
    if expected_spec is not None and spec.fingerprint() != expected_spec.fingerprint():
        raise ValueError("Checkpoint and dataset graph schemas differ")
    model = GraphAutoencoder(spec, ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _fraction(numerator: int | float, denominator: int) -> float | None:
    """Leave metrics with no eligible observations undefined, rather than perfect."""
    return numerator / denominator if denominator else None


def _valid_smiles(value: str | None) -> bool:
    """Treat only a nonempty canonical decode as valid; no empty-molecule credit."""
    return isinstance(value, str) and bool(value.strip())


def summarize_neighborhood(seeds, baselines, decoded, training_smiles=None):
    """Compare perturbations with their own unperturbed decodes and true inputs.

    ``valid_changed_from_baseline_fraction`` divides by all sampled molecules and
    includes invalid-to-valid rescues. The separate rescue fraction conditions on
    invalid baselines. Different-from-input includes existing reconstruction
    errors, so it is explicitly not a measure of an effect caused by perturbation.
    Canonical strings must already have been produced by the chemistry decoder.
    """
    if not (len(seeds) == len(baselines) == len(decoded)) or not seeds:
        raise ValueError("Neighborhood lists must be nonempty and have equal lengths")
    valid = [value for value in decoded if _valid_smiles(value)]
    baseline_valid = sum(_valid_smiles(value) for value in baselines)
    baseline_invalid = len(seeds) - baseline_valid
    baseline_exact = sum(_valid_smiles(base) and base == original
                         for original, base in zip(seeds, baselines))
    different_from_input = sum(_valid_smiles(value) and value != original
                               for original, value in zip(seeds, decoded))
    changed_from_baseline = sum(_valid_smiles(value) and value != base
                                for base, value in zip(baselines, decoded))
    rescued = sum(not _valid_smiles(base) and _valid_smiles(value)
                  for base, value in zip(baselines, decoded))
    results = {
        "samples": len(seeds), "valid_count": len(valid), "validity": len(valid) / len(seeds),
        "unique_valid_count": len(set(valid)),
        "unique_valid_fraction": _fraction(len(set(valid)), len(valid)),
        "different_from_input_valid_count": different_from_input,
        "different_from_input_valid_fraction": different_from_input / len(seeds),
        "valid_changed_from_baseline_count": changed_from_baseline,
        "valid_changed_from_baseline_fraction": changed_from_baseline / len(seeds),
        "baseline_valid_count": baseline_valid, "baseline_validity": baseline_valid / len(seeds),
        "baseline_exact_reconstruction_count": baseline_exact,
        "baseline_exact_reconstruction_fraction": baseline_exact / len(seeds),
        "baseline_invalid_count": baseline_invalid,
        "invalid_to_valid_rescue_count": rescued,
        "invalid_to_valid_rescue_fraction": _fraction(rescued, baseline_invalid),
    }
    if training_smiles is not None:
        novel = sum(value not in training_smiles for value in valid)
        results.update(novel_vs_training_split_count=novel,
                       novel_vs_training_split_fraction=_fraction(novel, len(valid)))
    return results


@torch.no_grad()
def reconstruction_metrics(model, loader, loss_config: LossConfig, device, chemistry_limit: int = 256,
                           training_mode: str | None = None, active_variance_threshold: float = 0.01,
                           training_smiles=None, quality_config: QualityScoreConfig | None = None,
                           graded: bool = False, retain_graphs: bool = False, *,
                           from_canonical_cache: bool = False):
    """Evaluate deterministic posterior-mean reconstruction, not posterior samples.

    Bond precision includes every predicted edge, including bonds to extra atoms.
    Charge correctness requires a predicted atom at the target's occupied slot.
    Active latent dimensions have population variance of mu across this evaluation
    set strictly greater than 0.01 by default; this is a diagnostic convention, not
    a universal threshold. KL/std have posterior meaning only for a trained VAE;
    an AE's unused log-variance head is deliberately not interpreted as a posterior.
    """
    if chemistry_limit < 0 or active_variance_threshold < 0:
        raise ValueError("Chemistry limit and active variance threshold must be nonnegative")
    if training_mode not in (None, "ae", "vae"):
        raise ValueError("training_mode must be ae, vae, or unspecified")
    totals, count = {}, 0
    atom_correct = atom_total = charge_correct = edge_correct = edge_total = 0
    predicted_edges_total = true_positive_edges = 0
    chemistry_count = valid = exact = 0
    size_error = 0.0
    mu_sum = mu_square_sum = kl_by_dimension = None
    kl_sum = posterior_std_sum = 0.0
    examples = []
    failures = {}
    quality_records = []
    quality_evaluator = GradedEvaluator(model.spec, quality_config) if graded else None
    saved_failed_graphs = 0
    size_groups = {}
    atom_entropy_sum = bond_entropy_sum = 0.0
    entropy_pairs = 0
    bond_classes = len(model.spec.bond_names)
    bond_confusion = torch.zeros((bond_classes, bond_classes), dtype=torch.int64)
    forward_options = {"from_canonical_cache": True} if from_canonical_cache else {}
    for target in loader:
        target = target.to(device)
        graph = target if from_canonical_cache else target.to_relaxed(model.spec)
        output = model(graph, sample_latent=False, **forward_options)
        losses = reconstruction_losses(output.decoded, target, loss_config)
        batch_size = target.atoms.shape[0]
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
        hard = output.decoded.harden(model.spec)
        active = target.atoms != 0
        atom_logp = output.decoded.atoms.log_softmax(-1)
        atom_entropy_sum += float((-(atom_logp.exp() * atom_logp).sum(-1) * active).sum())
        atom_correct += int(((hard.atoms == target.atoms) & active).sum())
        atom_total += int(active.sum())
        charge_correct += int(((hard.charges == target.charges) & active & (hard.atoms != 0)).sum())
        upper = torch.ones_like(target.bonds, dtype=torch.bool).triu(1)
        present_pairs = upper & active[:, :, None] & active[:, None, :]
        bond_logp = output.decoded.symmetric_bond_logits.log_softmax(-1)
        bond_entropy_sum += float((-(bond_logp.exp() * bond_logp).sum(-1) * present_pairs).sum())
        entropy_pairs += int(present_pairs.sum())
        pair_codes = target.bonds[present_pairs] * bond_classes + hard.bonds[present_pairs]
        bond_confusion += torch.bincount(pair_codes.cpu(), minlength=bond_classes ** 2).reshape(bond_classes, bond_classes)
        true_edges = (target.bonds != 0) & upper
        predicted_edges = (hard.bonds != 0) & upper
        edge_correct += int(((hard.bonds == target.bonds) & true_edges).sum())
        edge_total += int(true_edges.sum())
        predicted_edges_total += int(predicted_edges.sum())
        true_positive_edges += int((true_edges & predicted_edges).sum())
        size_error += float(((hard.atoms != 0).sum(-1) - active.sum(-1)).abs().sum())
        # Accumulate moments on CPU in double precision without retaining graphs.
        mu = output.mu.detach().to(device="cpu", dtype=torch.float64)
        if mu_sum is None:
            mu_sum, mu_square_sum = mu.sum(0), mu.square().sum(0)
        else:
            mu_sum += mu.sum(0)
            mu_square_sum += mu.square().sum(0)
        if training_mode == "vae":
            logvar = output.logvar.detach().to(device="cpu", dtype=torch.float64)
            batch_kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(0)
            kl_sum += float(batch_kl.sum())
            kl_by_dimension = batch_kl if kl_by_dimension is None else kl_by_dimension + batch_kl
            posterior_std_sum += float((0.5 * logvar).exp().sum())
        for i in range(min(batch_size, max(0, chemistry_limit - chemistry_count))):
            expected = decode_graph(target.atoms[i], target.charges[i], target.bonds[i], model.spec)
            if graded:
                diagnostic = quality_evaluator.diagnose(hard.atoms[i], hard.charges[i], hard.bonds[i])
                quality_records.append(diagnostic)
                observed, reason, detail = (diagnostic[key] for key in ("smiles", "failure_reason", "failure_detail"))
            else:
                inspected = inspect_graph(hard.atoms[i], hard.charges[i], hard.bonds[i], model.spec)
                observed, reason, detail = inspected.smiles, inspected.failure_reason, inspected.detail
            target_size = int(active[i].sum())
            predicted_size = int((hard.atoms[i] != 0).sum())
            lower = ((target_size - 1) // 10) * 10 + 1
            group = size_groups.setdefault(f"{lower}-{lower + 9}", {"samples": 0, "valid": 0, "exact": 0})
            group["samples"] += 1
            group["valid"] += int(_valid_smiles(observed))
            group["exact"] += int(_valid_smiles(observed) and observed == expected)
            chemistry_count += 1
            valid += _valid_smiles(observed)
            exact += _valid_smiles(observed) and observed == expected
            example = {"target": expected, "decoded": observed,
                       "target_heavy_atoms": target_size, "decoded_heavy_atoms": predicted_size,
                       "failure_reason": reason, "failure_detail": detail}
            if graded:
                example["graded_chemistry"] = diagnostic
            # Training callers discard examples: retain the original bounded
            # failure sample there. Standalone evaluation explicitly keeps all.
            if retain_graphs or (reason and saved_failed_graphs < 8):
                example["predicted_graph"] = {"atoms": hard.atoms[i].detach().cpu().tolist(),
                                              "charges": hard.charges[i].detach().cpu().tolist(),
                                              "bonds": hard.bonds[i].detach().cpu().tolist()}
                saved_failed_graphs += int(reason is not None)
            if reason:
                failure_key = reason + (":" + detail if detail else "")
                failures[failure_key] = failures.get(failure_key, 0) + 1
            examples.append(example)
    if count == 0:
        raise ValueError("Evaluation split is empty")
    metrics = {key: value / count for key, value in totals.items()}
    mu_variance = (mu_square_sum / count - (mu_sum / count).square()).clamp_min(0)
    latent_dimensions = int(mu_variance.numel())
    metrics.update(
        reconstruction_latent="deterministic_mu", molecules=count,
        active_atom_correct=atom_correct, active_atom_targets=atom_total,
        active_atom_accuracy=_fraction(atom_correct, atom_total),
        active_atom_charge_correct=charge_correct,
        active_atom_charge_accuracy=_fraction(charge_correct, atom_total),
        target_bonded_pairs=edge_total, predicted_bonded_pairs=predicted_edges_total,
        correct_bonded_pairs=true_positive_edges, correct_bonded_pair_types=edge_correct,
        bonded_pair_precision=_fraction(true_positive_edges, predicted_edges_total),
        bonded_pair_recall=_fraction(true_positive_edges, edge_total),
        bonded_pair_type_precision=_fraction(edge_correct, predicted_edges_total),
        bonded_pair_type_recall=_fraction(edge_correct, edge_total),
        discrete_size_mae=size_error / count, chemistry_samples=chemistry_count,
        valid_chemistry_count=valid, exact_reconstruction_count=exact,
        chemistry_failure_counts=failures,
        validity=_fraction(valid, chemistry_count), exact_reconstruction=_fraction(exact, chemistry_count),
        validity_wilson95=binomial_interval(valid, chemistry_count),
        exact_reconstruction_wilson95=binomial_interval(exact, chemistry_count),
        chemistry_by_target_size=size_groups,
        active_atom_entropy_mean=_fraction(atom_entropy_sum, atom_total),
        target_present_pair_bond_entropy_mean=_fraction(bond_entropy_sum, entropy_pairs),
        bond_confusion_target_present_pairs={"labels": list(model.spec.bond_names),
                                            "rows_target_columns_prediction": bond_confusion.tolist()},
        latent_diagnostics={
            "samples": count, "training_mode": training_mode or "unspecified",
            "posterior_statistics_applicable": training_mode == "vae",
            "posterior_kl_mean": kl_sum / count if training_mode == "vae" else None,
            "posterior_std_mean": posterior_std_sum / (count * latent_dimensions) if training_mode == "vae" else None,
            "posterior_kl_by_dimension": (kl_by_dimension / count).tolist() if training_mode == "vae" else None,
            "posterior_note": "Gaussian KL to N(0,I), summed over dimensions, averaged over molecules"
                              if training_mode == "vae" else "AE/unspecified log-variance head is not a trained posterior",
            "mu_variance_mean": float(mu_variance.mean()),
            "mu_variance_by_dimension": mu_variance.tolist(),
            "mu_variance_estimator": "population variance across evaluated molecules",
            "latent_dimensions": latent_dimensions,
            "active_dimensions": int((mu_variance > active_variance_threshold).sum()),
            "active_variance_threshold": active_variance_threshold,
        },
    )
    if training_smiles is not None:
        novel = sum(_valid_smiles(item["decoded"]) and item["decoded"] not in training_smiles
                    for item in examples)
        metrics.update(novel_vs_training_split_count=novel,
                       novel_vs_training_split_fraction=_fraction(novel, valid))
    if graded:
        metrics["graded_chemistry"] = summarize_quality(quality_records, quality_config)
        metrics["graded_chemistry"]["compute"] = quality_evaluator.compute()
    return metrics, examples


@torch.no_grad()
def _decode_latent_batches(model, codes, device, batch_size: int = 32, *, diagnostics=False,
                          quality_evaluator=None):
    """Decode at most 32 graphs simultaneously, and do RDKit work on CPU tensors."""
    if not 1 <= batch_size <= 32:
        raise ValueError("Latent decoding batch_size must be between 1 and 32")
    smiles, records = [], []
    quality_evaluator = quality_evaluator or (GradedEvaluator(model.spec) if diagnostics else None)
    for chunk in codes.split(batch_size):
        hard = model.decode(chunk.to(device)).harden(model.spec).to("cpu")
        for i in range(len(chunk)):
            if diagnostics:
                record = quality_evaluator.diagnose(hard.atoms[i], hard.charges[i], hard.bonds[i])
                record["predicted_graph"] = {"atoms": hard.atoms[i].tolist(),
                                             "charges": hard.charges[i].tolist(),
                                             "bonds": hard.bonds[i].tolist()}
                records.append(record)
                smiles.append(record["smiles"])
            else:
                smiles.append(decode_graph(hard.atoms[i], hard.charges[i], hard.bonds[i], model.spec))
    return (smiles, records) if diagnostics else smiles


@torch.no_grad()
def neighborhood_metrics(model, loader, device, seed: int, scales=(0.0, 0.05, 0.1, 0.2), limit=128,
                         decode_batch_size=32, training_smiles=None,
                         quality_config: QualityScoreConfig | None = None):
    """Perturb encoded molecules using relative latent scale for AE/VAE comparability.

    RMS latent scale is measured on the provided seed molecules. Each arm gets
    the same seeded noise directions. Changes are compared to each molecule's
    own unperturbed decode, including when scale zero is not a requested arm.
    CPU-held codes/noise and bounded decoder batches avoid a limit-sized GPU
    decoder allocation. This measures local variation, not property improvement.
    """
    if limit < 1:
        raise ValueError("Neighborhood limit must be positive")
    codes, seeds = [], []
    for target in loader:
        target = target.to(device)
        mu, _ = model.encode(target.to_relaxed(model.spec))
        keep = min(limit - len(seeds), len(mu))
        codes.append(mu[:keep].detach().cpu())
        for i in range(keep):
            seeds.append(decode_graph(target.atoms[i], target.charges[i], target.bonds[i], model.spec))
        if len(seeds) >= limit:
            break
    if not seeds:
        raise ValueError("Neighborhood evaluation split is empty")
    z = torch.cat(codes)
    measured_spread = float(z.double().std(dim=0, unbiased=False).square().mean().sqrt())
    spread = max(measured_spread, 1e-3)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(z.shape, generator=generator, dtype=z.dtype)
    quality_evaluator = GradedEvaluator(model.spec, quality_config)
    baseline, baseline_records = _decode_latent_batches(model, z, device, decode_batch_size, diagnostics=True,
                                                       quality_evaluator=quality_evaluator)
    results, examples = [], []
    for scale in scales:
        smiles, records = (baseline, baseline_records) if float(scale) == 0.0 else _decode_latent_batches(
            model, z + float(scale) * spread * noise, device, decode_batch_size, diagnostics=True,
            quality_evaluator=quality_evaluator)
        results.append({"relative_scale": scale, "latent_rms_std": measured_spread,
                        "perturbation_reference_std": spread,
                        "graded_chemistry": summarize_quality(records, quality_config),
                        "graded_compute_cumulative": quality_evaluator.compute(),
                        **summarize_neighborhood(seeds, baseline, smiles, training_smiles)})
        examples.extend({"scale": scale, "seed": original, "baseline_decoded": base, "decoded": value,
                         "graded_chemistry": record}
                        for original, base, value, record in zip(seeds, baseline, smiles, records))
    return results, examples


@torch.no_grad()
def prior_metrics(model, device, seed: int, limit: int = 256, training_smiles=None,
                  quality_config: QualityScoreConfig | None = None):
    """Sample the VAE's declared N(0,I) prior without encoding any input graph.

    This differs from posterior-mean reconstruction and local perturbation.
    Invalid and duplicate draws remain in the attempt denominator. Call only
    for a VAE checkpoint; an ordinary AE has no trained standard-normal prior.
    """
    if limit < 1:
        raise ValueError("Prior sample count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randn(limit, model.config.latent, generator=generator)
    quality_evaluator = GradedEvaluator(model.spec, quality_config)
    decoded, records = _decode_latent_batches(model, codes, device, diagnostics=True,
                                            quality_evaluator=quality_evaluator)
    valid = [value for value in decoded if _valid_smiles(value)]
    metrics = {"distribution": "standard_normal", "seed": seed, "samples": limit,
               "valid_count": len(valid), "validity": len(valid) / limit,
               "unique_valid_count": len(set(valid)),
               "unique_valid_fraction": _fraction(len(set(valid)), len(valid)),
               "graded_chemistry": summarize_quality(records, quality_config),
               "graded_compute": quality_evaluator.compute()}
    if training_smiles is not None:
        novel = sum(value not in training_smiles for value in valid)
        metrics.update(novel_vs_training_split_count=novel,
                       novel_vs_training_split_fraction=_fraction(novel, len(valid)))
    return metrics, [{"sample": index, "decoded": value, "graded_chemistry": records[index]}
                     for index, value in enumerate(decoded)]


def evaluate_checkpoint(checkpoint_path: Path, data_path: Path, run_root: Path, split: str = "val",
                        limit: int = 256, seed: int = 17, device: str = "cuda",
                        novelty: bool = False, *, chemistry_limit: int | None = None,
                        neighborhood_limit: int | None = None, prior_limit: int | None = None,
                        shuffle_full: bool = False, batch_size: int = 32,
                        quality_config: QualityScoreConfig | None = None) -> Path:
    """Evaluate one checkpoint in this process and return its recorded run directory.

    Keeping training and evaluation in the same process reuses its CUDA context,
    including on GPUs configured for exclusive-process access. Every invocation
    retains an independent ExperimentRun with source/configuration manifests,
    metrics, artifacts, and traceback capture if evaluation fails.
    """
    checkpoint_path, data_path, run_root = Path(checkpoint_path), Path(data_path), Path(run_root)
    chemistry_limit = limit if chemistry_limit is None else chemistry_limit
    neighborhood_limit = limit if neighborhood_limit is None else neighborhood_limit
    prior_limit = limit if prior_limit is None else prior_limit
    if min(limit, chemistry_limit, neighborhood_limit, prior_limit, batch_size) < 1:
        raise ValueError("Evaluation budgets must be positive")
    quality_config = quality_config or QualityScoreConfig()
    config = {"checkpoint": str(checkpoint_path), "data": str(data_path), "run_root": str(run_root),
              "split": split, "limit": limit, "seed": seed, "device": device, "novelty": novelty,
              "chemistry_limit": chemistry_limit, "neighborhood_limit": neighborhood_limit,
              "prior_limit": prior_limit, "shuffle_full": shuffle_full, "batch_size": batch_size,
              "quality_weights": asdict(quality_config)}
    with ExperimentRun(run_root, "evaluation", config, Path.cwd()) as run:
        data = GraphDataset(data_path, split)
        model, checkpoint = load_model(checkpoint_path, device, data.spec)
        # A checkpoint must be evaluated against the same prepared dataset unless
        # an explicit external-generalization workflow is added in the future.
        if checkpoint["dataset_sha256"] != data.manifest["dataset_sha256"]:
            raise ValueError("Checkpoint and evaluation dataset hashes differ")
        indices = subset_indices(len(data), limit, seed, shuffle_full=shuffle_full)
        subset = Subset(data, indices)
        loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_graphs)
        losses = LossConfig(**checkpoint["loss_config"])
        training = checkpoint.get("train_config", {})
        mode = training.get("mode")
        training_smiles = None
        if novelty:
            with np.load(data_path / "dataset.npz", allow_pickle=False) as archive:
                training_smiles = set(archive["smiles"][archive["split"] == 0].tolist())
        metrics, examples = reconstruction_metrics(model, loader, losses, device, chemistry_limit,
                                                   training_mode=mode, training_smiles=training_smiles,
                                                   quality_config=quality_config, graded=True, retain_graphs=True)
        neighborhood, neighbors = neighborhood_metrics(model, loader, device, seed,
                                                        limit=neighborhood_limit, training_smiles=training_smiles,
                                                        quality_config=quality_config)
        prior = None
        if mode == "vae":
            prior, prior_examples = prior_metrics(model, device, seed, prior_limit, training_smiles,
                                                 quality_config=quality_config)
            write_json(run.path / "prior_examples.json", prior_examples)
        beta_target = training.get("beta")
        effective_beta = checkpoint.get("beta_at_checkpoint", checkpoint.get("beta"))
        beta_source = "saved checkpoint" if effective_beta is not None else "unavailable"
        if effective_beta is None and mode == "ae":
            effective_beta, beta_source = 0.0, "AE objective has no KL term"
        elif effective_beta is None and mode == "vae" and beta_target is not None and "epoch" in checkpoint:
            effective_beta = beta_target * min(checkpoint["epoch"] / max(training.get("kl_warmup_epochs", 1), 1), 1.0)
            beta_source = "reconstructed from saved epoch and training warmup configuration"
        evaluation_metadata = {
            "checkpoint_training_mode": mode, "reconstruction_latent": "deterministic_mu",
            "checkpoint_beta_target": beta_target, "checkpoint_effective_beta": effective_beta,
            "checkpoint_beta_source": beta_source,
            "novelty_reference": "full prepared training split, possibly larger than the trained subset"
                                 if novelty else None,
        }
        run.annotate(checkpoint_sha256=file_sha256(checkpoint_path), dataset_manifest=data.manifest,
                     evaluation_indices=indices, evaluation_metadata=evaluation_metadata)
        run.log({"stage": "reconstruction", **metrics})
        for item in neighborhood:
            run.log({"stage": "neighborhood", **item})
        if prior is not None:
            run.log({"stage": "prior", **prior})
        write_json(run.path / "examples.json", examples)
        write_json(run.path / "neighbors.json", neighbors)
        write_json(run.path / "summary.json", {"metadata": evaluation_metadata,
                                               "reconstruction": metrics, "neighborhood": neighborhood,
                                               "prior": prior})
    return run.path


def main():
    """Keep the standalone CLI as a thin wrapper over the reusable evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--novelty", action="store_true",
                        help="Compare valid decodes to the full prepared training split")
    parser.add_argument("--chemistry-limit", type=int)
    parser.add_argument("--neighborhood-limit", type=int)
    parser.add_argument("--prior-limit", type=int)
    parser.add_argument("--shuffle-full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quality-config", type=Path, help="JSON QualityScoreConfig weights")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.data, args.run_root, args.split, args.limit,
                        args.seed, args.device, args.novelty, chemistry_limit=args.chemistry_limit,
                        neighborhood_limit=args.neighborhood_limit, prior_limit=args.prior_limit,
                        shuffle_full=args.shuffle_full, batch_size=args.batch_size,
                        quality_config=QualityScoreConfig.from_dict(json.loads(args.quality_config.read_text()))
                        if args.quality_config else None)


if __name__ == "__main__":
    main()
