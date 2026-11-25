"""Paired, unrepaired checkpoint evaluation; never trains or selects valid draws.

The same standard-normal latents, categorical RNG stream, held-out molecules
and perturbation directions are reused across checkpoints. Posterior neighbors
use each checkpoint's own mean/std, so their physical displacements are logged.
Raw invalid graphs are retained. Chemical and representation failures remain
separate, using the existing native chemistry classifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_epoch45_generation import structural_diversity
from latentmol.datasets.data import GraphDataset, collate_graphs
from latentmol.evaluation.evaluate import binomial_interval, load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import DecoderOutput
from latentmol.sampling import sample_hard_graph, subset_indices
from latentmol.train import state_digest
from latentmol.training.losses import LossConfig, reconstruction_losses


def select_indices(size, count, seed, excluded=()):
    """A fresh selection without replacement; never silently shrink the budget."""
    available = np.array(sorted(set(range(size)) - set(excluded)), dtype=np.int64)
    if type(count) is not int or not 0 < count <= len(available):
        raise ValueError("Insufficient distinct held-out rows for the declared panel")
    return np.random.default_rng(seed).choice(available, count, replace=False).tolist()


def paired_difference(before, after):
    """Paired proportion change with a large-sample interval, not independent CIs."""
    if not before or len(before) != len(after) or any(type(v) is not bool for v in [*before, *after]):
        raise ValueError("Paired outcomes need equal nonempty panels")
    differences = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    if not np.isin(differences, (-1., 0., 1.)).all():
        raise ValueError("Paired outcomes must be Boolean")
    delta = float(differences.mean())
    se = float(differences.std(ddof=1) / math.sqrt(len(differences))) if len(differences) > 1 else None
    return dict(attempts=len(differences), difference=delta,
        gained=int((differences == 1).sum()), lost=int((differences == -1).sum()),
        paired_normal95=[max(-1., delta - 1.959964 * se), min(1., delta + 1.959964 * se)]
                        if se is not None else None,
        interval_scope="Approximate paired interval across independent latent/molecule units; no multiplicity correction")


def unit_rms_directions(count, dimensions, seed):
    """CPU common random directions; standardized RMS is exactly one per row."""
    value = torch.randn(count, dimensions, generator=torch.Generator().manual_seed(seed))
    return value / value.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)


def _components(atoms, bonds):
    """Raw adjacency connectivity, including graphs that fail sanitization."""
    unseen = set((atoms != 0).nonzero().flatten().tolist())
    count = 0
    while unseen:
        count += 1
        stack = [unseen.pop()]
        while stack:
            i = stack.pop()
            neighbors = {j for j in unseen if int(bonds[i, j]) != 0}
            unseen -= neighbors
            stack.extend(neighbors)
    return count


def graph_record(hard, i, spec, **identity):
    """Save all labels, including invalid graphs and bonds to absent slots."""
    atoms, charges, bonds = (getattr(hard, name)[i] for name in ("atoms", "charges", "bonds"))
    assessment = classify_repair_graph(atoms, charges, bonds, spec)
    if assessment["failure_kind"] == "input":
        raise ValueError("Decoder produced incompatible hard graph: " + str(assessment))
    active = atoms != 0
    return dict(**identity, valid=assessment["smiles_parse_valid"] is True,
        strict_valid=assessment["strict_teacher_valid"], smiles=assessment["smiles"],
        classification=assessment, atoms=int(active.sum()), components=_components(atoms, bonds),
        elements=dict(Counter(str(spec.atomic_numbers[label]) for label in atoms[active].tolist())),
        formal_charges=dict(Counter(str(spec.formal_charges[label]) for label in charges[active].tolist())),
        graph=dict(atoms=atoms.tolist(), charges=charges.tolist(),
            edges=[[int(a), int(b), int(bonds[a, b])] for a, b in (bonds != 0).triu(1).nonzero().tolist()]))


def summarize_rows(rows):
    """Every attempted graph remains in validity/yield/composition denominators."""
    count = len(rows)
    if not count:
        raise ValueError("Cannot summarize an empty panel")
    valid = sum(r["valid"] for r in rows)
    elements, reasons, kinds = Counter(), Counter(), Counter()
    for row in rows:
        elements.update(row["elements"])
        assessment = row["classification"]
        kinds[assessment["failure_kind"]] += 1
        if assessment["failure_kind"] != "none":
            reasons[assessment["failure_kind"] + ":" + str(assessment["reason"])] += 1
    atom_total = sum(elements.values())
    strict = sum(r["strict_valid"] for r in rows)
    result = dict(attempts=count, valid_count=valid, validity=valid / count,
        validity_wilson95=binomial_interval(valid, count), strict_valid_count=strict,
        strict_validity=strict / count, strict_validity_wilson95=binomial_interval(strict, count),
        failure_kinds=dict(kinds), failure_reasons=dict(reasons),
        raw_connected_fraction=sum(r["components"] == 1 for r in rows) / count,
        empty_count=sum(r["atoms"] == 0 for r in rows),
        component_histogram=dict(Counter(str(r["components"]) for r in rows)),
        atom_count_histogram=dict(Counter(str(r["atoms"]) for r in rows)),
        mean_atom_count=sum(r["atoms"] for r in rows) / count,
        all_attempt_element_counts=dict(elements),
        sulfur_atom_fraction=elements.get("16", 0) / atom_total if atom_total else None,
        sulfur_molecule_fraction=sum(r["elements"].get("16", 0) > 0 for r in rows) / count,
        structural_diversity=structural_diversity(rows))
    result["unique_valid_yield"] = result["structural_diversity"]["unique_valid"] / count
    if "exact" in rows[0]:
        exact = sum(r["exact"] for r in rows)
        result.update(exact_count=exact, exact_reconstruction=exact / count,
                      exact_wilson95=binomial_interval(exact, count))
    return result


def _checked_decode(model, codes):
    device = next(model.parameters()).device
    if not bool(torch.isfinite(codes).all()):
        raise FloatingPointError("Nonfinite latent inputs")
    value = model.decode(codes.to(device))
    if any(not bool(torch.isfinite(getattr(value, field)).all()) for field in ("atoms", "charges", "bonds")):
        raise FloatingPointError("Nonfinite decoder outputs")
    batch, nodes = len(codes), model.spec.max_nodes
    expected = ((batch, nodes, len(model.spec.atomic_numbers)),
                (batch, nodes, len(model.spec.formal_charges)),
                (batch, nodes, nodes, len(model.spec.bond_names)))
    if tuple(tuple(getattr(value, name).shape) for name in ("atoms", "charges", "bonds")) != expected:
        raise ValueError("Incompatible decoder output shapes")
    return DecoderOutput(*(getattr(value, name).cpu() for name in ("atoms", "charges", "bonds")))


def _compact(rows):
    """Raw labels are already saved in gzip; comparison retains scalar metadata."""
    return [{k: v for k, v in r.items() if k != "graph"} for r in rows]


@torch.inference_mode()
def decode_panel(model, codes, output, *, batch_size, categorical_seed=None, temperature=1.):
    rows = {"argmax": [], "categorical": []}
    generator = torch.Generator().manual_seed(categorical_seed) if categorical_seed is not None else None
    expected_atoms = entropy = 0.
    expected_elements = torch.zeros(len(model.spec.atomic_numbers), dtype=torch.float64)
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for start in range(0, len(codes), batch_size):
            decoded = _checked_decode(model, codes[start:start + batch_size])
            logp = (decoded.atoms / temperature).log_softmax(-1)
            prob = logp.exp()
            expected_atoms += float((1 - prob[..., 0]).sum())
            expected_elements += prob.double().sum((0, 1))
            entropy += float(-(prob * logp).sum())
            methods = [("argmax", decoded.harden(model.spec))]
            if generator is not None:
                methods.append(("categorical", sample_hard_graph(decoded, model.spec,
                    generator=generator, temperature=temperature)))
            for method, hard in methods:
                batch = [graph_record(hard, i, model.spec, sample=start+i, method=method)
                         for i in range(len(hard.atoms))]
                for row in batch:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows[method].extend(_compact(batch))
    element_total = float(expected_elements[1:].sum())
    sulfur = model.spec.atomic_numbers.index(16) if 16 in model.spec.atomic_numbers else None
    metrics = {method: summarize_rows(values) for method, values in rows.items() if values}
    metrics["relaxed"] = dict(expected_atoms=expected_atoms / len(codes),
        atom_entropy_per_slot=entropy / (len(codes) * model.spec.max_nodes),
        expected_elements_per_latent=(expected_elements / len(codes)).tolist(),
        expected_sulfur_fraction=float(expected_elements[sulfur]) / element_total
                                 if sulfur is not None and element_total else None)
    return metrics, rows


@torch.inference_mode()
def reconstruct(model, dataset, indices, output, *, batch_size, loss_config, anchor_count):
    """Posterior-mean reconstruction, with complete hard chemistry on every row."""
    rows, means, stds = [], [], []
    totals, label_counts = Counter(), Counter()
    device = next(model.parameters()).device
    with gzip.open(output, "xt", encoding="utf-8") as stream:
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start+batch_size]
            target = collate_graphs([dataset[i] for i in selected]).to(device)
            mu, logvar = model.encode(target.to_relaxed(model.spec))
            if not bool(torch.isfinite(mu).all() and torch.isfinite(logvar).all()):
                raise FloatingPointError("Nonfinite posterior parameters")
            decoded = _checked_decode(model, mu)
            losses = reconstruction_losses(decoded, target.to("cpu"), loss_config)
            for name, value in losses.items():
                if not bool(torch.isfinite(value)):
                    raise FloatingPointError("Nonfinite reconstruction loss")
                totals[name] += float(value) * len(selected)
            hard = decoded.harden(model.spec)
            target_cpu = target.to("cpu")
            active = target_cpu.atoms != 0
            label_counts["active_atoms"] += int(active.sum())
            label_counts["correct_atoms"] += int(((hard.atoms == target_cpu.atoms) & active).sum())
            label_counts["correct_charges"] += int(((hard.charges == target_cpu.charges) & active &
                                                    (hard.atoms != 0)).sum())
            occupied_pairs = (active[:, :, None] & active[:, None, :]).triu(1)
            label_counts["occupied_pairs"] += int(occupied_pairs.sum())
            label_counts["correct_pairs"] += int(((hard.bonds == target_cpu.bonds) & occupied_pairs).sum())
            batch = []
            for i, index in enumerate(selected):
                row = graph_record(hard, i, model.spec, sample=start+i, dataset_index=index, method="argmax")
                reference = dataset.smiles[index]
                row.update(target_smiles=reference, exact=row["valid"] and row["smiles"] == reference,
                    exact_labels=bool((hard.atoms[i] == target_cpu.atoms[i]).all() and
                        (hard.charges[i] == target_cpu.charges[i]).all() and
                        (hard.bonds[i] == target_cpu.bonds[i]).all()))
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                batch.append(row)
            rows.extend(_compact(batch))
            keep = min(len(selected), max(0, anchor_count - start))
            if keep:
                means.append(mu[:keep].cpu())
                std = (logvar[:keep] * .5).exp().cpu()
                if not bool(torch.isfinite(std).all()):
                    raise FloatingPointError("Nonfinite posterior standard deviation")
                stds.append(std)
    metrics = summarize_rows(rows)
    metrics.update(loss={name: value / len(rows) for name, value in totals.items()},
        exact_labels_count=sum(r["exact_labels"] for r in rows), label_counts=dict(label_counts))
    return metrics, rows, torch.cat(means), torch.cat(stds)


def compare_neighbors(rows, baselines):
    """Compare identities/fingerprints to each anchor's own unperturbed decode."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {}
    def fingerprint(smiles):
        if smiles not in fingerprints:
            with rdBase.BlockLogs():
                molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError("Reported valid molecule failed fingerprint parse")
            fingerprints[smiles] = generator.GetFingerprint(molecule)
        return fingerprints[smiles]
    changed = same = rescued = eligible = 0
    similarities = []
    for row, baseline in zip(rows, baselines, strict=True):
        if row["valid"]:
            changed += row["smiles"] != baseline["smiles"]
            same += row["smiles"] == baseline["smiles"]
            rescued += not baseline["valid"]
        if row["valid"] and baseline["valid"]:
            eligible += 1
            similarities.append(DataStructs.TanimotoSimilarity(
                fingerprint(row["smiles"]), fingerprint(baseline["smiles"])))
    return dict(valid_changed_count=changed, valid_unchanged_count=same,
        invalid_to_valid_count=rescued, valid_changed_fraction=changed / len(rows),
        fingerprint_eligible=eligible, mean_tanimoto_to_baseline=sum(similarities) / eligible if eligible else None,
        interpretation="Fingerprint similarity conditions on both graphs being valid; failures remain in validity denominator")


def _validated_config(config):
    if config.get("version") != "paired-vae-checkpoints-v1":
        raise ValueError("Unknown comparison version")
    candidates = config["checkpoints"]
    if not candidates or len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("Checkpoint names must be unique")
    if any(not c["id"] or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in c["id"])
           for c in candidates):
        raise ValueError("Unsafe candidate name")
    for key in ("prior_count", "reconstruction_count", "anchor_count", "directions", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("Positive integer budget required: " + key)
    if config["anchor_count"] > min(config["reconstruction_count"], config["prior_count"]):
        raise ValueError("Anchor count exceeds shared panels")
    if (not config["prior_seeds"] or config["prior_count"] % len(config["prior_seeds"]) or
            len(set(config["prior_seeds"])) != len(config["prior_seeds"])):
        raise ValueError("Prior count must divide across distinct seed blocks")
    if not config["scales"] or len(set(config["scales"])) != len(config["scales"]) or any(
            not math.isfinite(scale) or scale <= 0 for scale in config["scales"]):
        raise ValueError("Positive finite distinct perturbation scales required")
    if config["split"] != "val" or config["temperature"] != 1.:
        raise ValueError("This workflow reserves the final test split and evaluates temperature one")
    return config


def _summary_markdown(result):
    lines = ["# Paired checkpoint comparison", "", "Raw outputs; no repair, filtering or training.", "",
             "| Checkpoint | Prior argmax validity | Categorical validity | Unique-valid yield | Reconstruction exact | Sulfur atom fraction |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, value in result["candidates"].items():
        a, c, r = value["prior"]["argmax"], value["prior"]["categorical"], value["reconstruction"]
        sulfur = f'{100*a["sulfur_atom_fraction"]:.3f}%' if a["sulfur_atom_fraction"] is not None else "undefined"
        lines.append(f'| {name} | {100*a["validity"]:.2f}% | {100*c["validity"]:.2f}% | '
                     f'{100*a["unique_valid_yield"]:.2f}% | {100*r["exact_reconstruction"]:.2f}% | {sulfur} |')
    lines += ["", "See summary.json for paired differences, failure classes, latent-neighbor results and intervals.",
              "The validation panel informs research decisions; it is not an untouched final test.",
              "Perturbation draws share anchors; intervals across those draws are not independent-sample evidence.", ""]
    return "\n".join(lines)


@torch.inference_mode()
def run_comparison(config, output, *, device="cuda"):
    """Sequential comparison in one process/allocation; all model state frozen."""
    _validated_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(output / "config.json", config)
    try:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Requested allocated CUDA device is unavailable")
        dataset_path = Path(config["data"])
        if file_sha256(dataset_path / "manifest.json") != config["data_manifest_sha256"]:
            raise ValueError("Dataset manifest differs")
        for candidate in config["checkpoints"]:
            if file_sha256(Path(candidate["path"])) != candidate["sha256"]:
                raise ValueError("Checkpoint bytes differ: " + candidate["id"])
        dataset = GraphDataset(dataset_path, config["split"])
        if dataset.manifest["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Dataset identity differs")
        excluded = set()
        for panel in config.get("exclude_reconstruction_panels", []):
            excluded.update(subset_indices(len(dataset), panel["count"], panel["seed"]))
        indices = select_indices(len(dataset), config["reconstruction_count"], config["row_seed"], excluded)
        panel_identity = dict(split=config["split"], indices=indices, exclusions=sorted(excluded),
            exclusion_scope="Declared historical panels only; not proof of never having been observed elsewhere",
            prior_seeds=config["prior_seeds"], prior_count=config["prior_count"],
            direction_seed=config["direction_seed"], categorical_seed=config["categorical_seed"])
        write_json(output / "panels.json", panel_identity)
        results, flags = {}, {}
        latent_dimensions = None
        for candidate in config["checkpoints"]:
            begin = time.monotonic()
            folder = output / candidate["id"]
            folder.mkdir()
            model, checkpoint = load_model(Path(candidate["path"]), device, dataset.spec)
            if (checkpoint.get("epoch") != candidate["epoch"] or
                    checkpoint.get("dataset_sha256") != config["dataset_sha256"]):
                raise ValueError("Checkpoint epoch/dataset identity differs")
            if checkpoint.get("train_config", {}).get("mode") != "vae":
                raise ValueError("A standard-normal prior comparison requires a VAE checkpoint")
            if latent_dimensions is not None and model.config.latent != latent_dimensions:
                raise ValueError("Paired checkpoints need equal latent dimensions")
            latent_dimensions = model.config.latent
            model.requires_grad_(False)
            before = state_digest(model.state_dict())
            del checkpoint["model_state"]
            count_per_seed = config["prior_count"] // len(config["prior_seeds"])
            codes = torch.cat([torch.randn(count_per_seed, latent_dimensions,
                generator=torch.Generator().manual_seed(seed)) for seed in config["prior_seeds"]])
            noise = unit_rms_directions(config["anchor_count"] * config["directions"],
                                        latent_dimensions, config["direction_seed"])
            if not (output / "prior-latents.npy").exists():
                np.save(output / "prior-latents.npy", codes.numpy(), allow_pickle=False)
                np.save(output / "perturbation-directions.npy", noise.numpy(), allow_pickle=False)
            if os.environ.get("DBMOL_LAUNCH_READY") and not Path(os.environ["DBMOL_LAUNCH_READY"]).exists():
                write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                    nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
            print(json.dumps(dict(event="candidate_started", candidate=candidate["id"])), flush=True)
            prior, prior_rows = decode_panel(model, codes, folder / "prior.jsonl.gz",
                batch_size=config["batch_size"], categorical_seed=config["categorical_seed"],
                temperature=config["temperature"])
            prior["seed_blocks"] = [{"seed": seed, **summarize_rows(prior_rows["argmax"][
                i*count_per_seed:(i+1)*count_per_seed])} for i, seed in enumerate(config["prior_seeds"])]
            rec, rec_rows, mu, std = reconstruct(model, dataset, indices, folder / "reconstruction.jsonl.gz",
                batch_size=config["batch_size"], loss_config=LossConfig(**checkpoint["loss_config"]),
                anchor_count=config["anchor_count"])
            np.savez_compressed(folder / "posterior-anchors.npz", mu=mu.numpy(), std=std.numpy())
            neighbors = []
            for kind, centers, factors, baseline in (
                    ("posterior", mu, std, rec_rows[:config["anchor_count"]]),
                    ("prior", codes[:config["anchor_count"]], torch.ones_like(std),
                     prior_rows["argmax"][:config["anchor_count"]])):
                repeated = centers.repeat_interleave(config["directions"], 0)
                repeated_factors = factors.repeat_interleave(config["directions"], 0)
                baselines = [r for r in baseline for _ in range(config["directions"])]
                for scale in config["scales"]:
                    displacement = scale * repeated_factors * noise
                    metrics, rows = decode_panel(model, repeated + displacement,
                        folder / f"neighbors-{kind}-{scale:g}.jsonl.gz", batch_size=config["batch_size"])
                    values = rows["argmax"]
                    # Cluster by anchor: four directions are correlated observations.
                    rates = [sum(r["valid"] for r in values[i:i+config["directions"]]) / config["directions"]
                             for i in range(0, len(values), config["directions"])]
                    metrics["argmax"].pop("validity_wilson95")
                    metrics["argmax"].pop("strict_validity_wilson95")
                    metrics.update(kind=kind, scale=scale, anchors=len(centers),
                        directions_per_anchor=config["directions"], anchor_validity_rates=rates,
                        mean_displacement_l2=float(displacement.norm(dim=-1).mean()),
                        max_displacement_l2=float(displacement.norm(dim=-1).max()),
                        baseline_comparison=compare_neighbors(values, baselines))
                    neighbors.append(metrics)
            unchanged = state_digest(model.state_dict()) == before
            if not unchanged or file_sha256(Path(candidate["path"])) != candidate["sha256"]:
                raise RuntimeError("Unexpected frozen model/checkpoint writes")
            results[candidate["id"]] = dict(checkpoint=candidate, prior=prior, reconstruction=rec,
                neighborhoods=neighbors, model_state_sha256=before, unchanged=unchanged,
                elapsed_seconds=time.monotonic()-begin)
            if config.get("atom_order"):
                from latentmol.evaluation.atom_order import permutation_metrics
                order_config = config["atom_order"]
                if not 1 <= order_config["count"] <= len(indices):
                    raise ValueError("Atom-order panel must fit inside reconstruction panel")
                results[candidate["id"]]["atom_order"] = permutation_metrics(model, dataset,
                    indices[:order_config["count"]], folder/"atom-order.jsonl.gz",
                    seed=order_config["seed"], permutations=order_config["permutations"], batch_size=config["batch_size"])
                if state_digest(model.state_dict()) != before:
                    raise RuntimeError("Atom-order evaluation changed frozen model weights")
                results[candidate["id"]]["elapsed_seconds"] = time.monotonic()-begin
            flags[candidate["id"]] = dict(
                prior_valid=[r["valid"] for r in prior_rows["argmax"]],
                prior_categorical_valid=[r["valid"] for r in prior_rows["categorical"]],
                reconstruction_exact=[r["exact"] for r in rec_rows])
            write_json(folder / "summary.json", results[candidate["id"]])
            write_json(output / "progress.json", dict(status="evaluating", completed=list(results),
                elapsed_seconds=time.monotonic()-started, job_id=os.environ.get("LSB_JOBID")))
            print(json.dumps(dict(event="candidate_finished", candidate=candidate["id"],
                validity=prior["argmax"]["validity"], exact=rec["exact_reconstruction"],
                elapsed_seconds=time.monotonic()-begin)), flush=True)
            del model, checkpoint
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        paired = {}
        for left, right in itertools.combinations(flags, 2):
            paired[left + "_to_" + right] = {
                name: paired_difference(flags[left][name], flags[right][name]) for name in flags[left]}
            changes = []
            for a, b in zip(results[left]["neighborhoods"], results[right]["neighborhoods"], strict=True):
                delta = np.array(b["anchor_validity_rates"]) - np.array(a["anchor_validity_rates"])
                se = float(delta.std(ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else None
                mean = float(delta.mean())
                changes.append(dict(kind=a["kind"], scale=a["scale"], anchors=len(delta), difference=mean,
                    anchor_cluster_normal95=[max(-1., mean-1.959964*se), min(1., mean+1.959964*se)] if se is not None else None))
            paired[left + "_to_" + right]["neighborhood_validity"] = changes
        result = dict(version=config["version"], status="completed", candidates=results, paired=paired,
            elapsed_seconds=time.monotonic()-started, device=device, job_id=os.environ.get("LSB_JOBID"),
            policy="Evaluation only; VAE training paused; no repair, rejection sampling or model selection by this script",
            validity_definition="Nonempty, connected, radical-free sanitized graph with reparsable SMILES; schema validity reported separately",
            perturbation_definition="Shared unit-RMS directions per anchor; counts in config; posterior scale times per-dimension std, prior scale in N(0,I) units",
            uncertainty="95% intervals are pointwise, approximate and unadjusted for multiple comparisons; same-lineage checkpoints are paired",
            test_split_untouched=True)
        write_json(output / "summary.json", result)
        (output / "summary.md").write_text(_summary_markdown(result), encoding="utf-8")
        with (output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["checkpoint", "prior_validity", "categorical_validity", "reconstruction_exact", "prior_unique_valid_yield"])
            for name, value in results.items():
                writer.writerow([name, value["prior"]["argmax"]["validity"], value["prior"]["categorical"]["validity"],
                    value["reconstruction"]["exact_reconstruction"], value["prior"]["argmax"]["unique_valid_yield"]])
        return result
    except BaseException as error:
        write_json(output / "failure.json", dict(status="failed", error=type(error).__name__+": "+str(error),
            elapsed_seconds=time.monotonic()-started, training_started=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_comparison(json.loads(args.config.read_text(encoding="utf-8")), args.output, device=args.device)
    print(json.dumps(dict(status=result["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
