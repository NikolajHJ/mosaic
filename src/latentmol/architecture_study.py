"""Matched fixed-z hard-draw diagnostics for graph-decoder comparisons.

This measures the *actual factorized sampling law* at encoded posterior means.
It retains every attempted graph and reports validity without repair. It does
not estimate binding, synthesizability, or a calibrated molecular likelihood.
Only run this numerical study in a DTU HPC allocation.
"""

import argparse
from dataclasses import asdict
import json
from collections import Counter
from pathlib import Path

import torch

from latentmol.data import GraphDataset, collate_graphs
from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.sampling import sample_hard_graph, subset_indices
from latentmol.quality import GradedEvaluator, QualityScoreConfig, summarize_quality


def graph_patterns(graph: HardGraphBatch, spec) -> dict[str, Counter]:
    """Count element/charge and local bond patterns without sanitizing graphs.

    Counts include invalid generated graphs; filtering to valid ones would make
    the learned distribution look artificially more chemically coherent.
    Bond signatures count incident bond categories rather than imposing an
    approximate valence formula on aromatic or charged atoms.
    """
    counts = {"element_charge": Counter(), "element_bond_signature": Counter(),
              "bond_endpoint_chemistry": Counter(), "size": Counter()}
    for atoms, charges, bonds in zip(graph.atoms, graph.charges, graph.bonds):
        active = (atoms != 0).nonzero(as_tuple=True)[0].tolist()
        counts["size"][str(len(active))] += 1
        for i in active:
            element = spec.atomic_numbers[int(atoms[i])]
            charge = spec.formal_charges[int(charges[i])]
            counts["element_charge"][f"{element}|{charge}"] += 1
            signature = ",".join(str(int((bonds[i, active] == category).sum()))
                                 for category in range(1, len(spec.bond_names)))
            counts["element_bond_signature"][f"{element}|{charge}|{signature}"] += 1
        for position, i in enumerate(active):
            for j in active[position + 1:]:
                # Include NONE pairs as the denominator needed to estimate
                # P(bond category | endpoint elements and charges).
                bond = int(bonds[i, j])
                left = (spec.atomic_numbers[int(atoms[i])], spec.formal_charges[int(charges[i])])
                right = (spec.atomic_numbers[int(atoms[j])], spec.formal_charges[int(charges[j])])
                first, second = sorted((left, right))
                counts["bond_endpoint_chemistry"][f"{first}|{second}|{spec.bond_names[bond]}"] += 1
    return counts


def _accumulate(destination: dict[str, Counter], source: dict[str, Counter]) -> None:
    for name, values in source.items():
        destination[name].update(values)


def _json_counts(counts: dict[str, Counter]) -> dict:
    return {name: dict(sorted(values.items())) for name, values in counts.items()}


@torch.no_grad()
def run_study(checkpoint_path: Path, data_path: Path, run_root: Path, *,
              split: str = "val", seed: int = 17, molecules: int = 128,
              draws_per_molecule: int = 16, temperature: float = 1.0,
              device: str = "cuda", source_indices: list[int] | None = None,
              quality_config: QualityScoreConfig | None = None) -> Path:
    """Evaluate the same fixed-z sampling budget for any compatible checkpoint.

    Per-molecule encoded means are used for both AE and VAE. An AE has no trained
    standard-normal prior, and a VAE posterior sample would add another noise
    source; neither is mixed into this controlled categorical-draw experiment.
    """
    if molecules < 1 or draws_per_molecule < 1 or not 0 < temperature < float("inf"):
        raise ValueError("Molecule/draw budgets and temperature must be positive")
    checkpoint_path, data_path, run_root = map(Path, (checkpoint_path, data_path, run_root))
    quality_config = quality_config or QualityScoreConfig()
    options = {"checkpoint": str(checkpoint_path), "data": str(data_path),
               "split": split, "seed": seed, "molecules": molecules,
               "draws_per_molecule": draws_per_molecule, "temperature": temperature,
               "latent_policy": "encoded_posterior_mean", "occupancy_policy": "variable",
               "device": device, "quality_weights": asdict(quality_config),
               "source_indices": source_indices}
    with ExperimentRun(run_root, "architecture-fixed-z-draws", options, Path.cwd()) as run:
        data = GraphDataset(data_path, split)
        model, checkpoint = load_model(checkpoint_path, device, data.spec)
        model.eval()
        if checkpoint["dataset_sha256"] != data.manifest["dataset_sha256"]:
            raise ValueError("Checkpoint and data identities differ")
        if molecules > len(data):
            raise ValueError("Requested fixed-z molecule budget exceeds the split")
        indices = (subset_indices(len(data), molecules, seed, shuffle_full=True)
                   if source_indices is None else list(source_indices))
        if (len(indices) != molecules or len(set(indices)) != molecules
                or any(type(i) is not int or not 0 <= i < len(data) for i in indices)):
            raise ValueError("Explicit cohort must contain exactly the requested unique split indices")
        generator = torch.Generator(device=device).manual_seed(seed + 701)
        patterns = {label: {name: Counter() for name in
                            ("element_charge", "element_bond_signature",
                             "bond_endpoint_chemistry", "size")}
                    for label in ("target", "argmax", "categorical")}
        chemistry = {label: {"attempts": 0, "valid": 0, "exact": 0,
                             "failure_reasons": Counter(), "valid_smiles": set()}
                     for label in ("argmax", "categorical")}
        saved = {"source_indices": indices, "targets": [], "argmax": [], "categorical": [],
                 "draws_per_molecule": draws_per_molecule,
                 "checkpoint_sha256": file_sha256(checkpoint_path)}
        per_seed = []
        graded = {label: [] for label in ("argmax", "categorical")}
        quality_evaluator = GradedEvaluator(data.spec, quality_config)
        for index in indices:
            target = collate_graphs([data[index]])
            target_smiles = data.smiles[index]
            mean, _ = model.encode(target.to(device).to_relaxed(data.spec))
            decoded = model.decode(mean)
            best = decoded.harden(data.spec).to("cpu")
            repeated = DecoderOutput(*(tensor.expand(draws_per_molecule, *tensor.shape[1:])
                                       for tensor in (decoded.atoms, decoded.charges, decoded.bonds)))
            sampled = sample_hard_graph(repeated, data.spec, generator=generator,
                                        temperature=temperature).to("cpu")
            _accumulate(patterns["target"], graph_patterns(target, data.spec))
            _accumulate(patterns["argmax"], graph_patterns(best, data.spec))
            _accumulate(patterns["categorical"], graph_patterns(sampled, data.spec))
            seed_result = {"source_index": index, "target": target_smiles,
                           "categorical_valid": 0, "categorical_exact": 0,
                           "categorical_unique_valid": 0}
            for label, graphs in (("argmax", best), ("categorical", sampled)):
                local_valid = set()
                for row in range(len(graphs.atoms)):
                    diagnostic = quality_evaluator.diagnose(graphs.atoms[row], graphs.charges[row], graphs.bonds[row])
                    graded[label].append({"source_index": index, "draw_index": row,
                                          "sampling_mode": label, **diagnostic})
                    result = chemistry[label]
                    result["attempts"] += 1
                    if diagnostic["smiles"]:
                        result["valid"] += 1
                        result["valid_smiles"].add(diagnostic["smiles"])
                        local_valid.add(diagnostic["smiles"])
                        if diagnostic["smiles"] == target_smiles:
                            result["exact"] += 1
                            if label == "categorical":
                                seed_result["categorical_exact"] += 1
                        if label == "categorical":
                            seed_result["categorical_valid"] += 1
                    else:
                        result["failure_reasons"][diagnostic["failure_reason"] or "unknown"] += 1
                if label == "categorical":
                    seed_result["categorical_unique_valid"] = len(local_valid)
            per_seed.append(seed_result)
            saved["targets"].append(tuple(value.to(torch.uint8) for value in
                                          (target.atoms, target.charges, target.bonds)))
            saved["argmax"].append(tuple(value.to(torch.uint8) for value in
                                         (best.atoms, best.charges, best.bonds)))
            saved["categorical"].append(tuple(value.to(torch.uint8) for value in
                                              (sampled.atoms, sampled.charges, sampled.bonds)))
        for label in ("targets", "argmax", "categorical"):
            saved[label] = tuple(torch.cat([row[field] for row in saved[label]])
                                 for field in range(3))
        artifact = run.path / "attempted_graphs.pt"
        torch.save(saved, artifact)
        results = {label: {"patterns": _json_counts(patterns[label]),
                           **({"attempts": chemistry[label]["attempts"],
                               "valid": chemistry[label]["valid"],
                               "exact": chemistry[label]["exact"],
                               "unique_valid": len(chemistry[label]["valid_smiles"]),
                               "failure_reasons": dict(chemistry[label]["failure_reasons"])}
                              if label != "target" else {})}
                   for label in patterns}
        for label in graded:
            results[label]["graded_chemistry"] = summarize_quality(graded[label], quality_config)
        # Every row maps into attempted_graphs.pt, including invalid and duplicate
        # draws. Store metrics separately so reviewers need not load torch data.
        write_json(run.path / "graded_attempts.json", graded)
        write_json(run.path / "summary.json", {"options": options, "results": results,
                                                "per_seed": per_seed, "graded_compute": quality_evaluator.compute()})
        run.annotate(dataset_manifest=data.manifest, source_indices=indices,
                     checkpoint_sha256=file_sha256(checkpoint_path),
                     saved_attempted_graphs=str(artifact),
                     attempted_graphs_sha256=file_sha256(artifact))
        run.log({"stage": "fixed_z_draws", "molecules": len(indices),
                 "draws_per_molecule": draws_per_molecule,
                 "argmax_valid": chemistry["argmax"]["valid"],
                 "argmax_attempts": chemistry["argmax"]["attempts"],
                 "categorical_valid": chemistry["categorical"]["valid"],
                 "categorical_attempts": chemistry["categorical"]["attempts"]})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--molecules", type=int, default=128)
    parser.add_argument("--draws-per-molecule", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--quality-config", type=Path, help="JSON QualityScoreConfig weights; recorded verbatim")
    args = parser.parse_args()
    run_study(args.checkpoint, args.data, args.run_root, split=args.split, seed=args.seed,
              molecules=args.molecules, draws_per_molecule=args.draws_per_molecule,
              temperature=args.temperature, device=args.device,
              quality_config=QualityScoreConfig.from_dict(json.loads(args.quality_config.read_text()))
              if args.quality_config else None)


if __name__ == "__main__":
    main()
