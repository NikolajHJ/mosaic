"""Matched VAE fine-tuning on MOSES and a MOSES/ChEMBL mixture."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import GraphDataset, validate_large_cache
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.evaluation.checkpoint_comparison import paired_difference, reconstruct, run_comparison
from latentmol.evaluation.evaluate import load_model
from latentmol.evaluation.interpolation import _encode, evaluate_arm
from latentmol.experiments import file_sha256, write_json
from latentmol.sampling import subset_indices
from latentmol.train import state_digest
from latentmol.training.interpolation_pilot import _pair_panel, _train_arm
from latentmol.training.losses import LossConfig


VERSION = "matched-chembl-comparison-v1"


def replacement_slots(updates, batch_size, percent, seed):
    """Allocate the exact rounded-down budget with at most one row variation per update."""
    target = updates * batch_size * percent // 100
    rng = random.Random(seed)
    slots = []
    for step in range(updates):
        count = target * (step + 1) // updates - target * step // updates
        slots.extend(step * batch_size + slot for slot in rng.sample(range(batch_size), count))
    return sorted(slots)


def _holdout_scaffold(scaffold, seed, percent):
    digest = hashlib.blake2b(f"{seed}:{scaffold}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 100 < percent


def select_chembl(path, *, train_count, holdout_count, seed, split_seed, holdout_percent):
    """Stream eligible rows into deterministic reservoirs, keeping unseen scaffold families apart."""
    pools = {"train": [], "holdout": []}
    capacities = {"train": train_count, "holdout": holdout_count}
    seen = dict.fromkeys(pools, 0)
    rng = {"train": random.Random(seed), "holdout": random.Random(seed + 1)}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"chembl_id", "smiles", "scaffold", "scaffold_in_moses_train",
                    "scaffold_in_moses_any", "heavy_atoms"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Unexpected audited ChEMBL row schema")
        for row in reader:
            scaffold = row["scaffold"]
            flag = row["scaffold_in_moses_any"]
            if flag not in ("0", "1") or not row["smiles"] or not row["chembl_id"]:
                raise ValueError("Invalid audited ChEMBL row")
            name = ("holdout" if scaffold and flag == "0" and
                    _holdout_scaffold(scaffold, split_seed, holdout_percent) else "train")
            seen[name] += 1
            pool = pools[name]
            if len(pool) < capacities[name]:
                pool.append(row)
            else:
                position = rng[name].randrange(seen[name])
                if position < capacities[name]:
                    pool[position] = row
    if any(len(pools[name]) != capacities[name] for name in pools):
        raise ValueError(f"Audited ChEMBL does not meet both pilot budgets: {seen}")
    for name, pool in pools.items():
        rng[name].shuffle(pool)
    train_scaffolds = {row["scaffold"] for row in pools["train"]}
    holdout_scaffolds = {row["scaffold"] for row in pools["holdout"]}
    rows = pools["train"] + pools["holdout"]
    if train_scaffolds & holdout_scaffolds or len({row["smiles"] for row in rows}) != len(rows):
        raise ValueError("ChEMBL panels have duplicate structures or overlapping scaffolds")
    return pools, seen


class _ChemblGraphs:
    def __init__(self, rows, spec):
        self.spec = spec
        self.smiles = [row["smiles"] for row in rows]
        self.items = []
        for smiles in self.smiles:
            graph = encode_smiles(smiles, spec)
            if graph["smiles"] != smiles:
                raise ValueError("Audited ChEMBL input changed under canonical encoding")
            self.items.append({name: torch.as_tensor(graph[field], dtype=torch.long) for name, field in
                               (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class _ReplacementRows:
    def __init__(self, moses, replacements):
        self.moses = moses
        self.replacements = replacements

    def __getitem__(self, index):
        return self.replacements[index] if index in self.replacements else self.moses[index]


def _validated_config(config):
    if config.get("version") != VERSION:
        raise ValueError("Expected the matched ChEMBL comparison config")
    for key in ("updates", "batch_size", "prior_anchor_count", "chembl_holdout_count"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    for key in ("chembl_percent", "chembl_holdout_scaffold_percent"):
        if type(config.get(key)) is not int or not 0 < config[key] < 100:
            raise ValueError(f"Integer percentage must be in (0, 100): {key}")
    for key in ("encoder_learning_rate", "decoder_learning_rate", "clip_grad"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Positive finite setting required: {key}")
    for key in ("beta", "prior_anchor_weight", "weight_decay"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Nonnegative finite setting required: {key}")


def _evaluate_additional_panels(candidates, holdout, val, config, output, device):
    paths = output / "interpolation-evaluation"
    paths.mkdir()
    pairs, (graph_a, graph_b, mapped_b), selection = _pair_panel(
        val, config["validation_pairs"], paths / "pairs.json",
        lambda value: print(json.dumps(dict(value, event="validation_pairs")), flush=True))
    panel = config["path_evaluation"]
    reconstruction, exact, interpolation, argmax = {}, {}, {}, {}
    for candidate in candidates:
        name, path = candidate["id"], Path(candidate["path"])
        if file_sha256(path) != candidate["sha256"]:
            raise ValueError(f"Candidate checkpoint identity differs: {name}")
        model, checkpoint = load_model(path, device, val.spec)
        model.requires_grad_(False)
        before = state_digest(model.state_dict())
        reconstruction[name], rows, _, _ = reconstruct(
            model, holdout, range(len(holdout)), output / f"chembl-reconstruction-{name}.jsonl.gz",
            batch_size=config["comparison"]["batch_size"], loss_config=LossConfig(**checkpoint["loss_config"]),
            anchor_count=1)
        exact[name] = [row["exact"] for row in rows]
        z_a, z_b = (_encode(model, graphs, panel["batch_size"]) for graphs in (graph_a, graph_b))
        interpolation[name], raw = evaluate_arm(
            model, pairs, graph_a, graph_b, z_a, z_b, panel, paths / name, mapped_b=mapped_b)
        argmax[name] = {fraction: [row["valid"] for row in raw[(fraction, "argmax")]]
                        for fraction in panel["fractions"]}
        if state_digest(model.state_dict()) != before or file_sha256(path) != candidate["sha256"]:
            raise RuntimeError(f"Unexpected frozen candidate writes: {name}")
    return dict(
        chembl_holdout_reconstruction=reconstruction,
        paired_chembl_holdout_exact=paired_difference(exact["moses"], exact["mixture"]),
        interpolation=interpolation,
        interpolation_pairs=selection,
        paired_path_argmax_validity={
            str(fraction): paired_difference(argmax["moses"][fraction], argmax["mixture"][fraction])
            for fraction in panel["fractions"]
        },
    )


def run_pilot(config, audit, audit_sha256, output, *, device="cuda"):
    _validated_config(config)
    audit, output = Path(audit), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = dict(config, chembl_audit=str(audit), chembl_audit_sha256=audit_sha256)
    write_json(output / "config.json", config)

    def progress(value):
        value = dict(value, elapsed_seconds=time.monotonic() - started)
        write_json(output / "progress.json", value)
        print(json.dumps(value, allow_nan=False), flush=True)

    try:
        parent_path, data_path = Path(config["checkpoint"]), Path(config["data"])
        manifest_path = data_path / "manifest.json"
        audit_path = audit / "summary.json"
        if file_sha256(parent_path) != config["checkpoint_sha256"] or file_sha256(manifest_path) != config["data_manifest_sha256"]:
            raise ValueError("Epoch-45 checkpoint or MOSES manifest identity differs")
        if file_sha256(audit_path) != audit_sha256:
            raise ValueError("ChEMBL audit summary identity differs")
        audit_summary = json.loads(audit_path.read_text(encoding="utf-8"))
        eligible_path = audit / "eligible.csv.gz"
        if (audit_summary.get("version") != "chembl-moses-audit-v1"
                or not audit_summary["source"]["complete"]
                or audit_summary["moses"]["dataset_sha256"] != config["dataset_sha256"]
                or audit_summary["moses"]["manifest_sha256"] != config["data_manifest_sha256"]
                or audit_summary["eligible"]["path"] != eligible_path.name
                or file_sha256(eligible_path) != audit_summary["eligible"]["sha256"]):
            raise ValueError("ChEMBL audit, eligible file and MOSES data identities differ")
        config["chembl_eligible_sha256"] = audit_summary["eligible"]["sha256"]
        write_json(output / "config.json", config)

        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache = validate_large_cache(data_path) if manifest["dataset_format_version"] == 2 else None
        train = GraphDataset(data_path, "train", validated_cache=cache)
        val = GraphDataset(data_path, "val", validated_cache=cache)
        reference, parent = load_model(parent_path, device, train.spec)
        if (parent.get("epoch") != config["epoch"]
                or parent.get("dataset_sha256") != config["dataset_sha256"]
                or manifest["dataset_sha256"] != config["dataset_sha256"]
                or parent.get("train_config", {}).get("mode") != "vae"):
            raise ValueError("Warm start requires the declared epoch-45 MOSES VAE")
        reference.requires_grad_(False)
        before = state_digest(reference.state_dict())
        loss_config = LossConfig(**parent["loss_config"])
        count = config["updates"] * config["batch_size"]
        if count > len(train):
            raise ValueError("Insufficient distinct MOSES training rows for the requested budget")
        rows = subset_indices(len(train), count, config["row_seed"], shuffle_full=True)
        slots = replacement_slots(config["updates"], config["batch_size"],
                                  config["chembl_percent"], config["replacement_seed"])
        if not slots:
            raise ValueError("Requested mixture has no ChEMBL replacements")
        cache_checks = dict(train=verify_canonical_training_cache(train, rows),
                            val=verify_canonical_training_cache(val, list(range(min(16, len(val))))))
        if os.environ.get("DBMOL_LAUNCH_READY"):
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
        progress(dict(stage="selecting_chembl"))
        pools, seen = select_chembl(eligible_path, train_count=len(slots), holdout_count=config["chembl_holdout_count"],
            seed=config["chembl_row_seed"], split_seed=config["chembl_scaffold_seed"],
            holdout_percent=config["chembl_holdout_scaffold_percent"])
        if sum(seen.values()) != audit_summary["eligible"]["count"]:
            raise ValueError("ChEMBL eligible row count differs from the audit")
        chembl = _ChemblGraphs(pools["train"], train.spec)
        holdout = _ChemblGraphs(pools["holdout"], train.spec)
        replacements = {rows[slot]: chembl[i] for i, slot in enumerate(slots)}
        write_json(output / "training-panels.json", dict(
            moses_rows=rows, replacement_slots=slots, chembl=pools, eligible_split_counts=seen,
            canonical_cache_checks=cache_checks, chembl_roundtrip_checks=len(chembl) + len(holdout)))
        schedule = [()] * config["updates"]
        checkpoints = {}
        for name, data in (("moses", train), ("mixture", _ReplacementRows(train, replacements))):
            checkpoints[name] = _train_arm(name, 0., reference, parent, data, rows, (), None,
                schedule, dict(config, arm=name), loss_config, output / name, progress)
        if state_digest(reference.state_dict()) != before or file_sha256(parent_path) != config["checkpoint_sha256"]:
            raise RuntimeError("Unexpected write to the frozen epoch-45 model or checkpoint")
        del reference, parent, chembl, replacements
        write_json(output / "training-completed.json", dict(checkpoints=checkpoints, frozen_parent_verified=True))

        candidates = [dict(id="epoch45", path=str(parent_path), epoch=config["epoch"], sha256=config["checkpoint_sha256"])]
        candidates += [dict(id=name, **{key: record[key] for key in ("path", "epoch", "sha256")})
                       for name, record in checkpoints.items()]
        progress(dict(stage="evaluating_moses_reconstruction_and_prior"))
        comparison = run_comparison(dict(config["comparison"], version="paired-vae-checkpoints-v1",
            data=config["data"], data_manifest_sha256=config["data_manifest_sha256"],
            dataset_sha256=config["dataset_sha256"], split="val", checkpoints=candidates),
            output / "comparison", device=device)
        progress(dict(stage="evaluating_chembl_holdout_and_moses_interpolation"))
        extra = _evaluate_additional_panels(candidates, holdout, val, config, output, device)
        result = dict(status="completed", job_id=os.environ.get("LSB_JOBID"),
            checkpoints=checkpoints, parent_sha256=config["checkpoint_sha256"], frozen_parent_verified=True,
            updates_per_arm=config["updates"], rows_per_arm=count, chembl_replacements=len(slots),
            chembl_holdout_count=len(holdout),
            chembl_holdout_scaffolds=len({row["scaffold"] for row in pools["holdout"]}),
            comparison={name: dict(exact_reconstruction=value["reconstruction"]["exact_reconstruction"],
                argmax_prior_validity=value["prior"]["argmax"]["validity"],
                categorical_prior_validity=value["prior"]["categorical"]["validity"],
                argmax_unique_valid_yield=value["prior"]["argmax"]["unique_valid_yield"],
                categorical_unique_valid_yield=value["prior"]["categorical"]["unique_valid_yield"])
                for name, value in comparison["candidates"].items()},
            elapsed_seconds=time.monotonic() - started, **extra)
        write_json(output / "summary.json", result)
        progress(dict(stage="completed"))
        return result
    except BaseException as error:
        write_json(output / "failure.json", dict(status="failed", error=type(error).__name__ + ": " + str(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True, help="SHA-256 of audit/summary.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_pilot(json.loads(args.config.read_text(encoding="utf-8")), args.audit, args.audit_sha256,
              args.output, device=args.device)


if __name__ == "__main__":
    main()
