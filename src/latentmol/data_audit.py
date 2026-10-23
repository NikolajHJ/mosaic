"""Verify actual expanded-cache split isolation before expensive training."""

import argparse
import json
from pathlib import Path

import numpy as np

from latentmol.data import SPLIT_NAMES, _split_digest, sha256_file
from latentmol.experiments import ExperimentRun, write_json


def _identities(cache):
    """Read identity arrays after verifying both byte and assignment checksums."""
    manifest = json.loads((cache / "manifest.json").read_text())
    if sha256_file(cache / "dataset.npz") != manifest["dataset_sha256"]:
        raise ValueError("Dataset digest mismatch")
    with np.load(cache / "dataset.npz", allow_pickle=False) as archive:
        smiles, split = archive["smiles"].tolist(), archive["split"]
    if _split_digest(smiles, split) != manifest["split_sha256"]:
        raise ValueError("Split identity digest mismatch")
    if len(smiles) != len(set(smiles)) or len(smiles) != manifest["counts"]["total"]:
        raise ValueError("Duplicate or inconsistent molecular identity count")
    if not np.isin(split, list(SPLIT_NAMES.values())).all():
        raise ValueError("Unknown split label")
    sets = {name: {smiles[i] for i in np.flatnonzero(split == label)}
            for name, label in SPLIT_NAMES.items()}
    if any(len(values) != manifest["counts"][name] for name, values in sets.items()):
        raise ValueError("Observed split count differs from manifest")
    return manifest, sets


def audit_cache(cache: Path, parent: Path, expected_counts=None):
    """Require disjoint full splits and exact preservation of every old role."""
    manifest, sets = _identities(cache)
    old, prior = _identities(parent)
    if any(manifest[key] != old[key] for key in ("spec_fingerprint", "raw_sha256")):
        raise ValueError("Expanded and parent caches differ in schema or raw source")
    counts = {name: len(values) for name, values in sets.items()}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {counts}")
    intersections = {f"{a}:{b}": len(sets[a] & sets[b])
                     for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    overlaps = {old_name: {new_name: len(old_values & new_values) for new_name, new_values in sets.items()}
                for old_name, old_values in prior.items()}
    if any(intersections.values()) or any(prior[name] - sets[name] for name in prior):
        raise ValueError("Holdout isolation or prior split-role preservation failed")
    return {"passed": True, "counts": counts, "pairwise_intersections": intersections,
            "old_split_rows_new_split_columns": overlaps,
            "dataset_sha256": manifest["dataset_sha256"], "split_sha256": manifest["split_sha256"],
            "parent_dataset_sha256": old["dataset_sha256"],
            "identity_policy": "canonical stereo-free SMILES; molecule split, not scaffold split"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    args = parser.parse_args()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with ExperimentRun(args.run_root, "full-data-isolation-audit", config, Path.cwd()) as run:
        report = audit_cache(args.cache, args.parent, {"train": args.train_count, "val": args.val_count, "test": args.test_count})
        write_json(run.path / "audit.json", report)
        run.log(report)


if __name__ == "__main__":
    main()
