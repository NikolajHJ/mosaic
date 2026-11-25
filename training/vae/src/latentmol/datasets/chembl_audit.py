"""Audit ChEMBL structures against the complete MOSES graph cache."""

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
import gzip
import io
from itertools import islice
import json
import os
from pathlib import Path
import time

from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from latentmol.chemistry import MoleculeRejected, encode_smiles
from latentmol.datasets.data import GraphDataset, SPLIT_NAMES, validate_large_cache
from latentmol.experiments import file_sha256, write_json


VERSION = "chembl-moses-audit-v1"
ELIGIBLE_FILE = "eligible.csv.gz"
ELIGIBLE_FIELDS = (
    "chembl_id", "smiles", "scaffold", "scaffold_in_moses_train", "scaffold_in_moses_any",
    "heavy_atoms",
)


def _structure_rows(path, limit):
    """Read the official chemreps TSV without keeping the archive in memory."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"chembl_id", "canonical_smiles"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected the chemreps TSV fields chembl_id and canonical_smiles")
        for row in islice(reader, limit):
            if None in row or not row["chembl_id"] or row["canonical_smiles"] is None:
                raise ValueError(f"Malformed chemreps record at TSV line {reader.line_num}")
            yield row["chembl_id"], row["canonical_smiles"]


def _scaffold_and_size(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Canonical cache identity failed to parse: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False), mol.GetNumHeavyAtoms()


def _scaffold_batch(batch):
    return [_scaffold_and_size(smiles)[0] for smiles in batch]


def _encode_batch(batch, spec):
    results = []
    for chembl_id, source_smiles in batch:
        try:
            smiles = encode_smiles(source_smiles, spec)["smiles"]
        except MoleculeRejected as exc:
            results.append((chembl_id, None, None, None, exc.reason))
        else:
            scaffold, size = _scaffold_and_size(smiles)
            results.append((chembl_id, smiles, scaffold, size, None))
    return results


def _ordered_results(rows, function, pool, workers, *args):
    """Bound submitted chunks and keep file order independent of worker timing."""
    rows = iter(rows)
    pending = deque()
    while batch := list(islice(rows, 512)):
        if pool is None:
            yield from function(batch, *args)
        else:
            pending.append(pool.submit(function, batch, *args))
            if len(pending) >= 2 * workers:
                yield from pending.popleft().result()
    while pending:
        yield from pending.popleft().result()


def _progress(stage, **metrics):
    print(json.dumps({"stage": stage, **metrics}), flush=True)


def _moses_index(cache, manifest, pool, workers):
    verified = validate_large_cache(cache) if manifest["dataset_format_version"] == 2 else None
    identities = {}
    scaffold_sets = {}
    for split in SPLIT_NAMES:
        dataset = GraphDataset(cache, split, validated_cache=verified)
        if len(dataset) != manifest["counts"][split]:
            raise ValueError(f"MOSES {split} count differs from the manifest")
        for smiles in dataset.smiles:
            if smiles in identities:
                raise ValueError("MOSES contains a duplicate graph identity")
            identities[smiles] = split
        scaffolds = set()
        for count, scaffold in enumerate(
            _ordered_results(dataset.smiles, _scaffold_batch, pool, workers), start=1
        ):
            if scaffold:
                scaffolds.add(scaffold)
            if count % 100_000 == 0:
                _progress("moses_scaffolds", split=split, molecules=count)
        scaffold_sets[split] = scaffolds
        _progress("moses_split_ready", split=split, molecules=len(dataset), scaffolds=len(scaffolds))
    if len(identities) != manifest["counts"]["total"]:
        raise ValueError("MOSES total count differs from the manifest")
    return identities, scaffold_sets, dataset.spec


def run_audit(structures, moses_cache, output, *, workers=1, limit=None, source_url=None,
              source_sha256=None, moses_manifest_sha256=None):
    """Count accepted novel identities and scaffold coverage, without modifying inputs."""
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("Workers and an optional row limit must be positive")
    started = time.monotonic()
    structures = Path(structures).resolve(strict=True)
    moses_cache = Path(moses_cache).resolve(strict=True)
    manifest_path = moses_cache / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    if moses_manifest_sha256 is not None and manifest_hash != moses_manifest_sha256:
        raise ValueError("MOSES manifest SHA-256 differs from the expected digest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if rdBase.rdkitVersion != manifest.get("rdkit_version"):
        raise ValueError("Audit RDKit version must match the MOSES canonicalization version")
    source_stat = structures.stat()
    source_hash = file_sha256(structures)
    if source_sha256 is not None and source_hash != source_sha256:
        raise ValueError("ChEMBL source SHA-256 differs from the expected digest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    eligible_path = output / ELIGIBLE_FILE
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)

    source_rows = 0
    duplicate_rows = 0
    seen = set()
    rejected = Counter()
    overlap = Counter({name: 0 for name in SPLIT_NAMES})
    chembl_scaffolds = set()
    novel_scaffolds = set()
    heavy_atoms = Counter()
    scaffold_membership = Counter({name: 0 for name in (
        "acyclic", "seen_train", "seen_holdouts_only", "unseen_moses",
    )})
    with executor as pool:
        moses, moses_scaffolds, spec = _moses_index(moses_cache, manifest, pool, workers)
        train_scaffolds = moses_scaffolds["train"]
        all_scaffolds = set.union(*moses_scaffolds.values())
        records = _ordered_results(_structure_rows(structures, limit), _encode_batch, pool, workers, spec)
        with eligible_path.open("xb") as raw, gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed, io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(ELIGIBLE_FIELDS)
            for source_rows, (chembl_id, smiles, scaffold, size, reason) in enumerate(records, 1):
                if reason is not None:
                    rejected[reason] += 1
                elif smiles in seen:
                    duplicate_rows += 1
                else:
                    seen.add(smiles)
                    if scaffold:
                        chembl_scaffolds.add(scaffold)
                    split = moses.get(smiles)
                    if split is not None:
                        overlap[split] += 1
                    else:
                        in_train = scaffold in train_scaffolds
                        in_moses = scaffold in all_scaffolds
                        if scaffold:
                            novel_scaffolds.add(scaffold)
                            membership = ("seen_train" if in_train else
                                          "seen_holdouts_only" if in_moses else "unseen_moses")
                        else:
                            membership = "acyclic"
                        scaffold_membership[membership] += 1
                        heavy_atoms[size] += 1
                        writer.writerow((chembl_id, smiles, scaffold, int(in_train), int(in_moses), size))
                if source_rows % 100_000 == 0:
                    _progress("chembl_structures", rows=source_rows, new_molecules=heavy_atoms.total())
    if not source_rows:
        raise ValueError("No ChEMBL structure records were read")
    current_stat = structures.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (current_stat.st_size, current_stat.st_mtime_ns):
        raise ValueError("ChEMBL structure file changed during audit")
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError("MOSES manifest changed during audit")

    result = {
        "version": VERSION,
        "source": {
            "path": str(structures), "url": source_url, "sha256": source_hash,
            "bytes": source_stat.st_size, "row_limit": limit, "complete": limit is None,
        },
        "moses": {
            "cache": str(moses_cache), "dataset_sha256": manifest["dataset_sha256"],
            "manifest_sha256": manifest_hash, "split_sha256": manifest["split_sha256"],
            "counts": manifest["counts"], "graph_spec_fingerprint": spec.fingerprint(),
        },
        "rdkit_version": rdBase.rdkitVersion,
        "identity_policy": "existing strict encode_smiles with stereo-free canonical identity",
        "scaffold_policy": "atom-typed RDKit Bemis–Murcko, no stereo; acyclic reported separately",
        "counts": {
            "source_rows": source_rows, "rejected_rows": rejected.total(),
            "accepted_rows": source_rows - rejected.total(), "duplicate_rows": duplicate_rows,
            "unique_molecules": len(seen), "overlap_moses": dict(overlap),
            "new_molecules": heavy_atoms.total(),
        },
        "rejections": dict(rejected),
        "scaffolds": {
            "moses_by_split": {name: len(values) for name, values in moses_scaffolds.items()},
            "moses_all": len(all_scaffolds), "chembl_accepted": len(chembl_scaffolds),
            "of_new_molecules": len(novel_scaffolds),
            "new_vs_moses_train": len(novel_scaffolds - train_scaffolds),
            "new_vs_moses_all": len(novel_scaffolds - all_scaffolds),
            "new_molecule_membership": dict(scaffold_membership),
        },
        "new_molecule_heavy_atom_counts": dict(heavy_atoms),
        "eligible": {"path": ELIGIBLE_FILE, "sha256": file_sha256(eligible_path), "count": heavy_atoms.total()},
        "elapsed_seconds": time.monotonic() - started,
        "workers": workers,
    }
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True, help="Official chemreps TSV, plain or gzip")
    parser.add_argument("--moses-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    parser.add_argument("--workers", type=int, default=min(8, available_cpus))
    parser.add_argument("--limit", type=int, help="Optional source-row cap for a smoke audit")
    parser.add_argument("--source-url")
    parser.add_argument("--source-sha256")
    parser.add_argument("--moses-manifest-sha256")
    args = parser.parse_args()
    summary = run_audit(**vars(args))
    _progress("complete", summary=str(args.output / "summary.json"), **summary["counts"])


if __name__ == "__main__":
    main()
