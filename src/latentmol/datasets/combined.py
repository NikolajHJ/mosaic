"""Compose the immutable MOSES cache with a compact, scaffold-split ChEMBL cache."""

import argparse
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase
import torch
from torch.utils.data import Dataset

from latentmol.chemistry import encode_smiles
from latentmol.datasets.chembl_audit import ELIGIBLE_FIELDS, VERSION as AUDIT_VERSION, _ordered_results
from latentmol.datasets.data import GraphDataset, SPLIT_NAMES, large_manifest_digest, validate_large_cache
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


VERSION = "moses-chembl-combined-v1"
FILES = ("atoms.npy", "charges.npy", "edge_offsets.npy", "edges.bin", "split.npy",
         "smiles_offsets.npy", "smiles.utf8")


def _checked_json(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"Input metadata hash differs: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _eligible_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ELIGIBLE_FIELDS:
            raise ValueError("Unexpected ChEMBL audit table schema")
        for row in reader:
            if (None in row or any(value is None for value in row.values()) or not row["smiles"]
                    or row["scaffold_in_moses_any"] not in ("0", "1")):
                raise ValueError(f"Malformed ChEMBL audit table row {reader.line_num}")
            scaffold = row["scaffold"] if row["scaffold_in_moses_any"] == "0" else ""
            yield row["smiles"], scaffold, int(row["heavy_atoms"])


def _holdout_groups(groups, seed, targets):
    """Assign whole unseen scaffold groups in a reproducible seeded order."""
    ranked = sorted(groups, key=lambda group: (
        hashlib.blake2b(f"{seed}:{group}".encode(), digest_size=16).digest(), group))
    assignment = {}
    for name, target in targets.items():
        remaining = target
        for group in ranked:
            if group not in assignment and groups[group] <= remaining:
                assignment[group] = SPLIT_NAMES[name]
                remaining -= groups[group]
                if remaining == 0:
                    break
        if remaining == target:
            raise ValueError(f"No unseen ChEMBL scaffolds could be reserved for {name}")
    return assignment


def _encode_rows(batch, spec):
    result = []
    for smiles, split, expected_atoms in batch:
        graph = encode_smiles(smiles, spec)
        atoms, charges, bonds = (graph[key] for key in ("atom_types", "charges", "bond_types"))
        if graph["smiles"] != smiles or np.count_nonzero(atoms) != expected_atoms:
            raise ValueError("Audited ChEMBL identity or heavy atom count changed")
        left, right = np.nonzero(bonds)
        upper = left < right
        left, right = left[upper], right[upper]
        edges = np.column_stack((left, right, bonds[left, right])).astype(np.uint8)
        result.append((smiles.encode("utf-8"), atoms, charges, edges, split))
    return result


def _cache_arrays(output, count, nodes):
    shapes = {"atoms": ((count, nodes), "int8"), "charges": ((count, nodes), "int8"),
              "split": ((count,), "int8"), "edge_offsets": ((count + 1,), "int64"),
              "smiles_offsets": ((count + 1,), "int64")}
    return {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            for name, (shape, dtype) in shapes.items()}


def _write_graphs(output, eligible, spec, holdouts, count, workers):
    arrays = _cache_arrays(output, count, spec.max_nodes)
    arrays["edge_offsets"][0] = arrays["smiles_offsets"][0] = 0
    split_counts = Counter()
    edge_end, smiles_end = 0, 0
    rows_written = 0
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    with executor as pool, (output / "edges.bin").open("xb") as edges, (output / "smiles.utf8").open("xb") as smiles:
        rows = ((smi, holdouts.get(scaffold, 0), atoms)
                for smi, scaffold, atoms in _eligible_rows(eligible))
        for i, (smi, atom, charge, triples, split) in enumerate(
                _ordered_results(rows, _encode_rows, pool, workers, spec)):
            arrays["atoms"][i], arrays["charges"][i], arrays["split"][i] = atom, charge, split
            edges.write(triples.tobytes())
            smiles.write(smi)
            edge_end += len(triples)
            smiles_end += len(smi)
            arrays["edge_offsets"][i + 1], arrays["smiles_offsets"][i + 1] = edge_end, smiles_end
            split_counts[split] += 1
            rows_written = i + 1
            if rows_written % 100_000 == 0:
                print(json.dumps(dict(stage="encode_chembl", rows=rows_written)), flush=True)
    if rows_written != count:
        raise ValueError("Encoded ChEMBL row count differs from the audit")
    for array in arrays.values():
        array.flush()
    return {name: split_counts[label] for name, label in SPLIT_NAMES.items()}


def prepare_combined(config, output, *, workers=1):
    """Revalidate and encode audited ChEMBL once, leaving MOSES on its original cache."""
    if config["version"] != VERSION or workers < 1:
        raise ValueError("Require the combined data config and a positive worker count")
    started = time.monotonic()
    moses_path = Path(config["moses_cache"]).resolve(strict=True)
    audit_path = Path(config["audit_summary"]).resolve(strict=True)
    moses = _checked_json(moses_path / "manifest.json", config["moses_manifest_sha256"])
    audit = _checked_json(audit_path, config["audit_summary_sha256"])
    spec = GraphSpec.from_dict(moses["spec"])
    if spec.max_nodes > 256:
        raise ValueError("Sparse bond indices must fit in one byte")
    if (audit["version"] != AUDIT_VERSION or not audit["source"]["complete"]
            or audit["moses"]["manifest_sha256"] != config["moses_manifest_sha256"]
            or audit["moses"]["dataset_sha256"] != moses["dataset_sha256"]
            or audit["moses"]["graph_spec_fingerprint"] != spec.fingerprint()
            or audit["rdkit_version"] != rdBase.rdkitVersion or moses["rdkit_version"] != rdBase.rdkitVersion):
        raise ValueError("Audit, MOSES, graph schema and RDKit identities differ")
    eligible = audit_path.parent / "eligible.csv.gz"
    if audit["eligible"]["path"] != eligible.name or file_sha256(eligible) != audit["eligible"]["sha256"]:
        raise ValueError("Audited ChEMBL table hash differs")
    targets = {name: config["chembl_holdout_counts"][name] for name in ("val", "test")}
    if any(type(count) is not int or count < 1 for count in targets.values()):
        raise ValueError("ChEMBL holdout targets must be positive integers")
    groups = Counter(scaffold for _, scaffold, _ in _eligible_rows(eligible) if scaffold)
    if len(groups) != audit["scaffolds"]["new_vs_moses_all"]:
        raise ValueError("Unseen scaffold inventory differs from the audit")
    holdouts = _holdout_groups(groups, config["split_seed"], targets)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    chembl_count = audit["eligible"]["count"]
    chembl_counts = _write_graphs(output, eligible, spec, holdouts, chembl_count, workers)
    for name in targets:
        expected = sum(groups[group] for group, label in holdouts.items() if label == SPLIT_NAMES[name])
        if chembl_counts[name] != expected:
            raise ValueError("Encoded holdout count differs from scaffold assignment")
    if not chembl_counts["train"]:
        raise ValueError("No ChEMBL training molecules remained after the holdouts")
    moses_counts = {name: moses["counts"][name] for name in SPLIT_NAMES}
    counts = {name: moses_counts[name] + chembl_counts[name] for name in SPLIT_NAMES}
    counts["total"] = sum(counts.values())
    manifest = dict(version=VERSION, spec=moses["spec"], spec_fingerprint=spec.fingerprint(),
        rdkit_version=rdBase.rdkitVersion, counts=counts,
        sources={"moses": dict(path=str(moses_path), manifest_sha256=config["moses_manifest_sha256"],
                             dataset_sha256=moses["dataset_sha256"], counts=moses_counts),
                 "chembl": dict(audit_summary_sha256=config["audit_summary_sha256"],
                     eligible_sha256=audit["eligible"]["sha256"], counts=chembl_counts,
                     holdout_targets=targets, holdout_scaffolds={name: sum(
                         label == SPLIT_NAMES[name] for label in holdouts.values()) for name in targets})},
        files={name: dict(bytes=(output / name).stat().st_size, sha256=file_sha256(output / name)) for name in FILES},
        provenance=dict(canonicalization="RDKit canonical stereo-free SMILES; reparsed for ordered atom slots",
                        chembl_split_seed=config["split_seed"],
                        chembl_split_policy="whole atom-typed Murcko scaffolds absent from all MOSES splits"))
    manifest["dataset_sha256"] = large_manifest_digest(manifest)
    write_json(output / "manifest.json", manifest)
    print(json.dumps(dict(stage="complete", seconds=time.monotonic() - started, counts=counts,
                          chembl_counts=chembl_counts, dataset_sha256=manifest["dataset_sha256"])), flush=True)
    return manifest


class _Smiles(Sequence):
    def __init__(self, moses, corpus, indices):
        self.moses, self.corpus, self.indices = moses, corpus, indices

    def __len__(self):
        return len(self.moses) + len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.moses):
            return self.moses[index]
        row = self.indices[index - len(self.moses)]
        offsets = self.corpus.smiles_offsets
        return self.corpus.smiles_bytes[offsets[row]:offsets[row + 1]].tobytes().decode("utf-8")


class _CombinedSplit(Dataset):
    def __init__(self, corpus, split, source):
        self.corpus, self.spec, self.split = corpus, corpus.spec, split
        self.manifest = corpus.manifest
        self.moses = corpus._moses_split(split) if source != "chembl" else None
        self.moses_count = len(self.moses) if self.moses is not None else 0
        self.indices = (np.flatnonzero(corpus.labels == SPLIT_NAMES[split]) if source != "moses"
                        else np.empty(0, dtype=np.int64))
        self.smiles = _Smiles(self.moses.smiles if self.moses is not None else (), corpus, self.indices)

    def __len__(self):
        return self.moses_count + len(self.indices)

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.moses_count:
            return self.moses[index]
        corpus, row = self.corpus, self.indices[index - self.moses_count]
        left, right, types = corpus.edges[corpus.edge_offsets[row]:corpus.edge_offsets[row + 1]].T
        bonds = np.zeros((self.spec.max_nodes, self.spec.max_nodes), dtype=np.int64)
        bonds[left, right] = bonds[right, left] = types
        return dict(atoms=torch.tensor(corpus.atoms[row], dtype=torch.long),
                    charges=torch.tensor(corpus.charges[row], dtype=torch.long), bonds=torch.from_numpy(bonds))


class CombinedCorpus:
    """Authenticated read-only corpus with split views for both sources."""

    def __init__(self, path, *, expected_sha256=None):
        path = Path(path).resolve(strict=True)
        self.manifest = manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        self.spec = GraphSpec.from_dict(manifest["spec"])
        self.dataset_sha256 = manifest["dataset_sha256"]
        if (manifest["version"] != VERSION or large_manifest_digest(manifest) != self.dataset_sha256
                or expected_sha256 is not None and expected_sha256 != self.dataset_sha256
                or self.spec.fingerprint() != manifest["spec_fingerprint"]
                or manifest["rdkit_version"] != rdBase.rdkitVersion or set(manifest["files"]) != set(FILES)):
            raise ValueError("Combined corpus schema or identity differs")
        for name, metadata in manifest["files"].items():
            if (path / name).stat().st_size != metadata["bytes"] or file_sha256(path / name) != metadata["sha256"]:
                raise ValueError(f"Combined cache data hash differs: {name}")
        moses = manifest["sources"]["moses"]
        self.moses_path = Path(moses["path"])
        source = _checked_json(self.moses_path / "manifest.json", moses["manifest_sha256"])
        if source["dataset_sha256"] != moses["dataset_sha256"] or source["spec"] != manifest["spec"]:
            raise ValueError("Combined MOSES source differs")
        self._moses_receipt = validate_large_cache(self.moses_path) if source["dataset_format_version"] == 2 else None
        self._moses_views = {}

        def read(name):
            return np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)

        self.atoms, self.charges, self.labels = read("atoms"), read("charges"), read("split")
        self.edge_offsets, self.smiles_offsets = read("edge_offsets"), read("smiles_offsets")
        self.smiles_bytes = np.memmap(path / "smiles.utf8", dtype=np.uint8, mode="r")
        edge_bytes = manifest["files"]["edges.bin"]["bytes"]
        if edge_bytes % 3:
            raise ValueError("ChEMBL sparse bond records have inconsistent byte count")
        self.edges = (np.memmap(path / "edges.bin", dtype=np.uint8, mode="r", shape=(edge_bytes // 3, 3))
                      if edge_bytes else np.empty((0, 3), dtype=np.uint8))
        chembl_counts = manifest["sources"]["chembl"]["counts"]
        count = sum(chembl_counts.values())
        if (self.atoms.shape != (count, self.spec.max_nodes) or self.charges.shape != self.atoms.shape
                or self.labels.shape != (count,) or any(a.dtype != np.int8 for a in (self.atoms, self.charges, self.labels))
                or np.any((self.labels < 0) | (self.labels > 2))
                or any(np.count_nonzero(self.labels == label) != chembl_counts[name]
                       or manifest["counts"][name] != moses["counts"][name] + chembl_counts[name]
                       for name, label in SPLIT_NAMES.items())):
            raise ValueError("Combined cache arrays or split counts differ")
        for offsets, size in ((self.edge_offsets, len(self.edges)), (self.smiles_offsets, len(self.smiles_bytes))):
            if (offsets.shape != (count + 1,) or offsets.dtype != np.int64
                    or offsets[0] != 0 or offsets[-1] != size or np.any(offsets[1:] < offsets[:-1])):
                raise ValueError("Combined cache sparse offsets differ")

    def _moses_split(self, split):
        if split not in self._moses_views:
            self._moses_views[split] = GraphDataset(self.moses_path, split, validated_cache=self._moses_receipt)
        return self._moses_views[split]

    def split(self, split, *, source="all"):
        if split not in SPLIT_NAMES or source not in ("all", "moses", "chembl"):
            raise ValueError("Unknown combined split or source")
        return _CombinedSplit(self, split, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    prepare_combined(json.loads(args.config.read_text(encoding="utf-8")), args.output, workers=args.workers)


if __name__ == "__main__":
    main()
