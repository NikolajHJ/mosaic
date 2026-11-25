"""Build a size-balanced training view over an immutable packed graph cache."""

from collections.abc import Sequence
import json
import operator
import os
from pathlib import Path
from shutil import disk_usage
from tempfile import TemporaryDirectory

import numpy as np
from torch.utils.data import Dataset

from latentmol.datasets.bulk_cache import SparseGraphDataset, validate_sparse_cache
from latentmol.datasets.bulk_cache_build import CanonicalMembership
from latentmol.datasets.bulk_split_policy import capped_waterfill, select_pubchem_holdouts, source_aligned_candidates
from latentmol.datasets.data import SPLIT_NAMES, large_manifest_digest
from latentmol.experiments import file_sha256, write_json


VERSION = "balanced-packed-sparse-view-v1"
INDEX_FILES = {"train": "train.u32", "val": "val-added.u32", "test": "test-added.u32"}
INDEX_TYPE = np.dtype("<u4")
CHUNK = 1_000_000
MANIFEST_RESERVE = 32 * 1024


def _progress(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def _json_counts(counts):
    return {str(size): count for size, count in counts.items()}


def _sizes_and_counts(dataset):
    max_nodes = dataset.spec.max_nodes
    sizes = np.empty(len(dataset), dtype=np.uint8)
    counts = np.zeros(max_nodes + 1, dtype=np.int64)
    for start in range(0, len(dataset), CHUNK):
        end = min(start + CHUNK, len(dataset))
        batch = dataset.active_counts(start, end)
        if np.any((batch == 0) | (batch > max_nodes)):
            raise ValueError(f"Invalid occupied counts in parent {dataset.split} rows {start}:{end}")
        sizes[start:end] = batch
        counts += np.bincount(batch, minlength=max_nodes + 1)
        if dataset.split == "train" and (end % (5 * CHUNK) == 0 or end == len(dataset)):
            _progress("count_active_atoms", rows=end, total=len(dataset))
    return sizes, {size: int(counts[size]) for size in range(1, max_nodes + 1)}


def _write_train(path, sizes, quotas, excluded):
    """Take the seeded parent prefix in each size bin, omitting new holdouts."""
    remaining = np.zeros(max(quotas) + 1, dtype=np.int64)
    for size, count in quotas.items():
        remaining[size] = count
    done = 0
    with path.open("xb") as stream:
        for start in range(0, len(sizes), CHUNK):
            end = min(start + CHUNK, len(sizes))
            batch = sizes[start:end].copy()
            left, right = excluded.searchsorted([start, end])
            batch[excluded[left:right] - start] = 0
            present = np.bincount(batch, minlength=len(remaining))
            present[0] = 0
            keep = ((remaining >= present) & (present > 0))[batch]
            for size in np.flatnonzero((remaining > 0) & (remaining < present)):
                keep[np.flatnonzero(batch == size)[:remaining[size]]] = True
            indices = (np.flatnonzero(keep) + start).astype(INDEX_TYPE)
            stream.write(indices.tobytes())
            done += len(indices)
            remaining -= np.minimum(remaining, present)
            if end % (5 * CHUNK) == 0 or end == len(sizes) or not np.any(remaining):
                _progress("select_balanced_train", scanned=end, selected=done, total=len(sizes))
            if not np.any(remaining):
                break
    if np.any(remaining) or done != sum(quotas.values()):
        raise ValueError("Source cache did not supply the promised training size quotas")


def _parent(corpus, cache_path, expected_cache_sha256, corpus_manifest_sha256):
    receipt = validate_sparse_cache(cache_path, expected_sha256=expected_cache_sha256)
    train = SparseGraphDataset(cache_path, "train", validated_cache=receipt)
    manifest = train.manifest
    selection = manifest.get("provenance", {}).get("selection", {})
    seed = selection.get("seed")
    if (manifest["source_dataset_sha256"] != corpus.dataset_sha256 or manifest["counts"] != corpus.counts
            or train.spec != corpus.spec or manifest["rdkit_version"] != corpus.manifest["rdkit_version"]
            or selection.get("canonical_manifest_sha256") != corpus_manifest_sha256
            or selection.get("rule") != "seeded_blake2b_bucket_and_row_prefix"
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("Balanced view needs the complete, authenticated source corpus in seeded cache order")
    return receipt, train, seed


def build_balanced_view(corpus_path, cache_path, output, *, corpus_manifest_sha256, expected_cache_sha256,
                        train_count=60_000_000, add_per_holdout=100_000, seed=550019, max_output_bytes=None):
    """Build exact, disjoint split indexes and preserve both original holdouts."""
    if (type(train_count) is not int or train_count < 1 or type(add_per_holdout) is not int or add_per_holdout < 1
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("Require positive train and holdout budgets and a 64-bit seed")
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    projected = (train_count + 2 * add_per_holdout) * INDEX_TYPE.itemsize + MANIFEST_RESERVE
    if (max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes < projected)
            or disk_usage(output.parent).free < projected):
        raise OSError("Insufficient storage for the requested balanced split indexes")
    corpus = CanonicalMembership(corpus_path, manifest_sha256=corpus_manifest_sha256)
    if not train_count + 2 * add_per_holdout <= corpus.counts["train"] <= 2**32:
        raise ValueError("Source train count cannot supply the requested view or exceed uint32 row capacity")
    receipt, parent_train, parent_seed = _parent(corpus, cache_path, expected_cache_sha256, corpus_manifest_sha256)
    sizes, capacities = _sizes_and_counts(parent_train)
    quotas = capped_waterfill(capacities, train_count, seed=seed)
    candidates = source_aligned_candidates(corpus.rows(parent_seed), parent_train.smiles, sizes)
    added = select_pubchem_holdouts(candidates, capacities=capacities, train_quotas=quotas,
                                    count_per_split=add_per_holdout)
    indexes = {split: np.asarray(rows, dtype=INDEX_TYPE) for split, rows in added.items()}
    excluded = np.sort(np.concatenate(tuple(indexes.values())))
    if len(excluded) != 2 * add_per_holdout or np.any(excluded[:-1] >= excluded[1:]):
        raise ValueError("New validation/test indices are not unique and disjoint")
    max_nodes = parent_train.spec.max_nodes
    atom_counts = {"train": quotas}
    added_counts = {}
    for split, index in indexes.items():
        original = SparseGraphDataset(cache_path, split, validated_cache=receipt)
        _, original_counts = _sizes_and_counts(original)
        additions = np.bincount(sizes[index], minlength=max_nodes + 1)
        added_counts[split] = {size: int(additions[size]) for size in capacities}
        atom_counts[split] = {size: original_counts[size] + int(additions[size]) for size in capacities}
    _progress("holdouts_selected", validation=len(indexes["val"]), test=len(indexes["test"]))
    with TemporaryDirectory(prefix=".balanced-view-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        for split, index in indexes.items():
            (stage / INDEX_FILES[split]).write_bytes(index.tobytes())
        _write_train(stage / INDEX_FILES["train"], sizes, quotas, excluded)
        files = {name: {"bytes": (stage / name).stat().st_size, "sha256": file_sha256(stage / name)}
                 for name in INDEX_FILES.values()}
        counts = {"train": train_count, **{split: corpus.counts[split] + add_per_holdout for split in indexes}}
        counts["total"] = sum(counts.values())
        manifest = dict(version=VERSION, spec=parent_train.manifest["spec"],
                        spec_fingerprint=parent_train.spec.fingerprint(), rdkit_version=parent_train.manifest["rdkit_version"],
                        counts=counts, active_atom_counts={split: _json_counts(hist) for split, hist in atom_counts.items()},
                        added_holdout_active_atom_counts={split: _json_counts(hist) for split, hist in added_counts.items()},
                        available_train_active_atom_counts=_json_counts(capacities), files=files,
                        provenance={"canonicalization": parent_train.manifest["provenance"]["canonicalization"],
                                    "parent_cache": str(parent_train.cache_dir),
                                    "parent_cache_sha256": receipt.dataset_sha256,
                                    "canonical_corpus_sha256": corpus.dataset_sha256,
                                    "canonical_manifest_sha256": corpus_manifest_sha256,
                                    "source_counts": corpus.counts, "parent_selection_seed": parent_seed,
                                    "waterfill_tie_seed": seed, "added_holdout_source": "pubchem",
                                    "added_holdouts_per_split": add_per_holdout,
                                    "split_policy": "Capped waterfill by heavy atom count; use only surplus PubChem rows for added molecule-disjoint holdouts"})
        manifest["dataset_sha256"] = large_manifest_digest(manifest)
        manifest_path = stage / "manifest.json"
        write_json(manifest_path, manifest)
        if sum(meta["bytes"] for meta in files.values()) + manifest_path.stat().st_size > projected:
            raise OSError("Balanced view exceeded its reserved output bytes")
        if os.path.lexists(output):
            raise FileExistsError(output)
        stage.rename(output)
    _progress("balanced_view_ready", counts=counts, dataset_sha256=manifest["dataset_sha256"])
    return manifest


def _check_index(path, metadata, length, parent_count):
    if path.stat().st_size != length * INDEX_TYPE.itemsize or path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Invalid balanced view index size: {path.name}")
    if file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"Balanced view index hash differs: {path.name}")
    values = np.memmap(path, dtype=INDEX_TYPE, mode="r")
    if int(values[-1]) >= parent_count:
        raise ValueError(f"Balanced view index exceeds the parent row count: {path.name}")
    for start in range(0, length - 1, CHUNK):
        end = min(start + CHUNK, length - 1)
        if np.any(values[start:end] >= values[start + 1:end + 1]):
            raise ValueError(f"Balanced view index is not strictly increasing: {path.name}")
    return values


def _overlap(sorted_rows, candidates):
    positions = np.minimum(sorted_rows.searchsorted(candidates), len(sorted_rows) - 1)
    return np.any(sorted_rows[positions] == candidates)


class BalancedSparseCorpus:
    """Verify a balanced view and open each split without rewriting graphs."""

    def __init__(self, path, *, expected_sha256=None):
        self.path = Path(path).resolve(strict=True)
        self.manifest = manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        self.dataset_sha256 = manifest["dataset_sha256"]
        if (manifest["version"] != VERSION or self.dataset_sha256 != large_manifest_digest(manifest)
                or expected_sha256 is not None and self.dataset_sha256 != expected_sha256):
            raise ValueError("Balanced view manifest differs from the expected dataset")
        provenance = manifest["provenance"]
        self._source = Path(provenance["parent_cache"])
        self._receipt = validate_sparse_cache(self._source, expected_sha256=provenance["parent_cache_sha256"])
        self._train = SparseGraphDataset(self._source, "train", validated_cache=self._receipt)
        self.spec = self._train.spec
        if (manifest["spec"] != self.spec.to_dict() or manifest["spec_fingerprint"] != self.spec.fingerprint()
                or provenance["source_counts"] != self._train.manifest["counts"]
                or set(manifest["files"]) != set(INDEX_FILES.values())):
            raise ValueError("Balanced view is incompatible with its source cache")
        counts = manifest["counts"]
        lengths = {"train": counts["train"], **{split: counts[split] - provenance["source_counts"][split]
                                               for split in ("val", "test")}}
        if (counts["total"] != sum(counts[split] for split in SPLIT_NAMES) or any(length < 1 for length in lengths.values())
                or any(lengths[split] != provenance["added_holdouts_per_split"] for split in ("val", "test"))):
            raise ValueError("Balanced view has inconsistent split counts")
        train, val, test = (_check_index(self.path / INDEX_FILES[split], manifest["files"][INDEX_FILES[split]],
                                        lengths[split], len(self._train)) for split in ("train", "val", "test"))
        if _overlap(train, val) or _overlap(train, test) or _overlap(val, test):
            raise ValueError("Balanced view splits overlap")

    def split(self, split):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown balanced view split: {split}")
        base = SparseGraphDataset(self._source, split, validated_cache=self._receipt) if split != "train" else None
        return _IndexedSplit(self.path, self.manifest, self._train, base, split)


class _IndexedSmiles(Sequence):
    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        dataset, row = self._dataset._row(index)
        return dataset.smiles[row]


class _IndexedSplit(Dataset):
    def __init__(self, path, manifest, train, base, split):
        self.cache_dir, self.manifest, self.split = path, manifest, split
        self.spec, self.dataset_sha256 = train.spec, manifest["dataset_sha256"]
        self._train, self._base = train, base
        self._base_count = len(base) if base is not None else 0
        self._indices = None

    def __len__(self):
        return self.manifest["counts"][self.split]

    @property
    def smiles(self):
        return _IndexedSmiles(self)

    def __getstate__(self):
        return dict(self.__dict__, _indices=None)

    def _row(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self._base_count:
            return self._base, index
        if self._indices is None:
            self._indices = np.memmap(self.cache_dir / INDEX_FILES[self.split], dtype=INDEX_TYPE, mode="r")
        return self._train, int(self._indices[index - self._base_count])

    def __getitem__(self, index):
        dataset, row = self._row(index)
        return dataset[row]
