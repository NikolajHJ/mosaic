"""Pack canonical training graphs into quota-bounded, memory-mapped sparse splits."""

from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
import re
import struct
from tempfile import TemporaryDirectory
import time

import numpy as np
from rdkit import rdBase
import torch
from torch.utils.data import Dataset

from latentmol.chemistry import MoleculeRejected, encode_canonical_smiles
from latentmol.datasets.chembl_audit import _ordered_results
from latentmol.datasets.data import SPLIT_NAMES, large_manifest_digest
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec


VERSION = "packed-sparse-graph-cache-v1"
CANONICALIZATION = "RDKit canonical stereo-free SMILES; reparsed for ordered atom slots"
MANIFEST_RESERVE_BYTES = 32 * 1024
HEADER = struct.Struct("<BH")
OFFSET = struct.Struct("<Q")
SOURCE_LABEL = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
CHECK_BYTES = 32 * 1024**2
CHECK_SECONDS = 60


def _spec_capacity(spec):
    if spec.max_nodes > 255 or max(map(len, (spec.atomic_numbers, spec.formal_charges, spec.bond_names))) > 128:
        raise ValueError("Sparse cache supports at most 255 nodes and 128 labels per feature")


def _encode_batch(batch, spec):
    packed = []
    for split, smiles, source in batch:
        if split not in SPLIT_NAMES or SOURCE_LABEL.fullmatch(source) is None:
            raise ValueError("Invalid canonical corpus row split or source")
        try:
            graph = encode_canonical_smiles(smiles, spec)
        except MoleculeRejected as error:
            raise ValueError(f"Corpus SMILES is not canonical or representable: {split}/{source} {smiles!r}") from error
        atoms, charges, bonds = (graph[key] for key in ("atom_types", "charges", "bond_types"))
        size = np.count_nonzero(atoms)
        left, right = np.nonzero(bonds[:size, :size])
        upper = left < right
        left, right = left[upper], right[upper]
        edges = np.column_stack((left, right, bonds[left, right])).astype(np.uint8)
        record = HEADER.pack(size, len(edges)) + atoms[:size].tobytes() + charges[:size].tobytes()
        packed.append((split, source, record + edges.tobytes() + smiles.encode("utf-8")))
    return packed


class _SplitWriter:
    """One packed record stream and a little-endian byte-offset index."""

    def __init__(self, directory, split, stack):
        self.graph_name = f"{split}.graphs.bin"
        self.offset_name = f"{split}.offsets.bin"
        self.graphs = stack.enter_context((directory / self.graph_name).open("xb"))
        self.offsets = stack.enter_context((directory / self.offset_name).open("xb"))
        sentinel = OFFSET.pack(0)
        self.offsets.write(sentinel)
        self.graph_hash = hashlib.sha256()
        self.offset_hash = hashlib.sha256(sentinel)
        self.graph_bytes = 0
        self.sources = Counter()
        self.count = 0

    def append(self, source, record):
        self.graph_bytes += len(record)
        end = OFFSET.pack(self.graph_bytes)
        self.graphs.write(record)
        self.offsets.write(end)
        self.graph_hash.update(record)
        self.offset_hash.update(end)
        self.count += 1
        self.sources[source] += 1

    def metadata(self):
        return {
            self.graph_name: {"bytes": self.graph_bytes, "sha256": self.graph_hash.hexdigest()},
            self.offset_name: {"bytes": OFFSET.size * (self.count + 1), "sha256": self.offset_hash.hexdigest()},
        }


def _current_cap(max_bytes, check_storage, written):
    available = check_storage(written) if check_storage is not None else None
    if available is not None and type(available) is not int:
        raise ValueError("Storage callback must return an integer cap or None")
    return min(max_bytes, available) if available is not None else max_bytes


def write_sparse_cache(rows, output, *, spec, source_dataset_sha256, max_bytes, workers=1,
                       check_storage=None, selection=None, expected_holdout_counts=None, should_stop=None):
    """Write unique ``(split, canonical_smiles, source)`` rows in supplied order.

    Validation/test rows must precede train rows and always fit in full. The
    first train row that cannot fit ends the deterministic training prefix.
    The storage callback receives bytes already staged and can shrink the total
    cap as disk headroom changes. A true stop callback also ends the train prefix
    early. Partial output remains hidden until rename.
    """
    _spec_capacity(spec)
    fixed_bytes = len(SPLIT_NAMES) * OFFSET.size
    if (type(workers) is not int or workers < 1 or type(max_bytes) is not int
            or max_bytes <= MANIFEST_RESERVE_BYTES + fixed_bytes
            or not isinstance(source_dataset_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", source_dataset_sha256) is None):
        raise ValueError("Require positive workers, sufficient byte budget and a source dataset SHA-256")
    if expected_holdout_counts is not None and (set(expected_holdout_counts) != {"val", "test"}
            or any(type(count) is not int or count < 1 for count in expected_holdout_counts.values())):
        raise ValueError("Expected holdout counts must give positive validation and test sizes")
    cap = _current_cap(max_bytes, check_storage, 0)
    if cap <= MANIFEST_RESERVE_BYTES + fixed_bytes:
        raise OSError("Insufficient storage for sparse cache metadata and split indexes")
    output = Path(output).absolute()
    if os.path.lexists(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".sparse-cache-", dir=output.parent) as staging:
        stage = Path(staging) / output.name
        stage.mkdir()
        written = checked_bytes = fixed_bytes
        checked_time = time.monotonic()
        started_train = stopped_for_budget = stopped_for_time = False
        rows_written = 0
        with ExitStack() as stack:
            splits = {name: _SplitWriter(stage, name, stack) for name in SPLIT_NAMES}
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers)) if workers > 1 else None
            for split, source, record in _ordered_results(rows, _encode_batch, pool, workers, spec):
                if started_train and split != "train":
                    raise ValueError("Validation and test rows must precede the training rows")
                started_train = started_train or split == "train"
                if split == "train" and should_stop is not None and should_stop():
                    stopped_for_time = True
                    break
                cost = len(record) + OFFSET.size
                now = time.monotonic()
                if check_storage is not None and (written - checked_bytes >= CHECK_BYTES
                        or now - checked_time >= CHECK_SECONDS or written + cost + MANIFEST_RESERVE_BYTES > cap):
                    for writer in splits.values():
                        writer.graphs.flush()
                        writer.offsets.flush()
                    cap = _current_cap(max_bytes, check_storage, written)
                    checked_bytes, checked_time = written, now
                if written + cost + MANIFEST_RESERVE_BYTES > cap:
                    if split != "train":
                        raise OSError(f"Storage cap cannot fit all {split} rows")
                    stopped_for_budget = True
                    break
                splits[split].append(source, record)
                written += cost
                rows_written += 1
                if rows_written % 250_000 == 0:
                    print(json.dumps(dict(stage="pack_sparse_cache", rows=rows_written,
                                          train=splits["train"].count, file_bytes=written, cap_bytes=cap)), flush=True)
        counts = {name: writer.count for name, writer in splits.items()}
        if not all(counts.values()):
            raise ValueError("Sparse cache needs nonempty train, validation and test splits")
        if expected_holdout_counts is not None and any(counts[name] != count for name, count in expected_holdout_counts.items()):
            raise ValueError("Sparse cache validation/test counts differ from the source corpus")
        counts["total"] = sum(counts.values())
        files = {name: meta for writer in splits.values() for name, meta in writer.metadata().items()}
        if (sum(meta["bytes"] for meta in files.values()) != written
                or any((stage / name).stat().st_size != meta["bytes"] for name, meta in files.items())):
            raise OSError("Sparse cache byte accounting differs from the staged files")
        cap = _current_cap(max_bytes, check_storage, written)
        manifest = dict(version=VERSION, spec=spec.to_dict(), spec_fingerprint=spec.fingerprint(),
                        rdkit_version=rdBase.rdkitVersion, source_dataset_sha256=source_dataset_sha256,
                        provenance={"canonicalization": CANONICALIZATION, "selection": selection or {}},
                        counts=counts, source_counts={name: dict(writer.sources) for name, writer in splits.items()},
                        files=files, budget=dict(requested_bytes=max_bytes, effective_bytes=cap, file_bytes=written,
                            stopped_for_budget=stopped_for_budget, stopped_for_time=stopped_for_time,
                            manifest_reserve_bytes=MANIFEST_RESERVE_BYTES))
        manifest["dataset_sha256"] = large_manifest_digest(manifest)
        metadata = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if len(metadata) > MANIFEST_RESERVE_BYTES or written + len(metadata) > cap:
            raise OSError("Sparse cache and manifest exceed the current storage budget")
        (stage / "manifest.json").write_bytes(metadata)
        if os.path.lexists(output):
            raise FileExistsError(output)
        stage.rename(output)
    return manifest


@dataclass(frozen=True)
class ValidatedSparseCache:
    path: Path
    dataset_sha256: str
    file_stats: tuple[tuple[str, int, int], ...]


def _manifest(cache):
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    spec = GraphSpec.from_dict(manifest["spec"])
    _spec_capacity(spec)
    if (manifest.get("version") != VERSION or manifest.get("spec_fingerprint") != spec.fingerprint()
            or manifest.get("dataset_sha256") != large_manifest_digest(manifest)):
        raise ValueError("Sparse cache manifest or graph schema differs from its fingerprint")
    return manifest, spec


def _file_stats(path):
    stat = path.stat()
    return path.name, stat.st_size, stat.st_mtime_ns


def validate_sparse_cache(cache, *, expected_sha256=None):
    """Verify every file and index once, then share the receipt across splits."""
    cache = Path(cache).resolve(strict=True)
    manifest, _ = _manifest(cache)
    digest = manifest["dataset_sha256"]
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Sparse cache differs from the expected dataset")
    files = manifest["files"]
    counts = manifest["counts"]
    if (set(files) != {f"{split}.{kind}.bin" for split in SPLIT_NAMES for kind in ("graphs", "offsets")}
            or set(counts) != {*SPLIT_NAMES, "total"}
            or any(type(count) is not int or count < 1 for count in counts.values())
            or counts["total"] != sum(counts[split] for split in SPLIT_NAMES)):
        raise ValueError("Sparse cache has unexpected files or split counts")
    stats = []
    for name, metadata in files.items():
        path = cache / name
        before = _file_stats(path)
        if before[1] != metadata["bytes"] or file_sha256(path) != metadata["sha256"] or _file_stats(path) != before:
            raise ValueError(f"Sparse cache file differs from its manifest: {name}")
        stats.append(before)
    for split in SPLIT_NAMES:
        offset_name, graph_name = f"{split}.offsets.bin", f"{split}.graphs.bin"
        count = counts[split]
        if files[offset_name]["bytes"] != (count + 1) * OFFSET.size:
            raise ValueError(f"Sparse cache index size differs from the {split} count")
        offsets = np.memmap(cache / offset_name, dtype="<u8", mode="r")
        if offsets[0] != 0 or offsets[-1] != files[graph_name]["bytes"]:
            raise ValueError(f"Sparse cache {split} index does not cover its records")
        for start in range(0, count, 131072):
            end = min(start + 131072, count)
            if np.any(offsets[start:end] >= offsets[start + 1:end + 1]):
                raise ValueError(f"Sparse cache {split} record offsets are not strictly increasing")
        del offsets
    return ValidatedSparseCache(cache, digest, tuple(stats))


class _Smiles(Sequence):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[row] for row in range(*index.indices(len(self)))]
        record, _, _, smiles_start = self.dataset._record(index)
        return record[smiles_start:].tobytes().decode("utf-8")


class SparseGraphDataset(Dataset):
    """GraphDataset-compatible samples with lazy SMILES and on-demand dense bonds."""

    def __init__(self, cache, split, *, validated_cache=None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"Unknown sparse cache split: {split}")
        self.cache_dir = Path(cache).resolve(strict=True)
        self.manifest, self.spec = _manifest(self.cache_dir)
        self.dataset_sha256 = self.manifest["dataset_sha256"]
        receipt = validated_cache or validate_sparse_cache(self.cache_dir)
        if (receipt.path != self.cache_dir or receipt.dataset_sha256 != self.dataset_sha256
                or any(_file_stats(self.cache_dir / stat[0]) != stat for stat in receipt.file_stats)):
            raise ValueError("Sparse cache changed or the validation receipt belongs to another dataset")
        self.split = split
        self._neutral_charge = self.spec.formal_charges.index(0)
        self._offsets = self._graphs = None

    @property
    def smiles(self):
        return _Smiles(self)

    def __len__(self):
        return self.manifest["counts"][self.split]

    def __getstate__(self):
        # Spawned workers reopen immutable maps instead of pickling their data.
        return dict(self.__dict__, _offsets=None, _graphs=None)

    def _open_maps(self):
        if self._offsets is None:
            self._offsets = np.memmap(self.cache_dir / f"{self.split}.offsets.bin", dtype="<u8", mode="r")
            self._graphs = np.memmap(self.cache_dir / f"{self.split}.graphs.bin", dtype=np.uint8, mode="r")

    def active_counts(self, start, end):
        """Read exact occupied counts from record headers without decoding graphs."""
        if not 0 <= start <= end <= len(self):
            raise IndexError((start, end))
        self._open_maps()
        return self._graphs[self._offsets[start:end]]

    def _record(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._open_maps()
        start, end = int(self._offsets[index]), int(self._offsets[index + 1])
        record = self._graphs[start:end]
        if len(record) < HEADER.size:
            raise ValueError("Sparse graph record has no complete header")
        size, edge_count = HEADER.unpack_from(record)
        smiles_start = HEADER.size + 2 * size + 3 * edge_count
        if (not 1 <= size <= self.spec.max_nodes or edge_count > size * (size - 1) // 2
                or smiles_start >= len(record)):
            raise ValueError("Sparse graph record has an invalid size")
        return record, size, edge_count, smiles_start

    def __getitem__(self, index):
        record, size, edge_count, smiles_start = self._record(index)
        atoms = np.zeros(self.spec.max_nodes, dtype=np.int64)
        charges = np.full(self.spec.max_nodes, self._neutral_charge, dtype=np.int64)
        bonds = np.zeros((self.spec.max_nodes, self.spec.max_nodes), dtype=np.int64)
        atoms[:size] = record[HEADER.size:HEADER.size + size]
        charges[:size] = record[HEADER.size + size:HEADER.size + 2 * size]
        left, right, labels = record[smiles_start - 3 * edge_count:smiles_start].reshape(-1, 3).T
        bonds[left, right] = bonds[right, left] = labels
        return {"atoms": torch.from_numpy(atoms), "charges": torch.from_numpy(charges), "bonds": torch.from_numpy(bonds)}
