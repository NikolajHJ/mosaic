"""Prepare auditable local molecular graph caches; never download source data.

Canonical, stereo-free molecules are deduplicated before random subset selection
and a molecule split (80/10/10 by default, or explicit counts). This is not a scaffold or target-aware split, and
does not establish generalization to novel scaffolds. Chemistry outside the graph
schema is counted and rejected. The original NPZ format is unchanged; the
separate large-corpus builder writes version-2 memory-mapped arrays.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import zipfile

import numpy as np
from rdkit import rdBase
import torch
from torch.utils.data import Dataset

from .chemistry import MoleculeRejected, encode_smiles
from .representation import GraphSpec, HardGraphBatch


SPLIT_NAMES = {"train": 0, "val": 1, "test": 2}
LARGE_DATA_FILES = ("atom_types.npy", "charges.npy", "bond_types.npy", "split.npy",
                    "smiles_offsets.npy", "smiles.utf8")


def sha256_file(path: str | Path) -> str:
    """Compute a file digest with bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_smiles(path: Path, smiles_column: str) -> Iterable[str]:
    """Yield CSV field values or the first whitespace-delimited .smi token."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() in {".smi", ".smiles"}:
            for line in stream:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    yield stripped.split()[0]
            return
        reader = csv.DictReader(stream)
        if not reader.fieldnames or smiles_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain the SMILES column {smiles_column!r}.")
        for row in reader:
            yield row.get(smiles_column) or ""


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write deterministic compressed NPZ bytes, without pickle or timestamps."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, arrays[name], allow_pickle=False)
            entry = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload.getvalue())


def _split_digest(smiles: list[str], split: np.ndarray) -> str:
    """Hash the ordered canonical molecule-to-split assignments."""
    assignments = [[value, int(label)] for value, label in zip(smiles, split)]
    payload = json.dumps(assignments, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def large_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Identify a v2 cache by its canonical manifest, excluding its own digest.

    A creation timestamp, if an external packager adds one, is deliberately not
    part of dataset identity. The manifest's file digests bind every data byte.
    Version 1 retains its historical ``dataset.npz`` file-hash convention.
    """
    payload = {key: value for key, value in manifest.items()
               if key not in {"dataset_sha256", "created_at"}}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                           separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _large_split_digest(smiles_path: Path, offsets: np.ndarray,
                        labels: np.ndarray) -> str:
    """Hash ordered v2 identity/role pairs without materializing all SMILES."""
    digest = hashlib.sha256()
    digest.update(b"[")
    previous: str | None = None
    with smiles_path.open("rb") as stream:
        for index, label in enumerate(labels):
            expected_start = int(offsets[index])
            expected_end = int(offsets[index + 1])
            if stream.tell() != expected_start or expected_end <= expected_start:
                raise ValueError("Large-cache SMILES offsets are invalid.")
            value = stream.read(expected_end - expected_start).decode("utf-8")
            if not value or (previous is not None and previous >= value):
                raise ValueError("Large-cache SMILES are empty, duplicate or out of order.")
            if index:
                digest.update(b",")
            digest.update(json.dumps([value, int(label)], ensure_ascii=True,
                                     separators=(",", ":")).encode("utf-8"))
            previous = value
        if stream.read(1):
            raise ValueError("Large-cache SMILES offsets omit trailing bytes.")
    digest.update(b"]")
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidatedLargeCache:
    """A process-local receipt for one full v2 cache integrity pass.

    Reuse it for train/validation/test loaders in the same process. File size and
    modification time are rechecked on reuse; keep the cache immutable while a
    loader is active. A fresh process should rehash the files once.
    """

    cache_dir: Path
    dataset_sha256: str
    file_stats: tuple[tuple[str, int, int], ...]


def validate_large_cache(cache_dir: str | Path) -> ValidatedLargeCache:
    """Verify all v2 file hashes, dimensions, identities and split assignments."""
    cache = Path(cache_dir).expanduser().resolve(strict=True)
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("dataset_format_version") != 2:
        raise ValueError("Expected a version-2 large graph cache.")
    spec = GraphSpec.from_dict(manifest["spec"])
    if spec.fingerprint() != manifest.get("spec_fingerprint"):
        raise ValueError("Manifest graph specification fingerprint does not match.")
    if large_manifest_digest(manifest) != manifest.get("dataset_sha256"):
        raise ValueError("Large-cache canonical manifest digest does not match.")
    files = manifest.get("files", {})
    if set(files) != set(LARGE_DATA_FILES):
        raise ValueError("Large-cache manifest has missing or unexpected files.")
    stats = []
    for name in LARGE_DATA_FILES:
        path = cache / name
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
                or after.st_size != files[name]["bytes"] or digest != files[name]["sha256"]):
            raise ValueError(f"Large-cache file differs from its manifest: {name}")
        stats.append((name, after.st_size, after.st_mtime_ns))
    atoms = np.load(cache / "atom_types.npy", mmap_mode="r", allow_pickle=False)
    charges = np.load(cache / "charges.npy", mmap_mode="r", allow_pickle=False)
    bonds = np.load(cache / "bond_types.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(cache / "split.npy", mmap_mode="r", allow_pickle=False)
    offsets = np.load(cache / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
    count = int(manifest["counts"]["total"])
    size = spec.max_nodes
    if (count < 1 or atoms.shape != (count, size) or charges.shape != (count, size)
            or bonds.shape != (count, size, size) or labels.shape != (count,)
            or offsets.shape != (count + 1,)
            or any(array.dtype != np.dtype(np.int8) for array in (atoms, charges, bonds, labels))
            or offsets.dtype != np.dtype(np.int64)):
        raise ValueError("Large-cache arrays do not conform to the graph schema.")
    if int(offsets[0]) != 0 or int(offsets[-1]) != files["smiles.utf8"]["bytes"]:
        raise ValueError("Large-cache SMILES offsets do not span the byte stream.")
    counts = np.zeros(3, dtype=np.int64)
    for begin in range(0, count, 131072):
        part = np.asarray(labels[begin:begin + 131072])
        if np.any((part < 0) | (part > 2)):
            raise ValueError("Large-cache split labels are outside train/val/test.")
        counts += np.bincount(part, minlength=3)
    if any(int(counts[label]) != manifest["counts"][name]
           for name, label in SPLIT_NAMES.items()):
        raise ValueError("Large-cache split counts do not match its manifest.")
    if _large_split_digest(cache / "smiles.utf8", offsets, labels) != manifest["split_sha256"]:
        raise ValueError("Large-cache split digest does not match its manifest.")
    # Structural graph-label validation is chunked; it does not re-run RDKit
    # chemistry or make a validity claim about model-generated graphs.
    neutral = spec.formal_charges.index(0)
    for begin in range(0, count, 512):
        end = min(begin + 512, count)
        a = np.asarray(atoms[begin:end])
        q = np.asarray(charges[begin:end])
        b = np.asarray(bonds[begin:end])
        if (np.any((a < 0) | (a >= len(spec.atomic_numbers)))
                or np.any((q < 0) | (q >= len(spec.formal_charges)))
                or np.any((b < 0) | (b >= len(spec.bond_names)))):
            raise ValueError("Large-cache graph contains an out-of-vocabulary label.")
        active = a != 0
        if (np.any(~np.any(active, axis=1))
                or np.any(np.diff(active.astype(np.int8), axis=1) > 0)
                or np.any(q[~active] != neutral)):
            raise ValueError("Large-cache padding or occupancy is inconsistent.")
        if (not np.array_equal(b, b.transpose(0, 2, 1))
                or np.any(np.diagonal(b, axis1=1, axis2=2) != 0)
                or np.any((b != 0) & ~(active[:, :, None] & active[:, None, :]))):
            raise ValueError("Large-cache bond symmetry, diagonal or padding is inconsistent.")
    for name, size_bytes, modified in stats:
        info = (cache / name).stat()
        if info.st_size != size_bytes or info.st_mtime_ns != modified:
            raise ValueError(f"Large-cache file changed during validation: {name}")
    return ValidatedLargeCache(cache, manifest["dataset_sha256"], tuple(stats))


def _check_validated_large_cache(cache: Path, manifest: Mapping[str, Any],
                                 receipt: ValidatedLargeCache) -> None:
    """Check a previously verified immutable cache before sharing its receipt."""
    if large_manifest_digest(manifest) != manifest.get("dataset_sha256"):
        raise ValueError("Large-cache canonical manifest digest does not match.")
    if receipt.cache_dir != cache or receipt.dataset_sha256 != manifest.get("dataset_sha256"):
        raise ValueError("Validated large-cache receipt refers to another dataset.")
    for name, size, modified in receipt.file_stats:
        info = (cache / name).stat()
        if info.st_size != size or info.st_mtime_ns != modified:
            raise ValueError(f"Large-cache file changed since validation: {name}")


def _selected_large_smiles(cache: Path, offsets: np.ndarray,
                           indices: np.ndarray) -> list[str]:
    """Load only the requested split's identity strings into ordinary memory."""
    result: list[str] = []
    with (cache / "smiles.utf8").open("rb") as stream:
        for index in indices:
            start, end = int(offsets[index]), int(offsets[index + 1])
            stream.seek(start)
            result.append(stream.read(end - start).decode("utf-8"))
    return result


def assign_splits(smiles: list[str], rng, train_count: int | None = None,
                  val_count: int | None = None, reserved: dict[str, int] | None = None):
    """Assign exact counts while preserving a previous cache's split identities.

    Prior train/validation/test examples keep their roles. Remaining molecules
    receive a seeded random assignment. This prevents an expanded experiment
    from training on its previous validation or test molecules.
    """
    count = len(smiles)
    if (train_count is None) != (val_count is None):
        raise ValueError("Explicit train_count and val_count must be supplied together")
    n_train = max(1, int(0.8 * count)) if train_count is None else train_count
    n_val = int(0.1 * count) if val_count is None else val_count
    if n_train < 1 or n_val < 0 or n_train + n_val > count:
        raise ValueError("Requested split counts exceed available molecules or are invalid")
    target = np.asarray([n_train, n_val, count - n_train - n_val])
    reserved = reserved or {}
    if set(reserved) - set(smiles):
        raise ValueError("Reserved cache contains molecules absent from the selected dataset")
    if any(label not in SPLIT_NAMES.values() for label in reserved.values()):
        raise ValueError("Reserved cache contains an unknown split label")
    split = np.asarray([reserved.get(value, -1) for value in smiles], dtype=np.int8)
    retained = np.asarray([np.count_nonzero(split == label) for label in range(3)])
    needed = target - retained
    if np.any(needed < 0):
        raise ValueError("A requested split is smaller than its reserved membership")
    unassigned = np.flatnonzero(split == -1)
    order = rng.permutation(unassigned)
    offset = 0
    for label, amount in enumerate(needed):
        split[order[offset:offset + int(amount)]] = label
        offset += int(amount)
    return split


def dataset_audit(arrays: Mapping[str, np.ndarray], spec: GraphSpec) -> dict:
    """Summarize size and chemistry coverage without storing a full bond mask."""
    sizes = np.count_nonzero(arrays["atom_types"], axis=1)
    audit = {"by_split": {}, "atomic_number_counts": {}, "formal_charge_counts": {}}
    for name, label in SPLIT_NAMES.items():
        chosen = sizes[arrays["split"] == label]
        audit["by_split"][name] = {
            "molecules": int(len(chosen)),
            "heavy_atom_histogram": np.bincount(chosen, minlength=spec.max_nodes + 1).tolist(),
            "heavy_atom_quantiles_0_25_50_75_100": np.quantile(chosen, [0, .25, .5, .75, 1]).tolist() if len(chosen) else [],
        }
    atoms = arrays["atom_types"]
    for index, number in enumerate(spec.atomic_numbers[1:], 1):
        audit["atomic_number_counts"][str(number)] = int(np.count_nonzero(atoms == index))
    for index, charge in enumerate(spec.formal_charges):
        audit["formal_charge_counts"][str(charge)] = int(np.count_nonzero((arrays["charges"] == index) & (atoms != 0)))
    bond_counts = np.zeros(len(spec.bond_names), dtype=np.int64)
    upper = np.triu(np.ones((spec.max_nodes, spec.max_nodes), dtype=bool), 1)
    for begin in range(0, len(atoms), 512):
        active = atoms[begin:begin + 512] != 0
        mask = active[:, :, None] & active[:, None, :] & upper
        values = arrays["bond_types"][begin:begin + 512][mask]
        bond_counts += np.bincount(values, minlength=len(spec.bond_names))
    audit["active_upper_triangle_bond_counts"] = dict(zip(spec.bond_names, bond_counts.tolist()))
    return audit


def prepare_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    smiles_column: str = "smiles",
    spec: GraphSpec | None = None,
    limit: int | None = None,
    seed: int = 17,
    train_count: int | None = None,
    val_count: int | None = None,
    reserve_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Build a new immutable cache directory and return its provenance manifest.

    All input rows are considered before an optional uniformly random subset of
    the unique accepted molecules is drawn. Sorting canonical keys makes dedup,
    subset choice, and split independent of input row order for the same set.
    Default split counts are floor(0.8M), floor(0.1M), remainder. Explicit train
    and validation counts override them; test receives the remainder. A previous
    cache can reserve its molecule-to-split assignments before allocating the rest.

    The cache is staged in a new sibling directory and renamed only when complete.
    An existing destination is never overwritten. Source data must already exist
    locally on the machine running this function; acquisition is a separate step.
    """
    spec = spec or GraphSpec()
    source = Path(input_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer when specified.")
    raw_sha256 = sha256_file(source)
    # Keep only canonical strings until subset selection: a small --limit should
    # not retain dense N x N graphs for every molecule in a large source file.
    accepted: set[str] = set()
    rejected: Counter[str] = Counter()
    rows_seen = duplicates = roundtrip_passed = 0
    for smiles in _read_smiles(source, smiles_column):
        rows_seen += 1
        try:
            encoded = encode_smiles(smiles, spec)
        except MoleculeRejected as error:
            rejected[error.reason] += 1
            continue
        roundtrip_passed += 1
        canonical = encoded["smiles"]
        if canonical in accepted:
            duplicates += 1
        else:
            accepted.add(canonical)
    if sha256_file(source) != raw_sha256:
        raise RuntimeError("Source file changed during preparation; cache was not written.")
    if not accepted:
        raise ValueError(f"No representable molecules found; rejection reasons: {dict(rejected)}")
    canonical_smiles = sorted(accepted)
    rng = np.random.default_rng(seed)
    available_count = len(canonical_smiles)
    if limit is not None and limit < available_count:
        selected = np.sort(rng.choice(available_count, size=limit, replace=False))
        canonical_smiles = [canonical_smiles[int(index)] for index in selected]
    records = [encode_smiles(smiles, spec) for smiles in canonical_smiles]
    count = len(records)
    reserved, parent = {}, None
    if reserve_cache is not None:
        reserve_path = Path(reserve_cache).resolve(strict=True)
        old = json.loads((reserve_path / "manifest.json").read_text())
        if old["spec_fingerprint"] != spec.fingerprint() or old["raw_sha256"] != raw_sha256:
            raise ValueError("Reserved cache must share the exact graph schema and raw source")
        if sha256_file(reserve_path / "dataset.npz") != old["dataset_sha256"]:
            raise ValueError("Reserved cache dataset hash differs from its manifest")
        with np.load(reserve_path / "dataset.npz", allow_pickle=False) as archive:
            old_smiles, old_split = archive["smiles"].tolist(), archive["split"]
        if _split_digest(old_smiles, old_split) != old["split_sha256"]:
            raise ValueError("Reserved cache split hash differs from its manifest")
        if len(set(old_smiles)) != len(old_smiles):
            raise ValueError("Reserved cache contains duplicate identities")
        reserved = dict(zip(old_smiles, map(int, old_split)))
        parent = {"path": str(reserve_path), "dataset_sha256": old["dataset_sha256"],
                  "split_sha256": old["split_sha256"], "reserved_molecules": len(reserved)}
    split = assign_splits(canonical_smiles, rng, train_count, val_count, reserved)
    arrays = {
        "atom_types": np.stack([record["atom_types"] for record in records]),
        "charges": np.stack([record["charges"] for record in records]),
        "bond_types": np.stack([record["bond_types"] for record in records]),
        "smiles": np.asarray(canonical_smiles, dtype=np.str_),
        "split": split,
    }
    counts: dict[str, Any] = {
        "source_rows": rows_seen,
        "rejected_rows": sum(rejected.values()),
        "duplicate_rows": duplicates,
        "unique_accepted_before_limit": available_count,
        "excluded_by_limit": available_count - count,
        "total": count,
        "roundtrip_valid_rows": roundtrip_passed,
        "roundtrip_rejected_rows": rejected.get("representation_roundtrip", 0),
        "selected_target_graphs_roundtrip_valid": count,
        **{name: int(np.count_nonzero(split == value)) for name, value in SPLIT_NAMES.items()},
    }
    manifest: dict[str, Any] = {
        "dataset_format_version": 1,
        "spec": spec.to_dict(),
        "spec_fingerprint": spec.fingerprint(),
        "raw_sha256": raw_sha256,
        "split_sha256": _split_digest(canonical_smiles, split),
        "counts": counts,
        "rejection_reasons": dict(sorted(rejected.items())),
        "rdkit_version": rdBase.rdkitVersion,
        "numpy_version": np.__version__,
        "provenance": {
            "source_path": str(source),
            "source_format": "smi" if source.suffix.lower() in {".smi", ".smiles"} else "csv",
            "smiles_column": smiles_column,
            "seed": seed,
            "limit": limit,
            "stereochemistry": "removed explicitly before canonicalization and deduplication",
            "canonicalization": "RDKit canonical SMILES; reparsed for stable atom ordering",
            "deduplication": "canonical stereo-free SMILES before subset selection and split",
            "split_method": "seeded molecule permutation with optional reserved identities; not scaffold-separated",
            "requested_train_count": train_count, "requested_val_count": val_count,
            "reserved_cache": parent,
            "target_validation": "connected sanitized graph; exact canonical SMILES roundtrip",
            "unsupported_features": ["stereochemistry", "isotopes", "radicals",
                                     "explicit hydrogen counts that fail graph roundtrip"],
            "repairs": "none; no salt stripping, charge normalization, or tautomer normalization",
        },
    }
    manifest["audit"] = dataset_audit(arrays, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent))
    try:
        _write_npz(stage / "dataset.npz", arrays)
        manifest["dataset_sha256"] = sha256_file(stage / "dataset.npz")
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Dataset destination appeared during preparation: {destination}")
        stage.rename(destination)
    finally:
        if stage.exists():
            # Only the fresh, private staging directory allocated above is removed.
            shutil.rmtree(stage)
    return manifest


class GraphDataset(Dataset):
    """One cached split with integrity-checked data and manifest/spec provenance.

    V1 arrays are loaded into RAM; V2 arrays remain read-only memory maps, and
    only the chosen split's SMILES/row indices are materialized. Samples are
    independent long tensors with keys ``atoms``, ``charges``, and ``bonds``.
    The constructor checks stored bytes, not new chemistry rules from another
    RDKit release. Rebuild explicitly to change the graph schema.
    """

    def __init__(self, cache_dir: str | Path, split: str,
                 *, validated_cache: ValidatedLargeCache | None = None) -> None:
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {tuple(SPLIT_NAMES)}, got {split!r}")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        version = self.manifest.get("dataset_format_version")
        if version not in (1, 2):
            raise ValueError("Unsupported dataset format version.")
        self.spec = GraphSpec.from_dict(self.manifest["spec"])
        if self.spec.fingerprint() != self.manifest["spec_fingerprint"]:
            raise ValueError("Manifest graph specification fingerprint does not match.")
        self._indices: np.ndarray | None = None
        if version == 2:
            receipt = validated_cache or validate_large_cache(self.cache_dir)
            _check_validated_large_cache(self.cache_dir, self.manifest, receipt)
            atoms = np.load(self.cache_dir / "atom_types.npy", mmap_mode="r", allow_pickle=False)
            charges = np.load(self.cache_dir / "charges.npy", mmap_mode="r", allow_pickle=False)
            bonds = np.load(self.cache_dir / "bond_types.npy", mmap_mode="r", allow_pickle=False)
            labels = np.load(self.cache_dir / "split.npy", mmap_mode="r", allow_pickle=False)
            offsets = np.load(self.cache_dir / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
            selected = np.flatnonzero(labels == SPLIT_NAMES[split])
            self._atoms, self._charges, self._bonds = atoms, charges, bonds
            self._indices = selected
            self.smiles = _selected_large_smiles(self.cache_dir, offsets, selected)
            self.split = split
            return
        if validated_cache is not None:
            raise ValueError("A validated large-cache receipt cannot be used with a v1 dataset.")
        dataset_path = self.cache_dir / "dataset.npz"
        if sha256_file(dataset_path) != self.manifest["dataset_sha256"]:
            raise ValueError("Dataset file hash does not match its manifest.")
        with np.load(dataset_path, allow_pickle=False) as archive:
            atoms, charges, bonds = archive["atom_types"], archive["charges"], archive["bond_types"]
            smiles, labels = archive["smiles"], archive["split"]
        count, size = len(smiles), self.spec.max_nodes
        if (atoms.shape != (count, size) or charges.shape != (count, size)
                or bonds.shape != (count, size, size) or labels.shape != (count,)
                or smiles.ndim != 1 or smiles.dtype.kind != "U"
                or any(array.dtype != np.dtype(np.int8) for array in (atoms, charges, bonds, labels))
                or not np.all(np.isin(labels, list(SPLIT_NAMES.values())))):
            raise ValueError("Dataset arrays do not conform to the manifest graph schema.")
        if _split_digest(smiles.tolist(), labels) != self.manifest["split_sha256"]:
            raise ValueError("Dataset split assignments do not match the manifest.")
        if count != self.manifest["counts"]["total"]:
            raise ValueError("Dataset molecule count does not match the manifest.")
        selected = labels == SPLIT_NAMES[split]
        self._atoms, self._charges, self._bonds = atoms[selected], charges[selected], bonds[selected]
        self.smiles = smiles[selected].tolist()
        self.split = split

    def __len__(self) -> int:
        """Return the number of unique molecules in this split."""
        return len(self.smiles)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a copy of one graph as categorical long tensors."""
        row = int(self._indices[index]) if self._indices is not None else index
        return {"atoms": torch.tensor(self._atoms[row], dtype=torch.long),
                "charges": torch.tensor(self._charges[row], dtype=torch.long),
                "bonds": torch.tensor(self._bonds[row], dtype=torch.long)}


def collate_graphs(samples: list[dict[str, torch.Tensor]]) -> HardGraphBatch:
    """Stack dataset samples into the shared model input type."""
    if not samples:
        raise ValueError("Cannot collate an empty graph batch.")
    return HardGraphBatch(atoms=torch.stack([sample["atoms"] for sample in samples]),
                          charges=torch.stack([sample["charges"] for sample in samples]),
                          bonds=torch.stack([sample["bonds"] for sample in samples]))


def main() -> None:
    """Prepare a cache from an already-local CSV/SMILES file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Existing CSV or .smi source file")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--output", required=True, help="New dataset cache directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-nodes", type=int, default=50)
    parser.add_argument("--train-count", type=int, help="Exact train size; requires --val-count")
    parser.add_argument("--val-count", type=int, help="Exact validation size; test receives remainder")
    parser.add_argument("--reserve-cache", type=Path, help="Preserve a previous cache's split memberships")
    arguments = parser.parse_args()
    manifest = prepare_dataset(arguments.input, arguments.output, arguments.smiles_column,
                               GraphSpec(max_nodes=arguments.max_nodes), arguments.limit,
                               arguments.seed, arguments.train_count, arguments.val_count,
                               arguments.reserve_cache)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
