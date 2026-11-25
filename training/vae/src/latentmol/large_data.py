"""Build a bounded-memory, exact-count graph cache from a large local corpus.

Usage::

    python -m latentmol.large_data --input corpus.smi --output cache-v2 \
        --reserve-splits original-cache --train-count 1900000 \
        --val-count 20000 --test-count 20000 --seed 17

Use ``--mode inventory`` with a new output directory first to count exactly
which source identities are supported. Inventory writes only a small JSON
report; it neither allocates graph arrays nor assigns new split roles.

The source must already be on the HPC filesystem. SQLite holds canonical
stereo-free identities and a seeded, stable SHA-256 ordering on disk; graph
arrays are written once as read-only-loadable ``.npy`` memory maps. No source
rows or full graph collection is retained in Python memory. A supported
molecule omitted by exact target counts is reported, never silently added.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Iterator

import numpy as np
from rdkit import rdBase

from .chemistry import MoleculeRejected, encode_smiles
from .data import (LARGE_DATA_FILES, SPLIT_NAMES, _read_smiles,
                   _split_digest, large_manifest_digest, sha256_file,
                   validate_large_cache)
from .representation import GraphSpec


def _rank(seed: int, smiles: str) -> bytes:
    """Provide a deterministic source-order-independent ordering of identities."""
    return hashlib.sha256(f"{seed}\0{smiles}".encode("utf-8")).digest()


def _preparation_code_hashes() -> dict[str, str]:
    """Bind the cache to the exact local preprocessing implementations used."""
    directory = Path(__file__).resolve().parent
    names = ("large_data.py", "data.py", "chemistry.py", "representation.py")
    return {name: sha256_file(directory / name) for name in names}


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_protected_unchanged(cache: Path, manifest_hash: str,
                               dataset_hash: str) -> None:
    """Reauthenticate protected identities before sealing a new manifest."""
    manifest_path = cache / "manifest.json"
    if sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("Protected cache manifest changed during preparation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_sha256") != dataset_hash:
        raise RuntimeError("Protected cache identity changed during preparation.")
    if manifest.get("dataset_format_version") == 1:
        if sha256_file(cache / "dataset.npz") != dataset_hash:
            raise RuntimeError("Protected cache file changed during preparation.")
    elif manifest.get("dataset_format_version") == 2:
        if validate_large_cache(cache).dataset_sha256 != dataset_hash:
            raise RuntimeError("Protected large cache changed during preparation.")
    else:
        raise ValueError("Protected cache has an unsupported format version.")


def _iter_protected(cache: Path, spec: GraphSpec,
                    manifest: dict) -> Iterator[tuple[str, int]]:
    """Verify an old cache and stream its protected split roles.

    A v1 cache is expected to be the original, comparatively small study cache.
    V2 protection also works without loading its graph arrays into RAM.
    Different raw source files are permitted; canonical identities and schema
    are the protection boundary.
    """
    protected_spec = GraphSpec.from_dict(manifest["spec"])
    if (protected_spec.fingerprint() != manifest.get("spec_fingerprint")
            or manifest["spec_fingerprint"] != spec.fingerprint()):
        raise ValueError("Protected cache graph schema differs from the new corpus.")
    if manifest.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Protected cache RDKit version differs; canonical identity parity is unverified.")
    version = manifest.get("dataset_format_version")
    if version == 1:
        source = cache / "dataset.npz"
        if sha256_file(source) != manifest.get("dataset_sha256"):
            raise ValueError("Protected v1 cache file hash differs from its manifest.")
        with np.load(source, allow_pickle=False) as archive:
            smiles_array = archive["smiles"]
            labels = archive["split"]
        if (smiles_array.ndim != 1 or smiles_array.dtype.kind != "U"
                or labels.shape != (len(smiles_array),)
                or labels.dtype != np.dtype(np.int8)):
            raise ValueError("Protected v1 cache has invalid SMILES/split array layout.")
        smiles = smiles_array.tolist()
        if _split_digest(smiles, labels) != manifest.get("split_sha256"):
            raise ValueError("Protected v1 cache split digest differs from its manifest.")
        if len(smiles) != manifest["counts"]["total"] or len(set(smiles)) != len(smiles):
            raise ValueError("Protected v1 cache contains inconsistent or duplicate identities.")
        if any(int(np.count_nonzero(labels == label)) != manifest["counts"][name]
               for name, label in SPLIT_NAMES.items()):
            raise ValueError("Protected v1 cache split counts differ from its manifest.")
        for value, label in zip(smiles, labels):
            if int(label) not in SPLIT_NAMES.values():
                raise ValueError("Protected v1 cache contains an unknown split label.")
            yield value, int(label)
    elif version == 2:
        validate_large_cache(cache)
        offsets = np.load(cache / "smiles_offsets.npy", mmap_mode="r", allow_pickle=False)
        labels = np.load(cache / "split.npy", mmap_mode="r", allow_pickle=False)
        with (cache / "smiles.utf8").open("rb") as stream:
            for index, label in enumerate(labels):
                value = stream.read(int(offsets[index + 1] - offsets[index])).decode("utf-8")
                yield value, int(label)
    else:
        raise ValueError("Protected cache has an unsupported format version.")


def _create_identity_tables(connection: sqlite3.Connection, reserve: Path,
                            spec: GraphSpec, manifest: dict) -> None:
    """Use one protected-role import and SQLite schema in both modes."""
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE molecules (smiles TEXT PRIMARY KEY, "
                       "rank BLOB NOT NULL, split INTEGER) WITHOUT ROWID")
    connection.execute("CREATE TABLE protected (smiles TEXT PRIMARY KEY, "
                       "split INTEGER NOT NULL) WITHOUT ROWID")
    for count, (smiles, label) in enumerate(_iter_protected(reserve, spec, manifest), 1):
        connection.execute("INSERT INTO protected(smiles, split) VALUES (?, ?)",
                           (smiles, label))
        if count % 10000 == 0:
            connection.commit()
    connection.commit()


def _composition_audit(spec: GraphSpec) -> dict:
    """Constant-size composition counts for supported unique hard graphs."""
    return {
        "molecules": 0,
        "heavy_atom_histogram": [0] * (spec.max_nodes + 1),
        "atomic_number_counts": {str(number): 0 for number in spec.atomic_numbers[1:]},
        "formal_charge_counts": {str(charge): 0 for charge in spec.formal_charges},
        "active_upper_triangle_bond_counts": {name: 0 for name in spec.bond_names},
    }


def _accumulate_composition(audit: dict, record: dict, spec: GraphSpec) -> None:
    """Count each active atom and each undirected active-slot pair once."""
    nodes = int(np.count_nonzero(record["atom_types"]))
    audit["molecules"] += 1
    audit["heavy_atom_histogram"][nodes] += 1
    for index, amount in enumerate(np.bincount(record["atom_types"][:nodes],
                                               minlength=len(spec.atomic_numbers))):
        if index:
            audit["atomic_number_counts"][str(spec.atomic_numbers[index])] += int(amount)
    for index, amount in enumerate(np.bincount(record["charges"][:nodes],
                                               minlength=len(spec.formal_charges))):
        audit["formal_charge_counts"][str(spec.formal_charges[index])] += int(amount)
    for index, amount in enumerate(np.bincount(
            record["bond_types"][:nodes, :nodes][np.triu_indices(nodes, k=1)],
            minlength=len(spec.bond_names))):
        audit["active_upper_triangle_bond_counts"][spec.bond_names[index]] += int(amount)


def _staging_bytes(stage: Path) -> int:
    """Sample bytes in the private staging directory, including SQLite journal."""
    return sum(path.stat().st_size for path in stage.iterdir() if path.is_file())


def _scan_supported(connection: sqlite3.Connection, source: Path, spec: GraphSpec,
                    seed: int, smiles_column: str, *, stage: Path,
                    collect_composition: bool = False, progress_every: int = 0) -> dict:
    """Shared roundtrip-valid canonical scan; never select or assign holdouts.

    Graph records are transient. Composition, when requested, counts only the
    first occurrence of each canonical identity, matching SQLite deduplication.
    Staging disk is sampled, not advertised as an exact OS peak measurement.
    """
    rejected: Counter[str] = Counter()
    source_rows = accepted_rows = duplicates = 0
    composition = _composition_audit(spec) if collect_composition else None
    sampled_peak = _staging_bytes(stage)
    started = time.perf_counter()
    for raw in _read_smiles(source, smiles_column):
        source_rows += 1
        try:
            record = encode_smiles(raw, spec)
        except MoleculeRejected as error:
            rejected[error.reason] += 1
        else:
            accepted_rows += 1
            canonical = record["smiles"]
            cursor = connection.execute("INSERT OR IGNORE INTO molecules(smiles, rank, split) "
                                        "VALUES (?, ?, NULL)", (canonical, _rank(seed, canonical)))
            if cursor.rowcount == 0:
                duplicates += 1
            elif composition is not None:
                _accumulate_composition(composition, record, spec)
        if source_rows % 10000 == 0:
            sampled_peak = max(sampled_peak, _staging_bytes(stage))
            connection.commit()
        if progress_every and source_rows % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(json.dumps({"event": "canonical_scan_progress", "source_rows": source_rows,
                              "roundtrip_valid_rows": accepted_rows,
                              "supported_unique_so_far": accepted_rows - duplicates,
                              "duplicate_rows": duplicates, "rejected_rows": sum(rejected.values()),
                              "elapsed_seconds": elapsed,
                              "source_rows_per_second": source_rows / elapsed if elapsed else None},
                             sort_keys=True, allow_nan=False), flush=True)
    sampled_peak = max(sampled_peak, _staging_bytes(stage))
    connection.commit()
    sampled_peak = max(sampled_peak, _staging_bytes(stage))
    return {"source_rows": source_rows, "accepted_rows": accepted_rows,
            "duplicates": duplicates, "rejected": rejected,
            "composition": composition, "sampled_peak_staging_bytes": sampled_peak}


def _protected_overlap(connection: sqlite3.Connection) -> dict:
    """Report imported roles without modifying any molecule split labels."""
    total = {name: int(connection.execute(
        "SELECT COUNT(*) FROM protected WHERE split = ?", (label,)).fetchone()[0])
        for name, label in SPLIT_NAMES.items()}
    present = {name: int(connection.execute(
        "SELECT COUNT(*) FROM protected INNER JOIN molecules USING(smiles) "
        "WHERE protected.split = ?", (label,)).fetchone()[0])
        for name, label in SPLIT_NAMES.items()}
    return {"total_protected": total, "present_protected": present,
            "absent_protected": {name: total[name] - present[name] for name in SPLIT_NAMES}}


def _peak_host_memory() -> dict:
    """Linux ru_maxrss is KiB; this is process-lifetime host RSS, not GPU VRAM."""
    try:
        import resource
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return {"peak_host_rss_bytes": None, "measurement": "resource unavailable"}
    factor = 1 if sys.platform == "darwin" else 1024
    return {"peak_host_rss_bytes": int(raw * factor),
            "measurement": "resource.RUSAGE_SELF.ru_maxrss; process lifetime including imports/protected cache"}


def inventory_large_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    reserve_splits: str | Path,
    seed: int = 17,
    smiles_column: str = "smiles",
    spec: GraphSpec | None = None,
) -> dict:
    """Publish an immutable eligibility report using preparation's exact scan.

    No arrays, train/validation/test assignment or cache manifest is produced.
    The caller must choose explicit attainable counts in a separately sealed
    preparation protocol. All inventory numerical work belongs on DTU HPC.
    """
    started = time.perf_counter()
    spec = spec or GraphSpec()
    source = Path(input_path).expanduser().resolve(strict=True)
    reserve = Path(reserve_splits).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Inventory destination already exists: {destination}")
    if (destination == source or destination == reserve
            or destination.is_relative_to(reserve) or reserve.is_relative_to(destination)):
        raise ValueError("Output must differ from source and protected cache.")
    raw_hash = sha256_file(source)
    protected_manifest_hash = sha256_file(reserve / "manifest.json")
    protected_manifest = json.loads((reserve / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(reserve / "manifest.json") != protected_manifest_hash:
        raise RuntimeError("Protected cache manifest changed before identity import.")
    protected_dataset_hash = protected_manifest["dataset_sha256"]
    code_hashes = _preparation_code_hashes()
    config = {"mode": "inventory", "spec_fingerprint": spec.fingerprint(), "seed": seed,
              "smiles_column": smiles_column, "protected_dataset_sha256": protected_dataset_hash}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.inventory-", dir=destination.parent))
    initial_free = shutil.disk_usage(stage).free
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(stage / "identities.sqlite")
        phase_start = time.perf_counter()
        _create_identity_tables(connection, reserve, spec, protected_manifest)
        protected_seconds = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        scan = _scan_supported(connection, source, spec, seed, smiles_column,
                               stage=stage, collect_composition=True, progress_every=100000)
        scan_seconds = time.perf_counter() - phase_start
        supported_unique = int(connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0])
        if connection.execute("SELECT COUNT(*) FROM molecules WHERE split IS NOT NULL").fetchone()[0]:
            raise RuntimeError("Inventory must not assign held-out or training roles.")
        overlap = _protected_overlap(connection)
        text_bytes = int(connection.execute(
            "SELECT COALESCE(SUM(LENGTH(CAST(smiles AS BLOB))), 0) FROM molecules").fetchone()[0])
        identities_hash = hashlib.sha256()
        identities_hash.update(b"[")
        for index, (smiles,) in enumerate(connection.execute("SELECT smiles FROM molecules ORDER BY smiles")):
            if index:
                identities_hash.update(b",")
            identities_hash.update(json.dumps(smiles, ensure_ascii=True).encode("utf-8"))
        identities_hash.update(b"]")
        # Same payload estimate as preparation, but for all supported unique
        # rows, not a selected quota. Excludes source/protected files and SQLite.
        array_bytes = supported_unique * (spec.max_nodes ** 2 + 2 * spec.max_nodes + 1)
        offsets_bytes = 8 * (supported_unique + 1)
        payload_bytes = array_bytes + offsets_bytes + text_bytes
        stage_bytes = _staging_bytes(stage)
        free_after_scan = shutil.disk_usage(stage).free
        connection.close()
        connection = None
        (stage / "identities.sqlite").unlink()
        phase_start = time.perf_counter()
        if sha256_file(source) != raw_hash:
            raise RuntimeError("Source file changed during inventory.")
        if _preparation_code_hashes() != code_hashes:
            raise RuntimeError("Preprocessing code changed during inventory.")
        _check_protected_unchanged(reserve, protected_manifest_hash, protected_dataset_hash)
        verification_seconds = time.perf_counter() - phase_start
        rejected = scan["rejected"]
        report = {
            "inventory_format_version": 1, "status": "completed_inventory_only",
            "spec": spec.to_dict(), "spec_fingerprint": spec.fingerprint(),
            "raw_sha256": raw_hash, "supported_identity_sha256": identities_hash.hexdigest(),
            "counts": {"source_rows": scan["source_rows"], "rejected_rows": sum(rejected.values()),
                       "roundtrip_valid_rows": scan["accepted_rows"], "duplicate_rows": scan["duplicates"],
                       "supported_unique": supported_unique,
                       "roundtrip_rejected_rows": rejected.get("representation_roundtrip", 0)},
            "rejection_reasons": dict(sorted(rejected.items())),
            "composition": {"scope": "all supported unique canonical identities; duplicates excluded",
                            "bond_scope": "active upper-triangle pairs, including NONE", **scan["composition"]},
            "protected_roles": overlap,
            "timing": {"protected_import_seconds": protected_seconds, "canonical_scan_seconds": scan_seconds,
                       "input_reverification_seconds": verification_seconds,
                       "total_seconds_before_publication": time.perf_counter() - started,
                       "source_rows_per_second": scan["source_rows"] / scan_seconds if scan_seconds else None},
            "host_memory": _peak_host_memory(),
            "disk": {"filesystem_free_before_sqlite_bytes": initial_free,
                     "filesystem_free_after_scan_bytes": free_after_scan,
                     "sqlite_staging_bytes_at_scan_end": stage_bytes,
                     "sampled_peak_staging_bytes": scan["sampled_peak_staging_bytes"],
                     "staging_peak_measurement": "file sizes sampled every 10000 source rows and scan end; not exact OS peak",
                     "all_supported_dense_array_payload_bytes": array_bytes,
                     "all_supported_smiles_offset_payload_bytes": offsets_bytes,
                     "all_supported_smiles_utf8_bytes": text_bytes,
                     "all_supported_cache_payload_bytes": payload_bytes,
                     "all_supported_cache_required_free_after_sqlite_estimate_bytes":
                         payload_bytes + max(64 * 1024 * 1024, payload_bytes // 10)},
            "rdkit_version": rdBase.rdkitVersion, "numpy_version": np.__version__,
            "inventory_identity": {"source_file_sha256": code_hashes, "config": config,
                                   "config_sha256": _config_hash(config)},
            "provenance": {"source_path": str(source), "smiles_column": smiles_column,
                           "reserved_cache": {"path": str(reserve), "manifest_sha256": protected_manifest_hash,
                                              "dataset_sha256": protected_dataset_hash,
                                              "split_sha256": protected_manifest["split_sha256"]},
                           "canonicalization": "same encode_smiles roundtrip and SQLite canonical dedup as preparation",
                           "source_split_labels": "ignored; no new split roles assigned",
                           "repairs": "none", "graph_arrays_written": False,
                           "selected_target_counts": None,
                           "limits": "inventory is not a training cache or a chosen split; disk peak is sampled"},
        }
        report["inventory_sha256"] = _config_hash(report)
        (stage / "inventory.json").write_text(json.dumps(report, indent=2, sort_keys=True,
                                                        allow_nan=False) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"Inventory destination appeared during inventory: {destination}")
        stage.rename(destination)
        return report
    finally:
        if connection is not None:
            connection.close()
        if stage.exists():
            shutil.rmtree(stage)


def _assign_exact_splits(connection: sqlite3.Connection, train_count: int,
                         val_count: int, test_count: int) -> dict:
    """Keep present protected roles, then fill quotas in stable hash order."""
    required = {0: train_count, 1: val_count, 2: test_count}
    available = connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
    target = sum(required.values())
    if available < target:
        raise ValueError(f"Requested {target:,} graphs, but only {available:,} unique "
                         "supported identities exist. Choose explicit lower --*-count values.")
    connection.execute("""
        UPDATE molecules SET split =
          (SELECT protected.split FROM protected WHERE protected.smiles = molecules.smiles)
        WHERE EXISTS (SELECT 1 FROM protected WHERE protected.smiles = molecules.smiles)
    """)
    overlap = _protected_overlap(connection)
    protected_total = overlap["total_protected"]
    protected_present = overlap["present_protected"]
    if any(protected_present[name] > required[label]
           for name, label in SPLIT_NAMES.items()):
        raise ValueError("Exact split target is smaller than present protected membership: "
                         f"{protected_present}")
    connection.execute("CREATE INDEX molecules_split_rank ON molecules(split, rank, smiles)")
    # Keep held-out identities fixed before allocating new rows. Remaining
    # training rows are the next seeded hashes; no protected old-train identity
    # may fall into a new held-out split.
    for label in (1, 2, 0):
        name = next(key for key, value in SPLIT_NAMES.items() if value == label)
        needed = required[label] - protected_present[name]
        if needed:
            connection.execute("""
                UPDATE molecules SET split = ? WHERE smiles IN
                  (SELECT smiles FROM molecules WHERE split IS NULL
                   ORDER BY rank, smiles LIMIT ?)
            """, (label, needed))
    observed = {
        name: int(connection.execute("SELECT COUNT(*) FROM molecules WHERE split = ?", (label,))
                  .fetchone()[0])
        for name, label in SPLIT_NAMES.items()
    }
    if any(observed[name] != required[label] for name, label in SPLIT_NAMES.items()):
        raise RuntimeError(f"Exact split assignment failed: {observed}")
    connection.commit()
    return {
        "total_protected": protected_total,
        "present_protected": protected_present,
        "absent_protected": {name: protected_total[name] - protected_present[name]
                             for name in SPLIT_NAMES},
        "observed": observed,
        "supported_unique": available,
        "excluded_supported_by_target": available - target,
    }


def _write_arrays(stage: Path, connection: sqlite3.Connection, spec: GraphSpec,
                  count: int) -> tuple[dict, str]:
    """Encode selected identities into numeric mmap arrays and streaming audit."""
    size = spec.max_nodes
    atoms = np.lib.format.open_memmap(stage / "atom_types.npy", mode="w+",
                                      dtype=np.int8, shape=(count, size))
    charges = np.lib.format.open_memmap(stage / "charges.npy", mode="w+",
                                        dtype=np.int8, shape=(count, size))
    bonds = np.lib.format.open_memmap(stage / "bond_types.npy", mode="w+",
                                      dtype=np.int8, shape=(count, size, size))
    labels = np.lib.format.open_memmap(stage / "split.npy", mode="w+",
                                       dtype=np.int8, shape=(count,))
    offsets = np.lib.format.open_memmap(stage / "smiles_offsets.npy", mode="w+",
                                        dtype=np.int64, shape=(count + 1,))
    offsets[0] = 0
    audit = {
        "by_split": {name: {"molecules": 0,
                             "heavy_atom_histogram": [0] * (size + 1)}
                     for name in SPLIT_NAMES},
        "atomic_number_counts": {str(number): 0 for number in spec.atomic_numbers[1:]},
        "formal_charge_counts": {str(charge): 0 for charge in spec.formal_charges},
        "active_upper_triangle_bond_counts": {name: 0 for name in spec.bond_names},
    }
    split_names = {value: name for name, value in SPLIT_NAMES.items()}
    upper = {nodes: np.triu_indices(nodes, k=1) for nodes in range(1, size + 1)}
    split_hash = hashlib.sha256()
    split_hash.update(b"[")
    with (stage / "smiles.utf8").open("wb") as text_stream:
        rows = connection.execute("SELECT smiles, split FROM molecules "
                                  "WHERE split IS NOT NULL ORDER BY smiles")
        seen = 0
        for seen, (smiles, label) in enumerate(rows, 1):
            index = seen - 1
            record = encode_smiles(smiles, spec)
            if record["smiles"] != smiles:
                raise RuntimeError("A selected identity changed during graph encoding.")
            atoms[index] = record["atom_types"]
            charges[index] = record["charges"]
            bonds[index] = record["bond_types"]
            labels[index] = label
            raw = smiles.encode("utf-8")
            text_stream.write(raw)
            offsets[seen] = offsets[index] + len(raw)
            if index:
                split_hash.update(b",")
            split_hash.update(json.dumps([smiles, label], ensure_ascii=True,
                                         separators=(",", ":")).encode("utf-8"))
            nodes = int(np.count_nonzero(record["atom_types"]))
            group = audit["by_split"][split_names[label]]
            group["molecules"] += 1
            group["heavy_atom_histogram"][nodes] += 1
            for atom_index, amount in enumerate(np.bincount(record["atom_types"][:nodes],
                                                            minlength=len(spec.atomic_numbers))):
                if atom_index:
                    audit["atomic_number_counts"][str(spec.atomic_numbers[atom_index])] += int(amount)
            for charge_index, amount in enumerate(np.bincount(record["charges"][:nodes],
                                                              minlength=len(spec.formal_charges))):
                audit["formal_charge_counts"][str(spec.formal_charges[charge_index])] += int(amount)
            for bond_index, amount in enumerate(np.bincount(
                    record["bond_types"][:nodes, :nodes][upper[nodes]],
                    minlength=len(spec.bond_names))):
                audit["active_upper_triangle_bond_counts"][spec.bond_names[bond_index]] += int(amount)
        if seen != count:
            raise RuntimeError(f"Selected-row count changed: wrote {seen}, expected {count}.")
    split_hash.update(b"]")
    for array in (atoms, charges, bonds, labels, offsets):
        array.flush()
    del atoms, charges, bonds, labels, offsets
    return audit, split_hash.hexdigest()


def prepare_large_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    reserve_splits: str | Path,
    train_count: int = 1_900_000,
    val_count: int = 20_000,
    test_count: int = 20_000,
    seed: int = 17,
    smiles_column: str = "smiles",
    spec: GraphSpec | None = None,
) -> dict:
    """Prepare one exact-size v2 cache, atomically, using bounded host memory.

    A new source need not match the protected cache's raw-file digest. Present
    protected identities keep their original roles, and absent identities are
    counted separately. If supported unique molecules cannot satisfy all three
    requested counts, fail with an explicit count instead of clamping them.
    """
    spec = spec or GraphSpec()
    if train_count < 1 or val_count < 0 or test_count < 0:
        raise ValueError("Train count must be positive; validation/test counts cannot be negative.")
    source = Path(input_path).expanduser().resolve(strict=True)
    reserve = Path(reserve_splits).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset destination already exists: {destination}")
    if destination == reserve or destination == source:
        raise ValueError("Output must differ from source and protected cache.")
    raw_hash = sha256_file(source)
    protected_manifest_hash = sha256_file(reserve / "manifest.json")
    protected_manifest = json.loads((reserve / "manifest.json").read_text(encoding="utf-8"))
    if sha256_file(reserve / "manifest.json") != protected_manifest_hash:
        raise RuntimeError("Protected cache manifest changed before identity import.")
    protected_dataset_hash = protected_manifest["dataset_sha256"]
    code_hashes = _preparation_code_hashes()
    prep_config = {"spec_fingerprint": spec.fingerprint(), "seed": seed,
                   "smiles_column": smiles_column, "train_count": train_count,
                   "val_count": val_count, "test_count": test_count,
                   "protected_dataset_sha256": protected_dataset_hash}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.building-",
                                        dir=destination.parent))
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(stage / "identities.sqlite")
        _create_identity_tables(connection, reserve, spec, protected_manifest)
        scan = _scan_supported(connection, source, spec, seed, smiles_column, stage=stage)
        source_rows = scan["source_rows"]
        accepted_rows = scan["accepted_rows"]
        duplicates = scan["duplicates"]
        rejected = scan["rejected"]
        if sha256_file(source) != raw_hash:
            raise RuntimeError("Source file changed during preparation.")
        supported_unique = connection.execute("SELECT COUNT(*) FROM molecules").fetchone()[0]
        requested_total = train_count + val_count + test_count
        if supported_unique < requested_total:
            raise ValueError(
                f"Requested {requested_total:,} graphs, but only {supported_unique:,} unique "
                f"supported identities exist among {source_rows:,} source rows "
                f"({duplicates:,} duplicates; {sum(rejected.values()):,} rejected; "
                f"reasons={dict(sorted(rejected.items()))}). Choose explicit lower --*-count values."
            )
        assignment = _assign_exact_splits(connection, train_count, val_count, test_count)
        count = train_count + val_count + test_count
        # The SQLite index is already on disk. Account for all numeric array
        # payloads and exact selected UTF-8 bytes, then reserve 10%/64 MiB for
        # headers, manifests and filesystem slack before allocating output maps.
        selected_smiles_bytes = connection.execute("""
            SELECT COALESCE(SUM(LENGTH(CAST(smiles AS BLOB))), 0)
            FROM molecules WHERE split IS NOT NULL
        """).fetchone()[0]
        estimated_bytes = (count * (spec.max_nodes * spec.max_nodes
                                    + 2 * spec.max_nodes + 1)
                           + 8 * (count + 1) + selected_smiles_bytes)
        required_bytes = estimated_bytes + max(64 * 1024 * 1024, estimated_bytes // 10)
        free_bytes = shutil.disk_usage(stage).free
        if free_bytes < required_bytes:
            raise OSError(f"Insufficient staging disk: need approximately {required_bytes:,} "
                          f"bytes free after SQLite, found {free_bytes:,}.")
        audit, split_hash = _write_arrays(stage, connection, spec, count)
        connection.close()
        connection = None
        (stage / "identities.sqlite").unlink()
        files = {name: {"sha256": sha256_file(stage / name),
                        "bytes": (stage / name).stat().st_size}
                 for name in LARGE_DATA_FILES}
        manifest = {
            "dataset_format_version": 2,
            "spec": spec.to_dict(),
            "spec_fingerprint": spec.fingerprint(),
            "raw_sha256": raw_hash,
            "split_sha256": split_hash,
            "files": files,
            "counts": {
                "source_rows": source_rows,
                "rejected_rows": sum(rejected.values()),
                "duplicate_rows": duplicates,
                "roundtrip_valid_rows": accepted_rows,
                "roundtrip_rejected_rows": rejected.get("representation_roundtrip", 0),
                "unique_accepted_before_selection": assignment["supported_unique"],
                "excluded_supported_by_target": assignment["excluded_supported_by_target"],
                "selected_target_graphs_roundtrip_valid": count,
                "total": count,
                **assignment["observed"],
            },
            "rejection_reasons": dict(sorted(rejected.items())),
            "audit": audit,
            "rdkit_version": rdBase.rdkitVersion,
            "numpy_version": np.__version__,
            "preparation_identity": {"source_file_sha256": code_hashes,
                                     "config": prep_config,
                                     "config_sha256": _config_hash(prep_config)},
            "provenance": {
                "source_path": str(source),
                "source_format": "smi" if source.suffix.lower() in {".smi", ".smiles"} else "csv",
                "smiles_column": smiles_column,
                "seed": seed,
                "selection_method": "SHA256(seed, NUL, canonical stereo-free SMILES) rank; heldouts first",
                "canonicalization": "RDKit canonical stereo-free SMILES; reparsed for ordered atom slots",
                "deduplication": "SQLite primary key on roundtrip-valid canonical identity",
                "split_method": "protected old roles then seeded rank, exact quotas; not scaffold-separated",
                "source_split_labels": "ignored; this is a custom canonical-identity resplit, not an official corpus split",
                "requested_train_count": train_count,
                "requested_val_count": val_count,
                "requested_test_count": test_count,
                "reserved_cache": {
                    "path": str(reserve),
                    "manifest_sha256": protected_manifest_hash,
                    "dataset_sha256": protected_dataset_hash,
                    "split_sha256": protected_manifest["split_sha256"],
                    **{key: assignment[key] for key in
                       ("total_protected", "present_protected", "absent_protected")},
                },
                "stereochemistry": "removed before identity deduplication",
                "target_validation": "connected sanitized graph; exact canonical SMILES roundtrip",
                "unsupported_features": ["stereochemistry", "isotopes", "radicals",
                                         "explicit hydrogen counts that fail graph roundtrip"],
                "repairs": "none; no salt stripping, charge normalization or tautomer normalization",
            },
        }
        manifest["dataset_sha256"] = large_manifest_digest(manifest)
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if sha256_file(source) != raw_hash:
            raise RuntimeError("Source file changed during graph-cache construction.")
        if _preparation_code_hashes() != code_hashes:
            raise RuntimeError("Preprocessing code changed during graph-cache construction.")
        _check_protected_unchanged(reserve, protected_manifest_hash,
                                   protected_dataset_hash)
        if destination.exists():
            raise FileExistsError(f"Dataset destination appeared during preparation: {destination}")
        stage.rename(destination)
        return manifest
    finally:
        if connection is not None:
            connection.close()
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "inventory"), default="prepare")
    parser.add_argument("--input", required=True, help="Existing local .smi/.smiles or CSV file")
    parser.add_argument("--output", required=True, help="New immutable cache or inventory directory")
    parser.add_argument("--reserve-splits", required=True,
                        help="Original cache whose canonical train/val/test roles are protected")
    parser.add_argument("--train-count", type=int, default=1_900_000)
    parser.add_argument("--val-count", type=int, default=20_000)
    parser.add_argument("--test-count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--smiles-column", default="smiles")
    arguments = parser.parse_args()
    if arguments.mode == "inventory":
        manifest = inventory_large_dataset(
            arguments.input, arguments.output, arguments.reserve_splits,
            arguments.seed, arguments.smiles_column)
    else:
        manifest = prepare_large_dataset(
            arguments.input, arguments.output, arguments.reserve_splits,
            arguments.train_count, arguments.val_count, arguments.test_count,
            arguments.seed, arguments.smiles_column)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
