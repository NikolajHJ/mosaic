"""Encode the largest affordable reproducible subset of finalized graph membership."""

import argparse
from decimal import Decimal
import gzip
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import re
from shutil import disk_usage
import subprocess
import time

from rdkit import rdBase

from latentmol.datasets.bulk_finalize import VERSION as FINALIZED_VERSION
from latentmol.datasets.data import large_manifest_digest
from latentmol.representation import GraphSpec


GIB = 1024**3
DEFAULT_RESERVE = 4 * GIB
DEFAULT_SEED = 550017
MAX_BUCKET_ROWS = 1_000_000


def parse_work3_headroom(text, account, *, pool=6):
    """Read the finite per-account byte quota from DTU's official pool table."""
    sections = re.split(r"storagepool\s+(\d+)", text)
    pools = [content for number, content in zip(sections[1::2], sections[2::2]) if int(number) == pool]
    rows = [line.strip().split("||") for content in pools for line in content.splitlines()
            if line.strip().startswith(account + "|")]
    if len(rows) != 1 or len(rows[0]) < 2:
        raise ValueError("Cannot verify the official work3 quota for the requested account and pool")

    units = {unit: Decimal(1024)**i for i, unit in enumerate(("Byte", "KiB", "MiB", "GiB", "TiB"))}

    def amount(value):
        fields = value.split()
        if len(fields) != 2 or fields[1] not in units or re.fullmatch(r"\d+(?:\.\d+)?", fields[0]) is None:
            raise ValueError("Work3 needs finite byte quota usage and a finite hard limit")
        return Decimal(fields[0]) * units[fields[1]]

    used, hard = map(amount, rows[0][1].split("|"))
    # The command prints rounded human-readable values, not exact byte usage.
    return max(0, int(hard - used - Decimal("0.02") * GIB))


def work3_headroom(account):
    quota = subprocess.run(["getquota_work3.sh"], check=True, capture_output=True, text=True, timeout=30)
    return parse_work3_headroom(quota.stdout, account)


class LiveCacheBudget:
    """Track quota and filesystem availability while our staged artifact grows."""

    def __init__(self, path, *, reserve_bytes, max_output_bytes=None, quota=None):
        if (type(reserve_bytes) is not int or reserve_bytes < 0 or max_output_bytes is not None
                and (type(max_output_bytes) is not int or max_output_bytes < 1)):
            raise ValueError("Storage reserve must be nonnegative and an explicit cap must be positive")
        self.path = Path(path).resolve()
        while not self.path.exists():
            self.path = self.path.parent
        self.quota = quota
        self.cap = max_output_bytes
        self.report = dict(reserve_bytes=reserve_bytes, explicit_output_cap_bytes=max_output_bytes,
                           measured_official_user_quota=quota is not None, checks=0)

    def check(self, staged_data_bytes):
        """Return maximum allowed total artifact bytes, including staged data."""
        filesystem = disk_usage(self.path).free
        quota = self.quota() if self.quota is not None else None
        available = min(filesystem, quota) if quota is not None else filesystem
        current_cap = max(0, staged_data_bytes + available - self.report["reserve_bytes"])
        self.cap = current_cap if self.cap is None else min(self.cap, current_cap)
        observation = dict(filesystem_free_bytes=filesystem, quota_free_bytes=quota,
                           staged_data_bytes=staged_data_bytes, effective_cap_bytes=self.cap)
        self.report.setdefault("initial", observation)
        self.report["latest"] = observation
        self.report["checks"] += 1
        return self.cap


class CanonicalMembership:
    """Read finalized gzip parts on demand, with one train hash bucket in RAM."""

    def __init__(self, root, *, manifest_sha256):
        self.root = Path(root).resolve(strict=True)
        raw = (self.root / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise ValueError("Canonical corpus manifest hash differs")
        self.manifest = manifest = json.loads(raw)
        self.spec = GraphSpec.from_dict(manifest["spec"])
        self.dataset_sha256 = manifest["dataset_sha256"]
        self.counts = manifest["counts"]
        self.files = manifest["files"]
        self.sources = frozenset(manifest["source_priority"])
        self.train_parts = tuple(name for name in self.files if re.fullmatch(r"train/part-\d{4}\.tsv\.gz", name))
        if (manifest["version"] != FINALIZED_VERSION or large_manifest_digest(manifest) != self.dataset_sha256
                or manifest["spec_fingerprint"] != self.spec.fingerprint() or manifest["rdkit_version"] != rdBase.rdkitVersion
                or manifest["columns"] != ["canonical_smiles", "source"] or not self.sources or not self.train_parts
                or set(self.files) != {*self.train_parts, "val.tsv.gz", "test.tsv.gz"}
                or set(self.counts) != {"train", "val", "test", "total"}
                or any(type(count) is not int or count < 1 for count in self.counts.values())
                or self.counts["total"] != sum(self.counts[split] for split in ("train", "val", "test"))):
            raise ValueError("Canonical corpus manifest or graph schema is incompatible")
        for name, metadata in self.files.items():
            path = self.root / name
            if (not path.resolve(strict=True).is_relative_to(self.root)
                    or path.stat().st_size != metadata["bytes"]):
                raise ValueError(f"Canonical corpus artifact size or path differs: {name}")

    def _read_part(self, name):
        path = self.root / name
        with path.open("rb") as raw:
            digest = hashlib.sha256()
            while block := raw.read(1024 * 1024):
                digest.update(block)
            if digest.hexdigest() != self.files[name]["sha256"]:
                raise ValueError(f"Canonical corpus artifact hash differs: {name}")
            raw.seek(0)
            with gzip.open(raw, "rt", encoding="utf-8", newline="") as stream:
                for number, line in enumerate(stream, 1):
                    fields = line.removesuffix("\n").split("\t")
                    if (len(fields) != 2 or not fields[0] or fields[1] not in self.sources
                            or "\r" in line or not line.endswith("\n")):
                        raise ValueError(f"Malformed canonical corpus row {number} in {name}")
                    yield fields[0], fields[1]

    def rows(self, seed):
        for split in ("val", "test"):
            count = 0
            for count, (smiles, source) in enumerate(self._read_part(f"{split}.tsv.gz"), 1):
                yield split, smiles, source
            if count != self.counts[split]:
                raise ValueError(f"Canonical corpus {split} count differs")

        key = seed.to_bytes(8, "little")

        def rank(text, person):
            return hashlib.blake2b(text.encode("utf-8"), key=key, person=person, digest_size=16).digest()

        count = 0
        for name in sorted(self.train_parts, key=lambda part: (rank(part, b"cache-bucket"), part)):
            rows = list(islice(self._read_part(name), MAX_BUCKET_ROWS + 1))
            if len(rows) > MAX_BUCKET_ROWS:
                raise ValueError(f"Canonical corpus shard exceeds the bounded row-ranking limit: {name}")
            rows.sort(key=lambda row: (rank(row[0], b"cache-molecule"), row[0]))
            for smiles, source in rows:
                yield "train", smiles, source
                count += 1
        if count != self.counts["train"]:
            raise ValueError("Canonical corpus train count differs")


def build_sparse_corpus(corpus_path, output, *, manifest_sha256, workers=1, work3_account=None,
                        reserve_bytes=DEFAULT_RESERVE, max_output_bytes=None, seed=DEFAULT_SEED, deadline=None):
    """Authenticate membership and encode all holdouts plus the affordable train prefix."""
    from latentmol.datasets.bulk_cache import write_sparse_cache

    if type(seed) is not int or not 0 <= seed < 2**64 or deadline is not None and not math.isfinite(deadline):
        raise ValueError("Selection needs a 64-bit unsigned seed and finite deadline")
    corpus = CanonicalMembership(corpus_path, manifest_sha256=manifest_sha256)
    quota = None if work3_account is None else lambda: work3_headroom(work3_account)
    budget = LiveCacheBudget(Path(output).parent, reserve_bytes=reserve_bytes,
                             max_output_bytes=max_output_bytes, quota=quota)
    maximum = budget.check(0)
    if maximum <= 0:
        raise OSError("No graph-cache space remains after the storage reserve")
    selection = dict(rule="seeded_blake2b_bucket_and_row_prefix", seed=seed,
                     canonical_manifest_sha256=manifest_sha256, available_counts=corpus.counts,
                     available_train_by_source={source: stats["unique_train"]
                                                for source, stats in corpus.manifest["source_counts"].items()},
                     storage=budget.report)
    print(json.dumps(dict(stage="graph_cache_capacity", available_counts=corpus.counts,
                          max_bytes=maximum, storage=budget.report)), flush=True)
    manifest = write_sparse_cache(corpus.rows(seed), output,
        spec=corpus.spec, source_dataset_sha256=corpus.dataset_sha256, max_bytes=maximum, workers=workers,
        check_storage=budget.check, selection=selection,
        expected_holdout_counts={split: corpus.counts[split] for split in ("val", "test")},
        should_stop=(lambda: time.monotonic() >= deadline) if deadline is not None else None)
    selected = manifest["source_counts"]["train"]
    excluded = {source: count - selected.get(source, 0)
                for source, count in selection["available_train_by_source"].items()}
    print(json.dumps(dict(stage="complete", dataset_sha256=manifest["dataset_sha256"], counts=manifest["counts"],
                          excluded_train=sum(excluded.values()), excluded_train_by_source=excluded,
                          budget=manifest["budget"], storage=budget.report)), flush=True)
    return manifest


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--work3-account")
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE)
    parser.add_argument("--max-output-bytes", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    if args.wall_seconds is not None and (not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0):
        parser.error("Wall budget must be positive and finite")
    build_sparse_corpus(args.corpus, args.output, manifest_sha256=args.manifest_sha256,
        workers=args.workers, work3_account=args.work3_account, reserve_bytes=args.reserve_bytes,
        max_output_bytes=args.max_output_bytes, seed=args.seed,
        deadline=started + args.wall_seconds if args.wall_seconds is not None else None)


if __name__ == "__main__":
    main()
