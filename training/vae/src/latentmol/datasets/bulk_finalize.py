"""Finalize a large, exact canonical-SMILES corpus with bounded local scratch."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from latentmol.datasets.chembl_audit import _scaffold_and_size
from latentmol.datasets.combined import CombinedCorpus
from latentmol.datasets.data import large_manifest_digest
from latentmol.experiments import file_sha256, write_json


VERSION = "canonical-corpus-finalization-v1"
INGEST_VERSION = "bulk-canonical-smiles-ingest-v1"
BASELINE_SOURCES = {"moses": "moses", "chembl": "chembl37"}
COUNTERS = ("normalized", "excluded_exact_holdout", "excluded_reserved_scaffold", "duplicates", "unique_train")


def _progress(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


@contextmanager
def _write_gzip(path):
    with path.open("xb") as stream, gzip.GzipFile(filename="", mode="wb", compresslevel=3,
                                                  fileobj=stream, mtime=0) as compressed:
        yield compressed


def _safe_sources(sources):
    names = [source["name"] for source in sources]
    if (len(set(names)) != len(names) or set(names) & set(BASELINE_SOURCES.values())
            or any(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None for name in names)):
        raise ValueError("Ingest source names must be distinct, lowercase labels separate from the baseline")
    return names


def _baseline_holdouts(corpus, stage):
    seen, reserved_scaffolds = set(), {}
    counts = Counter()
    for split in ("val", "test"):
        with _write_gzip(stage / f"{split}.tsv.gz") as stream:
            for key, source in BASELINE_SOURCES.items():
                suffix = b"\t" + source.encode("ascii") + b"\n"
                for smiles in corpus.split(split, source=key).smiles:
                    identity = smiles.encode("utf-8")
                    if identity in seen:
                        raise ValueError("The baseline has duplicate exact holdout identities")
                    seen.add(identity)
                    if key == "chembl":
                        scaffold, _ = _scaffold_and_size(smiles)
                        if scaffold:
                            previous = reserved_scaffolds.setdefault(scaffold.encode("utf-8"), split)
                            if previous != split:
                                raise ValueError("Baseline ChEMBL scaffold spans validation and test")
                    stream.write(identity + suffix)
                    counts[split] += 1
    return seen, reserved_scaffolds, {split: counts[split] for split in ("val", "test")}


def _read_shards(shard_root, name, corpus):
    root = Path(shard_root).resolve(strict=True)
    # Ingest keeps incomplete work in a hidden sibling staging directory.
    shards = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not shards:
        raise ValueError(f"No normalized shards for {name} in {root}")
    for shard in shards:
        receipt_path = shard / "manifest.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        table = shard / "accepted.tsv.gz"
        count = receipt["counts"]["accepted"]
        if (receipt["version"] != INGEST_VERSION or receipt["source"] != name
                or receipt["spec_fingerprint"] != corpus.spec.fingerprint()
                or receipt["spec"] != corpus.manifest["spec"]
                or receipt["rdkit_version"] != corpus.manifest["rdkit_version"]
                or type(count) is not int or count < 0 or receipt["output"]["path"] != table.name
                or receipt["output"]["bytes"] != table.stat().st_size
                or receipt["output"]["sha256"] != file_sha256(table)):
            raise ValueError(f"Incompatible normalized shard or output digest: {shard}")
        provenance = {"name": shard.name, "manifest_sha256": file_sha256(receipt_path),
                      "input": receipt["input"], "output": receipt["output"], "counts": receipt["counts"],
                      "limit": receipt["limit"], "min_atoms": receipt["min_atoms"]}
        yield table, count, provenance


def _bucketize(corpus, sources, work, bucket_count, heldout, reserved_scaffolds, stats):
    """Stream routing by stable hash. Full strings still determine identity."""
    shard_receipts = {}
    with ExitStack() as stack:
        writers = [stack.enter_context((work / f"{index:04d}.tsv").open("xb", buffering=64 * 1024))
                   for index in range(bucket_count)]

        def write_row(identity, suffix):
            digest = hashlib.blake2b(identity, digest_size=8).digest()
            index = int.from_bytes(digest, "little") & (bucket_count - 1)
            writers[index].write(identity + suffix)

        for key, source in BASELINE_SOURCES.items():
            suffix = b"\t" + source.encode("ascii") + b"\n"
            for smiles in corpus.split("train", source=key).smiles:
                identity = smiles.encode("utf-8")
                if identity in heldout:
                    raise ValueError("The baseline contains an exact train/holdout overlap")
                write_row(identity, suffix)
                stats[source]["normalized"] += 1

        for source in sources:
            name = source["name"]
            suffix = b"\t" + name.encode("ascii") + b"\n"
            counter = stats[name]
            receipts = shard_receipts[name] = []
            for table, expected, receipt in _read_shards(source["shard_root"], name, corpus):
                with gzip.open(table, "rb") as stream:
                    rows = 0
                    for rows, line in enumerate(stream, 1):
                        identity, sep, scaffold = line.removesuffix(b"\n").partition(b"\t")
                        if not identity or not sep or b"\t" in scaffold or not line.endswith(b"\n") or b"\r" in line:
                            raise ValueError(f"Invalid normalized row {rows} in {table}")
                        if identity in heldout:
                            counter["excluded_exact_holdout"] += 1
                        elif scaffold in reserved_scaffolds:
                            counter["excluded_reserved_scaffold"] += 1
                        else:
                            write_row(identity, suffix)
                        if rows % 1_000_000 == 0:
                            _progress("route_source", source=name, shard=table.parent.name, shard_rows=rows)
                if rows != expected:
                    raise ValueError(f"Normalized row count differs from its receipt: {table}")
                counter["normalized"] += rows
                receipts.append(receipt)
    return shard_receipts


def _finalize_buckets(work, stage, bucket_count, stats, max_unique, output_cap):
    train = stage / "train"
    train.mkdir()
    stats_by_suffix = {name.encode("ascii") + b"\n": values for name, values in stats.items()}
    written_bytes = sum(path.stat().st_size for path in stage.glob("*.tsv.gz"))
    train_count = 0
    for index in range(bucket_count):
        route = work / f"{index:04d}.tsv"
        unique = {}
        with route.open("rb") as rows:
            for line in rows:
                identity, suffix = line.split(b"\t", 1)
                if identity in unique:
                    stats_by_suffix[suffix]["duplicates"] += 1
                else:
                    unique[identity] = suffix
                    stats_by_suffix[suffix]["unique_train"] += 1
                    if len(unique) > max_unique:
                        raise ValueError(f"Bucket {index} exceeds the unique-row memory bound; increase hash_buckets")
        route.unlink()
        if not unique:
            continue
        path = train / f"part-{index:04d}.tsv.gz"
        with _write_gzip(path) as stream:
            for identity in sorted(unique):
                stream.write(identity + b"\t" + unique[identity])
        train_count += len(unique)
        written_bytes += path.stat().st_size
        if output_cap is not None and written_bytes > output_cap:
            raise ValueError(f"Compressed corpus exceeds publication cap of {output_cap} bytes")
        if (index + 1) % 16 == 0 or index + 1 == bucket_count:
            _progress("finalize_buckets", buckets=index + 1, unique_train=train_count, compressed_bytes=written_bytes)
    return train_count


def _copy_checked(source, destination, metadata):
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        while block := reader.read(1024 * 1024):
            writer.write(block)
            digest.update(block)
            size += len(block)
    if size != metadata["bytes"] or digest.hexdigest() != metadata["sha256"]:
        raise ValueError(f"Staged corpus artifact changed during publication: {source}")


def _publish(stage, output, manifest):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bulk-corpus-publication-", dir=output.parent) as temporary:
        publication = Path(temporary) / output.name
        (publication / "train").mkdir(parents=True)
        for relative_path, metadata in manifest["files"].items():
            _copy_checked(stage / relative_path, publication / relative_path, metadata)
        write_json(publication / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(output)
        publication.rename(output)


def finalize_corpus(config, output, *, scratch_root=None, max_output_bytes=None):
    """Deduplicate across sources and keep baseline validation/test untouched."""
    if config.get("version") != VERSION:
        raise ValueError("Unexpected canonical corpus finalization version")
    bucket_count = config.get("hash_buckets", 256)
    max_unique = config.get("max_unique_per_bucket", 1_000_000)
    minimum_train = config["minimum_train_count"]
    if (type(bucket_count) is not int or not 1 <= bucket_count <= 512 or bucket_count & (bucket_count - 1)
            or type(max_unique) is not int or max_unique < 1 or type(minimum_train) is not int or minimum_train < 1
            or max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes < 1)):
        raise ValueError("Invalid bucket, target or output budget")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if scratch_root is not None:
        scratch_root = Path(scratch_root).resolve(strict=True)
    names = _safe_sources(config["sources"])
    baseline = config["baseline"]
    corpus = CombinedCorpus(baseline["path"], expected_sha256=baseline["dataset_sha256"])
    stats = {name: Counter(dict.fromkeys(COUNTERS, 0)) for name in (*BASELINE_SOURCES.values(), *names)}
    with TemporaryDirectory(prefix="canonical-corpus-", dir=scratch_root) as scratch:
        work = Path(scratch)
        stage = work / "publish"
        stage.mkdir()
        heldout, reserved_scaffolds, counts = _baseline_holdouts(corpus, stage)
        receipts = _bucketize(corpus, config["sources"], work, bucket_count, heldout, reserved_scaffolds, stats)
        counts["train"] = _finalize_buckets(work, stage, bucket_count, stats, max_unique, max_output_bytes)
        counts["total"] = sum(counts.values())
        if (not counts["train"] or any(stats[name]["duplicates"] for name in BASELINE_SOURCES.values())
                or any(values["normalized"] != sum(values[key] for key in COUNTERS[1:]) for values in stats.values())
                or sum(values["unique_train"] for values in stats.values()) != counts["train"]):
            raise ValueError("Canonical corpus count conservation or baseline uniqueness failed")
        files = {path.relative_to(stage).as_posix(): {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                 for path in sorted(stage.rglob("*.tsv.gz"))}
        manifest = dict(version=VERSION, spec=corpus.manifest["spec"], spec_fingerprint=corpus.spec.fingerprint(),
                        rdkit_version=corpus.manifest["rdkit_version"], baseline_dataset_sha256=corpus.dataset_sha256,
                        columns=["canonical_smiles", "source"], counts=counts, minimum_train_count=minimum_train,
                        target_reached=counts["train"] >= minimum_train, source_counts=stats,
                        input_shards=receipts, source_priority=[*BASELINE_SOURCES.values(), *names], files=files,
                        hash_buckets=bucket_count, reserved_scaffold_counts=dict(Counter(reserved_scaffolds.values())),
                        split_policy="Immutable baseline holdouts; exclude exact holdouts and ChEMBL holdout scaffolds from additions")
        manifest["dataset_sha256"] = large_manifest_digest(manifest)
        write_json(stage / "manifest.json", manifest)
        output_bytes = sum(meta["bytes"] for meta in files.values()) + (stage / "manifest.json").stat().st_size
        if max_output_bytes is not None and output_bytes > max_output_bytes:
            raise ValueError(f"Compressed corpus and manifest exceed publication cap of {max_output_bytes} bytes")
        _publish(stage, output, manifest)
    _progress("complete", counts=counts, target_reached=manifest["target_reached"], bytes=output_bytes,
              dataset_sha256=manifest["dataset_sha256"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--max-output-bytes", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    finalize_corpus(config, args.output, scratch_root=args.scratch_root, max_output_bytes=args.max_output_bytes)


if __name__ == "__main__":
    main()
