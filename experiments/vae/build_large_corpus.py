"""Build one large corpus from a pinned SMILES snapshot in bounded chunks."""

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

from latentmol.datasets.bulk_finalize import finalize_corpus
from latentmol.datasets.bulk_ingest import ingest
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


def download_snapshot(source, target):
    """Authenticate the release before any source rows can enter a corpus."""
    request = Request(source["url"], headers={"If-Unmodified-Since": source["last_modified"]})
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with urlopen(request, timeout=120) as reply, target.open("xb") as stream:
        if (reply.status != 200 or reply.headers.get("Last-Modified") != source["last_modified"]
                or int(reply.headers.get("Content-Length", "-1")) != source["bytes"]):
            raise ValueError("Source server did not return the pinned snapshot")
        while block := reply.read(8 * 1024 * 1024):
            size += len(block)
            if size > source["bytes"]:
                raise ValueError("Downloaded snapshot exceeds the pinned length")
            stream.write(block)
            sha.update(block)
            md5.update(block)
    if size != source["bytes"] or md5.hexdigest() != source["md5"]:
        raise ValueError("Source length or upstream checksum differs")
    return dict(source, sha256=sha.hexdigest())


def write_chunk(source, target, limit):
    digest = hashlib.sha256()
    count = 0
    with target.open("xb") as stream:
        for _ in range(limit):
            row = source.readline()
            if not row:
                break
            stream.write(row)
            digest.update(row)
            count += 1
    return count, digest.hexdigest()


def work3_headroom():
    quota = subprocess.run(["getquota_work3.sh"], check=True, capture_output=True, text=True, timeout=30).stdout
    _, pool = quota.split("storagepool 6", 1)
    match = re.search(r"\|\|\s*([\d.]+) GiB\|\s*([\d.]+) GiB\|\|", pool)
    if match is None:
        raise ValueError("Cannot verify the active work3 quota")
    used, limit = map(float, match.groups())
    return max(0, int((limit - used - 0.02) * 1024**3))


def publish_shard(staged, destination, bytes_left):
    size = sum(path.stat().st_size for path in staged.iterdir() if path.is_file())
    if size > bytes_left:
        raise OSError("Compressed ingest shards exceed the declared work3 budget")
    if work3_headroom() < size + 1024**3:
        raise OSError("Work3 needs a 1 GiB reserve after writing the ingest shard")
    if destination.exists():
        raise FileExistsError(f"Preserve existing shard at {destination}")
    # A copy may cross filesystems. Rename only after the complete shard arrives.
    with tempfile.TemporaryDirectory(prefix=".ingest-publish-", dir=destination.parent.parent) as scratch:
        temporary = Path(scratch) / destination.name
        shutil.copytree(staged, temporary)
        receipt = json.loads((temporary / "manifest.json").read_text(encoding="utf-8"))
        if file_sha256(temporary / "accepted.tsv.gz") != receipt["output"]["sha256"]:
            raise OSError("Work3 shard differs from the locally verified copy")
        temporary.rename(destination)
    return size


def run(config, output, scratch_root, workers):
    started = time.monotonic()
    baseline = config["baseline"]
    baseline_path = Path(baseline["path"]) / "manifest.json"
    if file_sha256(baseline_path) != baseline["manifest_sha256"]:
        raise ValueError("Combined baseline manifest differs from the pinned input")
    spec = GraphSpec.from_dict(json.loads(baseline_path.read_text(encoding="utf-8"))["spec"])
    if scratch_root.stat().st_dev == output.stat().st_dev or shutil.disk_usage(scratch_root).free < 14 * 1024**3:
        raise OSError("Require 14 GiB on a separate node-local scratch filesystem")
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Resume requires the original corpus configuration")
    if (output / "corpus").exists():
        raise FileExistsError("Preserve the existing finalized corpus")
    write_json(config_path, config)
    accepted = output / "ingested"
    accepted.mkdir(exist_ok=True)
    if os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                   nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))

    def progress(stage, **metrics):
        record = dict(stage=stage, job_id=os.environ.get("LSB_JOBID"),
                      elapsed_seconds=time.monotonic() - started, **metrics)
        write_json(output / "progress.json", record)
        print(json.dumps(record), flush=True)

    source = config["source"]
    with tempfile.TemporaryDirectory(prefix="latentmol-corpus-", dir=scratch_root) as temp:
        local = Path(temp)
        raw = local / "snapshot.gz"
        progress("downloading_snapshot")
        snapshot = download_snapshot(source, raw)
        snapshot_record = output / "snapshot.json"
        if snapshot_record.exists() and json.loads(snapshot_record.read_text(encoding="utf-8")) != snapshot:
            raise ValueError("Existing shards belong to a different source snapshot")
        write_json(snapshot_record, snapshot)
        input_rows = accepted_rows = accepted_bytes = shard_count = 0
        with gzip.open(raw, "rb") as stream:
            while True:
                chunk = local / "chunk.tsv"
                count, digest = write_chunk(stream, chunk, config["chunk_rows"])
                if not count:
                    break
                shard_count += 1
                destination = accepted / f"part-{shard_count:05}"
                staged = local / destination.name
                receipt = ingest(chunk, destination if destination.exists() else staged, source=source["name"],
                                 input_format=source["format"], spec=spec, workers=workers,
                                 input_sha256=digest, min_atoms=config["min_atoms"])
                if staged.exists():
                    accepted_bytes += publish_shard(staged, destination, config["max_ingest_bytes"] - accepted_bytes)
                    shutil.rmtree(staged)
                else:
                    accepted_bytes += sum(path.stat().st_size for path in destination.iterdir() if path.is_file())
                chunk.unlink()
                input_rows += count
                accepted_rows += receipt["counts"]["accepted"]
                progress("ingesting", input_rows=input_rows, accepted_rows=accepted_rows,
                         shards=shard_count, compressed_bytes=accepted_bytes)
        if len(list(accepted.iterdir())) != shard_count or not shard_count:
            raise ValueError("Committed ingest shards differ from the pinned source partition")
        raw.unlink()
        progress("finalizing", input_rows=input_rows, accepted_rows=accepted_rows,
                 shards=shard_count, compressed_bytes=accepted_bytes)
        max_output = min(config["max_output_bytes"], work3_headroom() - 1024**3)
        if max_output < 1:
            raise OSError("Work3 needs a 1 GiB reserve before final publication")
        report = finalize_corpus(dict(version="canonical-corpus-finalization-v1",
            baseline={key: baseline[key] for key in ("path", "dataset_sha256")},
            sources=[dict(name=source["name"], shard_root=str(accepted))],
            minimum_train_count=config["minimum_train_count"], hash_buckets=config["hash_buckets"]),
            output / "corpus", scratch_root=local, max_output_bytes=max_output)
        progress("completed", input_rows=input_rows, accepted_rows=accepted_rows, shards=shard_count,
                 manifest_sha256=file_sha256(output / "corpus" / "manifest.json"))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not 1 <= args.workers < int(os.environ.get("LSB_DJOB_NUMPROC", "0")):
        parser.error("Run inside an LSF allocation with a CPU slot reserved for the coordinator")
    args.output.mkdir(exist_ok=True)
    with (args.output / ".writer.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(json.loads(args.config.read_text(encoding="utf-8")), args.output.resolve(),
            args.scratch_root.resolve(strict=True), args.workers)


if __name__ == "__main__":
    main()
