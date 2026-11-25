"""Build a sparse graph cache after the pinned corpus job publishes its manifest."""

import argparse
import json
import os
from pathlib import Path
import time

from latentmol.datasets.bulk_cache import SparseGraphDataset, validate_sparse_cache
from latentmol.datasets.bulk_cache_build import build_sparse_corpus
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json


def completed_corpus(config):
    root = Path(config["parent_build"])
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    corpus = root / "corpus"
    manifest_path = corpus / "manifest.json"
    if progress["stage"] != "completed" or progress["job_id"] != str(config["parent_job_id"]):
        raise ValueError("Corpus completion marker is missing or belongs to a different job")
    digest = file_sha256(manifest_path)
    if progress["manifest_sha256"] != digest:
        raise ValueError("Finalized corpus manifest differs from the completion marker")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest["baseline_dataset_sha256"] != config["baseline_dataset_sha256"]
            or any(manifest["counts"][key] != value for key, value in config["holdout_counts"].items())):
        raise ValueError("Finalized corpus has a different baseline or holdout membership budget")
    return corpus, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    workers = int(os.environ.get("LSB_DJOB_NUMPROC", "0")) - 1
    if not os.environ.get("LSB_JOBID") or workers < 1 or args.wall_seconds < 1:
        parser.error("Run inside an allocated CPU job with a positive wall budget")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus, digest = completed_corpus(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                   nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ["LSB_JOBID"]))
    manifest = build_sparse_corpus(corpus, args.output, manifest_sha256=digest, workers=workers,
                                   work3_account=config["work3_account"], reserve_bytes=config["reserve_bytes"],
                                   seed=config["selection_seed"], deadline=started + args.wall_seconds)
    receipt = validate_sparse_cache(args.output, expected_sha256=manifest["dataset_sha256"])
    checks = {}
    for split in ("train", "val", "test"):
        data = SparseGraphDataset(args.output, split, validated_cache=receipt)
        checks[split] = verify_canonical_training_cache(data, range(min(16, len(data))))
    print(json.dumps(dict(stage="cache_validated", counts=manifest["counts"],
                          dataset_sha256=receipt.dataset_sha256, canonical_graph_checks=checks)), flush=True)


if __name__ == "__main__":
    main()
