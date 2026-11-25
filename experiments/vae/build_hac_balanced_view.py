"""Select a new atom-count-balanced membership from the completed sparse cache."""

import argparse
import json
import os
from pathlib import Path
from shutil import disk_usage

from latentmol.datasets.bulk_cache_build import work3_headroom
from latentmol.datasets.bulk_view import BalancedSparseCorpus, build_balanced_view
from latentmol.datasets.ordering import verify_canonical_training_cache
from latentmol.experiments import file_sha256, write_json


def completed_inputs(config):
    launch = json.loads(Path(config["cache_launch_receipt"]).read_text(encoding="utf-8"))
    if (launch["job_id"] != str(config["cache_job_id"]) or launch["status"] != "worker_exited"
            or launch["returncode"] != 0 or launch["ready"] is not True
            or launch["source_archive_sha256"] != config["cache_source_archive_sha256"]):
        raise ValueError("Expected the pinned sparse-cache job to finish successfully")
    corpus = Path(config["corpus"])
    if file_sha256(corpus / "manifest.json") != config["corpus_manifest_sha256"]:
        raise ValueError("The canonical corpus differs from the pinned source")
    cache = Path(config["sparse_cache"])
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    return corpus, cache, manifest["dataset_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        parser.error("Run inside an allocated CPU job")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus, cache, digest = completed_inputs(config)
    if args.output.exists():
        raise FileExistsError(args.output)
    cap = min(disk_usage(args.output.parent).free, work3_headroom(config["work3_account"])) - config["reserve_bytes"]
    if cap < 1:
        raise OSError("No dataset view space remains after the storage reserve")
    if os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                   nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ["LSB_JOBID"]))
    manifest = build_balanced_view(corpus, cache, args.output,
        corpus_manifest_sha256=config["corpus_manifest_sha256"], expected_cache_sha256=digest,
        train_count=config["train_count"], add_per_holdout=config["add_per_holdout"],
        seed=config["seed"], max_output_bytes=cap)
    view = BalancedSparseCorpus(args.output, expected_sha256=manifest["dataset_sha256"])
    for split in ("train", "val", "test"):
        data = view.split(split)
        if len(data) != manifest["counts"][split]:
            raise ValueError(f"Published view {split} count differs from its manifest")
        verify_canonical_training_cache(data, [0, len(data) - 1])
    print(json.dumps(dict(stage="balanced_view_complete", counts=manifest["counts"],
                          dataset_sha256=manifest["dataset_sha256"])), flush=True)


if __name__ == "__main__":
    main()
