"""Allocated stage dispatcher; inputs are pinned before any numerical imports.

Full preparation and CUDA benchmarking await allocated validation. The CPU
contracts and inventory passed in source094. Every queued benchmark independently
authenticates successful preparation before numerical imports. No downloads or
VPN/SSH changes are performed here.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from latentmol.experiments import file_sha256, source_manifest, write_json
from large_triangle_evidence import authenticate_inventory_reference, authenticate_prepared_reference, require_stage_completion


def bindings(path, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Campaign binding file changed")
    value = json.loads(path.read_text())
    if value["version"] != 1:
        raise ValueError("Unsupported binding version")
    work = Path(value["work_root"]).resolve()
    if not str(work).startswith("/work3/"):
        raise ValueError("Require a DTU work3 project root")
    paths = [value[key] for key in ("raw_input", "reserve_cache", "data", "campaign")]
    paths += [row["checkpoint"] for row in value["references"].values()]
    if any(not isinstance(p, str) or not Path(p).is_absolute() or not Path(p).resolve().is_relative_to(work) for p in paths):
        raise ValueError("Bind absolute input/output paths inside the project work root")
    source, reserved, data, campaign = (Path(value[key]).resolve() for key in
                                      ("raw_input", "reserve_cache", "data", "campaign"))
    if (data == reserved or data.is_relative_to(reserved) or reserved.is_relative_to(data)
            or campaign == data or campaign.is_relative_to(data) or data.is_relative_to(campaign)
            or source.is_relative_to(data) or source.is_relative_to(campaign)):
        raise ValueError("Require distinct source, protected cache, new cache and campaign paths")
    hashes = [value["raw_input_sha256"], value["reserve_manifest_sha256"]]
    hashes += [row["sha256"] for row in value["references"].values()]
    if any(not isinstance(h, str) or not re.fullmatch("[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("All inputs/reference checkpoints need authenticated SHA256 values")
    if "original" not in value["references"]:
        raise ValueError("Retain the original checkpoint comparison reference")
    if any(not re.fullmatch("[a-zA-Z0-9_-]+", label) for label in value["references"]):
        raise ValueError("Reference labels must be simple names")
    if set(value["references"]) & {"epoch100", "validation_reconstruction", "validation_validity"}:
        raise ValueError("Historical references cannot replace candidate roles")
    return value


def completed(args, value, artifacts=()):
    """Allocated evidence remains usable after successful LSF jobs age out.

    Submission receipts still bind the actual job ID; missing scheduler history
    alone never establishes success. Every declared artifact is pinned by hash.
    """
    key = args.stage + "-" + args.run_id if args.stage in ("segment", "recover", "final") else args.stage
    root = Path(value["campaign"]) / "stage-completions"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (key + ".json")
    if path.exists():
        raise FileExistsError("Completion receipts are immutable: " + str(path))
    write_json(path, {"status": "passed", "stage": key, "job_id": os.environ["LSB_JOBID"],
                      "bindings_sha256": args.bindings_sha256,
                      "source_sha256": source_manifest(Path.cwd())["sha256"],
                      "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
                      "artifacts": {str(p): file_sha256(p) for p in artifacts}, "stage_returncode": 0})


def require_processed_input_identity(report, value):
    """Bind the inputs actually processed, closing the pre-call hash race."""
    protected = report["provenance"]["reserved_cache"]
    if (report["raw_sha256"] != value["raw_input_sha256"]
            or protected["manifest_sha256"] != value["reserve_manifest_sha256"]):
        raise ValueError("Processed source/protected inputs differ from campaign bindings")


def require_benchmark_preparation(args, value, release):
    """Certify the bound cache before any GPU tests or numerical imports."""
    campaign = Path(value["campaign"])
    identity = {"release": str(release.resolve()),
                "archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"],
                "manifest_sha256": source_manifest(release)["sha256"],
                "bindings_sha256": args.bindings_sha256, "campaign": str(campaign)}
    stage = "cache" if value.get("prepared_reference") else "prepare"
    if stage == "cache" and (args.prepare_receipt_sha256 or args.prepare_job_id):
        raise ValueError("Prepared-cache reuse needs completed current validation, not queued preparation")
    if args.prepare_receipt_sha256 and file_sha256(campaign / "receipts/prepare-receipt.json") != args.prepare_receipt_sha256:
        raise ValueError("Queued preparation receipt changed")
    job = require_stage_completion(campaign / "receipts", stage, identity)
    if args.prepare_job_id and job != args.prepare_job_id:
        raise ValueError("Queued preparation job identity differs")
    completion = json.loads((campaign / "stage-completions" / (stage + ".json")).read_text())
    manifest = Path(value["data"]) / "manifest.json"
    if completion["artifacts"].get(str(manifest)) != file_sha256(manifest):
        raise ValueError("Preparation did not certify the bound data manifest")
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "inventory", "prepare", "cache", "benchmark", "segment", "recover", "final"))
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--segment-id")
    parser.add_argument("--run-id", default="attempt001")
    parser.add_argument("--prepare-job-id")
    parser.add_argument("--prepare-receipt-sha256")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("All numerical stages must run in an allocated DTU job")
    value = bindings(args.bindings, args.bindings_sha256)
    campaign = Path(value["campaign"])
    protocol = Path("configs/large_triangle_protocol.json")
    config = Path("configs/vae_large_triangle.yaml")
    if bool(args.prepare_job_id) != bool(args.prepare_receipt_sha256):
        raise ValueError("Queued preparation requires both its job and receipt identity")
    if args.prepare_job_id and args.stage != "benchmark":
        raise ValueError("Queued preparation is supported only for the benchmark")
    if args.stage == "benchmark":
        # LSF's done() dependency controls scheduling, but is not an artifact
        # certificate. Authenticate the same immutable preparation again before
        # CUDA tests, model construction or opening its newly written arrays.
        require_benchmark_preparation(args, value, Path.cwd())
    if args.stage == "cpu":
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_large_data.py",
                        "tests/test_triangle_vae.py", "tests/test_large_vae_pipeline.py",
                        "tests/training/test_training.py", "tests/test_refined_decoder.py",
                        "tests/test_large_triangle_evidence.py", "tests/test_large_vae_evaluation.py"], check=True)
        completed(args, value)
        return
    if args.stage == "cache":
        # This is a new validation allocation, not a recreated preparation run.
        # The prior source095 dataset/receipts remain unchanged and explicitly
        # identified; current CPU and all fresh benchmark gates stay mandatory.
        prior = authenticate_prepared_reference(value, json.loads(protocol.read_text()), Path.cwd())
        from latentmol.data import validate_large_cache
        validated = validate_large_cache(Path(value["data"]))
        if validated.dataset_sha256 != prior["dataset_sha256"]:
            raise ValueError("Validated cache differs from its authenticated preparation")
        report_path = campaign / "cache-validation.json"
        if report_path.exists():
            raise FileExistsError("Current cache validation report is immutable")
        write_json(report_path, {"status": "passed", "prior_preparation": prior,
                                 "job_id": os.environ["LSB_JOBID"],
                                 "scope": "Reused prepared dataset; allocated complete file/array/schema/split validation, no re-preparation"})
        completed(args, value, [Path(value["data"]) / "manifest.json", report_path])
        return
    if args.stage in ("inventory", "prepare"):
        # The reused source094 inventory has separate pinned provenance; its
        # receipt never stands in for this release's CPU or prepare validation.
        if args.stage == "prepare":
            authenticate_inventory_reference(value, json.loads(protocol.read_text()), Path.cwd())
        from latentmol.large_data import inventory_large_dataset, prepare_large_dataset
        for path, expected in ((Path(value["raw_input"]), value["raw_input_sha256"]),
                               (Path(value["reserve_cache"]) / "manifest.json", value["reserve_manifest_sha256"])):
            if file_sha256(path) != expected:
                raise ValueError("Raw/protected preparation input identity changed")
        if args.stage == "inventory":
            # Counts are deliberately not read from the unattainable initial
            # protocol. This immutable report chooses no train/held-out roles;
            # the next source must explicitly seal counts after its review.
            output = campaign / "inventory"
            report = inventory_large_dataset(Path(value["raw_input"]), output, Path(value["reserve_cache"]),
                                             seed=17, smiles_column=value["smiles_column"])
            require_processed_input_identity(report, value)
            completed(args, value, [output / "inventory.json"])
            return
        counts = json.loads(protocol.read_text())["requested_counts"]
        report = prepare_large_dataset(Path(value["raw_input"]), Path(value["data"]), Path(value["reserve_cache"]),
                                       train_count=counts["train"], val_count=counts["val"], test_count=counts["test"],
                                       seed=17, smiles_column=value["smiles_column"])
        require_processed_input_identity(report, value)
        completed(args, value, [Path(value["data"]) / "manifest.json"])
        return
    if args.stage == "benchmark":
        # A GPU gate additionally exercises deterministic native CUDA gradients.
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_triangle_vae.py",
                        "tests/test_large_vae_pipeline.py", "tests/training/test_training.py"], check=True)
    command = [sys.executable, "-m", "latentmol.large_vae", args.stage, "--config", str(config),
               "--protocol", str(protocol), "--data", value["data"], "--campaign", str(campaign),
               "--gate", str(campaign / "benchmark" / "passed.json")]
    if args.stage == "benchmark":
        command += ["--output", str(campaign / "benchmark")]
    elif args.stage in ("segment", "recover"):
        command += ["--segment-id", args.segment_id]
        if args.stage == "recover":
            command += ["--output", str(campaign / "recoveries" / args.run_id)]
    else:
        for label, row in value["references"].items():
            if file_sha256(Path(row["checkpoint"])) != row["sha256"]:
                raise ValueError("Historical reference checkpoint changed")
            command += ["--reference", label + "=" + row["checkpoint"]]
            command += ["--reference-sha256", label + "=" + row["sha256"]]
        command += ["--output", str(campaign / "final" / args.run_id)]
    subprocess.run(command, check=True)
    if args.stage == "benchmark":
        reference_command = [sys.executable, "-m", "latentmol.large_vae", "reference-validation",
                             "--config", str(config), "--protocol", str(protocol), "--data", value["data"],
                             "--gate", str(campaign / "benchmark/passed.json"),
                             "--output", str(campaign / "reference-validation")]
        for label, row in value["references"].items():
            reference_command += ["--reference", label + "=" + row["checkpoint"],
                                  "--reference-sha256", label + "=" + row["sha256"]]
        subprocess.run(reference_command, check=True)
        artifacts = [campaign / "benchmark/passed.json", campaign / "benchmark/selected-training.yaml",
                     campaign / "reference-validation/comparison.json"]
    elif args.stage in ("segment", "recover"):
        progress = json.loads((campaign / "progress.json").read_text())
        # Progress itself changes in later segments. Its immutable checkpoint
        # and optional recovery receipt carry the durable completion evidence.
        artifacts = [Path(progress["checkpoint"])]
        if "recovery_receipt" in progress:
            artifacts.append(Path(progress["recovery_receipt"]))
    else:
        artifacts = [campaign / "final" / args.run_id / "comparison.json"]
    completed(args, value, artifacts)


if __name__ == "__main__":
    main()
