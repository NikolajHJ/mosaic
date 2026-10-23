"""Duplicate-safe submission: CPU engineering gate then one serial GPU study.

Source, protected data, initializer and config bytes are authenticated before
bsub. This is metadata inspection only on the login node. Native inference and
training happen exclusively inside an allocation. Keep uncertain intents;
never retry a submission after an unknown outcome without reconciling it.
"""

import argparse
import json
from pathlib import Path
import subprocess

from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "native"), required=True)
    parser.add_argument("--kind", choices=("defect", "search"), default="defect")
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require an immutable release inside the declared DTU work root")
    if not args.receipts.is_absolute() or not args.receipts.resolve().is_relative_to(work):
        raise ValueError("Receipts must stay inside the declared DTU work root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Source changed after sealing")
    config_path = release / ("configs/search_decoder_pilot.json" if args.kind == "search" else "configs/decoder_validity_study.json")
    config = read(config_path)
    authenticated(Path(config["initializer"]), config["initializer_sha256"])
    authenticated(Path(config["data"]) / "dataset.npz", config["dataset_sha256"])
    data_manifest = read(Path(config["data"]) / "manifest.json")
    if data_manifest["split_sha256"] != config["split_sha256"]:
        raise ValueError("Dataset split identity mismatch")
    config_sha = file_sha256(config_path)
    campaign = work / "runs" / (release.name + ("-search-study" if args.kind == "search" else "-decoder-study"))
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
        "manifest_sha256": manifest_sha, "config_sha256": config_sha,
        "initializer_sha256": config["initializer_sha256"], "dataset_sha256": config["dataset_sha256"],
        "split_sha256": config["split_sha256"], "output_root": str(campaign), "kind": args.kind}
    dependency = None
    if args.stage == "cpu":
        command = ["bash", "scripts/hpc/search_decoder_tests.sh" if args.kind == "search" else "scripts/hpc/decoder_validity_tests.sh"]
        queue, cores, memory, wall = "hpc", 2, "4GB", "0:30"
    else:
        receipt = read(args.receipts / "cpu-receipt.json")
        intent = read(args.receipts / "cpu-intent.json")
        if receipt["intent"] != intent or intent["identity"] != identity:
            raise ValueError("CPU receipt belongs to another immutable experiment")
        state = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", receipt["job_id"]],
                               text=True, capture_output=True, check=True, timeout=30)
        if state.stdout.strip() != "DONE":
            raise ValueError("Native experiment requires a DONE CPU engineering gate")
        dependency = receipt["job_id"]
        command = ["bash", "scripts/hpc/search_decoder_study.sh" if args.kind == "search" else "scripts/hpc/decoder_validity_study.sh",
                   str(config_path), config_sha, str(campaign)]
        queue, cores, memory, wall = "gpuv100", 4, "16GB", "8:00" if args.kind == "search" else "23:00"
    label = release.name + "-" + args.stage
    arguments = ["bsub", "-J", label, "-q", queue, "-n", str(cores), "-R", f"rusage[mem={memory}]",
        "-R", "span[hosts=1]", "-W", wall, "-oo", str(work / "logs" / (label + ".%J.out")),
        "-eo", str(work / "logs" / (label + ".%J.err"))]
    if dependency:
        expression = f"done({dependency})"
        if args.kind == "search":
            preceding = config["preceding_job"]
            if not isinstance(preceding, str) or not preceding.isdigit():
                raise ValueError("Search pilot preceding job must be a scheduler ID")
            expression += f" && ended({preceding})"
        arguments += ["-gpu", "num=1:mode=exclusive_process", "-w", expression]
    job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
        "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
        "--archive-sha256", args.archive_sha256, "--manifest-sha256", manifest_sha,
        "--", "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / (release.name + "-" + args.stage)),
        "--name", "decoder-validity-" + args.stage, "--", *command]
    plan = [{"stage": args.stage, "cwd": str(release), "arguments": arguments,
             "job_command": job, "dependency": None}]
    (work / "logs").mkdir(exist_ok=True)
    print(json.dumps(submit(plan, args.receipts, identity) if args.submit else
                     {"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
