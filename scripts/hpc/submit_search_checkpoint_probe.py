"""Immutable saved-script actual-checkpoint CPU recovery; duplicate-safe bsub."""

import argparse
import json
from pathlib import Path

from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require sealed release under DTU work root")
    if not args.receipts.is_absolute() or not args.receipts.resolve().is_relative_to(work):
        raise ValueError("Receipt path must stay within declared DTU root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Source changed after sealing")
    config_path = release / "configs/search_decoder_pilot.json"
    config_sha = file_sha256(config_path)
    config = read(config_path)
    authenticated(Path(config["initializer"]), config["initializer_sha256"])
    output = work / "runs" / (release.name + "-probe")
    identity = {"source_archive_sha256": args.archive_sha256, "manifest_sha256": manifest_sha,
        "config_sha256": config_sha, "initializer_sha256": config["initializer_sha256"],
        "output_root": str(output), "device": "cpu", "superseded_launch_job": "29623910"}
    label = release.name + "-cpu-probe"
    arguments = ["bsub", "-J", label, "-q", "hpc", "-n", "4", "-R", "rusage[mem=4GB]",
        "-R", "span[hosts=1]", "-W", "0:45", "-oo", str(work / "logs" / (label + ".%J.out")),
        "-eo", str(work / "logs" / (label + ".%J.err"))]
    command = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
        "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
        "--archive-sha256", args.archive_sha256, "--manifest-sha256", manifest_sha, "--",
        "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / (release.name + "-cpu-probe")),
        "--name", "search-actual-checkpoint-cpu", "--", "bash", "scripts/hpc/search_checkpoint_probe.sh",
        str(config_path), config_sha, str(output)]
    plan = [{"stage": "actual-probe", "cwd": str(release), "arguments": arguments,
             "job_command": command, "dependency": None}]
    print(json.dumps(submit(plan, args.receipts, identity) if args.submit else
                     {"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
