"""Metadata-only duplicate-safe submission of CPU gate then two serial GPUs.

The new comparison follows the existing search pilot without replacing jobs.
Both GPU stages require CPU success. Segment1 follows segment0 even if it
fails, retaining independent-seed results; its own native fixture/FD gates
must still pass. Source/config/data are authenticated again inside LSF.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

from latentmol.experiments import file_sha256, source_manifest
from latentmol.search_decoder_protocol import read_full_protocol
from submit_stability_study import authenticated, read, submit


def workflow(release, work, protocol, path, config_sha, archive_sha, manifest_sha, cpu_job=None):
    """Return scheduler argv only; no model imports or scheduler mutations."""
    stages = ["cpu"] if cpu_job is None else ["seed17", "seed23"]
    rows = []
    for stage in stages:
        cpu = stage == "cpu"
        label = release.name + "-" + stage
        arguments = ["bsub", "-J", label, "-q", "hpc" if cpu else "gpuv100", "-n", "2" if cpu else "4",
            "-R", "rusage[mem=4GB]" if cpu else "rusage[mem=16GB]", "-R", "span[hosts=1]",
            "-W", "0:45" if cpu else "23:00", "-oo", str(work / "logs" / (label + ".%J.out")),
            "-eo", str(work / "logs" / (label + ".%J.err"))]
        dependency = None
        if not cpu:
            arguments += ["-gpu", "num=1:mode=exclusive_process"]
            # Explicit dependencies retain CPU protection for both seeds.
            # The submission phase resolves the preceding new job ID for seed23.
            if stage == "seed17":
                arguments += ["-w", f"done({cpu_job}) && ended({protocol['preceding_job']})"]
            else:
                dependency = "ended:seed17"
        tail = ["bash", "scripts/hpc/search_decoder_full_tests.sh"] if cpu else [
            "bash", "scripts/hpc/search_decoder_full_segment.sh", str(path), config_sha,
            str(work / "runs" / (release.name + "-search-full")), "0" if stage == "seed17" else "1"]
        command = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
            "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
            "--archive-sha256", archive_sha, "--manifest-sha256", manifest_sha, "--", "python", "-m", "latentmol.experiments",
            "--run-root", str(work / "runs" / (release.name + "-" + stage)), "--name", "search-full-" + stage, "--", *tail]
        rows.append({"stage": stage, "cwd": str(release), "arguments": arguments,
                     "job_command": command, "dependency": dependency})
    return rows


def following_segment(row, cpu_job, first_job):
    """Keep the CPU gate explicit when resolving the second serial allocation."""
    if not all(isinstance(job, str) and re.fullmatch(r"[0-9]+", job) for job in (cpu_job, first_job)):
        raise ValueError("Scheduler dependencies require verified numeric job IDs")
    return {**row, "dependency": None,
            "arguments": [*row["arguments"], "-w", f"done({cpu_job}) && ended({first_job})"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "native"), required=True)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    work, release = args.work_root.resolve(strict=True), args.release.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases") or not args.receipts.resolve().is_relative_to(work):
        raise ValueError("Require immutable DTU release and receipt root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Source changed after sealing")
    path = release / "configs/search_decoder_full.json"
    config_sha = file_sha256(path)
    protocol = read_full_protocol(path, config_sha)
    authenticated(Path(protocol["initializer"]), protocol["initializer_sha256"])
    authenticated(Path(protocol["data"]) / "dataset.npz", protocol["dataset_sha256"])
    if read(Path(protocol["data"]) / "manifest.json")["split_sha256"] != protocol["split_sha256"]:
        raise ValueError("Protected split changed")
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
        "manifest_sha256": manifest_sha, "config_sha256": config_sha,
        "initializer_sha256": protocol["initializer_sha256"], "dataset_sha256": protocol["dataset_sha256"],
        "split_sha256": protocol["split_sha256"], "output_root": str(work / "runs" / (release.name + "-search-full"))}
    cpu_job = None
    if args.stage == "native":
        receipt = read(args.receipts / "cpu-receipt.json")
        intent = read(args.receipts / "cpu-intent.json")
        if receipt["intent"] != intent or intent["identity"] != identity:
            raise ValueError("CPU gate belongs to another source/config")
        cpu_job = receipt["job_id"]
        result = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", cpu_job],
                                text=True, capture_output=True, timeout=30, check=True)
        if result.stdout.strip() != "DONE":
            raise ValueError("Full native submission requires completed CPU gate")
    plan = workflow(release, work, protocol, path, config_sha, args.archive_sha256, manifest_sha, cpu_job)
    (work / "logs").mkdir(exist_ok=True)
    if not args.submit:
        print(json.dumps({"identity": identity, "plan": plan}, indent=2))
        return
    if args.stage == "cpu":
        jobs = submit(plan, args.receipts, identity)
    else:
        # Exact receipts make both calls idempotent. If the second reply is
        # lost, an exclusive intent prevents blindly queuing another job.
        jobs = submit(plan[:1], args.receipts, identity)
        second = following_segment(plan[1], cpu_job, jobs["seed17"])
        jobs.update(submit([second], args.receipts, identity))
    print(json.dumps(jobs), flush=True)


if __name__ == "__main__":
    main()
