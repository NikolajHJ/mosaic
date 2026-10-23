"""Seal-aware LSF plan/submission for one CPU gate or serial native comparison.

Default prints an authenticated plan only. Native submission requires a DONE
CPU gate and its authenticated preparation manifest. Unknown scheduler replies
are retained and never automatically retried. No login-node model inference.
"""

import argparse
import json
from pathlib import Path
import subprocess

from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cpu", "native"), required=True)
    p.add_argument("--release", type=Path, default=Path.cwd())
    p.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--base-sha256", required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--comparison-sha256")
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    release, work = a.release.resolve(strict=True), a.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require an immutable release under the declared DTU root")
    if any(not path.is_absolute() or not path.resolve().is_relative_to(work)
           for path in (a.base_config, a.prepared_root, a.receipts)):
        raise ValueError("All case/receipt paths must stay inside the declared work root")
    authenticated(Path(str(release) + ".tar.gz"), a.archive_sha256)
    authenticated(a.base_config, a.base_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Executable source changed after sealing")
    guidance_sha = file_sha256(release / "configs/defog_repair_guidance.example.yaml")
    identity = {"release": str(release), "archive_sha256": a.archive_sha256,
                "manifest_sha256": manifest_sha, "base_config": str(a.base_config),
                "base_sha256": a.base_sha256, "guidance_sha256": guidance_sha,
                "prepared_root": str(a.prepared_root)}
    if a.stage == "cpu":
        if a.comparison_sha256 is not None:
            raise ValueError("CPU preparation has not produced a comparison yet")
        command = ["bash", "scripts/hpc/boltz_repair_preflight.sh", str(a.base_config), a.base_sha256,
                   guidance_sha, str(a.prepared_root)]
        queue, cores, memory, wall = "hpc", 2, "4GB", "0:30"
        dependency = None
    else:
        if not a.comparison_sha256:
            raise ValueError("Native comparison requires an authenticated prepared manifest")
        comparison = a.prepared_root / "comparison.json"
        authenticated(comparison, a.comparison_sha256)
        prepared = read(comparison)
        if (prepared["source_manifest_sha256"] != manifest_sha
                or prepared["source_archive_sha256"] != a.archive_sha256
                or prepared["base"]["sha256"] != a.base_sha256):
            raise ValueError("Preparation belongs to another source/base")
        receipt, intent = read(a.receipts / "cpu-receipt.json"), read(a.receipts / "cpu-intent.json")
        if receipt["intent"] != intent or intent["identity"] != identity:
            raise ValueError("CPU receipt differs from this immutable workflow")
        reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", receipt["job_id"]],
                               capture_output=True, text=True, check=True, timeout=30)
        if reply.stdout.strip() != "DONE":
            raise ValueError("Native release requires a completed CPU gate")
        command = ["bash", "scripts/hpc/boltz_repair_smoke.sh", str(comparison), a.comparison_sha256,
                   str(work / "runs" / (release.name + "-native"))]
        queue, cores, memory, wall = "gpua100", 4, "16GB", "6:15"
        dependency = receipt["job_id"]
    label = release.name + "-" + a.stage
    args = ["bsub", "-J", label, "-q", queue, "-n", str(cores), "-R", f"rusage[mem={memory}]",
            "-R", "span[hosts=1]", "-W", wall,
            "-oo", str(work / "logs" / (label + ".%J.out")), "-eo", str(work / "logs" / (label + ".%J.err"))]
    if dependency is not None:
        args += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process", "-w", f"done({dependency})"]
    job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
           "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
           "--archive-sha256", a.archive_sha256, "--manifest-sha256", manifest_sha,
           "--", "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / (release.name + "-" + a.stage)),
           "--name", "repair-" + a.stage, "--", *command]
    plan = [{"stage": a.stage, "cwd": str(release), "arguments": args, "job_command": job, "dependency": None}]
    (work / "logs").mkdir(exist_ok=True)
    print(json.dumps(submit(plan, a.receipts, identity) if a.submit else
                     {"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
