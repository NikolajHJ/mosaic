"""Queue one CPU gate and one sequential matched pilot, without retries.

All model/fixture execution is allocated on DTU. The shared durable submitter
retains intents and receipts so a lost SSH reply cannot duplicate submission.
Fresh fail-closed gates run in each 25-update child; historical failures remain.
"""

import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_ramped_random import gpu_request
from submit_stability_study import authenticated, read, submit


def workflow(release, work, plan, plan_sha, archive_sha, manifest_sha):
    branch_mode = read(plan).get("version") in (2, 3, 4) if plan.is_file() else False
    stages = [
        ("cpu", "hpc", "0:45", 2, "4GB", None,
         ["bash", "scripts/hpc/boltz_matched_pilot_preflight.sh", str(plan), plan_sha]),
        ("native", "gpua100", "12:00" if branch_mode else "8:00", 4, "16GB", "done:cpu",
         ["python", "-m", "latentmol.repair_guidance.matched_pilot",
          "--plan", str(plan), "--sha256", plan_sha]),
    ]
    result = []
    for stage, queue, wall, cores, memory, dependency, command in stages:
        label = release.name + "-" + stage
        arguments = ["bsub", "-J", label, "-q", queue, "-n", str(cores),
            "-R", "rusage[mem=" + memory + "]", "-R", "span[hosts=1]", "-W", wall,
            "-oo", str(work / "logs" / (label + ".%J.out")),
            "-eo", str(work / "logs" / (label + ".%J.err"))]
        if stage == "native":
            arguments += ["-R", "select[gpu80gb]", "-gpu", gpu_request(managed_mps=True)]
        job = ["env", "LATENTMOL_SOURCE_ARCHIVE=" + str(release) + ".tar.gz",
            "LATENTMOL_WORK_ROOT=" + str(work), "bash", "scripts/hpc/boltz_run.sh",
            "python", "scripts/verify_stability_release.py", "--archive-sha256", archive_sha,
            "--manifest-sha256", manifest_sha, "--", "python", "-m", "latentmol.experiments",
            "--run-root", str(work / "runs"), "--name", label, "--", *command]
        result.append({"stage": stage, "cwd": str(release), "arguments": arguments,
                       "job_command": job, "dependency": dependency})
    return result


def authenticate_plan_assets(plan):
    """Authenticate the paired plan's schema, including both independent YAMLs."""
    references = [plan[key] for key in ("base_config", "prior_smoke", "prior_run", "native_template")]
    if [row.get("id") for row in plan["cases"]] != ["boltz_only", "boltz_defog"]:
        raise ValueError("Require the complete prescribed pair in control-first order")
    references.extend(row["config"] for row in plan["cases"])
    for reference in references:
        authenticated(Path(reference["path"]), reference["sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if (not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases")
            or any(not path.is_absolute() or not path.resolve().is_relative_to(work)
                   for path in (args.plan, args.receipts))):
        raise ValueError("Require sealed release and controls inside the declared DTU root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    plan = read(args.plan)
    if (source_manifest(release)["sha256"] != args.manifest_sha256
            or read(release / "release_manifest.json")["sha256"] != args.manifest_sha256
            or plan["source_archive_sha256"] != args.archive_sha256
            or plan["source_manifest_sha256"] != args.manifest_sha256
            or Path(plan["output_root"]) != work / "runs" / (release.name + "-pilot")):
        raise ValueError("Changed source identity or nonunique pilot output root")
    authenticate_plan_assets(plan)
    jobs = workflow(release, work, args.plan, args.plan_sha256, args.archive_sha256, args.manifest_sha256)
    identity = {"release": str(release), "plan": str(args.plan), "plan_sha256": args.plan_sha256,
                "archive_sha256": args.archive_sha256, "manifest_sha256": args.manifest_sha256,
                "scope": "Matched 25-update pilot; historical failed-state guards remain uncertified"}
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "jobs": jobs, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
