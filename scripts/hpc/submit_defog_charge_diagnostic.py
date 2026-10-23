"""Seal-aware durable submission of CPU parity then exact repair-only GPU replay.

No model imports or numerical work on the login node. A failed CPU gate keeps
native work pending; unknown bsub outcomes cannot be blindly retried.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    release = args.release.resolve(strict=True)
    work = Path("/work3/s234818/latentmol")
    if release.parent != work / "releases" or args.control.parent != work / "submissions":
        raise ValueError("Require release/control directly inside the declared work root")
    sys.path.insert(0, str(release / "src"))
    sys.path.insert(0, str(release / "scripts/hpc"))
    from latentmol.experiments import source_manifest, file_sha256
    from submit_stability_study import authenticated, read, submit
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    if (source_manifest(release)["sha256"] != args.manifest_sha256
            or read(release / "release_manifest.json")["sha256"] != args.manifest_sha256):
        raise ValueError("Executable source differs from sealed identity")
    label = release.name
    parity = work / "runs" / (label + "-parity")
    replay = work / "runs" / (label + "-replay")
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
                "manifest_sha256": args.manifest_sha256, "plan": str(args.plan),
                "plan_sha256": args.plan_sha256, "submission_script_sha256": file_sha256(Path(__file__)),
                "scope": "CPU parity followed by repair-only exact candidates; no Boltz or latent updates"}
    plan = []
    for stage, queue, limit, dependency, command in (
        ("cpu", "hpc", "0:30", None, ["bash", "scripts/hpc/defog_charge_cpu_gate.sh",
            str(work / "cache/defog-official-guacamol/DeFoG"), str(parity)]),
        ("native", "gpul40s", "1:30", "done:cpu", ["python", "-m", "latentmol.defog.charge_repair_diagnostic",
            "--plan", str(args.plan), "--plan-sha256", args.plan_sha256,
            "--run-root", str(replay), "--parity-root", str(parity)])):
        job_label = label + "-" + stage
        arguments = ["bsub", "-J", job_label, "-q", queue, "-n", "4", "-R", "rusage[mem=8GB]",
                     "-R", "span[hosts=1]", "-W", limit,
                     "-oo", str(work / "logs" / (job_label + ".%J.out")),
                     "-eo", str(work / "logs" / (job_label + ".%J.err"))]
        if stage == "native":
            arguments += ["-gpu", "num=1:mode=exclusive_process"]
        job = ["env", "LATENTMOL_SOURCE_ARCHIVE=" + str(release) + ".tar.gz",
            "LATENTMOL_WORK_ROOT=" + str(work), "bash", "scripts/hpc/boltz_run.sh",
            "python", "scripts/verify_stability_release.py", "--archive-sha256", args.archive_sha256,
            "--manifest-sha256", args.manifest_sha256, "--", "python", "-m", "latentmol.experiments",
            "--run-root", str(work / "runs"), "--name", job_label, "--", *command]
        plan.append({"stage": stage, "cwd": str(release), "arguments": arguments,
                     "dependency": dependency, "job_command": job})
    print(json.dumps(submit(plan, args.control / "receipts", identity) if args.submit
                     else {"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
