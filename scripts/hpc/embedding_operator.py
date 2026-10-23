"""Standard-library immutable staging, quota and LSF actions for one prototype.

Login-node operations are byte/metadata/scheduler work only. No model imports.
Each submission writes exclusive intent before bsub and never retries an
ambiguous submission. Previous campaigns and all VAE jobs remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def authenticate(release, expected_archive, expected_plan):
    if digest(str(release) + ".tar.gz") != expected_archive:
        raise ValueError("Release archive hash mismatch")
    with tarfile.open(str(release) + ".tar.gz") as archive:
        sealed_manifest = archive.extractfile("embedding-release-manifest.json").read()
    manifest_bytes = (release / "embedding-release-manifest.json").read_bytes()
    if manifest_bytes != sealed_manifest:
        raise ValueError("Extracted release manifest differs from authenticated archive")
    manifest = json.loads(manifest_bytes)
    for name, sha in manifest["members"].items():
        path = (release / name).resolve()
        if not path.is_relative_to(release) or digest(path) != sha:
            raise ValueError("Release member differs: " + name)
    if digest(release / "configs/epoch36_boltz_embedding_bridge.json") != expected_plan:
        raise ValueError("Prototype plan hash mismatch")


def quota():
    run = subprocess.run(["getquota_work3.sh"], text=True, capture_output=True, timeout=60)
    if run.returncode:
        raise RuntimeError("Fresh work3 quota unavailable: " + run.stderr)
    rows = re.findall(r"\|\|\s*([0-9.]+)\s*GiB\|\s*([0-9.]+)\s*GiB", run.stdout)
    if not rows or any(float(limit) - float(used) < 5 for used, limit in rows):
        raise RuntimeError("Fresh work3 quota lacks required 5 GiB headroom")
    return {"stdout": run.stdout, "stderr": run.stderr, "returncode": run.returncode,
            "minimum_free_gib": min(float(limit)-float(used) for used, limit in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit", "verify-run", "collect"))
    parser.add_argument("--stage", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    work = Path("/work3/s234818/latentmol")
    release, campaign = args.release.resolve(), args.campaign.resolve()
    if (release.parent != work / "releases" or campaign.parent != work / "runs"
            or not campaign.name.startswith("source137-epoch36-boltz-embedding-bridge-attempt")):
        raise ValueError("Prototype paths must belong to independent source137 namespace")
    authenticate(release, args.archive_sha256, args.plan_sha256)
    control = work / "submissions" / campaign.name
    control.mkdir(parents=True, exist_ok=True)
    identity = dict(release=str(release), campaign=str(campaign),
        archive_sha256=args.archive_sha256, plan_sha256=args.plan_sha256)
    if args.action == "verify-run":
        if not os.environ.get("LSB_JOBID"):
            raise RuntimeError("Execution requires LSF allocation")
        fresh_quota = quota()
        write_new(control / (args.stage + "-launch-quota.json"), fresh_quota)
        os.chdir(release)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        command = [sys.executable, "-m", "latentmol.boltz.embedding_experiment", args.stage,
            "--plan", str(release / "configs/epoch36_boltz_embedding_bridge.json"),
            "--output", str(campaign / args.stage)]
        os.execv(sys.executable, command)
    if args.action == "collect":
        output = {"identity": identity, "files": {}}
        for stage in ("cpu", "gpu"):
            for name in ("receipt.json", "intent.json", "launch-quota.json"):
                path = control / (stage + "-" + name)
                if path.exists():
                    output["files"][str(path)] = path.read_text()
            receipt_path = control / (stage + "-receipt.json")
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                job = receipt.get("job_id")
                if job:
                    run = subprocess.run(["bjobs", "-a", "-l", job], text=True, capture_output=True, timeout=30)
                    output[stage + "_scheduler"] = dict(stdout=run.stdout, stderr=run.stderr, returncode=run.returncode)
                    for suffix in ("out", "err"):
                        log = work / "logs" / f"{campaign.name}-{stage}.{job}.{suffix}"
                        if log.exists():
                            output["files"][str(log)] = log.read_text(errors="replace")[-24000:]
            for name in ("result.json", "cpu-gate.json", "cpu-tests.xml", "cpu-tests.log", "native-gate.json",
                         "native-projection-bank.json", "learned-diagnostic.json", "fit/training-result.json", "paired-data/manifest.json"):
                path = campaign / stage / name
                if path.exists():
                    output["files"][str(path)] = path.read_text(errors="replace")[-100000:]
        print(json.dumps(output))
        return
    receipt_path = control / (args.stage + "-receipt.json")
    intent_path = control / (args.stage + "-intent.json")
    if intent_path.exists() or receipt_path.exists():
        raise FileExistsError("Submission already attempted; inspect receipt before any new action")
    if args.stage == "gpu":
        cpu_path = campaign / "cpu/result.json"
        cpu = json.loads(cpu_path.read_text())
        if (cpu["status"] != "passed" or cpu["source_sha256"] != args.archive_sha256
                or cpu["plan_sha256"] != args.plan_sha256):
            raise ValueError("GPU requires this release's completed CPU pass")
        receipt = json.loads((control / "cpu-receipt.json").read_text())
        state = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", receipt["job_id"]],
                               text=True, capture_output=True, timeout=30)
        if state.stdout.strip() not in ("DONE", "") or (state.returncode and "not found" not in state.stderr.lower()):
            raise ValueError("CPU predecessor allocation has not completed")
    fresh_quota = quota()
    campaign.mkdir(parents=True, exist_ok=True)
    label = campaign.name + "-" + args.stage
    bsub = ["bsub", "-J", label, "-q", "hpc" if args.stage == "cpu" else "gpua100",
        "-n", "4", "-R", "rusage[mem=8GB]" if args.stage == "cpu" else "rusage[mem=16GB]",
        "-R", "span[hosts=1]", "-W", "0:45" if args.stage == "cpu" else "4:00",
        "-oo", str(work / "logs" / (label + ".%J.out")),
        "-eo", str(work / "logs" / (label + ".%J.err"))]
    if args.stage == "gpu":
        bsub += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process"]
    job = ["env", "PYTHONDONTWRITEBYTECODE=1", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz",
        "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/hpc/embedding_operator.py", "verify-run",
        "--stage", args.stage, "--release", str(release), "--archive-sha256", args.archive_sha256,
        "--plan-sha256", args.plan_sha256, "--campaign", str(campaign)]
    intent = dict(identity=identity, stage=args.stage, arguments=bsub, command=job, fresh_quota=fresh_quota)
    write_new(intent_path, intent)
    run = subprocess.run(bsub, input="#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(release)) +
        "\nexec " + shlex.join(job) + "\n", text=True, capture_output=True, timeout=60)
    match = re.search(r"Job <(\d+)>", run.stdout)
    receipt = dict(intent=intent, returncode=run.returncode, stdout=run.stdout, stderr=run.stderr,
                   job_id=match.group(1) if match else None, status="submitted_not_passed" if run.returncode == 0 and match else "submission_failed")
    write_new(receipt_path, receipt)
    print(json.dumps(receipt))
    if run.returncode or not match:
        raise RuntimeError("LSF did not confirm submission")


if __name__ == "__main__":
    main()
