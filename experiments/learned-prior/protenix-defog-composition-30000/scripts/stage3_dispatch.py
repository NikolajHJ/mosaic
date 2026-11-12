"""Prepare reviewable exclusive-GPU bundles; submit only with --execute.

Run on the HPC login host using an already transferred, stable source snapshot.
This command neither modifies source nor cancels/restarts any existing job.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

PROFILES = {
    "v100": ("gpuv100", "30G"), "l40s": ("gpul40s", "44G"),
    "a10": ("gpua10", "22G"), "a100": ("gpua100", "38G"), "a40": ("gpua40", "44G"),
}
SCRIPT = "experiments/learned-prior/protenix-defog-composition-30000/scripts/stage3_campaign.py"


def utc():
    return datetime.now(timezone.utc).isoformat()


def assignment(seed, initialization, profiles):
    """Keep shared controls and all matched conditions on one profile per block."""
    digest = hashlib.sha256(f"{seed}|{initialization}".encode()).digest()
    return profiles[int.from_bytes(digest[:4], "big") % len(profiles)]


def prepare(plan_path, output, source, python, assets, profiles, hours=2, modules=()):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("mock"):
        raise ValueError("Mock software fixtures are not GPU experiments")
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles) - PROFILES.keys():
        raise ValueError(f"Choose distinct compatible profiles from {list(PROFILES)}")
    if not 1 <= hours <= 24:
        raise ValueError("Requested walltime must lie between1and24hours")
    for value in (source, python, assets):
        if not str(value).startswith("/") or "\n" in str(value):
            raise ValueError("HPC paths must be explicit absolute POSIX paths")
    for module in modules:
        if not re.fullmatch(r"[A-Za-z0-9_./+:-]+", module):
            raise ValueError("Unexpected module name")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    scripts = output / "scripts"
    logs = output / "logs"
    for directory in (scripts, logs):
        directory.mkdir()
    rows = []
    for bundle in plan["bundles"]:
        bid = bundle["bundle_id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", bid):
            raise ValueError("Unsafe bundle ID")
        profile = assignment(bundle["seed"], bundle["initialization"], profiles)
        queue, memory = PROFILES[profile]
        payload = [
            "#!/bin/bash", "set -euo pipefail", ': "${LSB_JOBID:?Allocated GPU required}"',
            *(["module load " + " ".join(map(shlex.quote, modules))] if modules else []),
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "export JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false JAX_PLATFORMS=cuda",
            "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2",
            "export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1",
            "export MOSAIC_GPU_PROFILE=" + shlex.quote(profile),
            "unset PYTHONHOME PYTHONOPTIMIZE XLA_FLAGS",
            "export PYTHONPATH=" + shlex.quote(f"{source}/src:{source}/models/defog/src"),
            "cd " + shlex.quote(source),
            "exec " + shlex.join([str(python), f"{source}/{SCRIPT}", "run-bundle", "--plan", str(plan_path),
                                  "--bundle-id", bid, "--assets", str(assets)]),
        ]
        path = scripts / f"{bid}.sh"
        path.write_text("\n".join(payload) + "\n", encoding="utf-8", newline="\n")
        command = ["bsub", "-q", queue, "-J", f"s3-{bid}"[:120], "-n", "4",
                   "-R", "rusage[mem=4096] span[hosts=1]", "-W", f"{hours}:00",
                   "-gpu", f"num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem={memory}",
                   "-o", str(logs / f"{bid}.%J.out"), "-e", str(logs / f"{bid}.%J.err")]
        rows.append({"bundle_id": bid, "seed": bundle["seed"], "initialization": bundle["initialization"],
                     "gpu_profile": profile, "queue": queue, "run_ids": bundle["run_ids"],
                     "script": str(path.resolve()), "bsub": command,
                     "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "receipt": str((plan_path.parent / "bundles" / f"{bid}.submission.json").resolve())})
    result = {"created_utc": utc(), "plan": str(plan_path),
              "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "source": str(source), "source_policy": "explicit stable snapshot; never edit submitted source",
              "profiles": profiles, "jobs": rows}
    (output / "dispatch.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def submit(dispatch_path, bundle_ids=(), execute=False):
    path = Path(dispatch_path)
    dispatch = json.loads(path.read_text(encoding="utf-8"))
    if hashlib.sha256(Path(dispatch["plan"]).read_bytes()).hexdigest() != dispatch["plan_sha256"]:
        raise ValueError("Scientific plan changed after dispatch preparation")
    wanted = set(bundle_ids)
    jobs = [r for r in dispatch["jobs"] if not wanted or r["bundle_id"] in wanted]
    if wanted - {r["bundle_id"] for r in jobs}:
        raise ValueError("Requested unknown bundle IDs")
    responses = []
    for row in jobs:
        if hashlib.sha256(Path(row["script"]).read_bytes()).hexdigest() != row["script_sha256"]:
            raise ValueError("Job script changed after dispatch preparation")
        if not execute:
            responses.append({"bundle_id": row["bundle_id"], "command": row["bsub"], "submitted": False})
            continue
        # A durable pre-submission receipt is intentional. If transport dies
        # after bsub accepts the job, the uncertain claim prevents duplicates.
        receipt = Path(row["receipt"])
        record = {"bundle_id": row["bundle_id"], "state": "unknown", "attempted_utc": utc(),
                  "bsub": row["bsub"], "run_ids": row["run_ids"]}
        with receipt.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        try:
            result = subprocess.run(row["bsub"], input=Path(row["script"]).read_text(),
                                    text=True, capture_output=True, timeout=60)
            match = re.search(r"Job <(\d+)>", result.stdout)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode,
                          state="accepted" if result.returncode == 0 and match else "unknown",
                          job_id=match.group(1) if match else None)
        finally:
            receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        responses.append(record)
        if record["state"] != "accepted":
            raise RuntimeError(f"Uncertain submission for{row['bundle_id']}; reconcile scheduler/receipt before retrying")
    return responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in ("plan", "output", "source", "python", "assets"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, required=True)
    p.add_argument("--hours", type=int, default=2)
    p.add_argument("--modules", nargs="*", default=[])
    s = sub.add_parser("submit")
    s.add_argument("--dispatch", required=True)
    s.add_argument("--bundle-id", action="append", default=[])
    s.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.plan, args.output, args.source, args.python, args.assets,
                         args.profiles, args.hours, args.modules)
        print(json.dumps({"jobs_prepared": len(result["jobs"]), "submitted": False}))
    else:
        print(json.dumps(submit(args.dispatch, args.bundle_id, args.execute), indent=2))


if __name__ == "__main__":
    main()
