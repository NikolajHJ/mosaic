"""Allocated controller: submit independent arms only after actual native success."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "root", "cpu", "ready"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ready-job", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Submission controller requires allocated DTU CPU")
    state = subprocess.run(
        ["bjobs", "-noheader", "-o", "stat", args.ready_job],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if state != "DONE":
        raise ValueError("Native readiness scheduler outcome is not DONE")
    assessment = args.ready / "assessment.json"
    result = json.loads(assessment.read_text())
    if (
        result["status"] != "passed"
        or result["source_manifest_sha256"] != args.manifest
    ):
        raise ValueError("Native assessment does not certify this source")
    for proof in result["proofs"]:
        if (
            hashlib.sha256(Path(proof["path"]).read_bytes()).hexdigest()
            != proof["sha256"]
        ):
            raise ValueError("Native evidence bytes changed")
    digest = hashlib.sha256(assessment.read_bytes()).hexdigest()
    args.root.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "submitting",
        "controller_job": os.environ["LSB_JOBID"],
        "ready_job": args.ready_job,
        "assessment_sha256": digest,
        "source_manifest_sha256": args.manifest,
        "arms": {},
    }
    output = args.root / "submissions.json"
    output.write_text(json.dumps(status, indent=2) + "\n")
    for arm in ("mix50", "random"):
        native = args.root / arm
        native.mkdir()
        subprocess.run(
            [
                "/work3/s234818/mosaic/envs/protenix-tiny/bin/python",
                str(args.release / "scripts/hpc/zhome_quota_preflight.py"),
                "--run-dir",
                str(native),
                "--minimum-free-gib",
                "8",
            ],
            check=True,
        )
        command = [
            "bsub",
            "-J",
            "mosaic-boltz-aff-" + arm,
            "-q",
            "gpua10",
            "-n",
            "4",
            "-R",
            "span[hosts=1]",
            "-R",
            "rusage[mem=8GB]",
            "-gpu",
            "num=1:mode=exclusive_process",
            "-W",
            "23:00",
            "-oo",
            str(native / "stdout.%J.log"),
            "-eo",
            str(native / "stderr.%J.log"),
            "bash",
            "-l",
            str(args.release / "scripts/hpc/brd4_boltz_run.lsf"),
            str(args.release),
            str(native),
            args.manifest,
            str(args.ready),
            str(args.cpu),
            digest,
            arm,
        ]
        # Persist intent before bsub; an ambiguous response is never retried.
        (native / "intent.json").write_text(json.dumps(command, indent=2) + "\n")
        reply = subprocess.run(command, check=True, capture_output=True, text=True)
        (native / "submission.txt").write_text(reply.stdout + reply.stderr)
        match = re.search(r"Job <(\d+)>", reply.stdout)
        if not match:
            raise RuntimeError(
                "Ambiguous bsub result; inspect evidence before retrying"
            )
        job = match.group(1)
        status["arms"][arm] = {
            "job_id": job,
            "directory": str(native),
            "initial_state": "submitted; native result pending",
        }
        output.write_text(json.dumps(status, indent=2) + "\n")
        snapshot = subprocess.run(
            ["bjobs", "-l", job], check=True, capture_output=True, text=True
        )
        (native / "scheduler-submitted.txt").write_text(snapshot.stdout)
    status["status"] = "submitted"
    output.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
