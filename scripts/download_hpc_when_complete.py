"""One-shot bounded HPC completion download; local work is file I/O only.

Polls one evidence job every two minutes, downloads once, verifies all bytes,
copies the archived offline report and appends actual outcomes to the ledger.
No local model/chemistry evaluation, job resubmission or automatic Git mutation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import time


def command(arguments, *, timeout=60):
    return subprocess.run(
        arguments, capture_output=True, text=True, timeout=timeout, check=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study-run", required=True)
    parser.add_argument("--report-run", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--timeout-hours", type=float, default=8)
    args = parser.parse_args()
    remote = PurePosixPath(args.archive)
    if (
        args.job <= 0
        or not re.fullmatch(
            r"/work3/s234818/mosaic/runs/[A-Za-z0-9._/-]+\.tar\.gz", args.archive
        )
        or ".." in remote.parts
    ):
        raise ValueError("Unexpected job/archive identity")
    for name in (args.study_run, args.report_run):
        if not re.fullmatch("[A-Za-z0-9_-]+", name):
            raise ValueError("Expected a single run directory name")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "waiting_for_allocated_hpc_evidence",
        "job_id": args.job,
        "archive": args.archive,
        "scope": "one-shot file transfer/integrity; no local numerical work",
    }

    def persist():
        status["utc"] = datetime.now(timezone.utc).isoformat()
        (output / "completion.json").write_text(json.dumps(status, indent=2) + "\n")

    persist()
    try:
        deadline = time.monotonic() + args.timeout_hours * 3600
        while time.monotonic() < deadline:
            result = command(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "dtu-hpc",
                    f"bjobs -noheader -o stat {args.job}",
                ]
            )
            states = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip()
                in (
                    "DONE",
                    "EXIT",
                    "RUN",
                    "PEND",
                    "PSUSP",
                    "USUSP",
                    "SSUSP",
                    "UNKWN",
                    "ZOMBI",
                )
            ]
            if len(states) != 1:
                raise RuntimeError("Could not authenticate a unique scheduler state")
            status["scheduler_state"] = states[0]
            persist()
            if states[0] == "EXIT":
                raise RuntimeError(
                    "Evidence job exited; inspect HPC logs, no success claimed"
                )
            if states[0] == "DONE":
                break
            time.sleep(120)
        else:
            raise TimeoutError("One-shot download deadline reached")
        scheduler = command(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "dtu-hpc",
                f"bjobs -l {args.job}",
            ]
        )
        (output / "evidence-job-final.txt").write_text(
            scheduler.stdout + scheduler.stderr
        )
        receipt_remote = str(remote.with_suffix(".receipt.json"))
        receipt_local = output / "archive-receipt.json"
        command(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "dtu-hpc:" + receipt_remote,
                str(receipt_local),
            ]
        )
        receipt = json.loads(receipt_local.read_text())
        if receipt["path"] != args.archive or not re.fullmatch(
            "[0-9a-f]{64}", receipt["sha256"]
        ):
            raise ValueError("Unexpected archive receipt")
        archive = output / remote.name
        command(
            [
                "scp",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "dtu-hpc:" + args.archive,
                str(archive),
            ],
            timeout=1800,
        )
        verification = command(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_hpc_evidence.py")),
                str(archive),
                receipt["sha256"],
            ],
            timeout=1800,
        )
        status["verification"] = json.loads(verification.stdout)
        extracted = archive.with_suffix("").with_suffix("")
        study = json.loads(
            (extracted / args.study_run / "payload/study/summary.json").read_text()
        )
        status["native_status"] = study["status"]
        report = extracted / args.report_run / "payload/report"
        if (report / "report.html").is_file():
            shutil.copytree(report, output / "report")
            status["report"] = str(output / "report/report.html")
        status["status"] = "downloaded_and_verified"
        persist()
        entry = (
            f"\n\n### Automatic completion receipt for known-ligand job {args.job}\n\n"
            f"Recorded {status['utc']}. Evidence job **{args.job} DONE**; "
            f"native study status **{study['status']}**, model job **{study['job_id']}**. "
            f"Archive SHA-256 `{receipt['sha256']}`, {receipt['bytes']} bytes; "
            f"{status['verification']['verified_files']} declared files verified locally "
            "using stdlib byte checks only. "
            f"Local completion receipt `{output / 'completion.json'}`. "
            f"Offline report: `{status.get('report', 'not produced; inspect retained failed run')}`. "
            "No local model/chemistry evaluation was performed; failed native outcomes "
            "are retained and do not establish a 1000-update success.\n"
        )
        with args.ledger.open("a", encoding="utf-8") as writer:
            writer.write(entry)
    except Exception as error:
        status.update(status="stopped", error=f"{type(error).__name__}: {error}")
        persist()
        with args.ledger.open("a", encoding="utf-8") as writer:
            writer.write(
                f"\n\n### Known-ligand completion download stopped\n\nEvidence job {args.job}; {status['error']}. See `{output / 'completion.json'}`. No successful download or native outcome inferred.\n"
            )
        raise


if __name__ == "__main__":
    main()
