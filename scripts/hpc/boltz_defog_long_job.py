"""One allocated preflight -> authenticated preparation -> serial 500 pair.

No scheduler callbacks or extra GPU jobs are created. Source/config bindings
and official user quota are checked again inside the allocation. A failed
fixture, preparation or optimization stops this job; it is never retried here.
"""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from latentmol.boltz.workflow_process import run_logged
from latentmol.experiments import file_sha256, source_manifest, write_json


TESTS = (
    "tests/test_long_execution_controls.py", "tests/test_long_report_matching.py",
    "tests/test_boltz_defog_long_comparison.py", "tests/test_optimization_report.py",
    "tests/test_repair_weight_schedule.py", "tests/test_defog_matched_pilot.py",
    "tests/test_boltz_active_set_runtime.py", "tests/test_pointwise_contract.py",
    "tests/test_pointwise_ad.py", "tests/test_pointwise_cuts.py",
)


def quota_receipt(minimum_gib=4):
    """Use official per-user quota, never global filesystem free space."""
    reply = subprocess.run(["getquota_work3.sh"], capture_output=True, text=True,
        timeout=30, check=True)
    units = {"Byte": Decimal(1), "KiB": Decimal(1024), "MiB": Decimal(1024)**2,
        "GiB": Decimal(1024)**3, "TiB": Decimal(1024)**4}
    rows = []
    for line in reply.stdout.splitlines():
        if not line.strip().startswith("s234818|"):
            continue
        fields = line.split("||")[1].split("|")
        if len(fields) != 2:
            raise ValueError("Unrecognized official quota row")
        used, hard = [v.strip() for v in fields]
        if hard == "unlimited":
            continue
        def amount(value):
            count, unit = value.split()
            result = Decimal(count) * units[unit]
            if not result.is_finite() or result < 0:
                raise ValueError("Invalid official quota amount")
            return result
        remaining = (amount(hard) - amount(used)) / Decimal(1024)**3
        if remaining < minimum_gib:
            raise ValueError("Insufficient user quota for preserved long trajectories/reports")
        rows.append(str(remaining))
    if not rows:
        raise ValueError("Finite official user quota missing")
    return {"utc": datetime.now(timezone.utc).isoformat(), "minimum_gib": minimum_gib,
        "remaining_gib": rows, "stdout": reply.stdout, "stderr": reply.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an allocated immutable DTU GPU job")
    if file_sha256(args.bootstrap) != args.sha256:
        raise ValueError("Changed long-comparison bootstrap")
    protocol = json.loads(args.bootstrap.read_text())
    if (protocol["archive_sha256"] != os.environ["LATENTMOL_SOURCE_SHA256"]
            or protocol["manifest_sha256"] != source_manifest(Path.cwd())["sha256"]
            or protocol["allocation_seconds"] != 172800):
        raise ValueError("Changed long-comparison source/allocation binding")
    root = Path(protocol["control"])
    if not root.is_absolute() or not root.resolve().is_relative_to(Path("/work3/s234818/latentmol/submissions")):
        raise ValueError("Require controls under the declared user submissions root")
    record = root / "allocated-execution.json"
    started = time.monotonic()
    result = {"status": "running", "job_id": os.environ["LSB_JOBID"],
        "bootstrap_sha256": args.sha256, "source": protocol, "stages": {}}
    # Exclusive creation is the attempt claim. An existence check followed by
    # replacement would race with another allocation using the same bootstrap.
    with record.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())

    def execute(label, command, ceiling):
        # Keep an hour below LSF's hard kill for partial evidence/report flushing.
        left = int(protocol["allocation_seconds"] - 3600 - (time.monotonic() - started))
        if left < ceiling:
            raise TimeoutError("Insufficient complete stage reservation: " + label)
        execution = run_logged(command, root / (label + ".log"), timeout_seconds=ceiling)
        result["stages"][label] = execution
        write_json(record, result)
        if execution["timed_out"] or execution["returncode"] != 0:
            raise RuntimeError("Allocated stage failed; retained without retry: " + label)

    try:
        result["quota"] = quota_receipt()
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        execute("preflight", [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], 1800)
        runner = "scripts/run_boltz_defog_long_comparison.py"
        execute("prepare", [sys.executable, runner, "prepare",
            "--reference-plan", protocol["reference_plan"]["path"],
            "--reference-plan-sha256", protocol["reference_plan"]["sha256"],
            "--reference-pilot", protocol["reference_pilot"]["path"],
            "--reference-pilot-sha256", protocol["reference_pilot"]["sha256"],
            "--prepared-root", str(root / "prepared"), "--output-root", protocol["campaign"]], 900)
        plan = root / "prepared/plan.json"
        result["plan"] = {"path": str(plan), "sha256": file_sha256(plan)}
        execute("validate", [sys.executable, runner, "validate", "--plan", str(plan),
            "--sha256", result["plan"]["sha256"]], 300)
        result["launch_quota"] = quota_receipt()
        execute("comparison", [sys.executable, runner, "run", "--plan", str(plan),
            "--sha256", result["plan"]["sha256"]], 165600)
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_json(record, result)


if __name__ == "__main__":
    main()
