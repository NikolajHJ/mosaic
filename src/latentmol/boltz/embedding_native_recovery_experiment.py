"""Allocated fixture/authentication gate and CUDA-free native recovery adapter.

The CPU pytest process exits before any recovery coordinator runs. The GPU path
authenticates the completed CPU stage, then executes the recovery CLI in a new
process. Neither adapter stage initializes a CUDA context in its coordinator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from latentmol.cuda_process import require_cuda_free_coordinator


FIXTURES = ["tests/test_embedding_comparison_rescore.py", "tests/test_embedding_native_recovery.py",
            "tests/test_cuda_process.py", "tests/test_embedding_comparison_experiment.py",
            "tests/test_embedding_native_recovery_adapter.py"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    result = dict(tests=len(cases), failures=sum(c.find("failure") is not None for c in cases),
                  errors=sum(c.find("error") is not None for c in cases),
                  skipped=sum(c.find("skipped") is not None for c in cases))
    if not cases or any(result[key] for key in ("failures", "errors", "skipped")):
        raise ValueError("Native recovery requires complete allocated fixture coverage without skips")
    return result


def require_cpu_gate(cpu, source_sha, plan_sha):
    """Bind both fixture evidence and the actual saved-parent CLI validation."""
    cpu = Path(cpu)
    record = json.loads((cpu / "result.json").read_text())
    validation = cpu / "validation/result.json"
    if (record.get("status") != "passed" or record.get("source_sha256") != source_sha
            or record.get("plan_sha256") != plan_sha
            or record.get("junit_sha256") != digest(cpu / "cpu-tests.xml")
            or record.get("validation_sha256") != digest(validation)):
        raise ValueError("GPU recovery requires this exact completed CPU gate")
    observed = json.loads(validation.read_text())
    if (observed.get("status") != "passed_authentication_not_prediction"
            or observed.get("source_sha256") != source_sha or observed.get("plan_sha256") != plan_sha):
        raise ValueError("GPU recovery parent-validation identity differs")
    junit(cpu / "cpu-tests.xml")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "gpu"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Native recovery adapter requires a sealed LSF allocation")
    require_cuda_free_coordinator()
    plan_sha = digest(args.plan)
    source_sha = os.environ["LATENTMOL_SOURCE_SHA256"]
    command = [sys.executable, "-m", "latentmol.boltz.embedding_native_recovery", "--plan", str(args.plan),
               "--plan-sha256", plan_sha, "--output", str(args.output), "--mode", "run"]
    if args.stage == "gpu":
        cpu = args.output.parent / "cpu"
        require_cpu_gate(cpu, source_sha, plan_sha)
        subprocess.run(command, check=True)
        require_cuda_free_coordinator()
        return
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    provenance = dict(source_sha256=source_sha, plan_sha256=plan_sha, stage="cpu",
                      job_id=os.environ["LSB_JOBID"], coordinator=require_cuda_free_coordinator())
    environment = dict(os.environ, MKL_CBWR="COMPATIBLE", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       OPENBLAS_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    xml = args.output / "cpu-tests.xml"
    tests = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml", str(xml), *FIXTURES]
    try:
        with (args.output / "cpu-tests.log").open("x", encoding="utf-8") as log:
            subprocess.run(tests, check=True, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=1200)
        coverage = junit(xml)
        validation = args.output / "validation"
        command[command.index("--output") + 1] = str(validation)
        command[command.index("--mode") + 1] = "validate"
        subprocess.run(command, check=True, env=environment, timeout=600)
        observed = json.loads((validation / "result.json").read_text())
        if (observed.get("status") != "passed_authentication_not_prediction"
                or observed.get("plan_sha256") != plan_sha or observed.get("source_sha256") != source_sha):
            raise ValueError("Exact saved-parent CPU authentication did not pass")
        require_cuda_free_coordinator()
        write(args.output / "result.json", dict(**provenance, status="passed", junit=coverage,
            junit_sha256=digest(xml), validation_sha256=digest(validation / "result.json"),
            fixture_command=tests, elapsed_seconds=time.monotonic() - started,
            finished_at=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        write(args.output / "result.json", dict(**provenance, status="failed_not_passed",
            error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == "__main__":
    main()
