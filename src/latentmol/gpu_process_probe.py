"""Allocated two-process CUDA startup gate for isolated frozen-model workers.

Hold a parent CUDA tensor while the exact configured worker interpreter makes
its own CUDA allocation. This checks launch compatibility only: no molecular
model, repaired target or gradient is validated here. LSF owns MPS setup and
GPU selection; this module never starts a daemon or changes device visibility.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from latentmol.experiments import file_sha256, write_json

ENV_KEYS = ("LSB_JOBID", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER",
            "CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY")


def process_identity():
    """Retain interpreter/allocation evidence even if the first CUDA call fails."""
    return {"pid": os.getpid(), "python": sys.executable, "prefix": sys.prefix,
            "environment": {key: os.environ.get(key) for key in ENV_KEYS}}


def cuda_operation():
    """A retained real context, using no pretrained model or mutable weights."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one visible allocated CUDA device")
    value = torch.ones(16, device="cuda:0")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    metadata = {**process_identity(), "status": "passed",
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count(), "device_name": properties.name,
        "device_uuid": str(properties.uuid) if hasattr(properties, "uuid") else None,
        "tensor_sum": float(value.sum())}
    return value, metadata


def assess_processes(parent, child, worker_python):
    """Authenticate even a failed child's interpreter and inherited allocation."""
    if not child.get("pid") or parent.get("pid") == child["pid"]:
        raise ValueError("Require a distinct isolated worker process")
    if not parent.get("environment", {}).get("LSB_JOBID") or parent["environment"] != child.get("environment"):
        raise ValueError("Worker allocation/device/MPS environment differs")
    # Do not resolve the interpreter symlink: its spelling selects the venv.
    expected = Path(worker_python)
    if child.get("python") != str(expected) or child.get("prefix") != str(expected.parent.parent):
        raise ValueError("Worker did not use the exact configured isolated environment")


def assess_pair(parent, child, worker_python):
    """Fail closed on wrong allocation/interpreter, context or device identity.

    MPS can remap numeric GPU IDs. Preserve the inherited environment rather
    than interpreting those IDs; compare actual UUIDs when both runtimes expose
    them. Missing UUIDs are disclosed as visibility-only identity evidence.
    """
    if parent.get("status") != "passed" or child.get("status") != "passed":
        raise RuntimeError("Parent or isolated worker CUDA startup failed")
    if (parent.get("pid") == child.get("pid") or parent.get("device_count") != 1
            or child.get("device_count") != 1 or parent.get("tensor_sum") != 16
            or child.get("tensor_sum") != 16 or parent.get("retained_sum_after_child") != 16):
        raise ValueError("Both distinct processes must retain a working CUDA context")
    assess_processes(parent, child, worker_python)
    if parent.get("device_name") != child.get("device_name"):
        raise ValueError("Worker uses a different GPU model")
    left, right = parent.get("device_uuid"), child.get("device_uuid")
    if left is not None and right is not None and left != right:
        raise ValueError("Worker uses a different physical GPU")
    return "matched_gpu_uuid" if left is not None and right is not None else "one_device_inherited_visibility_only"


def protocol_worker(protocol_path, expected_sha):
    """Read authenticated prepared inputs without constructing any model."""
    import yaml
    if file_sha256(protocol_path) != expected_sha:
        raise ValueError("GPU gate protocol identity differs")
    protocol = json.loads(protocol_path.read_text())
    ref = protocol["gate"]["config"]
    path = Path(ref["path"])
    if file_sha256(path) != ref["sha256"]:
        raise ValueError("GPU gate's scientific config changed")
    provider = yaml.safe_load(path.read_text())["repair_guidance"]["provider"]
    if provider["device"] != "cuda":
        raise ValueError("This launch gate requires the declared CUDA worker")
    return provider["python"]


def run_probe(worker_python, output, *, negative_control=False, timeout_seconds=120):
    """Record actual concurrency, including expected or unexpected old-policy outcomes."""
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("CUDA process probe requires an allocated DTU job")
    if not Path(worker_python).is_absolute() or not Path(worker_python).is_file():
        raise ValueError("Require the configured existing isolated worker interpreter")
    if output.exists():
        raise FileExistsError("GPU startup receipts are immutable")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "status": "running", "worker_python": worker_python,
        "negative_control": negative_control, "timeout_seconds": timeout_seconds,
        "scope": "Concurrent CUDA startup only; not native repair/gradient validation"}
    write_json(output, report)
    started = time.monotonic()
    retained = None
    try:
        retained, parent = cuda_operation()
        report["parent"] = parent
        inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,compute_mode",
                                    "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        report["compute_mode_snapshot"] = {"returncode": inventory.returncode,
                                          "stdout": inventory.stdout, "stderr": inventory.stderr}
        command = [worker_python, "-m", "latentmol.gpu_process_probe", "--child"]
        report["child_command"] = command
        completed = subprocess.run(command, env=dict(os.environ), capture_output=True,
                                   text=True, timeout=timeout_seconds)
        report["child_execution"] = {"returncode": completed.returncode,
            "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]}
        child = json.loads(completed.stdout.splitlines()[-1])
        report["child"] = child
        parent["retained_sum_after_child"] = float(retained.sum())
        assess_processes(parent, child, worker_python)
        unavailable = (completed.returncode != 0 and child.get("status") == "error"
            and child.get("stage") == "cuda_operation"
            and "CUDA-capable device(s) is/are busy or unavailable" in child.get("detail", ""))
        if unavailable and negative_control:
            report.update(status="observed_child_cuda_unavailable", exclusivity_failure_reproduced=True)
        else:
            if completed.returncode != 0:
                raise RuntimeError("Isolated worker CUDA process failed; see retained receipt")
            report["device_identity_scope"] = assess_pair(parent, child, worker_python)
            report.update(status="passed", exclusivity_failure_reproduced=False)
            if negative_control:
                report["interpretation"] = "Unexpected old-policy child success; exclusivity cause not confirmed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        write_json(output, report)
        del retained
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--worker-python")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("GPU process check requires an allocated DTU job")
    if args.child:
        try:
            retained, value = cuda_operation()
        except Exception as error:
            print(json.dumps({**process_identity(), "status": "error", "stage": "cuda_operation",
                              "detail": str(error), "error_type": type(error).__name__}))
            raise SystemExit(1)
        print(json.dumps(value))
        return
    if args.output is None or bool(args.worker_python) == bool(args.protocol):
        parser.error("Declare output and exactly one of worker-python or authenticated protocol")
    worker = protocol_worker(args.protocol, args.protocol_sha256) if args.protocol else args.worker_python
    result = run_probe(worker, args.output, negative_control=args.negative_control)
    print(json.dumps({"status": result["status"], "receipt": str(args.output)}))


if __name__ == "__main__":
    main()
