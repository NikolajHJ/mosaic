"""HPC CPU metadata fixtures; actual CUDA concurrency requires allocated probes."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from latentmol.gpu_process_probe import assess_pair, assess_processes, protocol_worker
from latentmol.experiments import file_sha256, write_json


def pair():
    worker = "/work3/user/env/defog311/bin/python"
    parent = {"status": "passed", "pid": 1, "device_count": 1, "tensor_sum": 16,
        "retained_sum_after_child": 16, "device_uuid": "gpu-uuid", "device_name": "V100",
        "environment": {"LSB_JOBID": "123", "CUDA_VISIBLE_DEVICES": "0", "CUDA_MPS_PIPE_DIRECTORY": "/job-pipe"}}
    child = {**deepcopy(parent), "pid": 2, "python": worker, "prefix": "/work3/user/env/defog311"}
    return parent, child, worker


def test_same_gpu_two_contexts_and_exact_isolated_interpreter():
    assert assess_pair(*pair()) == "matched_gpu_uuid"


@pytest.mark.parametrize("mutation", ["child_failed", "same_pid", "two_gpus", "tensor", "parent_lost_context",
                                     "environment", "interpreter", "prefix", "uuid", "gpu_name"])
def test_concurrency_gate_rejects_invalid_receipts(mutation):
    parent, child, worker = pair()
    if mutation == "child_failed": child["status"] = "error"
    elif mutation == "same_pid": child["pid"] = parent["pid"]
    elif mutation == "two_gpus": child["device_count"] = 2
    elif mutation == "tensor": child["tensor_sum"] = 0
    elif mutation == "parent_lost_context": parent["retained_sum_after_child"] = 0
    elif mutation == "environment": child["environment"]["CUDA_VISIBLE_DEVICES"] = "1"
    elif mutation == "interpreter": child["python"] = "/usr/bin/python"
    elif mutation == "prefix": child["prefix"] = "/usr"
    elif mutation == "uuid": child["device_uuid"] = "other-uuid"
    else: child["device_name"] = "other-gpu"
    with pytest.raises((ValueError, RuntimeError)): assess_pair(parent, child, worker)


def test_missing_uuid_discloses_weaker_visibility_identity():
    parent, child, worker = pair()
    child["device_uuid"] = None
    assert assess_pair(parent, child, worker) == "one_device_inherited_visibility_only"


def test_failed_child_still_requires_correct_interpreter_and_allocation():
    parent, child, worker = pair()
    child["status"] = "error"
    assess_processes(parent, child, worker)
    child["prefix"] = "/usr"
    with pytest.raises(ValueError, match="isolated environment"):
        assess_processes(parent, child, worker)


def test_probe_reads_authenticated_actual_gate_provider(tmp_path):
    config = tmp_path / "case.yaml"
    config.write_text("repair_guidance:\n  provider:\n    device: cuda\n    python: /work3/user/env/defog311/bin/python\n")
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {"gate": {"config": {"path": str(config), "sha256": file_sha256(config)}}})
    assert protocol_worker(protocol, file_sha256(protocol)) == "/work3/user/env/defog311/bin/python"
    config.write_text(config.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="config changed"): protocol_worker(protocol, file_sha256(protocol))


def submission_module():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/submit_ramped_random.py"
    spec = importlib.util.spec_from_file_location("ramped_submit_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try: spec.loader.exec_module(module)
    finally: sys.path.pop(0)
    return module


def test_submission_keeps_gpu_exclusive_and_enables_managed_mps():
    module = submission_module()
    assert module.gpu_request(managed_mps=True) == "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes"
    assert module.gpu_request(managed_mps=False) == "num=1:mode=exclusive_process:mps=no:j_exclusive=yes"


@pytest.mark.parametrize("change", ["none", "identity", "receipt", "job_id", "scheduler"])
def test_native_requires_authenticated_done_gate(tmp_path, monkeypatch, change):
    module = submission_module()
    identity = {"release": "immutable-release"}
    intent = {"identity": identity}
    receipt = {"intent": deepcopy(intent), "job_id": "123"}
    if change == "identity": intent["identity"] = {"release": "other"}
    elif change == "receipt": receipt["intent"] = {}
    elif change == "job_id": receipt["job_id"] = "unknown"
    write_json(tmp_path / "mps-probe-intent.json", intent)
    write_json(tmp_path / "mps-probe-receipt.json", receipt)
    def scheduler(command, **kwargs):
        assert command == ["bjobs", "-a", "-noheader", "-o", "stat", "123"]
        assert kwargs["check"] and kwargs["timeout"] == 30
        return module.subprocess.CompletedProcess(command, 0, "RUN\n" if change == "scheduler" else "DONE\n", "")
    monkeypatch.setattr(module.subprocess, "run", scheduler)
    if change == "none":
        assert module.require_completed_stage(tmp_path, "mps-probe", identity) == "123"
    else:
        with pytest.raises(ValueError): module.require_completed_stage(tmp_path, "mps-probe", identity)
