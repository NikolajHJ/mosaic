"""Allocated CPU protocol/process controls; actual CUDA smoke is a GPU gate.

These fixtures execute no JAX/model/optimizer code.  Real subprocesses exercise
the IPC co-residency/deadline/owned cleanup contract under a metadata protocol.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/cuda_ownership.py"
SPEC = importlib.util.spec_from_file_location("cuda_ownership_fixture", PATH)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
NONCE = "a" * 32


@pytest.fixture(autouse=True)
def allocated_only():
    assert str(os.environ.get("LSB_JOBID", "")).isdigit(), "CUDA ownership fixtures require allocated DTU HPC"


def environment():
    return {"LSB_JOBID": "123456", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "4", "CUDA_VISIBLE_DEVICES": "GPU-abc123"}


def message(role="production_fp32", sequence=0, pid=111):
    own = c.allocation(environment(), "123456")
    values = {"float32": {"dtype": "float32", "sum_squares": sum((n + sequence)**2 for n in (1, 2, 4)),
                         "gradient": [2 * (n + sequence) for n in (1, 2, 4)]}}
    if role == "reference_fp64":
        values["float64"] = {"dtype": "float64", "sum_squares": sum((n + sequence)**2 for n in (.5, 1.5, 2.5)),
                             "gradient": [2 * (n + sequence) for n in (.5, 1.5, 2.5)]}
    return {"version": c.VERSION, "status": "cuda_work_complete", "role": role, "nonce": NONCE, "pid": pid,
            "job_id": own["job_id"], "queue": own["queue"], "cuda_visible_devices": own["cuda_visible_devices"],
            "platform": "gpu", "device_kind": "NVIDIA L40S", "device_id": 0, "device_count": 1,
            "x64_enabled": role == "reference_fp64", "sequence": sequence, "measurements": values}


@pytest.mark.parametrize("key,value", [("LSB_JOBID", "999"), ("LSB_QUEUE", "hpc"), ("LSB_DJOB_NUMPROC", "16"),
    ("CUDA_VISIBLE_DEVICES", ""), ("CUDA_VISIBLE_DEVICES", "0,1"), ("CUDA_VISIBLE_DEVICES", "-1"),
    ("CUDA_VISIBLE_DEVICES", "GPU-abc 123"), ("CUDA_VISIBLE_DEVICES", "all")])
def test_allocation_refuses_wrong_job_queue_slots_or_visibility(key, value):
    current = environment(); current[key] = value
    with pytest.raises(ValueError):
        c.allocation(current, "123456")


@pytest.mark.parametrize("visible", ["0", "2", "GPU-abc-123"])
def test_one_explicit_gpu_identity_preserved(visible):
    current = environment(); current["CUDA_VISIBLE_DEVICES"] = visible
    assert c.allocation(current, "123456")["cuda_visible_devices"] == visible


def test_actual_worker_command_and_isolated_precision_environments():
    for role in c.ROLES:
        command = c.worker_command(role, NONCE)
        assert command == [sys.executable, "-B", "-u", str(PATH), "--worker", role, "--nonce", NONCE]
        original = environment()
        child = c.worker_environment(original, role)
        assert child["CUDA_VISIBLE_DEVICES"] == original["CUDA_VISIBLE_DEVICES"]
        assert child["JAX_ENABLE_X64"] == ("1" if role == "reference_fp64" else "0")
        assert child["JAX_PLATFORMS"] == "cuda,cpu" and child["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert "JAX_PLATFORMS" not in original
    tree = ast.parse(PATH.read_text(encoding="utf8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all("jax" not in ast.unparse(node) and "numpy" not in ast.unparse(node) for node in top_imports)
    assert "mps-control" not in PATH.read_text(encoding="utf8")


@pytest.mark.parametrize("role", c.ROLES)
def test_ready_and_both_actual_repeat_responses(role):
    own = c.allocation(environment(), "123456")
    first = message(role)
    assert c._validate_message(first, own, role, NONCE, 0) == first
    for sequence in (1, 2):
        assert c._validate_message(message(role, sequence), own, role, NONCE, sequence, first)["sequence"] == sequence


@pytest.mark.parametrize("field,value", [("pid", 0), ("pid", True), ("nonce", "b" * 32), ("job_id", "9"),
    ("queue", "hpc"), ("cuda_visible_devices", "GPU-other"), ("platform", "cpu"), ("device_kind", "NVIDIA A100"),
    ("device_count", 2), ("device_id", 1), ("x64_enabled", True), ("sequence", 2), ("status", "ready")])
def test_wrong_transport_or_cuda_precision_identity_rejected(field, value):
    row = message(); row[field] = value
    with pytest.raises(ValueError):
        c._validate_message(row, c.allocation(environment(), "123456"), "production_fp32", NONCE, 0)


def test_wrong_dtype_repeat_result_and_changed_context_pid_rejected():
    own = c.allocation(environment(), "123456")
    for mutate in ("dtype", "primal", "gradient", "extra"):
        row = message("reference_fp64")
        if mutate == "dtype": row["measurements"]["float64"]["dtype"] = "float32"
        elif mutate == "primal": row["measurements"]["float64"]["sum_squares"] = 0
        elif mutate == "gradient": row["measurements"]["float32"]["gradient"] = [0, 0, 0]
        else: row["other"] = True
        with pytest.raises(ValueError):
            c._validate_message(row, own, "reference_fp64", NONCE, 0)
    with pytest.raises(ValueError, match="process identity"):
        c._validate_message(message(sequence=1, pid=222), own, "production_fp32", NONCE, 1, message())


@pytest.fixture
def probe_inputs(tmp_path, monkeypatch):
    for key, value in environment().items(): monkeypatch.setenv(key, value)
    identities = {}
    for name in ("source_manifest", "plan", "context_identity"):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"name": name}) + "\n", encoding="utf8")
        identities[name] = c.binding(path)
    submission = tmp_path / "submission.json"
    row = {"status": "submitted", "returncode": 0, "job_id": "123456",
           "source_manifest_sha256": identities["source_manifest"]["sha256"],
           "argv": ["bsub", "-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                    "-gpu", c.GPU_REQUEST, "-W", "02:00", "bash", "/sealed/stage.lsf"]}
    submission.write_text(json.dumps(row) + "\n", encoding="utf8")
    return tmp_path, identities, submission, row


@pytest.mark.parametrize("change", ["job", "source", "mps", "slots", "duplicate", "host", "memory"])
def test_same_job_submission_requires_exact_mps_exclusive_gpu_resource(probe_inputs, change):
    root, identities, path, row = probe_inputs
    if change == "job": row["job_id"] = "123457"
    elif change == "source": row["source_manifest_sha256"] = "f" * 64
    elif change == "mps": row["argv"][row["argv"].index("-gpu") + 1] = "num=1:mode=exclusive_process"
    elif change == "slots": row["argv"][row["argv"].index("-n") + 1] = "16"
    elif change == "duplicate": row["argv"] += ["-gpu", c.GPU_REQUEST]
    elif change == "host": row["argv"].remove("span[hosts=1]")
    else: row["argv"].remove("rusage[mem=8GB]")
    path.write_text(json.dumps(row), encoding="utf8")
    with pytest.raises(ValueError): c._submission(path, c.allocation(environment(), "123456"), identities)


def test_source_context_proof_changes_and_output_reuse_fail_closed(probe_inputs):
    root, identities, path, row = probe_inputs
    assert c._bindings(identities) == identities
    Path(identities["context_identity"]["path"]).write_text("changed", encoding="utf8")
    with pytest.raises(ValueError, match="identity changed"): c._bindings(identities)
    with pytest.raises(ValueError): c._bindings({"source_manifest": identities["source_manifest"]})
    output = root / "existing"; output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        c.run_probe(output, submission=path, expected_job_id="123456", input_bindings=identities, deadline=time.monotonic() + 30)


FAKE_WORKER = r'''
import importlib.util,json,os,sys,time
spec=importlib.util.spec_from_file_location('fake_protocol',sys.argv[1]); c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
role,nonce,mode=sys.argv[2:5]
if mode=='second-fails' and role=='reference_fp64': sys.exit(7)
if mode=='hang' and role=='reference_fp64': time.sleep(60)
if mode=='malformed' and role=='reference_fp64': print('not-json',flush=True);time.sleep(60)
def emit(sequence):
 values={'float32':{'dtype':'float32','sum_squares':sum((n+sequence)**2 for n in (1,2,4)),'gradient':[2*(n+sequence) for n in (1,2,4)]}}
 if role=='reference_fp64':values['float64']={'dtype':'float64','sum_squares':sum((n+sequence)**2 for n in (.5,1.5,2.5)),'gradient':[2*(n+sequence) for n in (.5,1.5,2.5)]}
 print(json.dumps({'version':c.VERSION,'status':'cuda_work_complete','role':role,'nonce':nonce,'pid':os.getpid(),'job_id':os.environ['LSB_JOBID'],'queue':os.environ['LSB_QUEUE'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'platform':'gpu','device_kind':'NVIDIA L40S','device_id':0,'device_count':1,'x64_enabled':role=='reference_fp64','sequence':sequence,'measurements':values}),flush=True)
emit(0)
for sequence in (1,2):
 row=json.loads(sys.stdin.readline());assert row=={'command':'repeat','sequence':sequence,'nonce':nonce}
 if mode=='dies-on-repeat' and role=='production_fp32':sys.exit(8)
 emit(sequence)
assert json.loads(sys.stdin.readline())=={'command':'stop','nonce':nonce}
'''


def fake_transport(monkeypatch, mode):
    real_command = c.worker_command
    monkeypatch.setattr(c, "worker_command", lambda role, nonce: [sys.executable, "-B", "-u", "-c", FAKE_WORKER, str(PATH), role, nonce, mode])
    return real_command


def test_real_two_child_ipc_holds_first_context_and_repeats_before_success(probe_inputs, monkeypatch):
    root, identities, path, row = probe_inputs
    fake_transport(monkeypatch, "success")
    receipt = c.run_probe(root / "smoke", submission=path, expected_job_id="123456", input_bindings=identities,
                          deadline=time.monotonic() + 40)
    assert receipt["status"] == "passed_two_client_cuda_ownership" and receipt["optimizer_updates"] == 0
    assert receipt["native_or_gradient_gate_replaced"] is False and receipt["mps_daemon_or_compute_mode_changed"] is False
    assert receipt["input_bindings"] == receipt["input_bindings_after"] == identities
    assert len(receipt["cleanup"]) == 2 and all(r["reaped"] and not r["forced_termination"] for r in receipt["cleanup"])
    assert receipt["records"]["production_fp32"][0]["pid"] != receipt["records"]["reference_fp64"][0]["pid"]
    assert all([r["sequence"] for r in receipt["records"][role]] == [0, 1, 2] for role in c.ROLES)


@pytest.mark.parametrize("mode", ["second-fails", "hang", "malformed", "dies-on-repeat"])
def test_worker_failure_timeout_malformed_or_lost_context_preserves_failure_and_reaps_only_owned(probe_inputs, monkeypatch, mode):
    root, identities, path, row = probe_inputs
    fake_transport(monkeypatch, mode)
    monkeypatch.setattr(c, "MAX_SECONDS", 2 if mode == "hang" else 10)
    output = root / "failed"
    with pytest.raises(ValueError, match="ownership admission failed"):
        c.run_probe(output, submission=path, expected_job_id="123456", input_bindings=identities, deadline=time.monotonic() + 35)
    result = json.loads((output / "assessment.json").read_text())
    assert result["status"] == "failed" and result["error"] and result["input_bindings_after"] == identities
    assert len(result["cleanup"]) == 2 and all(r["reaped"] for r in result["cleanup"])
    assert result["native_or_gradient_gate_replaced"] is False and result["scientific_results_changed"] is False


def test_output_budget_and_shared_deadline_are_mandatory(probe_inputs):
    root, identities, path, row = probe_inputs
    with pytest.raises(ValueError, match="runtime remains"):
        c.run_probe(root / "no-time", submission=path, expected_job_id="123456", input_bindings=identities, deadline=time.monotonic())
    with pytest.raises(ValueError, match="cleanup runtime"):
        c.run_probe(root / "no-cleanup", submission=path, expected_job_id="123456", input_bindings=identities, deadline=time.monotonic() + 5)
    output = root / "oversize"; output.mkdir()
    with (output / "large.log").open("wb") as stream: stream.truncate(c.MAX_BYTES + 1)
    with pytest.raises(ValueError, match="output budget"): c._budget(output)
