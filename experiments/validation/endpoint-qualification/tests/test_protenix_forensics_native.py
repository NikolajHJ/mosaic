"""Allocated metadata/startup controls; no pretrained inference or optimizer run."""
from copy import deepcopy
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_forensics_native.py"
SPEC = importlib.util.spec_from_file_location("protenix_forensics_native_fixture", PATH)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Forensics controls require allocated DTU HPC"


def valid_plan():
    endpoint = {"path": str(d.ENDPOINT), "bytes": 14284, "sha256": d.ENDPOINT_SHA}
    arm = {"id": "protenix", "rate": .3, "start_global": 20000, "target_global": 50000,
           "segment_updates": 10000,
           "source": {"path": str(d.SCIENCE), "manifest_sha256": d.SCIENCE_SHA, "files": 643},
           "config": {"path": str(d.CONFIG), "bytes": 1733, "sha256": d.CONFIG_SHA},
           "parent_snapshot": endpoint,
           "failed_parent": {"endpoint": deepcopy(endpoint), "job_id": "29809291"},
           "inputs": {"pocket": {"sha256": d.POCKET_SHA}, "reference": {"sha256": d.REFERENCE_SHA}},
           "expected": {"saved_parent_scientific_status": "failed"}}
    return {"version": d.VERSION, "max_bytes": d.MAX_BYTES, "internal_seconds": d.INTERNAL_SECONDS,
            "output": str(d.ROOT / "gpu/payload"), "cache": str(d.CACHE),
            "hook_source": {"path": str(d.HOOK), "manifest_sha256": d.HOOK_SHA, "files": 14},
            "original_plan": {"path": str(d.ORIGINAL_PLAN), "bytes": 46041, "sha256": d.ORIGINAL_PLAN_SHA},
            "failed_admission": {"path": str(d.FAILED), "bytes": 2531142, "sha256": d.FAILED_SHA},
            "context_ready": {"path": str(d.CONTEXT_READY), "bytes": 21266, "sha256": d.CONTEXT_READY_SHA},
            "arm": arm}


def test_exact_scope_metadata_has_no_execution():
    plan = valid_plan()
    assert d.validate_plan(plan, plan["output"]) == plan["arm"]


@pytest.mark.parametrize("change", ["version", "cap", "deadline", "output", "cache", "helper", "original",
                                   "failed", "context", "rate", "boundary", "source", "config", "snapshot",
                                   "producer", "pocket", "reference", "failure_status"])
def test_any_changed_protected_scope_fails_before_model(change):
    plan = valid_plan()
    if change == "version": plan["version"] = "continuation"
    if change == "cap": plan["max_bytes"] += 1
    if change == "deadline": plan["internal_seconds"] += 1
    if change == "output": plan["output"] = str(d.ENDPOINT.parent / "diagnostics")
    if change == "cache": plan["cache"] = "/different/native/assets"
    if change == "helper": plan["hook_source"]["files"] += 1
    if change == "original": plan["original_plan"]["sha256"] = "0" * 64
    if change == "failed": plan["failed_admission"]["bytes"] += 1
    if change == "context": plan["context_ready"]["path"] = str(d.CONFIG)
    if change == "rate": plan["arm"]["rate"] = .03
    if change == "boundary": plan["arm"]["start_global"] = 10000
    if change == "source": plan["arm"]["source"]["manifest_sha256"] = "0" * 64
    if change == "config": plan["arm"]["config"]["bytes"] += 1
    if change == "snapshot": plan["arm"]["parent_snapshot"]["sha256"] = "0" * 64
    if change == "producer": plan["arm"]["failed_parent"]["job_id"] = "replacement"
    if change == "pocket": plan["arm"]["inputs"]["pocket"]["sha256"] = "0" * 64
    if change == "reference": plan["arm"]["inputs"]["reference"]["sha256"] = "0" * 64
    if change == "failure_status": plan["arm"]["expected"]["saved_parent_scientific_status"] = "passed"
    with pytest.raises(ValueError):
        d.validate_plan(plan, plan["output"])


def test_output_argument_cannot_override_plan_scope():
    with pytest.raises(ValueError, match="output"):
        d.validate_plan(valid_plan(), d.ROOT / "another/payload")


@pytest.mark.parametrize("slots", [{"LSB_DJOB_NUMPROC": "4"}, {"LSB_MCPU_HOSTS": "node 4"},
                                  {"LSB_DJOB_NUMPROC": "4", "LSB_MCPU_HOSTS": "node 4"}])
def test_only_four_slot_native_allocation_is_accepted(slots):
    env = {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", **slots}
    assert d.allocated_environment(env) == env


@pytest.mark.parametrize("env", [{}, {"LSB_JOBID": "123", "LSB_QUEUE": "hpc", "LSB_DJOB_NUMPROC": "4"},
    {"LSB_JOBID": "bad", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "4"},
    {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s"},
    {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "8"},
    {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "four"},
    {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", "LSB_MCPU_HOSTS": "node"},
    {"LSB_JOBID": "123", "LSB_QUEUE": "gpul40s", "LSB_DJOB_NUMPROC": "4", "LSB_MCPU_HOSTS": "node 8"}])
def test_wrong_or_unknown_allocation_fails_closed(env):
    with pytest.raises(ValueError):
        d.allocated_environment(env)


def test_protected_input_is_hashed_and_not_written(tmp_path):
    path = tmp_path / "original.json"
    path.write_bytes(b'{"status":"failed"}\n')
    receipt = d.binding(path)
    assert d.bound(receipt) == path
    before = path.read_bytes()
    assert d.protected_bindings({"a": receipt, "nested": [receipt]}) == {str(path): receipt}
    assert path.read_bytes() == before
    path.write_bytes(before + b" ")
    with pytest.raises(ValueError, match="changed"):
        d.bound(receipt)


def test_source_manifest_rejects_unlisted_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "helper.py").write_text("# allocated-only helper\n")
    manifest = source / "source_manifest.json"
    manifest.write_text(json.dumps({"files": {"helper.py": d.sha(source / "helper.py")}}))
    assert d.verify_tree(source, d.sha(manifest), 1)["helper.py"] == d.sha(source / "helper.py")
    (source / "unexpected.py").write_text("# must not import\n")
    with pytest.raises(ValueError, match="Unexpected"):
        d.verify_tree(source, d.sha(manifest), 1)


def test_source_manifest_rejects_changed_member(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    member = source / "helper.py"
    member.write_text("# original\n")
    manifest = source / "source_manifest.json"
    manifest.write_text(json.dumps({"files": {"helper.py": d.sha(member)}}))
    member.write_text("# changed\n")
    with pytest.raises(ValueError, match="member changed"):
        d.verify_tree(source, d.sha(manifest), 1)


def test_persistent_budget_counts_nested_regular_files(tmp_path):
    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "child").mkdir()
    (tmp_path / "child/b").write_bytes(b"45")
    assert d.output_bytes(tmp_path) == 5


def test_timeout_restores_previous_signal_even_when_expired(monkeypatch):
    events = []
    original = object()
    def handler(_number, callback):
        events.append(("signal", callback))
        return original
    monkeypatch.setattr(d, "signal", SimpleNamespace(SIGALRM=1, ITIMER_REAL=2, signal=handler,
                       setitimer=lambda *args: events.append(("timer", args))))
    with pytest.raises(TimeoutError, match="deadline"):
        with d.deadline(5400):
            events[0][1](1, None)
    assert events[-2:] == [("timer", (2, 0)), ("signal", original)]


def test_all_preservation_proofs_attempted_after_protected_file_failure(tmp_path, monkeypatch):
    touched = []
    def failed_binding(path):
        raise ValueError("changed saved input")
    monkeypatch.setattr(d, "binding", failed_binding)
    monkeypatch.setattr(d, "verify_tree", lambda *args: touched.append("helper"))
    lifecycle = {"frozen": lambda: touched.append("frozen") or {"model": "same"},
                 "frozen_before": {"model": "same"}}
    hooks = SimpleNamespace(verify_source=lambda *args: touched.append("science"))
    result = d.final_proofs(tmp_path, {"/original": {}}, lifecycle, hooks)
    assert touched == ["frozen", "helper", "science"]
    assert result["frozen_unchanged"] is True
    assert result["source_trees_unchanged"] is True
    assert len(result["preservation_errors"]) == 1
    assert (tmp_path / "frozen-after.json").is_file()


def test_partial_model_failure_still_proves_loaded_model_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(d, "verify_tree", lambda *args: None)
    lifecycle = {"model_frozen": lambda: {"model": "same"}, "model_before": {"model": "same"}}
    result = d.final_proofs(tmp_path, None, lifecycle, None)
    assert result["frozen_model_unchanged"] is True
    assert result["frozen_unchanged"] == "full_context_not_available"
    assert result["protected_inputs_unchanged"] == "not_established_before_model"


def test_harness_has_no_production_or_optimizer_execution_route():
    tree = ast.parse(PATH.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in calls if isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not names & {"run_chemical_updates", "run_trajectory", "project_graph", "projected_step", "advance",
                        "check_graph_gradients", "check_protenix_raw_gradients", "sample", "Popen", "run_path"}
    direct_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("jax" in ast.unparse(node) or "numpy" in ast.unparse(node) or "mosaic" in ast.unparse(node)
                   for node in direct_imports)
    source = PATH.read_text()
    assert '"can_certify_or_restart_original_study": False' in source
    assert '"updates": 0' in source
