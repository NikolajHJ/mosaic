"""Allocated CPU adapter controls; these never claim pretrained validation."""
from copy import deepcopy
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/continuation_driver.py"
SPEC = importlib.util.spec_from_file_location("prospective_driver_fixture", PATH)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Adapter controls require allocated DTU HPC"


def plan_for(tmp_path, name):
    rel, count, pin, driver, rate, steps, target = d.SOURCES[name]
    source = str(d.BASE / rel)
    start = {"protenix": 20000, "boltz": 10000, "dbmol": 5000}[name]
    argv = ["/original/python", str(Path(source) / driver), "--mode", "run", "--config", source + "/config.json",
            "--output", "/old/results", "--manifest", pin, "--offset", "0", "--segment-steps", str(steps),
            "--resume-snapshot", "/old/parent.npz", "--resume-sha256", "a" * 64,
            "--cpu-certificate", "/old/cpu.json", "--cpu-certificate-sha256", "b" * 64,
            "--ready", "/old/ready.json", "--ready-sha256", "c" * 64, "--reference", "/original/reference.json"]
    arm = {"id": name, "source": {"path": source, "files": count, "manifest_sha256": pin},
           "config": {"path": source + "/config.json", "bytes": 100, "sha256": "d" * 64},
           "start_global": start, "target_global": target, "segment_updates": steps,
           "original_argv": argv, "inputs": {"original_driver_argv": argv[:]},
           "original_directions": {"X": [{"row": 1, "source": 0, "target": 1}], "E_upper": []}}
    return {"root": str(tmp_path / "new"), "arms": [arm]}, arm


def inputs(tmp_path):
    return ({"path": str(tmp_path / "parent.npz"), "bytes": 17, "sha256": "e" * 64},
            {"path": str(tmp_path / "fresh.json"), "bytes": 18, "sha256": "f" * 64})


@pytest.mark.parametrize("name", ["protenix", "boltz", "dbmol"])
@pytest.mark.parametrize("phase", ["validate", "segment"])
def test_exact_generated_argv_preserves_scientific_fields(tmp_path, name, phase):
    plan, arm = plan_for(tmp_path, name)
    snapshot, ready = inputs(tmp_path)
    output = Path(plan["root"]) / "producer/payload/study"
    argv = d.build_original_argv(plan, name, phase, arm["start_global"], output, snapshot,
                                 ready if phase == "segment" else None)
    assert argv[:2] == arm["original_argv"][:2]
    for key in ("--manifest", "--config", "--cpu-certificate", "--cpu-certificate-sha256", "--reference"):
        assert d._option(argv, key) == d._option(arm["original_argv"], key)
    assert d._option(argv, "--mode") == ("run" if phase == "segment" else "validate")
    assert d._option(argv, "--offset") == str(arm["start_global"])
    assert d._option(argv, "--segment-steps") == str(arm["segment_updates"])
    assert d._option(argv, "--resume-sha256") == snapshot["sha256"]
    assert d._option(argv, "--ready", optional=True) == (ready["path"] if phase == "segment" else None)
    assert "/old/results" not in argv and "/old/parent.npz" not in argv


@pytest.mark.parametrize("failure", ["offset", "outside", "rate_scope", "duplicate", "argv_alias", "missing_ready"])
def test_undeclared_generated_argv_fails_closed(tmp_path, failure):
    plan, arm = plan_for(tmp_path, "boltz")
    snapshot, ready = inputs(tmp_path)
    offset = arm["start_global"]
    output = Path(plan["root"]) / "producer/payload/study"
    if failure == "offset": offset += 1
    if failure == "outside": output = tmp_path / "old/payload/study"
    if failure == "rate_scope": arm["segment_updates"] = 9999
    if failure == "duplicate":
        arm["original_argv"] += ["--offset", "0"]
        arm["inputs"]["original_driver_argv"] = arm["original_argv"][:]
    if failure == "argv_alias": arm["inputs"]["original_driver_argv"][0] = "/different/python"
    if failure == "missing_ready": ready = None
    with pytest.raises(ValueError):
        d.build_original_argv(plan, "boltz", "segment", offset, output, snapshot, ready)


@pytest.mark.parametrize("name", ["protenix", "boltz", "dbmol"])
def test_cpu_startup_scope_is_exact(tmp_path, name):
    plan, arm = plan_for(tmp_path, name)
    snapshot, ready = inputs(tmp_path)
    output = Path(plan["root"]) / "cpu/payload" / ("startup-" + name + "-segment") / "study"
    assert d.build_original_argv(plan, name, "segment", arm["start_global"], output, snapshot, ready)
    with pytest.raises(ValueError):
        d.build_original_argv(plan, name, "segment", arm["start_global"], output.with_name("wrong"), snapshot, ready)


STARTUP = '''
import argparse
events = []
def verify_readiness(value):
    events.append('readiness')
    if value != 'fresh': raise ValueError('not fresh')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('validate','run'), required=True)
    parser.add_argument('--ready')
    args = parser.parse_args()
    events.append('parsed')
    if args.mode == 'run':
        verify_readiness(args.ready)
    import jax
    events.append('numerical body')
'''


@pytest.mark.parametrize("defer", [False, True])
def test_real_parser_sentinel_preserves_or_explicitly_defers_readiness(monkeypatch, defer):
    namespace = {"PremodelBoundaryReached": d.PremodelBoundaryReached}
    exec(compile(d.startup_tree(STARTUP, "original.py", defer_ready=defer), "original.py", "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["original.py", "--mode", "run", "--ready", "old" if defer else "fresh"])
    with pytest.raises(d.PremodelBoundaryReached): namespace["main"]()
    assert namespace["events"] == (["parsed"] if defer else ["parsed", "readiness"])


def test_full_startup_rejects_bad_actual_ready_before_model(monkeypatch):
    namespace = {"PremodelBoundaryReached": d.PremodelBoundaryReached}
    exec(compile(d.startup_tree(STARTUP, "original.py"), "original.py", "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["original.py", "--mode", "run", "--ready", "old"])
    with pytest.raises(ValueError, match="not fresh"): namespace["main"]()
    assert namespace["events"] == ["parsed", "readiness"]


def test_actual_parser_rejects_unknown_argument(monkeypatch):
    namespace = {"PremodelBoundaryReached": d.PremodelBoundaryReached}
    exec(compile(d.startup_tree(STARTUP, "original.py"), "original.py", "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["original.py", "--mode", "validate", "--unknown"])
    with pytest.raises(SystemExit): namespace["main"]()
    assert namespace["events"] == []


def test_startup_boundary_rejects_unrecognized_driver_shape():
    with pytest.raises(ValueError): d.startup_tree("def main():\n    return 0\n", "different.py")


def original_checker(value):
    report = {"parity": value["parity"]}
    if not value["parity"]: raise ValueError("parity failed")
    for field in ("X", "E_upper"):
        raise AssertionError("legacy coarse failure")
    report["nulls"] = value["nulls"]
    if not value["nulls"]: raise ValueError("null failed")
    report["typed"] = True
    return report


def original_db_checker(value):
    x, e, gx, ge = 1, 2, 3, 4
    report = {"parity": True}
    for field, values, gradient in (("X", x, gx), ("E_upper", e, ge)):
        raise AssertionError("legacy raw")
    if not value: raise ValueError("typed contrast failed")
    report["typed_bond_contrast_nonzero"] = True
    return report


def unrecognized_checker(value):
    report = {}
    for category in ("a", "b"): report[category] = value
    return report


def test_raw_loop_replacement_retains_original_parity_nulls_and_tail():
    replacement, legacy = d.replace_raw_loop(original_checker, lambda context: {"new_endpoint": True}, kind="boltz")
    assert replacement({"parity": True, "nulls": True}) == {"parity": True, "new_endpoint": True, "nulls": True, "typed": True}
    with pytest.raises(ValueError, match="parity failed"): replacement({"parity": False, "nulls": True})
    with pytest.raises(ValueError, match="null failed"): replacement({"parity": True, "nulls": False})
    with pytest.raises(AssertionError, match="legacy coarse failure"):
        exec(legacy, original_checker.__globals__, {"report": {}})


def test_dbmol_raw_loop_replacement_preserves_typed_nonnull_guard():
    replacement, legacy = d.replace_raw_loop(original_db_checker, lambda context: {"new_endpoint": True}, kind="dbmol")
    assert replacement(True)["typed_bond_contrast_nonzero"] is True
    with pytest.raises(ValueError, match="typed contrast failed"): replacement(False)
    with pytest.raises(AssertionError, match="legacy raw"):
        exec(legacy, original_db_checker.__globals__, dict(report={}, x=1, e=2, gx=3, ge=4))


def test_raw_loop_shape_change_fails_closed():
    with pytest.raises(ValueError, match="raw loop shape changed"):
        d.replace_raw_loop(unrecognized_checker, lambda context: {}, kind="boltz")


def verify_native_report(report):
    raise AssertionError("old raw verifier must be replaced explicitly")


def original_readiness(record):
    if not (record["status"] == "passed" and record["parity"] is True
            and record["all_components_and_total_raw_fd_passed"] is True and record["runtime"] is True):
        raise ValueError("original nonraw scope failed")
    verify_native_report(record)
    return record


def test_explicit_readiness_adapter_keeps_nonraw_predicates(monkeypatch):
    monkeypatch.setattr(d, "verify_dbmol_native_report", lambda record: d.require(record["new_endpoint"] is True, "new endpoint failed"))
    adapted = d.adapt_dbmol_readiness(original_readiness)
    good = dict(status="passed", parity=True, runtime=True, new_endpoint=True,
                all_components_and_total_raw_fd_passed=False)
    assert adapted(good) is good
    for key in ("parity", "runtime", "new_endpoint"):
        bad = dict(good); bad[key] = False
        with pytest.raises(ValueError): adapted(bad)
    assert good["all_components_and_total_raw_fd_passed"] is False


def test_bindings_reject_changed_bytes_and_extra_metadata(tmp_path):
    path = tmp_path / "proof.json"; path.write_text('{"status":"passed"}\n')
    record = d.binding(path)
    assert d.bound(record) == path
    bad = dict(record, other=True)
    with pytest.raises(ValueError): d.bound(bad)
    path.write_text('{"status":"failed"}\n')
    with pytest.raises(ValueError): d.bound(record)


def test_exclusive_writes_preserve_original_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    d.write(path, {"original_failure": True})
    with pytest.raises(FileExistsError): d.write(path, {"original_failure": False})
    assert d.read(path) == {"original_failure": True}


def test_execution_context_restores_process_on_error(tmp_path):
    before = (sys.path[:], sys.argv[:], Path.cwd())
    with pytest.raises(RuntimeError):
        with d.execution_context(tmp_path, ["python", "original.py", "--mode", "validate"]):
            assert sys.argv == ["original.py", "--mode", "validate"]
            assert Path.cwd() == tmp_path
            raise RuntimeError("disposable")
    assert (sys.path, sys.argv, Path.cwd()) == before


def test_hook_restores_original_checker_on_failure(monkeypatch):
    original = lambda *args: None
    module = SimpleNamespace(check_affinity_gradients=original)
    monkeypatch.setattr(d.importlib, "import_module", lambda name: module)
    adapted = lambda *args: True
    state = SimpleNamespace(name="boltz", simple=lambda function, kind: adapted)
    with pytest.raises(RuntimeError):
        with d.install_hooks(state):
            assert module.check_affinity_gradients is adapted
            raise RuntimeError("failed admission")
    assert module.check_affinity_gradients is original


def test_boundary_replay_requires_exact_all_five_saved_arrays(tmp_path):
    import numpy as np
    arrays = {"X": np.full((2, 2), .5, np.float32), "E_upper": np.full((1, 2), .5, np.float32),
              "atom_logits": np.zeros((2, 2), np.float32), "bond_logits": np.zeros((1, 2), np.float32),
              "key_data": np.asarray([0, 17], np.uint32)}
    first, second = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez_compressed(first, **arrays); np.savez_compressed(second, **arrays)
    assert d._boundary(first, second) is True
    arrays["key_data"] = np.asarray([0, 18], np.uint32)
    np.savez_compressed(second, **arrays)
    with pytest.raises(ValueError, match="key_data"): d._boundary(first, second)


@pytest.mark.parametrize("name", ["boltz", "dbmol"])
def test_frozen_state_excludes_volatile_transport_but_preserves_raw_reply(tmp_path, monkeypatch, name):
    keys = ("model_digest", "affinity_model_digest", "structural_model_digest", "context_digest",
            "full_context_digest", "context_file_sha256", "checkpoint_sha256", "structural_checkpoint_sha256")
    identity = {key: str(index) * 64 for index, key in enumerate(keys)}
    replies = []
    def request(action):
        assert action == "verify_identity"
        row = dict(identity, status="passed", frozen=True, id=len(replies), elapsed_seconds=.1 + len(replies))
        replies.append(row)
        return row
    def frozen(after, before):
        assert after["frozen"] is True and all(after[key] == before[key] for key in keys)
    def verify(after, model, context):
        assert after["status"] == "passed" and after["model_digest"] == model and after["context_digest"] == context
    monkeypatch.setitem(sys.modules, "dbmol_protocol", SimpleNamespace(frozen=frozen))
    monkeypatch.setitem(sys.modules, "campaign25_contract", SimpleNamespace(verify_frozen_response=verify))
    state = d.EndpointHooks({}, name, tmp_path / "study", 0)
    callback = state._frozen(None, SimpleNamespace(identity=identity, request=request), tmp_path)
    assert callback() == callback()
    first, second = d.read(tmp_path / "frozen-0000.json"), d.read(tmp_path / "frozen-0001.json")
    assert first["id"] != second["id"] and first["elapsed_seconds"] != second["elapsed_seconds"]


@pytest.mark.parametrize("scope,known,expected", [
    ("saved_failed_endpoint", True, "saved_failed_endpoint"),
    ("saved_failed_endpoint", False, "original_probe_not_reached"),
    ("not_preexisting_at_new_endpoint", False, "not_preexisting_at_new_endpoint"),
])
def test_authentic_unreached_and_future_scopes_stay_distinct(tmp_path, monkeypatch, scope, known, expected):
    def plan(values, original, *, original_direction_scope):
        return {"values": values, "original": original, "scope": original_direction_scope}
    monkeypatch.setitem(sys.modules, "admission", SimpleNamespace(plan=plan))
    state = d.EndpointHooks({}, "dbmol", tmp_path, 5000)
    original = {"X": [{"row": 0, "source": 0, "target": 1}] if known else [], "E_upper": []}
    point = {"values": {"X": "raw", "E_upper": "raw"}, "original": {"component": original}, "scope": scope}
    result = state._new_plan(point, "component")
    assert result["scope"] == expected and result["original"] == original


def test_raw_points_preserve_original_per_leaf_devices_without_default_placement():
    values = {"X": object(), "E_upper": object()}
    template = SimpleNamespace(X=SimpleNamespace(device="original-atoms-CPU"),
                               E_upper=SimpleNamespace(device="original-bonds-GPU"))
    calls = []
    def device_put(value, device):
        calls.append((value, device))
        return SimpleNamespace(value=value, device=device)
    point = d.restore_point_placement(values, template, lambda x, e: SimpleNamespace(X=x, E_upper=e), device_put)
    assert calls == [(values["X"], "original-atoms-CPU"), (values["E_upper"], "original-bonds-GPU")]
    assert point.X.value is values["X"] and point.E_upper.value is values["E_upper"]
