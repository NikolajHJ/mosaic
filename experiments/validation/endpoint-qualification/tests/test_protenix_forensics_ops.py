"""Allocated metadata contracts for diagnostic-only Protenix orchestration."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_forensics_ops.py"
SPEC = importlib.util.spec_from_file_location("protenix_forensics_ops_tests", PATH)
ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ops
SPEC.loader.exec_module(ops)
PIN = "a" * 64


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Forensic contracts execute on allocated DTU HPC only"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf8")
    return ops.binding(path)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / "release/source"
    root = tmp_path / "run"
    source.mkdir(parents=True)
    root.mkdir()
    monkeypatch.setattr(ops, "SOURCE", source)
    monkeypatch.setattr(ops, "ROOT", root)
    return source, root


def seal(source, members=None):
    members = members or {"scripts/driver.py": "# unchanged\n"}
    files = {}
    for name, data in members.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf8")
        files[name] = ops.sha(path)
    save(source / "source_manifest.json", {"version": ops.VERSION, "files": files})
    return ops.sha(source / "source_manifest.json")


@pytest.fixture
def plan(owned, monkeypatch):
    source, root = owned
    arm = {"id": "protenix", "rate": 0.3, "global_start": 20000, "target_global": 50000,
           "source": {"path": "/unchanged/scientific/source", "manifest_sha256": PIN, "files": 643},
           "config": {"path": "/unchanged/config", "bytes": 1733, "sha256": PIN},
           "failed_parent": {"job_id": "29809291", "endpoint": {"path": "/unchanged/snapshot", "bytes": 14284, "sha256": PIN}},
           "inputs": {"original_driver_argv": ["python", "original.py", "--lr", "0.3"]}}
    original = save(root / "original-plan.json", {"arms": [arm]})
    failure = save(root / "original-admission.json", {"status": "failed", "can_admit_new_child": False,
                   "original_run_failed_preserved": True, "can_certify_original_study": False})
    monkeypatch.setattr(ops, "ORIGINAL_PLAN", {"path": original["path"], "sha256": original["sha256"]})
    monkeypatch.setattr(ops, "FAILED_ADMISSION", failure)
    return {"version": ops.PLAN_VERSION, "source_root": str(source), "control_root": str(root), "arm": arm,
            "original_plan": original, "failed_admission": failure, "output": str(root / "gpu/payload"),
            "max_bytes": ops.OUTPUT_CAP, "cap_bytes": ops.OUTPUT_CAP, "metadata_cap_bytes": ops.METADATA_CAP,
            "internal_seconds": 5400, "maximum_unique_graphs": 109, "original_results_changed": False,
            "optimization_settings_changed": False, "can_certify_or_restart_original_study": False,
            "monitor_remains_paused": True}


def good_lsf(job="111", name="mosaic-pro-forensics-cpu", queue="hpc"):
    return ("Subject: Job " + job + ": <" + name + "> in cluster <dcc> Done\n"
            "Job was executed in queue <" + queue + ">.\nSuccessfully completed.\n")


def test_authentic_plan_cannot_authorize_an_optimizer(plan):
    assert ops.validate_plan(plan) == plan
    assert ops.PHASES == ("cpu", "after-cpu", "gpu")
    assert not plan["can_certify_or_restart_original_study"]


@pytest.mark.parametrize("change", ["rate", "target", "snapshot", "argv", "source", "output", "cap", "seconds", "graphs", "certify", "settings", "results", "monitor"])
def test_plan_rejects_settings_checkpoint_or_scope_changes(plan, change):
    value = copy.deepcopy(plan)
    if change == "rate":
        value["arm"]["rate"] = 0.03
    elif change == "target":
        value["arm"]["target_global"] = 60000
    elif change == "snapshot":
        value["arm"]["failed_parent"]["endpoint"]["sha256"] = "b" * 64
    elif change == "argv":
        value["arm"]["inputs"]["original_driver_argv"].append("--optimizer")
    elif change == "source":
        value["source_root"] += "/other"
    elif change == "output":
        value["output"] = "/original/saved/study"
    elif change == "cap":
        value["cap_bytes"] += 1
    elif change == "seconds":
        value["internal_seconds"] = 7200
    elif change == "graphs":
        value["maximum_unique_graphs"] = 110
    elif change == "certify":
        value["can_certify_or_restart_original_study"] = True
    elif change == "settings":
        value["optimization_settings_changed"] = True
    elif change == "results":
        value["original_results_changed"] = True
    else:
        value["monitor_remains_paused"] = False
    with pytest.raises(ValueError):
        ops.validate_plan(value)


def test_original_admission_change_is_detected_even_if_plan_is_unchanged(plan):
    Path(plan["failed_admission"]["path"]).write_text('{"status":"passed"}\n')
    with pytest.raises(ValueError, match="Proof bytes changed"):
        ops.validate_plan(plan)


def test_source_manifest_rejects_changed_and_extra_files(owned):
    source, _ = owned
    pin = seal(source)
    assert ops.verify(source, pin)["files"]
    extra = source / "unexpected.py"
    extra.write_text("# extra\n")
    with pytest.raises(ValueError, match="scope changed"):
        ops.verify(source, pin)
    extra.unlink()
    (source / "scripts/driver.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="changed/linked"):
        ops.verify(source, pin)


@pytest.mark.parametrize("name", ["../escape.py", "/absolute.py", "scripts\\escape.py"])
def test_source_inventory_rejects_unsafe_paths(owned, name):
    source, _ = owned
    save(source / "source_manifest.json", {"files": {name: PIN}})
    with pytest.raises(ValueError, match="Unsafe source member"):
        ops.verify(source, ops.sha(source / "source_manifest.json"))


def test_binding_rejects_linked_original_proof(owned):
    _, root = owned
    proof = root / "proof.json"
    proof.write_text("{}\n")
    link = root / "alias.json"
    link.symlink_to(proof)
    with pytest.raises(ValueError, match="absolute regular file"):
        ops.binding(link)


@pytest.mark.parametrize("phase", ["cpu", "after-cpu", "gpu"])
def test_declared_resource_vectors_and_actual_wrapper_arguments(owned, phase):
    dependency = "ended(111)" if phase == "after-cpu" else None
    command = ops.argv(phase, PIN, dependency)
    assert command[-4:] == [str(ops.SOURCE / "scripts/protenix_forensics_stage.lsf"), str(ops.SOURCE), PIN, phase]
    assert command[command.index("-q") + 1] == ("gpul40s" if phase == "gpu" else "hpc")
    assert command[command.index("-n") + 1] == ("1" if phase == "after-cpu" else "4")
    assert command[command.index("-W") + 1] == {"cpu": "01:00", "after-cpu": "00:20", "gpu": "02:00"}[phase]
    assert ("num=1:mode=exclusive_process" in command) == (phase == "gpu")
    assert ("-w" in command) == (phase == "after-cpu")


@pytest.mark.parametrize("phase,dependency", [("after-cpu", None), ("after-cpu", "done(111)"), ("gpu", "ended(111)"), ("cpu", "ended(111)"), ("segment", None), ("validate", None)])
def test_unauthorized_phases_or_dependencies_are_not_submittable(owned, phase, dependency):
    with pytest.raises(ValueError):
        ops.argv(phase, PIN, dependency)


@pytest.mark.parametrize("phase", ["cpu", "after-cpu", "gpu"])
def test_own_job_authenticates_actual_submission_and_allocation(owned, monkeypatch, phase):
    _, root = owned
    dependency = "ended(222)" if phase == "after-cpu" else None
    save(root / phase / "submission.json", {"status": "submitted", "job_id": "111", "phase": phase,
         "source_manifest_sha256": PIN, "argv": ops.argv(phase, PIN, dependency), "dependency": dependency})
    if dependency:
        save(root / "cpu/submission.json", {"status": "submitted", "job_id": "222"})
    monkeypatch.setenv("LSB_JOBID", "111")
    monkeypatch.setenv("LSB_QUEUE", "gpul40s" if phase == "gpu" else "hpc")
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "1" if phase == "after-cpu" else "4")
    assert ops.own(phase, PIN)["job_id"] == "111"
    monkeypatch.setenv("LSB_QUEUE", "other")
    with pytest.raises(ValueError, match="queue/slots"):
        ops.own(phase, PIN)


def test_unknown_submission_keeps_markers_and_cannot_be_replayed(owned, monkeypatch):
    _, root = owned
    calls = []
    def unknown(command, **kwargs):
        calls.append(command)
        raise subprocess_timeout()
    monkeypatch.setattr(ops.subprocess, "run", unknown)
    with pytest.raises(ValueError, match="Unknown submission"):
        ops.submit("gpu", PIN)
    assert ops.read(root / "gpu/submission-started.json")["status"] == "unknown"
    assert ops.read(root / "gpu/submission.json")["status"] == "unknown"
    with pytest.raises(FileExistsError):
        ops.submit("gpu", PIN)
    assert len(calls) == 1


def subprocess_timeout():
    return TimeoutError("unknown remote submission outcome")


def test_lsf_accepts_wrapped_identity_without_changing_raw_bytes():
    raw = good_lsf().replace("forensics-cpu", "forensics-\n    cpu").replace("queue <hpc>", "queue <\n hpc>")
    assert ops.lsf_success(raw, "111", "mosaic-pro-forensics-cpu", "hpc")


@pytest.mark.parametrize("change", ["job", "name", "queue", "duplicate", "exit", "missing_success"])
def test_lsf_rejects_wrong_or_ambiguous_terminal_evidence(change):
    raw = good_lsf()
    if change == "job":
        raw = good_lsf(job="222")
    elif change == "name":
        raw = good_lsf(name="mosaic-endpoint02-cpu")
    elif change == "queue":
        raw = good_lsf(queue="gpul40s")
    elif change == "duplicate":
        raw += raw
    elif change == "exit":
        raw += "Exited with exit code 1.\n"
    else:
        raw = raw.replace("Successfully completed.", "")
    with pytest.raises(ValueError, match="LSF success"):
        ops.lsf_success(raw, "111", "mosaic-pro-forensics-cpu", "hpc")


def make_xml(path, child="", duplicate=False, omitted=False):
    modules = sorted(ops.MODULES)
    rows = [(name, "case", child if index == 0 else "") for index, name in enumerate(modules)]
    if duplicate:
        rows.append(rows[0])
    if omitted:
        rows.pop()
    path.write_text('<testsuite tests="999">' + ''.join('<testcase classname="' + module + '" name="' + name + '">' + extra + '</testcase>' for module, name, extra in rows) + '</testsuite>')


def test_raw_cpu_rows_override_untrusted_xml_header(owned):
    _, root = owned
    path = root / "tests.xml"
    make_xml(path)
    assert ops.xml_cases(path, dict.fromkeys(ops.MODULES, 1)) == 3


@pytest.mark.parametrize("child", ["<failure/>", "<error/>", "<skipped/>"])
def test_cpu_raw_failure_error_or_skip_blocks_certificate(owned, child):
    _, root = owned
    path = root / "tests.xml"
    make_xml(path, child=child)
    with pytest.raises(ValueError, match="failed/errored/skipped"):
        ops.xml_cases(path, dict.fromkeys(ops.MODULES, 1))


@pytest.mark.parametrize("change", ["duplicate", "omitted", "wrong_count", "bool_count", "wrong_module"])
def test_exact_cpu_module_attribution_is_mandatory(owned, change):
    _, root = owned
    path = root / "tests.xml"
    make_xml(path, duplicate=change == "duplicate", omitted=change == "omitted")
    counts = dict.fromkeys(ops.MODULES, 1)
    if change == "wrong_count":
        counts[next(iter(counts))] = 2
    elif change == "bool_count":
        counts[next(iter(counts))] = True
    elif change == "wrong_module":
        counts["unrelated_test"] = counts.pop(next(iter(counts)))
    with pytest.raises(ValueError):
        ops.xml_cases(path, counts)


@pytest.fixture
def parent_gate(monkeypatch):
    value = {"status": "passed", "required_bytes": 20 * ops.GIB, "available_bytes": 30 * ops.GIB,
             "reserve_bytes": 4 * ops.GIB, "file_reserve": 100000, "original_gate_changed": False}
    monkeypatch.setattr(ops, "parent_ops", lambda: SimpleNamespace(storage=lambda parent, phase: copy.deepcopy(value)))
    return value


def test_storage_adds_full_new_caps_and_keeps_parent_reserves(owned, parent_gate):
    _, root = owned
    parent = root / "preflight"
    parent.mkdir()
    value = ops.storage(parent, "preflight")
    assert value["status"] == "passed"
    assert value["required_bytes"] == parent_gate["required_bytes"] + ops.OUTPUT_CAP + ops.METADATA_CAP
    assert value["reserve_bytes"] == 4 * ops.GIB and value["file_reserve"] == 100000
    assert ops.read(parent / "forensics-storage-preflight.json") == value


@pytest.mark.parametrize("change", ["failed_parent", "bytes", "reserve", "inodes", "waiver", "output_cap", "metadata_cap"])
def test_storage_failure_preserves_its_receipt_without_discounting_caps(owned, parent_gate, monkeypatch, change):
    _, root = owned
    parent = root / "entry"
    parent.mkdir()
    if change == "failed_parent":
        parent_gate["status"] = "failed"
    elif change == "bytes":
        parent_gate["available_bytes"] = parent_gate["required_bytes"] + ops.OUTPUT_CAP + ops.METADATA_CAP - 1
    elif change == "reserve":
        parent_gate["reserve_bytes"] -= 1
    elif change == "inodes":
        parent_gate["file_reserve"] -= 1
    elif change == "waiver":
        parent_gate["original_gate_changed"] = True
    elif change == "output_cap":
        monkeypatch.setattr(ops, "OUTPUT_CAP", 32)
        output = root / "gpu/payload"
        output.mkdir(parents=True)
        (output / "raw.bin").write_bytes(b"x" * 33)
    else:
        monkeypatch.setattr(ops, "METADATA_CAP", 16)
        (root / "metadata.bin").write_bytes(b"x" * 17)
    with pytest.raises(ValueError):
        ops.storage(parent)
    assert ops.read(parent / "forensics-storage-entry.json")["status"] == "failed"


def test_owned_metadata_rejects_symlink_even_if_target_is_inside_root(owned):
    _, root = owned
    target = root / "raw.bin"
    target.write_bytes(b"raw")
    (root / "alias.bin").symlink_to(target)
    with pytest.raises(ValueError, match="linked/special"):
        ops.owned_bytes(root)


def test_native_command_is_diagnostic_only_and_plan_bound(owned, plan):
    source, root = owned
    save(source / "plan.json", plan)
    command = ops.native_argv(plan)
    assert command == [ops.PYTHON, str(source / "scripts/protenix_forensics_native.py"), "--plan", str(source / "plan.json"),
                       "--plan-sha256", ops.sha(source / "plan.json"), "--output", str(root / "gpu/payload")]
    assert not any(value in command for value in ("--resume", "--lr", "--updates", "--segment"))
