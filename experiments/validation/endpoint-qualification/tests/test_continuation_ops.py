"""Allocated-only stdlib controller fixtures; no native scientific evaluations."""
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/continuation_ops.py"
SPEC = importlib.util.spec_from_file_location("endpoint_continuation_ops_tests", PATH)
ops = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ops
SPEC.loader.exec_module(ops)
PIN = "a" * 64


@pytest.fixture(autouse=True)
def allocated_only(monkeypatch):
    assert os.environ.get("LSB_JOBID"), "Controller contracts execute only on allocated HPC"
    # Isolate original controller unit budgets. Separate capacity fixtures and
    # actual allocated startup invoke the complete authenticated new guard.
    monkeypatch.setattr(ops, 'current_capacity_gate', lambda *args: {
        'status':'passed','required_bytes':0,'available_bytes':10**30})


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / "release/source"
    root = tmp_path / "run"
    source.mkdir(parents=True)
    root.mkdir()
    monkeypatch.setattr(ops, "SOURCE", source)
    monkeypatch.setattr(ops, "ROOT", root)
    return source, root


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf8")
    return ops.binding(path)


def plan():
    return {"version": ops.PLAN_VERSION, "admission_policy": "prospective-endpoint-admission-v2",
            "user_authorized_prospective_checks_before_children": True, "original_failures_preserved": True,
            "optimization_settings_changed": False, "monitor_remains_paused": True,
            "matrix_stride": 20, "loss_stride": 1, "dispatch_arms": list(ops.DISPATCH_ARMS),
            "blocked_arms": copy.deepcopy(ops.BLOCKED_ARMS), "root":str(ops.ROOT), "control_root": str(ops.ROOT), "source_root": str(ops.SOURCE), "arms": [
        {"id": name, "offsets": list(values["offsets"]), "segment_steps": values["segment_steps"],
         "global_target": values["global_target"], "rate": values["rate"],
         "source": {"path": str(ops.BASE / ops.ORIGINALS[name]["release"]), "manifest_sha256": ops.ORIGINALS[name]["manifest"], "files": ops.ORIGINALS[name]["files"]},
         "config": {"path": str(ops.BASE / ops.ORIGINALS[name]["release"] / ops.ORIGINALS[name]["config"]), "bytes": ops.ORIGINALS[name]["config_bytes"], "sha256": ops.ORIGINALS[name]["config_sha256"]},
         "failed_parent": {"job_id": ops.ORIGINALS[name]["job_id"], "endpoint": {"path": "/snapshot", "bytes": 1, "sha256": PIN}},
         "inputs": {"original_driver_argv": [ops.PYTHON, "/original/driver.py", "--mode", "run"]}}
        for name, values in ops.ARMS.items()]}


def scheduler(status="DONE", job="111", queue="hpc"):
    return {"job_id": job, "returncode": 0, "stdout": "Job <" + job + ">, Status <" + status + ">, Queue <" + queue + ">", "stderr": ""}


def test_scientific_root_alias_cannot_point_to_predecessor():
    value=plan();value['root']=str(ops.ROOT)+'-predecessor'
    with pytest.raises(ValueError,match='scientific/control/source'):ops.validate_plan(value)


def lsf(job="111", name="mosaic-endpoint06-cpu", queue="hpc", status="Done"):
    return ("Subject: Job " + job + ": <" + name + "> in cluster <dcc> " + status
            + "\nJob was executed in queue <" + queue + ">.\nSuccessfully completed.\n")


def xml(path, rows):
    path.write_text("<testsuite>" + "".join('<testcase classname="' + module + '" name="' + name + '">' + child + "</testcase>"
                                          for module, name, child in rows) + "</testsuite>", encoding="utf8")


def sealed(source, members=None):
    members = members or {"driver.py": "# unchanged\n"}
    files = {}
    for name, data in members.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf8")
        files[name] = ops.sha(path)
    save(source / "source_manifest.json", {"version": ops.VERSION, "files": files})
    return ops.sha(source / "source_manifest.json")


def valid_result(owned, arm="boltz", phase="validate", offset=10000):
    source, root = owned
    current_plan = plan()
    save(source / "plan.json", current_plan)
    save(source / "source_manifest.json", {"files": {"plan.json": ops.sha(source / "plan.json")}})
    snapshot = save(root / "original-snapshot.json", {"saved": "exact"})
    row = next(item for item in current_plan["arms"] if item["id"] == arm)
    row["failed_parent"]["endpoint"] = snapshot
    save(source / "plan.json", current_plan)
    original = save(root / "raw-native-proof.json", {"status": "passed", "can_admit_new_child": True})
    result = {"version": ops.RESULT_VERSION, "status": "passed", "job_id": "111", "arm": arm, "phase": phase, "offset": offset,
              "global_end": offset + (ops.ARMS[arm]["segment_steps"] if phase == "segment" else 0),
              "source_manifest_sha256": PIN, "plan_sha256": ops.sha(source / "plan.json"),
              "config_sha256": row["config"]["sha256"], "original_results_changed": False,
              "frozen_unchanged": True, "checks": dict.fromkeys(ops.CHECKS, True), "proofs": [original],
              "original_failed_parent_preserved": True, "endpoint_policy": "prospective-endpoint-admission-v2",
              "scientific_source_manifest_sha256": row["source"]["manifest_sha256"],
              "applicability": {"pilot_high": "not_applicable_chemistry_disabled",
                                "terminal_admission": "not_applicable_validation_phase" if phase == "validate" else "required",
                                "case_jsonl": "not_applicable_validation_phase" if phase == "validate" else "required",
                                "fp32_boundary": "required", "null_controls": "typed_bonds_required_nonnull" if arm == "boltz" else "original_null_controls"},
              "input_snapshot": snapshot, "projections": {"maximum_pilot_update_bytes": 2000}}
    if phase == "segment":
        result["terminal_snapshot"] = save(root / "new-endpoint.json", {"new": "exact"})
    path = ops.parent_path(phase, arm, offset) / "payload/assessment.json"
    save(path.parent.parent / "submission.json", {"status": "submitted", "job_id": "111"})
    save(path.parent.parent / "post-execution-preservation.json", {"preservation_status": "passed", "source_manifest_sha256": PIN,
                                                                 "original_results_changed": False})
    save(path, result)
    return current_plan, result, path


@pytest.mark.parametrize("arm", ["boltz", "dbmol"])
def test_exact_authorized_ranges_and_own_labels(arm):
    value = ops.arm_plan(plan(), arm)
    for offset in value["offsets"]:
        for phase in ("validate", "segment", "after-validate", "after-segment"):
            assert ops.label(phase, arm, offset) == arm + "-" + phase + "-" + str(offset).zfill(5)


@pytest.mark.parametrize("change", ["missing_arm", "duplicate_arm", "wrong_offset", "wrong_steps", "wrong_target", "missing_argv", "enable_protenix", "missing_block", "changed_block_proof"])
def test_plan_rejects_scope_changes(change):
    value = plan()
    if change == "missing_arm":
        value["arms"].pop()
    elif change == "duplicate_arm":
        value["arms"][2] = copy.deepcopy(value["arms"][0])
    elif change == "wrong_offset":
        value["arms"][0]["offsets"] = [10000, 20000, 30000]
    elif change == "wrong_steps":
        value["arms"][0]["segment_steps"] = 9999
    elif change == "wrong_target":
        value["arms"][0]["global_target"] = 60000
    elif change == "missing_argv":
        value["arms"][0]["inputs"].pop("original_driver_argv")
    elif change == "enable_protenix":
        value["dispatch_arms"].append("protenix")
    elif change == "missing_block":
        value["blocked_arms"] = {}
    else:
        value["blocked_arms"]["protenix"]["source01_admission"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        ops.validate_plan(value)


@pytest.mark.parametrize("phase", ["cpu", "after-cpu", "validate", "after-validate", "segment", "after-segment"])
def test_actual_resource_vector_and_wrapper_arguments(phase):
    callback = phase.startswith("after-")
    arm, offset = ("all", 0) if phase in ("cpu", "after-cpu") else ("dbmol", 5000)
    command = ops.argv(phase, PIN, arm, offset, "ended(111)" if callback else None)
    assert command[-5:] == [str(ops.SOURCE), PIN, phase, arm, str(offset)]
    assert command[command.index("-n") + 1] == ("4" if phase in ("cpu", "validate", "segment") else "1")
    assert command[command.index("-q") + 1] == ("gpul40s" if phase in ("validate", "segment") else "hpc")
    assert ("num=1:mode=exclusive_process" in command) == (phase in ("validate", "segment"))
    assert ("-w" in command) == callback


@pytest.mark.parametrize("phase,dependency", [("validate", "done(111)"), ("segment", "ended(222)"), ("after-validate", None), ("after-segment", "done(111)")])
def test_dependencies_never_couple_independent_arms(phase, dependency):
    with pytest.raises(ValueError):
        ops.argv(phase, PIN, "boltz", 10000, dependency)


def test_source_authenticates_all_members_and_rejects_extra_or_changed_bytes(owned):
    source, _ = owned
    pin = sealed(source)
    assert ops.verify(source, pin)["files"] == {"driver.py": ops.sha(source / "driver.py")}
    (source / "unsealed.py").write_text("extra")
    with pytest.raises(ValueError, match="scope"):
        ops.verify(source, pin)
    (source / "unsealed.py").unlink()
    (source / "driver.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        ops.verify(source, pin)


def test_source_rejects_linked_member(owned):
    source, root = owned
    target = root / "real.py"
    target.write_text("preserved")
    (source / "linked.py").symlink_to(target)
    save(source / "source_manifest.json", {"files": {"linked.py": ops.sha(target)}})
    with pytest.raises(ValueError, match="changed"):
        ops.verify(source, ops.sha(source / "source_manifest.json"))


def test_binding_requires_exact_bytes_length_and_schema(owned):
    _, root = owned
    row = save(root / "proof.json", {"native": "preserved"})
    assert ops.check_binding(row) == row
    with pytest.raises(ValueError):
        ops.check_binding({**row, "bytes": row["bytes"] + 1})
    with pytest.raises(ValueError):
        ops.check_binding({**row, "unbound": "ignored"})


@pytest.mark.parametrize("state", ["RUN", "PEND", "DONE"])
def test_actual_scheduler_states_are_observed_without_invention(state):
    assert ops.scheduler_status(scheduler(state), "111", "hpc") == state


@pytest.mark.parametrize("change", ["wrong_job", "wrong_queue", "duplicate_job", "exit", "nonzero", "bool_returncode"])
def test_scheduler_failures_identity_and_query_errors_fail_closed(change):
    row = scheduler()
    if change == "wrong_job":
        row["stdout"] = row["stdout"].replace("111", "222")
    elif change == "wrong_queue":
        row["stdout"] = row["stdout"].replace("hpc", "gpul40s")
    elif change == "duplicate_job":
        row["stdout"] += " Job <111>"
    elif change == "exit":
        row["stdout"] = row["stdout"].replace("DONE", "EXIT")
    elif change == "nonzero":
        row["returncode"] = 255
    else:
        row["returncode"] = False
    with pytest.raises(ValueError):
        ops.scheduler_status(row, "111", "hpc")


def test_lsf_wraps_whitespace_without_losing_exact_job_name():
    raw = lsf().replace("mosaic-endpoint06-cpu", "mosaic-endpoint06-\n cpu")
    assert ops.lsf_success(raw, "111", "mosaic-endpoint06-cpu", "hpc")


@pytest.mark.parametrize("change", ["wrong_job", "wrong_name", "wrong_queue", "duplicate_subject", "fake_done_suffix", "no_success"])
def test_lsf_failure_does_not_become_a_success(change):
    raw = lsf()
    if change == "wrong_job":
        raw = raw.replace("111", "112")
    elif change == "wrong_name":
        raw = raw.replace("mosaic-", "wrong-mosaic-", 1)
    elif change == "wrong_queue":
        raw = raw.replace("hpc", "gpul40s")
    elif change == "duplicate_subject":
        raw += raw
    elif change == "fake_done_suffix":
        raw = raw.replace("Done\n", "DoneInvalid\n")
    else:
        raw = raw.replace("Successfully completed.", "Exited with exit code 1.")
    assert raw != lsf(), "Negative control must actually mutate the raw LSF proof"
    with pytest.raises(ValueError):
        ops.lsf_success(raw, "111", "mosaic-endpoint06-cpu", "hpc")


def test_raw_xml_counts_actual_rows_and_exact_modules(owned):
    _, root = owned
    path = root / "tests.xml"
    xml(path, [("tests.test_ops", "a", ""), ("tests.test_admission", "b", "")])
    assert ops.xml_cases(path, {"tests.test_ops": 1, "tests.test_admission": 1}) == 2
    with pytest.raises(ValueError, match="attribution"):
        ops.xml_cases(path, {"test_ops": 1, "test_admission": 1})


@pytest.mark.parametrize("bad", ["<failure/>", "<error/>", "<skipped/>", "duplicate"])
def test_raw_xml_failure_error_skip_and_duplicate_block_cpu_certificate(owned, bad):
    _, root = owned
    path = root / "tests.xml"
    rows = [("test_ops", "same", "" if bad == "duplicate" else bad)]
    if bad == "duplicate":
        rows *= 2
    xml(path, rows)
    with pytest.raises(ValueError):
        ops.xml_cases(path, {"test_ops": len(rows)})


def test_new_sparse_projection_adds_only_positive_excess_without_reducing_old_budget():
    old = {"selected": {"boltz30k": {"remaining": 4 * ops.GIB}}}
    forecast = {"boltz": {"unit_bytes": 2000, "remaining_offsets": [10000, 20000]}}
    amount, details = ops.projection_excess(old, forecast)
    assert amount == 0
    assert details["boltz"]["original_unfunded_projection"] == 4 * ops.GIB
    forecast["boltz"]["unit_bytes"] = 10000000
    amount, details = ops.projection_excess(old, forecast)
    assert amount > 0
    assert amount == details["boltz"]["new_full_remaining_projection"] - 4 * ops.GIB
    assert old["selected"]["boltz30k"]["remaining"] == 4 * ops.GIB


@pytest.mark.parametrize("offsets", [[20000, 10000], [9999], [10000], []])
def test_projection_suffix_cannot_skip_reorder_or_invent_remaining_segments(offsets):
    with pytest.raises(ValueError):
        ops.projection_excess({"selected": {"boltz30k": {"remaining": 1}}}, {"boltz": {"unit_bytes": 10, "remaining_offsets": offsets}})


def test_unknown_bsub_outcome_keeps_exclusive_markers_and_prevents_retry(owned, monkeypatch):
    _, root = owned
    def fail(*args, **kwargs):
        raise TimeoutError("unknown actual bsub outcome")
    monkeypatch.setattr(ops.subprocess, "run", fail)
    with pytest.raises(ValueError, match="Unknown"):
        ops.submit("cpu", PIN)
    assert ops.read(root / "cpu/submission.json")["status"] == "unknown"
    assert (root / "cpu/submission-started.json").is_file()
    with pytest.raises(FileExistsError):
        ops.submit("cpu", PIN)


def test_own_pair_uses_only_its_exact_ended_dependency(monkeypatch):
    calls = []
    def submit(phase, pin, arm, offset, dependency=None):
        calls.append((phase, arm, offset, dependency))
        return {"job_id": str(111 + len(calls))}
    monkeypatch.setattr(ops, "submit", submit)
    result = ops._pair("validate", PIN, "dbmol", 5000)
    assert calls == [("validate", "dbmol", 5000, None), ("after-validate", "dbmol", 5000, "ended(112)")]
    assert result == {"producer": "112", "callback": "113"}


@pytest.mark.parametrize("change", ["scientific_failed", "wrong_pin", "wrong_config", "frozen_changed", "old_mutated", "missing_check", "false_check", "wrong_boundary", "wrong_input"])
def test_result_authentication_requires_every_native_runtime_and_boundary_gate(owned, change):
    current_plan, result, path = valid_result(owned)
    if change == "scientific_failed":
        result["status"] = "failed"
    elif change == "wrong_pin":
        result["source_manifest_sha256"] = "b" * 64
    elif change == "wrong_config":
        result["config_sha256"] = "b" * 64
    elif change == "frozen_changed":
        result["frozen_unchanged"] = False
    elif change == "old_mutated":
        result["original_results_changed"] = True
    elif change == "missing_check":
        result["checks"].pop("null_controls")
    elif change == "false_check":
        result["checks"]["terminal_admission"] = False
    elif change == "wrong_boundary":
        result["global_end"] += 1
    else:
        result["input_snapshot"] = {**result["input_snapshot"], "sha256": "b" * 64}
    save(path, result)
    with pytest.raises(ValueError):
        ops.verify_result(current_plan, PIN, "validate", "boltz", 10000)


def test_validated_result_is_file_bound_and_original_failure_stays_preserved(owned):
    current_plan, result, _ = valid_result(owned)
    assert ops.verify_result(current_plan, PIN, "validate", "boltz", 10000) == result
    proof = Path(result["proofs"][0]["path"])
    proof.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        ops.verify_result(current_plan, PIN, "validate", "boltz", 10000)


def test_new_child_boundary_cannot_use_original_failed_summary_as_certificate(owned):
    source, root = owned
    current_plan = plan()
    save(source / "plan.json", current_plan)
    save(source / "source_manifest.json", {"files": {"plan.json": ops.sha(source / "plan.json")}})
    save(root / (ops.label("segment", "boltz", 10000) + "-certificate.json"), {"status": "failed", "arm": "boltz", "global_end": 20000})
    with pytest.raises(ValueError, match="not certified"):
        ops.input_snapshot(current_plan, "boltz", 20000)


def test_producer_cannot_be_certified_from_scheduler_done_without_native_assessment(owned, monkeypatch):
    current_plan, result, path = valid_result(owned)
    result["checks"]["native_components"] = False
    save(path, result)
    monkeypatch.setattr(ops, "terminal", lambda *args: [])
    with pytest.raises(ValueError):
        ops.certify_producer(current_plan, PIN, "validate", "boltz", 10000)
    assert not (owned[1] / (ops.label("validate", "boltz", 10000) + "-certificate.json")).exists()


def test_terminal_observation_preserves_run_then_done_raw_queries(owned, monkeypatch):
    _, root = owned
    parent = root / "cpu"
    save(parent / "submission.json", {"job_id": "111"})
    (parent / "stdout.111.log").write_text(lsf())
    responses = iter([scheduler("RUN"), scheduler("DONE")])
    def run(*args, **kwargs):
        row = next(responses)
        return SimpleNamespace(returncode=row["returncode"], stdout=row["stdout"], stderr=row["stderr"])
    monkeypatch.setattr(ops.subprocess, "run", run)
    monkeypatch.setattr(ops.time, "sleep", lambda _: None)
    proofs = ops.terminal(parent, "hpc")
    assert len(proofs) == 4
    assert "Status <RUN>" in ops.read(parent / "terminal-query-000.json")["stdout"]
    assert "Status <DONE>" in ops.read(parent / "terminal-query-001.json")["stdout"]


def test_nonfinite_json_metadata_is_rejected(owned):
    path = owned[1] / "nonfinite.json"
    path.write_text('{"projection": NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        ops.read(path)


def test_control_allowance_counts_validation_source_and_logs_but_not_projected_segment_study(owned):
    source, root = owned
    (source / "module.py").write_bytes(b"source")
    validation = ops.parent_path("validate", "boltz", 10000) / "payload/study"
    validation.mkdir(parents=True)
    (validation / "proof.bin").write_bytes(b"native-proof")
    segment = ops.parent_path("segment", "boltz", 10000)
    (segment / "payload/study").mkdir(parents=True)
    (segment / "payload/study/projected.bin").write_bytes(b"funded-in-production-projection")
    (segment / "stderr.log").write_bytes(b"raw-log")
    assert ops.metadata_bytes() == len(b"source" + b"native-proof" + b"raw-log")


def test_control_inventory_rejects_external_alias_even_within_owned_directory(owned):
    source, root = owned
    target = root / "original-proof.json"
    target.write_bytes(b"read-only-original")
    (source / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="linked"):
        ops.metadata_bytes()


def startup_fixture(owned, monkeypatch, fail=False):
    source, root = owned
    current = plan()
    snapshot = save(root / "original-snapshot.json", {"saved": True})
    ready = save(root / "original-ready.json", {"status": "passed_original_only"})
    for arm in current["arms"]:
        arm["failed_parent"]["endpoint"] = snapshot
        arm["inputs"]["ready"] = ready
    save(source / "plan.json", current)
    calls = []
    def builder(plan, arm, phase, offset, output, snapshot, ready):
        return ["original", arm, phase, str(offset), str(output), snapshot["path"], ready["path"] if ready else "no-ready"]
    monkeypatch.setattr(ops, "_module", lambda *args: SimpleNamespace(build_original_argv=builder))
    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        arm = command[command.index("--arm") + 1]
        phase = command[command.index("--phase") + 1]
        offset = int(command[command.index("--offset") + 1])
        if "--worker-startup-only" in command:
            row = ops.arm_plan(current, arm)
            save(output.parent / "worker-startup.json", {"status": "passed_metadata_only", "arm": arm,
                 "worker_python": "/work3/s234818/latentmol/env/boltz311/bin/python",
                 "mosaic_origin": str(Path(row["source"]["path"]) / "src/mosaic/__init__.py"),
                 "numerical_setup_entered": False, "updates": 0, "original_scientific_source_unchanged": True})
            return SimpleNamespace(returncode=1 if fail else 0, stdout="actual worker imports", stderr="")
        save(output.parent / "endpoint-startup.json", {"status": "passed_metadata_only", "numerical_setup_entered": False,
                                                       "updates": 0, "job_id": os.environ["LSB_JOBID"], "readiness_deferred": "--startup-defer-ready" in command,
                                                       "original_argv": builder(current, arm, phase, offset, output, snapshot, ready if phase == "segment" else None)})
        return SimpleNamespace(returncode=1 if fail else 0, stdout="metadata only", stderr="")
    monkeypatch.setattr(ops.subprocess, "run", run)
    payload = root / "cpu/payload"
    payload.mkdir(parents=True)
    return current, payload, calls, ready


def test_cpu_executes_four_cli_routes_and_two_actual_worker_imports(owned, monkeypatch):
    current, payload, calls, ready = startup_fixture(owned, monkeypatch)
    result = ops.cpu_startups(current, PIN, payload, ops.time.monotonic() + 1000)
    assert len(result) == 4 and len(calls) == 6
    assert {(row["arm"], row["phase"]) for row in result} == {(arm, phase) for arm in ops.DISPATCH_ARMS for phase in ("validate", "segment")}
    for command in calls:
        assert command[command.index("--arm") + 1] in ops.DISPATCH_ARMS
        if "--worker-startup-only" in command:
            assert "--startup-only" not in command and "--ready" not in command
        else:
            assert "--startup-only" in command
            phase = command[command.index("--phase") + 1]
            assert ("--startup-defer-ready" in command) == (phase == "segment")
            assert ("--ready" in command) == (phase == "segment")
            if phase == "segment":
                assert command[command.index("--ready") + 1] == ready["path"]
        assert Path(command[command.index("--output") + 1]).is_relative_to(payload)
    assessment = ops.read(payload / "actual-startup-checks.json")
    assert assessment["native_readiness_established"] is False
    assert assessment["fresh_prospective_readiness_deferred_until_native_validation"] is True
    assert assessment["blocked_arms"] == ops.BLOCKED_ARMS
    assert len(ops.startup_proofs(assessment, current)) == 12


def test_actual_startup_failure_stops_before_any_native_submission(owned, monkeypatch):
    current, payload, calls, _ = startup_fixture(owned, monkeypatch, fail=True)
    with pytest.raises(ValueError, match="startup command failed"):
        ops.cpu_startups(current, PIN, payload, ops.time.monotonic() + 1000)
    assert len(calls) == 1
    assert not (payload / "actual-startup-checks.json").exists()
    assert ops.read(payload / "startup-boltz-validate/invocation.json")["returncode"] == 1


def test_cpu_startup_shared_deadline_is_not_reset_per_command(owned, monkeypatch):
    current, payload, calls, _ = startup_fixture(owned, monkeypatch)
    with pytest.raises(ValueError, match="shared startup/fixture budget"):
        ops.cpu_startups(current, PIN, payload, ops.time.monotonic() - 1)
    assert calls == []


@pytest.mark.parametrize("outcome", ["passed", "deferred", "failed"])
def test_callback_actual_new_readiness_startup_must_pass_without_defer(owned, monkeypatch, outcome):
    current, result, _ = valid_result(owned)
    source, root = owned
    ready_path = ops.parent_path("validate", "boltz", 10000) / "payload/study/assessment.json"
    ready = save(ready_path, {"status": "passed_new_prospective_readiness"})
    certificate = root / (ops.label("validate", "boltz", 10000) + "-certificate.json")
    save(certificate, {"status": "passed", "new_prospective": True})
    parent = ops.parent_path("after-validate", "boltz", 10000)
    save(parent / "submission.json", {"status": "submitted", "job_id": "222"})
    monkeypatch.setattr(ops, "verify_ready", lambda *args: None)
    def builder(plan, arm, phase, offset, output, snapshot, ready):
        return ["actual-original-driver", arm, phase, str(output), ready["path"], ready["sha256"]]
    monkeypatch.setattr(ops, "_module", lambda *args: SimpleNamespace(build_original_argv=builder))
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        save(output.parent / "endpoint-startup.json", {"status": "passed_metadata_only", "numerical_setup_entered": False,
                                                       "updates": 0, "job_id": "222", "readiness_deferred": outcome == "deferred",
                                                       "original_argv": builder(current, "boltz", "segment", 10000, output, result["input_snapshot"], ready)})
        return SimpleNamespace(returncode=1 if outcome == "failed" else 0, stdout="original readiness route", stderr="")
    monkeypatch.setattr(ops.subprocess, "run", run)
    if outcome == "passed":
        value = ops.final_startup(current, PIN, "boltz", 10000, parent, ops.time.monotonic() + 1000)
        assert value["readiness"] == ready and value["readiness_deferred"] is False
        assert ops.verify_segment_startup(current, PIN, "boltz", 10000) == value
    else:
        with pytest.raises(ValueError):
            ops.final_startup(current, PIN, "boltz", 10000, parent, ops.time.monotonic() + 1000)
        assert not (root / (ops.label("validate", "boltz", 10000) + "-segment-startup.json")).exists()
    assert len(calls) == 1 and "--startup-defer-ready" not in calls[0]
    assert calls[0][calls[0].index("--ready") + 1] == ready["path"]


@pytest.mark.parametrize("failure", ["timeout", "original_mutation"])
def test_child_failure_or_mutation_preserves_error_and_always_rehashes_originals(owned, monkeypatch, failure):
    _, root = owned
    parent = root / "allocated-child"
    parent.mkdir()
    observations = []
    def run(*args, **kwargs):
        if failure == "timeout":
            raise ops.subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return SimpleNamespace(returncode=0)
    def verify(*args):
        observations.append("capsule_rehashed")
    def external(*args):
        observations.append("originals_rehashed")
        if failure == "original_mutation":
            raise ValueError("original input bytes changed")
    monkeypatch.setattr(ops.subprocess, "run", run)
    monkeypatch.setattr(ops, "verify", verify)
    monkeypatch.setattr(ops, "external", external)
    with pytest.raises(ops.subprocess.TimeoutExpired if failure == "timeout" else ValueError):
        ops.run_child(["allocated-native-child"], 10, parent, {}, plan(), PIN)
    assert observations == ["capsule_rehashed", "originals_rehashed"]
    result = ops.read(parent / "post-execution-preservation.json")
    if failure == "timeout":
        assert result["preservation_status"] == "passed" and "execution_error" in result
    else:
        assert result["preservation_status"] == "failed" and "preservation_error" in result
        assert result["original_results_changed"] is None


@pytest.mark.parametrize("phase", ["validate", "segment", "after-validate", "after-segment"])
def test_blocked_protenix_has_no_submission_or_external_cli_route(phase):
    value = ops.validate_plan(plan())
    assert next(row for row in value["arms"] if row["id"] == "protenix")["global_target"] == 50000
    with pytest.raises(ValueError, match="Blocked"):
        ops.arm_plan(value, "protenix")
    with pytest.raises(ValueError):
        ops.argv(phase, PIN, "protenix", 20000, "ended(111)" if phase.startswith("after-") else None)


@pytest.mark.parametrize("failure", ["nonzero", "wrong_origin", "setup_entered", "updates", "source_changed"])
def test_actual_worker_failure_blocks_before_native_dispatch(owned, monkeypatch, failure):
    current, payload, calls, _ = startup_fixture(owned, monkeypatch)
    original = ops.subprocess.run
    def worker(command, **kwargs):
        result = original(command, **kwargs)
        if "--worker-startup-only" in command:
            output = Path(command[command.index("--output") + 1])
            receipt = output.parent / "worker-startup.json"
            row = ops.read(receipt)
            if failure == "wrong_origin":
                row["mosaic_origin"] = "/unverified/site-packages/mosaic/__init__.py"
            elif failure == "setup_entered":
                row["numerical_setup_entered"] = True
            elif failure == "updates":
                row["updates"] = 1
            elif failure == "source_changed":
                row["original_scientific_source_unchanged"] = False
            save(receipt, row)
            if failure == "nonzero":
                return SimpleNamespace(returncode=1, stdout="", stderr="ModuleNotFoundError: mosaic")
        return result
    monkeypatch.setattr(ops.subprocess, "run", worker)
    with pytest.raises(ValueError, match="worker"):
        ops.cpu_startups(current, PIN, payload, ops.time.monotonic() + 1000)
    assert len(calls) == 5
    assert not (payload / "actual-startup-checks.json").exists()
    assert not (owned[1] / "jobs.json").exists()


@pytest.mark.parametrize("failure", ["missing_worker", "duplicate_worker", "protenix_worker", "missing_cli", "block_changed"])
def test_cpu_callback_rejects_incomplete_or_changed_worker_proofs(owned, monkeypatch, failure):
    current, payload, _, _ = startup_fixture(owned, monkeypatch)
    ops.cpu_startups(current, PIN, payload, ops.time.monotonic() + 1000)
    value = ops.read(payload / "actual-startup-checks.json")
    if failure == "missing_worker":
        value["worker_checks"].pop()
    elif failure == "duplicate_worker":
        value["worker_checks"][1] = copy.deepcopy(value["worker_checks"][0])
    elif failure == "protenix_worker":
        value["worker_checks"][0]["arm"] = "protenix"
    elif failure == "missing_cli":
        value["checks"].pop()
    else:
        value["blocked_arms"] = {}
    with pytest.raises(ValueError, match="startup commands"):
        ops.startup_proofs(value, current)


@pytest.mark.parametrize("change", ["none", "manifest_missing", "proof_missing", "passed", "can_admit", "certify_old", "frozen_changed", "bond_passed", "wrong_historical_version"])
def test_rejected_original_protenix_admission_stays_file_bound_and_failed(monkeypatch, change):
    current = plan()
    blocked = ops.BLOCKED_ARMS["protenix"]["source01_admission"]
    manifest = {"blocked_arms": copy.deepcopy(ops.BLOCKED_ARMS), "preserved_proofs": [blocked]}
    observation = {"version": "prospective-endpoint-admission-v1", "status": "failed", "can_admit_new_child": False,
                   "original_run_failed_preserved": True, "can_certify_original_study": False,
                   "frozen_unchanged": True, "field_outcomes": {"E_upper": {"passed": False}}}
    if change == "manifest_missing":
        manifest.pop("blocked_arms")
    elif change == "proof_missing":
        manifest["preserved_proofs"] = []
    elif change == "passed":
        observation["status"] = "passed"
    elif change == "can_admit":
        observation["can_admit_new_child"] = True
    elif change == "certify_old":
        observation["can_certify_original_study"] = True
    elif change == "frozen_changed":
        observation["frozen_unchanged"] = False
    elif change == "bond_passed":
        observation["field_outcomes"]["E_upper"]["passed"] = True
    elif change == "wrong_historical_version":
        observation["version"] = "prospective-endpoint-admission-v2"
    seen = []
    def check(row):
        seen.append(row)
        assert row == blocked
        return row
    monkeypatch.setattr(ops, "check_binding", check)
    monkeypatch.setattr(ops, "read", lambda path: observation)
    if change == "none":
        assert ops.blocked_admission(manifest, current) == blocked
        assert seen == [blocked]
    else:
        with pytest.raises(ValueError):
            ops.blocked_admission(manifest, current)
