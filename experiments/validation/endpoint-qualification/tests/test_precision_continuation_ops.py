"""Allocated metadata-only controls for the two-arm precision child controller."""
from copy import deepcopy
import ast
import errno
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/precision_continuation_ops.py"
SPEC = importlib.util.spec_from_file_location("precision_continuation_ops_fixture", PATH)
o = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = o
SPEC.loader.exec_module(o)
PIN = "a" * 64


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Controller fixtures require allocated DTU HPC"


@pytest.fixture
def owned(tmp_path, monkeypatch):
    source = tmp_path / "release/source"
    root = tmp_path / "run"
    source.mkdir(parents=True)
    root.mkdir()
    monkeypatch.setattr(o, "SOURCE", source)
    monkeypatch.setattr(o, "ROOT", root)
    return source, root


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf8")
    return o.binding(path)


def plan():
    policies = {"protenix": {"protenix_contact_loss": "protenix-separate-precision-admission-v1"},
                "dbmol": {"dbmol_affinity_loss": "dbmol-separate-precision-finer-admission-v1",
                          "dbmol_contact_loss": "prospective-endpoint-admission-v3",
                          "dbmol_confidence_loss": "prospective-endpoint-admission-v3",
                          "dbmol_total_loss": "prospective-endpoint-admission-v3"}}
    refs = {name: {"command_prefix": ["/reference/python", "/reference/worker.py"], "environment": {},
                  "expected_identity": {"status": "ready_full_double_reference"},
                  "historical_global": o.ARMS[name]["offsets"][0],
                  "historical_original_directions": {"X": [], "E_upper": []},
                  "max_session_bytes": 64 * 1024**2,
                  "policy_kind": "protenix-original-fp64" if name == "protenix" else "dbmol-finer-fp64-affinity"}
            for name in o.DISPATCH_ARMS}
    arms = []
    for name in o.DISPATCH_ARMS:
        values, old = o.ARMS[name], o.ORIGINALS[name]
        source = o.BASE / old["release"]
        arms.append({"id": name, "offsets": list(values["offsets"]), "segment_steps": values["segment_steps"],
                     "segment_updates": values["segment_steps"], "start_global": values["offsets"][0],
                     "target_global": values["global_target"], "global_target": values["global_target"], "rate": values["rate"],
                     "source": {"path": str(source), "manifest_sha256": old["manifest"], "files": old["files"]},
                     "config": {"path": str(source / old["config"]), "bytes": old["config_bytes"], "sha256": old["config_sha256"]},
                     "failed_parent": {"job_id": old["job_id"], "endpoint": {"path": "/saved/endpoint.npz", "bytes": 1, "sha256": PIN}},
                     "inputs": {"original_driver_argv": [o.PYTHON, "/original/driver.py", "--mode", "run"]}})
    return {"version": o.PLAN_VERSION, "admission_policy": "precision-reference-continuation-policy-v1",
            "logit_policy": "prospective-fixed-reference-logit-admission-v1",
            "user_authorized_prospective_checks_before_children": True, "original_failures_preserved": True,
            "optimization_settings_changed": False, "monitor_remains_paused": True,
            "matrix_stride": 20, "loss_stride": 1, "dispatch_arms": list(o.DISPATCH_ARMS),
            "automatic_dispatch_arms": list(o.AUTOMATIC_ARMS),
            "preserved_failed_admissions": deepcopy(o.PRESERVED_FAILED_ADMISSIONS),
            "root": str(o.ROOT), "control_root": str(o.ROOT), "source_root": str(o.SOURCE),
            "precision_adapter_version": "precision-reference-continuation-adapter-v1",
            "precision_policy_by_arm": policies, "reference_configuration": {"arms": refs}, "arms": arms}


@pytest.mark.parametrize("arm,offsets,rate,target,steps", [
    ("protenix", [20000, 30000, 40000], .3, 50000, 10000), ("dbmol", [7500], .0025, 10000, 2500)])
def test_exact_two_arm_range_rate_and_immediate_saved_boundary(arm, offsets, rate, target, steps):
    current = plan()
    row = o.arm_plan(current, arm)
    assert row["offsets"] == offsets and row["rate"] == rate
    assert row["global_target"] == target and row["segment_steps"] == steps
    assert {r["id"] for r in current["arms"]} == {"protenix", "dbmol"}
    assert o.ORIGINALS["dbmol"]["job_id"] == "29830999"
    for offset in offsets:
        assert o.label("validate", arm, offset) == arm + "-validate-" + str(offset).zfill(5)


@pytest.mark.parametrize("failure", ["enable_boltz", "missing_arm", "duplicate_arm", "rate", "dbmol_5000",
                                    "target", "steps", "source", "config", "predecessor", "missing_argv",
                                    "policy", "preserved_failure", "stride", "loss_stride", "automatic", "root_alias"])
def test_changed_scope_source_history_or_science_rejected(failure):
    current = plan()
    if failure == "enable_boltz": current["dispatch_arms"].append("boltz")
    elif failure == "missing_arm": current["arms"].pop()
    elif failure == "duplicate_arm": current["arms"][1] = deepcopy(current["arms"][0])
    elif failure == "rate": current["arms"][0]["rate"] = .03
    elif failure == "dbmol_5000": current["arms"][1]["offsets"] = [5000, 7500]
    elif failure == "target": current["arms"][0]["global_target"] = 60000
    elif failure == "steps": current["arms"][1]["segment_steps"] = 1000
    elif failure == "source": current["arms"][1]["source"]["manifest_sha256"] = "b" * 64
    elif failure == "config": current["arms"][0]["config"]["sha256"] = "b" * 64
    elif failure == "predecessor": current["arms"][1]["failed_parent"]["job_id"] = "29810971"
    elif failure == "missing_argv": current["arms"][1]["inputs"].pop("original_driver_argv")
    elif failure == "policy": current["admission_policy"] = "prospective-endpoint-admission-v3"
    elif failure == "preserved_failure": current["preserved_failed_admissions"]["protenix"]["status"] = "passed"
    elif failure == "stride": current["matrix_stride"] = 1
    elif failure == "loss_stride": current["loss_stride"] = 20
    elif failure == "automatic": current["automatic_dispatch_arms"] = ["boltz"]
    else: current["root"] += "-old"
    with pytest.raises(ValueError): o.validate_plan(current)


@pytest.mark.parametrize("phase", ["cpu", "after-cpu", "validate", "after-validate", "segment", "after-segment"])
def test_actual_resources_wrapper_and_dependency_are_declared(phase):
    callback = phase.startswith("after-")
    arm, offset = ("all", 0) if phase in ("cpu", "after-cpu") else ("dbmol", 7500)
    argv = o.argv(phase, PIN, arm, offset, "ended(111)" if callback else None)
    assert argv[-5:] == [str(o.SOURCE), PIN, phase, arm, str(offset)]
    assert str(o.SOURCE / "scripts/precision_continuation_stage.lsf") in argv
    assert argv[argv.index("-n") + 1] == ("4" if phase in ("cpu", "validate", "segment") else "1")
    assert argv[argv.index("-q") + 1] == ("gpul40s" if phase in ("validate", "segment") else "hpc")
    assert ("num=1:mode=exclusive_process" in argv) is (phase in ("validate", "segment"))
    assert ("-w" in argv) is callback
    assert argv[argv.index("-W") + 1] == ("23:00" if phase == "segment" else "02:00" if phase == "validate"
                                          else "01:00" if phase == "cpu" else "00:20")


@pytest.mark.parametrize("phase,dependency", [("validate", "ended(111)"), ("segment", "done(111)"),
                                              ("after-validate", None), ("after-segment", "done(111)")])
def test_interarm_or_wrong_dependency_cannot_be_introduced(phase, dependency):
    with pytest.raises(ValueError): o.argv(phase, PIN, "dbmol", 7500, dependency)


def test_unknown_submission_preserves_exclusive_marker_and_prevents_duplicate(owned, monkeypatch):
    def unknown(*args, **kwargs): raise TimeoutError("unknown actual bsub outcome")
    monkeypatch.setattr(o.subprocess, "run", unknown)
    with pytest.raises(ValueError, match="Unknown"): o.submit("cpu", PIN)
    root = owned[1]
    assert o.read(root / "cpu/submission.json")["status"] == "unknown"
    assert (root / "cpu/submission-started.json").is_file()
    with pytest.raises(FileExistsError): o.submit("cpu", PIN)


def test_actual_submission_keeps_only_own_ended_pair(owned, monkeypatch):
    calls = []
    def submit(phase, pin, arm, offset, dependency=None):
        calls.append((phase, arm, offset, dependency))
        return {"job_id": str(111 + len(calls))}
    monkeypatch.setattr(o, "submit", submit)
    assert o._pair("validate", PIN, "dbmol", 7500) == {"producer": "112", "callback": "113"}
    assert calls == [("validate", "dbmol", 7500, None), ("after-validate", "dbmol", 7500, "ended(112)")]


def test_initial_checkpoint_is_immediate7500_failed_parent(owned):
    current = plan()
    exact = save(owned[1] / "saved7500.json", {"original": "unchanged7500"})
    current["arms"][1]["failed_parent"]["endpoint"] = exact
    assert o.input_snapshot(current, "dbmol", 7500) == exact
    with pytest.raises(ValueError): o.input_snapshot(current, "dbmol", 5000)


def test_later_pro_checkpoint_requires_own_passed_parent_certificate(owned):
    source, root = owned
    current = plan()
    save(source / "plan.json", current)
    save(source / "source_manifest.json", {"files": {"plan.json": o.sha(source / "plan.json")}})
    cert_path = root / (o.label("segment", "protenix", 20000) + "-certificate.json")
    save(cert_path, {"status": "failed", "arm": "protenix", "global_end": 30000})
    with pytest.raises(ValueError, match="not certified"): o.input_snapshot(current, "protenix", 30000)
    proof = save(root / "parent-proof.json", {"status": "passed_new_child"})
    terminal = save(root / "terminal30000.json", {"exact": "new_child_endpoint"})
    save(cert_path, {"status": "passed", "arm": "protenix", "global_end": 30000,
                     "source_manifest_sha256": o.sha(source / "source_manifest.json"),
                     "plan_sha256": o.sha(source / "plan.json"), "proofs": [proof], "terminal_snapshot": terminal})
    assert o.input_snapshot(current, "protenix", 30000) == terminal


@pytest.mark.parametrize("failure", ["version", "status", "certify", "changed", "unbound"])
def test_original_pro_failure_is_preserved_and_never_recertified(owned, monkeypatch, failure):
    root = owned[1]
    observation = {"version": "prospective-endpoint-admission-v1", "status": "failed", "can_admit_new_child": False,
                   "original_run_failed_preserved": True, "can_certify_original_study": False,
                   "frozen_unchanged": True, "field_outcomes": {"E_upper": {"passed": False}}}
    original = save(root / "original-failure.json", observation)
    declared = {"protenix": {"status": "blocked_failed_endpoint_admission", "source01_admission": original}}
    monkeypatch.setattr(o, "PRESERVED_FAILED_ADMISSIONS", declared)
    current = plan()
    manifest = {"preserved_failed_admissions": deepcopy(declared), "preserved_proofs": [original]}
    assert o.preserved_failed_admission(manifest, current) == original
    if failure == "version": observation["version"] = "new"
    elif failure == "status": observation["status"] = "passed"
    elif failure == "certify": observation["can_certify_original_study"] = True
    elif failure == "changed": observation["field_outcomes"]["E_upper"]["passed"] = True
    else: manifest["preserved_proofs"] = []
    if failure != "unbound":
        rebound = save(root / "original-failure.json", observation)
        declared["protenix"]["source01_admission"] = rebound
        current["preserved_failed_admissions"] = deepcopy(declared)
        manifest = {"preserved_failed_admissions": deepcopy(declared), "preserved_proofs": [rebound]}
    with pytest.raises(ValueError): o.preserved_failed_admission(manifest, current)


def qualified_fixture(owned, monkeypatch):
    root = owned[1]
    expected = deepcopy(o.QUALIFIED_NATIVE)
    declarations = {}
    for name, spec in expected.items():
        native_root = root / ("qualified-" + name)
        source = root / ("qualified-source-" + name)
        source.mkdir()
        spec["root"] = str(native_root)
        spec["source"]["path"] = str(source)
        proofs = []
        def artifact(relative, value):
            row = save(native_root / relative, value)
            proofs.append(row)
            return row
        plan_row = save(source / "plan.json", {"scope": "metadata toy qualification"})
        proofs.append(plan_row)
        preserved_proof = artifact("cpu/raw-proof.json", {"fixture": "passed"})
        artifact("cpu-certificate.json", {"status": "passed", "source_manifest_sha256": spec["source"]["manifest_sha256"],
            "cases": 1, "proofs": [preserved_proof]})
        policy = "protenix-separate-precision-admission-v1" if name == "protenix" else "dbmol-separate-precision-finer-admission-v1"
        admission = {"version": policy, "status": "passed", "can_admit_new_child": True,
                     "can_certify_original_study": False, "original_failed_outcome_preserved": True,
                     "frozen_unchanged": True, "baseline_fp32_unchanged": True, "optimization_updates": 0,
                     "failures": [], "field_outcomes": {"X": {"passed": True}, "E_upper": {"passed": True}},
                     "directions": [{"field": "X", "passed": True}, {"field": "E_upper", "passed": True}]}
        prefix = "gpu/payload/reference/" if name == "protenix" else "gpu/payload/native/"
        admission_row = artifact(prefix + ("admission.json" if name == "protenix" else "native-admission.json"), admission)
        for negative in ("zero", "reversed", "scaled"):
            artifact(prefix + negative + ("-control.json" if name == "protenix" else "-admission.json"),
                     {"status": "failed", "can_admit_new_child": False, "can_certify_original_study": False})
        if name == "protenix":
            reference = artifact(prefix + "assessment.json", {"status": "passed_new_prospective_precision_measurement",
                "updates": 0, "admission": admission_row, "original_results_changed": False})
            production = artifact("gpu/payload/production/assessment.json", {"status": "passed_fp32_measurements"})
            workers = {}
            for worker in ("fp32", "fp64"):
                preservation = artifact("gpu/payload/" + worker + "-final-preservation.json", {
                    "status": "passed", "updates": 0, "protected_inputs_unchanged": True,
                    "frozen_unchanged": True, "preservation_errors": []})
                workers[worker] = {"assessment": production if worker == "fp32" else reference, "preservation": preservation}
            artifact("gpu/payload/protected-before.json", {"saved": "unchanged"})
            artifact("gpu/payload/protected-after.json", {"saved": "unchanged"})
            artifact(prefix + "native-dtype-audit.json", {"status": "passed", "floating_operations_float64": True,
                                                        "narrow_float_avals": []})
            assessment = {"status": "prospective_precision_gate_passed_original_failures_preserved", "job_id": spec["job_id"],
                "updates": 0, "original_results_changed": False, "optimization_settings_changed": False,
                "can_certify_or_restart_original_study": False, "production_optimizer_executed": False,
                "frozen_unchanged": True, "preservation_errors": [], "plan": plan_row,
                "reference_assessment": reference, "worker_proofs": workers}
        else:
            inner = artifact(prefix + "assessment.json", {"status": "passed_affinity_reference_only", "error": None,
                "optimizer_updates": 0, "original_results_changed": False, "can_submit_continuation": False,
                "remaining_admission": "all_components_raw_null_parity_pilot_runtime_startup_full_storage"})
            artifact(prefix + "preservation.json", {"status": "passed", "before": {"saved": "unchanged"},
                                                    "after": {"saved": "unchanged"}})
            artifact(prefix + "reference-operation-audit.json", {"status": "closed", "audit": {
                "status": "passed_full_operation_FP64_audit", "narrow_floating_operations": 0,
                "production_execution_changed": False}})
            artifact(prefix + "production-frozen-after.json", {"frozen": True})
            assessment = {"status": "passed_affinity_reference_only", "optimizer_jobs_submitted": False, "native_assessment": inner}
        assessment_row = artifact("gpu/payload/assessment.json", assessment)
        command = ["bsub", "-q", "gpul40s", "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]",
                   "-gpu", "num=1:mode=exclusive_process", "-W", "02:00", "-J", spec["name"],
                   "-oo", str(native_root / "gpu/stdout.%J.log"), "-eo", str(native_root / "gpu/stderr.%J.log"),
                   "bash", "-l", str(source / "scripts" / spec["wrapper"]), str(source),
                   spec["source"]["manifest_sha256"], "gpu"]
        submitted = artifact("gpu/submission.json", {"status": "submitted", "job_id": spec["job_id"], "phase": "gpu",
            "dependency": None, "source_manifest_sha256": spec["source"]["manifest_sha256"], "argv": command, "returncode": 0})
        stdout = native_root / "gpu" / ("stdout." + spec["job_id"] + ".log")
        stdout.write_text("Subject: Job " + spec["job_id"] + ": <" + spec["name"] + "> in cluster <dcc> Done\n"
            "Job was executed in queue <gpul40s>.\nSuccessfully completed.\n", encoding="utf8")
        stdout_row = o.binding(stdout); proofs.append(stdout_row)
        declarations[name] = {"source": deepcopy(spec["source"]), "job_id": spec["job_id"], "assessment": assessment_row,
                              "submission": submitted, "stdout": stdout_row, "proofs": proofs}
    monkeypatch.setattr(o, "QUALIFIED_NATIVE", expected)
    def verified(source, pin):
        spec = next(spec for spec in expected.values() if spec["source"]["path"] == str(source))
        assert spec["source"]["manifest_sha256"] == pin
        return {"files": {str(i): PIN for i in range(spec["source"]["files"])}, "cpu_counts": {"tests.test_toy": 1}}
    monkeypatch.setattr(o, "verify", verified)
    return {"qualified_admissions": declarations}


def test_both_exact_qualified_native_passes_are_component_only_adoption(owned, monkeypatch):
    current = qualified_fixture(owned, monkeypatch)
    result = o.qualified_admissions(current)
    assert result["status"] == "passed_readonly_native_qualification_adoption"
    assert result["full_continuation_admission_established"] is result["original_results_changed"] is False
    assert set(result["arms"]) == {"protenix", "dbmol"}
    assert all(row["full_continuation_admission_established"] is False for row in result["arms"].values())


@pytest.mark.parametrize("failure", ["missing_arm", "wrong_job", "source", "proof_changed", "lsf_failed",
                                    "overall_native_failed", "negative_admitted", "original_recertified"])
def test_qualified_native_missing_wrong_or_failed_evidence_cannot_authorize_children(owned, monkeypatch, failure):
    current = qualified_fixture(owned, monkeypatch)
    declarations = current["qualified_admissions"]
    row = declarations["dbmol"]
    if failure == "missing_arm": del declarations["protenix"]
    elif failure == "wrong_job": row["job_id"] = "999"
    elif failure == "source": row["source"]["manifest_sha256"] = "f" * 64
    else:
        if failure == "proof_changed":
            Path(row["assessment"]["path"]).write_text("changed", encoding="utf8")
        elif failure == "lsf_failed":
            Path(row["stdout"]["path"]).write_text("Exited with exit code 1.\n", encoding="utf8")
        else:
            proof = (row["assessment"] if failure == "overall_native_failed" else next(
                proof for proof in row["proofs"] if proof["path"].endswith("/zero-admission.json")))
            value = o.read(proof["path"])
            if failure == "overall_native_failed": value["status"] = "failed"
            elif failure == "negative_admitted": value.update(status="passed", can_admit_new_child=True)
            else: value["can_certify_original_study"] = True
            replacement = save(Path(proof["path"]), value)
            row["proofs"] = [replacement if item["path"] == replacement["path"] else item for item in row["proofs"]]
            if failure == "overall_native_failed": row["assessment"] = replacement
    with pytest.raises(ValueError): o.qualified_admissions(current)


def capacity_fixture(owned, monkeypatch, available=80 * 1024**3):
    source, root = owned
    parent = root / "storage-test"
    parent.mkdir()
    prior = {"status": "passed", "required_bytes": 50 * o.GIB, "available_bytes": available,
             "reserve_bytes": 4 * o.GIB, "file_reserve": 100000, "original_gate_changed": False,
             "original_full_forecast": {"version": "selected-nonchem-full-storage-v1", "selected": {
                 "protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}}
    calls = []
    def gate(where, phase):
        calls.append((Path(where), phase))
        return deepcopy(prior)
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=gate))
    monkeypatch.setattr(o, "metadata_bytes", lambda: 0)
    monkeypatch.setattr(o, "owned_bytes", lambda path: 100 if path == root else 200)
    return parent, prior, calls, gate


def test_storage_adds_all_twelve_reference_caps_control_and_entire_new_tree_charge(owned, monkeypatch):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    result = o.storage(parent, "dispatch")
    assert result["status"] == "passed" and len(calls) == 1
    assert result["required_bytes"] == prior["required_bytes"] + 1280 * 1024**2
    assert result["available_bytes"] == prior["available_bytes"] - 300
    assert result["additional_complete_reference_and_control_bytes"] == 1280 * 1024**2
    assert result["reference_sessions"] == 12 and result["reference_session_cap"] == 64 * 1024**2
    assert result["forecasts_removed"] is result["original_gate_changed"] is False
    assert result["reserve_bytes"] == 4 * o.GIB and result["file_reserve"] == 100000
    assert o.read(parent / "continuation-storage-dispatch.json") == result


def test_actual_storage_funds_fresh_pilot_positive_excess_without_borrowing_old_future(owned, monkeypatch):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    save(owned[1] / "projections.json", {"dbmol": {"unit_bytes": 20000000, "remaining_offsets": [7500]}})
    excess, details = o.projection_excess_from_gate(prior)
    assert excess > 0
    result = o.storage(parent, "dispatch")
    assert result["required_bytes"] == prior["required_bytes"] + 1280 * 1024**2 + excess
    assert result["additional_positive_pilot_projection_bytes"] == excess
    assert result["new_pilot_projection_details"] == details
    assert prior["original_full_forecast"]["selected"]["dbmol10k"]["remaining"] == o.GIB


def test_projection_uses_smallest_authenticated_inherited_remaining_budget(owned):
    save(owned[1] / "projections.json", {"dbmol": {"unit_bytes": 20000000, "remaining_offsets": [7500]}})
    gates = {"one": {"version": "selected-nonchem-full-storage-v1", "selected": {
        "protenix50k40k": {"remaining": 9 * o.GIB}, "dbmol10k": {"remaining": 5 * o.GIB}}},
        "two": [{"version": "selected-nonchem-full-storage-v1", "selected": {
            "protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}]}
    amount, details = o.projection_excess_from_gate(gates)
    assert amount > 0 and details["dbmol"]["original_unfunded_projection"] == o.GIB


@pytest.mark.parametrize("failure", ["missing_profiles", "negative_remaining", "boolean_remaining"])
def test_fresh_projection_cannot_use_missing_or_nonfinite_inherited_full_budget(owned, failure):
    prior = {"version": "selected-nonchem-full-storage-v1", "selected": {
        "protenix50k40k": {"remaining": o.GIB}, "dbmol10k": {"remaining": o.GIB}}}
    if failure == "missing_profiles": del prior["selected"]["dbmol10k"]
    elif failure == "negative_remaining": prior["selected"]["dbmol10k"]["remaining"] = -1
    else: prior["selected"]["dbmol10k"]["remaining"] = True
    with pytest.raises(ValueError): o.projection_excess_from_gate(prior)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("original_gate_changed", True),
                                        ("reserve_bytes", 3 * 1024**3), ("file_reserve", 99999)])
def test_inherited_storage_or_reserve_failure_cannot_be_waived(owned, monkeypatch, field, value):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch)
    prior[field] = value
    with pytest.raises(ValueError): o.storage(parent, "entry")
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["new_capacity", "control_cap"])
def test_new_full_storage_failure_is_preserved_without_retry(owned, monkeypatch, failure):
    parent, prior, calls, gate = capacity_fixture(owned, monkeypatch, available=50 * 1024**3 + 1000)
    if failure == "control_cap":
        prior["available_bytes"] = 80 * o.GIB
        monkeypatch.setattr(o, "metadata_bytes", lambda: o.CONTROL_CAP + 1)
    with pytest.raises(ValueError): o.storage(parent, "entry")
    assert len(calls) == 1
    failure_path = parent / "whole-precision-entry-1/full-precision-storage.json"
    assert failure_path.is_file() and o.read(failure_path)["status"] == "failed"
    assert not (parent / "whole-precision-entry-2").exists()


def test_exact_owned_atomic_race_retries_whole_inventory_and_keeps_failed_attempt(owned, monkeypatch):
    parent, prior, calls, normal = capacity_fixture(owned, monkeypatch)
    atomic = o.BASE / "nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp"
    attempts = []
    def racing(where, phase):
        attempts.append(Path(where))
        if len(attempts) == 1: raise FileNotFoundError(errno.ENOENT, "owned atomic reference replaced", str(atomic))
        return deepcopy(prior)
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=racing))
    assert o.storage(parent, "dispatch")["status"] == "passed"
    assert len(attempts) == 2
    incomplete = o.read(attempts[0] / "incomplete-owned-atomic-inventory.json")
    assert incomplete["partial_totals_used"] is incomplete["files_skipped"] is False


@pytest.mark.parametrize("kind", ["unrelated_missing", "exhausted_atomic", "capacity"])
def test_inventory_retry_never_skips_unrelated_files_or_retries_other_failures(owned, monkeypatch, kind):
    parent, prior, calls, normal = capacity_fixture(owned, monkeypatch)
    atomic = o.BASE / "nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study/case-0/references/latest-reference.json.tmp"
    attempts = []
    def failing(where, phase):
        attempts.append(Path(where))
        if kind == "capacity": raise ValueError("no capacity")
        raise FileNotFoundError(errno.ENOENT, "missing", str(atomic if kind == "exhausted_atomic" else parent / "unrelated-proof"))
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(gate=failing))
    with pytest.raises((ValueError, FileNotFoundError)): o.storage(parent, "entry")
    assert len(attempts) == (3 if kind == "exhausted_atomic" else 1)


def test_projection_only_adds_positive_fresh_pilot_excess_and_never_reduces_old_future():
    original = {"selected": {"protenix50k40k": {"remaining": 3 * o.GIB}, "dbmol10k": {"remaining": o.GIB}}}
    forecast = {"dbmol": {"unit_bytes": 20, "remaining_offsets": [7500]}}
    extra, details = o.projection_excess(original, forecast)
    assert extra >= 0 and original["selected"]["dbmol10k"]["remaining"] == o.GIB
    forecast["dbmol"]["unit_bytes"] = 100000000
    larger, details = o.projection_excess(original, forecast)
    assert larger > extra and larger == details["dbmol"]["positive_excess_bytes"]
    assert details["dbmol"]["remaining_offsets"] == [7500]


@pytest.mark.parametrize("offsets", [[5000, 7500], [7500, 5000], [], [9999]])
def test_dbmol_projection_suffix_cannot_restore_completed_or_invented_updates(offsets):
    with pytest.raises(ValueError):
        o.projection_excess({"selected": {"dbmol10k": {"remaining": 1}}},
                            {"dbmol": {"unit_bytes": 20, "remaining_offsets": offsets}})


def result_fixture(owned, monkeypatch):
    source, root = owned
    current = plan()
    snapshot = save(root / "saved7500.json", {"original": "preserved7500"})
    row = next(item for item in current["arms"] if item["id"] == "dbmol")
    row["failed_parent"]["endpoint"] = snapshot
    save(source / "plan.json", current)
    proof = save(root / "new-native-proof.json", {"passed": True})
    result = {"version": o.RESULT_VERSION, "status": "passed", "job_id": "111", "arm": "dbmol",
              "phase": "validate", "offset": 7500, "global_end": 7500, "source_manifest_sha256": PIN,
              "plan_sha256": o.sha(source / "plan.json"), "config_sha256": row["config"]["sha256"],
              "original_results_changed": False, "frozen_unchanged": True, "original_failed_parent_preserved": True,
              "scientific_source_manifest_sha256": row["source"]["manifest_sha256"],
              "endpoint_policy": "precision-reference-continuation-policy-v1", "logit_records": [],
              "checks": {key: True for key in o.CHECKS}, "proofs": [proof], "input_snapshot": snapshot,
              "projections": {"maximum_pilot_update_bytes": 2000},
              "applicability": {"pilot_high": "not_applicable_chemistry_disabled",
                                "terminal_admission": "not_applicable_validation_phase",
                                "case_jsonl": "not_applicable_validation_phase", "fp32_boundary": "required",
                                "null_controls": "original_null_controls"}}
    parent = o.parent_path("validate", "dbmol", 7500)
    save(parent / "submission.json", {"status": "submitted", "job_id": "111"})
    save(parent / "post-execution-preservation.json", {"preservation_status": "passed",
          "source_manifest_sha256": PIN, "original_results_changed": False})
    path = parent / "payload/assessment.json"
    save(path, result)
    calls = []
    def verify_precision(*args):
        calls.append(args)
        return args[0]
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(verify_precision_result=verify_precision))
    return current, result, path, calls


def test_result_requires_additional_reference_adapter_verification(owned, monkeypatch):
    current, result, path, calls = result_fixture(owned, monkeypatch)
    assert o.verify_result(current, PIN, "validate", "dbmol", 7500) == result
    assert len(calls) == 1 and calls[0][0] == result and calls[0][2:6] == (PIN, "validate", "dbmol", 7500)
    def reject(*args): raise ValueError("Reference co-residency failed")
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(verify_precision_result=reject))
    with pytest.raises(ValueError, match="co-residency"): o.verify_result(current, PIN, "validate", "dbmol", 7500)


@pytest.mark.parametrize("failure", ["result_failed", "old_policy", "wrong_source", "wrong_config", "wrong_job",
                                    "frozen", "original_changed", "missing_check", "false_check", "snapshot", "global_end"])
def test_failed_identity_nonraw_gate_or_policy_blocks_before_reference_verifier(owned, monkeypatch, failure):
    current, result, path, calls = result_fixture(owned, monkeypatch)
    if failure == "result_failed": result["status"] = "failed"
    elif failure == "old_policy": result["endpoint_policy"] = "prospective-endpoint-admission-v3"
    elif failure == "wrong_source": result["source_manifest_sha256"] = "b" * 64
    elif failure == "wrong_config": result["config_sha256"] = "b" * 64
    elif failure == "wrong_job": result["job_id"] = "112"
    elif failure == "frozen": result["frozen_unchanged"] = False
    elif failure == "original_changed": result["original_results_changed"] = True
    elif failure == "missing_check": del result["checks"]["null_controls"]
    elif failure == "false_check": result["checks"]["hard_parity"] = False
    elif failure == "snapshot": result["input_snapshot"] = {**result["input_snapshot"], "sha256": "b" * 64}
    else: result["global_end"] = 10000
    save(path, result)
    with pytest.raises(ValueError): o.verify_result(current, PIN, "validate", "dbmol", 7500)
    if failure != "global_end": assert calls == []


def reference_startup_fixture(owned, monkeypatch):
    source, root = owned
    current = plan()
    for arm, config in current["reference_configuration"]["arms"].items():
        config["command_prefix"] = [o.PYTHON, str(source / "scripts" /
            ("protenix_reference_worker.py" if arm == "protenix" else "dbmol_reference.py"))]
        if arm == "protenix": config["command_prefix"] += ["--reference-plan-sha256", "${OWN_PLAN_SHA256}"]
    save(source / "plan.json", current)
    save(root / "cpu/submission.json", {"status": "submitted", "job_id": os.environ["LSB_JOBID"]})
    payload = root / "cpu/payload"
    payload.mkdir()
    calls = []
    def expanded(config, plan_sha):
        return [plan_sha if item == "${OWN_PLAN_SHA256}" else item for item in config["command_prefix"]]
    monkeypatch.setattr(o, "_module", lambda *args: SimpleNamespace(expanded_command=expanded))
    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--startup-output") + 1])
        receipt = {"status": "passed_metadata_only", "numerical_setup_entered": False,
                   "optimizer_updates": 0, "job_id": os.environ["LSB_JOBID"]}
        if output.name.startswith("protenix"):
            receipt.update(own_source_manifest_sha256=PIN, plan_sha256=o.sha(source / "plan.json"))
        save(output, receipt)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(o.subprocess, "run", run)
    return current, payload, calls


def test_both_actual_reference_startup_commands_are_metadata_only_and_file_bound(owned, monkeypatch):
    current, payload, calls = reference_startup_fixture(owned, monkeypatch)
    reports = o.cpu_reference_startups(current, PIN, payload, o.time.monotonic() + 1000)
    assert len(reports) == len(calls) == 2
    for command, kwargs in calls:
        assert command[-3:] == ["--startup-only", "--startup-output", command[-1]]
        assert kwargs["env"]["JAX_PLATFORMS"] == "cpu" and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "${OWN_PLAN_SHA256}" not in command
        assert Path(command[-1]).is_relative_to(payload)
    assert len(o.reference_startup_proofs({"reference_checks": reports}, current, PIN)) == 4


@pytest.mark.parametrize("failure", ["numerical", "updates", "wrong_owner", "source_pin", "argv", "missing_reference"])
def test_reference_startup_missing_changed_or_cross_job_receipts_block_native_dispatch(owned, monkeypatch, failure):
    current, payload, calls = reference_startup_fixture(owned, monkeypatch)
    reports = o.cpu_reference_startups(current, PIN, payload, o.time.monotonic() + 1000)
    if failure == "missing_reference": reports.pop()
    else:
        row = reports[0]
        key = "invocation" if failure == "argv" else "receipt"
        value = o.read(row[key]["path"])
        if failure == "numerical": value["numerical_setup_entered"] = True
        elif failure == "updates": value["optimizer_updates"] = 1
        elif failure == "wrong_owner": value["job_id"] = "99999999"
        elif failure == "source_pin": value["own_source_manifest_sha256"] = "b" * 64
        else: value["argv"][0] = "/unbound/python"
        row[key] = save(Path(row[key]["path"]), value)
    with pytest.raises(ValueError): o.reference_startup_proofs({"reference_checks": reports}, current, PIN)


def test_no_scientific_import_or_optimizer_entry_exists_in_controller():
    tree = ast.parse(PATH.read_bytes())
    top = {a.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for a in node.names}
    assert not top & {"numpy", "jax", "torch", "equinox", "mosaic"}
    assert o.DISPATCH_ARMS == ("protenix", "dbmol") and o.AUTOMATIC_ARMS == o.DISPATCH_ARMS
    assert o.SOURCE != o.BASE / "nonchem-endpoint-release-09/source"
