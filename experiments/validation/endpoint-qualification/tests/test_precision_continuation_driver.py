"""Allocated metadata controls for the actual-entry precision adapter.

These controls use toy JSON/proof files, never pretrained numerical evaluation.
Real original CLI startup, full native/production/pilot and reference residency
must pass separately in the authenticated allocated capsule.
"""
from contextlib import contextmanager
from copy import deepcopy
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/precision_continuation_driver.py"
SPEC = importlib.util.spec_from_file_location("precision_continuation_driver_fixture", PATH)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Adapter controls require allocated DTU HPC"


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def plan_for(tmp_path):
    reference = {}
    arms = []
    for name, (start, target, steps, offsets) in d.BOUNDARIES.items():
        reference[name] = {"command_prefix": ["/reference/python", "/reference/worker.py"], "environment": {},
                           "expected_identity": {"status": "ready"},
                           "historical_global": start,
                           "policy_kind": "protenix-original-fp64" if name == "protenix" else "dbmol-finer-fp64-affinity"}
        arms.append({"id": name, "start_global": start, "target_global": target,
                     "segment_updates": steps, "offsets": offsets,
                     "source": {"manifest_sha256": "a" * 64}, "config": {"sha256": "b" * 64}})
    return {"precision_adapter_version": d.ADAPTER_VERSION, "precision_policy_by_arm": deepcopy(d.POLICIES),
            "optimization_settings_changed": False, "original_failures_preserved": True,
            "matrix_stride": 20, "loss_stride": 1, "dispatch_arms": ["protenix", "dbmol"],
            "arms": arms, "root": str(tmp_path / "new"), "reference_configuration": {"arms": reference},
            "base_driver": {"path": str(tmp_path / "historical.py"), "bytes": 1, "sha256": "c" * 64},
            "_sha256": "d" * 64}


@pytest.mark.parametrize("failure", ["adapter", "policy", "settings", "failure_preservation", "matrix_stride",
                                    "loss_stride", "dispatch", "start", "offsets", "target", "steps", "environment",
                                    "historical_global", "reference_kind"])
def test_plan_rejects_changed_scientific_boundary_scope_or_policy(tmp_path, failure):
    plan = plan_for(tmp_path)
    assert d.validate_adapter_plan(plan) is plan
    if failure == "adapter": plan["precision_adapter_version"] = "old"
    elif failure == "policy": plan["precision_policy_by_arm"]["dbmol"]["dbmol_affinity_loss"] = d.RAW_POLICY
    elif failure == "settings": plan["optimization_settings_changed"] = True
    elif failure == "failure_preservation": plan["original_failures_preserved"] = False
    elif failure == "matrix_stride": plan["matrix_stride"] = 1
    elif failure == "loss_stride": plan["loss_stride"] = 20
    elif failure == "dispatch": plan["dispatch_arms"] = ["protenix", "dbmol", "boltz"]
    elif failure == "start": plan["arms"][1]["start_global"] = 5000
    elif failure == "offsets": plan["arms"][1]["offsets"] = [5000, 7500]
    elif failure == "target": plan["arms"][0]["target_global"] = 60000
    elif failure == "steps": plan["arms"][0]["segment_updates"] = 9999
    elif failure == "environment": plan["reference_configuration"]["arms"]["protenix"]["environment"] = {"FLAG": False}
    elif failure == "historical_global": plan["reference_configuration"]["arms"]["dbmol"]["historical_global"] = 5000
    else: plan["reference_configuration"]["arms"]["protenix"]["policy_kind"] = "production-fp64"
    with pytest.raises(ValueError):
        d.validate_adapter_plan(plan)


def native_report():
    return {"status": "passed", "policy": d.DB_REPORT_POLICY, "policy_by_objective": deepcopy(d.POLICIES["dbmol"]),
            "all_components_and_total_endpoint_policy_passed": True,
            "hard_native_parity": {"status": "passed"}, "frozen_unchanged": True,
            "key_and_logit_row_shift_nulls": "passed", "live_primitive_recomposition": "passed_exact_scalar_and_raw_gradients",
            "components": {name: {"status": "passed", "endpoint_policy": policy,
                "typed_bond_contrast_nonzero": True, "production_jax_transport": "passed",
                "independent_primal": True, "raw_unclipped_gradients": True,
                "prospective_endpoint_admission": {"version": policy, "status": "passed",
                                                   "can_admit_new_child": True, "frozen_unchanged": True}}
                for name, policy in d.POLICIES["dbmol"].items()}}


@pytest.mark.parametrize("failure", ["missing_component", "old_outer_policy", "raw_affinity", "old_raw_success",
                                    "hard_parity", "null", "recomposition", "typed_nonzero", "transport", "frozen"])
def test_dbmol_complete_native_metadata_cannot_masquerade_or_drop_nonraw_checks(failure):
    report = native_report()
    assert d.verify_dbmol_native_report(report) is report
    if failure == "missing_component": del report["components"]["dbmol_total_loss"]
    elif failure == "old_outer_policy": report["policy"] = "dbmol-prospective-endpoint-v1"
    elif failure == "raw_affinity": report["components"]["dbmol_affinity_loss"]["endpoint_policy"] = d.RAW_POLICY
    elif failure == "old_raw_success": report["all_components_and_total_raw_fd_passed"] = True
    elif failure == "hard_parity": report["hard_native_parity"]["status"] = "failed"
    elif failure == "null": report["key_and_logit_row_shift_nulls"] = "failed"
    elif failure == "recomposition": report["live_primitive_recomposition"] = "failed"
    elif failure == "typed_nonzero": report["components"]["dbmol_contact_loss"]["typed_bond_contrast_nonzero"] = False
    elif failure == "transport": report["components"]["dbmol_confidence_loss"]["production_jax_transport"] = "failed"
    else: report["components"]["dbmol_total_loss"]["prospective_endpoint_admission"]["frozen_unchanged"] = False
    with pytest.raises(ValueError):
        d.verify_dbmol_native_report(report)


def session_fixture(tmp_path, arm="protenix"):
    plan = plan_for(tmp_path)
    output = tmp_path / "new/validate/payload/study"
    session_dir = output / "native_gradient_checks-prospective/reference-worker"
    config = plan["reference_configuration"]["arms"][arm]
    command = config["command_prefix"][:]
    if arm == "protenix": command += ["--output", str(session_dir)]
    identity = {"status": "ready", "frozen": {"model": "model-hash", "context": "context-hash"},
                "identity": {"own_source_manifest_sha256": "f" * 64},
                "optimizer_updates": 0,
                "precision": {"full_model_float64": True, "parameters_float64": True,
                              "native_features_float64": True, "native_operations_float64": True,
                              "final_reduction_only": False}}
    audit = {"status": "passed_full_FP64_reference_audit", "floating_operations_float64": True,
             "narrow_float_avals": [], "frozen_unchanged": True, "original_results_changed": False,
             "optimizer_updates": 0} if arm == "protenix" else {
                "status": "passed_full_operation_FP64_audit", "narrow_floating_operations": 0,
                "production_execution_changed": False}
    residency = {"status": "passed_live_production_reference_co_residency", "input_FP32_baseline_unchanged": True,
                 "production_optimizer_updates": 0, "reference_closed_before_return": True, "no_runtime_fallback": True}
    for field in ("production_measurements", "frozen"):
        for when in ("before", "during", "after"):
            residency[field + "_" + when] = {"sha256": "unchanged-production-" + field}
    values = {"invocation": {"argv": command, "environment": {}, "optimizer_updates": 0},
              "identity": identity, "close": {"status": "closed", "frozen": deepcopy(identity["frozen"]), "audit": audit},
              "residency": residency}
    raw = put(session_dir / "point-client-0000.json", {"scope": "toy raw payload binding; no native result"})
    response = put(session_dir / "response-0000.json", {"id": 0, "graph_sha256": "a" * 64})
    index = put(session_dir / "client-raw-point-index.json", {"status": "complete_actual_reference_points",
        "requests": 1, "points": [{"id": 0, "gradient": True, "graph_sha256": "a" * 64,
                                  "key_data": [0, 17], "raw": raw, "response": response}],
        "optimizer_updates": 0, "input_transport": "exact_current_FP64_graph_and_actual_native_key",
        "reference_frozen": identity["frozen"], "cached_or_interpolated_points": 0})
    values["invocation"].update(client_raw_point_index_path=index["path"], key_data=[0, 17])
    values["residency"]["reference_client_raw_points"] = index
    session = {name: put(session_dir / (name + ".json"), value) for name, value in values.items()}
    return plan, output, config, values, {"reference_session": session}


@pytest.mark.parametrize("arm", ["protenix", "dbmol"])
def test_live_reference_session_requires_closed_full_precision_and_original_co_residency(tmp_path, arm):
    plan, output, config, values, record = session_fixture(tmp_path, arm)
    assert d._session(record, output, config) == record["reference_session"]


@pytest.mark.parametrize("failure", ["changed_bytes", "different_command", "environment", "reference_updates",
                                    "close_missing", "close_frozen", "narrow_reference", "production_changed",
                                    "input_changed", "optimizer_updates", "runtime_fallback", "reference_not_closed"])
def test_live_reference_session_failures_remain_blocking(tmp_path, failure):
    plan, output, config, values, record = session_fixture(tmp_path)
    if failure == "changed_bytes":
        Path(record["reference_session"]["identity"]["path"]).write_text("changed", encoding="utf8")
    else:
        if failure == "different_command": values["invocation"]["argv"][0] = "/unbound/python"
        elif failure == "environment": values["invocation"]["environment"] = {"FALLBACK": "1"}
        elif failure == "reference_updates": values["identity"]["optimizer_updates"] = 1
        elif failure == "close_missing": values["close"]["status"] = "not_closed"
        elif failure == "close_frozen": values["close"]["frozen"]["model"] = "changed"
        elif failure == "narrow_reference": values["close"]["audit"]["narrow_float_avals"] = ["float32"]
        elif failure == "production_changed": values["residency"]["production_measurements_during"]["sha256"] = "changed"
        elif failure == "input_changed": values["residency"]["input_FP32_baseline_unchanged"] = False
        elif failure == "optimizer_updates": values["residency"]["production_optimizer_updates"] = 1
        elif failure == "runtime_fallback": values["residency"]["no_runtime_fallback"] = False
        else: values["residency"]["reference_closed_before_return"] = False
        for name, value in values.items():
            record["reference_session"][name] = put(Path(record["reference_session"][name]["path"]), value)
    with pytest.raises(ValueError):
        d._session(record, output, config)


def test_actual_entry_adapter_restores_private_metadata_and_factory_after_error(tmp_path):
    plan = plan_for(tmp_path)
    driver = SimpleNamespace(VERSION="old-result", HOOK_VERSION="old-hook", READINESS_VERSION="old-ready",
        endpoint_policy_version=lambda: "old-policy", verify_dbmol_native_report=lambda value: value,
        assessment=lambda *args, **kwargs: None, failed_assessment=lambda *args, **kwargs: {}, EndpointHooks=object())
    originals = driver.__dict__.copy()
    @contextmanager
    def install(current, declared, configuration):
        assert current is driver and declared is plan and configuration == plan["reference_configuration"]
        old = current.EndpointHooks
        current.EndpointHooks = "reference-factory"
        try: yield
        finally: current.EndpointHooks = old
    hooks = SimpleNamespace(install_precision_hooks=install)
    with pytest.raises(RuntimeError, match="original native failure"):
        with d.adapted_driver(driver, plan, hooks):
            assert driver.VERSION == d.RESULT_VERSION
            assert driver.endpoint_policy_version() == d.POLICY_VERSION
            assert driver.EndpointHooks == "reference-factory"
            raise RuntimeError("original native failure")
    assert driver.__dict__ == originals


def test_actual_argv_builder_is_inherited_and_cannot_enable_boltz(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    calls = []
    inherited = SimpleNamespace(build_original_argv=lambda *args: calls.append(args) or ["original-command"])
    monkeypatch.setattr(d, "load_base", lambda value: inherited)
    assert d.build_original_argv(plan, "protenix", "validate", 20000, "output", "snapshot") == ["original-command"]
    assert calls == [(plan, "protenix", "validate", 20000, "output", "snapshot", None)]
    plan["dispatch_arms"].append("boltz")
    with pytest.raises(ValueError): d.build_original_argv(plan, "boltz", "segment", 10000, "output", "snapshot")
    assert len(calls) == 1


def test_real_main_is_delegated_without_forking_original_numerical_driver():
    tree = ast.parse(PATH.read_bytes())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    assert sum(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
               and node.func.value.id == "base" and node.func.attr == "main" for node in calls) == 1
    top_imports = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert not top_imports & {"numpy", "jax", "torch", "equinox", "mosaic", "rdkit"}


def test_metadata_entry_args_cannot_duplicate_source_or_plan(tmp_path):
    assert d._argument(["--plan", "authentic.json"], "--plan") == "authentic.json"
    for argv in ([], ["--plan"], ["--plan", "a", "--plan", "b"]):
        with pytest.raises(ValueError): d._argument(argv, "--plan")


def test_authenticated_parent_driver_and_packaged_copy_must_match(tmp_path):
    code = b'''VERSION="endpoint-aware-continuation-result-v1"\nHOOK_VERSION="prospective-endpoint-driver-hook-v1"\ndef main(): return "actual-original-main"\ndef build_original_argv(*args): return args\n'''
    historical = tmp_path / "historical.py"
    sibling = tmp_path / "scripts/continuation_driver.py"
    sibling.parent.mkdir()
    historical.write_bytes(code)
    sibling.write_bytes(code)
    plan = {"base_driver": {"path": str(historical), "bytes": len(code), "sha256": hashlib.sha256(code).hexdigest()}}
    assert d.load_base(plan, directory=sibling.parent).main() == "actual-original-main"
    sibling.write_bytes(code + b"# changed packaged copy\n")
    with pytest.raises(ValueError, match="byte-identical"):
        d.load_base(plan, directory=sibling.parent)


def passed_result_fixture(tmp_path):
    plan, output, config, values, record = session_fixture(tmp_path)
    plan["base_driver"] = put(tmp_path / "historical.json", {"source": "preserved-driver"})
    report = {"version": d.PRO_POLICY, "status": "passed", "can_admit_new_child": True,
              "frozen_unchanged": True, "can_certify_original_study": False}
    binding = put(output / "native_gradient_checks-prospective/admission.json", report)
    endpoint = {"stage": "native_gradient_checks", "objective": "protenix_contact_loss",
                "endpoint_policy": d.PRO_POLICY, "path": binding["path"], "status": "passed",
                "can_admit_new_child": True, **record}
    summary = put(output / "summary.json", {"status": "passed", "job_id": "42",
                                           "runtime_gate": {"status": "passed", "conservative_projected_seconds": 10000}})
    raw_proofs = d._raw_reference_points(record["reference_session"], output)[1]
    proofs = [binding, summary, *record["reference_session"].values(), *raw_proofs]
    snapshot = {"path": "/preserved/checkpoint.npz", "bytes": 12, "sha256": "e" * 64}
    result = {"version": d.RESULT_VERSION, "status": "passed", "arm": "protenix", "phase": "validate", "offset": 20000,
              "job_id": "42", "source_manifest_sha256": "f" * 64, "plan_sha256": plan["_sha256"],
              "scientific_source_manifest_sha256": "a" * 64, "config_sha256": "b" * 64, "global_end": 20000,
              "endpoint_policy": d.POLICY_VERSION, "driver_hook": d.HOOK_VERSION,
              "readiness_adapter": d.READINESS_VERSION, "precision_adapter_version": d.ADAPTER_VERSION,
              "policy_by_objective": deepcopy(d.POLICIES["protenix"]), "base_driver": plan["base_driver"],
              "original_results_changed": False, "original_failed_parent_preserved": True, "frozen_unchanged": True,
              "checks": {key: True for key in d.CHECKS}, "input_snapshot": snapshot, "original_summary": summary,
              "endpoint_records": [endpoint], "proofs": proofs}
    result["precision_policy_records"] = d.policy_records([endpoint], "protenix", output, plan)
    result["precision_runtime_budget"] = d.runtime_budget(Path(summary["path"]))
    return plan, result, snapshot


@pytest.mark.parametrize("failure", ["raw_masquerade", "old_result_version", "lost_nonraw_check", "missing_session_proof",
                                    "missing_endpoint", "duplicate_endpoint", "checkpoint", "global_step", "failed_summary"])
def test_controller_precision_result_rejects_incomplete_or_wrong_boundary_admission(tmp_path, failure):
    plan, result, snapshot = passed_result_fixture(tmp_path)
    assert d.verify_precision_result(result, plan, "f" * 64, "validate", "protenix", 20000, snapshot) is result
    if failure == "raw_masquerade": result["endpoint_policy"] = d.RAW_POLICY
    elif failure == "old_result_version": result["version"] = "endpoint-aware-continuation-result-v1"
    elif failure == "lost_nonraw_check": result["checks"]["hard_parity"] = False
    elif failure == "missing_session_proof": result["proofs"] = result["proofs"][:-1]
    elif failure == "missing_endpoint": result["endpoint_records"] = []
    elif failure == "duplicate_endpoint":
        result["endpoint_records"] *= 2
        result["precision_policy_records"] *= 2
    elif failure == "checkpoint": result["input_snapshot"] = {**snapshot, "sha256": "different"}
    elif failure == "global_step": result["global_end"] = 30000
    else:
        result["original_summary"] = put(Path(result["original_summary"]["path"]), {"status": "failed", "job_id": "42"})
    with pytest.raises(ValueError):
        d.verify_precision_result(result, plan, "f" * 64, "validate", "protenix", 20000, snapshot)


@pytest.mark.parametrize("projection", [0, False, float("inf"), float("nan"), 71400, 80000])
def test_reference_runtime_is_reserved_without_reducing_original_guard(tmp_path, projection):
    path = tmp_path / "summary.json"
    # Nonfinite test documents are deliberately malformed input metadata.
    path.write_text(json.dumps({"runtime_gate": {"status": "passed", "conservative_projected_seconds": projection}}), encoding="utf8")
    with pytest.raises(ValueError): d.runtime_budget(path)


def test_reference_runtime_budget_preserves_both_endpoint_allowances(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": {"status": "passed", "conservative_projected_seconds": 50000}})
    value = d.runtime_budget(path)
    assert value["required_total_seconds"] == 60800 and value["reference_seconds"] == 10800
    assert value["limit_seconds"] == 82200 and value["original_runtime_guard_changed"] is False


@pytest.mark.parametrize("name", ["assessment.json", "point-0000.npz", "stdout.log", "native-dtype-audit.json"])
def test_zero_byte_scientific_proofs_are_never_accepted_as_empty_raw_stderr(tmp_path, name):
    output = tmp_path / "study"
    path = output / "reference-session" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    proof = {"path": str(path), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    with pytest.raises(ValueError): d.check_child_proof(proof, output)


def test_honest_empty_owned_reference_stderr_is_preserved_not_fabricated(tmp_path):
    output = tmp_path / "study"
    path = output / "native-gradient-checks-prospective/reference-session/stderr.log"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    proof = {"path": str(path), "bytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    assert d.check_child_proof(proof, output) == path
    with pytest.raises(ValueError): d.check_child_proof(proof, tmp_path / "another-study")
