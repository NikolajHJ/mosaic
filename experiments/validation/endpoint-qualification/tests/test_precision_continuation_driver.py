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


def original_sources():
    return {"protenix": ("original-pro", 643, "a" * 64, "original.py", .3, 10000, 50000),
            "dbmol": ("original-db", 657, "b" * 64, "dbmol_driver.py", .0025, 2500, 10000),
            "boltz": ("original-bol", 635, "c" * 64, "original.py", .0025, 10000, 30000)}


def pro_runtime(projected):
    return {"status": "passed", "mean_update_seconds": [(projected - 1800) / 15000],
            "conservative_projected_seconds": projected}


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
        assessment=lambda *args, **kwargs: None, failed_assessment=lambda *args, **kwargs: {}, EndpointHooks=object(),
        startup_tree=lambda *args, **kwargs: None, execution_context=lambda *args: None,
        runpy=SimpleNamespace(run_path=lambda *args: None), SOURCES=original_sources())
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
            assert driver.SOURCES["protenix"][5] == 2500 and driver.SOURCES["dbmol"][5] == 1250
            raise RuntimeError("original native failure")
    assert driver.__dict__ == originals


def test_actual_argv_builder_is_inherited_and_cannot_enable_boltz(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    calls = []
    inherited = SimpleNamespace(build_original_argv=lambda *args: calls.append(args) or ["original-command"],
                                SOURCES=original_sources())
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
    summary = put(output / "summary.json", {"status": "passed", "job_id": "42", "runtime_gate": pro_runtime(10000)})
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
    result["precision_runtime_budget"] = d.runtime_budget(Path(summary["path"]), "protenix")
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


@pytest.mark.parametrize("projection", [0, False, float("inf"), float("nan"), 82800, 90000])
def test_reference_runtime_is_reserved_without_reducing_original_guard(tmp_path, projection):
    path = tmp_path / "summary.json"
    # Nonfinite test documents are deliberately malformed input metadata.
    runtime = pro_runtime(projection) if type(projection) in (int, float) else {"status": "passed", "conservative_projected_seconds": projection}
    path.write_text(json.dumps({"runtime_gate": runtime}), encoding="utf8")
    with pytest.raises(ValueError): d.runtime_budget(path, "protenix")


def test_reference_runtime_budget_preserves_both_endpoint_allowances(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": pro_runtime(50000)})
    value = d.runtime_budget(path, "protenix")
    assert value["original_projected_seconds"] == 50000
    assert value["bounded_segment_projected_seconds"] == 13850
    assert value["required_total_seconds"] == 25250 and value["reference_seconds"] == 10800
    assert value["walltime"] == "07:15" and value["walltime_seconds"] == 26100
    assert value["limit_seconds"] == 25500 and value["original_runtime_guard_changed"] is False


def test_dbmol_runtime_keeps_all_measured_fixed_overhead(tmp_path):
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": {"status": "passed", "mean_update_seconds": 2,
        "conservative_projected_seconds": 6850, "segment_updates": 1250,
        "observed_startup_entry_check_seconds": 1000, "observed_native_check_seconds": 200,
        "terminal_check_safety_factor": 1.5, "update_safety_factor": 1.5, "reserve_seconds": 1800}})
    value = d.runtime_budget(path, "dbmol")
    assert value["bounded_segment_projected_seconds"] == 6850
    assert value["required_total_seconds"] == 18250
    assert value["walltime"] == "05:15" and value["limit_seconds"] == 18300


def test_dbmol_segment_still_rejects_two_reference_checks_beyond_maximum_walltime(tmp_path):
    path = tmp_path / "summary.json"
    # The inherited 23-hour check passes; the complete two-reference budget does
    # not. Shorter production intervals cannot scale away this measured cost.
    put(path, {"runtime_gate": {"status": "passed", "mean_update_seconds": 2,
        "conservative_projected_seconds": 75000, "segment_updates": 1250,
        "observed_startup_entry_check_seconds": 69150, "observed_native_check_seconds": 200,
        "terminal_check_safety_factor": 1.5, "update_safety_factor": 1.5, "reserve_seconds": 1800}})
    with pytest.raises(ValueError): d.runtime_budget(path, "dbmol")


@pytest.mark.parametrize("change", ["status", "missing_means", "different_formula", "nonfinite_mean", "unknown_arm"])
def test_segment_walltime_never_comes_from_unpassed_or_incomplete_measurements(tmp_path, change):
    runtime = pro_runtime(50000)
    if change == "status": runtime["status"] = "failed"
    elif change == "missing_means": del runtime["mean_update_seconds"]
    elif change == "different_formula": runtime["mean_update_seconds"] = [1]
    elif change == "nonfinite_mean": runtime["mean_update_seconds"] = [float("inf")]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runtime_gate": runtime}), encoding="utf8")
    with pytest.raises(ValueError): d.runtime_budget(path, "missing" if change == "unknown_arm" else "protenix")


@pytest.mark.parametrize("field", ["segment_updates", "terminal_check_safety_factor", "update_safety_factor", "reserve_seconds",
                                   "observed_startup_entry_check_seconds", "observed_native_check_seconds", "conservative_projected_seconds"])
def test_dbmol_runtime_rejects_incomplete_or_rescaled_fixed_terms(tmp_path, field):
    runtime = {"status": "passed", "mean_update_seconds": 2, "conservative_projected_seconds": 6850,
               "segment_updates": 1250, "observed_startup_entry_check_seconds": 1000,
               "observed_native_check_seconds": 200, "terminal_check_safety_factor": 1.5,
               "update_safety_factor": 1.5, "reserve_seconds": 1800}
    runtime[field] = 0
    path = tmp_path / "summary.json"
    put(path, {"runtime_gate": runtime})
    with pytest.raises(ValueError): d.runtime_budget(path, "dbmol")


def test_shorter_segments_cover_only_requested_unsaved_future_once(tmp_path):
    plan = plan_for(tmp_path)
    for arm in plan["arms"]:
        intervals = [(offset, offset + arm["segment_updates"]) for offset in arm["offsets"]]
        assert intervals[0][0] == arm["start_global"] and intervals[-1][1] == arm["target_global"]
        assert all(left[1] == right[0] for left, right in zip(intervals, intervals[1:]))
    assert d.BOUNDARIES["protenix"] == (20000, 50000, 2500, list(range(20000, 50000, 2500)))
    assert d.BOUNDARIES["dbmol"] == (7500, 10000, 1250, [7500, 8750])


def test_private_source_step_override_restores_exact_original_science_even_on_failure():
    driver = SimpleNamespace(SOURCES=original_sources())
    original = driver.SOURCES
    with pytest.raises(RuntimeError):
        with d.short_segment_sources(driver):
            assert driver.SOURCES["protenix"] == (*original["protenix"][:5], 2500, 50000)
            assert driver.SOURCES["dbmol"] == (*original["dbmol"][:5], 1250, 10000)
            assert driver.SOURCES["boltz"] is original["boltz"]
            raise RuntimeError("preserved original error")
    assert driver.SOURCES is original


PRO_MAIN = '''def main():
    result = original_optimizer(loss, initial, steps=args.segment_steps, rate=0.3)
    if result["status"] != "passed" or len(result["updates"]) != 10000 or [r["step"] for r in result["states"]] != list(range(10001)):
        raise ValueError("Incomplete declared 10,000-update budget")
    terminal_gradient_checks(result)
'''
DB_MAIN = '''def main():
    case = original_optimizer(loss, initial, steps=a.segment_steps, rate=0.0025)
    if case['status'] != 'passed' or len(case['updates']) != a.segment_steps or len(case['states']) != a.segment_steps+1:
        raise ValueError("Incomplete declared segment")
    terminal_gradient_checks(case)
'''


def test_driver_completion_only_ast_adaptation_preserves_optimizer_and_terminal_checker():
    original = ast.parse(PRO_MAIN)
    changed = d.short_segment_driver_tree(PRO_MAIN, "original.py", "protenix")
    original_main, changed_main = original.body[0], changed.body[0]
    assert ast.dump(original_main.body[0]) == ast.dump(changed_main.body[0])
    assert ast.dump(original_main.body[2]) == ast.dump(changed_main.body[2])
    assert ast.dump(original_main.body[1].body[0]) == ast.dump(changed_main.body[1].body[0])
    assert "args.segment_steps + 1" in ast.unparse(changed_main.body[1].test)
    assert ast.dump(d.short_segment_driver_tree(DB_MAIN, "db.py", "dbmol")) == ast.dump(ast.parse(DB_MAIN))


def test_cpu_startup_and_production_compile_use_same_audited_completion_adapter(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    for row in plan['arms']:
        row['source']['path'] = str(tmp_path / row['id'])
    filename = tmp_path / 'protenix/examples/original.py'
    filename.parent.mkdir(parents=True)
    filename.write_text(PRO_MAIN, encoding='utf8')
    calls = []
    transform = d.short_segment_driver_tree
    def observe(raw, filename, arm):
        calls.append((str(filename), arm))
        return transform(raw, filename, arm)
    monkeypatch.setattr(d, 'short_segment_driver_tree', observe)
    @contextmanager
    def install(*args):
        yield
    driver = SimpleNamespace(VERSION='old', HOOK_VERSION='old', READINESS_VERSION='old',
        endpoint_policy_version=lambda: 'old', verify_dbmol_native_report=lambda value: value,
        assessment=lambda *args, **kwargs: None, failed_assessment=lambda *args, **kwargs: {},
        startup_tree=lambda raw, filename, **kwargs: ast.parse(raw), execution_context=lambda *args: None,
        runpy=SimpleNamespace(run_path=lambda *args: None), SOURCES=original_sources())
    with d.adapted_driver(driver, plan, SimpleNamespace(install_precision_hooks=install)):
        startup = driver.startup_tree(PRO_MAIN, str(filename))
        namespace = driver.runpy.run_path(str(filename), run_name='prospective_original_driver')
        assert isinstance(startup, ast.Module) and callable(namespace['main'])
        assert namespace['__file__'] == str(filename) and namespace['__name__'] == 'prospective_original_driver'
        assert namespace['main'].__code__.co_filename == str(filename)
        # Neither route has entered the fixture optimizer/model or any update.
        assert calls == [(str(filename), 'protenix'), (str(filename), 'protenix')]


@pytest.mark.parametrize("failure", ["different_count", "duplicate_guard", "missing_terminal_count", "different_db_guard"])
def test_short_segment_driver_rejects_unexpected_original_completion_shape(failure):
    raw, arm = PRO_MAIN, "protenix"
    if failure == "different_count": raw = raw.replace("!= 10000", "!= 9999")
    elif failure == "duplicate_guard": raw += "\n" + PRO_MAIN.replace("def main():", "def another():")
    elif failure == "missing_terminal_count": raw = raw.replace("range(10001)", "range(10000)")
    else: raw, arm = DB_MAIN.replace("len(case['states']) != a.segment_steps+1", "len(case['states']) != a.segment_steps"), "dbmol"
    with pytest.raises(ValueError): d.short_segment_driver_tree(raw, "original.py", arm)


def segment_scope(config, offset, steps, snapshot, digest):
    start = config['extension']['start_global_update']
    if (type(offset) is not int or type(steps) is not int or steps != 10000
            or offset not in range(start, 50000, 10000) or offset + steps > 50000):
        raise ValueError('Undeclared global segment range')
    if (offset > 0) != bool(snapshot and digest):
        raise ValueError('Every continuation needs exact predecessor checkpoint')
    return {'start': offset, 'end': offset + steps, 'target': 50000, 'updates': steps,
            'input_snapshot': snapshot, 'input_snapshot_sha256': digest}


def require_boundary(summary, assessment=None):
    execution = summary.get('extended_execution', {})
    if (execution.get('updates') != 10000 or execution.get('target') != 50000
            or summary.get('exact_boundary_replay') != 'passed'):
        raise ValueError('Extended boundary/placement replay is not authenticated')
    if assessment is not None and assessment.get('extended_execution') != execution:
        raise ValueError('Readiness global scope differs from actual execution')
    return execution


SEGMENT = 2500
TARGET = 10000


def scope(offset, steps, snapshot, digest):
    if not (type(offset) is int and offset in range(0, TARGET, SEGMENT) and steps == SEGMENT):
        raise ValueError('Undeclared segment')
    if (offset > 0) != bool(snapshot and digest):
        raise ValueError('Missing exact predecessor')
    return {'start': offset, 'end': offset + steps, 'updates': steps, 'target': TARGET}


def test_protocol_shortening_keeps_config_constant_exact_predecessor_target_and_replay_gates():
    module = SimpleNamespace(segment_scope=segment_scope, require_boundary=require_boundary)
    with d.short_segment_protocol(module, "protenix"):
        config = {'extension': {'start_global_update': 10000, 'segment_updates': 10000}}
        assert module.segment_scope(config, 22500, 2500, 'exact', 'digest')['end'] == 25000
        assert config['extension']['segment_updates'] == 10000
        for offset, steps, snapshot in [(20000, 10000, 'exact'), (21000, 2500, 'exact'), (47500, 5000, 'exact'), (20000, 2500, None)]:
            with pytest.raises(ValueError): module.segment_scope(config, offset, steps, snapshot, 'digest')
        with pytest.raises(ValueError): module.require_boundary({'extended_execution': {'updates': 2500, 'target': 50000}})
        summary = {'extended_execution': {'updates': 2500, 'target': 50000}, 'exact_boundary_replay': 'passed'}
        assert module.require_boundary(summary) == summary['extended_execution']
        with pytest.raises(ValueError): module.require_boundary(summary, {'extended_execution': {'updates': 10000}})
    assert module.segment_scope is segment_scope and module.require_boundary is require_boundary
    db = SimpleNamespace(scope=scope, SEGMENT=SEGMENT)
    with d.short_segment_protocol(db, "dbmol"):
        assert db.SEGMENT == 2500 and db.scope(8750, 1250, 'exact', 'digest')['end'] == 10000
        for offset, steps, snapshot in [(7500, 2500, 'exact'), (8000, 1250, 'exact'), (8750, 1250, None)]:
            with pytest.raises(ValueError): db.scope(offset, steps, snapshot, 'digest')
    assert db.scope is scope


@pytest.mark.parametrize("failure", ["different_steps", "different_range", "duplicate_range"])
def test_protocol_metadata_ast_rejects_changed_or_duplicated_original_shape(monkeypatch, failure):
    original = d.inspect.getsource(segment_scope)
    if failure == "different_steps": raw = original.replace('steps != 10000', 'steps != 9999')
    elif failure == "different_range": raw = original.replace('range(start, 50000, 10000)', 'range(start, 60000, 10000)')
    else: raw = original.replace("return {", "another = range(start, 50000, 10000)\n    return {")
    monkeypatch.setattr(d.inspect, 'getsource', lambda function: raw)
    with pytest.raises(ValueError): d.short_segment_protocol_function(segment_scope, 'protenix', 'segment_scope')


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
