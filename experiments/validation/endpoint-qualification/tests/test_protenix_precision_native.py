"""Allocated metadata and JAX dtype-audit contracts for the precision gate.

No pretrained model is loaded here. The small numerical dtype controls run only
in the new allocated CPU capsule with x64 enabled; local inspection is AST only.
"""
from copy import deepcopy
import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/protenix_precision_native.py"
SPEC = importlib.util.spec_from_file_location("precision_native_fixture", PATH)
n = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = n
SPEC.loader.exec_module(n)
PIN = "a" * 64


@pytest.fixture(autouse=True)
def allocated_only():
    assert os.environ.get("LSB_JOBID"), "Native harness contracts require allocated DTU HPC"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf8")
    return n.binding(path)


@pytest.fixture
def owned(tmp_path, monkeypatch):
    root, source, forensic = tmp_path / "new-run", tmp_path / "new-release/source", tmp_path / "old/source"
    root.mkdir(); source.mkdir(parents=True); forensic.mkdir(parents=True)
    monkeypatch.setattr(n, "ROOT", root); monkeypatch.setattr(n, "SOURCE", source)
    monkeypatch.setattr(n, "FORENSICS", forensic)
    arm = {"id": "protenix", "rate": .3, "start_global": 20000, "target_global": 50000,
           "config": {"path": "/original/config", "bytes": 1733, "sha256": PIN},
           "parent_snapshot": {"path": "/original/step-10000.npz", "bytes": 14284, "sha256": PIN},
           "original_argv": ["python", "/original/driver.py", "--mode", "run"]}
    prior = {"arm": arm, "output": "/prior/gpu/payload"}
    old = save(forensic / "plan.json", prior)
    monkeypatch.setattr(n, "FORENSICS_PLAN_SHA", old["sha256"])
    report = save(forensic / "report.json", {"points": [], "original_failures_preserved": True})
    monkeypatch.setattr(n, "PRIOR_REPORT", report)
    runtime = [save(forensic / name, {"fixed": name}) for name in ("jax.json", "native.json", "backend.json")]
    monkeypatch.setattr(n, "RUNTIME_FILES", runtime)
    plan = {"version": n.VERSION, "source_root": str(source), "control_root": str(root),
            "original_results_changed": False, "optimization_settings_changed": False,
            "can_certify_or_restart_original_study": False, "monitor_remains_paused": True,
            "max_bytes": n.CAP, "cap_bytes": n.CAP, "metadata_cap_bytes": 16 * 1024**2,
            "internal_seconds": 5400, "output": str(root / "gpu/payload"),
            "original_forensics_plan": old, "arm": deepcopy(arm), "runtime_files": deepcopy(runtime),
            "prior_forensic_report": report, "original_directions": deepcopy(n.ORIGINAL_DIRECTIONS)}
    plan_path = source / "plan.json"; save(plan_path, plan)
    return root, source, forensic, plan, prior


def arguments(owned, *, startup=False, worker=None):
    root, source, _, _, _ = owned
    output = root / ("cpu/payload/native-startup" if startup else "gpu/payload")
    result = ["--plan", str(source / "plan.json"), "--plan-sha256", n.sha(source / "plan.json"),
              "--output", str(output)]
    if startup: result.append("--startup-only")
    if worker: result += ["--worker", worker]
    return result


def test_exact_scientific_plan_allows_only_new_validation_output(owned):
    root, _, _, plan, prior = owned
    assert n.validate_plan(plan, root / "gpu/payload") == prior
    assert n.validate_plan(plan, root / "cpu/payload/native-startup", True) == prior


@pytest.mark.parametrize("change", ["version", "source", "root", "rate", "snapshot", "argv", "target", "output",
                                    "results", "settings", "certify", "monitor", "cap", "seconds", "bool_cap", "old_path", "old_digest",
                                    "runtime", "directions", "report"])
def test_scope_mutation_never_changes_original_or_admits_an_optimizer(owned, change):
    root, _, _, plan, _ = owned
    changed = deepcopy(plan)
    if change == "version": changed["version"] += "-other"
    elif change == "source": changed["source_root"] += "/other"
    elif change == "root": changed["control_root"] += "/other"
    elif change == "rate": changed["arm"]["rate"] = .03
    elif change == "snapshot": changed["arm"]["parent_snapshot"]["sha256"] = "b" * 64
    elif change == "argv": changed["arm"]["original_argv"].append("--lower-lr")
    elif change == "target": changed["arm"]["target_global"] = 60000
    elif change == "output": changed["output"] = "/original/saved/study"
    elif change == "results": changed["original_results_changed"] = True
    elif change == "settings": changed["optimization_settings_changed"] = True
    elif change == "certify": changed["can_certify_or_restart_original_study"] = True
    elif change == "monitor": changed["monitor_remains_paused"] = False
    elif change == "cap": changed["max_bytes"] += 1
    elif change == "seconds": changed["internal_seconds"] += 1
    elif change == "bool_cap": changed["max_bytes"] = True
    elif change == "old_path": changed["original_forensics_plan"]["path"] += ".copy"
    elif change == "old_digest": changed["original_forensics_plan"]["sha256"] = "b" * 64
    elif change == "runtime": changed["runtime_files"] = changed["runtime_files"][:-1]
    elif change == "directions": changed["original_directions"]["X"][0]["row"] = 4
    else: changed["prior_forensic_report"]["sha256"] = "b" * 64
    with pytest.raises(ValueError): n.validate_plan(changed, root / "gpu/payload")


@pytest.mark.parametrize("path,startup", [("gpu/payload", True), ("cpu/payload/native-startup", False),
                                           ("gpu/other", False), ("cpu/payload/other", True)])
def test_real_cli_scope_rejects_wrong_mode_output(owned, path, startup):
    root, _, _, plan, _ = owned
    with pytest.raises(ValueError, match="output scope"):
        n.validate_plan(plan, root / path, startup)


def test_original_plan_replacement_detected_before_any_model_import(owned):
    root, _, forensic, plan, _ = owned
    (forensic / "plan.json").write_text('{"arm":{}}\n')
    with pytest.raises(ValueError, match="bytes differ"):
        n.validate_plan(plan, root / "gpu/payload")


def test_actual_startup_command_accepts_metadata_then_stops_before_numerical_import(owned, monkeypatch):
    root, source, _, _, _ = owned
    (root / "cpu/payload").mkdir(parents=True)
    calls = []
    def checked(plan, output, startup=False):
        calls.append((output, startup)); assert startup
        assert "jax" not in sys.modules or not getattr(sys.modules["jax"], "_model_loaded_for_this_fixture", False)
        return (None,) * 7
    monkeypatch.setattr(n, "verify", checked)
    monkeypatch.setenv("LSB_JOBID", "111"); monkeypatch.setenv("LSB_QUEUE", "hpc")
    result = n.main(arguments(owned, startup=True))
    assert calls == [(root / "cpu/payload/native-startup", True)]
    assert result == {"status": "passed_metadata_only", "numerical_setup_entered": False, "updates": 0,
                      "plan_sha256": n.sha(source / "plan.json"), "job_id": "111"}
    assert n.read(root / "cpu/payload/native-startup/endpoint-startup.json") == result


@pytest.mark.parametrize("change", ["worker", "queue", "allocation", "plan_sha"])
def test_actual_startup_rejects_worker_bad_queue_or_wrong_plan(owned, monkeypatch, change):
    monkeypatch.setattr(n, "verify", lambda *a, **k: pytest.fail("Rejected command entered verification"))
    monkeypatch.setenv("LSB_JOBID", "111" if change != "allocation" else "")
    monkeypatch.setenv("LSB_QUEUE", "gpul40s" if change == "queue" else "hpc")
    argv = arguments(owned, startup=True, worker="fp32" if change == "worker" else None)
    if change == "plan_sha": argv[argv.index("--plan-sha256") + 1] = PIN
    with pytest.raises(ValueError): n.main(argv)


def test_actual_parser_rejects_unknown_optimizer_options(owned):
    with pytest.raises(SystemExit): n.main(arguments(owned, startup=True) + ["--lr", ".03"])


@pytest.mark.parametrize("worker", ["fp32", "fp64"])
def test_worker_command_has_only_declared_fresh_phase_and_bound_inputs(owned, worker):
    root, source, _, _, _ = owned
    pin = n.sha(source / "plan.json")
    command = n.worker_command(source / "plan.json", pin, root / "gpu/payload", worker)
    assert command == [n.PYTHON, str(source / "scripts/protenix_precision_native.py"), "--plan", str(source / "plan.json"),
                       "--plan-sha256", pin, "--output", str(root / "gpu/payload"), "--worker", worker]
    assert not set(command) & {"--lr", "--resume", "--train", "--optimize"}


def test_undeclared_worker_is_impossible(owned):
    root, source, _, _, _ = owned
    with pytest.raises(ValueError): n.worker_command(source / "plan.json", PIN, root / "gpu/payload", "optimizer")


def variable(dtype):
    return SimpleNamespace(aval=SimpleNamespace(dtype=dtype))


def graph(dtype="float64", *, params=None, constants=()):
    arg = variable(dtype)
    eqn = SimpleNamespace(invars=(arg,), outvars=(arg,), primitive="fake_add", params=params or {})
    return SimpleNamespace(jaxpr=SimpleNamespace(constvars=(), invars=(arg,), outvars=(arg,), eqns=(eqn,)), consts=constants)


def test_jaxpr_dtype_walk_includes_nested_maps_sequences_constants_and_shared_cycles():
    nested = graph()
    params = {"scan": [nested, {"call": nested}], "ordinary_integer": 7}
    params["cycle"] = nested
    outer = graph(params=params, constants=(SimpleNamespace(dtype="float64"),))
    result = n.audit_jaxpr(outer)
    assert result["status"] == "passed" and result["floating_operations_float64"]
    assert result["aval_counts"]["float64"] > 0 and result["nested_graphs"] >= 4
    assert not result["narrow_float_avals"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "complex64"])
@pytest.mark.parametrize("where", ["input", "nested", "constant"])
def test_hidden_narrow_floating_operations_cannot_be_called_full_fp64(dtype, where):
    if where == "input": value = graph(dtype)
    elif where == "nested": value = graph(params={"call": (graph(dtype),)})
    else: value = graph(constants=(SimpleNamespace(dtype=dtype),))
    with pytest.raises(ValueError, match="non-FP64"):
        n.audit_jaxpr(value)


def test_integer_only_or_empty_jaxpr_cannot_claim_native_double_precision():
    with pytest.raises(ValueError): n.audit_jaxpr(graph("int32"))


def test_executable_jaxpr_self_alias_cannot_hide_valid_double_operations():
    value = graph()
    value.jaxpr.jaxpr = value.jaxpr
    result = n.audit_jaxpr(value)
    assert result["status"] == "passed" and result["aval_counts"]["float64"] > 0


def test_nested_jaxpr_self_alias_cannot_hide_narrow_operations():
    nested = graph("float32")
    nested.jaxpr.jaxpr = nested.jaxpr
    value = graph(params={"call": nested})
    with pytest.raises(ValueError, match="non-FP64"):
        n.audit_jaxpr(value)


def test_actual_nested_cpu_jaxpr_accepts_float64_and_rejects_downcast():
    # Runs only in the allocated CPU capsule; no model or pretrained inference.
    import jax
    import jax.numpy as jnp
    assert jax.config.jax_enable_x64, "Precision fixtures require declared CPU x64"
    x = jnp.asarray([.25, .5], dtype=jnp.float64)
    double = jax.make_jaxpr(jax.jit(lambda v: jnp.sin(v) + jnp.sum(v)))(x)
    assert n.audit_jaxpr(double)["status"] == "passed"
    narrow = jax.make_jaxpr(jax.jit(lambda v: v.astype(jnp.float32)))(x)
    with pytest.raises(ValueError, match="non-FP64"): n.audit_jaxpr(narrow)


def test_preservation_rehashes_exact_originals_and_detects_changed_bytes(owned):
    root, _, _, _, _ = owned
    original = root / "original.json"
    before = {str(original): save(original, {"failed": True})}
    assert n.preserved(before) == before
    original.write_text('{"passed":true}\n')
    with pytest.raises(ValueError, match="Protected original"):
        n.preserved(before)


def test_existing_proof_never_gets_rewritten(owned):
    root, _, _, _, _ = owned
    value = root / "immutable.json"; save(value, {"failed": True})
    with pytest.raises(FileExistsError): n.write(value, {"failed": False})


def test_coordinator_imports_no_numerical_module_before_worker_dispatch():
    tree = ast.parse(PATH.read_bytes())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import): top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom): top.add((node.module or "").split(".")[0])
    assert not top & {"jax", "numpy", "equinox", "mosaic", "torch", "rdkit"}


@pytest.mark.parametrize("failure", ["none", "protected", "frozen", "unloaded", "budget"])
def test_worker_finally_attempts_independent_preservation_checks_and_keeps_primary(owned, monkeypatch, failure):
    root, _, _, _, _ = owned
    proof = root / "original-proof.json"; row = save(proof, {"original": True})
    before = {str(proof): row}
    calls = []
    def frozen():
        calls.append("frozen")
        if failure == "frozen": raise ValueError("frozen after failed")
        return {"original": "same"}
    lifecycle = {} if failure == "unloaded" else {"frozen": frozen, "frozen_before": {"original": "same"}}
    if failure == "protected": proof.write_text('{"changed":true}\n')
    if failure == "budget": monkeypatch.setattr(n, "CAP", 1)
    primary = RuntimeError("real native failure")
    result = n.worker_preservation(root, "fp32", before, lifecycle, primary)
    assert result["primary_error"] == repr(primary)
    assert result["status"] == "failed_native_or_preservation"
    assert result["exception_chain"][0] == repr(primary)
    assert (root / "fp32-final-preservation.json").is_file()
    if failure != "unloaded": assert calls == ["frozen"]
    if failure in ("protected", "frozen", "budget"): assert result["preservation_errors"]


def test_worker_failure_never_masks_primary_even_when_aftercheck_also_fails(owned, monkeypatch):
    root, _, _, plan, _ = owned
    proof = root / "original-proof.json"; row = save(proof, {"original": True})
    primary = RuntimeError("original numerical failure")
    def failed(plan, output, lifecycle, protected):
        protected.update({str(proof): row})
        lifecycle.update(frozen=lambda: (_ for _ in ()).throw(ValueError("secondary frozen failure")), frozen_before={})
        raise primary
    monkeypatch.setattr(n, "_production", failed)
    with pytest.raises(RuntimeError, match="original numerical") as caught:
        n.run_worker(plan, root, "fp32")
    assert caught.value is primary
    final = n.read(root / "fp32-final-preservation.json")
    assert final["primary_error"] == repr(primary) and len(final["preservation_errors"]) == 1


def test_worker_success_requires_full_input_and_frozen_preservation(owned, monkeypatch):
    root, _, _, plan, _ = owned
    row = save(root / "proof.json", {"original": True})
    def good(plan, output, lifecycle, protected):
        protected[row["path"]] = row
        lifecycle.update(frozen=lambda: {"same": True}, frozen_before={"same": True})
    monkeypatch.setattr(n, "_production", good)
    assert n.run_worker(plan, root, "fp32")["status"] == "passed"


def fake_worker_artifacts(output, worker):
    directory = output / ("production" if worker == "fp32" else "reference")
    directory.mkdir()
    final = {"status": "passed", "worker": worker, "primary_error": None, "preservation_errors": [],
             "protected_inputs_unchanged": True, "frozen_unchanged": True}
    save(output / (worker + "-final-preservation.json"), final)
    if worker == "fp32":
        artifact = save(directory / "data.json", {"standin": "metadata fixture"})
        assessment = {"status": "passed_fp32_measurements", "updates": 0, "data": artifact,
                      "production_x64_enabled": False, "separate_process": True}
    else:
        admission = save(directory / "admission.json", {"status": "passed", "can_admit_new_child": True,
                                                       "frozen_unchanged": True})
        oracle = save(directory / "oracle.json", {"cached_native_points": "metadata fixture"})
        assessment = {"status": "passed_new_prospective_precision_measurement", "updates": 0,
                      "admission": admission, "oracle": oracle, "original_results_changed": False,
                      "can_certify_or_restart_original_study": False}
    save(directory / "assessment.json", assessment)
    return directory, assessment, final


@pytest.mark.parametrize("worker", ["fp32", "fp64"])
@pytest.mark.parametrize("change", ["none", "status", "updates", "frozen", "binding", "original_scope"])
def test_zero_exit_is_insufficient_without_exact_raw_worker_assessment(owned, worker, change):
    root, _, _, _, _ = owned
    directory, assessment, final = fake_worker_artifacts(root, worker)
    if change == "status": assessment["status"] = "failed"
    elif change == "updates": assessment["updates"] = 1
    elif change == "frozen": final["frozen_unchanged"] = False
    elif change == "binding": assessment["data" if worker == "fp32" else "admission"]["sha256"] = "b" * 64
    elif change == "original_scope":
        if worker == "fp32": assessment["production_x64_enabled"] = True
        else: assessment["can_certify_or_restart_original_study"] = True
    save(directory / "assessment.json", assessment)
    save(root / (worker + "-final-preservation.json"), final)
    if change == "none": assert set(n.verified_worker(root, worker)) == {"assessment", "preservation"}
    else:
        with pytest.raises(ValueError): n.verified_worker(root, worker)


def coordinator_setup(owned, monkeypatch):
    root, _, _, _, _ = owned
    (root / "gpu").mkdir()
    row = save(root / "original-proof.json", {"original": "intact"})
    native = SimpleNamespace(allocated_environment=lambda env: None)
    monkeypatch.setattr(n, "original", lambda: native)
    monkeypatch.setattr(n, "verify", lambda *a, **k: (native, None, None, None, None, None, {row["path"]: row}))
    monkeypatch.setenv("LSB_JOBID", "111"); monkeypatch.setenv("LSB_QUEUE", "gpul40s")
    return root


def test_coordinator_separates_fp32_fp64_processes_with_verified_exit_between_them(owned, monkeypatch):
    root = coordinator_setup(owned, monkeypatch)
    calls = []
    def run(command, **kwargs):
        worker = command[-1]
        assert kwargs["env"]["JAX_ENABLE_X64"] == ("0" if worker == "fp32" else "1")
        assert 0 < kwargs["timeout"] <= 5400
        if worker == "fp64": assert len(calls) == 1 and calls[0] == "fp32"
        fake_worker_artifacts(root / "gpu/payload", worker)
        calls.append(worker)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(n.subprocess, "run", run)
    result = n.main(arguments(owned))
    assert calls == ["fp32", "fp64"]
    assert result["status"] == "prospective_precision_gate_passed_original_failures_preserved"
    assert not result["production_optimizer_executed"] and result["frozen_unchanged"]
    assert set(result["worker_proofs"]) == {"fp32", "fp64"}


def test_fp32_failure_blocks_reference_and_keeps_original_inputs(owned, monkeypatch):
    root = coordinator_setup(owned, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command[-1]); return SimpleNamespace(returncode=1)
    monkeypatch.setattr(n.subprocess, "run", run)
    with pytest.raises(ValueError, match="Isolated fp32"):
        n.main(arguments(owned))
    assert calls == ["fp32"]
    assessment = n.read(root / "gpu/payload/assessment.json")
    assert assessment["status"] == "failed" and assessment["updates"] == 0
    assert (root / "gpu/payload/protected-after.json").is_file()


def test_coordinator_preserves_primary_worker_failure_if_input_aftercheck_fails(owned, monkeypatch):
    root = coordinator_setup(owned, monkeypatch)
    def run(command, **kwargs):
        (root / "original-proof.json").write_text('{"bad":true}\n')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(n.subprocess, "run", run)
    with pytest.raises(ValueError, match="Isolated fp32"):
        n.main(arguments(owned))
    assessment = n.read(root / "gpu/payload/assessment.json")
    assert "Isolated fp32" in assessment["primary_error"]
    assert len(assessment["preservation_errors"]) == 1


def test_shared_runtime_expiry_never_starts_a_second_worker(owned, monkeypatch):
    root = coordinator_setup(owned, monkeypatch)
    clock = iter([0.0, 0.0, 5401.0, 5402.0])
    monkeypatch.setattr(n.time, "monotonic", lambda: next(clock))
    calls = []
    def run(command, **kwargs):
        calls.append(command[-1]); fake_worker_artifacts(root / "gpu/payload", command[-1])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(n.subprocess, "run", run)
    with pytest.raises(ValueError, match="runtime exhausted"):
        n.main(arguments(owned))
    assert calls == ["fp32"]
