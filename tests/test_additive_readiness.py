"""Allocated CPU metadata/binding fixtures; no native scientific approval."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("_additive_readiness_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/additive_readiness.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sealed_plan(tmp_path):
    cases = [{"id": label, "parent_run": str(tmp_path / "old" / label), "arm": "fixed_initial_mask", "step": 15,
              "parent_artifacts_sha256": {str(tmp_path / "old" / label / "run.json"): "a" * 64}}
             for label in observer.CASE_IDS]
    original = {"version": 1, "name": "mechanism", "output_root": str(tmp_path / "diagnostic"),
        "production_release": str(tmp_path / "releases/old105"), "production_archive_sha256": observer.SOURCE105_ARCHIVE,
        "production_manifest_sha256": observer.SOURCE105_MANIFEST, "cases": cases}
    reference_path = tmp_path / "mechanism-plan.json"
    write(reference_path, original)
    plan = {**original, "name": "additive-readiness", "output_root": str(tmp_path / "readiness"),
        "fd_policy": observer.POLICY, "objective_arithmetic": observer.ARITHMETIC,
        "diagnostic_reference": {"plan": {"path": str(reference_path), "sha256": observer.helper()._sha(reference_path)},
                                 "output_root": original["output_root"]}}
    path = tmp_path / "plan.json"
    write(path, plan)
    binding = {"plan": {"path": str(path), "sha256": observer.helper()._sha(path)}, "output_root": plan["output_root"]}
    return plan, binding, original


def test_plan_requires_same_source105_states_explicit_policy_and_reference(tmp_path):
    plan, binding, _ = sealed_plan(tmp_path)
    assert observer.validate_readiness_binding(binding) == plan
    path = Path(binding["plan"]["path"])
    before = observer.helper()._read
    for update in ({"fd_policy": "branch_local_v2"}, {"objective_arithmetic": "native_v1"},
                   {"cases": plan["cases"][:1]}, {"silent_waiver": True},
                   {"production_manifest_sha256": "b" * 64}):
        write(path, {**plan, **update})
        with pytest.raises(ValueError): observer.load_plan(path, observer.helper()._sha(path))
    assert observer.helper()._read is before
    write(path, plan)
    with pytest.raises(ValueError, match="SHA256"): observer.load_plan(path, "b" * 64)
    with pytest.raises(ValueError): observer.validate_readiness_binding({**binding, "passed": True})
    with pytest.raises(ValueError): observer.validate_readiness_binding({**binding, "output_root": str(tmp_path / "other")})


def test_reference_plan_cannot_change_point_or_asset_bindings(tmp_path):
    plan, _, original = sealed_plan(tmp_path)
    changed = deepcopy(plan)
    changed["cases"][0]["parent_artifacts_sha256"][next(iter(changed["cases"][0]["parent_artifacts_sha256"]))] = "f" * 64
    with pytest.raises(ValueError, match="different historical"):
        observer.diagnostic_plan(plan["diagnostic_reference"], changed)
    assert observer.diagnostic_plan(plan["diagnostic_reference"], plan) == original


def test_module_and_package_bindings_restore_after_exception():
    import latentmol.boltz as package
    name = "latentmol.boltz._readiness_binding_fixture"
    old_module, old_attribute, temporary = ModuleType("old_module"), ModuleType("old_attribute"), ModuleType("new")
    sys.modules[name] = old_module
    setattr(package, "_readiness_binding_fixture", old_attribute)
    try:
        with pytest.raises(RuntimeError):
            with observer.module_bindings({name: temporary}):
                assert sys.modules[name] is temporary and package._readiness_binding_fixture is temporary
                raise RuntimeError("fixture")
        assert sys.modules[name] is old_module and package._readiness_binding_fixture is old_attribute
    finally:
        sys.modules.pop(name)
        delattr(package, "_readiness_binding_fixture")
    with observer.module_bindings({name: temporary}):
        assert sys.modules[name] is temporary
    assert name not in sys.modules and not hasattr(package, "_readiness_binding_fixture")


def test_actual_candidate_relative_imports_use_adjacent_bytes_and_restore():
    import latentmol.boltz as package
    old_module = sys.modules.get("latentmol.boltz.additive_fd")
    sentinel = object()
    old_attribute = getattr(package, "additive_fd", sentinel)
    with observer.candidate_modules() as modules:
        assert modules["additive_branch_fd"].additive_fd is modules["additive_fd"]
        assert package.additive_fd is modules["additive_fd"]
        assert Path(modules["additive_fd"].__file__).parent == Path(observer.__file__).parent
        assert modules["additive_branch_fd"].POLICY == observer.POLICY
    assert sys.modules.get("latentmol.boltz.additive_fd") is old_module
    assert getattr(package, "additive_fd", sentinel) is old_attribute


def test_forward_budget_counts_original_parity_dual_and_native_calls(monkeypatch):
    rows, report = [], {}
    counters = SimpleNamespace(call=lambda *args: rows.append(args[:3]) or args[3]())
    budget = observer.FullForwardBudget(counters, report)
    monkeypatch.setattr(observer, "MAXIMUM_FULL_FORWARDS_PER_CASE", 5)
    for action in ("forward", "native_forward", "forward_ad"):
        assert budget.call("boltz", "fixture", action, lambda: 4, "cpu") == 4
    assert report["actual_full_forward_reservations"] == 5
    assert budget.call("repair", "fixture", "backward", lambda: 7, "cpu") == 7
    with pytest.raises(ValueError, match="reservation exhausted"):
        budget.call("boltz", "fixture", "forward", lambda: 4, "cpu")
    assert len(rows) == 4


def readiness_fixture(tmp_path, monkeypatch):
    """Fabricate metadata to test the reader, never simulate native readiness."""
    from latentmol import experiments
    from latentmol.boltz.additive_contract import profile_record
    cfg, h = observer.config_identity(), observer.helper()
    plan, binding, diagnostic = sealed_plan(tmp_path)
    files = {name: "a" * 64 for name in observer.SOURCE_PATHS}
    identity = {"source_manifest_sha256": h._digest(files), "source_archive_sha256": "b" * 64}
    monkeypatch.setattr(observer, "SOURCE106_MANIFEST", identity["source_manifest_sha256"])
    monkeypatch.setattr(observer, "SOURCE106_ARCHIVE", identity["source_archive_sha256"])
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", identity["source_archive_sha256"])
    monkeypatch.setattr(experiments, "source_manifest", lambda _: {"sha256": identity["source_manifest_sha256"]})
    monkeypatch.setattr(h, "authenticated_artifact", lambda case, path: Path(path))
    monkeypatch.setattr(h, "authenticate_source", lambda *args: None)
    monkeypatch.setattr(h, "failed_item", lambda *args, **kwargs: ("combined", {}))
    @contextmanager
    def modules():
        yield {"additive_contract": SimpleNamespace(profile_record=profile_record)}
    monkeypatch.setattr(observer, "candidate_modules", modules)
    checked = []
    def verify(gate, profile, require_repair):
        checked.append((profile, require_repair))
        return gate.get("verified_fixture") is True
    monkeypatch.setattr(observer, "_verify_gate", verify)
    def record(name, config, job="29799999"):
        return {"status": "completed", "name": name, "config": config, "config_hash_version": cfg.CONFIG_HASH_VERSION,
            "config_sha256": cfg.config_sha256(config), "source": {"files": files, "sha256": identity["source_manifest_sha256"]},
            "source_archive_sha256": identity["source_archive_sha256"], "environment": {"job_id": job}}
    dbmol = {"profile": "dbmol_fixed_pose_v1", "objective_arithmetic": observer.ARITHMETIC, "affinity_regression_weight": 1., "affinity_logit_weight": 1.,
        "contact_weight": 2.5, "confidence_weight": 2.5, "confidence_weights":
        {"iptm": 1., "complex_iplddt": .3, "complex_ipde": .3, "complex_pde": .1, "confidence_score": .3, "ptm": .1}}
    original = {"fd_policy": "branch_local_v2", "dbmol": dbmol,
                "objective": {"output": "dbmol_total_loss", "direction": "minimize", "scale": 1., "offset": 0.}}
    parent, reference_parent = Path(plan["output_root"]) / "unique-parent", Path(diagnostic["output_root"]) / "unique-parent"
    rows, reference_rows = [], []
    for case in plan["cases"]:
        guided = case["id"] == "boltz_defog"
        weight = .1 if guided else 0.
        target = {"target_sha256": "f" * 64 if guided else None, "new_worker_calls": 0}
        identities = {key: "a" * 64 for key in ("decoder_state_sha256", "scorer_state_sha256", "point_sha256", "context_sha256")}
        gate = {"measured": {"loss": 1., "components": {"primitive": 2.}}, "point_sha256": identities["point_sha256"],
                "context_sha256": identities["context_sha256"], "active_slots": [0, 1], "effective_repair_weight": weight}
        gate_path = Path(case["parent_run"]) / case["arm"] / "gate-0015.json"
        write(gate_path, gate)
        write(Path(case["parent_run"]) / "run.json", {"config": original})
        configuration = {"case": case, "production_release": plan["production_release"],
            "production_archive_sha256": plan["production_archive_sha256"], "production_manifest_sha256": plan["production_manifest_sha256"],
            "effective_weight": weight, "target": target}
        reference = {"status": "completed", "diagnostic_completed": True, "frozen_state_unchanged": True, "parent_unchanged": True,
            "native_seed_parity": True, "original_gate_not_reclassified": True, "identities_before": identities,
            "identities_after": identities, "optimizer_updates": 0, "new_defog_calls": 0, "old_gate_certified_by_this_diagnostic": False,
            "baseline_graph_sha256": "a" * 64, "target": target, "directions": {"gradient": {}},
            "epsilons": [1e-5, 3e-5, 1e-4, 3e-4, .001, .003],
            "endpoint_rows": [{"direction": "gradient", "epsilon": epsilon, "mode": mode}
                for epsilon in (1e-5, 3e-5, 1e-4, 3e-4, .001, .003) for mode in ("natural", "held_readout")]}
        reference_child = reference_parent / case["id"] / "unique-child"
        write(reference_child / "run.json", record(case["id"], configuration, "29799998"))
        write(reference_child / "attribution.json", reference)
        reference_rows.append({"case_id": case["id"], "diagnostic_completed": True,
            "execution": {"returncode": 0, "timed_out": False, "run_paths": [str(reference_child)]}})
        child = parent / case["id"] / "unique-child"
        write(child / "run.json", record(case["id"], configuration))
        candidate = {**deepcopy(original), "fd_policy": observer.POLICY}
        proof = {**reference, "validation_passed": True, "candidate_gate_passed": True, "declared_policy_only_changed": True,
            "baseline_primal_and_gradient_exact": True, "original_config": original, "original_config_sha256": cfg.config_sha256(original),
            "candidate_config": candidate, "candidate_config_sha256": cfg.config_sha256(candidate),
            "candidate_source_files": files, "repair_target_available": guided, "effective_repair_weight": weight,
            "expected_profile": profile_record(dbmol, original["objective"], repair_weight=weight if guided else None),
            "branch_validation": {**gate, "verified_fixture": True}, "actual_full_forward_reservations": 10,
            "diagnostic_reference": plan["diagnostic_reference"], "old_gate": {"path": str(gate_path), "sha256": h._sha(gate_path)}}
        write(child / "attribution.json", proof)
        write(child / "oracle_cost.json", {"boltz": {"fixture": {"forward_attempted": 10}}})
        rows.append({"case_id": case["id"], "validation_completed": True, "validation_passed": True, "old_gate_certified": False,
            "execution": {"returncode": 0, "timed_out": False, "run_paths": [str(child)]}})
    write(reference_parent / "run.json", record(diagnostic["name"],
        {"plan": diagnostic, "plan_sha256": plan["diagnostic_reference"]["plan"]["sha256"]}, "29799998"))
    write(reference_parent / "cases.json", reference_rows)
    write(parent / "run.json", record(plan["name"], {"plan": plan, "plan_sha256": binding["plan"]["sha256"]}))
    write(parent / "cases.json", rows)
    write(parent / "validation.json", {"status": "completed", "validation_passed": True, "cases": rows,
        "plan_sha256": binding["plan"]["sha256"], "fd_policy": observer.POLICY, "objective_arithmetic": observer.ARITHMETIC,
        "diagnostic_reference": plan["diagnostic_reference"], "original_gates_not_reclassified": True,
        "optimizer_updates": 0, "new_defog_calls": 0})
    return binding, identity, parent, reference_parent, checked


def test_complete_paired_reader_binds_actual_profile_and_allocation(tmp_path, monkeypatch):
    binding, identity, _, _, checked = readiness_fixture(tmp_path, monkeypatch)
    proof = observer.require_readiness(binding, identity)
    assert proof["passed"] and len(proof["cases"]) == 2 and proof["allocation_job_id"] == "29799999"
    assert [requires for _, requires in checked] == [False, True]
    assert checked[0][0]["repair_weight"] is None and checked[1][0]["repair_weight"] == .1
    assert proof["diagnostic_reference"]["allocation_job_id"] == "29799998"


@pytest.mark.parametrize("mutation", ["failed_gate", "different_allocation", "changed_target", "changed_config",
    "changed_source", "changed_profile", "old_gate_scalar", "incomplete_reference", "missing_reference_endpoints",
    "ambiguous_child", "counter_disagreement", "missing_native_source"])
def test_reader_rejects_failed_stale_or_incomplete_native_proofs(tmp_path, monkeypatch, mutation):
    binding, identity, parent, reference_parent, _ = readiness_fixture(tmp_path, monkeypatch)
    child = parent / "boltz_defog" / "unique-child"
    path = child / "attribution.json"
    proof = json.loads(path.read_text())
    if mutation == "failed_gate": proof["branch_validation"]["verified_fixture"] = False
    if mutation == "changed_target": proof["target"]["target_sha256"] = "e" * 64
    if mutation == "changed_config": proof["candidate_config"]["dbmol"]["contact_weight"] = 3.
    if mutation == "changed_profile": proof["expected_profile"]["dbmol"]["contact_weight"] = 3.
    if mutation == "old_gate_scalar": proof["branch_validation"]["measured"]["loss"] = 2.
    if mutation == "missing_native_source": proof["candidate_source_files"].pop(observer.SOURCE_PATHS[0])
    write(path, proof)
    if mutation in ("different_allocation", "changed_source"):
        path = child / "run.json"
        record = json.loads(path.read_text())
        if mutation == "different_allocation": record["environment"]["job_id"] = "29799997"
        else: record["source_archive_sha256"] = "c" * 64
        write(path, record)
    if mutation in ("incomplete_reference", "missing_reference_endpoints"):
        path = reference_parent / "boltz_defog/unique-child/attribution.json"
        reference = json.loads(path.read_text())
        if mutation == "incomplete_reference": reference["diagnostic_completed"] = False
        else: reference["endpoint_rows"].pop()
        write(path, reference)
    if mutation == "ambiguous_child": write(parent / "boltz_defog/another-child/run.json", {})
    if mutation == "counter_disagreement": write(child / "oracle_cost.json", {"boltz": {"fixture": {"forward_attempted": 9}}})
    with pytest.raises(ValueError): observer.require_readiness(binding, identity)


def test_cli_refuses_unallocated_native_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setattr(sys, "argv", ["readiness", "--plan", str(tmp_path / "plan.json"), "--sha256", "a" * 64])
    with pytest.raises(ValueError, match="allocated DTU HPC"):
        observer.main()
