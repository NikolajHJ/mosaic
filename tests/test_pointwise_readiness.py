"""Allocated CPU fixtures for metadata gates, not native scientific approval."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location("_pointwise_readiness_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/pointwise_readiness.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def sealed_plan(tmp_path, monkeypatch):
    """Small fabricated metadata tests authentication, never native inference."""
    h = observer.helper()
    identities = {key: "a" * 64 for key in
        ("point_sha256", "context_sha256", "decoder_state_sha256", "scorer_state_sha256")}
    dbmol = {"profile": "dbmol_fixed_pose_v1", "objective_arithmetic": observer.ARITHMETIC,
        "affinity_regression_weight": 1., "affinity_logit_weight": 1., "contact_weight": 2.5,
        "confidence_weight": 2.5, "confidence_weights": {"iptm": 1., "complex_iplddt": .3,
        "complex_ipde": .3, "complex_pde": .1, "confidence_score": .3, "ptm": .1}}
    original = {"fd_policy": "branch_local_v2", "dbmol": dbmol,
        "objective": {"output": "dbmol_total_loss", "direction": "minimize", "scale": 1., "offset": 0.}}
    cases, old_cases = [], deepcopy(observer.OLD_CASES)
    review = {"source_capsule_sha256": observer.SOURCE105_CAPSULE}
    for label in observer.CASE_IDS:
        run_id, _, key = old_cases[label]
        relative = "/".join(("source-105-contact-precision-branch-validation-attempt002-pilot", label, "optimization", run_id))
        parent = tmp_path / "runs" / relative
        gate = {"status": "failed", "passed": False, "native_seed_parity": True,
            "point_sha256": identities["point_sha256"], "context_sha256": identities["context_sha256"],
            "active_slots": [0, 1], "measured": {"loss": 1., "components": {"primitive": 2.},
                "repair_guidance": {"target_sha256": "f" * 64 if label == "boltz_defog" else None}},
            "effective_repair_weight": .1 if label == "boltz_defog" else 0.}
        gate_path, run_path = parent / "fixed_initial_mask/gate-0015.json", parent / "run.json"
        write(gate_path, gate)
        write(run_path, {"config": original, **identities})
        digest = h._sha(gate_path)
        old_cases[label] = (run_id, digest, key)
        # The archived review embeds local Windows paths; only the exact /runs/
        # suffix and actual gate bytes identify its remote scientific state.
        review[key] = {"path": "outputs\\hpc-results\\saved\\runs\\" + relative.replace("/", "\\")
            + "\\fixed_initial_mask\\gate-0015.json", "sha256": digest}
        cases.append({"id": label, "parent_run": str(parent), "arm": "fixed_initial_mask", "step": 15,
            "parent_artifacts_sha256": {str(path): h._sha(path) for path in (gate_path, run_path)}})
    monkeypatch.setattr(observer, "OLD_CASES", old_cases)
    review_path = tmp_path / "controls/existing-evidence-review.json"
    write(review_path, review)
    monkeypatch.setattr(observer, "EVIDENCE_REVIEW_SHA256", h._sha(review_path))
    plan = {"version": 1, "name": "focused-readiness", "output_root": str(tmp_path / "readiness"),
        "production_release": str(tmp_path / "releases/old105"), "production_archive_sha256": observer.SOURCE105_ARCHIVE,
        "production_manifest_sha256": observer.SOURCE105_MANIFEST, "cases": cases,
        "fd_policy": observer.POLICY, "objective_arithmetic": observer.ARITHMETIC,
        "evidence_review": {"path": str(review_path), "sha256": h._sha(review_path)}}
    path = tmp_path / "plan.json"
    write(path, plan)
    binding = {"plan": {"path": str(path), "sha256": h._sha(path)}, "output_root": plan["output_root"]}
    return plan, binding, original, identities


def test_plan_accepts_only_two_original_failures_and_pinned_review(tmp_path, monkeypatch):
    plan, binding, _, _ = sealed_plan(tmp_path, monkeypatch)
    assert observer.validate_readiness_binding(binding) == plan
    path, before = Path(binding["plan"]["path"]), observer.helper()._read
    for update in ({"fd_policy": "additive_branch_v3"}, {"objective_arithmetic": "native_v1"},
                   {"cases": plan["cases"][:1]}, {"silent_waiver": True},
                   {"diagnostic_reference": {}}, {"production_manifest_sha256": "b" * 64}):
        write(path, {**plan, **update})
        with pytest.raises(ValueError): observer.load_plan(path, observer.helper()._sha(path))
    assert observer.helper()._read is before
    write(path, plan)
    with pytest.raises(ValueError, match="SHA256"): observer.load_plan(path, "b" * 64)
    with pytest.raises(ValueError): observer.validate_readiness_binding({**binding, "passed": True})
    with pytest.raises(ValueError): observer.validate_readiness_binding({**binding, "output_root": str(tmp_path / "other")})
    changed = deepcopy(plan)
    changed["cases"][0]["parent_run"] += "-different"
    write(path, changed)
    with pytest.raises(ValueError, match="exact two"): observer.load_plan(path, observer.helper()._sha(path))


def test_changed_review_bytes_or_source_capsule_are_rejected(tmp_path, monkeypatch):
    plan, _, _, _ = sealed_plan(tmp_path, monkeypatch)
    path = Path(plan["evidence_review"]["path"])
    review = observer._read(path)
    write(path, {**review, "source_capsule_sha256": "b" * 64})
    with pytest.raises(ValueError, match="bytes"): observer.review_asset(plan["evidence_review"], plan["cases"])
    digest = observer.helper()._sha(path)
    monkeypatch.setattr(observer, "EVIDENCE_REVIEW_SHA256", digest)
    with pytest.raises(ValueError, match="another source105"):
        observer.review_asset({"path": str(path), "sha256": digest}, plan["cases"])


def test_native_forward_cap_includes_original_seed_calls(monkeypatch):
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
    """Fabricated metadata checks reader rejection, not scientific correctness."""
    from latentmol import experiments
    from latentmol.boltz.additive_contract import profile_record
    plan, binding, original, identities = sealed_plan(tmp_path, monkeypatch)
    h, cfg = observer.helper(), observer.config_identity()
    files = {name: "a" * 64 for name in observer.SOURCE_PATHS}
    identity = {"source_manifest_sha256": h._digest(files), "source_archive_sha256": "b" * 64}
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", identity["source_archive_sha256"])
    monkeypatch.setattr(experiments, "source_manifest", lambda _: {"sha256": identity["source_manifest_sha256"]})
    monkeypatch.setattr(h, "authenticate_source", lambda *args: None)
    monkeypatch.setattr(h, "failed_item", lambda *args, **kwargs: ("combined", {}))
    checked = []
    def verify(gate, *, require_repair, expected_profile):
        checked.append((expected_profile, require_repair))
        return gate.get("verified_fixture") is True
    @contextmanager
    def modules():
        yield {"pointwise_contract": SimpleNamespace(profile_record=profile_record),
               "integration_smoke": SimpleNamespace(full_gate_passed=verify)}
    monkeypatch.setattr(observer, "candidate_modules", modules)
    def record(name, config):
        return {"status": "completed", "name": name, "config": config, "config_hash_version": cfg.CONFIG_HASH_VERSION,
            "config_sha256": cfg.config_sha256(config), "source": {"files": files, "sha256": identity["source_manifest_sha256"]},
            "source_archive_sha256": identity["source_archive_sha256"], "environment": {"job_id": "29799999"}}
    parent, rows = Path(plan["output_root"]) / "unique-parent", []
    for case in plan["cases"]:
        guided = case["id"] == "boltz_defog"
        weight, target = (.1 if guided else 0.), {"target_sha256": "f" * 64 if guided else None}
        old_path = Path(case["parent_run"]) / case["arm"] / "gate-0015.json"
        gate = observer._read(old_path)
        configuration = {"case": case, "production_release": plan["production_release"],
            "production_archive_sha256": plan["production_archive_sha256"], "production_manifest_sha256": plan["production_manifest_sha256"],
            "effective_weight": weight, "target": target, "case_timeout_seconds": observer.CASE_TIMEOUT_SECONDS,
            "repeats_per_mode": 1}
        child = parent / case["id"] / "unique-child"
        write(child / "run.json", record(case["id"], configuration))
        candidate = {**deepcopy(original), "fd_policy": observer.POLICY}
        proof = {"status": "completed", "diagnostic_completed": True, "validation_passed": True,
            "candidate_gate_passed": True, "frozen_state_unchanged": True, "parent_unchanged": True,
            "native_seed_parity": True, "original_gate_not_reclassified": True, "declared_policy_only_changed": True,
            "baseline_primal_and_gradient_exact": True, "identities_before": identities, "identities_after": identities,
            "optimizer_updates": 0, "new_defog_calls": 0, "old_gate_certified_by_this_diagnostic": False,
            "original_config": original, "original_config_sha256": cfg.config_sha256(original),
            "candidate_config": candidate, "candidate_config_sha256": cfg.config_sha256(candidate),
            "candidate_source_files": files, "repair_target_available": guided, "effective_repair_weight": weight,
            "target": target, "expected_profile": profile_record(original["dbmol"], original["objective"],
                repair_weight=weight if guided else None), "baseline_graph_sha256": "a" * 64,
            "branch_validation": {**gate, "verified_fixture": True, "derivative_contract": {"graph_sha256": "a" * 64}},
            "actual_full_forward_reservations": 10, "evidence_review": plan["evidence_review"],
            "old_gate": {"path": str(old_path), "sha256": h._sha(old_path)}}
        write(child / "attribution.json", proof)
        write(child / "oracle_cost.json", {"boltz": {"fixture": {"forward_attempted": 10}}})
        rows.append({"case_id": case["id"], "validation_completed": True, "validation_passed": True,
            "old_gate_certified": False, "execution": {"returncode": 0, "timed_out": False, "run_paths": [str(child)]}})
    write(parent / "run.json", record(plan["name"], {"plan": plan, "plan_sha256": binding["plan"]["sha256"],
        "case_timeout_seconds": observer.CASE_TIMEOUT_SECONDS, "maximum_full_forwards_per_case": observer.MAXIMUM_FULL_FORWARDS_PER_CASE,
        "optimizer_updates": 0, "new_defog_calls": 0}))
    write(parent / "cases.json", rows)
    write(parent / "validation.json", {"status": "completed", "validation_passed": True, "cases": rows,
        "plan_sha256": binding["plan"]["sha256"], "fd_policy": observer.POLICY, "objective_arithmetic": observer.ARITHMETIC,
        "evidence_review": plan["evidence_review"], "original_gates_not_reclassified": True,
        "optimizer_updates": 0, "new_defog_calls": 0})
    return binding, identity, parent, checked


def test_reader_requires_both_actual_cases_and_external_profile(tmp_path, monkeypatch):
    binding, identity, _, checked = readiness_fixture(tmp_path, monkeypatch)
    proof = observer.require_readiness(binding, identity)
    assert proof["passed"] is True and len(proof["cases"]) == 2
    assert proof["allocation_job_id"] == "29799999" and proof["fd_policy"] == observer.POLICY
    assert [repair for _, repair in checked] == [False, True]
    assert checked[0][0]["repair_weight"] is None and checked[1][0]["repair_weight"] == .1
    assert "diagnostic_reference" not in proof


@pytest.mark.parametrize("mutation", ["failed_gate", "different_allocation", "changed_target", "changed_config",
    "changed_source", "changed_profile", "old_gate_scalar", "ambiguous_child", "counter_disagreement",
    "missing_native_source", "changed_identity", "new_defog_calls", "old_gate_approved", "forward_cap"])
def test_reader_rejects_failed_stale_or_incomplete_native_proofs(tmp_path, monkeypatch, mutation):
    binding, identity, parent, _ = readiness_fixture(tmp_path, monkeypatch)
    child = parent / "boltz_defog/unique-child"
    path = child / "attribution.json"
    proof = observer._read(path)
    if mutation == "failed_gate": proof["branch_validation"]["verified_fixture"] = False
    if mutation == "changed_target": proof["target"]["target_sha256"] = "e" * 64
    if mutation == "changed_config": proof["candidate_config"]["dbmol"]["contact_weight"] = 3.
    if mutation == "changed_profile": proof["expected_profile"]["dbmol"]["contact_weight"] = 3.
    if mutation == "old_gate_scalar": proof["branch_validation"]["measured"]["loss"] = 2.
    if mutation == "missing_native_source": proof["candidate_source_files"].pop(observer.SOURCE_PATHS[0])
    if mutation == "changed_identity": proof["identities_after"]["point_sha256"] = "e" * 64
    if mutation == "new_defog_calls": proof["new_defog_calls"] = 1
    if mutation == "old_gate_approved": proof["old_gate_certified_by_this_diagnostic"] = True
    if mutation == "forward_cap": proof["actual_full_forward_reservations"] = 81
    write(path, proof)
    if mutation in ("different_allocation", "changed_source"):
        path = child / "run.json"; value = observer._read(path)
        if mutation == "different_allocation": value["environment"]["job_id"] = "29799998"
        else: value["source_archive_sha256"] = "c" * 64
        write(path, value)
    if mutation == "ambiguous_child": write(parent / "boltz_defog/other/run.json", {})
    if mutation == "counter_disagreement": write(child / "oracle_cost.json", {"boltz": {"fixture": {"forward_attempted": 9}}})
    with pytest.raises(ValueError): observer.require_readiness(binding, identity)


def test_cli_refuses_unallocated_execution(tmp_path, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setattr(sys, "argv", ["readiness", "--plan", str(tmp_path / "plan.json"), "--sha256", "a" * 64])
    with pytest.raises(ValueError, match="allocated DTU HPC"): observer.main()
