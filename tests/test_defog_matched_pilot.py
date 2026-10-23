"""HPC fixtures for an honest zero-prior control and fresh paired-pilot gates."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from latentmol.repair_guidance.matched_pilot import (
    AUDIT_INTERVAL, BUDGETS, HARD_SAMPLES, STEPS, case_record, validate_case, validate_configs,
    validate_prior_run,
)
from latentmol.config_identity import CONFIG_HASH_VERSION, config_sha256


def prior_fixture():
    """Round-trip the real container types that source090's first gate rejected."""
    base = {"arms": ("fixed_initial_mask",), "contexts": {1: {"sha256": "a" * 64},
        22: {"sha256": "b" * 64}}, "diagnostics": {"fd_epsilons": (.001, .003)},
        "optimizer": {"steps": 6}, "context_manifest": None, "temperature": 1.0}
    ref = {"path": "/work3/u/submissions/source089/native-smoke.yaml", "sha256": "c" * 64}
    prior = {"status": "completed", "config_hash_version": CONFIG_HASH_VERSION,
        "config": {**deepcopy(base), "config_path": ref["path"],
                   "config_file_sha256": ref["sha256"], "runtime_extra": {"immutable": True}}}
    prior["config_sha256"] = config_sha256(prior["config"])
    return base, json.loads(json.dumps(prior)), {"base_config": ref,
        "prior_run_path": "/work3/u/runs/short-child/run.json", "child_run": "/work3/u/runs/short-child"}


def test_authenticated_prior_accepts_only_json_container_roundtrip():
    base, prior, args = prior_fixture()
    assert prior["config"]["arms"] != base["arms"]
    assert prior["config"]["contexts"] != base["contexts"]
    assert prior["config"]["diagnostics"] != base["diagnostics"]
    preserved = deepcopy(prior)
    validate_prior_run(base, prior, **args)
    assert prior == preserved


@pytest.mark.parametrize("change", ["incomplete", "wrong_child", "wrong_file_path", "wrong_file_sha",
    "wrong_steps", "missing_none", "integer_instead_of_float", "array_order", "stale_digest",
    "ambiguous_context_keys"])
def test_prior_roundtrip_does_not_waive_real_provenance_or_value_changes(change):
    base, prior, args = prior_fixture()
    if change == "incomplete":
        prior["status"] = "failed"
    elif change == "wrong_child":
        args["child_run"] = "/work3/u/runs/other-child"
    elif change == "wrong_file_path":
        prior["config"]["config_path"] += ".changed"
    elif change == "wrong_file_sha":
        prior["config"]["config_file_sha256"] = "d" * 64
    elif change == "wrong_steps":
        prior["config"]["optimizer"]["steps"] = 25
    elif change == "missing_none":
        prior["config"].pop("context_manifest")
    elif change == "integer_instead_of_float":
        prior["config"]["temperature"] = 1
    elif change == "array_order":
        prior["config"]["diagnostics"]["fd_epsilons"].reverse()
    elif change == "stale_digest":
        prior["config"]["optimizer"]["steps"] = 25
    else:
        base["contexts"]["1"] = {"sha256": "e" * 64}
    if change != "stale_digest":
        # Even a self-consistent but wrong receipt must fail the base/child link.
        prior["config_sha256"] = config_sha256(prior["config"])
    with pytest.raises(ValueError):
        validate_prior_run(base, prior, **args)


def fixture(guided):
    evidence = {key: 0 for key in ("native_attempts", "accepted_native_targets", "accepted_targets",
        "guided_accepted_updates", "nonzero_guided_updates", "target_only_fd_passes", "refreshes")}
    if guided:
        evidence.update(native_attempts=20, accepted_native_targets=20, accepted_targets=5, refreshes=5,
            guided_accepted_updates=STEPS, nonzero_guided_updates=STEPS, target_only_fd_passes=6,
            repair_validation_passed=True)
    result = {"status": "completed", "accepted_updates": STEPS, "repair_evidence": evidence}
    trajectory = {"completed_requested_budget": True, "requested_steps": STEPS, "fixed_counts": [22],
        "prescribed_candidate_count": (STEPS + 1) * (HARD_SAMPLES + 1),
        "trace": [{"step": step, "accepted": step < STEPS, "used_target": guided,
            "applied_weight": .01 if guided else 0., "gradient_measured": guided and step < STEPS,
            "gradient_diagnostics": {"binding_norm": 1., "weighted_repair_norm": .1} if guided else None}
            for step in range(STEPS + 1)]}
    gates = {f"step-{s:04d}": {"passed": True, "status": "passed", "native_seed_parity": True,
        "local_fd_validation": {"passed": True},
        "component_gradients": {key: {"passed": True} for key in
            ("dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss")},
        **({"repair_only_validation": {"passed": True, "status": "validated_nonzero"}} if guided else {})}
        for s in range(0, STEPS + 1, AUDIT_INTERVAL)}
    return result, trajectory, gates


@pytest.mark.parametrize("guided", [False, True])
def test_complete_declared_pilot_passes_without_chemical_validity_requirement(guided):
    result, trajectory, gates = fixture(guided)
    trajectory.update(valid_candidate_count=0, raw_argmax_valid_states=0)
    assert validate_case(result, trajectory, gates, guided=guided)["passed"]


@pytest.mark.parametrize("change", ["short", "missing_state", "missing_gate", "component_failure",
    "parity_failure", "combined_failure", "wrong_count", "short_sampling", "lost_target",
    "zero_guidance", "nonfinite_gradient", "missing_repair_fd"])
def test_incomplete_or_false_guidance_does_not_pass(change):
    result, trajectory, gates = fixture(True)
    if change == "short":
        result["accepted_updates"] -= 1
    elif change == "missing_state":
        trajectory["trace"].pop()
    elif change == "missing_gate":
        gates.pop("step-0010")
    elif change == "component_failure":
        gates["step-0010"]["component_gradients"]["dbmol_contact_loss"] = {"passed": False}
    elif change == "parity_failure":
        gates["step-0000"]["native_seed_parity"] = False
    elif change == "combined_failure":
        gates["step-0025"]["local_fd_validation"]["passed"] = False
    elif change == "wrong_count":
        trajectory["fixed_counts"] = [22, 23]
    elif change == "short_sampling":
        trajectory["prescribed_candidate_count"] -= 1
    elif change == "lost_target":
        result["repair_evidence"]["accepted_targets"] = 4
    elif change == "zero_guidance":
        result["repair_evidence"]["nonzero_guided_updates"] = 0
    elif change == "nonfinite_gradient":
        trajectory["trace"][0]["gradient_diagnostics"]["weighted_repair_norm"] = float("nan")
    elif change == "missing_repair_fd":
        gates["step-0025"].pop("repair_only_validation")
    assert not validate_case(result, trajectory, gates, guided=True)["passed"]


@pytest.mark.parametrize("key", ["native_attempts", "accepted_targets", "guided_accepted_updates", "nonzero_guided_updates"])
def test_control_cannot_hide_repair_oracle_calls(key):
    result, trajectory, gates = fixture(False)
    result["repair_evidence"][key] = 1
    assert not validate_case(result, trajectory, gates, guided=False)["passed"]


def fake_configs(root):
    original = {"name": "parent", "output_root": "/old", "gradient_audit_interval": 6,
        "optimizer": {"steps": 6}, "diagnostics": {"hard_samples": 1},
        "dbmol": {"objective_arithmetic": "contact_and_assembly_float64_v1"},
        "decoder": {"sha256": "a" * 64}, "repair_guidance": {"weight": .001,
            "weight_schedule": {"start_weight": .001, "end_weight": .1}}}
    base = SimpleNamespace(record=lambda: deepcopy(original), optimizer=SimpleNamespace(steps=6),
        gradient_audit_interval=6, arms=("fixed_initial_mask",), fd_policy="local_v1", step_control=object(),
        dbmol=SimpleNamespace(objective_arithmetic="contact_and_assembly_float64_v1"),
        start=SimpleNamespace(kind="standard_normal", seed=0), initial_latent=object(),
        repair_guidance=SimpleNamespace(refresh_interval=5, attempts_per_refresh=4, weight_schedule=object(), provider={}))
    raw = {name: case_record(base, name, root / name / "optimization") for name in ("boltz_only", "boltz_defog")}
    configs = {name: SimpleNamespace(record=lambda raw=raw: deepcopy(raw),
        repair_guidance=SimpleNamespace(provider_activation_weight=0 if name == "boltz_only" else .1))
        for name, raw in raw.items()}
    return base, configs, raw


def test_zero_weight_control_removes_positive_schedule(tmp_path):
    base, configs, raw = fake_configs(tmp_path)
    assert raw["boltz_only"]["repair_guidance"]["weight"] == 0
    assert "weight_schedule" not in raw["boltz_only"]["repair_guidance"]
    assert "weight_schedule" in raw["boltz_defog"]["repair_guidance"]
    validate_configs(base, configs, tmp_path)


@pytest.mark.parametrize("policy", ["additive_branch_v3", "pointwise_native_ad_v4"])
def test_additive_contract_changes_validation_only_not_model_or_objective(tmp_path, policy):
    base, _, _ = fake_configs(tmp_path)
    for case in ("boltz_only", "boltz_defog"):
        previous = case_record(base, case, tmp_path / case, derivative_contract="branch_local_v2")
        candidate = case_record(base, case, tmp_path / case, derivative_contract=policy)
        assert candidate.pop("fd_policy") == policy
        assert previous.pop("fd_policy") == "branch_local_v2"
        assert previous == candidate


@pytest.mark.parametrize("guided", [False, True])
def test_additive_pilot_requires_profile_from_config_at_each_scheduled_audit(monkeypatch, guided):
    from latentmol.repair_guidance import matched_pilot as module
    from latentmol.boltz import additive_contract
    result, trajectory, gates = fixture(guided)
    for gate in gates.values():
        gate["derivative_contract"] = {"version": "additive_branch_v3"}
    assert not validate_case(result, trajectory, gates, guided=guided)["passed"]
    observed = []
    monkeypatch.setattr(additive_contract, "profile_record", lambda dbmol, objective, *, repair_weight:
        {"dbmol": dbmol, "objective": objective, "weight": repair_weight})
    monkeypatch.setattr(module, "full_gate_passed", lambda gate, *, require_repair, expected_profile:
        observed.append((require_repair, expected_profile)) or True)
    cfg = SimpleNamespace(fd_policy="additive_branch_v3", dbmol="pinned-dbm", objective="pinned-objective",
        repair_guidance=SimpleNamespace(weight_at=lambda step, steps: step / steps + .001))
    assert validate_case(result, trajectory, gates, guided=guided, config=cfg)["passed"]
    assert observed == [(guided, {"dbmol": "pinned-dbm", "objective": "pinned-objective",
        "weight": (step / STEPS + .001) if guided else None}) for step in range(0, STEPS + 1, AUDIT_INTERVAL)]


def test_additive_config_rejects_receipt_from_old_contract():
    result, trajectory, gates = fixture(False)
    assert not validate_case(result, trajectory, gates, guided=False,
        config=SimpleNamespace(fd_policy="additive_branch_v3"))["passed"]


def descent_receipt():
    from latentmol.optimization import StepControl
    import math
    cfg = SimpleNamespace(step_control=StepControl(version="armijo_v1"), optimizer=SimpleNamespace(trust_radius=4.),
        repair_guidance=SimpleNamespace(weight_at=lambda step, steps: .01))
    trace = {"step": 4, "accepted": True, "total_loss": 4., "repair_target_sha256": "a" * 64, "used_target": True}
    trial = {"accepted": True, "reason": "accepted", "candidate_loss": 3.5,
        "original_gradient_dot_actual_displacement": -1., "observed_improvement_guard": 2 * math.ulp(4.),
        "actual_displacement_norm": .1, "actual_initial_displacement_norm": .3,
        "actual_loss_improvement": .5, "candidate_loss_dtype": "torch.float64", "output_ulp": math.ulp(3.5),
        "strictly_decreased": True, "informative_decrease": True, "armijo_satisfied": True,
        "objective_metadata": {"loss": 3.5, "context_sha256": "c" * 64,
            "repair_guidance": {"weight": .01, "target_sha256": "a" * 64, "used_target": True}}}
    report = {"accepted_trial": 0, "trials": [trial], "policy": cfg.step_control.record(),
        "status": "accepted", "accepted": True, "stalled": False, "baseline_loss": 4.,
        "actual_loss_improvement": .5, "baseline_loss_dtype": "torch.float64",
        "baseline_output_ulp": math.ulp(4.), "baseline_variation": None, "baseline_variation_measured": False,
        "initial_displacement_norm": .2}
    return report, trace, cfg


def test_pointwise_contract_rechecks_actual_native_descent_with_held_target():
    from latentmol.repair_guidance.matched_pilot import measured_descent_passed
    report, trace, cfg = descent_receipt()
    assert measured_descent_passed(report, trace, cfg, "c" * 64)


def test_global_trust_region_allows_valid_inward_step():
    from latentmol.repair_guidance.matched_pilot import measured_descent_passed
    report, trace, cfg = descent_receipt()
    report["initial_displacement_norm"] = .5
    report["trials"][0]["actual_displacement_norm"] = .6
    report["trials"][0]["actual_initial_displacement_norm"] = .3
    assert measured_descent_passed(report, trace, cfg, "c" * 64)


@pytest.mark.parametrize("change", ["increase", "context", "target", "weight", "guard", "armijo",
    "reported_improvement", "baseline", "radius", "displacement", "missing_trial", "extra_trial", "dtype"])
def test_pointwise_contract_rejects_false_or_transplanted_descent(change):
    from latentmol.repair_guidance.matched_pilot import measured_descent_passed
    report, trace, cfg = descent_receipt()
    trial = report["trials"][0]
    if change == "increase": trial["candidate_loss"] = 4.1
    if change == "context": trial["objective_metadata"]["context_sha256"] = "d" * 64
    if change == "target": trial["objective_metadata"]["repair_guidance"]["target_sha256"] = "b" * 64
    if change == "weight": trial["objective_metadata"]["repair_guidance"]["weight"] = .02
    if change == "guard": trial["observed_improvement_guard"] = 0.
    if change == "armijo": trial["original_gradient_dot_actual_displacement"] = -1e9
    if change == "reported_improvement": report["actual_loss_improvement"] = 10.
    if change == "baseline": report["baseline_loss"] = 5.
    if change == "radius": trial["actual_initial_displacement_norm"] = 4.1
    if change == "displacement": trial["actual_displacement_norm"] = 4.1
    if change == "missing_trial": report["trials"] = []
    if change == "extra_trial": report["trials"].append(deepcopy(trial))
    if change == "dtype": trial["candidate_loss_dtype"] = "torch.float32"
    assert not measured_descent_passed(report, trace, cfg, "c" * 64)


def best_trial_receipt(losses, *, zero=()):
    """Recorded native observations for the new policy, not a model evaluation."""
    import math
    from latentmol.optimization import StepControl
    report, trace, cfg = descent_receipt()
    cfg.step_control = StepControl(version="best_trial_v2", max_trials=3)
    template = deepcopy(report["trials"][0])
    improving = [index for index, loss in enumerate(losses) if loss < 4. and index not in zero]
    full_nonincrease = losses[0] <= 4.
    selected = 0 if full_nonincrease else improving[0] if improving else min(range(len(losses)), key=lambda index: (losses[index], index))
    kind = ("first_improvement" if selected in improving else "full_step_nonincrease") if full_nonincrease else (
        "first_improvement" if improving else "best_finite_fallback")
    report.update(policy=cfg.step_control.record(), accepted_trial=selected,
        accepted_scale=2. ** -selected, acceptance_kind=kind,
        monotone_descent=bool(improving), no_op=selected in zero, selected_loss=losses[selected],
        actual_loss_improvement=4. - losses[selected], actual_displacement_norm=0. if selected in zero else .1,
        oracle_calls_attempted=len(losses), oracle_calls_completed=len(losses),
        oracle_calls_nonfinite=0, oracle_calls_unexpected_errors=0, trials=[])
    for index, loss in enumerate(losses):
        trial = deepcopy(template)
        trial.update(index=index, scale=2. ** -index, oracle_called=True, accepted=index == selected,
            reason="accepted" if index == selected else "no_strict_decrease", finite_proposal=True, finite_candidate=True,
            candidate_loss=loss, actual_loss_improvement=4. - loss, output_ulp=math.ulp(loss),
            strictly_decreased=loss < 4., zero_displacement=index in zero,
            actual_displacement_norm=0. if index in zero else .1,
            original_gradient_dot_actual_displacement=0. if index in zero else -1.)
        trial["objective_metadata"]["loss"] = loss
        # These v1-only acceptance fields are intentionally absent in v2.
        for key in ("observed_improvement_guard", "informative_decrease", "armijo_satisfied"):
            trial.pop(key)
        report["trials"].append(trial)
    return report, trace, cfg


@pytest.mark.parametrize("losses,zero", [([3.9999999999999996], ()),
    ([4.2, 3.9], ()), ([4.2, 4.1, 4.3], ()), ([4.], ()), ([4.], (0,)), ([3.9], (0,))])
def test_native_best_trial_policy_accepts_recorded_finite_selection_without_descent_claim(losses, zero):
    from latentmol.repair_guidance.matched_pilot import measured_descent_passed, measured_step_passed
    report, trace, cfg = best_trial_receipt(losses, zero=zero)
    assert measured_step_passed(report, trace, cfg, "c" * 64)
    assert not measured_descent_passed(report, trace, cfg, "c" * 64)
    if report["acceptance_kind"] == "best_finite_fallback":
        assert report["monotone_descent"] is False


@pytest.mark.parametrize("defect", ["context", "target", "weight", "unselected_context", "baseline", "dtype",
    "radius", "displacement", "output_ulp", "selection", "monotonicity", "hidden_improvement"])
def test_native_best_trial_policy_retains_context_target_dtype_and_trust_checks(defect):
    from latentmol.repair_guidance.matched_pilot import measured_step_passed
    report, trace, cfg = best_trial_receipt([4.2, 4.1, 4.3])
    selected = report["trials"][1]
    if defect == "context": selected["objective_metadata"]["context_sha256"] = "d" * 64
    if defect == "unselected_context": report["trials"][0]["objective_metadata"]["context_sha256"] = "d" * 64
    if defect == "target": selected["objective_metadata"]["repair_guidance"]["target_sha256"] = "b" * 64
    if defect == "weight": selected["objective_metadata"]["repair_guidance"]["weight"] = .02
    if defect == "baseline": report["baseline_loss"] = 5.
    if defect == "dtype": selected["candidate_loss_dtype"] = "torch.float32"
    if defect == "radius": selected["actual_initial_displacement_norm"] = 5.
    if defect == "displacement": selected["actual_displacement_norm"] = report["actual_displacement_norm"] = 5.
    if defect == "output_ulp": selected["output_ulp"] = 0.
    if defect == "selection": report["accepted_trial"] = 0
    if defect == "monotonicity": report["monotone_descent"] = True
    if defect == "hidden_improvement": report["trials"][0].update(candidate_loss=3.9, actual_loss_improvement=4. - 3.9, strictly_decreased=True)
    assert not measured_step_passed(report, trace, cfg, "c" * 64)


@pytest.mark.parametrize("change", ["ramp", "decoder", "steps", "sampling", "tolerance", "order_extra"])
def test_undeclared_config_changes_fail(tmp_path, change):
    base, configs, raw = fake_configs(tmp_path)
    if change == "ramp":
        raw["boltz_only"]["repair_guidance"]["weight_schedule"] = {"end_weight": .1}
    elif change == "decoder":
        raw["boltz_defog"]["decoder"]["sha256"] = "b" * 64
    elif change == "steps":
        raw["boltz_defog"]["optimizer"]["steps"] = 100
    elif change == "sampling":
        raw["boltz_only"]["diagnostics"]["hard_samples"] = 1
    elif change == "tolerance":
        raw["boltz_defog"]["diagnostics"]["fd_rtol"] = .5
    else:
        configs["extra"] = configs["boltz_only"]
    with pytest.raises(ValueError):
        validate_configs(base, configs, tmp_path)


def test_budget_reserves_both_complete_arms_before_start():
    reserve = 180 + 2 * sum(BUDGETS[k] for k in
        ("optimization_timeout_seconds", "repair_timeout_seconds", "rescore_timeout_seconds")) + 120
    assert reserve < BUDGETS["allocation_seconds"] < 8 * 3600


def test_submitter_has_one_sequential_gpu_job_after_cpu_gate():
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts/hpc"))
    spec = importlib.util.spec_from_file_location("pilot_submitter", root / "scripts/hpc/submit_defog_matched_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cpu, native = module.workflow(Path("/work3/u/release"), Path("/work3/u"), Path("/work3/u/plan.json"),
                                  "a" * 64, "b" * 64, "c" * 64)
    assert native["dependency"] == "done:cpu" and "8:00" in native["arguments"]
    assert "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes" in native["arguments"]
    assert "latentmol.repair_guidance.matched_pilot" in native["job_command"]


def test_submitter_authenticates_new_pair_schema_not_smoke_fields(monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts/hpc"))
    spec = importlib.util.spec_from_file_location("pilot_asset_submitter", root / "scripts/hpc/submit_defog_matched_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = []
    monkeypatch.setattr(module, "authenticated", lambda path, sha: checked.append((str(path), sha)))
    plan = {key: {"path": "/work3/u/" + key, "sha256": "a" * 64}
            for key in ("base_config", "prior_smoke", "prior_run", "native_template")}
    plan["cases"] = [{"id": name, "config": {"path": "/work3/u/" + name, "sha256": "b" * 64}}
                     for name in ("boltz_only", "boltz_defog")]
    module.authenticate_plan_assets(plan)
    assert len(checked) == 6 and {p for p, _ in checked} == {
        "/work3/u/" + key for key in ("base_config", "prior_smoke", "prior_run", "native_template", "boltz_only", "boltz_defog")}
    plan["cases"].reverse()
    with pytest.raises(ValueError, match="control-first"):
        module.authenticate_plan_assets(plan)
