"""Allocated-only policy fixtures; stronger self objectives never promote."""
from copy import deepcopy

import pytest

from latentmol import chemistry_self_guidance_iteration as iteration
from latentmol import chemistry_short_continuation as short
from latentmol import chemistry_early_health as health
from latentmol import chemistry_restart_policy as restart
from test_chemistry_short_continuation import baseline_plan, _quarter_proof


def iteration_plan(tmp_path=None, weight=.1):
    plan = baseline_plan(tmp_path, "self_guidance_test")
    plan["self_guidance_iteration"] = dict(version=iteration.VERSION, weight=weight)
    plan["protocol"]["valid_prior_self_guidance"]["weight"] = weight
    plan["optimizer_step_control"] = dict(version="adamw-displacement-scale-v1", factor=.25)
    return plan


def test_absent_marker_preserves_exact_historical_alpha01_and_baseline_boundaries(tmp_path):
    assert iteration.validate_self_guidance_iteration(dict(protocol={})) is None
    for purpose in ("baseline_training", "self_guidance_test"):
        plan = baseline_plan(tmp_path, purpose)
        assert short.validate_short_continuation(plan) == plan["short_continuation"]
        assert iteration.validate_self_guidance_iteration(plan) is None
    for weight in (.1, 1.0):
        plan = baseline_plan(tmp_path, "self_guidance_test")
        plan["protocol"]["valid_prior_self_guidance"]["weight"] = weight
        with pytest.raises(ValueError):
            short.validate_short_continuation(plan)


@pytest.mark.parametrize("weight", [.1, 1.0])
def test_explicit_closed_weight_matches_loss_and_remains_disposable_from36(weight, tmp_path):
    plan = iteration_plan(tmp_path, weight)
    marker = iteration.validate_self_guidance_iteration(plan)
    assert marker == plan["self_guidance_iteration"]
    marker["weight"] = 999.
    assert plan["self_guidance_iteration"]["weight"] == weight
    assert short.validate_short_continuation(plan) == plan["short_continuation"]
    assert restart.validate_guarded_restart(plan) == short.GUARDED_POLICY
    assert short.continuation_end_epoch(plan) == 39 and plan["protocol"]["total_epochs"] == 100
    for call in (short.require_training_branch, lambda p: short.expected_stop_reason(p, 39)):
        with pytest.raises(ValueError, match="disposable"):
            call(plan)


@pytest.mark.parametrize("value", [None, {}, {"version": "unknown", "weight": .1},
    {"version": iteration.VERSION, "weight": .01}, {"version": iteration.VERSION, "weight": .2},
    {"version": iteration.VERSION, "weight": 1}, {"version": iteration.VERSION, "weight": True},
    {"version": iteration.VERSION, "weight": "0.1"}, {"version": iteration.VERSION, "weight": float("nan")},
    {"version": iteration.VERSION, "weight": float("inf")},
    {"version": iteration.VERSION, "weight": .1, "allow_training": False}])
def test_unknown_implicit_or_nonexact_iteration_policies_are_rejected(value):
    plan = iteration_plan(); plan["self_guidance_iteration"] = value
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("change", [lambda p: p.pop("short_continuation"),
    lambda p: p["short_continuation"].update(purpose="baseline_training"),
    lambda p: p["protocol"]["valid_prior_self_guidance"].update(weight=.01),
    lambda p: p["protocol"]["valid_prior_self_guidance"].update(weight=1.0),
    lambda p: p["protocol"]["valid_prior_self_guidance"].update(extra=True),
    lambda p: p.pop("optimizer_step_control"), lambda p: p.update(optimizer_step_control=None),
    lambda p: p["optimizer_step_control"].update(factor=.5),
    lambda p: p["protocol"].update(prior_ratio=3),
    lambda p: p["protocol"].update(repair_weight=1.0),
    lambda p: p["protocol"]["search"].update(max_edits=6),
    lambda p: p["starting_checkpoint"].update(epoch=37),
    lambda p: p["starting_checkpoint"].update(sha256="9" * 64),
    lambda p: p["guarded_restart"].update(validation_updates=2499),
    lambda p: p["protocol"]["early_health"].update(interval_updates=200)])
def test_marker_cannot_change_original_start_ratio_search_repair_guards_or_train(change):
    plan = iteration_plan(); change(plan)
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


def _iteration_proof(tmp_path):
    from test_chemistry_guarded_restart import _bind
    plan, gate, native, run, rows, benchmark = _quarter_proof(tmp_path)
    plan["short_continuation"]["purpose"] = "self_guidance_test"
    plan["self_guidance_iteration"] = dict(version=iteration.VERSION, weight=.1)
    plan["protocol"]["valid_prior_self_guidance"] = dict(version=short.SELF_VERSION, weight=.1)
    gate["short_continuation"] = deepcopy(plan["short_continuation"])
    gate["self_guidance_iteration"] = deepcopy(plan["self_guidance_iteration"])
    _bind(tmp_path, run, gate, native)
    return plan, gate, native, run, rows


def test_stronger_weight_still_requires_full_same_plan_disposable_evidence(tmp_path):
    plan, gate, _, _, _ = _iteration_proof(tmp_path)
    assert restart.authenticate_stability(plan) == gate
    with pytest.raises(ValueError, match="disposable"):
        short.require_training_branch(plan)


@pytest.mark.parametrize("defect", ["absent_marker", "changed_marker", "partial", "failed_retention"])
def test_rebound_hashes_cannot_hide_marker_tamper_partial_proof_or_failed_health(tmp_path, defect):
    from test_chemistry_guarded_restart import _bind
    plan, gate, native, run, rows = _iteration_proof(tmp_path)
    if defect == "absent_marker":
        gate.pop("self_guidance_iteration")
    elif defect == "changed_marker":
        gate["self_guidance_iteration"]["weight"] = 1.0
    elif defect == "partial":
        gate["completed_updates"] = 200
    else:
        gate["gates"]["all_early_panels_passed"] = False
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError):
        restart.authenticate_stability(plan)


def test_old_proof_cannot_claim_an_undeclared_iteration_even_as_null(tmp_path):
    from test_chemistry_guarded_restart import _bind
    plan, gate, native, run, _, _ = _quarter_proof(tmp_path)
    gate["self_guidance_iteration"] = None
    _bind(tmp_path, run, gate, native)
    with pytest.raises(ValueError, match="iteration"):
        restart.authenticate_stability(plan)


def test_fixed_starting_baseline_explicitly_binds_iteration_and_cannot_change_coefficients(tmp_path, monkeypatch):
    from test_chemistry_early_health import observation
    monkeypatch.setenv("LSB_JOBID", "123456")
    plan = iteration_plan(tmp_path)
    record, sha = health.materialize_early_health_baseline(plan, observation(), completed_epoch=36,
        updates=short.START_UPDATES, model_sha256="1" * 64)
    assert record["self_guidance_iteration"] == plan["self_guidance_iteration"]
    assert record["valid_prior_self_guidance"]["weight"] == .1
    assert health.require_early_health_baseline(plan, sha) == record
    plan["self_guidance_iteration"]["weight"] = 1.0
    plan["protocol"]["valid_prior_self_guidance"]["weight"] = 1.0
    with pytest.raises(ValueError, match="identity differs"):
        health.require_early_health_baseline(plan, sha)
