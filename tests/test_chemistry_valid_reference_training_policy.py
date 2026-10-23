"""Allocated metadata contracts for a new through40 training authorization.

Fake paths/checkpoint payloads below test applicability and byte bindings only;
they never establish scientific readiness or promote a disposable proof state.
All execution, including these fixtures, belongs on allocated DTU HPC.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol import chemistry_short_continuation as short
from latentmol import chemistry_early_health as health
from latentmol import chemistry_valid_reference_policy as old_policy
from latentmol import chemistry_valid_reference_training_policy as policy
from latentmol.experiments import file_sha256, write_json
from test_chemistry_valid_reference_policy import valid_reference_plan


def training_plan(tmp_path=None):
    """Declare a fresh metadata plan; the source135 proof still needs auth."""
    plan = valid_reference_plan(tmp_path)
    for name in policy.RETIRED_MARKERS:
        plan.pop(name, None)
    plan["valid_reference_continuation"] = deepcopy(policy.POLICY)
    plan["valid_reference_adoption"] = dict(version=policy.ADOPTION_VERSION,
        source_plan="/work3/s234818/latentmol/submissions/source135-epoch36-valid-reference-attempt001.json",
        source_plan_sha256=policy.SOURCE_PLAN_SHA, source_stability_sha256=policy.SOURCE_STABILITY_SHA,
        artifacts={"/work3/s234818/latentmol/runs/source135-epoch36-valid-reference-attempt001/stability_gate.json":"d"*64})
    plan["worker_startup"] = dict(version="chemistry-worker-startup-v1")
    plan["classification_execution"] = dict(version="repair-classification-pool-v1")
    return plan


def test_explicit_new_training_policy_preserves_protocol_guards_and_ends_at40(tmp_path):
    old = valid_reference_plan(tmp_path)
    plan = training_plan(tmp_path)
    assert plan["protocol"] == old["protocol"]
    assert policy.validate_valid_reference_continuation(plan) == policy.POLICY
    assert old_policy.validate_valid_reference_test(plan) is None
    assert short.validate_short_continuation(plan) == policy.POLICY
    assert short.require_training_branch(plan) == policy.POLICY
    assert short.continuation_end_epoch(plan) == 40 and plan["protocol"]["total_epochs"] == 100
    assert short.expected_stop_reason(plan,39) == "wall_budget_epoch_boundary"
    assert short.expected_stop_reason(plan,40) == "bounded_horizon"
    with pytest.raises(ValueError): short.expected_stop_reason(plan,41)
    assert plan["guarded_restart"] == short.GUARDED_POLICY
    assert health.validate_early_health_policy(plan["protocol"]) == short.EARLY_POLICY
    result = policy.validate_valid_reference_continuation(plan); result["end_epoch"] = 100
    assert plan["valid_reference_continuation"]["end_epoch"] == 40


def test_all_historical39_and_validation_only_contracts_remain_unchanged(tmp_path):
    from test_chemistry_short_continuation import baseline_plan
    baseline = baseline_plan(tmp_path)
    test = valid_reference_plan(tmp_path)
    assert policy.validate_valid_reference_continuation(baseline) is None
    assert policy.validate_valid_reference_continuation(test) is None
    assert short.continuation_end_epoch(baseline) == short.continuation_end_epoch(test) == 39
    assert old_policy.validate_valid_reference_test(test) == old_policy.POLICY
    assert short.require_training_branch(baseline)["purpose"] == "baseline_training"
    with pytest.raises(ValueError): short.require_training_branch(test)
    with pytest.raises(ValueError): short.expected_stop_reason(baseline,40)


@pytest.mark.parametrize("change", [lambda x:x.update(version="unknown"),lambda x:x.update(start_epoch=35),
    lambda x:x.update(start_epoch=36.),lambda x:x.update(end_epoch=39),lambda x:x.update(end_epoch=100),
    lambda x:x.update(end_epoch=True),lambda x:x.update(purpose="self_guidance_test"),
    lambda x:x.update(purpose="baseline_training"),lambda x:x.update(extra=True)])
def test_new_marker_is_exact_and_cannot_amend_old_purpose_or_endpoint(change):
    plan = training_plan(); change(plan["valid_reference_continuation"])
    with pytest.raises(ValueError): short.validate_short_continuation(plan)


@pytest.mark.parametrize("key", policy.RETIRED_MARKERS)
def test_new_training_cannot_coexist_even_with_null_historical_validation_markers(key):
    plan = training_plan(); plan[key] = None
    with pytest.raises(ValueError): policy.validate_valid_reference_continuation(plan)


@pytest.mark.parametrize("change", [lambda p:p.pop("valid_reference_continuation"),
    lambda p:p.update(valid_reference_continuation=None),lambda p:p.pop("valid_reference_adoption"),
    lambda p:p.update(valid_reference_adoption=None),lambda p:p["protocol"].pop("valid_prior_reference_guidance"),
    lambda p:p["protocol"].update(valid_prior_reference_guidance=None),
    lambda p:p["protocol"]["valid_prior_reference_guidance"].update(weight=.01),
    lambda p:p["protocol"].update(valid_prior_self_guidance=None),
    lambda p:p["protocol"].update(chemistry_target_weighting=None),lambda p:p["protocol"].update(failure_guidance=None),
    lambda p:p.pop("optimizer_step_control"),lambda p:p["optimizer_step_control"].update(factor=.5),
    lambda p:p.pop("worker_startup"),lambda p:p.pop("classification_execution"),
    lambda p:p["classification_execution"].update(version="unknown"),
    lambda p:p["protocol"].update(prior_ratio=3),lambda p:p["protocol"].update(total_epochs=40),
    lambda p:p["protocol"]["search"].update(max_edits=6),lambda p:p["protocol"].update(repair_weight=1.),
    lambda p:p["guarded_restart"].update(validation_updates=2499),
    lambda p:p["starting_checkpoint"].update(epoch=37),lambda p:p["starting_checkpoint"].update(sha256="d"*64),
    lambda p:p.update(dataset_sha256="e"*64),lambda p:p.update(parent_epoch=36),
    lambda p:p["protocol"]["early_health"].update(max_strict_validity_drop=.03)])
def test_new_authorization_never_waives_source_objective_sampling_or_guards(change):
    plan = training_plan(); change(plan)
    with pytest.raises(ValueError): short.validate_short_continuation(plan)


@pytest.mark.parametrize("change", [lambda x:x.update(version="unknown"),lambda x:x.update(source_plan="relative.json"),
    lambda x:x.update(source_plan="/tmp/plan.json"),lambda x:x.update(source_plan="/work3/s234818/latentmol/../escape.json"),
    lambda x:x.update(source_plan_sha256="f"*64),lambda x:x.update(source_stability_sha256="f"*64),
    lambda x:x.update(artifacts={}),lambda x:x.update(artifacts=[]),
    lambda x:x.update(artifacts={"relative.json":"d"*64}),
    lambda x:x.update(artifacts={"/work3/s234818/latentmol/proof.json":"D"*64}),lambda x:x.update(extra=0)])
def test_adoption_requires_the_exact_passed_source135_proof_and_bounded_artifact_hashes(change):
    plan = training_plan(); change(plan["valid_reference_adoption"])
    with pytest.raises(ValueError): policy.validate_valid_reference_continuation(plan)


def test_adoption_metadata_returns_copy_and_is_not_a_file_or_scientific_authentication():
    plan = training_plan()
    result = policy.validate_adoption_metadata(plan["valid_reference_adoption"])
    result["artifacts"].clear()
    assert plan["valid_reference_adoption"]["artifacts"]
    # The metadata-only policy can be inspected before the declared remote
    # files exist locally. Actual launch still calls authenticate_adoption.
    assert policy.validate_valid_reference_continuation(plan)["end_epoch"] == 40


def test_new_baseline_context_binds_training_scope_guidance_and_original_proof_without_reobserving(tmp_path):
    old = valid_reference_plan(tmp_path)
    plan = training_plan(tmp_path)
    before = health._baseline_context(old)
    context = health._baseline_context(plan)
    assert "short_continuation" not in context and "valid_reference_test" not in context
    assert context["valid_reference_continuation"] == policy.POLICY
    assert context["valid_reference_adoption"] == plan["valid_reference_adoption"]
    assert context["valid_prior_reference_guidance"] == plan["protocol"]["valid_prior_reference_guidance"]
    assert context["starting_checkpoint"] == before["starting_checkpoint"]
    assert context["policy"] == before["policy"] and context["panel_seed"] == before["panel_seed"]
    assert health._baseline_context(old) == before


def test_additive_starting_baseline_reuses_observation_but_rejects_changed_adoption_binding(tmp_path):
    from test_chemistry_early_health import observation
    plan = training_plan(tmp_path)
    record = dict(health._baseline_context(plan),status="authenticated_starting_state_observed",
        completed_epoch=36,updates=short.START_UPDATES,model_sha256="d"*64,job_id="29780820",
        created_at="2026-10-22T10:00:00+00:00",observation=observation())
    # This is a fake metadata observation fixture, not a new scientific draw.
    path = tmp_path/"early-health-baseline.json"; write_json(path,record)
    digest = file_sha256(path)
    assert health.require_early_health_baseline(plan,digest) == record
    plan["valid_reference_adoption"]["artifacts"]["/work3/s234818/latentmol/another.json"] = "e"*64
    with pytest.raises(ValueError): health.require_early_health_baseline(plan,digest)


def test_phase_checkpoint_persists_new_policy_adoption_and_readiness_without_changing_numerical_payload(tmp_path,monkeypatch):
    from latentmol import chemistry_continuation as c
    from latentmol import chemistry_optimizer_step as step
    plan = training_plan(tmp_path)
    plan.update(_benchmark_gate_sha256="1"*64,_failure_readiness_sha256="2"*64,_early_health_baseline_sha256="3"*64)
    payload, saved, checks = {"existing_numerical_payload":"unchanged"}, [], []
    def original_save(path,*args): Path(path).write_bytes(b"fake native checkpoint")
    def save(value,path): saved.append(deepcopy(value)); Path(path).write_bytes(b"fake phase checkpoint")
    monkeypatch.setattr(c,"save_checkpoint",original_save)
    monkeypatch.setattr(c,"require_failure_readiness",lambda p:checks.append("readiness"))
    monkeypatch.setattr(c.torch,"load",lambda *a,**k:deepcopy(payload))
    monkeypatch.setattr(c.torch,"save",save)
    monkeypatch.setattr(step,"authenticate_optimizer_step_evidence",lambda *a,**k:checks.append("step_evidence"))
    state = dict(updates=short.START_UPDATES+math.ceil(plan["training_molecules"]/256),
                 optimizer_step_evidence={"fixture":"separately authenticated"})
    c.phase_checkpoint(tmp_path/"checkpoint.pt",None,None,None,None,37,SimpleNamespace(path=tmp_path),state,plan)
    assert checks == ["readiness","step_evidence"]
    assert saved[0]["existing_numerical_payload"] == "unchanged"
    phase = saved[0]["chemistry_phase"]
    assert phase["valid_reference_continuation"] == policy.POLICY
    assert phase["valid_reference_adoption"] == plan["valid_reference_adoption"]
    assert phase["failure_readiness_sha256"] == "2"*64 and phase["early_health_baseline_sha256"] == "3"*64
    assert "short_continuation" not in phase and "valid_reference_test" not in phase
