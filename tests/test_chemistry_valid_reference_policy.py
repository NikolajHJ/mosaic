"""Allocated policy/controller fixtures for the separate frozen36 valid anchor."""
from copy import deepcopy
import hashlib
import json

import pytest

from latentmol import chemistry_valid_reference_policy as policy
from latentmol import chemistry_short_continuation as short
from test_chemistry_short_continuation import baseline_plan
from test_chemistry_self_guidance_iteration import iteration_plan
from test_chemistry_target_weighting_policy import target_weighting_plan
from test_chemistry_independent_self_test import _fixture


def valid_reference_plan(tmp_path=None):
    plan = baseline_plan(tmp_path, "self_guidance_test")
    plan["protocol"].pop("valid_prior_self_guidance")
    plan["protocol"]["valid_prior_reference_guidance"] = deepcopy(policy.LOSS_POLICY)
    plan["valid_reference_test"] = deepcopy(policy.POLICY)
    plan["valid_reference_probe"] = deepcopy(policy.PROBE_POLICY)
    plan["optimizer_step_control"] = dict(version="adamw-displacement-scale-v1", factor=.25)
    return plan


def test_absent_marker_keeps_all_historical_launch_contracts(tmp_path):
    plans = [baseline_plan(tmp_path, purpose) for purpose in ("baseline_training", "self_guidance_test")]
    plans += [iteration_plan(tmp_path), target_weighting_plan(tmp_path)]
    for plan in plans:
        assert policy.validate_valid_reference_test(plan) is None
        assert short.validate_short_continuation(plan) == plan["short_continuation"]


def test_reference_marker_binds_original36_alpha_quarter_and_disposable_scope(tmp_path):
    plan = valid_reference_plan(tmp_path)
    marker = policy.validate_valid_reference_test(plan)
    assert marker == plan["valid_reference_test"]
    marker["reference_epoch"] = 37
    assert plan["valid_reference_test"] == policy.POLICY
    assert short.validate_short_continuation(plan) == plan["short_continuation"]
    assert plan["guarded_restart"] == short.GUARDED_POLICY
    assert short.continuation_end_epoch(plan) == 39
    for action in (short.require_training_branch, lambda p:short.expected_stop_reason(p,39)):
        with pytest.raises(ValueError, match="disposable"):
            action(plan)


def test_protocol_schema_reader_cannot_authorize_markerless_reference_run():
    plan = valid_reference_plan(); plan.pop("valid_reference_test")
    assert short.validate_baseline_continuation(plan["protocol"]) == plan["protocol"]["baseline_continuation"]
    with pytest.raises(ValueError, match="marker"):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("marker", [None, {}, dict(policy.POLICY, version="unknown"),
    dict(policy.POLICY, reference_epoch=35), dict(policy.POLICY, reference_epoch=36.),
    dict(policy.POLICY, reference_epoch=True), dict(policy.POLICY, reference_sha256="a"*64),
    dict(policy.POLICY, reference_sha256=None), dict(policy.POLICY, allow_training=False)])
def test_nonexact_reference_identity_or_unknown_top_fields_fail(marker):
    plan = valid_reference_plan(); plan["valid_reference_test"] = marker
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("value", [None, {}, dict(policy.LOSS_POLICY, version="unknown"),
    dict(policy.LOSS_POLICY, weight=.01), dict(policy.LOSS_POLICY, weight=1.0),
    dict(policy.LOSS_POLICY, weight=True), dict(policy.LOSS_POLICY, weight=float("nan")),
    dict(policy.LOSS_POLICY, weight=float("inf")), dict(policy.LOSS_POLICY, extra=False)])
def test_primitive_protocol_and_full_plan_reject_nonexact_loss_policies(value):
    plan = valid_reference_plan(); plan["protocol"]["valid_prior_reference_guidance"] = value
    for action in (lambda:short.validate_baseline_continuation(plan["protocol"]),
                   lambda:short.validate_short_continuation(plan)):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("key", ["self_guidance_iteration", "self_strength_probe",
    "target_weighting_test", "target_weighting_probe"])
@pytest.mark.parametrize("value", [None, {}])
def test_even_null_obsolete_top_fields_are_not_compatible(key, value):
    plan = valid_reference_plan(); plan[key] = value
    with pytest.raises(ValueError, match="cannot mix"):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("key", ["valid_prior_self_guidance", "chemistry_target_weighting"])
@pytest.mark.parametrize("value", [None, {}])
def test_even_null_old_loss_fields_cannot_coexist(key, value):
    plan = valid_reference_plan(); plan["protocol"][key] = value
    for action in (lambda:short.validate_baseline_continuation(plan["protocol"]),
                   lambda:short.validate_short_continuation(plan)):
        with pytest.raises(ValueError):
            action()


@pytest.mark.parametrize("change", [lambda p:p.pop("valid_reference_test"),
    lambda p:p["protocol"].pop("valid_prior_reference_guidance"),
    lambda p:p.pop("valid_reference_probe"), lambda p:p.update(valid_reference_probe=None),
    lambda p:p["valid_reference_probe"].update(version="unknown"),
    lambda p:p["valid_reference_probe"].update(updates=100),
    lambda p:p["valid_reference_probe"].update(updates=True),
    lambda p:p["valid_reference_probe"].update(gradient_updates=[1,39,96,100]),
    lambda p:p["valid_reference_probe"].update(promotable=False),
    lambda p:p.pop("short_continuation"),
    lambda p:p["short_continuation"].update(purpose="baseline_training"),
    lambda p:p["short_continuation"].update(end_epoch=100),
    lambda p:p.pop("optimizer_step_control"),
    lambda p:p["optimizer_step_control"].update(factor=1.0),
    lambda p:p["starting_checkpoint"].update(epoch=37),
    lambda p:p["starting_checkpoint"].update(sha256="f"*64),
    lambda p:p["protocol"].update(prior_ratio=3),
    lambda p:p["protocol"].update(repair_weight=1.0),
    lambda p:p["protocol"]["search"].update(max_edits=6),
    lambda p:p["guarded_restart"].update(validation_updates=2499),
    lambda p:p["protocol"]["early_health"].update(interval_updates=200)])
def test_reference_marker_cannot_loosen_probe_source_or_scientific_guards(change):
    plan = valid_reference_plan(); change(plan)
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


def test_unmatched_probe_and_missing_baseline_are_not_implicit_authorization():
    plan = baseline_plan(None,"self_guidance_test")
    plan["valid_reference_probe"] = deepcopy(policy.PROBE_POLICY)
    with pytest.raises(ValueError, match="marker"):
        policy.validate_valid_reference_test(plan)
    plan = valid_reference_plan(); plan["protocol"].pop("baseline_continuation")
    with pytest.raises(ValueError, match="baseline continuation"):
        short.validate_short_continuation(plan)


def _campaign(monkeypatch, tmp_path):
    campaign, events, queued = _fixture(monkeypatch, tmp_path, quarter=True)
    campaign.plan["protocol"].pop("valid_prior_self_guidance")
    campaign.plan["protocol"]["valid_prior_reference_guidance"] = deepcopy(policy.LOSS_POLICY)
    campaign.plan["valid_reference_test"] = deepcopy(policy.POLICY)
    campaign.plan["valid_reference_probe"] = deepcopy(policy.PROBE_POLICY)
    campaign.path.write_text(json.dumps(campaign.plan))
    campaign.sha = hashlib.sha256(campaign.path.read_bytes()).hexdigest()
    campaign.identity["plan_sha256"] = campaign.sha
    return campaign, events, queued


def test_reference_fixtures_are_mandatory_cpu_cuda_with_exact_new_worker(monkeypatch, tmp_path):
    import chemistry_independent_self_test as controller
    campaign, events, queued = _campaign(monkeypatch,tmp_path)
    cpu = campaign.cpu_contract_command("python")
    result = campaign.after_cpu()
    assert events == ["cpu","predecessor","quota","reference"] and len(queued) == 1
    command = queued[0][1][-1]
    assert all(path in cpu and path in command for path in controller.VALID_REFERENCE_FIXTURES)
    assert "tests/test_chemistry_objective_gradient_diagnostics.py" in cpu
    assert "tests/test_chemistry_objective_gradient_diagnostics.py" in command
    assert len(cpu) == len(set(cpu))
    assert "latentmol.chemistry_valid_reference_validation" in command
    assert "latentmol.chemistry_target_weighting_validation" not in command
    assert "latentmol.chemistry_self_strength_validation" not in command
    assert "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1" in command
    assert result["training"] == "forbidden" and result["promotable"] is False
    assert result["training_callback"] == "none" and result["validation_updates"] == 2500
    assert result["valid_reference_test"] == policy.POLICY
    assert campaign.after_cpu() == result and len(queued) == 1


def test_unmatched_reference_policy_stops_before_gpu_and_retains_failure(monkeypatch,tmp_path):
    campaign, _, queued = _campaign(monkeypatch,tmp_path)
    campaign.plan.pop("valid_reference_test")
    with pytest.raises(ValueError, match="marker"):
        campaign.after_cpu()
    assert not queued
    saved = json.loads((campaign.root/"after-cpu-decision.json").read_text())
    assert saved["status"] == "stopped" and saved["training"] == "forbidden"


@pytest.mark.parametrize("mode", ["advance","train","benchmark","segment-001","final"])
def test_valid_reference_proof_cannot_train_or_promote_even_if_successful(monkeypatch,tmp_path,mode):
    campaign, _, queued = _campaign(monkeypatch,tmp_path)
    with pytest.raises(ValueError, match="cannot promote or train"):
        campaign.controller(mode)
    with pytest.raises(ValueError, match="cannot promote or train"):
        campaign.advance(mode)
    assert not queued
