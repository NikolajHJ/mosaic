"""Allocated metadata/controller contracts for one nonpromotable target test."""
from copy import deepcopy
import hashlib
import json

import pytest

from latentmol import chemistry_target_weighting_policy as policy
from latentmol import chemistry_short_continuation as short
from test_chemistry_self_guidance_iteration import iteration_plan
from test_chemistry_independent_self_test import _fixture


def target_weighting_plan(tmp_path=None):
    plan = iteration_plan(tmp_path, .1)
    plan["protocol"]["chemistry_target_weighting"] = deepcopy(policy.LOSS_POLICY)
    plan["target_weighting_test"] = deepcopy(policy.POLICY)
    plan["target_weighting_probe"] = deepcopy(policy.PROBE_POLICY)
    return plan


def test_absent_marker_preserves_historical_baseline_and_self_plans(tmp_path):
    from test_chemistry_short_continuation import baseline_plan
    for purpose in ("baseline_training", "self_guidance_test"):
        plan = baseline_plan(tmp_path, purpose)
        assert policy.validate_target_weighting_test(plan) is None
        assert short.validate_short_continuation(plan) == plan["short_continuation"]
    plan = iteration_plan(tmp_path)
    assert policy.validate_target_weighting_test(plan) is None
    assert short.validate_short_continuation(plan) == plan["short_continuation"]


def test_exact_weighted_test_retains_all_original_short_guards_and_returns_fresh_marker(tmp_path):
    plan = target_weighting_plan(tmp_path)
    marker = policy.validate_target_weighting_test(plan)
    assert marker == plan["target_weighting_test"]
    marker["version"] = "mutated"
    assert plan["target_weighting_test"] == policy.POLICY
    assert short.validate_short_continuation(plan) == plan["short_continuation"]
    assert plan["guarded_restart"] == short.GUARDED_POLICY
    assert short.continuation_end_epoch(plan) == 39
    with pytest.raises(ValueError, match="disposable"):
        short.require_training_branch(plan)


def test_protocol_only_schema_reader_is_not_full_plan_launch_authorization(tmp_path):
    plan = target_weighting_plan(tmp_path)
    plan.pop("target_weighting_test")
    assert short.validate_baseline_continuation(plan["protocol"]) == plan["protocol"]["baseline_continuation"]
    with pytest.raises(ValueError, match="marker"):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("value", [None, {}, {"version": "unknown"},
    {"version": policy.VERSION, "allow_training": False}, True, "epoch36-bond-weighting-test-v1"])
def test_unknown_or_nonexact_top_markers_fail(value):
    plan = target_weighting_plan(); plan["target_weighting_test"] = value
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


@pytest.mark.parametrize("value", [None, {}, {"version": "unknown", "none_bond_weight": .1},
    {"version": policy.LOSS_POLICY["version"], "none_bond_weight": True},
    {"version": policy.LOSS_POLICY["version"], "none_bond_weight": 0},
    {"version": policy.LOSS_POLICY["version"], "none_bond_weight": 1.0},
    {"version": policy.LOSS_POLICY["version"], "none_bond_weight": float("nan")},
    {"version": policy.LOSS_POLICY["version"], "none_bond_weight": float("inf")},
    dict(policy.LOSS_POLICY, extra=True)])
def test_loss_policy_is_closed_and_exact_in_both_readers(value):
    plan = target_weighting_plan(); plan["protocol"]["chemistry_target_weighting"] = value
    for reader in (lambda: short.validate_baseline_continuation(plan["protocol"]),
                   lambda: short.validate_short_continuation(plan)):
        with pytest.raises(ValueError):
            reader()


@pytest.mark.parametrize("change", [lambda p:p["protocol"].pop("chemistry_target_weighting"),
    lambda p:p.pop("target_weighting_test"), lambda p:p.pop("target_weighting_probe"),
    lambda p:p.update(target_weighting_probe=None),
    lambda p:p["target_weighting_probe"].update(version="unknown"),
    lambda p:p["target_weighting_probe"].update(updates=100),
    lambda p:p["target_weighting_probe"].update(updates=True),
    lambda p:p["target_weighting_probe"].update(gradient_updates=[1,39,96,100]),
    lambda p:p["target_weighting_probe"].update(allow_train=False),
    lambda p:p.update(self_strength_probe=None),
    lambda p:p.update(self_strength_probe={"version":"valid-self-strength-probe-v1"}),
    lambda p:p.pop("self_guidance_iteration"),
    lambda p:p["self_guidance_iteration"].update(weight=1.0),
    lambda p:p["protocol"]["valid_prior_self_guidance"].update(weight=.01),
    lambda p:p.pop("optimizer_step_control"),
    lambda p:p["optimizer_step_control"].update(factor=1.0),
    lambda p:p.pop("short_continuation"),
    lambda p:p["short_continuation"].update(purpose="baseline_training"),
    lambda p:p["starting_checkpoint"].update(epoch=37),
    lambda p:p["protocol"].update(prior_ratio=3),
    lambda p:p["protocol"]["search"].update(max_edits=6),
    lambda p:p["protocol"].update(repair_weight=1.0),
    lambda p:p["guarded_restart"].update(validation_updates=2499),
    lambda p:p["protocol"]["early_health"].update(interval_updates=200)])
def test_marker_cannot_bypass_declared_probe_ancestry_science_or_guard_scope(change):
    plan = target_weighting_plan(); change(plan)
    with pytest.raises(ValueError):
        short.validate_short_continuation(plan)


def test_unmatched_probe_and_weighted_protocol_without_baseline_are_rejected():
    plan = iteration_plan(); plan["target_weighting_probe"] = deepcopy(policy.PROBE_POLICY)
    with pytest.raises(ValueError, match="marker"):
        policy.validate_target_weighting_test(plan)
    plan = target_weighting_plan(); plan["protocol"].pop("baseline_continuation")
    with pytest.raises(ValueError, match="baseline continuation"):
        short.validate_short_continuation(plan)


def _campaign(monkeypatch, tmp_path):
    campaign, events, queued = _fixture(monkeypatch, tmp_path, iteration=True)
    campaign.plan["protocol"]["chemistry_target_weighting"] = deepcopy(policy.LOSS_POLICY)
    campaign.plan["target_weighting_test"] = deepcopy(policy.POLICY)
    campaign.plan["target_weighting_probe"] = deepcopy(policy.PROBE_POLICY)
    campaign.path.write_text(json.dumps(campaign.plan))
    campaign.sha = hashlib.sha256(campaign.path.read_bytes()).hexdigest()
    campaign.identity["plan_sha256"] = campaign.sha
    return campaign, events, queued


def test_target_fixtures_are_mandatory_in_cpu_and_cuda_and_worker_is_explicit(monkeypatch, tmp_path):
    import chemistry_independent_self_test as controller
    campaign, events, queued = _campaign(monkeypatch, tmp_path)
    cpu = campaign.cpu_contract_command("python")
    result = campaign.after_cpu()
    assert events == ["cpu", "predecessor", "quota", "reference"]
    assert len(queued) == 1
    command = queued[0][1][-1]
    assert all(path in cpu and path in command for path in controller.TARGET_WEIGHTING_FIXTURES)
    assert len(cpu) == len(set(cpu))
    assert "latentmol.chemistry_target_weighting_validation" in command
    assert "latentmol.chemistry_self_strength_validation" not in command
    assert "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1" in command
    assert result["training"] == "forbidden" and result["promotable"] is False
    assert result["training_callback"] == "none" and result["validation_updates"] == 2500
    assert campaign.after_cpu() == result and len(queued) == 1


def test_invalid_weighting_is_retained_as_stopped_before_gpu_queue(monkeypatch, tmp_path):
    campaign, _, queued = _campaign(monkeypatch, tmp_path)
    campaign.plan.pop("target_weighting_test")
    with pytest.raises(ValueError, match="marker"):
        campaign.after_cpu()
    assert not queued
    saved = json.loads((campaign.root / "after-cpu-decision.json").read_text())
    assert saved["status"] == "stopped" and saved["training"] == "forbidden"


@pytest.mark.parametrize("mode", ["benchmark", "advance", "train", "segment-001", "final"])
def test_weighted_test_can_never_schedule_training_or_promotion(monkeypatch, tmp_path, mode):
    campaign, _, queued = _campaign(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="cannot promote or train"):
        campaign.controller(mode)
    with pytest.raises(ValueError, match="cannot promote or train"):
        campaign.advance(mode)
    assert not queued
