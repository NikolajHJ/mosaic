"""Allocated-HPC metadata fixtures for an explicitly reduced runtime reserve.

These checks establish policy/identity wiring, not measured epoch duration or
chemical readiness. No training objective, sampling ratio or numerical gradient
is changed by the top-level amendment. Execute all fixtures on DTU HPC.
"""
from copy import deepcopy

import pytest

from latentmol.chemistry_runtime import (EPOCH_FIT_AMENDMENT_VERSION,
    epoch_fit_policy, epoch_fit_seconds, segment_runtime_policy, validate_runtime_policy)


def _plan(*, amended=True):
    value = {"protocol": {"segment_wall_seconds": 82800, "benchmark_wall_seconds": 6300,
        "prior_ratio": 2, "composition_prior_weight": 1.5, "repair_weight": .5,
        "real_repair_weight": .25, "prior_repair_weight": .75, "total_epochs": 100}}
    if amended:
        value["runtime_policy_amendment"] = {"version": EPOCH_FIT_AMENDMENT_VERSION,
                                             "factor": 1.25, "reserve_seconds": 1800}
    return value


@pytest.mark.parametrize("estimate", [1, 36000, 54000., 55000.])
def test_absent_amendment_keeps_exact_historical_epoch_guard(estimate):
    plan = _plan(amended=False)
    assert epoch_fit_policy(plan) == {"version": "chemistry-epoch-fit-legacy-v1",
                                      "factor": 1.5, "reserve_seconds": 1800}
    assert epoch_fit_seconds(plan, estimate) == 1.5 * estimate + 1800


@pytest.mark.parametrize("estimate", [1, 36000, 54000., 55000.])
def test_declared_amendment_changes_only_runtime_headroom(estimate):
    plan = _plan()
    before = deepcopy(plan)
    assert epoch_fit_policy(plan) == plan["runtime_policy_amendment"]
    assert epoch_fit_seconds(plan, estimate) == 1.25 * estimate + 1800
    assert plan == before
    assert plan["protocol"]["composition_prior_weight"] == 1.5
    assert segment_runtime_policy(plan["protocol"])["lsf_limit"] == "24:00"


def test_boundary_decisions_retain_strict_fit_and_original_reserve():
    # At the exact historical boundary a strict benchmark fit remains false;
    # the new decision differs only because the explicit factor is reduced.
    old, new = _plan(amended=False), _plan()
    assert epoch_fit_seconds(old, 54000) == old["protocol"]["segment_wall_seconds"]
    assert not epoch_fit_seconds(old, 54000) < old["protocol"]["segment_wall_seconds"]
    assert epoch_fit_seconds(new, 54000) < new["protocol"]["segment_wall_seconds"]
    assert epoch_fit_policy(new)["reserve_seconds"] == 1800


@pytest.mark.parametrize("value", [None, {}, [], True,
    {"version": EPOCH_FIT_AMENDMENT_VERSION, "factor": 1.25},
    {"version": EPOCH_FIT_AMENDMENT_VERSION, "factor": 1.25, "reserve_seconds": 1800, "extra": True},
    {"version": "unknown", "factor": 1.25, "reserve_seconds": 1800},
    *[{"version": EPOCH_FIT_AMENDMENT_VERSION, "factor": factor, "reserve_seconds": 1800}
      for factor in (True, 1, 1.0, 1.5, 0., -1., float("nan"), float("inf"), "1.25")],
    *[{"version": EPOCH_FIT_AMENDMENT_VERSION, "factor": 1.25, "reserve_seconds": reserve}
      for reserve in (True, 1800., 0, 1799, 1801, "1800")]])
def test_malformed_or_arbitrary_amendment_cannot_silently_relax_runtime(value):
    plan = _plan()
    plan["runtime_policy_amendment"] = value
    with pytest.raises(ValueError):
        epoch_fit_policy(plan)
    with pytest.raises(ValueError):
        epoch_fit_seconds(plan, 54000)


@pytest.mark.parametrize("internal", [None, True, 22 * 3600, 24 * 3600, 82800., 82801, 71 * 3600])
def test_amendment_requires_exact_23_internal_24_allocated_hours(internal):
    plan = _plan()
    plan["protocol"]["segment_wall_seconds"] = internal
    with pytest.raises(ValueError, match="23 internal hours"):
        epoch_fit_policy(plan)


@pytest.mark.parametrize("estimate", [None, True, False, 0, -1, -1., float("nan"), float("inf"),
                                      float("-inf"), "54000", [], 1.7e308])
def test_invalid_or_overflowing_epoch_estimate_fails_closed(estimate):
    with pytest.raises(ValueError):
        epoch_fit_seconds(_plan(), estimate)


def test_returned_policy_is_a_copy_and_legacy_allocation_contract_is_unchanged():
    plan = _plan()
    declared = deepcopy(plan["runtime_policy_amendment"])
    returned = epoch_fit_policy(plan)
    returned["factor"] = .1
    assert plan["runtime_policy_amendment"] == declared
    assert epoch_fit_policy(plan) == declared
    historical = {"protocol": {"segment_wall_seconds": 71 * 3600, "benchmark_wall_seconds": 6300}}
    assert validate_runtime_policy(historical["protocol"])["lsf_limit"] == "72:00"
    assert epoch_fit_policy(historical)["factor"] == 1.5


@pytest.mark.parametrize("amended", [False, True])
def test_phase_checkpoint_records_only_explicit_runtime_amendment(tmp_path, monkeypatch, amended):
    import torch
    from latentmol import chemistry_continuation as continuation
    plan = _plan(amended=amended)
    plan.update(_plan_sha256="plan", parent_checkpoint_sha256="parent", _benchmark_gate_sha256="gate",
                parent_epoch=35, execution=None)
    monkeypatch.setattr(continuation, "require_failure_readiness", lambda _: None)
    monkeypatch.setattr(continuation, "save_checkpoint", lambda path, *args: torch.save({"kept": "metadata"}, path))
    path = tmp_path / "phase.pt"
    continuation.phase_checkpoint(path, None, None, None, None, 37, None, {}, plan)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    phase = payload["chemistry_phase"]
    assert phase["protocol"] == plan["protocol"] and phase["plan_sha256"] == "plan"
    if amended:
        assert phase["runtime_policy_amendment"] == plan["runtime_policy_amendment"]
    else:
        assert "runtime_policy_amendment" not in phase
