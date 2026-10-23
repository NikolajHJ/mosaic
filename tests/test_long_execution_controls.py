"""Allocated engineering tests of absolute ramps and pre-call resource guards."""

import pytest

from latentmol.execution_budget import ExecutionBudgetConfig, ExecutionBudgetExceeded, TrackedNativeBudget
from latentmol.repair_guidance.config import AbsoluteRepairWeightSchedule, LinearRepairWeightSchedule


def test_extended_run_reproduces_all_pilot_weights_then_holds():
    original = LinearRepairWeightSchedule("linear_v1", .001, .1, .5, 1.)
    extended = AbsoluteRepairWeightSchedule("linear_updates_v1", .001, .1, 12, 24)
    # Equality is exact, including the original floating-point expression.
    assert [original.at(i, 25) for i in range(26)] == [extended.at(i, 500) for i in range(26)]
    assert all(extended.at(i, 500) == .1 for i in range(24, 501))
    assert extended.at(10, 10) == extended.at(9, 10) == .001


@pytest.mark.parametrize("change", [
    {"version": "unknown"}, {"start_update": True}, {"start_update": -1},
    {"end_update": 12}, {"end_update": 1.2}, {"start_weight": -1},
    {"end_weight": float("nan")}, {"end_weight": .0001},
])
def test_absolute_schedule_rejects_ambiguous_or_invalid_inputs(change):
    raw = dict(version="linear_updates_v1", start_weight=.001, end_weight=.1, start_update=12, end_update=24)
    raw.update(change)
    with pytest.raises(ValueError):
        AbsoluteRepairWeightSchedule.from_mapping(raw)


def test_absolute_schedule_rejects_unknown_keys_and_invalid_index():
    schedule = AbsoluteRepairWeightSchedule("linear_updates_v1", .001, .1, 12, 24)
    with pytest.raises(ValueError):
        AbsoluteRepairWeightSchedule.from_mapping({**schedule.__dict__, "start_fraction": .5})
    for index, budget in ((-1, 500), (501, 500), (True, 500), (0, 0)):
        with pytest.raises(ValueError):
            schedule.at(index, budget)


@pytest.mark.parametrize("raw", [
    {"version": "unknown", "wall_seconds": 1, "max_boltz_calls": 2},
    {"version": "tracked_native_v1", "wall_seconds": 0, "max_boltz_calls": 2},
    {"version": "tracked_native_v1", "wall_seconds": 1, "max_boltz_calls": True},
    {"version": "tracked_native_v1", "wall_seconds": 1, "max_boltz_calls": 2, "extra": 1},
])
def test_execution_budget_rejects_invalid_or_undeclared_policy(raw):
    with pytest.raises(ValueError):
        ExecutionBudgetConfig.from_mapping(raw)


def test_count_limit_blocks_before_function_and_keeps_unfailed_counter_truth():
    from latentmol.multimodel.engine import Counters
    now, receipts, called = [0.], [], []
    budget = TrackedNativeBudget(ExecutionBudgetConfig("tracked_native_v1", 10, 1),
        receipts.append, clock=lambda: now[0])
    counters = Counters(before_call=budget.before_call)
    counters.call("boltz", "audit", "forward", lambda: called.append("native"), "cpu")
    counters.call("boltz", "audit", "backward", lambda: called.append("backward"), "cpu")
    counters.call("repair", "audit", "decoder_probe", lambda: None, "cpu")
    with pytest.raises(ExecutionBudgetExceeded, match="max_boltz_calls"):
        counters.call("boltz", "trial", "forward", lambda: called.append("forbidden"), "cpu")
    assert called == ["native", "backward"]
    assert "trial" not in counters.values["boltz"]
    assert receipts[-1]["status"] == "exhausted"
    assert receipts[-1]["tracked_boltz_calls"] == 1


def test_wall_limit_and_terminal_receipt_do_not_spend_extra_native_calls():
    now, receipts = [0.], []
    policy = ExecutionBudgetConfig("tracked_native_v1", 10, 2)
    guard = TrackedNativeBudget(policy, receipts.append, clock=lambda: now[0])
    guard.before_call("boltz", "seed", "native_forward")
    now[0] = 9.
    done = guard.complete()
    assert done["status"] == "completed" and done["within_limits"] is True
    assert done["tracked_boltz_calls"] == 1 and done["limits"] == policy.record()
    now[0] = 10.
    with pytest.raises(ExecutionBudgetExceeded, match="wall_seconds"):
        guard.before_call("repair", "probe", "decoder_probe")


def test_completion_receipt_uses_the_exact_checked_clock_sample():
    ticks, receipts = iter([0., 9., 10.]), []
    guard = TrackedNativeBudget(ExecutionBudgetConfig("tracked_native_v1", 10, 2),
        receipts.append, clock=lambda: next(ticks))
    assert guard.complete()["elapsed_seconds"] == 9.
