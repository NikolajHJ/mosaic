"""HPC-only schedule/config/probe fixtures; no native chemistry efficacy claim."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

import latentmol.boltz.active_set as active_set
from latentmol.boltz.active_set import ContextBank, make_branch, run_arm, score_at, select_slots, validate_branch
from latentmol.multimodel.engine import Counters
from latentmol.optimization import StepControl
from latentmol.repair_guidance.config import LinearRepairWeightSchedule, RepairGuidanceConfig
from test_boltz_active_set_runtime import SPEC, _Decoder, _Run, _Scorer
from test_repair_guidance import _guided_cfg, _target
from test_repair_guidance_provider import _config as _provider_config


def _schedule(**changes):
    value = {"version": "linear_v1", "start_weight": .001, "end_weight": .1,
             "start_fraction": .5, "end_fraction": 1.}
    value.update(changes)
    return LinearRepairWeightSchedule.from_mapping(value)


def _scheduled_cfg(tmp_path, *, steps=5, interval=4, schedule=None):
    base = _guided_cfg(tmp_path, steps=steps, interval=interval)
    return replace(base, repair_guidance=replace(base.repair_guidance,
        weight_schedule=_schedule() if schedule is None else schedule))


def test_late_linear_schedule_last_update_and_endpoint_use_end_weight():
    schedule = _schedule()
    assert [schedule.at(i, 5) for i in range(6)] == pytest.approx([.001, .001, .001, .0505, .1, .1])
    assert schedule.at(0, 1) == schedule.at(1, 1) == .1
    # A stall after just one or two updates never compresses the 100-update ramp.
    assert schedule.at(2, 100) == .001
    assert schedule.at(99, 100) == schedule.at(100, 100) == .1


@pytest.mark.parametrize("changes", [
    {"version": "cosine_v1"}, {"start_weight": -1}, {"end_weight": float("inf")},
    {"start_weight": True}, {"end_weight": .0001}, {"start_fraction": -.1},
    {"start_fraction": .9, "end_fraction": .8}, {"end_fraction": 1.1},
])
def test_schedule_rejects_invalid_or_nonincreasing_definitions(changes):
    with pytest.raises(ValueError):
        _schedule(**changes)


def test_schedule_requires_full_versioned_fields_and_valid_budget():
    with pytest.raises(ValueError):
        LinearRepairWeightSchedule.from_mapping({"version": "linear_v1"})
    raw = dict(_schedule().__dict__, typo=1)
    with pytest.raises(ValueError):
        LinearRepairWeightSchedule.from_mapping(raw)
    for index, budget in ((-1, 5), (0, 0), (6, 5), (True, 5)):
        with pytest.raises(ValueError):
            _schedule().at(index, budget)


def test_config_roundtrip_and_absent_schedule_preserve_constant_record(tmp_path):
    constant = _provider_config(tmp_path)
    old_record = constant.record()
    assert "weight_schedule" not in old_record
    assert constant.weight_at(0, 5) == constant.weight_at(5, 5) == constant.weight
    assert constant.provider_activation_weight == constant.weight
    raw = {**old_record, "weight_schedule": dict(_schedule().__dict__)}
    scheduled = RepairGuidanceConfig.from_mapping(raw, tmp_path)
    assert scheduled.record()["weight_schedule"] == raw["weight_schedule"]
    assert scheduled.weight_at(0, 5) == .001 and scheduled.weight_at(4, 5) == .1
    assert scheduled.provider_activation_weight == .1
    assert RepairGuidanceConfig.from_mapping(old_record, tmp_path).record() == old_record


def test_scheduled_score_requires_explicit_weight_and_fd_holds_it_fixed(tmp_path):
    cfg = _scheduled_cfg(tmp_path)
    decoder, point = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(point), temperature=1., threshold=.5)
    branch = make_branch(_Scorer(), ContextBank(cfg, SPEC), initial, cfg, "c" * 64)
    target = _target(SPEC, initial.active)
    counters = Counters()
    with pytest.raises(ValueError, match="explicit fixed state weight"):
        score_at(decoder, point, branch, cfg, counters, "missing_weight", gradients=True, repair_target=target)
    assert counters.values == {}  # Reject before spending a native forward.
    for weight in (.001, .1):
        row, _ = score_at(decoder, point, branch, cfg, counters, "fixture", gradients=True,
                          repair_target=target, repair_weight=weight)
        assert row["repair_guidance"]["weight"] == weight
        report = validate_branch(decoder, point, branch, cfg, counters, repair_target=target, repair_weight=weight)
        assert report["passed"] and report["effective_repair_weight"] == weight
        assert report["repair_only_validation"]["status"] == "validated_nonzero"


def test_runner_holds_each_weight_through_fd_and_trials_and_records_jumps(tmp_path, monkeypatch):
    cfg = _scheduled_cfg(tmp_path, steps=5, interval=4)
    provider_calls, seen_weights = [], []
    class ProviderFixture:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            provider_calls.append(step)
            return _target(SPEC, active, step=step), {"attempts": [{"status": "accepted"}], "step": step}
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", lambda *args: ProviderFixture())
    original_score = active_set.score_at
    def observed_score(*args, **kwargs):
        result = original_score(*args, **kwargs)
        seen_weights.append((args[5], kwargs.get("repair_weight"), result[0]["repair_guidance"]["weight"]))
        return result
    monkeypatch.setattr(active_set, "score_at", observed_score)
    decoder, start = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    run = _Run(tmp_path)
    summary = run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC),
                      _Scorer(), cfg, "c" * 64, run, Counters())
    assert summary["accepted_updates"] == 5
    assert provider_calls == [0, 4]
    folder = tmp_path / "fixed_initial_mask"
    weights = [json.loads((folder / f"metrics-{i:04d}.json").read_text())["score"]["repair_guidance"]["weight"] for i in range(6)]
    assert weights == pytest.approx([.001, .001, .001, .0505, .1, .1])
    assert all(supplied == actual for _, supplied, actual in seen_weights)
    assert summary["repair_guidance"]["last_accepted_update_weight"] == .1
    assert summary["repair_guidance"]["final_observed_weight"] == .1
    jump = json.loads((folder / "repair-weight-0003.json").read_text())
    assert jump["target_unchanged"] and jump["binding_unchanged"]
    assert jump["weighted_objective_jump"] == pytest.approx((.0505 - .001) * jump["repair_nll_held_target"])
    assert jump["is_optimization_improvement"] is False
    refresh_jump = json.loads((folder / "repair-refresh-objective-0004.json").read_text())
    assert refresh_jump["effective_weight"] == .1 and refresh_jump["schedule_held_fixed"]


def test_zero_phase_spends_no_oracle_and_first_positive_state_refreshes_off_interval(tmp_path, monkeypatch):
    cfg = _scheduled_cfg(tmp_path, steps=5, interval=4, schedule=_schedule(start_weight=0))
    calls, provider_weights = [], []
    class ProviderFixture:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            calls.append(step)
            return _target(SPEC, active, step=step), {"attempts": [{"status": "accepted"}], "step": step}
    def make(config, source):
        provider_weights.append(config.weight)
        return ProviderFixture()
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", make)
    decoder, start = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    summary = run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC),
                      _Scorer(), cfg, "c" * 64, _Run(tmp_path), Counters())
    assert calls == [3, 4]  # Step3 is the declared first-positive exception.
    assert provider_weights == [.1]
    assert summary["repair_guidance"]["refreshes"] == 2
    assert summary["repair_guidance"]["attempts"] == 2
    for step in (0, 1, 2):
        row = json.loads((tmp_path / f"fixed_initial_mask/metrics-{step:04d}.json").read_text())
        assert row["score"]["repair_guidance"]["status"] == "disabled_weight_zero"
        assert row["repair_refresh"] is None


def test_absent_schedule_retains_constant_zero_disabled_refresh_receipts(tmp_path, monkeypatch):
    cfg = _guided_cfg(tmp_path, weight=0, steps=2, interval=1)
    calls = []
    class DisabledFixture:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            calls.append(step)
            return None, {"attempts": [], "status": "disabled"}
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", lambda *args: DisabledFixture())
    decoder, start = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    summary = run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC),
                      _Scorer(), cfg, "c" * 64, _Run(tmp_path), Counters())
    assert calls == [0, 1]
    assert "weight_schedule" not in summary["repair_guidance"]["config"]
    assert not list((tmp_path / "fixed_initial_mask").glob("repair-weight-*.json"))


def test_early_stall_keeps_requested_budget_and_distinguishes_last_accepted_weight(tmp_path, monkeypatch):
    cfg = _scheduled_cfg(tmp_path, steps=100, interval=10)
    provider_calls = []
    class ProviderFixture:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            provider_calls.append(step)
            return _target(SPEC, active, step=step), {"attempts": [{"status": "accepted"}]}
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", lambda *args: ProviderFixture())
    real_step = active_set.backtracking_step
    attempts = []
    def stall_third(point, *args, **kwargs):
        attempts.append(kwargs["baseline_loss"]["repair_guidance"]["weight"])
        if len(attempts) != 3:
            return real_step(point, *args, **kwargs)
        report = kwargs["report"]
        report.update(status="stalled", oracle_calls_attempted=0)
        kwargs["persist"]()
        return SimpleNamespace(point=point, accepted=False, report=report)
    monkeypatch.setattr(active_set, "backtracking_step", stall_third)
    decoder, start = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    initial = select_slots(decoder.decode(start), temperature=1., threshold=.5)
    summary = run_arm("fixed_initial_mask", decoder, start, initial, ContextBank(cfg, SPEC),
                      _Scorer(), cfg, "c" * 64, _Run(tmp_path), Counters())
    assert summary["stalled"] and summary["accepted_updates"] == 2
    assert summary["requested_steps"] == 100 and not summary["completed_requested_budget"]
    assert summary["endpoint"]["step"] == 2
    assert attempts == pytest.approx([.001, .001, .001])
    assert provider_calls == [0]  # No fictional endpoint refresh or compressed ramp.
    assert summary["repair_guidance"]["requested_final_update_weight"] == .1
    assert summary["repair_guidance"]["final_observed_weight"] == .001
    assert summary["repair_guidance"]["last_accepted_update_weight"] == .001
    assert not (tmp_path / "fixed_initial_mask/repair-weight-0100.json").exists()
