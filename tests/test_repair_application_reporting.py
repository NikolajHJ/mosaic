"""HPC CPU fixtures for telemetry only; no native models or finite differences."""
from dataclasses import replace
import json

import pytest
import torch

import latentmol.boltz.active_set as active_set
from latentmol.boltz.active_set import ContextBank, make_branch, repair_application, score_at, select_slots, validate_branch
from latentmol.multimodel.engine import Counters
from latentmol.repair_guidance.loss import target_score_at
from latentmol.repair_guidance.random_study import trajectory_report
from test_boltz_active_set_runtime import SPEC, _Decoder, _Run, _Scorer
from test_repair_guidance import _guided_cfg, _target


def _setup(tmp_path, *, weight=.2):
    config = _guided_cfg(tmp_path, weight=weight)
    decoder, point = _Decoder(), torch.zeros(1, 1, dtype=torch.float64)
    selection = select_slots(decoder.decode(point), temperature=1., threshold=.5)
    branch = make_branch(_Scorer(), ContextBank(config, SPEC), selection, config, "c" * 64)
    return config, decoder, point, selection, branch


@pytest.mark.parametrize("weight", [0., .03])
@pytest.mark.parametrize("gradients", [False, True])
def test_missing_target_has_zero_applied_weight_without_invented_gradient(tmp_path, weight, gradients):
    config, decoder, point, _, branch = _setup(tmp_path, weight=weight)
    before = config.record()
    plain, expected_gradient = score_at(decoder, point, branch, replace(config, repair_guidance=None),
        Counters(), "fixture", gradients=gradients)
    scored, gradient = score_at(decoder, point, branch, config, Counters(), "fixture", gradients=gradients)
    repair = scored["repair_guidance"]
    assert repair["weight"] == repair["scheduled_weight"] == weight
    assert repair["applied_weight"] == repair["weighted_loss"] == 0.
    assert not repair["target_available"] and not repair["used_target"] and not repair["guidance_active"]
    assert repair["gradient_measured"] is False and "gradient_diagnostics" not in repair
    assert scored["loss"] == scored["binding_loss"] == plain["loss"]
    assert gradient is None if not gradients else torch.equal(gradient, expected_gradient)
    assert config.record() == before


@pytest.mark.parametrize("gradients", [False, True])
def test_target_inclusion_and_repair_gradient_measurement_are_distinct(tmp_path, gradients):
    config, decoder, point, selection, branch = _setup(tmp_path)
    target = _target(SPEC, selection.active)
    plain, binding_gradient = score_at(decoder, point, branch, replace(config, repair_guidance=None),
        Counters(), "fixture", gradients=True)
    repair_row, repair_gradient = target_score_at(decoder, point, selection.active, target, gradients=True)
    scored, gradient = score_at(decoder, point, branch, config, Counters(), "fixture", gradients=gradients, repair_target=target)
    repair = scored["repair_guidance"]
    assert repair["weight"] == repair["scheduled_weight"] == repair["applied_weight"] == .2
    assert repair["target_available"] and repair["used_target"] and repair["guidance_active"]
    assert repair["gradient_measured"] is gradients
    assert ("gradient_diagnostics" in repair) is gradients
    assert scored["loss"] == pytest.approx(plain["loss"] + .2 * repair_row["loss"])
    if gradients:
        torch.testing.assert_close(gradient, binding_gradient + .2 * repair_gradient, rtol=0, atol=1e-15)


def test_zero_weight_target_availability_does_not_mean_used_or_measured():
    row = repair_application(0., object(), gradient_measured=True)
    assert row["target_available"] and not row["used_target"] and not row["guidance_active"]
    assert row["gradient_measured"] is False and row["weighted_loss"] == 0.


def test_partial_gate_records_application_before_a_gradient_failure(tmp_path, monkeypatch):
    config, decoder, point, selection, branch = _setup(tmp_path)
    path = tmp_path / "failed-gate.json"
    def fail_after_parity(*args, **kwargs):
        raise RuntimeError("fixture before gradient measurement")
    monkeypatch.setattr(active_set, "score_at", fail_after_parity)
    with pytest.raises(RuntimeError, match="before gradient"):
        validate_branch(decoder, point, branch, config, Counters(), report_path=path,
                        repair_target=_target(SPEC, selection.active))
    report = json.loads(path.read_text())
    assert report["native_seed_parity"] and report["status"] == "failed"
    assert report["repair_application"]["used_target"]
    assert report["repair_application"]["gradient_measured"] is False
    assert "effective_repair_weight" not in report  # Constant legacy field availability is unchanged.


@pytest.mark.parametrize("fail_second_gate", [False, True])
def test_refresh_clears_application_in_state_and_compact_trace(tmp_path, monkeypatch, fail_second_gate):
    config, decoder, point, selection, _ = _setup(tmp_path)
    class Provider:
        def refresh(self, decoder, point, active, *, temperature, step, directory, counters):
            target = _target(SPEC, active, step=step) if step == 0 else None
            return target, {"attempts": [{"status": "accepted" if target is not None else "rejected"}]}
    monkeypatch.setattr("latentmol.repair_guidance.provider.make_provider", lambda *args: Provider())
    def gate_without_fd(*args, report_path, repair_target=None, repair_weight=None, **kwargs):
        report = {"passed": True, "native_seed_parity": True}
        if fail_second_gate and report_path.name == "gate-0001.json":
            report["passed"] = False
            report_path.write_text(json.dumps(report))
            raise RuntimeError("fixture after rejected refresh")
        report_path.write_text(json.dumps(report))
        return report
    monkeypatch.setattr(active_set, "validate_branch", gate_without_fd)
    counters = Counters()
    def execute():
        return active_set.run_arm("fixed_initial_mask", decoder, point, selection, ContextBank(config, SPEC),
                                  _Scorer(), config, "c" * 64, _Run(tmp_path), counters)
    if fail_second_gate:
        with pytest.raises(RuntimeError, match="rejected refresh"):
            execute()
    else:
        arm = execute()
        trace = trajectory_report(tmp_path, {"arms": {"fixed_initial_mask": arm}, "oracle_cost": counters.values})["trace"]
        assert [row["effective_weight"] for row in trace] == [.2, .2, .2]
        assert [row["applied_weight"] for row in trace] == [.2, 0., 0.]
        assert [row["used_target"] for row in trace] == [True, False, False]
        assert [row["gradient_measured"] for row in trace] == [True, False, False]
    state = torch.load(tmp_path / "fixed_initial_mask/state-0001.pt", weights_only=True)
    assert state["repair_application"]["scheduled_weight"] == .2
    assert state["repair_application"]["applied_weight"] == 0.
    assert state["repair_application"]["used_target"] is False
    assert state["repair_application"]["gradient_measured"] is False
