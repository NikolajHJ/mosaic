"""Allocated integration scheduling tests; no model or native gate is mocked as passed."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch

from latentmol.diffsbdd.alternation import AlternationResult, SequentialResult
from latentmol.diffsbdd.native import FinalSample
from latentmol.diffsbdd.runs import require_T2_bond_change_example, run_cohort


def _T2_example():
    return {"applied": True, "hard_bonds_changed": True, "pose_changed": True,
            "transport": {"stage": "T2", "uff_convergence_code": 0,
                          "uff_max_iterations": 50,
                          "uff_position_constraint_max_displacement": 0.5,
                          "uff_position_constraint_force_constant": 10.0,
                          "max_displacement": 0.2,
                          "native_builder": {"complete_ordered_graph_matches": True}},
            "rebuilt_baseline": {"total": 1.0}, "rebuilt_proposed": {"total": 0.9}}


def _final(changed=False):
    xyz = torch.zeros(25, 3)
    one_hot = torch.tensor([[1., 0.]] * 25)
    if changed:
        one_hot[0] = torch.tensor([0., 1.])
    pocket = torch.zeros(3, 3)
    ph = torch.tensor([[1., 0.]] * 3)
    return FinalSample(xyz, one_hot, pocket, ph,
                       torch.zeros(25, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                       torch.cat((xyz, one_hot), -1), torch.cat((pocket, ph), -1))


def _engineering_driver(monkeypatch, *, transport_stage="T1", fail_arm=None):
    """Replace only scheduling boundary; this fixture issues no scientific gate."""
    import latentmol.diffsbdd.runs as runs

    events = []

    def factory_builder(output, mode, stage):
        arm = Path(output).parent.name
        events.append(("factory", arm, mode, stage))
        return SimpleNamespace(callbacks=SimpleNamespace(arm=arm),
                               admit_terminal=object(), qualification_evidence={
                                   "scope": "engineering scheduling fixture only"},
                               native_atom_decoder=("C", "N"))

    class EngineeringSampler:
        def __init__(self, stepper, callbacks, decoder, policy, qualification):
            self.arm = callbacks.arm

        def run(self, pocket, counts, stream, levels, max_updates, on_opportunity,
                before_transition=None):
            events.append(("native_run", self.arm, tuple(levels), max_updates))
            if before_transition:
                before_transition(None)
            stream.draw("engineering-native-draw", (1,), "cpu")
            if self.arm == fail_arm:
                raise RuntimeError("engineering native failure")
            records = []
            for level in levels:
                applied = max_updates > 0
                receipt = {"level": level, "applied": applied,
                           "hard_graph_changed": applied, "hard_bonds_changed": False,
                           "proposal": {"forward_calls": 2 if applied else 0,
                                        "backward_calls": 1 if applied else 0}}
                if transport_stage == "T2" and applied:
                    receipt.update(_T2_example())
                records.append(receipt)
                on_opportunity(receipt)
            return AlternationResult(_final(changed=max_updates > 0), tuple(records),
                                      tuple(stream.receipts))

        def run_sequential(self, final, admit_terminal, *, on_block, before_block):
            events.append(("sequential_after_terminal", self.arm))
            assert torch.equal(final.one_hot, _final().one_hot)
            records = []
            for index in range(20):
                before_block(index)
                receipt = {"block": index + 1, "applied": True,
                           "hard_graph_changed": True, **_T2_example()}
                on_block(receipt)
                records.append(receipt)
            return SequentialResult(_final(changed=True), tuple(records))

    monkeypatch.setattr(runs, "AlternatingSampler", EngineeringSampler)
    return factory_builder, events


def test_smoke_preserves_fixed_seed_arm_denominators_and_H0_equality(tmp_path, monkeypatch):
    builder, events = _engineering_driver(monkeypatch)
    result = run_cohort(SimpleNamespace(timesteps=500), {}, torch.tensor([25]), builder, {},
                        tmp_path, stage="smoke", transport_stage="T1",
                        deadline=time.monotonic() + 100)
    assert result["seeds"] == [17, 23, 29]
    assert result["arms"] == ["D", "H0", "A5", "A5-exact"]
    assert len(result["completed"]) == 12
    assert result["actual_effective_state_transfers"] == 18
    for seed in (17, 23, 29):
        root = tmp_path / f"seed-{seed:03d}"
        assert json.loads((root / "H0" / "H0-equality.json").read_text())["passed"]
        assert len((root / "A5" / "opportunities.jsonl").read_text().splitlines()) == 3
        raw = torch.load(root / "D" / "gaussians.pt", weights_only=False)
        assert len(raw["draws"]) == 1
    assert all(event[2] == (75, 50, 25) for event in events
               if event[0] == "native_run" and event[1] != "D")
    assert result["fresh_pose_native_followup"] == "required_separate_stage_pending"


def test_pilot_uses_S_after_actual_D_terminal_and_six_prespecified_arms(tmp_path, monkeypatch):
    builder, events = _engineering_driver(monkeypatch, transport_stage="T2")
    result = run_cohort(SimpleNamespace(timesteps=500), {}, torch.tensor([25]), builder,
                        {"T2_bond_change_example": _T2_example()}, tmp_path,
                        stage="pilot", transport_stage="T2", deadline=time.monotonic() + 100)
    assert result["seeds"] == list(range(31, 39))
    assert len(result["completed"]) == 48
    assert len([e for e in events if e[0] == "sequential_after_terminal"]) == 8
    assert not any(e[0] == "native_run" and e[1] == "S" for e in events)
    raw = torch.load(tmp_path / "seed-031" / "S" / "gaussians.pt", weights_only=False)
    assert raw["draws"] == []
    assert raw["S_baseline_draws"] == "../D/gaussians.pt"
    assert result["T2_bond_change_examples"]


def test_missing_actual_T2_bond_change_blocks_pilot_before_running(tmp_path, monkeypatch):
    builder, events = _engineering_driver(monkeypatch)
    with pytest.raises(RuntimeError, match="actual T2"):
        run_cohort(SimpleNamespace(timesteps=500), {}, torch.tensor([25]), builder,
                   {"T2_transport": True}, tmp_path, stage="pilot", transport_stage="T2",
                   deadline=time.monotonic() + 100)
    assert events == []


@pytest.mark.parametrize("changed", ["score", "graph", "force", "displacement"])
def test_T2_example_does_not_qualify_from_a_boolean_or_incomplete_receipt(changed):
    example = _T2_example()
    if changed == "score":
        example["rebuilt_proposed"]["total"] = 1.1
    elif changed == "graph":
        example["transport"]["native_builder"]["complete_ordered_graph_matches"] = False
    elif changed == "force":
        example["transport"]["uff_position_constraint_force_constant"] = 1
    else:
        example["transport"]["max_displacement"] = 0.51
    with pytest.raises(RuntimeError, match="T2 scientific"):
        require_T2_bond_change_example(example)


def test_arm_failure_is_durable_and_does_not_replace_seed(tmp_path, monkeypatch):
    builder, events = _engineering_driver(monkeypatch, fail_arm="A5")
    with pytest.raises(RuntimeError, match="engineering native failure"):
        run_cohort(SimpleNamespace(timesteps=500), {}, torch.tensor([25]), builder, {},
                   tmp_path, stage="smoke", deadline=time.monotonic() + 100)
    saved = json.loads((tmp_path / "cohort.json").read_text())
    assert saved["status"] == "failed_or_incomplete"
    assert len(saved["completed"]) == 2
    assert (tmp_path / "seed-017" / "A5" / "failure.json").exists()
    assert (tmp_path / "seed-017" / "A5" / "gaussians.pt").exists()
    assert not (tmp_path / "seed-023").exists()


def test_expired_deadline_records_incomplete_without_running_models(tmp_path, monkeypatch):
    builder, events = _engineering_driver(monkeypatch)
    with pytest.raises(TimeoutError, match="deadline"):
        run_cohort(SimpleNamespace(timesteps=500), {}, torch.tensor([25]), builder, {},
                   tmp_path, stage="smoke", deadline=time.monotonic() - 1)
    assert events == []
    saved = json.loads((tmp_path / "cohort.json").read_text())
    assert saved["completed"] == []
    assert saved["status"] == "failed_or_incomplete"
