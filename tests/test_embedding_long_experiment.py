"""Allocated-only long-run runner protocol/association fixtures."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.multimodel.study import tensor_digest
from latentmol.boltz import embedding_long_experiment as experiment
from latentmol.boltz.embedding_comparison_policy import FINAL_FLOAT64_POLICY, policy_binding


class ProductionFixture:
    def __init__(self):
        self.context_sha256 = "c" * 64
        self.calls = []
        self.base = self

    def value(self, path, z):
        return dict(scores={"dbmol_total_loss": 1. if path == "relaxed" else 2.},
            gradient=torch.full_like(z, .25 if path == "relaxed" else .5),
            context_sha256=self.context_sha256,
            comparison_graph_policy_sha256=policy_binding(FINAL_FLOAT64_POLICY))

    def evaluate(self, *args, require_gradient=False):
        path, z = args if len(args) == 2 else ("learned", args[0])
        assert require_gradient
        self.calls.append(path)
        return self.value(path, z)


def gate(objective, path, z):
    value = objective.value(path, z)
    return dict(passed=True, scoring_path=path, baseline_scores=value["scores"],
        production_gradient_sha256=tensor_digest({"gradient": value["gradient"]}),
        context_sha256=value["context_sha256"],
        comparison_graph_policy_sha256=value["comparison_graph_policy_sha256"])


def test_both_relaxed_and_learned_actual_callbacks_are_bound_to_fresh_gates():
    objective = ProductionFixture()
    z = torch.zeros(1, 2)
    gates = {path: gate(objective, path, z) for path in ("relaxed", "learned")}
    result = experiment.bind_production_gates(objective, z, gates, FINAL_FLOAT64_POLICY)
    assert objective.calls == ["relaxed", "learned"]
    assert all(value["passed"] for value in result.values())


@pytest.mark.parametrize("path", ["relaxed", "learned"])
@pytest.mark.parametrize("field", ["passed", "scoring_path", "baseline_scores", "production_gradient_sha256", "context_sha256", "comparison_graph_policy_sha256"])
def test_each_production_binding_rejects_wrong_path_scalar_derivative_context_or_policy(path, field):
    objective = ProductionFixture()
    z = torch.zeros(1, 2)
    gates = {name: gate(objective, name, z) for name in ("relaxed", "learned")}
    gates[path][field] = False if field == "passed" else {"dbmol_total_loss": 999.} if field == "baseline_scores" else "b" * 64
    with pytest.raises(ValueError, match="Fresh " + path):
        experiment.bind_production_gates(objective, z, gates, FINAL_FLOAT64_POLICY)


def test_stage_state_alias_is_node_local_identical_and_never_overwritten(tmp_path):
    source = tmp_path / "trajectories/case-001/states/state-000020.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable exact committed state")
    state = experiment.asset(source)
    node = tmp_path / "pilot"
    alias = experiment.publish_state(node, state, case="case-001", pilot=True)
    assert Path(alias["path"]).parent == node
    assert Path(alias["path"]).name == "case-001-state.pt"
    assert alias["sha256"] == state["sha256"]
    assert os.path.samefile(alias["path"], source)
    with pytest.raises(FileExistsError):
        experiment.publish_state(node, state, case="case-001", pilot=True)
    assert source.read_bytes() == b"immutable exact committed state"
    segment = experiment.publish_state(tmp_path / "segment", state, case="case-001", pilot=False)
    assert Path(segment["path"]).name == "current-state.pt"


def test_node_alias_rejects_changed_parent_state_before_link(tmp_path):
    source = tmp_path / "state.pt"
    source.write_bytes(b"original")
    asset = experiment.asset(source)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        experiment.publish_state(tmp_path / "node", asset, case="case-001", pilot=False)
    assert not (tmp_path / "node").exists()


def test_stage_result_authenticates_release_plan_status_and_artifacts(tmp_path):
    node = tmp_path / "cpu"
    node.mkdir()
    (node / "proof.txt").write_text("allocated proof")
    value = dict(status="passed", source_sha256="a" * 64, plan_sha256="b" * 64,
        artifacts={"proof": experiment.asset(node / "proof.txt")})
    experiment.write(node / "result.json", value)
    assert experiment.stage_result(tmp_path, "cpu", "a" * 64, "b" * 64) == value
    with pytest.raises(ValueError, match="exact release"):
        experiment.stage_result(tmp_path, "cpu", "c" * 64, "b" * 64)
    (node / "proof.txt").write_text("tampered")
    with pytest.raises(ValueError, match="Artifact hash"):
        experiment.stage_result(tmp_path, "cpu", "a" * 64, "b" * 64)


def test_exclusive_result_writer_preserves_child_result(tmp_path):
    path = tmp_path / "child-result.json"
    experiment.write(path, dict(status="passed"))
    with pytest.raises(FileExistsError):
        experiment.write(path, dict(status="failed_not_passed"))
    assert json.loads(path.read_text()) == dict(status="passed")


def test_all_three_fresh_oracles_and_six_gates_precede_any_pilot_optimization(tmp_path, monkeypatch):
    from latentmol.boltz import embedding_experiment, embedding_comparison
    from latentmol.boltz import embedding_validation, embedding_comparison_validation, embedding_long_optimization
    events = []
    campaign = tmp_path / "campaign"
    output = campaign / "pilot"
    output.mkdir(parents=True)
    starts = campaign / "cpu/starts.pt"
    starts.parent.mkdir()
    torch.save(dict(cases=[dict(record=dict(case_id=name, bound_context_sha256="c" * 64),
        z=torch.zeros(1, 2)) for name in experiment.CASES]), starts)
    fitted = dict(model_identity=dict(checkpoint_sha256="e" * 64, calibration_sha256="f" * 64))
    monkeypatch.setattr(experiment, "stage_result", lambda root, stage, source, plan:
        dict(artifacts={"starts": experiment.asset(starts)}) if stage == "cpu" else fitted)
    monkeypatch.setattr(embedding_experiment, "load_vae", lambda *a: (object(), None))
    monkeypatch.setattr(embedding_experiment, "load_native", lambda *a: (object(), object(), object()))
    monkeypatch.setattr(experiment, "load_long_bridge", lambda *a: (object(), {"head": 1.}))
    monkeypatch.setattr(experiment, "load_case", lambda item, device: (item["record"]["case_id"], item["z"]))

    class Base(ProductionFixture):
        def __init__(self, scorer, prepared, bridge, vae, **kwargs):
            super().__init__()
            self.case = prepared

    class Guard:
        def __init__(self, base, caps):
            self.base, self.context_sha256 = base, base.context_sha256
        def assess(self, z):
            return dict(record=dict(eligible=True))
        def evaluate(self, z, *, require_gradient=False):
            return self.base.evaluate("learned", z, require_gradient=require_gradient)

    monkeypatch.setattr(embedding_comparison, "FrozenComparisonObjective", Base)
    monkeypatch.setattr(embedding_long_optimization, "GuardedEmbeddingObjective", Guard)

    def native(scorer, prepared, bank, vae, path, *args, **kwargs):
        events.append(("native", prepared))
        experiment.write(path, dict(passed=True))
    monkeypatch.setattr(embedding_validation, "native_gate", native)

    def derivative(scorer, prepared, bridge, vae, z, path, *args, scoring_path, **kwargs):
        events.append(("gate", prepared, scoring_path))
        value = gate(ProductionFixture(), scoring_path, z)
        experiment.write(path, value)
        return value
    monkeypatch.setattr(embedding_comparison_validation, "learned_start_gate", derivative)

    def optimize(z, objective, root, **kwargs):
        events.append(("optimize", kwargs["case_id"]))
        assert sum(event[0] == "native" for event in events) == 3
        assert sum(event[0] == "gate" for event in events) == 6
        assert kwargs["target_updates"] == 10000 and kwargs["segment_updates"] == 20
        assert kwargs["max_gradient_norm"] == 1. and kwargs["trust_radius"] == 3.
        assert kwargs["learning_rate"] == 400. and kwargs["max_consecutive_noop"] == 25
        path = root / "states/state-000020.pt"
        path.parent.mkdir(parents=True)
        path.write_bytes(kwargs["case_id"].encode())
        return dict(status="segment_complete_target_pending", completed_update=20, starting_update=0,
            counts=dict(applied=17), best_valid={"hard_loss": 1.}, current_state=experiment.asset(path))
    monkeypatch.setattr(embedding_long_optimization, "run_guarded_optimization", optimize)
    config = dict(comparison_graph_policy=FINAL_FLOAT64_POLICY, native_tolerances={},
        vae={"sha256": "a" * 64}, optimization={"learning_rate": 400.},
        step_control={"version": "best_trial_v2", "max_trials": 16},
        long_campaign=dict(pilot_updates=20, segment_reserve_seconds=1800))
    result = experiment.trajectory(config, output, 1e30, campaign, "a" * 64, "b" * 64)
    assert [case["case"] for case in result["cases"]] == list(experiment.CASES)
    assert all(case["accepted_nonzero_steps"] == 17 for case in result["cases"])
    assert all(Path(item["path"]).parent == output for item in result["artifacts"].values())
    for name in experiment.CASES:
        production = json.loads((output / "gates" / name / "production-binding.json").read_text())
        assert production["relaxed"]["passed"] and production["learned"]["passed"]
