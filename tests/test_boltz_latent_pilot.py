"""Bounded pilot accounting and success logic; execute in the HPC Boltz environment."""

import pytest
import torch

from latentmol.boltz.latent_pilot import CountingScorer, _aggregate, _trust_value_gradient


def test_scorer_persists_attempts_even_when_a_call_fails(tmp_path):
    class Stub:
        def __call__(self, value):
            if value < 0:
                raise RuntimeError("failed oracle")
            return torch.tensor(float(value))

    path = tmp_path / "cost.json"
    scorer = CountingScorer(Stub(), path)
    scorer.purpose = "probe"
    assert float(scorer(1)) == 1
    with pytest.raises(RuntimeError, match="failed oracle"):
        scorer(-1)
    import json
    cost = json.loads(path.read_text())
    assert cost["forward_completed"] == 1
    assert cost["by_purpose"]["probe"]["forward_attempted"] == 2
    assert cost["by_purpose"]["probe"]["forward_completed"] == 1


def test_trust_penalty_is_stationary_inside_radius():
    anchor = torch.zeros(1, 3)
    point = torch.tensor([[2.0, 0.0, 0.0]])
    value, gradient = _trust_value_gradient(point, anchor, 3.5)
    assert float(value) == 0
    torch.testing.assert_close(gradient, torch.zeros_like(point))


def test_pilot_requires_change_and_hard_improvement_over_both_controls():
    def score(value):
        return {"fixed_context_discrete_affinity_probability_binary": value}

    baseline = {"hard": {"smiles": "C"}, "hard_projection_score": score(0.5)}
    row = {"arm": "joint", "start_radius": 4.0, "hard": {"smiles": "CC"},
           "start_hard": {"smiles": "CCC"}, "binary_probability": 0.9,
           "hard_projection_score": score(0.7), "start_hard_projection_score": score(0.6)}
    result = _aggregate([row], baseline)["joint"]
    assert result["valid_changed_fixed_projection_improved_over_start_and_baseline"] == 1
    row["hard_projection_score"] = score(0.55)
    assert _aggregate([row], baseline)["joint"]["valid_changed_fixed_projection_improved_over_start_and_baseline"] == 0


def test_carbon_anchor_summary_only_names_arms_that_ran():
    score = {"fixed_context_discrete_affinity_probability_binary": 0.5}
    baseline = {"hard": {"smiles": "CCCCCCCCCCCCC"}, "hard_projection_score": score}
    row = {"arm": "target", "start_radius": 0.0,
           "hard": {"smiles": "CCCCCCCCCCCCC"},
           "start_hard": {"smiles": "CCCCCCCCCCCCC"},
           "binary_probability": 0.5,
           "hard_projection_score": score, "start_hard_projection_score": score}
    result = _aggregate([row], baseline)
    assert list(result) == ["target"]
    assert result["target"]["by_start_radius"]["0.0"]["attempts"] == 1
