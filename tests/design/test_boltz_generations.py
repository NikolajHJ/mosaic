import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import boltz_allocation_pilot as allocation
from latentmol.design import boltz_search_methods as methods
from latentmol.design.boltz_generations import generation_budget, ranked_elites, run_generations
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_branches import BranchResult
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import HardGraphBatch


@pytest.mark.parametrize("policy,count", [("fixed400", 9), ("fixed150", 24), ("promote", 18)])
def test_generation_allocations_match(policy, count):
    assert generation_budget(policy) == dict(starts_per_generation=count, updates=10800, fresh_budget=144)
    with pytest.raises(ValueError, match="complete"):
        generation_budget(policy, updates_per_generation=301)


def test_four_parent_fixed75_budget_and_optional_fourth_diversity_slot():
    assert generation_budget("fixed75", 600, 18, 8, 4, 2, .85) == {
        "starts_per_generation": 8, "updates": 10800, "fresh_budget": 144}
    with pytest.raises(ValueError, match="Parent and child"):
        generation_budget("fixed75", 450, 18, 8, 4, 2)
    smiles = ["CCCCCCCC", "CCO", "CCN", "CCCCCCCCC", "c1ccccc1"]
    archive = {s: SimpleNamespace(fresh_loss=i, candidate=SimpleNamespace(smiles=s)) for i, s in enumerate(smiles)}
    assert [p.candidate.smiles for p in ranked_elites(archive, 4, .85)] == smiles[:3] + smiles[-1:]
    assert [p.candidate.smiles for p in ranked_elites(archive, 4)] == smiles[:4]


def test_generations_propagate_replayable_elites_cache_and_causal_promotion_metrics(tmp_path, monkeypatch):
    def point(tag, loss, branch="source", step=0):
        hard = HardGraphBatch(torch.tensor([[tag + 1]]), torch.zeros((1, 1), dtype=torch.long),
                              torch.zeros((1, 1, 1), dtype=torch.long))
        candidate = Candidate(str(tag), torch.tensor([[float(tag), float(step)]]), hard, 0.1, branch, step)
        return SearchPoint(candidate, torch.tensor([True]), loss, (branch,))

    initial = [point(tag, -tag) for tag in (1, 2, 3)]
    state = dict(archive={p.candidate.smiles: p for p in initial}, ineligible=set())
    design = SimpleNamespace(schedule_horizon=400, noise_initial=0, temperature_initial=1, temperature_final=1,
                             chemistry_initial=1, chemistry_final=1, max_step_norm_final=None)
    observed_parents, fresh_calls = [], []
    def parent_starts(decoder, cfg, state, count, seed, **kwargs):
        ordered = sorted(state["archive"].values(), key=lambda p: p.fresh_loss)[:count]
        observed_parents.append([p.candidate.smiles for p in ordered])
        return ordered, []

    def execute(decoder, scorer, cfg, stage, branches, **kwargs):
        branch, = branches
        (stage / branch.name).mkdir(parents=True)
        wave = int(stage.parent.parent.name.split("-")[-1])
        child = int(branch.name.split("-")[-1])
        # 8 repeats the same pose across generations. 99 is scored but cannot restart.
        tags = [(8, 99, 1), (8, 9, 3), (8, 9, 3)][wave - 1]
        tag = tags[child - 1]
        branch_path = f"{stage.name}/{branch.name}"
        candidate = point(tag, 0, branch_path, 7).candidate
        end = point(tag, 0, branch_path, branch.updates).candidate
        excluded_steps = frozenset([7]) if tag == 99 else frozenset()
        return [BranchResult(branch, [candidate], [], dict(updates=branch.updates, valid_states=1,
                             distinct_valid=1), excluded_steps, end)]

    def fresh(path, rows):
        fresh_calls.extend(r["smiles"] for r in rows)
        return [dict(r, loss=-float(r["smiles"])) for r in rows]

    monkeypatch.setattr(methods, "archive_starts", parent_starts)
    monkeypatch.setattr(allocation, "execute_branches", execute)
    result = run_generations(None, SimpleNamespace(score_valid_records=fresh), design, state, tmp_path,
                             policy="fixed150", seed=17, generations=3, updates_per_generation=450,
                             fresh_per_generation=3)
    assert observed_parents == [["3", "2", "1"], ["8", "3", "2"], ["9", "8", "3"]]
    assert fresh_calls == ["8", "99", "9"]
    assert [w["wave_fresh_attempts"] for w in result["waves"]] == [2, 1, 0]
    assert result["distinct_scored"] == result["fresh_attempts"] == 3
    assert result["updates"] == 1350 and result["fresh_budget"] == 9
    assert [m["updates"] for m in result["milestones"]] == list(range(150, 1351, 150))
    assert [m["fresh_attempts"] for m in result["milestones"]] == [1, 2, 2, 2, 3, 3, 3, 3, 3]
    assert result["milestones"][0]["best_loss"] == -8 and result["best_loss"] == -99
    second = json.loads((tmp_path / "generations/generation-02/starts.json").read_text())
    assert second["elites"][0]["source"].startswith("generations/generation-01/allocation/")
    latest = torch.load(tmp_path / "generations/generation-03/elites.pt", weights_only=True)
    assert latest[0]["smiles"] == "9" and latest[0]["latent"].tolist() == [[9.0, 7.0]]
    assert set(state["archive"]) == {"1", "2", "3"}
