import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import boltz_allocation_pilot as pilot
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_branches import BranchResult
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import HardGraphBatch


def _candidate(tag, latent, path, step, smiles=None):
    hard = HardGraphBatch(torch.tensor([[tag + 1]]), torch.zeros((1, 1), dtype=torch.long),
                          torch.zeros((1, 1, 1), dtype=torch.long))
    return Candidate(smiles, torch.tensor([[float(latent)]]), hard, 0.0, path, step)


def test_promotion_scores_prefix_only_and_resumes_exact_invalid_terminal(tmp_path, monkeypatch):
    starts = [SearchPoint(_candidate(index, index, f"parent-{index}", 0),
                          torch.tensor([True]), None, (f"parent-{index}",)) for index in range(6)]
    design = SimpleNamespace(schedule_horizon=400, noise_initial=0.0, temperature_initial=1.0,
                             temperature_final=1.0, chemistry_initial=1.0, chemistry_final=1.0,
                             max_step_norm_final=None)
    terminals = {}
    visited = []

    def execute(decoder, scorer, config, stage, branches, **kwargs):
        branch, = branches
        child = int(branch.name[-2:]) - 1
        assert branch.policy == "continuous"
        assert bool(kwargs["ready"]) == (not visited)
        visited.append((child, branch.updates))
        (stage / branch.name).mkdir(parents=True)
        if branch.updates == 300:
            assert branch.parent.candidate is terminals[child]
            assert branch.parent.active is starts[child].active
            assert branch.parent.candidate.smiles is None
        else:
            assert branch.parent is starts[child]
        tag = child if branch.updates == 100 else child + 6
        # Child 0's excluded source has an artificially favorable loss.
        smiles = "source" if tag == 0 else f"candidate-{tag}"
        path = f"{stage.name}/{branch.name}"
        candidate = _candidate(tag, tag, path, 7, smiles)
        terminal = _candidate(child, child + branch.updates, path, branch.updates)
        if branch.updates == 100:
            terminals[child] = terminal
        return [BranchResult(branch, [candidate], [], {"updates": branch.updates, "valid_states": 1,
                                                       "distinct_valid": 1}, frozenset(), terminal)]

    def score(path, records):
        assert all(row["smiles"] != "source" for row in records)
        rows = []
        for row in records:
            tag = int(row["smiles"].split("-")[-1])
            # Supplies a better score only after promotion. Child 2 wins at the prefix.
            loss = -10.0 if tag >= 6 else abs(tag - 2)
            rows.append(dict(row, loss=loss))
        return rows

    monkeypatch.setattr(pilot, "execute_branches", execute)
    result = pilot.run_allocation(None, None, design, starts, tmp_path, policy="promote", fresh_budget=24,
                                  seed=19, excluded_smiles={"source"}, fresh_score=score, ready="ready.json")
    assert visited[:6] == [(index, 100) for index in range(6)]
    decision, = result["promotions"]
    assert visited[6:] == [(2, 300), (decision["wildcard_child"], 300)]
    assert decision["merit_child"] == 2 and decision["wildcard_child"] != 2
    assert decision["prefix_best_new_fresh_losses"] == [None, 1, 0, 1, 2, 3]
    assert decision["fresh_calls_at_decision"] == 5
    assert result["fresh_attempts"] == 7
    assert result["top20"][0]["step"] == 107
    assert result["time_to_ten_seconds"] is None
    assert result["updates"] == 1200 and result["relaxed_observations"] == 1208
    assert json.loads((tmp_path / "allocation/cohort-01-promotion.json").read_text()) == decision
    with pytest.raises(FileExistsError):
        pilot.run_allocation(None, None, design, starts, tmp_path, policy="promote", fresh_budget=24, seed=19)


def test_budget_matches_both_cohorts_and_rejects_partial_promotion_group():
    assert pilot.allocation_budget("fixed400", 6) == pilot.allocation_budget("fixed150", 16) == 2400
    assert pilot.allocation_budget("fixed75", 32) == 2400
    assert pilot.allocation_budget("promote", 12) == 2400
    with pytest.raises(ValueError, match="six"):
        pilot.allocation_budget("promote", 10)
