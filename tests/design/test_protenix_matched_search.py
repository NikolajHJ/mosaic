import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from latentmol.design import protenix_matched_search as matched
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_branches import BranchResult
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import GraphSpec, HardGraphBatch


@pytest.mark.parametrize("policy", ("equal_allowance", "matched_minimum"))
def test_matched_search_keeps_feedback_when_continuous_has_no_new_candidates(tmp_path, monkeypatch, policy):
    config = json.loads((Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/"
                         "restart-matched-search.json").read_text(encoding="utf-8"))
    planned = matched.plan_budget(config, "seed-a")
    assert (planned["updates"], planned["fresh_attempts"]) == (144_000, 1728)
    config["fresh_scoring_policy"] = policy
    config["design"].update(updates=2, occupied_atoms=2)
    config["search"].update(waves=2, branches_per_arm=2, parents=2, children=1,
                            exact_children=1, fresh_per_branch=2)
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=3)
    hard = HardGraphBatch(torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3), dtype=torch.long),
                         torch.zeros((1, 3, 3), dtype=torch.long))
    starts, terminals, scoring = {}, {}, {}

    def gaussian(_decoder, _design, seed):
        name = f"prior-{seed}"
        return SearchPoint(Candidate(None, torch.tensor([[float(seed)]]), hard, 0., name, 0),
                           torch.tensor([True, True, False]), None, (name,))

    def execute(_decoder, _scorer, design, root, branches, **_kwargs):
        results = []
        for branch in branches:
            path = f"{root.name}/{branch.name}"
            starts[path] = branch
            folder = root / branch.name
            folder.mkdir()
            (folder / "metrics.jsonl").write_text(
                json.dumps(dict(step=0, raw_valid=branch.parent.candidate.smiles is not None,
                                hard_bond_count=0)) + "\n", encoding="utf-8")
            _, slot, arm, _ = branch.name.split("-")
            if root.name == "cycle-01":
                smiles = ("C",) if arm == "continuous" else ("C", "N")
            else:
                smiles = ("CO", "CC") if slot == "00" else ("O", "F")
                if arm == "continuous":
                    smiles = ("C",)  # Already scored: no new candidates in wave two.
            valid = [Candidate(s, branch.parent.candidate.latent, hard, 1. + i, path, 1)
                     for i, s in enumerate(smiles)]
            terminal = Candidate(None, branch.parent.candidate.latent + design.updates, hard,
                                 1., path, design.updates)
            terminals[path] = terminal
            results.append(BranchResult(branch, valid, [], dict(updates=design.updates), frozenset(), terminal))
        return results

    def fresh_score(root, records):
        scoring[(root.parent.name, root.name)] = [row["smiles"] for row in records]
        return [dict(row, status="scored", loss=2.1) for row in records]

    monkeypatch.setattr(matched, "gaussian_start", gaussian)
    monkeypatch.setattr(matched, "execute_branches", execute)
    output = tmp_path / "run"
    result = matched.run_matched(decoder, lambda *_: SimpleNamespace(identity={}, close=lambda: None),
                                 fresh_score, config, "seed-a", output, "fixture-checkpoint")
    assert result["completed_updates_by_arm"] == dict.fromkeys(matched.ARMS, 8)
    assert result["fresh_attempts_by_arm"] == (
        dict(adaptive=6, continuous=1, independent=6) if policy == "equal_allowance"
        else dict.fromkeys(matched.ARMS, 1))
    assert result["relaxed_calls_by_arm"] == dict.fromkeys(matched.ARMS, 12)
    assert result["fresh_scoring_policy"] == policy
    assert [row["fresh_attempt_shortfall"] for row in result["cycles"]] == [3, 4]
    wave_two = result["cycles"][1]
    assert wave_two["fresh_attempt_allowance_by_arm"] == dict.fromkeys(matched.ARMS, 4)
    assert wave_two["fresh_attempt_shortfall_by_arm"] == (
        dict(adaptive=0, continuous=4, independent=0) if policy == "equal_allowance"
        else dict.fromkeys(matched.ARMS, 4))
    for slot in range(2):
        adaptive = starts[f"cycle-01/parent-{slot:02d}-adaptive-00"].parent
        assert adaptive is starts[f"cycle-01/parent-{slot:02d}-independent-00"].parent
    assert starts["cycle-01/parent-00-continuous-00"].parent is starts["cycle-01/parent-00-adaptive-00"].parent
    prior_path = "cycle-01/parent-01-continuous-00"
    continued = starts["cycle-02/parent-00-continuous-00"]
    assert continued.policy == "continuous" and continued.parent.candidate is terminals[prior_path]
    assert continued.parent.candidate.smiles is None
    assert starts["cycle-02/parent-00-adaptive-00"].policy == "none"
    # Both first-wave candidates share a source step, so the spacing rule keeps one parent.
    assert starts["cycle-02/parent-01-adaptive-00"].policy == "bootstrap"
    assert result["cycles"][1]["adaptive_fallbacks"] == 1
    for arm in matched.ARMS:
        assert scoring[("cycle-01", f"fresh-{arm}")] == (
            ["C", "N"] if policy == "equal_allowance" and arm != "continuous" else ["C"])
        if policy == "equal_allowance" and arm != "continuous":
            assert set(scoring[("cycle-02", f"fresh-{arm}")]) == {"CO", "CC", "O", "F"}
        else:
            assert ("cycle-02", f"fresh-{arm}") not in scoring
    archive = json.loads((output / "cycle-02" / "archives.json").read_text())
    if policy == "equal_allowance":
        assert any(point["branch"].startswith("cycle-02/") for point in archive["adaptive"])


def test_matched_width_shares_initial_states_and_caps_exact_children_at_common_parent_count(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/"
                         "restart-matched-width.json").read_text(encoding="utf-8"))
    planned = matched.plan_budget(config, "seed-a")
    assert (planned["updates"], planned["fresh_attempts"]) == (108_000, 1296)
    config["design"].update(updates=2, occupied_atoms=2)
    config["search"]["waves"] = 3
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=2)
    active = torch.ones(2, dtype=torch.bool)
    hard = HardGraphBatch(torch.zeros((1, 2), dtype=torch.long), torch.zeros((1, 2), dtype=torch.long),
                         torch.zeros((1, 2, 2), dtype=torch.long))
    starts = {}

    def gaussian(_decoder, _design, seed):
        name = f"prior-{seed}"
        return SearchPoint(Candidate(None, torch.tensor([[float(seed)]]), hard, 0., name, 0),
                           active, None, (name,))

    def execute(_decoder, _scorer, design, root, branches, **_kwargs):
        starts.setdefault(root.name, []).append(branches)
        results = []
        for index, branch in enumerate(branches):
            folder = root / branch.name
            folder.mkdir()
            (folder / "metrics.jsonl").write_text('{"step":0,"raw_valid":false,"hard_bond_count":0}\n',
                                                  encoding="utf-8")
            smiles = {("cycle-01", 0): "C", ("cycle-02", 0): "N", ("cycle-02", 10): "O"}.get(
                (root.name, index))
            candidates = [Candidate(smiles, branch.parent.candidate.latent, hard, 1.,
                                    f"{root.name}/{branch.name}", 1)] if smiles else []
            results.append(BranchResult(branch, candidates, [], dict(updates=design.updates), frozenset()))
        return results

    def fresh_score(_root, records):
        return [dict(row, status="scored", loss=2.1) for row in records]

    monkeypatch.setattr(matched, "gaussian_start", gaussian)
    monkeypatch.setattr(matched, "execute_branches", execute)
    output = tmp_path / "width"
    result = matched.run_matched(decoder, lambda *_: SimpleNamespace(identity={}, close=lambda: None),
                                 fresh_score, config, "seed-a", output, "fixture-checkpoint")
    assert result["completed_updates_by_arm"] == dict.fromkeys(matched.WIDTH_ARMS, 72)
    assert result["fresh_attempts_by_arm"] == dict.fromkeys(matched.WIDTH_ARMS, 3)
    assert result["fresh_scoring_policy"] == "matched_minimum"
    assert result["checkpoints"]["wave_3"] == result["cycles"][2]
    for shared in zip(*starts["cycle-01"], strict=True):
        assert shared[0].parent is shared[1].parent is shared[2].parent
    for arm, children, second, third in zip(matched.WIDTH_ARMS, (2, 3, 4), starts["cycle-02"],
                                           starts["cycle-03"], strict=True):
        count = result["cycles"][1]["allocations"][arm]
        assert (count["selected_parents"], count["fallback_children"], count["exact_children"]) == (1, 12 - children, 1)
        assert second[0].parent.candidate.branch.endswith(f"-{arm}-00")
        assert len({row.parent.candidate.branch for row in third if row.policy == "none"}) == 2
        assert all(row.parent.candidate.branch.endswith(f"-{arm}-00") for row in third if row.policy == "none")
