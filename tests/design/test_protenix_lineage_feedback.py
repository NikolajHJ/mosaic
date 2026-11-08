from types import SimpleNamespace

import pytest
import torch

from latentmol.design import protenix_restart_study as study
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate, write_json
from latentmol.representation import GraphSpec, HardGraphBatch


def test_lineage_feedback_pairs_starts_and_matches_fresh_attempts(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=5)
    active = torch.ones(5, dtype=torch.bool)
    hard = HardGraphBatch(torch.ones((1, 5), dtype=torch.long),
                         torch.zeros((1, 5), dtype=torch.long),
                         torch.zeros((1, 5, 5), dtype=torch.long))

    def candidate(smiles, branch, step, relaxed_loss=0.0):
        return Candidate(smiles, torch.zeros((1, 2)), hard, relaxed_loss, branch, step)

    original = RestartParent(candidate("CC", "fixture", 0), active, 2.6, {"kind": "fixture"})
    prior_smiles = iter(("N", "CN", "CCN", "CCCN"))
    starts = {}
    fresh_calls = {}

    def make_prior_parent(_decoder, seed, **_kwargs):
        return RestartParent(candidate(next(prior_smiles), f"prior-{seed}", 0),
                             active, None, {"kind": "fresh_prior", "seed": seed})

    def propose_start(_decoder, parent, _active, *, policy, seed, **_kwargs):
        return parent.latent, None, {"policy": policy, "seed": seed}

    def run_branch(_decoder, _scorer, config, path, _latent, _active, *, parent, proposal,
                   select_shortlist, shortlist_limit, scored=(), **_kwargs):
        cycle = next(part for part in path.parts if part.startswith("cycle-"))
        _, _, arm, repeat = path.name.split("-")
        starts[cycle, arm, repeat] = (parent.smiles, proposal[2], config.noise_seed)
        if cycle == "cycle-01":
            if arm == "independent":
                smiles = ("CCF",)
            else:
                smiles = ("CCO", "CCCO", "CCCCO") if repeat == "00" else ("CCO", "CCCCCO")
        else:
            if arm == "independent":
                smiles = ("CCF", "CCCCl", "CCCCCl") if repeat == "00" else ("CCCBr", "CCCCBr")
            else:
                smiles = ("CCO", "CCCOC", "CCCCOC") if repeat == "00" else ("CCOC", "CCCCCOC")
        ranked = [candidate(parent.smiles, f"{cycle}/{path.name}", 0, -1.0)]
        ranked.extend(candidate(s, f"{cycle}/{path.name}", i, float(i))
                      for i, s in enumerate(smiles, 1) if s != parent.smiles)
        selected = select_shortlist([row for row in ranked if row.smiles not in scored], shortlist_limit)
        path.mkdir()
        write_json(path / "summary.json", {
            "updates": config.updates,
            "valid_states": config.updates + 1,
            "distinct_valid": len(ranked),
            "bounded_steps": 0,
            "fresh_shortlist": [row.smiles for row in selected],
        })
        return selected

    def fresh_score(root, records):
        cycle, arm = root.parent.name, root.name.removeprefix("fresh-")
        smiles = [row["smiles"] for row in records]
        assert (cycle, arm) not in fresh_calls
        assert len(smiles) == len(set(smiles))
        fresh_calls[cycle, arm] = smiles
        losses = {"CCO": 2.0, "CCCOC": 1.8, "CCF": None}
        return [dict(row, loss=losses.get(row["smiles"], 2.5),
                     status="embedding_failed" if row["smiles"] == "CCF" else "scored")
                for row in records]

    monkeypatch.setattr(study, "load_parent_panel", lambda *_a, **_k: [original])
    monkeypatch.setattr(study, "make_prior_parent", make_prior_parent)
    monkeypatch.setattr(study, "propose_start", propose_start)
    monkeypatch.setattr(study, "run_branch", run_branch)
    config = {
        "study": "lineage_feedback",
        "design": {"learning_rate": 70.0, "chemistry_initial": 1.0, "chemistry_final": 1.0,
                   "temperature_initial": 1.0, "temperature_final": 1.0, "noise_initial": 0.0},
        "parent_cohorts": ["fixture"],
        "panel_size": 1,
        "panel_seed": 41,
        "prior_seed": 43,
        "perturbation_seed": 47,
        "tasks": {"fixture": {"comparison": "feedback", "panel_offset": 0}},
        "comparisons": {"feedback": {"parents": 1, "cycles": 2, "branches_per_parent": 2,
                                    "updates": 3, "shortlist": 2, "high_score_ceiling": 2.6}},
    }
    scorer = SimpleNamespace(score_at=lambda *_args, **_kwargs: None,
                             close=lambda: None, identity="fixture")
    result = study.run_study(decoder, lambda *_: scorer, fresh_score, config,
                             "fixture", tmp_path / "run", "fixture-sha")

    assert result["budget"]["optimization_updates"] == 36
    for repeat in ("00", "01"):
        assert starts["cycle-01", "adaptive", repeat] == starts["cycle-01", "fixed", repeat]
        adaptive = starts["cycle-02", "adaptive", repeat]
        fixed = starts["cycle-02", "fixed", repeat]
        assert (adaptive[0], fixed[0]) == ("CCO", "CC")
        assert adaptive[1:] == fixed[1:]
    assert {parent for (_, arm, _), (parent, _, _) in starts.items() if arm == "independent"} == {
        "N", "CN", "CCN", "CCCN",
    }
    first, second = result["cycles"]
    assert first["fresh_attempt_limit"] == 4 and first["fresh_attempt_shortfall"] == 1
    assert first["fresh_attempts"] == dict(adaptive=3, fixed=3, independent=3)
    assert first["fresh_scored"] == dict(adaptive=3, fixed=3, independent=2)
    assert second["fresh_attempts"] == dict(adaptive=4, fixed=4, independent=4)
    assert first["best"] == pytest.approx(dict(adaptive=2.0, fixed=2.0, independent=2.5))
    assert set(fresh_calls["cycle-01", "independent"]) == {"N", "CN", "CCF"}
    for arm in ("adaptive", "fixed", "independent"):
        first_calls = fresh_calls["cycle-01", arm]
        second_calls = fresh_calls["cycle-02", arm]
        assert (len(first_calls), len(second_calls)) == (3, 4)
        assert set(first_calls).isdisjoint(second_calls)
