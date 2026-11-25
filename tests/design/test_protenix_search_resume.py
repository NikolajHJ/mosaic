import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import protenix_search as search
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate
from latentmol.representation import GraphSpec, HardGraphBatch


def test_resume_restores_archive_and_attempts_after_a_partial_wave(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=5)
    active = torch.ones(5, dtype=torch.bool)
    labels = torch.ones((1, 5), dtype=torch.long)
    hard = HardGraphBatch(labels, torch.zeros_like(labels), torch.zeros((1, 5, 5), dtype=torch.long))

    def point(smiles, branch, step):
        return Candidate(smiles, torch.full((1, 2), float(step)), hard, 1., branch, step)

    parents = [RestartParent(point("CC", "archived", 0), active, 3., {"fixture": True})]
    loaded = []

    def load(*_args, **_kwargs):
        loaded.append(True)
        return parents

    config = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-frontier-slot.json"
    design, options, settings = search.resolve_config(json.loads(config.read_text()), "recent", [
        "search.parents=1", "search.waves=3", "search.max_ancestor_fraction=null",
        "search.duplicate_refill_attempts=0", "search.min_step_gap=0",
    ])
    stop = False
    runs = []
    scored = []

    def branches(_decoder, _scorer, design, root, planned, **_kwargs):
        if stop and root.name == "cycle-02":
            raise RuntimeError("interrupted")
        branch, = planned
        runs.append((root.name, branch.parent.candidate.smiles))
        path = f"{root.name}/{branch.name}"
        novel = {"cycle-01": "CCN", "cycle-02": "CCO", "cycle-03": "CCC"}[root.name]
        valid = [point(smiles, path, index + 1) for index, smiles in enumerate(dict.fromkeys((novel, "CCN")))]
        return [SimpleNamespace(branch=branch, valid=valid, invalid=[], nonreplayable_steps=set(),
                                summary={"updates": design.updates})]

    def fresh_score(_root, records):
        scored.extend(row["smiles"] for row in records)
        return [dict(row, loss={"CCN": 2.9, "CCO": 2.8, "CCC": 2.7}[row["smiles"]]) for row in records]

    monkeypatch.setattr(search, "load_parent_panel", load)
    monkeypatch.setattr(search, "execute_branches", branches)
    factory = lambda *_args: SimpleNamespace(close=lambda: None)
    args = (decoder, factory, fresh_score, design, options, settings)
    reference = search.run_search(*args, tmp_path / "reference", "decoder")
    expected_runs, expected_scores = runs.copy(), scored.copy()
    runs.clear()
    scored.clear()
    stop = True
    partial = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="interrupted"):
        search.run_search(*args, partial, "decoder")
    checkpoint = partial / "search-state.pt"
    snapshot = checkpoint.read_bytes()
    stop = False
    resumed = tmp_path / "resume"
    result = search.run_search(*args, resumed, "decoder", resume_state=checkpoint)
    assert not (resumed / "cycle-01").exists()
    assert checkpoint.read_bytes() == snapshot
    assert runs == expected_runs and scored == expected_scores
    assert result["completed_updates"] == reference["completed_updates"]
    assert result["unique_fresh_attempts"] == reference["unique_fresh_attempts"] == 3
    assert [r["best_fresh_loss"] for r in result["cycles"]] == [2.9, 2.8, 2.7]
    assert len(loaded) == 2
    changed = settings | {"design": settings["design"] | {"learning_rate": 70.0}}
    with pytest.raises(ValueError, match="different model, scorer, or settings"):
        search.run_search(decoder, factory, fresh_score, replace(design, learning_rate=70.0),
                          options, changed, tmp_path / "incompatible", "decoder", resume_state=checkpoint)
