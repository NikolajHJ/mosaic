import json
from pathlib import Path
from types import SimpleNamespace

import torch

from latentmol.design import protenix_search as search
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import GraphSpec, HardGraphBatch


CONFIGS = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_long_presets_have_equal_budgets_and_legacy_config_can_be_overridden():
    config = read(CONFIGS / "restart-search-lr60-long.json")
    for name, waves, children, invalid in (("wide", 20, 6, 0), ("deep", 40, 3, 0), ("near-valid", 40, 3, 1)):
        design, options, settings = search.resolve_config(config, name)
        assert (design.learning_rate, options.waves, options.children, options.invalid_children) == (60, waves, children, invalid)
        assert options.parents == 4 and design.updates == 500
        assert settings["budget"] == dict(branches=480, updates=240_000, fresh_attempts=2880)
    _, options, settings = search.resolve_config(read(CONFIGS / "restart-lr70.json"), "screened",
        ["search.batch_size=2", "search.cpu_workers=2", "search.overlap=true"])
    assert (options.batch_size, options.cpu_workers, options.overlap) == (2, 2, True)
    assert settings["budget"] == dict(branches=10, updates=10_000, fresh_attempts=160)


def test_near_valid_wave_uses_explicit_source_and_never_fresh_scores_it(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=5)
    active = torch.ones(5, dtype=torch.bool)
    hard = HardGraphBatch(torch.ones((1, 5), dtype=torch.long),
                         torch.zeros((1, 5), dtype=torch.long),
                         torch.zeros((1, 5, 5), dtype=torch.long))

    def candidate(smiles, branch, step):
        return Candidate(smiles, torch.zeros((1, 2)), hard, 1., branch, step)

    parents = [RestartParent(candidate(smiles, "archive", 0), active, 3., {"fixture": smiles})
               for smiles in ("CC", "CN", "CO")]
    monkeypatch.setattr(search, "load_parent_panel", lambda *args, **kwargs: parents)
    seen = []

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        assert kwargs["batch_size"] == 2
        first = root.name == "cycle-01"
        results = []
        for index, branch in enumerate(branches):
            path = f"{root.name}/{branch.name}"
            valid, invalid = [], []
            if first and index in (0, 2, 4):
                valid.append(candidate({0: "CCCC", 2: "CCCN", 4: "CCCO"}[index], path, 120))
            if first and index == 1:
                invalid.append(SearchPoint(candidate(None, path, 170), active, None,
                    branch.parent.ancestry + (path,), 1, "CCO", "fixture"))
            results.append(SimpleNamespace(branch=branch, valid=valid, invalid=invalid,
                summary=dict(updates=design.updates, first_valid_step=12), nonreplayable_steps=frozenset()))
        return results

    def fresh_score(_root, records):
        seen.extend(row["smiles"] for row in records)
        return [dict(row, loss=2.1 + 0.05 * i) for i, row in enumerate(records)]

    monkeypatch.setattr(search, "execute_branches", execute)
    clock = iter((0., 0., 10., 10., 30., 30., 30.))
    monkeypatch.setattr(search, "perf_counter", lambda: next(clock))
    design, options, settings = search.resolve_config(read(CONFIGS / "restart-search-lr60-long.json"),
        "near-valid", ["search.waves=3", "search.parents=3", "search.children=2",
                       "search.exact_children=1", "search.batch_size=2", "search.time_budget_seconds=35"])
    output = tmp_path / "run"
    result = search.run_search(decoder, lambda *_: SimpleNamespace(close=lambda: None), fresh_score,
                               design, options, settings, output, "fixture-checkpoint")
    assert result["completed_updates"] == 6000
    assert result["status"] == "time_budget" and result["time_budget_stop"]["next_wave"] == 3
    assert not (output / "cycle-03").exists()
    assert result["unique_fresh_attempts"] == 3
    assert seen == ["CCCC", "CCCN", "CCCO"]
    assert result["cycles"][0]["invalid_fallback"]
    assert result["cycles"][1]["invalid_first_valid_step"] == 12
    wave = read(output / "cycle-02/wave-plan.json")
    raw = next(branch for branch in wave["branches"] if branch["policy"] == "raw-invalid")
    assert raw["source"]["branch"] == "cycle-01/branch-02" and raw["source"]["step"] == 170
    assert raw["source"]["smiles"] is None
