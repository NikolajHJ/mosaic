import json
from pathlib import Path
from types import SimpleNamespace

import torch

from latentmol.design import protenix_search as search
from latentmol.design import protenix_search_branches as branch_execution
from latentmol.design.protenix_restart_parents import RestartParent
from latentmol.design.protenix_restarts import Candidate, RestartOffset
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


CONFIGS = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_multitarget_bootstrap_width_has_72_windows_and_no_update_noise():
    config = read(CONFIGS / "restart-multitarget.json")
    assert len(config["arms"]) == 7
    for name in config["arms"]:
        design, options, settings = search.resolve_config(config, name)
        assert settings["budget"] == dict(branches=72, updates=14_400, fresh_attempts=432)
        assert options.bootstrap_branches == 12 and options.waves == 6
        assert (options.parents, options.children, design.updates) == (3, 4, 200)
        assert design.noise_initial == 0 and design.learning_rate == 1
        assert design.occupied_atoms == settings["target"]["initialized_atoms"]
        assert options.diversity_limit == 20 and options.diversity_max_similarity == 1


def test_short_waves_share_limited_fresh_scores_across_branches(tmp_path):
    config = read(CONFIGS / "restart-brd4-e48-100-wide-deep.json")
    design, options, settings = search.resolve_config(config, "seed-a")
    assert design.updates == 100
    assert options.wave_widths() == (4,) + (18,) * 12
    assert settings["budget"] == dict(branches=220, updates=22_000, fresh_attempts=264)

    scored = []
    attempted = {"CO"}
    results = []
    for index, identities in enumerate((("CC", "CN"), ("CN", "CCC", "CCCC"), ("CO", "CCO")), 1):
        branch = f"cycle-02/branch-{index:02d}"
        valid = [Candidate(smiles, None, None, 1., branch, step) for step, smiles in enumerate(identities)]
        parent = SimpleNamespace(active=None, ancestry=("founder",))
        results.append(SimpleNamespace(valid=valid, branch=SimpleNamespace(parent=parent)))

    def fresh_score(_root, records):
        scored.extend(row["smiles"] for row in records)
        return [dict(row, loss=2.) for row in records]

    archive = {}
    search.score_wave(tmp_path, results, fresh_score, archive, attempted, options, None, limit=3)
    assert scored == ["CC", "CCC", "CCO"]
    assert set(archive) == set(scored)
    assert attempted == {"CO", *scored}


def test_wide_bootstrap_starts_distinct_latents_and_counts_all_branches(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=5)
    active = torch.ones(5, dtype=torch.bool)
    hard = HardGraphBatch(torch.ones((1, 5), dtype=torch.long),
        torch.zeros((1, 5), dtype=torch.long), torch.zeros((1, 5, 5), dtype=torch.long))
    starts = []

    def bootstrap(_decoder, design, *, seed=None, name="bootstrap"):
        starts.append((seed, name))
        point = Candidate(None, torch.full((1, 2), float(seed)), hard, 0., name, 0)
        return SearchPoint(point, active, None, (name,))

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        assert all(branch.policy == "bootstrap" for branch in branches)
        return [SimpleNamespace(branch=branch, valid=[], invalid=[], summary=dict(updates=design.updates),
                                nonreplayable_steps=frozenset()) for branch in branches]

    monkeypatch.setattr(search, "bootstrap", bootstrap)
    monkeypatch.setattr(search, "execute_branches", execute)
    config = read(CONFIGS / "restart-search-lr60-short.json")
    design, options, settings = search.resolve_config(config, "short-screened", [
        "search.waves=1", "search.bootstrap_branches=12", "search.diversity_limit=0"])
    output = tmp_path / "wide-bootstrap"
    result = search.run_search(decoder, lambda *_: SimpleNamespace(close=lambda: None),
        lambda *_: [], design, options, settings, output, "fixture")
    assert len(starts) == len(set(starts)) == 12
    assert [seed for seed, _ in starts] == list(range(design.seed, design.seed + 12))
    assert result["completed_updates"] == 12 * design.updates
    assert result["cycles"][0]["requested_branches"] == 12


def test_long_presets_have_equal_budgets_and_legacy_config_can_be_overridden():
    config = read(CONFIGS / "restart-search-lr60-long.json")
    for name, waves, children, invalid in (("wide", 10, 6, 0), ("deep", 20, 3, 0), ("near-valid", 20, 3, 1)):
        design, options, settings = search.resolve_config(config, name)
        assert (design.learning_rate, options.waves, options.children, options.invalid_children) == (60, waves, children, invalid)
        assert options.parents == 4 and design.updates == 500
        assert design.noise_initial == 0.0
        assert settings["budget"] == dict(branches=240, updates=120_000, fresh_attempts=1440)
    del config["design"]["noise_initial"]
    design, _, _ = search.resolve_config(config, "deep")
    assert design.noise_initial == 0.0
    design, options, settings = search.resolve_config(read(CONFIGS / "restart-lr70.json"), "screened",
        ["search.batch_size=2", "search.cpu_workers=2", "search.overlap=true", "design.noise_initial=0.05"])
    assert design.noise_initial == 0.05
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


def test_frontier_reserves_one_child_only_for_a_displaced_best_parent(tmp_path, monkeypatch):
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=5)
    active = torch.ones(5, dtype=torch.bool)
    hard = HardGraphBatch(torch.ones((1, 5), dtype=torch.long),
                         torch.zeros((1, 5), dtype=torch.long),
                         torch.zeros((1, 5, 5), dtype=torch.long))

    def candidate(smiles, branch, step):
        return Candidate(smiles, torch.zeros((1, 2)), hard, 1., branch, step)

    founders = [RestartParent(candidate(smiles, "archive", 0), active, loss, {})
                for smiles, loss in (("CC", 1.5), ("CO", 2.9))]
    monkeypatch.setattr(search, "load_parent_panel", lambda *_args, **_kwargs: founders)

    def execute(_decoder, _scorer, design, root, branches, **_kwargs):
        results = []
        for index, branch in enumerate(branches):
            valid = []
            if root.name == "cycle-01" and index < 2:
                valid = [candidate(("CCN", "CCCO")[index], f"{root.name}/{branch.name}", 120)]
            results.append(SimpleNamespace(branch=branch, valid=valid, invalid=[],
                summary=dict(updates=design.updates), nonreplayable_steps=frozenset()))
        return results

    def fresh_score(_root, records):
        return [dict(row, loss=2.1 + .1 * i) for i, row in enumerate(records)]

    monkeypatch.setattr(search, "execute_branches", execute)
    config = read(CONFIGS / "restart-search-lr60-long.json")
    design, options, settings = search.resolve_config(config, "deep", [
        "search.parents=2", "search.children=2", "search.waves=2", "search.exact_children=0",
        "search.min_step_gap=0", "search.max_similarity=1.0", "search.frontier_keep_best=true",
        "search.max_ancestor_fraction=null",
    ])
    output = tmp_path / "reserved"
    result = search.run_search(decoder, lambda *_: SimpleNamespace(close=lambda: None), fresh_score,
                               design, options, settings, output, "fixture-checkpoint")
    first, second = result["cycles"]
    assert first["branches"] == 4 and first["reserved_best"] is None
    assert second["branches"] == 3 and second["requested_branches"] == 4
    assert second["reserved_best"]["smiles"] == "CC"
    plan = read(output / "cycle-02/wave-plan.json")
    assert [row["source"]["smiles"] for row in plan["branches"]] == ["CC", "CCN", "CCN"]


def test_exact_start_refill_compares_latent_mask_and_offset_instead_of_smiles(monkeypatch):
    hard = HardGraphBatch(torch.ones((1, 2), dtype=torch.long),
                         torch.zeros((1, 2), dtype=torch.long),
                         torch.zeros((1, 2, 2), dtype=torch.long))

    def point(latent, active):
        return SearchPoint(Candidate("CC", torch.tensor([latent]), hard, 1., "archive", 0),
                           torch.tensor(active), 2.1, ("founder",))

    source = point([0., 0.], [True, True])
    distinct_latent = point([1., 0.], [True, True])
    distinct_mask = point([0., 0.], [False, True])
    offset = RestartOffset(DecoderOutput(torch.ones((1, 2, 2)), torch.zeros((1, 2, 2)),
                                         torch.zeros((1, 2, 2, 2))), 25)

    def propose(_decoder, candidate, _active, *, policy, seed, **_kwargs):
        return candidate.latent, offset if policy == "sparse" else None, dict(policy=policy, seed=seed)

    monkeypatch.setattr(branch_execution, "propose_start", propose)
    Branch = branch_execution.Branch
    branches = [Branch(f"branch-{i:02d}", p, i, policy="none")
                for i, p in enumerate((source, source, distinct_latent, distinct_mask), 1)]
    design = search.LatentDesignConfig.for_restart({"noise_initial": 0.0})
    seen = set()
    accepted, collisions = branch_execution.refill_exact_starts(None, design, branches, seen,
        attempts=2, fallback_policy="sparse", repair_limit=2)
    assert [branch.name for branch in accepted] == [branch.name for branch in branches]
    assert [branch.policy for branch in accepted] == ["none", "sparse", "none", "none"]
    assert accepted[1].seed != branches[1].seed and accepted[1].prepared.offset is offset
    assert collisions == [dict(branch="branch-02", rejected_starts=1, refilled=True)]

    later = Branch("branch-later", source, 9, policy="sparse")
    accepted, collisions = branch_execution.refill_exact_starts(None, design, [later], seen,
        attempts=2, fallback_policy="sparse", repair_limit=2)
    assert not accepted
    assert collisions == [dict(branch="branch-later", rejected_starts=3, refilled=False)]
