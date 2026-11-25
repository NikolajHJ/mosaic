import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import protenix_synthesis_study as pilot
from latentmol.design import protenix_synthesis_waves as waves
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_branches import Branch, BranchResult
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch


def point(smiles, value=0., fresh=2., *, step=0, branch="original"):
    hard = HardGraphBatch(torch.ones((1, 2), dtype=torch.long),
                          torch.full((1, 2), 2, dtype=torch.long),
                          torch.zeros((1, 2, 2), dtype=torch.long))
    return SearchPoint(Candidate(smiles, torch.tensor([[value]]), hard, value, branch, step),
                       torch.ones(2, dtype=torch.bool), fresh, (branch,))


def test_pareto_selection_preserves_best_binder_and_adds_synthesis_tradeoff():
    entries = [point("C", fresh=2.), point("N", fresh=2.1), point("O", fresh=2.5)]
    archive = {p.candidate.smiles: (p, {"chemistry": {"sa_score": sa}})
               for p, sa in zip(entries, (8., 7., 2.))}
    binding, _ = waves.select_wave_parents(archive, 2, "binding", 1.)
    pareto, _ = waves.select_wave_parents(archive, 2, "pareto", 1.)
    assert binding[0] is pareto[0] is entries[0]
    assert binding[1] is entries[1] and pareto[1] is entries[2]


def test_vina_audit_and_selection_preserve_earlier_slots_and_strict_gates(tmp_path, monkeypatch):
    from latentmol.design import protenix_vina_parents as vina

    asset = tmp_path / "native-asset"
    asset.write_bytes(b"fixture")
    ref = {"path": str(asset), "sha256": file_sha256(asset)}
    panel = vina.VinaParentPanel(dict(receptor=ref, box=ref, panel_size=3, seed=17, exhaustiveness=4))
    calls = []

    def score(_config, names):
        calls.append(names)
        return {s: dict(status="ok", affinity_kcal_mol={"O": -5., "F": -8.}[s])
                if s != "Cl" else dict(status="failed", reason="embedding") for s in names}

    monkeypatch.setattr(vina, "_native_scores", score)
    pts = [point(s, fresh=loss) for s, loss in zip(("C", "N", "O", "F", "Cl", "Br"),
                                                   (2., 2.05, 2.1, 2.15, 2.2, 2.5))]
    archive = {p.candidate.smiles: (p, {"chemistry": {"sa_score": 6.-i/5, "valid": True}})
               for i, p in enumerate(pts)}
    params = dict(binding_ceiling=2.35, apply_binding_ceiling_to_all=True, ranking="diversity")
    plain, _ = waves.select_wave_parents(archive, 3, "pareto", 1., **params)
    for role in ("audit", "select"):
        receipt = {}
        chosen, relaxations = waves.select_wave_parents(archive, 3, "pareto", 1., **params,
            vina_role=role, vina=panel, vina_receipt=receipt)
        assert not relaxations and chosen[:2] == plain[:2] == pts[:2]
        assert chosen[-1] == (pts[3] if role == "select" else plain[-1])
        assert "Br" not in receipt["scores"] and receipt["standard_parent"] == "O"
    assert calls == [["O", "F", "Cl"]]
    assert panel.accounting()["failures"] == 1 and panel.accounting()["cache_hits"] == 3
    receipt = {}
    chosen, _ = waves.select_wave_parents(archive, 3, "pareto", 1.,
        **(params | {"binding_ceiling": 1.}), vina=panel, vina_role="select", vina_receipt=receipt)
    assert chosen == plain and receipt["scores"] == {} and not receipt["changed"]


def test_parent_spacing_and_pareto_ceiling_protect_elite_and_log_budget_fallback():
    entries = [point("C", fresh=2., step=0), point("N", fresh=2.1, step=5),
               point("O", fresh=2.2, step=120), point("F", fresh=2.5, step=220)]
    archive = {p.candidate.smiles: (p, {"chemistry": {"sa_score": sa}})
               for p, sa in zip(entries, (8., 7., 6., 1.))}
    chosen, relaxations = waves.select_wave_parents(archive, 2, "pareto", 1.,
        min_step_gap=100, history={"original": {0}}, binding_ceiling=2.35)
    assert chosen == [entries[0], entries[2]]
    assert relaxations == [{"smiles": "C", "reasons": ["protected_best_spacing_exemption"]}]
    chosen, relaxations = waves.select_wave_parents(archive, 4, "pareto", 1.,
        min_step_gap=100, history={"original": {0}}, binding_ceiling=2.35)
    assert len(chosen) == 4
    reasons = {reason for item in relaxations for reason in item["reasons"]}
    assert "source_step_spacing" in reasons and "binding_ceiling" in reasons


def test_recovery_replays_witnessed_invalid_latent_without_repairing_it():
    original = point("CC")
    invalid = point(None, 3., None, step=8, branch="cycle-03/branch")
    witness = SearchPoint(invalid.candidate, invalid.active, None, invalid.ancestry,
                          2, "CN", "a" * 64)
    terminal = point(None, 4., None, step=10, branch="cycle-03/branch").candidate
    row = BranchResult(Branch("previous", original, 1), [original.candidate], [witness], {},
                       frozenset(), terminal)
    recovered, source = waves.recovery_parent(row, original)
    assert recovered is witness and source == "near_valid_witness"
    assert recovered.candidate.smiles is None
    assert recovered.candidate.latent.item() == 3.
    valid_terminal = point("CO", 5., None, step=10, branch="cycle-03/branch").candidate
    row = BranchResult(row.branch, row.valid, row.invalid, {}, frozenset(), valid_terminal)
    recovered, source = waves.recovery_parent(row, original)
    assert recovered.candidate is valid_terminal and source == "valid_terminal"


def test_common_diversity_rank_and_ceiling_isolate_frontier_filtering(monkeypatch):
    entries = [point("C", fresh=2.), point("N", fresh=2.2), point("O", fresh=2.1),
               point("F", fresh=2.4), point("Cl", fresh=2.3)]
    archive = {p.candidate.smiles: (p, {"chemistry": {"sa_score": sa}})
               for p, sa in zip(entries, (8., 7., 6., 1., 3.))}
    monkeypatch.setattr(waves, "_morgan_fingerprints", lambda names: [(name, name) for name in names])
    similarities = {"N": .1, "O": .8, "F": 0., "Cl": .2}
    monkeypatch.setattr(waves.DataStructs, "BulkTanimotoSimilarity",
                        lambda name, others: [similarities.get(name, 0.)] * len(others))
    kwargs = dict(ranking="diversity", binding_ceiling=2.35, apply_binding_ceiling_to_all=True)
    binding, _ = waves.select_wave_parents(archive, 2, "binding", 1., **kwargs)
    pareto, _ = waves.select_wave_parents(archive, 2, "pareto", 1., **kwargs)
    # Binding chooses the most diverse eligible point N; O dominates N, so
    # the same ranking after frontier filtering chooses Cl. Both reject F.
    assert binding == [entries[0], entries[1]]
    assert pareto == [entries[0], entries[4]]
    legacy, _ = waves.select_wave_parents(archive, 2, "binding", 1.)
    assert legacy == [entries[0], entries[2]]
    slots, _ = waves.select_wave_parents(archive, 3, "sa-slot", 1., **kwargs)
    assert slots == [entries[0], entries[4], entries[1]]


def test_joint_recovery_replays_exact_intermediate_and_excludes_ineligible_archive_points():
    original = point("CC")
    middle = point("CO", 3., 2.2, step=278, branch="cycle-03/child")
    terminal = point("CN", 4., None, step=500, branch="cycle-03/child").candidate
    ineligible = [point("N", 5., 2.5, step=100, branch=middle.candidate.branch),
                  point("O", 6., 2.1, step=150, branch="other-child"),
                  point("F", 7., 2.1, step=200, branch=middle.candidate.branch)]
    row = BranchResult(Branch("child", original, 1), [middle.candidate, terminal], [], {},
                       frozenset({200}), terminal)
    archive = {p.candidate.smiles: (p, {"status": "scored", "chemistry": {"valid": True, "sa_score": sa}})
               for p, sa in zip([middle, *ineligible], (4.2, 1., 2., 3.))}
    recovered, kind = waves.recovery_parent(row, original, strategy="joint", archive=archive)
    assert recovered is middle and kind == "joint_archive"
    assert recovered.candidate.latent.item() == 3.
    default, kind = waves.recovery_parent(row, original)
    assert default.candidate is terminal and kind == "valid_terminal"
    archive[middle.candidate.smiles][1]["status"] = "failed"
    recovered, kind = waves.recovery_parent(row, original, strategy="joint", archive=archive)
    assert recovered.candidate is terminal and kind == "joint_fallback_valid_terminal"


@pytest.mark.parametrize("joint_recovery", [False, True])
def test_recovery_configuration_validation(joint_recovery):
    original = point("C")
    row = BranchResult(Branch("child", original, 1), [], [], {}, frozenset(), None)
    with pytest.raises(ValueError, match="recovery"):
        waves.recovery_parent(row, original, strategy="other" if joint_recovery else "joint",
                              binding_ceiling=2.35 if joint_recovery else float("nan"))


def test_disjoint_founder_groups_charge_scores_and_exclude_previous_search(tmp_path, monkeypatch):
    panel = tmp_path / "parents.json"
    panel.write_text(json.dumps({"historical_valid_smiles": ["CO"]}))
    variants = {arm: {"ratios": {"synfrag": .05}} for arm in ("hard", "easy")}
    cfg = dict(study="synthesis_pilot", waves=3, parent_count=1, fresh_per_parent=2,
               design=dict(updates=2, schedule_horizon=500, occupied_atoms=2, learning_rate=60.,
                           chemistry_initial=1., chemistry_final=1., chemistry_hold=0, chemistry_ramp_end=500,
                           temperature_initial=1., temperature_final=1., noise_initial=0., noise_zero_at=100),
               founder_groups={"hard": [0], "easy": [1]}, sa_mode="pre_smoothing",
               parent_panel=dict(path=str(panel), sha256=file_sha256(panel)), guided_controls=True,
               synthesis_assets={"synfrag": dict(path="unused", sha256="b" * 64)},
               variants=variants, tasks={"one": {"seed": 17}})
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=2)
    parents = [point("CC", fresh=1.), point("NN", fresh=None)]
    monkeypatch.setattr(pilot, "load_pilot_parents", lambda *_: (parents, []))
    monkeypatch.setattr(waves, "load_models", lambda *_: {})
    starts = {}

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        b, = branches
        phase = int(root.name[-2:])
        starts[phase, b.name] = (b.parent.candidate.smiles, b.parent.fresh_loss)
        child = point("CO" if phase == 1 else "CCO", fresh=None, branch=root.name + "/" + b.name).candidate
        return [BranchResult(b, [child], [], dict(updates=design.updates, valid_states=3), frozenset(), child)]

    checked = []
    def readout(c, active, models, spec, mode):
        checked.append(mode)
        return dict(smiles=c.smiles, chemistry={"valid": True, "sa_score": 4.})

    def fresh(root, rows):
        return [dict(r, status="embedding_failed" if r["smiles"] == "NN" else "scored",
                     loss=None if r["smiles"] == "NN" else 2.1) for r in rows]

    monkeypatch.setattr(waves, "execute_branches", execute)
    monkeypatch.setattr(pilot, "_candidate_readout", readout)
    result = waves.run_waves(decoder, lambda *_: SimpleNamespace(close=lambda: None), fresh,
                             cfg, "one", tmp_path / "run", "fixture")
    assert result["planned"]["fresh_attempts_per_arm"] == 7
    assert starts[1, "parent-00-hard-00"] == ("CC", 2.1)
    assert starts[1, "parent-00-easy-00"] == ("NN", None)
    assert set(checked) == {"pre_smoothing"}
    for arm in variants:
        row = result["arms"][arm]
        assert row["fresh_attempts"] == 3 and row["founder_fresh_attempts"] == 1
        assert row["joint_quality"]["sa<=5,loss<=2.3"]["distinct"] == 2
        assert row["joint_quality_new_to_history"]["sa<=5,loss<=2.3"]["distinct"] == 1


def test_shared_audited_founders_charge_once_and_record_vina_choices(tmp_path, monkeypatch):
    from latentmol.design import protenix_vina_parents as vina

    path = tmp_path / "panel.json"
    path.write_text(json.dumps({"historical_valid_smiles": ["CO", "CN"]}))
    asset = {"path": str(path), "sha256": file_sha256(path)}
    config = dict(study="synthesis_pilot", waves=3, parent_count=2, fresh_per_parent=2,
        design=dict(updates=2, schedule_horizon=500, occupied_atoms=2, learning_rate=60.,
                    chemistry_initial=1., chemistry_final=1., chemistry_hold=0, chemistry_ramp_end=500,
                    temperature_initial=1., temperature_final=1., noise_initial=0., noise_zero_at=100),
        founder_policy="audited-near-miss", sa_mode="pre_smoothing", guided_controls=True,
        parent_panel=asset, synthesis_assets={"synfrag": dict(path="unused", sha256="b" * 64)},
        variants={role: {"vina_parent_role": role} for role in ("audit", "select")},
        vina_parent_panel=dict(receptor=asset, box=asset, seed=1701, exhaustiveness=4, panel_size=8),
        tasks={"one": {"seed": 17}})
    parents = [point("CO"), point("CN")]
    monkeypatch.setattr(pilot, "load_pilot_parents", lambda *_: (parents, []))
    monkeypatch.setattr(waves, "load_models", lambda *_: {})
    monkeypatch.setattr(pilot, "_candidate_readout", lambda c, *_:
                        dict(smiles=c.smiles, chemistry=dict(valid=True, sa_score=4.)))
    monkeypatch.setattr(vina, "_native_scores", lambda _, names:
                        {s: dict(status="ok", affinity_kcal_mol=-9. if s == "F" else -5.) for s in names})
    calls, starts = [], {}

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        rows = []
        for i, b in enumerate(branches):
            starts[root.name, b.name] = b.parent.candidate.smiles
            names = (("C", "N"), ("O", "F"))[i]
            children = [point(name, fresh=None, step=j, branch=root.name + "/" + b.name).candidate
                        for j, name in enumerate(names)]
            rows.append(BranchResult(b, children, [], dict(updates=design.updates, valid_states=3),
                                     frozenset(), children[-1]))
        return rows

    losses = dict(C=2., N=2.1, O=2.2, F=2.25, CO=2.34, CN=2.35)
    def fresh(root, records):
        calls.append((root.name, [r["smiles"] for r in records]))
        return [dict(r, status="scored", loss=losses[r["smiles"]]) for r in records]

    monkeypatch.setattr(waves, "execute_branches", execute)
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=2)
    result = waves.run_waves(decoder, lambda *_: SimpleNamespace(close=lambda: None), fresh,
                             config, "one", tmp_path / "run", "fixture")
    assert calls[0] == ("founder-fresh", ["CO", "CN"])
    assert result["planned"]["fresh_attempts"] == 26
    assert result["planned"]["fresh_attempts_per_arm"] == 14
    assert result["physical_discovery_fresh_attempts"] == 10
    assert starts["cycle-02", "parent-00-audit-00"] == starts["cycle-02", "parent-00-select-00"] == "C"
    assert starts["cycle-02", "parent-01-audit-00"] == "N"
    assert starts["cycle-02", "parent-01-select-00"] == "F"
    assert result["cycles"][1]["arms"]["select"]["vina_decision"]["changed"]
    for arm in ("audit", "select"):
        record = result["arms"][arm]
        assert record["fresh_attempts"] == 6 and record["founder_fresh_attempts"] == 2
        assert record["vina_new_hits"]["eligible"] == 4
        assert all(r["vina"]["status"] == "ok" for r in record["vina_new_hits"]["candidates"])


@pytest.mark.parametrize("joint_recovery", [False, True])
def test_wave_budgets_shared_starts_independent_scoring_and_near_valid_recovery(tmp_path, monkeypatch,
                                                                            joint_recovery):
    panel = tmp_path / "parents.json"
    panel.write_text("fixture")
    cfg = dict(study="synthesis_pilot", waves=2, recovery_wave=2, chemistry_profile=[1., 2.],
        guided_controls=True, parent_count=2, fresh_per_parent=2,
        design=dict(updates=2, schedule_horizon=500, occupied_atoms=2, learning_rate=60.,
                    chemistry_initial=1., chemistry_final=1., chemistry_hold=0, chemistry_ramp_end=500,
                    temperature_initial=1., temperature_final=1., noise_initial=0., noise_zero_at=100),
        parent_panel=dict(path=str(panel), sha256=file_sha256(panel)),
        synthesis_assets={"synfrag": dict(path="unused.pt", sha256="b" * 64)},
        variants={"empty": {"ratios": {"synfrag": .1}}, "constant": {"ratios": {"synfrag": .1625}},
                  "focused": {"ratios": {"synfrag": .1}, "wave_schedule": [
                      {"ratios": {"synfrag": .4}}, {"ratios": {"synfrag": .05}}]},
                  "pareto": {"ratios": {"synfrag": .1}, "parent_selection": "pareto"}},
        tasks={"one": dict(seed=123)})
    if joint_recovery:
        cfg["variants"]["pareto"]["recovery_start"] = "joint"
    else:
        cfg["variants"]["empty"]["independent_final_slot"] = True
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=2)
    parents = [point("CC", fresh=2.), point("NN", 1., 2.1)]
    draws = []
    def gaussian(_decoder, design, seed):
        draws.append(seed)
        return point(None, 10., None, branch=f"prior-{seed}")
    monkeypatch.setattr(waves, "gaussian_start", gaussian)
    seen, scored, closed_scorers = {}, {}, []

    class Frozen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.closed = False

        def close(self):
            self.closed = True

    model = Frozen()
    monkeypatch.setattr(pilot, "load_pilot_parents", lambda *_: (parents, []))
    monkeypatch.setattr(waves, "load_models", lambda *_: {"synfrag": model})

    def readout(candidate, *_):
        return dict(smiles=candidate.smiles, step=candidate.step, branch=candidate.branch,
                    chemistry={"valid": bool(candidate.smiles), "sa_score": 4., "qed": .5, "rings": 0})

    monkeypatch.setattr(pilot, "_candidate_readout", readout)

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        arm = branches[0].name.split("-")[2]
        wave = int(root.name[-2:])
        assert design.chemistry_initial == design.chemistry_final == wave
        assert kwargs["repair_limit"] == 2 and kwargs["invalid_limit"] == 8
        rows = []
        for index, branch in enumerate(branches):
            seen[(wave, arm, index)] = branch
            name = "CO" if index == 0 else ("CN" if wave == 1 else "CCO")
            child = point(name, float(wave + index), None, step=2,
                          branch=f"{root.name}/{branch.name}").candidate
            invalid = []
            terminal = child
            if arm == "focused" and wave == 1 and index == 0:
                source = point(None, 7., None, step=1, branch=child.branch)
                invalid = [SearchPoint(source.candidate, source.active, None, source.ancestry,
                                       2, "CO", "c" * 64)]
                terminal = point(None, 9., None, step=2, branch=child.branch).candidate
            rows.append(BranchResult(branch, [] if arm == "empty" else [child], invalid,
                                     dict(updates=2, valid_states=3), frozenset(), terminal))
        return rows

    def fresh(root, records):
        arm = root.name.removeprefix("fresh-")
        scored.setdefault(arm, []).extend(row["smiles"] for row in records)
        return [dict(row, status="failed" if row["smiles"] == "CN" else "scored",
                     loss=None if row["smiles"] == "CN" else 2.05) for row in records]

    def make_scorer(_spec, _active, root):
        return SimpleNamespace(identity={}, close=lambda: closed_scorers.append(root.name))

    monkeypatch.setattr(waves, "execute_branches", execute)
    result = waves.run_waves(decoder, make_scorer, fresh, cfg, "one", tmp_path / "run", "checkpoint")
    assert result["status"] == "completed" and result["planned"]["updates"] == 32
    assert all(row["updates"] == 8 for row in result["arms"].values())
    assert result["arms"]["empty"]["fresh_attempts"] == 0
    for arm in ("constant", "focused", "pareto"):
        assert result["arms"][arm]["fresh_attempts"] == 3
        assert result["arms"][arm]["fresh_successes"] == 2
        assert scored[arm] == ["CO", "CN", "CCO"]  # Failed CN is cached too.
        for index in range(2):
            assert seen[(1, arm, index)].parent is parents[index]
            assert seen[(1, arm, index)].seed == seen[(1, "empty", index)].seed
    recovered = seen[(2, "focused", 0)]
    assert recovered.policy == "continuous" and recovered.parent.candidate.smiles is None
    assert recovered.parent.candidate.latent.item() == 7.
    assert closed_scorers == ["cycle-01", "cycle-02"] and model.closed
    read = json.loads((tmp_path / "run/cycle-02/batch.json").read_text())
    focus = next(row for row in read["branches"] if row["arm"] == "focused" and row["parent_index"] == 0)
    assert focus["parent_source"]["source"] == "near_valid_witness"
    if not joint_recovery:
        assert draws == [126]
        assert seen[1, "empty", 1].parent is parents[1]
        assert seen[2, "empty", 0].policy == "continuous"
        independent = seen[2, "empty", 1]
        assert independent.policy == "bootstrap" and independent.parent.candidate.latent.item() == 10.
        start = next(row["parent_source"] for row in read["branches"]
                     if row["arm"] == "empty" and row["parent_index"] == 1)
        assert start["source"] == "raw_gaussian" and start["draw_seed"] == draws[0]
        assert start["initial_valid"] is False and start["active_slots"] == [0, 1]
        assert start["displaced_continuation"]["source"] == "valid_terminal"
        assert result["arms"]["empty"]["final_slot"]["fresh_attempts"] == 0
        assert result["arms"]["constant"]["final_slot"]["fresh_successes"] == 1
    if joint_recovery:
        replay = seen[(2, "pareto", 0)]
        assert replay.parent.candidate.latent.item() == 1. and replay.policy == "continuous"
        source = next(row for row in read["branches"] if row["arm"] == "pareto" and row["parent_index"] == 0)
        assert source["parent_source"]["source"] == "joint_archive"
        assert source["parent_source"]["parent_sa"] == 4.
    assert (tmp_path / "run/cycle-02/archive-state.pt").is_file()
