import copy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import protenix_synthesis_study as study
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_branches import BranchResult
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.experiments import file_sha256
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def config():
    return dict(study="synthesis_pilot", design=dict(updates=500, schedule_horizon=500,
        occupied_atoms=3, learning_rate=60., chemistry_initial=1., chemistry_final=1.,
        chemistry_hold=0, chemistry_ramp_end=500, noise_zero_at=100,
        noise_initial=0., temperature_initial=1., temperature_final=1.),
        parent_count=3, fresh_per_parent=6, parent_panel=dict(path="parents.json", sha256="a" * 64),
        synthesis_assets=dict(synfrag=dict(path="synfrag.pth", sha256="b" * 64),
                              dfr=dict(path="dfr.pt", sha256="c" * 64)),
        variants=dict(baseline=dict(ratios={}), sfweak=dict(ratios=dict(synfrag=.1)),
                      sfstrong=dict(ratios=dict(synfrag=.5)), dfrweak=dict(ratios=dict(dfr=.1)),
                      dfrstrong=dict(ratios=dict(dfr=.5)), both=dict(ratios=dict(synfrag=.25, dfr=.25))),
        tasks={"seed-a": dict(seed=123, queue="gpuv100")})


def test_synthesis_pilot_budget_and_opt_in_validation():
    cfg = config()
    design, budget = study.resolve_pilot(cfg, "seed-a")
    assert design.learning_rate == 60 and design.noise_initial == 0
    assert budget == dict(updates=9000, branches=18, relaxed_calls=9018,
                          updates_per_arm=1500, fresh_attempts=108, fresh_attempts_per_arm=18)
    for key, value in (("noise_initial", .1), ("updates", 1001)):
        changed = copy.deepcopy(cfg)
        changed["design"][key] = value
        with pytest.raises(ValueError):
            study.resolve_pilot(changed, "seed-a")
    cfg["variants"]["both"]["ratios"]["unregistered"] = .1
    with pytest.raises(ValueError, match="variant"):
        study.resolve_pilot(cfg, "seed-a")


def test_synthesis_pilot_pairs_starts_and_matches_fresh_scores_per_parent(tmp_path, monkeypatch):
    cfg = config()
    cfg["design"]["updates"] = 2
    cfg["parent_count"] = 2
    cfg["fresh_per_parent"] = 2
    cfg["variants"] = dict(baseline=dict(ratios={}), guided=dict(ratios=dict(synfrag=.2), delay_updates=1))
    source = tmp_path / "parents.json"
    source.write_text("fixture")
    cfg["parent_panel"] = dict(path=str(source), sha256=file_sha256(source))
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=3)
    active = torch.ones(3, dtype=torch.bool)
    hard = HardGraphBatch(torch.ones((1, 3), dtype=torch.long),
                          torch.full((1, 3), 2, dtype=torch.long),
                          torch.zeros((1, 3, 3), dtype=torch.long))
    parents = [SearchPoint(Candidate(smiles, torch.tensor([[float(i)]]), hard, 2., "source", 0),
                           active, 2.5 + i, ("source",)) for i, smiles in enumerate(("CCC", "CCO"))]
    calls, seen, factories = {}, {}, []
    monkeypatch.setattr(study, "load_pilot_parents", lambda *_: (parents, []))
    monkeypatch.setattr(study, "load_models", lambda *_: {"synfrag": torch.nn.Module()})
    monkeypatch.setattr(study, "SynthesisGuidance", lambda *_args, **kwargs: kwargs)

    def execute(_decoder, _scorer, design, root, branches, **kwargs):
        factories.append(kwargs["guidance_factory"])
        rows = []
        for i, branch in enumerate(branches):
            seen[branch.name] = branch
            # Guided parent 0 retains only its founder. It must still receive a
            # real, charged fresh rescore instead of zeroing all matched pools.
            smiles = ("CCC", "N") if i == 0 else ("N", "F")
            if "guided" in branch.name and i == 0:
                smiles = smiles[:1]
            elif "guided" in branch.name:
                smiles = ("CCO",)
            valid = [Candidate(s, branch.parent.candidate.latent, hard, float(j),
                               "cycle-01/" + branch.name, j) for j, s in enumerate(smiles)]
            rows.append(BranchResult(branch, valid, [], dict(updates=design.updates, valid_states=2),
                                     frozenset(), valid[-1]))
        return rows

    def fresh(root, records):
        calls[root.name] = [row["smiles"] for row in records]
        return [dict(row, loss=2.1 + i * .1, status="scored") for i, row in enumerate(records)]

    monkeypatch.setattr(study, "execute_branches", execute)
    monkeypatch.setattr(study, "_candidate_readout", lambda c, *_: dict(smiles=c.smiles, step=c.step,
        branch=c.branch, chemistry=dict(valid=True, sa_score=4., qed=.7, rings=1)))
    result = study.run_pilot(decoder, lambda *_: SimpleNamespace(identity={}, close=lambda: None),
                             fresh, cfg, "seed-a", tmp_path / "run", "checkpoint")
    assert result["status"] == "completed"
    assert calls == {"fresh-baseline": ["CCC", "N"], "fresh-guided": ["CCC", "CCO"]}
    assert [r["fresh_attempts"] for r in result["arms"].values()] == [2, 2]
    assert [r["fresh_shortfall"] for r in result["arms"].values()] == [2, 2]
    for index in range(2):
        a, b = (seen[f"parent-{index:02d}-{arm}-00"] for arm in ("baseline", "guided"))
        assert a.parent is b.parent and a.seed == b.seed
    assert factories[0] is None and factories[1](None) == {"delay_updates": 1}
    assert result["frozen_synthesis_models_unchanged"]
    assert result["arms"]["guided"]["best_new_fresh_loss_by_sa_ceiling"]["4"] == 2.1
    assert result["arms"]["guided"]["best_novel_fresh_loss"] is None
    assert result["arms"]["guided"]["founder_rescore_successes"] == 2
    assert result["arms"]["baseline"]["fresh_novel_successes"] == 1
    branches = json.loads((tmp_path / "run/cycle-01/batch.json").read_text())["branches"]
    assert branches[0]["candidates"]["mixed"][0]["is_founder"] is True
    assert json.loads((tmp_path / "run/result.json").read_text())["planned"]["updates"] == 8


def test_independent_allowances_keep_scoring_when_another_guide_has_no_valid_candidates(tmp_path, monkeypatch):
    cfg = config()
    cfg["fresh_scoring_policy"] = "equal_allowance"
    cfg["design"]["updates"] = 2
    cfg["parent_count"] = 2
    cfg["fresh_per_parent"] = 2
    cfg["variants"] = dict(baseline=dict(ratios={}), guided=dict(ratios=dict(synfrag=.2)))
    source = tmp_path / "parents.json"
    source.write_text("fixture")
    cfg["parent_panel"] = dict(path=str(source), sha256=file_sha256(source))
    decoder = torch.nn.Module()
    decoder.spec = GraphSpec(max_nodes=2)
    hard = HardGraphBatch(torch.ones((1, 2), dtype=torch.long), torch.full((1, 2), 2),
                          torch.zeros((1, 2, 2), dtype=torch.long))
    active = torch.ones(2, dtype=torch.bool)
    parents = [SearchPoint(Candidate(s, torch.tensor([[float(i)]]), hard, 1., "source", 0),
                           active, 2.5, ("source",)) for i, s in enumerate(("CC", "CO"))]
    called = []
    monkeypatch.setattr(study, "load_pilot_parents", lambda *_: (parents, []))
    monkeypatch.setattr(study, "load_models", lambda *_: {"synfrag": torch.nn.Module()})

    def execute(_decoder, _scorer, design, _root, branches, **_kwargs):
        rows = []
        for index, branch in enumerate(branches):
            candidate = Candidate("C" if index == 0 else "N", branch.parent.candidate.latent,
                                  hard, 1., branch.name, 1)
            valid = [] if "guided" in branch.name else [candidate]
            rows.append(BranchResult(branch, valid, [], dict(updates=design.updates, valid_states=len(valid)),
                                     frozenset(), candidate))
        return rows

    def fresh(root, records):
        called.append((root.name, [r["smiles"] for r in records]))
        return [dict(r, loss=2.1 if r["smiles"] == "C" else None,
                     status="scored" if r["smiles"] == "C" else "failed") for r in records]

    monkeypatch.setattr(study, "execute_branches", execute)
    monkeypatch.setattr(study, "_candidate_readout", lambda c, *_: dict(smiles=c.smiles,
        chemistry=dict(valid=True, sa_score=4., qed=.7, rings=0)))
    output = tmp_path / "run"
    result = study.run_pilot(decoder, lambda *_: SimpleNamespace(identity={}, close=lambda: None),
                             fresh, cfg, "seed-a", output, "fixture-checkpoint")
    assert called == [("fresh-baseline", ["C", "N"])]
    assert result["fresh_scoring_policy"] == "equal_allowance"
    assert result["arms"]["baseline"]["fresh_attempts"] == 2
    assert result["arms"]["baseline"]["fresh_successes"] == 1
    assert result["arms"]["baseline"]["fresh_failures"] == 1
    assert result["arms"]["guided"]["fresh_attempts"] == 0
    assert result["arms"]["guided"]["fresh_shortfall"] == 4
    assert all(row["fresh_attempt_allowance"] == 4 for row in result["arms"].values())
    assert json.loads((output / "run.json").read_text())["fresh_scoring_policy"] == "equal_allowance"


def test_anchor_control_bank_requires_a_valid_checksum_before_loading():
    cfg = config()
    cfg["anchor_control_bank"] = dict(path="control.json", sha256="not-a-sha")
    with pytest.raises(ValueError, match="SHA-256"):
        study.resolve_pilot(cfg, "seed-a")


def test_saved_synthesis_founder_replays_exact_labels_and_rejects_changed_identity(tmp_path):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = GraphSpec(max_nodes=3)
            self.config = SimpleNamespace(latent=1)
            self.register_buffer("atoms", torch.tensor([[1, 1, 3]]))
            self.register_buffer("charges", torch.full((1, 3), 2, dtype=torch.long))
            self.register_buffer("bonds", torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]))

        def decode(self, _point):
            return DecoderOutput(*(torch.nn.functional.one_hot(value, categories).float() * 40
                for value, categories in ((self.atoms, 10), (self.charges, 5), (self.bonds, 5))))

    decoder = Decoder()
    cfg = config()
    cfg["parent_count"] = 1
    data = dict(version=1, checkpoint_sha256="checkpoint", decoder_sha256=study._model_identity(decoder),
                graph_spec=decoder.spec.to_dict(), points=[dict(smiles="CCO", fresh_loss=2.1,
                relaxed_loss=1.8, latent=[[0.]], active_slots=[0, 1, 2], branch="source", step=4,
                labels={key: getattr(decoder, key)[0].tolist() for key in ("atoms", "charges", "bonds")})])
    path = tmp_path / "parents.json"
    path.write_text(json.dumps(data))
    cfg["parent_panel"] = dict(path=str(path), sha256=file_sha256(path))
    design, _ = study.resolve_pilot(cfg, "seed-a")
    points, records = study.load_pilot_parents(decoder, cfg, design, "checkpoint")
    assert points[0].candidate.smiles == "CCO" and records[0]["step"] == 4
    data["points"][0]["labels"]["atoms"][0] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        study.load_pilot_parents(decoder, cfg, design, "checkpoint")
    cfg["parent_panel"]["sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="does not replay"):
        study.load_pilot_parents(decoder, cfg, design, "checkpoint")


def test_native_hard_synthesis_reference_accepts_scalar_difficulty():
    class Model(torch.nn.Module):
        def forward(self, *_):
            return dict(difficulty=torch.tensor([.3]), raw_score=torch.tensor([.7]))

        def reference_score(self, _mol):
            return .25

    spec = GraphSpec(max_nodes=1)
    hard = HardGraphBatch(torch.ones((1, 1), dtype=torch.long), torch.full((1, 1), 2),
                          torch.zeros((1, 1, 1), dtype=torch.long))
    candidate = Candidate("C", torch.zeros((1, 1)), hard, 1., "source", 0)
    row = study._candidate_readout(candidate, torch.ones(1, dtype=torch.bool),
                                    {"synfrag": Model(), "dfr": Model()}, spec)
    assert row["native_hard_scores"] == {"synfrag": dict(difficulty=.25, raw_score=.75),
                                         "dfr": dict(difficulty=.25, raw_score=.25)}


def test_joint_shortlist_preserves_binding_leaders_and_easy_witness(monkeypatch):
    rows = [SimpleNamespace(smiles=str(i), relaxed_loss=float(i), step=i) for i in range(8)]
    monkeypatch.setattr(study, "select_candidates", lambda rows, n, *_: {"mixed": rows[:n]})
    monkeypatch.setattr(study, "molecule_readout", lambda smi: {"sa_score": 9-int(smi), "valid": True})
    assert [r.smiles for r in study.synthesis_shortlist(rows, 6, "binding")] == list("012345")
    assert [r.smiles for r in study.synthesis_shortlist(rows, 6, "binding-sa")] == list("012765")
    with pytest.raises(ValueError, match="shortlist"):
        study.synthesis_shortlist(rows, 6, "unknown")


def test_joint_metrics_exclude_failures_founders_and_duplicate_scores():
    def row(smi, loss, sa):
        return dict(smiles=smi, loss=loss, chemistry=dict(valid=True, sa_score=sa))
    rows = [row("CCO", 2.2, 3.), row("CCO", 2.15, 3.), row("CCN", None, 2.),
            row("CCCC", 2.1, 2.), row("c1ccccc1", 2.5, 2.)]
    metrics = study.joint_quality_metrics(rows, {"CCCC"})
    assert metrics["sa<=5,loss<=2.3"]["distinct"] == 1
    assert metrics["sa<=5,loss<=2.3"]["best_loss"] == 2.15
    assert metrics["sa<=5,loss<=2.3"]["diverse"] == 1


def test_new_anchor_libraries_are_checked_before_model_loading():
    cfg = config()
    cfg["anchor_libraries"] = {"low_sa": dict(path="bank.json", sha256="bad")}
    with pytest.raises(ValueError, match="SHA-256"):
        study.resolve_pilot(cfg, "seed-a")
    cfg["anchor_libraries"]["low_sa"]["sha256"] = "a"*64
    cfg["variants"]["sfweak"].update(anchor_ratio=.05, anchor_library="missing")
    with pytest.raises(ValueError, match="Unknown anchor"):
        study.resolve_pilot(cfg, "seed-a")


def test_aligned_bank_factory_mask_calibration_and_geometry_logging(tmp_path):
    bank = dict(checkpoint_sha256="ckpt", active_slots=[0, 1, 3], scale=1.,
                latents=[[0., 0.], [1., 0.], [0., 1.], [2., 2.], [-1., 1.]])
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    cfg = dict(anchor_libraries={"local": dict(path=str(path), sha256=file_sha256(path),
                                              target_effective_anchors=2.)})
    mask = torch.tensor([True, True, False, True])
    parents = [SimpleNamespace(active=mask, candidate=SimpleNamespace(latent=torch.tensor([[2., 1.]])))]
    libraries = study.load_anchor_libraries(cfg, parents, "ckpt")
    factory = study.guidance_factory({}, dict(ratios={}, anchor_ratio=.05, anchor_library="local",
        controller=dict(max_ratio=.05)), libraries)
    z = torch.tensor([[2., 1.]], requires_grad=True)
    gradient, meta = factory(None).evaluate(None, mask, z, torch.ones_like(z), 0)
    assert gradient.isfinite().all() and gradient.norm() > 0
    assert meta["models"]["anchor"]["latent_geometry"]["effective_anchors"] == pytest.approx(2., rel=1e-5)
    assert libraries["local"].latents.grad is None


def test_nonanchor_controls_ignore_unselected_named_libraries():
    libraries = {"low-local": object()}
    assert study.guidance_factory({}, dict(ratios={}), libraries) is None
    factory = study.guidance_factory({"synfrag": torch.nn.Identity()},
                                    dict(ratios={"synfrag": .05}), libraries)
    assert factory(None).ratios == {"synfrag": .05}


def test_standard_start_and_shortlist_use_distinct_audited_hard_molecules(monkeypatch):
    from rdkit import Chem
    from latentmol.evaluation import synthesis_accessibility

    # Two enantiomers have identical non-chiral fingerprints; only the lower
    # binding loss should receive a slot, even with fractionally different SA.
    sa = {"C[C@H](N)O": 4.5, "C[C@@H](N)O": 4.6, "c1ccccc1": 5.0, "CCCO": 5.1, "CCN": 6.1}
    monkeypatch.setattr(synthesis_accessibility, "audited_sa_score",
                        lambda mol: sa[Chem.MolToSmiles(mol)])
    samples = [("C[C@H](N)O", 2.33), ("C[C@@H](N)O", 2.34), ("c1ccccc1", 2.4),
               ("CCCO", 2.31), ("CCN", 2.32)]
    points = [dict(smiles=s, fresh_loss=v, latent=[], labels={}, active_slots=[]) for s, v in samples]
    points += [dict(points[-1], smiles="C", fresh_loss=2.3), dict(points[-1], smiles="N", fresh_loss=None)]
    assert [p["smiles"] for p in study.select_near_miss_founders(points, 2)] == ["C[C@H](N)O", "c1ccccc1"]
    with pytest.raises(ValueError, match="Only 2 of 3"):
        study.select_near_miss_founders(points, 3)
    candidates = [SimpleNamespace(smiles=s, relaxed_loss=v, step=i) for i, (s, v) in enumerate(samples)]
    candidates.append(SimpleNamespace(smiles="c1ccccc1", relaxed_loss=2.2, step=20))
    chosen = study.synthesis_shortlist(candidates, 4, "audited-binding-diverse", "pre_smoothing")
    assert [(c.smiles, c.step) for c in chosen] == [("c1ccccc1", 20), ("C[C@H](N)O", 0), ("CCCO", 3)]
    with pytest.raises(ValueError, match="pre_smoothing"):
        study.synthesis_shortlist(candidates, 4, "audited-binding-diverse")


def test_standard_plan_single_and_paired_discovery_confirmations():
    standard = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-synthesis-standard.json"
    cfg = json.loads(standard.read_text())
    design, budget = study.resolve_pilot(cfg, "seed-1")
    assert (design.learning_rate, design.updates, design.max_step_norm, design.noise_initial) == (60, 500, .25, 0)
    assert (budget["updates"], budget["fresh_attempts"], budget["confirmation_fresh_attempts"]) == (4500, 57, 6)
    cfg["variants"]["paired"] = dict(cfg["variants"]["standard"])
    _, budget = study.resolve_pilot(cfg, "seed-1")
    assert (budget["updates"], budget["fresh_attempts"], budget["confirmation_fresh_attempts"]) == (9000, 111, 12)
    assert budget["fresh_attempts_per_arm"] == 57


def test_prepare_standard_panel_reads_valid_intermediates_and_replayable_parents(tmp_path):
    prepare = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-operations/prepare_synthesis_panel.py"
    extend = runpy.run_path(str(prepare))["extend_panel"]
    base = dict(version=1, graph_spec={}, checkpoint_sha256="ckpt", decoder_sha256="decoder",
                points=[dict(smiles="CCC")], historical_valid_smiles=["CCO"])
    root = tmp_path / "example"
    branch = root / "cycle-01/child"
    branch.mkdir(parents=True)
    run = {k: base[k] for k in ("graph_spec", "checkpoint_sha256", "decoder_sha256")}
    (root / "run.json").write_text(json.dumps(dict(run, parents=[], job_id="123", variants={"arm": {}},
                                                   runner="synthesis_waves")))
    (root / "result.json").write_text(json.dumps(dict(status="completed", cycles=[{"cycle": 1}], planned={"branches": 1})))
    (branch.parent / "batch.json").write_text(json.dumps(dict(branches=[dict(name="child", summary={"updates": 1})],
                                                             native_identity=dict(pocket_sha256="brd4"))))
    metrics = [dict(step=0, raw_valid=False, smiles=None), dict(step=1, raw_valid=True, smiles="CO")]
    path = branch / "metrics.jsonl"
    path.write_text("\n".join(map(json.dumps, metrics)))
    point = dict(smiles="CO", fresh_loss=2.31, relaxed_loss=2.02, branch="child", step=1, ancestry=["test"],
                 active=torch.tensor([True, True]), latent=torch.zeros(1, 1),
                 hard={k: torch.ones(1, 2, dtype=torch.long) for k in ("atoms", "charges", "bonds")})
    torch.save(dict(wave=1, archives=dict(arm={"CO": point})), branch.parent / "archive-state.pt")
    history = extend(base, [], [root], "brd4")
    assert history["points"] == base["points"] and history["historical_valid_smiles"] == ["CCC", "CCO", "CO"]
    expanded = extend(base, [root], [], "brd4")
    assert expanded["points"][-1]["smiles"] == "CO" and expanded["points"][-1]["provenance"]["arm"] == "arm"
    with pytest.raises(ValueError, match="protein pocket"):
        extend(base, [root], [], "other")
    path.write_text(json.dumps(metrics[0]))
    with pytest.raises(ValueError, match="Incomplete"):
        extend(base, [], [root], "brd4")
