"""Local toy regressions for shared-graph latent design, not native qualification."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design.latent_protenix import (LatentDesignConfig, choose_active,
    evaluate_point, latent_step, run_design, schedules)
from latentmol.representation import DecoderOutput, GraphSpec


class ToyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spec=GraphSpec(max_nodes=3)
        self.config=SimpleNamespace(latent=4)
        n,a,q,e=3,len(self.spec.atomic_numbers),len(self.spec.formal_charges),len(self.spec.bond_names)
        atoms=torch.full((1,n,a),-3.);atoms[...,1]=5.
        charges=torch.full((1,n,q),-3.);charges[...,2]=5.
        bonds=torch.full((1,n,n,e),-3.);bonds[...,0]=4.
        # Each central carbon has six bonds: structural discovery must repair.
        for i,j in [(0,1),(1,2)]:bonds[0,i,j,0]=bonds[0,j,i,0]=-3.;bonds[0,i,j,3]=bonds[0,j,i,3]=5.
        self.shapes=[v.shape for v in (atoms,charges,bonds)]
        self.register_buffer('base',torch.cat([v.flatten() for v in (atoms,charges,bonds)]))
        g=torch.Generator().manual_seed(7301)
        self.projection=torch.nn.Parameter(torch.randn(4,len(self.base),generator=g)*.13)

    def decode(self,z):
        values=(self.base+z@self.projection).flatten();out=[];offset=0
        for shape in self.shapes:
            count=math_prod(shape);out.append(values[offset:offset+count].reshape(shape));offset+=count
        return DecoderOutput(*out)


def math_prod(shape):
    result=1
    for n in shape:result*=n
    return result


class ToyScorer:
    identity={'model':'toy','context':'fixed'}
    closed=False
    def __call__(self,graph):
        # All represented branches can affect this scalar; no chemical conversion.
        loss=(graph.atoms[...,1].sum()*.2 + graph.charges[...,0].sum()*.4
              +graph.bonds[...,0].sum()*.07)
        return loss,{'contact_score':-float(loss.detach())}
    def close(self):self.closed=True


def config(**values):
    return LatentDesignConfig(occupied_atoms=3,updates=4,snapshot_interval=2,
                             checkpoint_interval=2,**values)


@pytest.mark.parametrize('step,weight,temp,noise',[(0,1,.7,.4),(20000,1,.58,.4/3),
    (25000,10**.5,.55,.4/6),(30000,10,.52,0),(50000,10,.4,0)])
def test_absolute_schedules_are_not_compressed_by_pilot(step,weight,temp,noise):
    d=schedules(config(),step)
    assert d['chemistry_weight']==pytest.approx(weight)
    assert d['temperature']==pytest.approx(temp)
    assert d['noise_ratio']==pytest.approx(noise)


def test_fixed_mask_and_temperature_shared_by_binding_and_structural_loss():
    model=ToyDecoder().eval().requires_grad_(False)
    z=torch.tensor([[.2,-.1,.3,.4]])
    active=torch.ones(3,dtype=torch.bool)
    result=evaluate_point(model,ToyScorer(),z,active,config(),0)
    assert torch.equal(result['graph'].atoms[...,0],torch.zeros(1,3))
    assert result['repair']['status']=='witness_found'
    assert result['repair_detail']['normalization']=='mean_per_decision'
    assert result['repair_detail']['decisions']=={'atoms':3,'charges':3,'bonds':3}
    assert result['gradient_norms']['binding']>0
    assert result['gradient_norms']['chemistry']>0
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for p in model.parameters())


def test_target_held_fixed_when_reusing_same_trial():
    model=ToyDecoder().eval().requires_grad_(False);z=torch.randn(1,4)
    active=torch.ones(3,dtype=torch.bool)
    first=evaluate_point(model,ToyScorer(),z,active,config(),0)
    trial=evaluate_point(model,ToyScorer(),z+.001,active,config(),0,target_bundle=first['target_bundle'])
    assert trial['repair_detail']['target_sha256']==first['repair_detail']['target_sha256']
    assert trial['schedule']==first['schedule']


def test_noise_is_scaled_by_binding_only_latent_step_and_bounded():
    cfg=config(learning_rate=.3,max_step_norm=.1)
    z=torch.zeros(1,4);binding=torch.ones_like(z);combined=binding*100
    result,row=latent_step(z,binding,combined,cfg,0,torch.Generator().manual_seed(1))
    assert row['noise_norm']==pytest.approx(.4*.3*2)
    assert result.norm()==pytest.approx(.1)
    assert row['bound_factor']<1
    _,zero=latent_step(z,binding*0,combined,cfg,0,torch.Generator().manual_seed(1))
    assert zero['noise_norm']==0


def test_detached_native_objective_is_rejected():
    model=ToyDecoder().requires_grad_(False)
    def detached(graph):return torch.tensor(1.),{}
    with pytest.raises(ValueError,match='detached'):
        evaluate_point(model,detached,torch.randn(1,4),torch.ones(3,dtype=torch.bool),config(),0)


def test_top_slots_selected_once_with_stable_ties():
    model=ToyDecoder();decoded=model.decode(torch.zeros(1,4))
    assert choose_active(decoded,model.spec,2).tolist()==[True,True,False]


def test_segment_resume_preserves_noise_latents_and_raw_invalid_graphs(tmp_path):
    model=ToyDecoder();cfg=config()
    factory=lambda spec,active:ToyScorer()
    full=run_design(model,factory,cfg,tmp_path/'full',checkpoint_sha256='toy-checkpoint')
    first=run_design(model,factory,cfg,tmp_path/'part1',checkpoint_sha256='toy-checkpoint',max_updates=2)
    second=run_design(model,factory,cfg,tmp_path/'part2',checkpoint_sha256='toy-checkpoint',resume=first['state'])
    complete_state=torch.load(full['state'],weights_only=True)
    resumed_state=torch.load(second['state'],weights_only=True)
    assert torch.equal(complete_state['latent'],resumed_state['latent'])
    assert torch.equal(complete_state['noise_rng'],resumed_state['noise_rng'])
    assert resumed_state['step']==4
    assert resumed_state['trajectory_statistics']==complete_state['trajectory_statistics']
    assert full['technical_checks']['frozen_decoder_unchanged']
    candidates=[json.loads(line) for line in (tmp_path/'full/arms/latent/candidates.jsonl').read_text().splitlines()]
    assert len(candidates)==5 and candidates[0]['valid'] is False
    assert candidates[0]['labels']
    assert len(list((tmp_path/'full/arms/latent/raw-blocks').glob('*.npz')))>0
    assert (tmp_path/'full/weight-probes.json').exists()
    with pytest.raises(FileExistsError):run_design(model,factory,cfg,tmp_path/'full',checkpoint_sha256='toy-checkpoint')


def test_resume_rejects_changed_context(tmp_path):
    model=ToyDecoder();cfg=config()
    first=run_design(model,lambda s,a:ToyScorer(),cfg,tmp_path/'part1',checkpoint_sha256='c',max_updates=1)
    scorer=ToyScorer();scorer.identity={'model':'toy','context':'changed'}
    with pytest.raises(ValueError,match='context identity'):
        run_design(model,lambda s,a:scorer,cfg,tmp_path/'part2',checkpoint_sha256='c',resume=first['state'])
    assert scorer.closed


def test_new_warm_start_preserves_point_slots_and_global_schedule(tmp_path):
    model=ToyDecoder();cfg=replace(config(),updates=184,noise_initial=0.,learning_rate=.1)
    point=torch.tensor([[.2,-.1,.4,.3]])
    active=torch.ones(3,dtype=torch.bool)
    result=run_design(model,lambda s,a:ToyScorer(),cfg,tmp_path/'warm',checkpoint_sha256='c',
        start_state=dict(latent=point,active=active,step=180,provenance={'step':180,'kind':'fixture'}))
    state=torch.load(result['state'],weights_only=True)
    assert result['start_step']==180 and result['completed_step']==184 and result['updates_this_segment']==4
    assert torch.equal(state['initial_latent'],point) and torch.equal(state['active'],active)
    metadata=json.loads((tmp_path/'warm/run.json').read_text())
    assert metadata['config']['start']['kind']=='saved_trajectory_latent'
    assert metadata['config']['start']['schedule_step']==180
    assert result['final']['temperature']==schedules(cfg,184)['temperature']
    updates=[json.loads(x) for x in (tmp_path/'warm/arms/latent/updates.jsonl').read_text().splitlines()]
    assert all(x['noise_norm']==0 and x['noise_ratio']==0 for x in updates)


def test_zero_noise_does_not_consume_rng_or_depend_on_seed():
    cfg=replace(config(),noise_initial=0.,learning_rate=.1)
    point=torch.ones(1,4);gradient=torch.tensor([[.1,.2,-.3,.4]])
    a=torch.Generator().manual_seed(11);b=torch.Generator().manual_seed(99)
    before=a.get_state().clone()
    left,details=latent_step(point,gradient,gradient,cfg,180,a)
    right,_=latent_step(point,gradient,gradient,cfg,180,b)
    assert torch.equal(left,right) and torch.equal(a.get_state(),before)
    assert details['noise_norm']==0


@pytest.mark.parametrize('changed',[{'updates':0},{'learning_rate':float('nan')},{'noise_initial':2.},
    {'chemistry_hold':30001},{'temperature_initial':0.}])
def test_invalid_configuration_rejected(changed):
    with pytest.raises(ValueError):LatentDesignConfig(**changed)


@pytest.mark.parametrize("study_mode", ("legacy", "warm", "search", "matched", "normalized_temperatures"))
def test_temperature_study_matches_starts_and_scores_saved_hard_graphs(tmp_path, monkeypatch, study_mode):
    from latentmol.design import temperature_study as study
    from latentmol.multimodel.external import file_sha256

    class ConnectedDecoder(ToyDecoder):
        def decode(self, z):
            decoded = super().decode(z)
            bonds = decoded.bonds.clone()
            bonds[..., 1] = decoded.bonds[..., 3]
            bonds[..., 3] = -3.
            return DecoderOutput(decoded.atoms, decoded.charges, bonds)

    class LocalScorer(ToyScorer):
        instances = []
        def __init__(self, *_, **__):
            self.checks = 0
            self.instances.append(self)

        def assert_unchanged(self):
            self.checks += 1

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"fixture")
    parent = ConnectedDecoder()
    monkeypatch.setattr(study, "load_model", lambda *_: (parent, {}))
    monkeypatch.setattr(study, "MosaicProtenixScorer", LocalScorer)
    native = tmp_path / "native.json"
    native.write_text("{}", encoding="utf-8")
    (tmp_path / "pilot.json").write_text(json.dumps(asdict(config())), encoding="utf-8")
    plan = tmp_path / "temperature.json"
    experiment = dict(checkpoint_sha256=file_sha256(checkpoint), base_design="pilot.json",
                      updates=2, starts=[dict(name="fixture", kind="prior", seed=51)])
    if study_mode == "warm":
        experiment.update(temperature_arms={
            "warm": dict(temperature_initial=1.5, temperature_final=1., cooling_updates=1),
            "constant": dict(temperature_initial=1., temperature_final=1., cooling_updates=1)},
            fresh_edit_limit=4)
    if study_mode in ("search", "matched", "normalized_temperatures"):
        experiment.update(step_search=dict(iterations=1, radii=[.1, .3], minimum_improvement=1e-7),
                          fresh_edit_limit=4)
    if study_mode == "normalized_temperatures":
        experiment.pop("updates")
        experiment["normalized_search_temperatures"] = {"ordinary": 1., "warm": 1.5}
    if study_mode == "matched":
        from latentmol.design.same_molecule_comparison import ARM_NAMES
        experiment["same_molecule_panel"] = dict(parents=1, path="unused", sha256="fixture")
        def fixture_starts(config, config_folder, model, configs, checkpoint_sha, count):
            states = {arm: dict(study.start_state(model, dict(kind="prior", seed=51+i), count),
                                native_order=[2,1,0] if i else [0,1,2]) for i,arm in enumerate(ARM_NAMES)}
            yield "fixture", "CCC", states
        monkeypatch.setattr(study, "study_starts", fixture_starts)
    plan.write_text(json.dumps(experiment), encoding="utf-8")
    output = tmp_path / "output"
    result = study.run(plan, checkpoint, output, "unused", native)
    arms = result["starts"]["fixture"]
    endpoints = [.4, .7, 1.] if study_mode == "legacy" else [1.] * len(arms)
    if study_mode == "normalized_temperatures":
        endpoints = [1., 1.5]
    assert [arm["design"]["final"]["temperature"] for arm in arms.values()] == endpoints
    updates = [3, 1] if study_mode == "search" else [1 if study_mode in ("matched", "normalized_temperatures") else 2] * len(arms)
    assert [arm["design"]["updates_this_segment"] for arm in arms.values()] == updates
    assert all(arm["candidates"]["initial"]["smiles"] == "CCC" for arm in arms.values())
    if study_mode not in ("search", "matched", "normalized_temperatures"):
        assert len({arm["candidates"]["initial"]["hard_fixed_context_loss"] for arm in arms.values()}) == 1
    fresh = json.loads((output / "fresh-candidates.json").read_text(encoding="utf-8"))["candidates"]
    assert {row["role"] for row in fresh} == ({"initial", "endpoint"} if study_mode == "legacy" else {"initial"})
    if study_mode == "warm":
        rows = (output / "fixture" / "warm" / "arms" / "latent" / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(row)["temperature"] for row in rows] == [1.5, 1., 1.]
    if study_mode == "search":
        assert [arm["scorer_budget"]["actual"] for arm in arms.values()] == [4, 4]
        assert arms["normalized_search"]["scorer_budget"]["trial_calls"] == 2
        rows = (output / "fixture" / "normalized_search" / "arms" / "latent" / "scorer-observations.jsonl").read_text().splitlines()
        assert [json.loads(row)["kind"] for row in rows] == ["center", "search_trial", "search_trial", "center"]
    if study_mode == "normalized_temperatures":
        assert result["updates_per_arm"] == {"ordinary": 1, "warm": 1}
        assert all(arm["scorer_budget"]["actual"] == 4 for arm in arms.values())
    if study_mode == "matched":
        assert result["parent_slice"] == [0,1]
        assert len(LocalScorer.instances) == 1 and LocalScorer.instances[0].checks == 3
        assert all(arm["scorer_budget"]["actual"] == 4 for arm in arms.values())
        assert [row["on_committed_trajectory"] for row in fresh] == [True, True, True]


def test_normalized_step_freezes_repair_witness_and_rejects_invalid_decrease(monkeypatch):
    from latentmol.design import protenix_step_search as search

    model = ToyDecoder().eval().requires_grad_(False)
    active = torch.ones(3, dtype=torch.bool)
    settings = config(temperature_initial=1., temperature_final=1.)
    scorer = ToyScorer()
    center = evaluate_point(model, scorer, torch.zeros(1, 4), active, settings, 0)
    target, _ = center["target_bundle"]
    assert target is not None
    repaired = target.graph.to_relaxed(model.spec)
    candidates = iter([(center["graph"], -0.2), (repaired, -0.1), (repaired, 0.)])

    def trial(_, __, point, mask, design, step, *, target_bundle):
        assert target_bundle is center["target_bundle"]
        assert mask is active and design is settings and step == 0
        graph, delta = next(candidates)
        return dict(center, graph=graph, binding=center["binding"] + delta)

    monkeypatch.setattr(search, "evaluate_point", trial)
    selected, row = search.normalized_update(model, scorer, center, active, settings, 0,
                                              radii=[.1, .3, 1.], minimum_improvement=1e-7)
    assert row["trials"][0]["improving"] and not row["trials"][0]["fixed"]["valid"]
    assert row["trials"][1]["selected"] and row["accepted"]
    assert not row["trials"][2]["improving"]
    assert selected.norm() == pytest.approx(.3)
