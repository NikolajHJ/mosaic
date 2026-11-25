from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.design.exchange_protenix import ObservedNativeScorer, choose_start, cohort_summary, arm_design
from latentmol.design.latent_protenix import LatentDesignConfig, run_design, evaluate_point
from latentmol.guidance.exchange import StructureReference
from test_latent_protenix import ToyDecoder, ToyScorer


def valid_toy():
    model=ToyDecoder().eval().requires_grad_(False)
    with torch.no_grad():
        decoded=model.decode(torch.zeros(1,4));decoded.bonds.fill_(-8.);decoded.bonds[...,0]=0.
        for i,j in ((0,1),(1,2)):
            decoded.bonds[0,i,j,0]=decoded.bonds[0,j,i,0]=-8.
            decoded.bonds[0,i,j,1]=decoded.bonds[0,j,i,1]=0.
        model.base.copy_(torch.cat([x.flatten() for x in (decoded.atoms,decoded.charges,decoded.bonds)]))
    return model


@pytest.mark.parametrize('exchange',[None,dict(edit_cost='one_minus_probability',atom_extra_cost=.5,substructure_extra_cost=.5),
                                    dict(edit_cost='one_minus_probability',atom_extra_cost=2,substructure_extra_cost=.5)])
def test_native_wrapper_keeps_observation_and_hard_readout_paths_separate(tmp_path,exchange):
    model=valid_toy();row=encode_smiles('CCC',model.spec)
    raw=tuple(np.asarray(row[k]) for k in ('atom_types','charges','bond_types'))
    reference=StructureReference.create(raw,model.spec)
    config=LatentDesignConfig(occupied_atoms=3,updates=2,learning_rate=70,noise_initial=0,
        temperature_initial=1,temperature_final=1,chemistry_initial=1,chemistry_final=1,
        checkpoint_interval=1,snapshot_interval=1,exchange_guidance=exchange)
    state=dict(latent=torch.zeros(1,4),active=torch.ones(3,dtype=torch.bool),step=0,
               exchange_reference={k:getattr(reference.target.graph,k)[0] for k in ('atoms','charges','bonds')})
    native=ToyScorer();native.spec=model.spec
    folder=tmp_path/'arm'
    def factory(*_):return ObservedNativeScorer(native,folder,reference)
    result=run_design(model,factory,config,folder,checkpoint_sha256='test',start_state=state)
    assert result['completed_step']==2 and native.closed
    observed=json.loads((folder/'chemistry-observations.json').read_text())
    assert [r['step'] for r in observed]==[0,1,2]
    readouts=json.loads((folder/'hard-readouts.json').read_text())
    assert readouts[0]['role']=='initial' and readouts[0]['loss'] is not None
    assert readouts[1]['role']=='endpoint'
    assert all(r['loss'] is None for r in readouts if not r['valid'])
    assert all(p.grad is None for p in model.parameters())
    summary=cohort_summary(tmp_path,dict(design=dict(updates=2),arms=dict(arm=exchange)))
    assert summary['arm']['native_updates']==2


def test_screening_is_repeatable_and_returns_shared_valid_reference():
    model=valid_toy();cfg=dict(seeds=[170],screen_limit=5,design=dict(occupied_atoms=3))
    first,reference=choose_start(model,cfg,170);second,_=choose_start(model,cfg,170)
    assert torch.equal(first['latent'],second['latent'])
    assert torch.equal(first['active'],reference.target.active)
    assert first['provenance']['smiles']
    with pytest.raises(ValueError,match='outside'):choose_start(model,cfg,171)


@pytest.mark.parametrize('valid',[False,True])
@pytest.mark.parametrize('atom,motif',[(0.,0.),(.5,.5),(2.,2.)])
def test_preservation_keeps_exact_current_targets_and_zero_recovers_gradient(valid,atom,motif):
    model=valid_toy() if valid else ToyDecoder().eval().requires_grad_(False)
    encoded=encode_smiles('CCC',model.spec)
    reference=StructureReference.create(tuple(np.asarray(encoded[k]) for k in ('atom_types','charges','bond_types')),model.spec)
    point=torch.zeros(1,4);active=reference.target.active
    config=LatentDesignConfig(occupied_atoms=3,temperature_initial=1,temperature_final=1)
    baseline=evaluate_point(model,ToyScorer(),point,active,config,0)
    guided=evaluate_point(model,ToyScorer(),point,active,replace(config,preservation_guidance=dict(
        atom_extra_cost=atom,substructure_extra_cost=motif)),0,exchange_reference=reference)
    assert baseline['repair']==guided['repair']
    assert baseline['repair_detail']['normalized_nll']==guided['repair_detail']['normalized_nll']
    assert torch.equal(baseline['binding_gradient'],guided['binding_gradient'])
    assert guided['repair']['status']==('already_valid' if valid else 'witness_found')
    if not atom and not motif:
        assert torch.equal(baseline['chemistry'],guided['chemistry'])
        assert torch.equal(baseline['gradient'],guided['gradient'])
    else:
        assert guided['repair_detail']['preservation_gradient_norm']>0
    if guided['target_bundle'][0] is not None:
        assert torch.equal(guided['target_bundle'][0].graph.atoms,guided['graph'].atoms.argmax(-1))
    assert all(p.grad is None for p in model.parameters())


def test_preservation_reference_survives_resume_and_modes_are_exclusive(tmp_path):
    model=valid_toy();cfg=LatentDesignConfig(occupied_atoms=3,updates=2,temperature_initial=1,
        temperature_final=1,noise_initial=0,preservation_guidance=dict(atom_extra_cost=.5,substructure_extra_cost=.5))
    state,ref=choose_start(model,dict(seeds=[170],screen_limit=5,design=dict(occupied_atoms=3,temperature_initial=1)),170)
    run_design(model,lambda *_:ToyScorer(),cfg,tmp_path/'first',checkpoint_sha256='toy',start_state=state,max_updates=1)
    run_design(model,lambda *_:ToyScorer(),cfg,tmp_path/'second',checkpoint_sha256='toy',resume=tmp_path/'first/state.pt')
    after=torch.load(tmp_path/'second/state.pt',weights_only=True)
    for k in ('atoms','charges','bonds'):assert torch.equal(after['exchange_reference'][k],getattr(ref.target.graph,k)[0])
    with pytest.raises(ValueError,match='not both'):
        replace(cfg,exchange_guidance={})
    with pytest.raises(ValueError,match='finite nonnegative'):
        replace(cfg,preservation_guidance=dict(atom_extra_cost=float('inf')))


def test_comparison_arm_routes_penalties_without_changing_search():
    config=dict(design={},guidance_mode='current_search_preservation',arms=dict(current=None,penalty=dict(atom_extra_cost=2)))
    baseline=arm_design(config,'current',1);penalty=arm_design(config,'penalty',1)
    assert penalty.exchange_guidance is None and penalty.preservation_guidance==dict(atom_extra_cost=2)
    assert baseline.repair_search==penalty.repair_search
    assert baseline.structural_proposals==penalty.structural_proposals
