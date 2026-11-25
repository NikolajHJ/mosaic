from dataclasses import asdict
import hashlib
import json
import numpy as np
import pytest
import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, evaluate_point, run_design
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.exchange import (ExchangeConfig, StructureReference, exchange_cost,
    preservation_loss, search_exchange, structure_distance, _mixed_neighbors)
from latentmol.representation import DecoderOutput, GraphSpec


def raw_smiles(smiles,spec):
    row=encode_smiles(smiles,spec)
    return tuple(np.array(row[k]) for k in ('atom_types','charges','bond_types'))


def output(raw,spec):
    return DecoderOutput(*(torch.nn.functional.one_hot(torch.tensor(x,dtype=torch.long),n).double()[None].mul(3).requires_grad_()
        for x,n in zip(raw,(len(spec.atomic_numbers),len(spec.formal_charges),len(spec.bond_names)))))


def test_atom_cost_can_reverse_sulfur_shortcut_without_banning_sulfur():
    spec=GraphSpec(max_nodes=4)
    reference_raw=raw_smiles('CC(C)C',spec)
    ref=StructureReference.create(reference_raw,spec)
    center=int(np.argmax((reference_raw[2]!=0).sum(-1)))
    raw=tuple(x.copy() for x in reference_raw)
    raw[2][raw[2]!=0]=spec.bond_names.index('DOUBLE')
    sulfur=tuple(x.copy() for x in raw);sulfur[0][center]=spec.atomic_numbers.index(16)
    bonds=tuple(x.copy() for x in raw)
    for j in [i for i in range(4) if i!=center][:2]:bonds[2][center,j]=bonds[2][j,center]=1
    assert inspect_graph(*raw,spec).smiles is None
    assert inspect_graph(*sulfur,spec).smiles is not None
    assert inspect_graph(*bonds,spec).smiles is not None
    p=tuple(np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5)))
    low=ExchangeConfig(atom_extra_cost=0,substructure_extra_cost=0)
    high=ExchangeConfig(atom_extra_cost=5,substructure_extra_cost=0)
    assert exchange_cost(raw,sulfur,p,spec,low,ref)['weighted_cost']<exchange_cost(raw,bonds,p,spec,low,ref)['weighted_cost']
    assert exchange_cost(raw,sulfur,p,spec,high,ref)['weighted_cost']>exchange_cost(raw,bonds,p,spec,high,ref)['weighted_cost']
    before=tuple(x.copy() for x in raw)
    result=search_exchange(raw,p,spec,ExchangeConfig(atom_extra_cost=5,substructure_extra_cost=0,checks_per_depth=128),ref)
    assert result['status']=='witness_found'
    assert result['checks']<=384
    assert all(np.array_equal(a,b) for a,b in zip(raw,before))


def test_ring_penalty_and_finite_live_gradient_with_valid_graph():
    spec=GraphSpec(max_nodes=9);raw=raw_smiles('CCc1ccccc1',spec)
    ref=StructureReference.create(raw,spec)
    assert ref.atom_mask.sum()==6
    ring=int(np.flatnonzero(ref.atom_mask)[0]);terminal=int(np.flatnonzero((raw[0]!=0)&~ref.atom_mask)[0])
    a=tuple(x.copy() for x in raw);b=tuple(x.copy() for x in raw)
    a[0][ring]=2;b[0][terminal]=2
    assert structure_distance(a,ref)>structure_distance(b,ref)
    decoded=output(raw,spec);graph=graph_from_decoder(decoded,spec,fixed_active=ref.target.active)
    loss,detail=preservation_loss(graph,ref,ExchangeConfig())
    gradients=torch.autograd.grad(loss,(decoded.atoms,decoded.charges,decoded.bonds))
    assert all(bool(g.isfinite().all()) for g in gradients)
    assert all(float(g.norm())>0 for g in gradients)
    assert detail['preservation_loss']>0
    assert gradients[0][0,ring,raw[0][ring]]<0
    p=tuple(getattr(graph,k)[0].detach().numpy() for k in ('atoms','charges','bonds'))
    assert search_exchange(raw,p,spec,ExchangeConfig(),ref)['status']=='already_valid'
    bad=tuple(x.copy() for x in raw);bad[0][terminal]=0
    with pytest.raises(ValueError,match='occupancy'):search_exchange(bad,p,spec,ExchangeConfig(),ref)


def test_original_config_identity_is_unchanged_and_invalid_costs_rejected():
    config=LatentDesignConfig();old=asdict(config);old.pop('exchange_guidance');old.pop('preservation_guidance')
    assert config.identity()==hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest()
    for value in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):ExchangeConfig(atom_extra_cost=value)


def test_reference_tampering_and_invalid_reference_rejected():
    spec=GraphSpec(max_nodes=6);raw=raw_smiles('c1ccccc1',spec)
    ref=StructureReference.create(raw,spec)
    ref.target.graph.atoms[0,0]=3
    with pytest.raises(ValueError,match='identity'):ref.labels()
    raw[2][0,0]=1
    with pytest.raises(ValueError):StructureReference.create(raw,spec)


@pytest.mark.parametrize('edit_cost',['log_regret','one_minus_probability'])
def test_opt_in_runner_preserves_reference_across_resume(tmp_path,edit_cost):
    from test_latent_protenix import ToyDecoder, ToyScorer
    model=ToyDecoder().eval().requires_grad_(False)
    cfg=LatentDesignConfig(occupied_atoms=3,updates=2,snapshot_interval=2,checkpoint_interval=1,
        exchange_guidance=dict(edit_cost=edit_cost,atom_extra_cost=2,substructure_extra_cost=2,checks_per_depth=16,max_edits=2))
    raw=raw_smiles('CCC',model.spec)
    start=dict(latent=torch.zeros(1,4),active=torch.ones(3,dtype=torch.bool),step=0,
               exchange_reference={k:torch.tensor(x,dtype=torch.long) for k,x in zip(('atoms','charges','bonds'),raw)})
    factory=lambda *_:ToyScorer()
    run_design(model,factory,cfg,tmp_path/'first',checkpoint_sha256='toy',start_state=start,max_updates=1)
    before=torch.load(tmp_path/'first'/'state.pt',weights_only=True)
    run_design(model,factory,cfg,tmp_path/'second',checkpoint_sha256='toy',resume=tmp_path/'first'/'state.pt')
    after=torch.load(tmp_path/'second'/'state.pt',weights_only=True)
    assert after['step']==2
    for k in ('atoms','charges','bonds'):assert torch.equal(before['exchange_reference'][k],after['exchange_reference'][k])
    assert all(p.grad is None for p in model.parameters())


def test_small_budget_visits_each_edit_family():
    from itertools import islice
    spec=GraphSpec(max_nodes=4);raw=raw_smiles('CC(C)C',spec)
    candidates=list(islice(_mixed_neighbors(raw,raw,spec,{},lambda op:op),3))
    fields=[next(i for i in range(3) if not np.array_equal(c[i],raw[i])) for c in candidates]
    assert fields==[2,0,1]


@pytest.mark.parametrize('field',[0,1,2])
@pytest.mark.parametrize('probability',[0.,.2,.8,1.])
def test_one_minus_probability_is_additive_per_actual_edit(field,probability):
    spec=GraphSpec(max_nodes=3);raw=raw_smiles('CCC',spec);ref=StructureReference.create(raw,spec)
    candidate=tuple(x.copy() for x in raw)
    index=(0,1) if field==2 else (0,)
    old=int(raw[field][index]);new=(old+1)%(len(spec.bond_names) if field==2 else len(spec.formal_charges) if field==1 else len(spec.atomic_numbers))
    candidate[field][index]=new
    if field==2:candidate[2][1,0]=new
    p=[np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5))]
    p[field][index]=0;p[field][index+(old,)]=1-probability;p[field][index+(new,)]=probability
    if field==2:p[2][1,0]=p[2][0,1]
    config=ExchangeConfig(edit_cost='one_minus_probability',atom_extra_cost=.5,charge_extra_cost=.25,substructure_extra_cost=0)
    expected=1-probability+(.5 if field==0 else .25 if field==1 else 0)
    cost=exchange_cost(raw,candidate,p,spec,config,ref)
    assert cost['weighted_cost']==pytest.approx(expected)
    assert sum(change['weighted_cost'] for change in cost['changes'])==pytest.approx(expected)
    assert exchange_cost(raw,raw,p,spec,config,ref)['weighted_cost']==0


@pytest.mark.parametrize('smiles,name,count',[
    ('CCC(=O)N','amide',3), ('CC(=O)O','carboxyl',3), ('CC(=O)OCC','carboxyl',3),
    ('CS(=O)(=O)C','sulfonyl',3), ('CS(=O)(=O)N','sulfonyl',3),
    ('c1ccccc1','ring',6), ('C1CCCCC1','ring',6), ('c1ccc2ccccc2c1','ring',10)])
def test_declared_motif_coverage_and_exact_additive_substructure_cost(smiles,name,count):
    spec=GraphSpec(max_nodes=16);raw=raw_smiles(smiles,spec);ref=StructureReference.create(raw,spec)
    assert ref.atom_mask.sum()==count
    assert name in [m[0] for m in ref.motifs]
    candidate=tuple(x.copy() for x in raw)
    i=int(np.flatnonzero(ref.atom_mask)[0]);candidate[0][i]=2 if raw[0][i]!=2 else 1
    p=tuple(np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5)))
    config=ExchangeConfig(edit_cost='one_minus_probability',atom_extra_cost=.5,substructure_extra_cost=.75)
    assert structure_distance(candidate,ref)==1
    assert exchange_cost(raw,candidate,p,spec,config,ref)['weighted_cost']==pytest.approx(.9+.5+.75)
    i,j=np.argwhere(np.triu(ref.pair_mask,1))[0]
    candidate=tuple(x.copy() for x in raw);candidate[2][i,j]=candidate[2][j,i]=(int(raw[2][i,j])+1)%5
    assert structure_distance(candidate,ref)==1  # Undirected bond counted once.


def test_reference_permutation_padding_and_directional_gradient():
    spec=GraphSpec(max_nodes=12);raw=raw_smiles('CC(=O)Nc1ccccc1',spec)
    ref=StructureReference.create(raw,spec);decoded=output(raw,spec)
    config=ExchangeConfig(edit_cost='one_minus_probability',atom_extra_cost=.5,substructure_extra_cost=.5)
    def value(d,reference):
        graph=graph_from_decoder(d,spec,fixed_active=reference.target.active)
        return preservation_loss(graph,reference,config)[0]
    loss=value(decoded,ref)
    derivatives=torch.autograd.grad(loss,(decoded.atoms,decoded.charges,decoded.bonds))
    assert torch.count_nonzero(derivatives[0][:,~ref.target.active])==0
    assert torch.count_nonzero(derivatives[1][:,~ref.target.active])==0
    generator=torch.Generator().manual_seed(18)
    directions=[torch.randn(x.shape,dtype=x.dtype,generator=generator) for x in (decoded.atoms,decoded.charges,decoded.bonds)]
    ad=sum((a*b).sum() for a,b in zip(derivatives,directions))
    h=1e-5
    plus=DecoderOutput(*(x.detach()+h*d for x,d in zip((decoded.atoms,decoded.charges,decoded.bonds),directions)))
    minus=DecoderOutput(*(x.detach()-h*d for x,d in zip((decoded.atoms,decoded.charges,decoded.bonds),directions)))
    fd=(value(plus,ref)-value(minus,ref))/(2*h)
    assert float(ad)==pytest.approx(float(fd),rel=1e-6,abs=1e-10)
    order=np.random.default_rng(17).permutation(spec.max_nodes)
    perm=(raw[0][order],raw[1][order],raw[2][np.ix_(order,order)])
    other=StructureReference.create(perm,spec)
    assert float(value(output(perm,spec),other).detach())==pytest.approx(float(loss.detach()))


@pytest.mark.parametrize('bad',['nan','negative','shape','asymmetric'])
def test_invalid_shared_probabilities_rejected(bad):
    spec=GraphSpec(max_nodes=3);raw=raw_smiles('CCC',spec);ref=StructureReference.create(raw,spec)
    p=[np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5))]
    if bad=='nan':p[0][0,0]=np.nan
    if bad=='negative':p[0][0,0]=-1
    if bad=='shape':p[0]=p[0][:-1]
    if bad=='asymmetric':p[2][0,1]=[1,0,0,0,0]
    with pytest.raises(ValueError):search_exchange(raw,p,spec,ExchangeConfig(edit_cost='one_minus_probability'),ref)


@pytest.mark.parametrize('extra',[0.,.5,2.])
def test_tiny_exhaustive_cost_matches_search_when_budget_covers_space(extra):
    from itertools import product
    spec=GraphSpec(max_nodes=3,atomic_numbers=(0,6,8,16),formal_charges=(0,),bond_names=('NONE','SINGLE','DOUBLE','TRIPLE'))
    ref=StructureReference.create(raw_smiles('CCC',spec),spec)
    raw=(np.ones(3,dtype=np.int64),np.zeros(3,dtype=np.int64),np.full((3,3),3,dtype=np.int64))
    np.fill_diagonal(raw[2],0)
    p=(np.full((3,4),.25),np.ones((3,1)),np.full((3,3,4),.25))
    config=ExchangeConfig(edit_cost='one_minus_probability',atom_extra_cost=extra,substructure_extra_cost=0,
                          max_edits=2,checks_per_depth=2000,beam_width=256)
    optimum=float('inf')
    for atoms in product((1,2,3),repeat=3):
        for edges in product(range(4),repeat=3):
            if sum(a!=1 for a in atoms)+sum(b!=3 for b in edges)>2:continue
            bonds=np.zeros((3,3),dtype=np.int64)
            for (i,j),b in zip(((0,1),(0,2),(1,2)),edges):bonds[i,j]=bonds[j,i]=b
            candidate=(np.array(atoms),raw[1],bonds)
            if inspect_graph(*candidate,spec).smiles is not None:
                optimum=min(optimum,exchange_cost(raw,candidate,p,spec,config,ref)['weighted_cost'])
    found=search_exchange(raw,p,spec,config,ref)
    assert found['weighted_witness']['weighted_cost']==pytest.approx(optimum)
    assert not found['distance_is_exact']  # No general optimality claim.


@pytest.mark.parametrize('element',[7,15,16])
def test_atom_surcharge_is_element_independent(element):
    spec=GraphSpec(max_nodes=3);raw=raw_smiles('CCC',spec);ref=StructureReference.create(raw,spec)
    candidate=tuple(x.copy() for x in raw);candidate[0][0]=spec.atomic_numbers.index(element)
    p=tuple(np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5)))
    policy=ExchangeConfig(edit_cost='one_minus_probability',atom_extra_cost=.5,substructure_extra_cost=0)
    assert exchange_cost(raw,candidate,p,spec,policy,ref)['weighted_cost']==pytest.approx(1.4)


def test_schema_reordering_is_not_silently_accepted():
    from dataclasses import replace
    spec=GraphSpec(max_nodes=3);raw=raw_smiles('CCC',spec);ref=StructureReference.create(raw,spec)
    p=tuple(np.full(x.shape+(n,),1/n) for x,n in zip(raw,(10,5,5)))
    other=replace(spec,atomic_numbers=(0,7,6,8,9,15,16,17,35,53))
    with pytest.raises(ValueError,match='schema'):
        search_exchange(raw,p,other,ExchangeConfig(),ref)


def test_ambiguous_or_invalid_cost_mode_rejected():
    with pytest.raises(ValueError):ExchangeConfig(edit_cost='unknown')
    with pytest.raises(ValueError):ExchangeConfig(edit_cost='one_minus_probability',probability_weight=2.)
