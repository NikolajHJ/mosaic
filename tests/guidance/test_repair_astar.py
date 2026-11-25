"""Exhaustive tiny-space cost/admissibility checks and opt-in integration."""
from itertools import product

import numpy as np
import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.guidance.repair_astar import (AStarRepairConfig, AStarRepairTeacher, _Space,
    probability_lower_bound, required_edit_bound, search_astar)
from latentmol.probability_repair import ProbabilityRepairConfig, measure_probability_edits
from latentmol.representation import DecoderOutput, GraphSpec


def output(spec, elements, edges):
    n = spec.max_nodes
    a = torch.full((1,n,len(spec.atomic_numbers)), -8., dtype=torch.float64)
    c = torch.full((1,n,len(spec.formal_charges)), -8., dtype=torch.float64)
    b = torch.full((1,n,n,len(spec.bond_names)), -8., dtype=torch.float64)
    a[...,0] = 0.;c[...,spec.formal_charges.index(0)] = 0.;b[...,0] = 0.
    for i, element in enumerate(elements):
        a[0,i,0] = -8.;a[0,i,spec.atomic_numbers.index(element)] = 0.
    for i,j,name in edges:
        b[0,i,j,0] = b[0,j,i,0] = -8.
        b[0,i,j,spec.bond_names.index(name)] = b[0,j,i,spec.bond_names.index(name)] = 0.
    return DecoderOutput(*(x.requires_grad_() for x in (a,c,b)))


def arrays(decoded, spec):
    hard = decoded.harden(spec)
    raw = tuple(x[0].detach().numpy().copy() for x in (hard.atoms,hard.charges,hard.bonds))
    p = tuple(x[0].detach().numpy().copy() for x in (decoded.atoms.softmax(-1),decoded.charges.softmax(-1),
                                                  decoded.symmetric_bond_logits.softmax(-1)))
    return raw,p


def test_cheapest_cost_bound_uses_distinct_remaining_bonds():
    assert required_edit_bound(6, 1) == 1
    assert required_edit_bound(7, 1) == 2
    assert required_edit_bound(0, 4) == 3
    assert probability_lower_bound(2, [(1.,0),(5.,1),(8.,2)], {0}) == 13.
    assert probability_lower_bound(0, [], set()) == 0.


@pytest.mark.parametrize("heuristic",["probability","zero"])
def test_astar_matches_exhaustive_minimum_probability_weighted_repair(heuristic):
    spec=GraphSpec(max_nodes=3)
    raw,p=arrays(output(spec,[6,6,8],[(0,1,"TRIPLE"),(0,2,"TRIPLE"),(1,2,"TRIPLE")]),spec)
    rng=np.random.default_rng(391)
    for i,j in [(0,1),(0,2),(1,2)]:
        law=rng.uniform(.01,1,len(spec.bond_names));law/=law.sum();p[2][i,j]=p[2][j,i]=law
    config=ProbabilityRepairConfig(max_edits=3)
    costs=[]
    for labels in product(range(len(spec.bond_names)),repeat=3):
        bonds=np.zeros_like(raw[2])
        for (i,j),label in zip([(0,1),(0,2),(1,2)],labels):bonds[i,j]=bonds[j,i]=label
        if inspect_graph(raw[0],raw[1],bonds,spec).smiles is not None:
            costs.append(measure_probability_edits(raw,(raw[0],raw[1],bonds),p,spec,config)['weighted_cost'])
    result=search_astar(raw,p,spec,config,AStarRepairConfig(candidate_checks=2000,heuristic=heuristic))
    assert result['status']=='witness_found' and result['goal_popped']
    assert result['best_observed_weighted_cost']==pytest.approx(min(costs),abs=1e-12)
    assert not result['interrupted']


def test_probability_heuristic_never_exceeds_exhaustive_remaining_cost_in_tiny_space():
    spec=GraphSpec(max_nodes=3)
    raw,p=arrays(output(spec,[6,6,8],[(0,1,"TRIPLE"),(0,2,"TRIPLE"),(1,2,"TRIPLE")]),spec)
    config=ProbabilityRepairConfig(max_edits=3);space=_Space(raw,p,spec,config)
    keys=[bytes(labels) for labels in product(range(len(spec.bond_names)),repeat=3)]
    goals=[]
    for key in keys:
        if inspect_graph(raw[0],raw[1],space.bonds(key),spec).smiles is not None:goals.append(key)
    checked=0
    for key in keys:
        edited={i for i,label in enumerate(key) if label!=space.original[i]}
        completions=[g for g in goals if all(g[i]==key[i] for i in edited)]
        if not completions:continue
        _,components,_,_,_,excess=space.topology(key)
        h=probability_lower_bound(required_edit_bound(excess,components),space.ordered_costs,edited)
        edits=tuple((i,key[i]) for i in sorted(edited));current=space.cost(edits)
        remaining=min(space.cost(tuple((i,g[i]) for i in range(3) if g[i]!=space.original[i]))-current for g in completions)
        assert h<=remaining+1e-12
        checked+=1
    assert checked>10


@pytest.mark.parametrize("smiles",["c1ccoc1","c1ccsc1","c1ccccc1","O=S(=O)=O","CCS(=O)(=O)C"])
def test_native_valid_aromatic_and_sulfur_goals_have_zero_valence_bound(smiles):
    spec=GraphSpec(max_nodes=10);encoded=encode_smiles(smiles,spec)
    raw=tuple(np.array(encoded[k]) for k in ("atom_types","charges","bond_types"))
    decoded=output(spec,[],[])
    _,p=arrays(decoded,spec)
    space=_Space(raw,p,spec,ProbabilityRepairConfig())
    _,components,_,_,_,excess=space.topology(space.root)
    assert required_edit_bound(excess,components)==0
    result=search_astar(raw,p,spec,ProbabilityRepairConfig(),AStarRepairConfig())
    assert result['status']=='already_valid' and result['checks']==0


@pytest.mark.parametrize("budget,termination",[
    ({"max_generated":2},"generated_state_budget"),
    ({"max_frontier":1},"frontier_memory_budget"),
    ({"candidate_checks":1},"candidate_check_budget")])
def test_budget_interruptions_are_censored_without_pruning_or_optimality_claim(budget,termination):
    spec=GraphSpec(max_nodes=4);raw,p=arrays(output(spec,[6]*4,[]),spec)
    result=search_astar(raw,p,spec,ProbabilityRepairConfig(max_edits=3),AStarRepairConfig(**budget))
    assert result['status']=='censored' and result['interrupted']
    assert result['termination']==termination and result['weighted_witness'] is None
    assert result['lower_bound_weighted_cost'] is not None
    assert not result['goal_popped']


def test_dense_lower_bound_distinguishes_edit_radius_from_global_infeasibility():
    spec=GraphSpec(max_nodes=8);edges=[(i,j,'SINGLE') for i in range(8) for j in range(i+1,8)]
    raw,p=arrays(output(spec,[6]*8,edges),spec)
    result=search_astar(raw,p,spec,ProbabilityRepairConfig(max_edits=3),AStarRepairConfig())
    assert result['termination']=='edit_lower_bound_exceeds_radius'
    assert result['status']=='censored' and result['checks']==0


def test_teacher_worker_and_existing_loss_keep_fixed_slots_and_attached_gradients():
    spec=GraphSpec(max_nodes=3);decoded=output(spec,[8,6],[(0,1,'TRIPLE')])
    config=ProbabilityRepairConfig(max_edits=1);astar=AStarRepairConfig()
    with AStarRepairTeacher(spec,config,astar=astar) as serial, \
         AStarRepairTeacher(spec,config,astar=astar,workers=1) as parallel:
        a=serial.build_targets(decoded,candidate_ids=[2],update=0)
        b=parallel.build_targets_async(decoded,candidate_ids=[2],update=0).result()
        assert a.targets[0].record()==b.targets[0].record()
        loss=repair_loss(decoded,a,spec);loss.backward()
        assert torch.isfinite(loss) and float(decoded.bonds.grad.norm())>0
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (decoded.atoms,decoded.charges,decoded.bonds))


def test_factory_is_explicit_and_search_cannot_mutate_inputs():
    from latentmol.failure_guidance_training import build_repair_teacher
    spec=GraphSpec(max_nodes=3);raw,p=arrays(output(spec,[8,6],[(0,1,'TRIPLE')]),spec)
    originals=[x.copy() for x in (*raw,*p)]
    search_astar(raw,p,spec,ProbabilityRepairConfig(),AStarRepairConfig())
    assert all(np.array_equal(a,b) for a,b in zip((*raw,*p),originals))
    protocol={'search':{'max_edits':1}}
    with build_repair_teacher(spec,protocol) as old,build_repair_teacher(spec,protocol,astar_config={}) as new:
        assert type(old) is AtomPreservingRepairTeacher and type(new) is AStarRepairTeacher
    with pytest.raises(ValueError):build_repair_teacher(spec,protocol,astar_config={},proposal_config={})
    with pytest.raises(ValueError):AStarRepairConfig(candidate_checks=0)
    p[2][0,1,0]=np.nan
    with pytest.raises(ValueError):search_astar(raw,p,spec,ProbabilityRepairConfig(),AStarRepairConfig())


def test_overflowing_cost_configuration_stops_instead_of_entering_priority_queue():
    spec=GraphSpec(max_nodes=3);raw,p=arrays(output(spec,[8,6],[(0,1,'TRIPLE')]),spec)
    with pytest.raises(ValueError,match='Nonfinite A\\* edit cost'):
        search_astar(raw,p,spec,ProbabilityRepairConfig(probability_weight=1e308),AStarRepairConfig())
