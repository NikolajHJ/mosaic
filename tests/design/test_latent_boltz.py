"""Small fixed-context slot/category boundary controls, no native models."""
import importlib.util
from pathlib import Path

import pytest
import torch

from latentmol.design.latent_boltz import rebind_slots
from latentmol.boltz.adapter import adapt_graph
from latentmol.representation import GraphSpec
from test_boltz_adapter import prepared_fixture


def test_slot_reindexing_preserves_native_seed_features_and_gradients():
    original=prepared_fixture()
    spec=GraphSpec(max_nodes=6)
    active=torch.tensor([False,True,False,False,True,False])
    view=rebind_slots(original,spec,active)
    assert torch.equal(view.slot_to_atom[active],original.slot_to_atom[original.active])
    graph=view.seed.to_relaxed(spec)
    graph.atoms.requires_grad_();graph.charges.requires_grad_();graph.bonds.requires_grad_()
    features,bonds=adapt_graph(graph,view)
    for key in ('ref_element','ref_charge','token_bonds'):
        assert torch.equal(features[key],original.features[key].to(features[key]))
    loss=features['ref_element'].square().sum()+features['ref_charge'].square().sum()+bonds.square().sum()
    gradients=torch.autograd.grad(loss,(graph.atoms,graph.charges,graph.bonds))
    assert all(g.isfinite().all() for g in gradients)
    assert view.coordinates is original.coordinates
    assert view.features is original.features


def test_reindex_refuses_wrong_occupied_count():
    with pytest.raises(ValueError,match='occupancy counts'):
        rebind_slots(prepared_fixture(),GraphSpec(max_nodes=6),torch.ones(6,dtype=torch.bool))


def test_measured_budget_uses_full_remainder_when_it_fits_and_bounds_long_runs():
    path=Path('experiments/latent-design/protenix-vae/dbmol_campaign.py')
    # Import only the portable budget function, without campaign dependencies.
    import ast,math
    tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='remaining_budget')
    namespace={'math':math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    budget=namespace['remaining_budget']
    assert budget(1.,49980)[0]==49980
    count,wall=budget(10.,49980)
    assert 0<count<49980 and wall=='72:00'
    with pytest.raises(ValueError):budget(float('nan'),10)
