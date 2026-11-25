"""Bounded checks for the exploratory, frozen DFRscore soft adapter."""
import importlib.util
from pathlib import Path

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.synthesis_dfr import FrozenDFRscore, _ring_walk_features, _WeightedAttention
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph


ASSETS = Path(__file__).resolve().parents[2] / "outputs/restart-synthesis-pilot-20261111a/assets/dfr"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def reference_module(name):
    path = ASSETS / name
    if not path.exists():
        pytest.skip("Public DFRscore assets have not been downloaded")
    spec = importlib.util.spec_from_file_location("_dfr_reference_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model():
    weights = ASSETS / "Best_model_163.pt"
    if not weights.exists():
        pytest.skip("Public DFRscore checkpoint has not been downloaded")
    return FrozenDFRscore(GraphSpec(max_nodes=8), weights)


def hard_input(smiles, spec):
    row = encode_smiles(smiles, spec)
    hard = HardGraphBatch(*(torch.tensor(row[key].astype("int64"))[None]
                            for key in ("atom_types", "charges", "bond_types")))
    return row["smiles"], hard.to_relaxed(spec), hard.atoms[0] != 0


@pytest.mark.parametrize("smiles", ["CCO", "CC(=O)N", "C#N", "[NH4+]", "c1ccccc1"])
def test_simple_hard_feature_scope_matches_official_featurization(model, smiles):
    from rdkit import Chem
    import numpy as np

    native = reference_module("preprocessing.py")
    canonical, graph, active = hard_input(smiles, model.spec)
    mol = Chem.MolFromSmiles(canonical)
    atoms = np.stack([native.get_node_feature(a) for a in mol.GetAtoms()])
    rings = native.sssr_to_ring_feature(Chem.GetSymmSSSR(mol), mol.GetNumAtoms())
    expected = torch.tensor(np.concatenate((atoms, rings), -1)).float()[None]
    actual, _ = model.features(graph, active)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)


def test_released_network_matches_official_binary_attention_and_features(model):
    native = reference_module("layers.py")
    torch.manual_seed(173)
    x = torch.randn(2, 6, 49)
    a = torch.eye(6)[None].expand(2, -1, -1).clone()
    a[:, torch.arange(5), torch.arange(1, 6)] = 1
    a[:, torch.arange(1, 6), torch.arange(5)] = 1
    h = model.embedding(x)
    for layer in model.GAT_layers:
        stock = native.GraphAttentionLayer(256, 8)
        stock.load_state_dict(layer.state_dict())
        h = stock(h, a)
    stock_dense = native.FeedForward(256, 1, [128, 128], 0)
    stock_dense.load_state_dict(model.dense.state_dict())
    expected = torch.nn.functional.elu(stock_dense(h.mean(1))).squeeze(-1) + 1.5
    actual = model.score_features(x, a)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-6)


def test_live_atom_charge_and_bond_paths_leave_frozen_model_unchanged(model):
    generator = torch.Generator().manual_seed(61311)
    spec = model.spec
    leaves = [torch.randn(shape, generator=generator, requires_grad=True) for shape in
              ((1, 8, len(spec.atomic_numbers)), (1, 8, len(spec.formal_charges)),
               (1, 8, 8, len(spec.bond_names)))]
    active = torch.tensor([True] * 6 + [False] * 2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    graph = graph_from_decoder(DecoderOutput(*leaves), spec, fixed_active=active)
    result = model(graph, active)
    gradients = torch.autograd.grad(result["difficulty"].sum(), leaves)
    assert result["difficulty"].shape == (1,)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_missing_bond_has_a_live_creation_direction(model):
    _, graph, active = hard_input("CCCC", model.spec)
    # A one-sided blend toward a ring is a genuine bond-existence direction.
    blend = torch.tensor(0.1, requires_grad=True)
    other = graph.bonds.clone()
    other[:, 0, 3] = 0
    other[:, 3, 0] = 0
    other[:, 0, 3, model.spec.bond_names.index("SINGLE")] = 1
    other[:, 3, 0, model.spec.bond_names.index("SINGLE")] = 1
    mixed = RelaxedGraph(graph.atoms, graph.charges, graph.bonds * (1 - blend) + other * blend)
    gradient, = torch.autograd.grad(model(mixed, active)["difficulty"].sum(), blend)
    assert torch.isfinite(gradient) and gradient.abs() > 0


def test_tiny_real_bond_mass_survives_rounded_none_probability(model):
    _, graph, active = hard_input("CCCC", model.spec)
    bonds = graph.bonds.clone()
    # The float32 simplex has real mass even though its NONE entry rounds1.
    bonds[:, 0, 3, 1] = bonds[:, 3, 0, 1] = 1e-9
    assert bonds[0, 0, 3, 0] == 1
    bonds.requires_grad_(True)
    graph = RelaxedGraph(graph.atoms, graph.charges, bonds)
    _, adjacency = model.features(graph, active)
    assert float(adjacency[0, 0, 3].detach()) == pytest.approx(1e-9)
    gradient, = torch.autograd.grad(model(graph, active)["difficulty"].sum(), bonds)
    assert torch.isfinite(gradient).all() and torch.isfinite(gradient.norm())
    assert gradient[..., 1:].norm() > 0


def test_absent_edge_boundary_derivative_uses_finite_attention_exponent():
    attention = _WeightedAttention(dim=8, heads=1).requires_grad_(False)
    with torch.no_grad():
        attention.W.weight.copy_(torch.eye(8))
        attention.W.bias.zero_()
        attention.a.weight.zero_()
        attention.a.bias.zero_()
        attention.a.weight[1, 0] = 100
    x = torch.zeros(1, 3, 8)
    x[:, :, 0] = 1
    x[:, 2, 0] = 2
    adjacency = torch.eye(3)[None].requires_grad_(True)
    gradient, = torch.autograd.grad(attention(x, adjacency).sum(), adjacency)
    assert torch.isfinite(gradient).all() and torch.isfinite(gradient.norm())
    assert gradient[0, 0, 2] > 0
    assert gradient.abs().max() < 1e10


@pytest.mark.parametrize("founder", range(3))
def test_recorded_actual_decoder_graphs_have_stable_real_bond_derivatives(founder):
    import json
    import numpy as np

    pilot = ASSETS.parent.parent
    probabilities = pilot / "actual-founder-graphs.npz"
    if not probabilities.exists() or not (ASSETS / "Best_model_163.pt").exists():
        pytest.skip("Allocated-CPU archived-founder probability fixture is not present")
    spec = GraphSpec.from_dict(json.loads((pilot / "parents.json").read_text())["graph_spec"])
    model = FrozenDFRscore(spec, ASSETS / "Best_model_163.pt")
    with np.load(probabilities, allow_pickle=False) as arrays:
        fields = [torch.tensor(arrays[f"{founder}_{name}"], requires_grad=True)
                  for name in ("atoms", "charges", "bonds")]
        active = torch.tensor(arrays[f"{founder}_active"])
    graph = RelaxedGraph(*fields)
    real = graph.bonds[..., 1:].sum(-1)
    assert ((graph.bonds[..., 0] == 1) & (real > 0)).any()
    gradients = torch.autograd.grad(model(graph, active)["difficulty"].sum(), fields)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and torch.isfinite(gradient.norm())
    # The previous subtract-NONE/cap60 adapter exceeded this by orders of
    # magnitude on these exact fixtures, with float32 norm overflow on deep.
    assert 0 < gradients[-1].double().norm() < 1e10


def test_ring_proxy_distinguishes_a_chain_and_isolated_triangle():
    chain = torch.tensor([[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]])
    triangle = torch.ones(1, 3, 3) - torch.eye(3)[None]
    assert torch.count_nonzero(_ring_walk_features(chain)) == 0
    expected = torch.zeros(1, 3, 6)
    expected[..., 0] = 1
    torch.testing.assert_close(_ring_walk_features(triangle), expected, atol=0, rtol=0)


def test_ring_proxy_does_not_treat_a_ring_tail_as_a_larger_ring():
    edge = torch.zeros(1, 7, 7)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6)):
        edge[:, left, right] = edge[:, right, left] = 1
    features = _ring_walk_features(edge)
    expected = torch.zeros(1, 7, 6)
    expected[:, :6, 3] = 1
    torch.testing.assert_close(features, expected, atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["nan", "asymmetric", "variable_occupancy"])
def test_rejects_incompatible_live_graphs(model, defect):
    _, graph, active = hard_input("CCO", model.spec)
    if defect == "nan":
        graph.charges[0, 0, 0] = torch.nan
    elif defect == "asymmetric":
        graph.bonds[0, 0, 2] = graph.bonds[0, 0, 1]
    else:
        graph.atoms[0, 0] = 0
        graph.atoms[0, 0, 0] = 0.1
        graph.atoms[0, 0, 1] = 0.9
    with pytest.raises(ValueError):
        model(graph, active)
