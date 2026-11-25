"""Bounded GASA bridge checks; soft features are explicitly approximations."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.synthesis_gasa import FrozenGASA, _incoming_attention, reference_features
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph


ASSETS = Path(__file__).resolve().parents[2] / "outputs/synthesis-next-models-20261113/gasa"


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def model():
    path = ASSETS / "gasa.pth"
    if not path.exists():
        pytest.skip("Public GASA checkpoint has not been downloaded")
    return FrozenGASA(GraphSpec(max_nodes=8), path)


def hard_input(smiles, spec):
    row = encode_smiles(smiles, spec)
    hard = HardGraphBatch(*(torch.tensor(row[key].astype("int64"))[None]
                            for key in ("atom_types", "charges", "bond_types")))
    return row["smiles"], hard.to_relaxed(spec), hard.atoms[0] != 0


def sparse_reference(model, edge, adjacency):
    """Edge-list reference of the released DGL incoming-reduction algebra.

    This is independent of the adapter's dense attention/matrix reductions.
    It checks network orientation/weights, not native DGL runtime or RDKit.
    """
    def linear(h, name, bias=True):
        return F.linear(h, model._weight(name + ".weight"),
                        model._weight(name + ".bias") if bias else None)

    outputs = []
    for features, matrix in zip(edge, adjacency):
        count = matrix.shape[0]
        sources = [torch.nonzero(matrix[:, dst], as_tuple=True)[0] for dst in range(count)]

        def attention(h, name, global_layer=False):
            heads, width = model._weight(name + ".attn_l").shape[1:]
            values = linear(h, name + ".fc", False).reshape(count, heads, width)
            left = (values * model._weight(name + ".attn_l")).sum(-1)
            right = (values * model._weight(name + ".attn_r")).sum(-1)
            result = []
            for dst, incoming in enumerate(sources):
                logits = F.leaky_relu(left[incoming] + right[dst], .2)
                result.append((logits.softmax(0)[..., None] * values[incoming]).sum(0))
            result = torch.stack(result)
            return result if global_layer else result + model._weight(name + ".bias").reshape(heads, width)

        degree = torch.tensor([len(s) for s in sources], dtype=edge.dtype)[:, None]
        h = attention(degree, "gnn", True)
        h = linear(F.leaky_relu(h).flatten(1), "gnn.full")
        reduced = []
        for dst, incoming in enumerate(sources):
            message = torch.tanh(linear(torch.cat((h[incoming], features[incoming, dst]), -1), "gnn.linears1"))
            reduced.append(torch.cat((h[dst].expand(len(incoming), -1), message), -1).mean(0))
        h = torch.tanh(linear(torch.stack(reduced), "gnn.linears2"))
        h = F.elu(attention(h, "gat1")).flatten(1)
        h = F.elu(attention(h, "gat2")).flatten(1)
        h = F.elu(attention(h, "gat3")).mean(1)
        weighted = h * linear(h, "readout.weight_and_sum.atom_weighting.0").sigmoid()
        pooled = torch.cat((weighted.sum(0), h.max(0).values), -1)
        outputs.append(linear(linear(pooled, "predict.0"), "predict.1").log_softmax(-1))
    logp = torch.stack(outputs)
    return {"difficulty": -logp[:, 0], "raw_score": logp[:, 1].exp()}


def test_binary_network_matches_independent_sparse_reference(model):
    generator = torch.Generator().manual_seed(81331)
    edge = torch.randn(2, 5, 5, 11, generator=generator)
    adjacency = torch.eye(5)[None].repeat(2, 1, 1)
    # Deliberately directed, so reversing source/destination is detectable.
    adjacency[:, 0, 1] = adjacency[:, 1, 3] = adjacency[:, 4, 1] = 1
    actual = model.forward_features(edge, adjacency)
    expected = sparse_reference(model, edge, adjacency)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=4e-6, rtol=2e-5)


@pytest.mark.parametrize("smiles", ["C", "CCO", "C#N", "CC(=O)N", "c1ccccc1"])
def test_simple_hard_edge_features_match_declared_rdkit_scope(model, smiles):
    from rdkit import Chem

    canonical, graph, active = hard_input(smiles, model.spec)
    actual, adjacency = model.features(graph, active)
    expected, native_adjacency = reference_features(Chem.MolFromSmiles(canonical))
    # Missing edges are not native GASA edges and their feature values do not
    # affect binary messages. Ring proxies are close on these simple fixtures.
    mask = native_adjacency.bool()
    torch.testing.assert_close(adjacency, native_adjacency, atol=0, rtol=0)
    torch.testing.assert_close(actual[mask], expected[mask], atol=2e-4, rtol=0)


def test_soft_graph_has_finite_connected_bond_path_and_frozen_state(model):
    generator = torch.Generator().manual_seed(13116)
    spec = model.spec
    leaves = [torch.randn(shape, generator=generator, requires_grad=True) for shape in
              ((1, 8, len(spec.atomic_numbers)), (1, 8, len(spec.formal_charges)),
               (1, 8, 8, len(spec.bond_names)))]
    active = torch.tensor([True] * 5 + [False] * 3)
    graph = graph_from_decoder(DecoderOutput(*leaves), spec, fixed_active=active)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = model(graph, active)
    gradients = torch.autograd.grad(result["difficulty"].sum(), leaves, allow_unused=True)
    assert result["difficulty"].shape == (1,)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert gradients[2] is not None and gradients[2].norm() > 0
    assert 0 <= result["raw_score"].item() <= 1
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)


def test_currently_absent_edge_has_finite_creation_gradient(model):
    _, graph, active = hard_input("CCCC", model.spec)
    other = graph.bonds.clone()
    other[:, 0, 3] = other[:, 3, 0] = 0
    other[:, 0, 3, 1] = other[:, 3, 0, 1] = 1
    mixing = torch.tensor(0., requires_grad=True)
    mixed = RelaxedGraph(graph.atoms, graph.charges, graph.bonds * (1 - mixing) + other * mixing)
    gradient, = torch.autograd.grad(model(mixed, active)["difficulty"].sum(), mixing)
    assert torch.isfinite(gradient) and gradient.abs() > 0


def test_attention_stabilization_preserves_supported_edges_and_finite_boundary():
    logits = torch.tensor([[[[0.], [1000.]], [[1000.], [0.]]]])
    adjacency = torch.eye(2)[None].requires_grad_(True)
    attention = _incoming_attention(logits, adjacency)
    torch.testing.assert_close(attention[..., 0], adjacency, atol=0, rtol=0)
    gradient, = torch.autograd.grad((attention * torch.tensor([0., 1.])[None, :, None, None]).sum(), adjacency)
    assert torch.isfinite(gradient).all()


def test_nonfinite_and_incompatible_graphs_fail(model):
    _, graph, active = hard_input("CCO", model.spec)
    bonds = graph.bonds.clone()
    bonds[0, 0, 1, 1] = float("nan")
    with pytest.raises(ValueError, match="finite simplex"):
        model(RelaxedGraph(graph.atoms, graph.charges, bonds), active)
    with pytest.raises(ValueError, match="active-slot"):
        model(graph, active[:2])
