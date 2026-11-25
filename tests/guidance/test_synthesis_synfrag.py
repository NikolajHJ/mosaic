"""Frozen SynFrag bridge: hard-network reference and connected soft inputs."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.synthesis_synfrag import FrozenSynFrag, reference_features, soft_features
from latentmol.representation import DecoderOutput, GraphSpec, RelaxedGraph


WEIGHTS = Path("outputs/restart-synthesis-pilot-20261111a/assets/synfrag/synfrag_default.pth")


@pytest.fixture
def model():
    if not WEIGHTS.exists():
        pytest.skip("Public SynFrag weights have not been downloaded")
    torch.set_num_threads(1)
    return FrozenSynFrag(GraphSpec(max_nodes=5), WEIGHTS).double()


def _graph(model):
    spec = model.spec
    generator = torch.Generator().manual_seed(123)
    logits = [torch.randn(shape, dtype=torch.float64, generator=generator, requires_grad=True)
              for shape in ((1, 5, len(spec.atomic_numbers)), (1, 5, len(spec.formal_charges)),
                            (1, 5, 5, len(spec.bond_names)))]
    active = torch.ones(5, dtype=torch.bool)
    graph = graph_from_decoder(DecoderOutput(*logits), spec, fixed_active=active)
    return graph, active, logits


def _edge_list_reference(model, node, edge, mask):
    """Present-edge-only reference following released DGL message equations.

    Uses separate per-node softmax and edge gathering rather than the dense
    probability-weighted attention. This checks hard graph limiting behavior,
    not native DGL/DeepChem runtime parity.
    """
    node, edge = node[0], edge[0]
    dest, src = mask[0].nonzero(as_tuple=True)
    prefix = "gnn.init_context."
    hidden = F.leaky_relu(model._linear(node, prefix + "project_node.0"))
    he = F.leaky_relu(model._linear(torch.cat((node[src], edge[dest, src]), -1), prefix + "project_edge1.0"))
    logit = F.leaky_relu(model._linear(torch.cat((hidden[dest], he), -1), prefix + "project_edge2.1")).squeeze(-1)
    message = model._linear(he, prefix + "attentive_gru.edge_transform.1")
    context = torch.stack([(logit[dest == i].softmax(0)[:, None] * message[dest == i]).sum(0)
                           for i in range(len(node))])
    hidden = F.relu(model._gru(F.elu(context), hidden, prefix + "attentive_gru.gru"))
    for layer in range(3):
        prefix = f"gnn.gnn_layers.{layer}."
        logit = F.leaky_relu(model._linear(torch.cat((hidden[dest], hidden[src]), -1), prefix + "project_edge.1")).squeeze(-1)
        projected = model._linear(hidden, prefix + "attentive_gru.project_node.1")
        context = torch.stack([(logit[dest == i].softmax(0)[:, None] * projected[src[dest == i]]).sum(0)
                               for i in range(len(node))])
        hidden = F.relu(model._gru(F.elu(context), hidden, prefix + "attentive_gru.gru"))
    graph = hidden.sum(0)
    prefix = "readout.readouts.0."
    logit = F.leaky_relu(model._linear(torch.cat((F.relu(graph).expand_as(hidden), hidden), -1),
                                      prefix + "compute_logits.0")).squeeze(-1)
    context = F.elu((logit.softmax(0)[:, None] * model._linear(hidden, prefix + "project_nodes.1")).sum(0))
    graph = model._gru(context, graph, prefix + "gru")
    return model._linear(graph, "predict.1").squeeze(-1)


def test_hard_feature_network_matches_present_edge_reference(model):
    from rdkit import Chem
    features = [x.double() for x in reference_features(Chem.MolFromSmiles("CC(=O)N"))]
    actual = model.forward_features(*features)["difficulty"][0]
    expected = _edge_list_reference(model, *features)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    # A nonexistent edge's arbitrary features cannot influence its hard score.
    node, edge, mask = features
    changed = edge.clone()
    changed[mask == 0] = 20
    torch.testing.assert_close(model.forward_features(node, changed, mask)["difficulty"], actual[None],
                               atol=1e-10, rtol=1e-10)


def test_atom_charge_bond_gradients_connected_and_weights_unchanged(model):
    graph, active, logits = _graph(model)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    score = model(graph, active)
    gradients = torch.autograd.grad(score["difficulty"].sum(), logits)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-8 for g in gradients)
    assert not list(model.parameters())
    assert all(not value.requires_grad for value in model.buffers())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    torch.testing.assert_close(score["difficulty"] + score["raw_score"], torch.ones(1, dtype=torch.float64))


def test_soft_graph_permutation_invariant(model):
    graph, active, _ = _graph(model)
    p = torch.tensor([3, 0, 4, 1, 2])
    permuted = RelaxedGraph(graph.atoms[:, p], graph.charges[:, p], graph.bonds[:, p][:, :, p])
    torch.testing.assert_close(model(graph, active)["difficulty"], model(permuted, active)["difficulty"],
                               atol=1e-8, rtol=1e-8)


def test_absent_edge_has_a_creation_gradient(model):
    from rdkit import Chem
    node, edge, mask = [t.double() for t in reference_features(Chem.MolFromSmiles("CCC"))]
    mask.requires_grad_()
    gradient, = torch.autograd.grad(model.forward_features(node, edge, mask)["difficulty"].sum(), mask)
    assert abs(float(gradient[0, 0, 2])) > 1e-8
    assert torch.isfinite(gradient).all()


def test_absent_extreme_logit_cannot_erase_supported_attention():
    logits = torch.tensor([[1000., 0.]], requires_grad=True)
    weights = torch.tensor([[0., 1.]], requires_grad=True)
    attention = FrozenSynFrag._attention(logits, weights)
    torch.testing.assert_close(attention, torch.tensor([[0., 1.]]))
    gradient, = torch.autograd.grad(attention[0, 0], weights)
    assert gradient[0, 0] > 0
    assert torch.isfinite(gradient).all() and torch.isfinite(gradient.norm())


def test_tiny_positive_bond_survives_none_rounding():
    spec = GraphSpec(max_nodes=3)
    atoms = F.one_hot(torch.full((1, 3), spec.atomic_numbers.index(6)),
                      len(spec.atomic_numbers)).float()
    charges = F.one_hot(torch.full((1, 3), spec.formal_charges.index(0)),
                        len(spec.formal_charges)).float()
    tiny = torch.tensor(1e-10, requires_grad=True)
    bonds = torch.zeros((1, 3, 3, len(spec.bond_names)))
    bonds[..., 0] = 1
    bonds[..., 1] = tiny * (1 - torch.eye(3))
    graph = RelaxedGraph(atoms, charges, bonds)
    _, _, weights = soft_features(graph, torch.ones(3, dtype=torch.bool), spec)
    assert float(1 - bonds[0, 0, 1, 0]) == 0
    torch.testing.assert_close(weights[0, 0, 1], tiny, atol=0, rtol=0)
    gradient, = torch.autograd.grad(weights[0, 0, 1], tiny)
    torch.testing.assert_close(gradient, torch.tensor(1.))


def test_one_soft_logit_direction_matches_finite_difference(model):
    graph, active, logits = _graph(model)
    score = model(graph, active)["difficulty"].sum()
    gradient, = torch.autograd.grad(score, logits[2])
    direction = torch.zeros_like(logits[2])
    direction[0, 0, 3, 1] = direction[0, 3, 0, 1] = 1
    epsilon = 1e-5
    def at(delta):
        decoded = DecoderOutput(logits[0], logits[1], logits[2] + delta * direction)
        return model(graph_from_decoder(decoded, model.spec, fixed_active=active), active)["difficulty"].sum()
    finite_difference = (at(epsilon) - at(-epsilon)) / (2 * epsilon)
    torch.testing.assert_close((gradient * direction).sum(), finite_difference, atol=1e-7, rtol=1e-4)


def test_incompatible_or_nonfinite_inputs_fail(model):
    graph, active, _ = _graph(model)
    with pytest.raises(ValueError, match="active mask"):
        model(graph, torch.zeros(5, dtype=torch.bool))
    graph.atoms = graph.atoms * float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        model(graph, active)
