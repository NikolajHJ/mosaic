"""Fixed-count pretrained graph projection has deliberately different element/bond heads."""
import pytest
import torch

from latentmol.models.pretrained_graph_vae import FixedZincStart, GraphLDAZincVAE, _input_graph


def test_native_zinc_encoder_features_and_domain():
    _, nodes, edges, mask = _input_graph("N#Cc1ccccc1")
    assert nodes.shape == (1, 26, 43) and mask.sum() == 8
    assert nodes[0, 0, 2] == 1  # N in the upstream vocabulary, distinct from shared order
    assert edges[0, 0, 1, 9] == 1  # channel 9 = triple
    assert edges[0, 2, 3, 10] == 1  # aromatic ring stays aromatic
    assert edges[0, 0, 2, 2] == 1 and edges[0, 0, 2, 6] == 1  # distance 2 without bond
    assert not mask[0, 8:].any() and not nodes[0, 8:].any()
    torch.testing.assert_close(edges, edges.transpose(1, 2))
    with pytest.raises(ValueError, match="individually charged"):
        _input_graph("[NH3+]CC(=O)[O-]")  # net neutral, two charged atoms


def test_conditional_elements_bond_slice_and_latent_gradient():
    class Heads(torch.nn.Module):
        def decode(self, z, permutation, mask):
            node = z[:, None, :43].expand(-1, 26, -1)
            edge = z[:, None, None, 43:56].expand(-1, 26, 26, -1)
            return node, edge, None

    model = GraphLDAZincVAE(Heads())
    state = FixedZincStart("CC", 2, torch.arange(26)[None] < 2, torch.eye(26)[None])
    z = torch.zeros(1, 90)
    z[0, 0] = 20  # Si is conditioned out
    z[0, 3] = 3  # S maps to shared index 6
    z[0, 43 + 10] = 3  # aromatic edge label
    z.requires_grad_()
    graph = model.decode(z, state)
    assert graph.atoms[0, 0].argmax() == 6 and graph.charges[0, 1].argmax() == 2
    assert graph.bonds[0, 0, 1].argmax() == 4 and graph.bonds[0, 0, 0, 0] == 1
    torch.testing.assert_close(graph.atoms.sum(-1), torch.ones(1, 2))
    torch.testing.assert_close(graph.bonds.sum(-1), torch.ones(1, 2, 2))
    grad, = torch.autograd.grad(graph.atoms[0, 0, 6] + graph.bonds[0, 0, 1, 4], z)
    assert grad.isfinite().all() and grad[0, 3] > 0 and grad[0, 53] > 0 and grad[0, 0] == 0
