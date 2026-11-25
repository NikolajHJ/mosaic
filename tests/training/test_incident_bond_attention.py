"""Bonds share a neighborhood at both ends, without using hard predicted bond labels."""

from dataclasses import replace

import torch

from latentmol.models.autoencoder import GraphAutoencoder, ModelConfig
from latentmol.models.bond_attention import IncidentBondAttention
from latentmol.representation import GraphSpec


def symmetric_inputs(nodes, width):
    raw = torch.randn(1, nodes, nodes, width)
    pair = (raw + raw.transpose(1, 2)) * 0.5
    return pair.masked_fill(torch.eye(nodes, dtype=torch.bool)[None, :, :, None], 0.)


def active_block():
    block = IncidentBondAttention(8, 2, 4)
    torch.nn.init.normal_(block.output.weight, std=.1)
    return block


def test_identity_graft_and_live_gradients_on_normal_decoder_path():
    torch.manual_seed(21)
    config = ModelConfig(hidden=12, latent=8, layers=1, decoder_variant="triangle_refine",
                         global_hidden=12, attention_heads=3, refinement_rounds=1, pair_hidden=48)
    base = GraphAutoencoder(GraphSpec(max_nodes=6), config)
    expanded = GraphAutoencoder(base.spec, replace(config, bond_attention_layers=2))
    expanded.load_state_dict({**expanded.state_dict(), **base.state_dict()})
    blocks = expanded.decoder.bond_attention
    assert len(blocks) == 2 and blocks[0] is not blocks[1]
    assert sum(p.numel() for p in blocks.parameters()) == 112800
    z = torch.randn(2, 8, requires_grad=True)
    old, new = base.decode(z), expanded.decode(z)
    for field in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(new, field), getattr(old, field), rtol=0, atol=0)
    new.bonds[0, 1, 2, 1].backward()
    assert z.grad.isfinite().all() and z.grad.norm() > 0
    for block in blocks:
        for out in (block.output, block.feedforward[-1]):
            assert out.weight.grad.isfinite().all() and out.weight.grad.norm() > 0


def test_incidence_symmetry_slot_permutation_and_soft_occupancy():
    torch.manual_seed(52)
    block = active_block()
    pair = symmetric_inputs(6, 8).requires_grad_()
    occ = torch.tensor([[1., .85, .55, .15, 0., .91]], requires_grad=True)
    out = block(pair, occ)
    torch.testing.assert_close(out, out.transpose(1, 2), rtol=0, atol=0)
    perm = torch.tensor([3, 1, 0, 4, 2, 5])
    torch.testing.assert_close(block(pair[:, perm][:, :, perm], occ[:, perm]),
                               out[:, perm][:, :, perm])
    dp, do = torch.autograd.grad(out[0, 0, 1, 0], (pair, occ))
    assert dp.isfinite().all() and do.isfinite().all()
    assert dp[0, 0, 3].norm() > 0 and do[0, 3].abs() > 0
    assert dp[0, 1, 3].norm() > 0
    assert dp[0, 0, 4].norm() == 0 and dp[0, 1, 4].norm() == 0
    assert dp[0, 2, 3].norm() == 0


def test_second_round_connects_disjoint_pairs_and_handles_empty_context():
    torch.manual_seed(17)
    first, second = active_block(), active_block()
    pair = symmetric_inputs(5, 8).requires_grad_()
    occ = torch.ones(1, 5)
    out = second(first(pair, occ), occ)
    assert torch.autograd.grad(out[0, 0, 1, 0], pair)[0][0, 2, 3].norm() > 0
    empty = first(pair, torch.zeros_like(occ))
    torch.testing.assert_close(empty, pair, rtol=0, atol=0)
