"""Exercise weight, latent-gradient and AdamW continuity across the production head split."""

from copy import deepcopy

import pytest
import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.training.attention_expansion import expand_attention


def test_attention_graft_and_optimizer_resume():
    torch.manual_seed(703)
    model = GraphAutoencoder(GraphSpec(max_nodes=5), ModelConfig(hidden=12, latent=8, layers=1,
        decoder_variant="triangle_refine", pair_hidden=8, global_hidden=16, attention_heads=4,
        refinement_rounds=4, triangle_hidden=3, encoder_pooling="canonical", encoder_degree_features=True,
        encoder_graph_attention_layers=3, extra_bond_rounds=3)).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Give the initially idle graph-attention outputs a learned value and make
    # historical AdamW moments nonuniform so incorrectly packed states are visible.
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    inputs = model.spec
    nodes = inputs.max_nodes
    atoms = torch.tensor([[1, 2, 3, 1, 0]])
    bonds = torch.zeros(1, nodes, nodes, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    data = HardGraphBatch(atoms, torch.full_like(atoms, 2), bonds)
    mu, lv = model.encoder(data, from_canonical_cache=True)
    z = torch.randn(1, 8, requires_grad=True)
    baseline = model.decode(z)
    old_gradient, = torch.autograd.grad(baseline.bonds[0, 0, 1, 2], z)
    original = dict(model.named_parameters())
    q = original["decoder.rounds.0.query.weight"]
    downstream = original["decoder.rounds.0.node_update.0.weight"]
    qstate, dstate = deepcopy(optimizer.state[q]), deepcopy(optimizer.state[downstream])
    rng = torch.get_rng_state()
    added = expand_attention(model, optimizer, seed=479)
    assert added > 0 and torch.equal(torch.get_rng_state(), rng)
    assert model.encoder.mu.weight is original["encoder.mu.weight"]
    assert model.config.attention_heads == 10 and model.config.bond_attention_layers == 2
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in model.parameters()]
    new_mu, new_lv = model.encoder(data, from_canonical_cache=True)
    torch.testing.assert_close(new_mu, mu)
    torch.testing.assert_close(new_lv, lv)
    actual = model.decode(z)
    for field in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(actual, field), getattr(baseline, field), rtol=1e-5, atol=1e-6)
    grad, = torch.autograd.grad(actual.bonds[0, 0, 1, 2], z, retain_graph=True)
    torch.testing.assert_close(grad, old_gradient, rtol=1e-5, atol=1e-6)
    widened = model.decoder.rounds[0].query.weight
    state = optimizer.state[widened]
    torch.testing.assert_close(state["exp_avg"][12:15], qstate["exp_avg"][:3] / 3)
    torch.testing.assert_close(state["exp_avg_sq"][12:15], qstate["exp_avg_sq"][:3] / 9)
    torch.testing.assert_close(optimizer.state[model.decoder.rounds[0].node_update[0].weight]["exp_avg"][:, 12:15],
                               dstate["exp_avg"][:, :3])
    assert state["step"].item() == 1 and q not in optimizer.state
    assert model.decoder.bond_attention[0].output.weight not in optimizer.state
    actual.bonds.square().sum().backward()
    assert torch.any(widened.grad[:3] != widened.grad[12:15])
    optimizer.step()
    # A checkpoint reload must use the same positional optimizer order.
    resumed = GraphAutoencoder(inputs, model.config)
    resumed.load_state_dict(model.state_dict(), strict=True)
    loaded = torch.optim.AdamW(resumed.parameters())
    loaded.load_state_dict(deepcopy(optimizer.state_dict()))
    current_params = dict(model.named_parameters())
    for name, p in resumed.named_parameters():
        current = current_params[name]
        assert loaded.state.get(p, {}).keys() == optimizer.state.get(current, {}).keys()
        for key, value in loaded.state.get(p, {}).items():
            torch.testing.assert_close(value, optimizer.state[current][key])
    with pytest.raises(ValueError, match="four atom heads"):
        expand_attention(model, optimizer, seed=479)
