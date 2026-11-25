"""Numerical contracts for the triangle VAE."""

from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from latentmol.evaluation.evaluate import load_model
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.models.triangle import TrianglePairUpdate
from latentmol.representation import GraphSpec


def triangle_config():
    return ModelConfig(hidden=96, latent=256, layers=4, decoder_variant="triangle_refine",
                       global_hidden=320, encoder_residual_layers=2, pair_hidden=48,
                       refinement_rounds=4, attention_heads=4, triangle_hidden=24)


def test_wide_atom_heads_match_explicit_attention_outputs_and_gradients():
    """Pooling narrow edge features commutes with the linear per-head value maps."""
    torch.manual_seed(4)
    cfg = replace(triangle_config(), encoder_pooling="canonical", encoder_degree_features=True,
                  encoder_graph_attention_layers=3, extra_bond_rounds=3)
    old = GraphAutoencoder(GraphSpec(max_nodes=5), cfg)
    new = GraphAutoencoder(GraphSpec(max_nodes=5), replace(cfg, attention_heads=10, attention_head_width=24))
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(new.encoder) - count(old.encoder) == 167_382
    assert count(new.decoder) - count(old.decoder) == 252_312
    assert all(b.head_dim == 24 and b.heads == 10 for b in new.encoder.graph_attention)
    assert all(b.node_update[0].in_features == 96 for b in new.decoder.extra_rounds)
    layer = new.decoder.rounds[0].double()
    x = torch.randn(2, 5, 96, dtype=torch.float64, requires_grad=True)
    e = torch.randn(2, 5, 5, 48, dtype=torch.float64, requires_grad=True)
    occ = torch.rand(2, 5, dtype=torch.float64, requires_grad=True)
    result = layer._attention_messages(x, e, occ)
    q, k, v = (op(x).reshape(2, 5, 10, 24).transpose(1, 2)
               for op in (layer.query, layer.key, layer.value))
    scores = q @ k.transpose(-1, -2) / 24 ** 0.5
    scores = scores + layer.edge_bias(e).permute(0, 3, 1, 2) + occ.log()[:, None, None, :]
    pair_v = layer.edge_value(e).reshape(2, 5, 5, 10, 24).permute(0, 3, 1, 2, 4)
    explicit = (scores.softmax(-1)[..., None] * (v[:, :, None] + pair_v)).sum(3)
    expected = explicit.transpose(1, 2).reshape(2, 5, 240)
    torch.testing.assert_close(result, expected)
    inputs = (x, e, occ, layer.query.weight, layer.edge_value.weight, layer.edge_value.bias)
    for actual, reference in zip(torch.autograd.grad(result.square().sum(), inputs),
                                 torch.autograd.grad(expected.square().sum(), inputs)):
        torch.testing.assert_close(actual, reference)


def test_dropout_masks_only_refinement_updates_and_is_off_for_checkpoint_inference(tmp_path):
    torch.manual_seed(31)
    spec = GraphSpec(max_nodes=6)
    cfg = triangle_config()
    control = GraphAutoencoder(spec, cfg)
    candidate = GraphAutoencoder(spec, replace(cfg, atom_refinement_dropout=.05))
    candidate.load_state_dict(control.state_dict(), strict=True)
    assert list(dict(control.named_parameters())) == list(dict(candidate.named_parameters()))
    layers = candidate.decoder.rounds
    assert [m for m in candidate.modules() if isinstance(m, nn.Dropout)] == [r.atom_update_dropout for r in layers]
    assert [r.atom_update_dropout.p for r in layers] == [.05] * 4
    z = torch.randn(4, cfg.latent)
    candidate.train()
    masks = []
    hooks = [layer.atom_update_dropout.register_forward_hook(
        lambda module, inputs, output: masks.append((inputs[0].detach(), output.detach())))
             for layer in layers]
    try:
        torch.manual_seed(99)
        candidate.decode(z).atoms.square().mean().backward()
    finally:
        for handle in hooks:
            handle.remove()
    assert len(masks) == 4
    for (before, after), layer in zip(masks, layers):
        assert (after == 0).any() and (after != 0).any()
        torch.testing.assert_close(after[after != 0], before[after != 0] / .95)
        grad = layer.node_update[-1].weight.grad
        assert grad.isfinite().all() and grad.abs().sum() > 0

    candidate.eval()
    control.eval()
    for model, legacy in [(candidate, False), (control, True)]:
        saved_config = asdict(model.config)
        if legacy:
            saved_config.pop("atom_refinement_dropout")
        path = tmp_path / ("legacy.pt" if legacy else "dropout.pt")
        torch.save(dict(format_version=1, graph_spec=spec.to_dict(), model_config=saved_config,
                        model_state=model.state_dict()), path)
        loaded, _ = load_model(path, "cpu", spec)
        assert loaded.config.atom_refinement_dropout == model.config.atom_refinement_dropout
        rng = torch.get_rng_state()
        actual = loaded.decode(z)
        expected = control.decode(z)
        assert torch.equal(torch.get_rng_state(), rng)
        for name in ("atoms", "charges", "bonds"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)


def test_declared_parameter_counts_structure_and_unshared_rounds():
    model = GraphAutoencoder(GraphSpec(), triangle_config())
    count = lambda module: sum(p.numel() for p in module.parameters())
    assert count(model.encoder) == 2_173_216
    assert count(model.decoder) == 1_768_644
    assert count(model) == 3_941_860
    assert isinstance(model.decoder.nodes, nn.Linear)
    assert (model.decoder.nodes.in_features, model.decoder.nodes.out_features) == (256, 4800)
    assert len(model.encoder.layers) == len(model.decoder.rounds) == 4
    assert len(model.encoder.pool_updates) == 2
    for layer in model.decoder.rounds:
        assert [(m.in_features, m.out_features) for m in layer.node_update if isinstance(m, nn.Linear)] == [(96, 96)] * 3
        assert [(m.in_features, m.out_features) for m in layer.edge_update if isinstance(m, nn.Linear)] == [(336, 96), (96, 96), (96, 48)]
        assert count(layer.triangle) == 8400
    parameter_ids = [{id(p) for p in layer.parameters()} for layer in model.decoder.rounds]
    assert all(not a & b for i, a in enumerate(parameter_ids) for b in parameter_ids[:i])


def test_triangle_starts_identity_and_suppresses_empty_and_self_paths_after_biases():
    layer = TrianglePairUpdate(6, 3)
    pair = torch.randn(2, 4, 4, 6)
    assert torch.equal(layer(pair, torch.ones(2, 4)), torch.zeros_like(pair))
    with torch.no_grad():
        layer.output.weight.normal_()
        layer.output.bias.fill_(10)
    assert torch.equal(layer(pair, torch.zeros(2, 4)), torch.zeros_like(pair))
    # Exactly two atoms provide no distinct intermediate k for their pair.
    mask = torch.tensor([[1., 1., 0., 0.]]).expand(2, -1)
    assert torch.equal(layer(pair, mask), torch.zeros_like(pair))
    update = layer(pair, torch.ones(2, 4))
    assert torch.allclose(update, update.transpose(1, 2))
    assert torch.equal(update[:, range(4), range(4)], torch.zeros(2, 4, 6))


def test_triangle_counts_intermediate_soft_occupancy_once_not_squared():
    layer = TrianglePairUpdate(4, 2)
    with torch.no_grad():
        for projection in (layer.left, layer.right, layer.left_gate, layer.right_gate):
            projection.weight.zero_()
            projection.bias.zero_()
        layer.left.bias.copy_(torch.tensor([1., 2.]))
        layer.right.bias.copy_(torch.tensor([3., 4.]))
    captured = []
    hook = layer.product_norm.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    layer(torch.randn(1, 3, 3, 4), torch.tensor([[1., 0.4, 1.]]))
    hook.remove()
    # Sigmoid gates=.5 each; denominator floor1 preserves soft intermediate mass.
    assert torch.allclose(captured[0][0, 0, 2], torch.tensor([0.3, 0.8]), atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires allocated GPU"))])
def test_shared_relaxed_contract_and_latent_triangle_gradients(device):
    torch.manual_seed(17)
    spec = GraphSpec(max_nodes=6)
    model = GraphAutoencoder(spec, triangle_config()).to(device)
    z = torch.randn(2, 256, device=device, requires_grad=True)
    logits = model.decode(z)
    graph = logits.probabilities()
    assert torch.allclose(logits.bonds, logits.bonds.transpose(1, 2), atol=1e-6)
    assert torch.allclose(graph.bonds.sum(-1), torch.ones_like(graph.bonds[..., 0]), atol=1e-6)
    assert torch.equal(graph.bonds[:, range(6), range(6), 0], torch.ones(2, 6, device=device))
    objective = logits.atoms[..., 1].sum() + logits.charges[..., 1].sum() + logits.bonds[..., 1].sum()
    gradient = torch.autograd.grad(objective, z, retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    triangle = model.decoder.rounds[0].triangle
    output_gradient = torch.autograd.grad(objective, triangle.output.weight)[0]
    assert torch.isfinite(output_gradient).all() and output_gradient.abs().sum() > 0
    # Earlier triangle projections receive gradients after the zero-initialized
    # output branch has moved; they need not receive one on its first update.
    with torch.no_grad():
        triangle.output.weight.normal_(std=.01)
    gradient = torch.autograd.grad(model.decode(z).bonds[..., 1].sum(), triangle.left.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_historical_config_strict_load_keeps_original_counts_and_keys():
    old = {"hidden": 96, "latent": 256, "layers": 3, "decoder_variant": "stable_deep_pair_refine",
           "global_hidden": 320, "pair_hidden": 48, "refinement_rounds": 2, "attention_heads": 4,
           "trunk_hidden": 512}
    model = GraphAutoencoder(GraphSpec(), ModelConfig(**old))
    state = model.state_dict()
    assert not any("pool_updates" in key or ".triangle." in key for key in state)
    assert sum(p.numel() for p in model.encoder.parameters()) == 1_901_600
    assert sum(p.numel() for p in model.decoder.parameters()) == 2_201_628
    restored = GraphAutoencoder(GraphSpec(), ModelConfig(**old))
    restored.load_state_dict(state, strict=True)
    z = torch.randn(1, 256)
    with torch.no_grad():
        a, b = model.decode(z), restored.decode(z)
    for left, right in ((a.atoms, b.atoms), (a.charges, b.charges), (a.bonds, b.bonds)):
        assert torch.equal(left, right)
