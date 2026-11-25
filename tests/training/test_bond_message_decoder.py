"""Bond-message ablation of the tiny VAE and its continuous latent scorer path."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from latentmol.evaluation.evaluate import load_model
from latentmol.evaluation.bond_message_ablation import bond_message_ablation
from latentmol.models import GraphAutoencoder
from latentmol.models.refined_decoder import BondMessageAggregation
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig


def tiny_model(nodes=50, variant="bond_message"):
    path = Path(__file__).resolve().parents[2] / "configs/vae_compact_triangle.yaml"
    config = replace(TrainConfig.from_yaml(path).model, decoder_variant=variant)
    return GraphAutoencoder(GraphSpec(max_nodes=nodes), config)


def test_heldout_message_ablation_measures_shared_masks_and_restores_on_exception():
    torch.manual_seed(101)
    model = tiny_model(nodes=4)
    atoms = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    bonds = torch.zeros(2, 4, 4, dtype=torch.long)
    bonds[:, 0, 1] = bonds[:, 1, 0] = 1
    bonds[0, 1, 2] = bonds[0, 2, 1] = 1
    batch = HardGraphBatch(atoms, torch.full_like(atoms, model.spec.formal_charges.index(0)), bonds)
    modules = [layer.bond_messages for layer in model.decoder.rounds]
    report = bond_message_ablation(model, [batch, batch], limit=3)
    assert report["samples"] == 3 and report["silenced_rounds"] == 4 and report["restored_logits_exact"]
    assert report["rows_per_measure"] == {"target_atoms": 8, "true_bonds": 5, "absent_active_pairs": 2}
    assert all(0 < tv < 1 for tv in report["mean_total_variation"].values())
    assert report["hard_label_change_fraction"] == report["hard_label_changes"] / 3
    assert report["different_molecules_among_both_valid"] <= report["both_valid"] <= 3
    assert model.training and all(not m._forward_hooks for m in modules)
    # Fail inside the second decode while the ablation hooks are active.
    def fail_when_silenced(_module, _args):
        if modules[0]._forward_hooks:
            raise RuntimeError("test ablation cleanup")
    handle = model.decoder.register_forward_pre_hook(fail_when_silenced)
    try:
        with pytest.raises(RuntimeError, match="test ablation cleanup"):
            bond_message_ablation(model, [batch])
    finally:
        handle.remove()
    assert model.training and all(not m._forward_hooks for m in modules)


def test_typed_messages_match_expected_chemistry_and_use_endpoint_mass_once():
    layer = BondMessageAggregation(width=1, bond_types=3).double()
    with torch.no_grad():
        layer.values.weight.copy_(torch.tensor([[2.], [5.]]))
    # Conditional NONE/SINGLE/DOUBLE distribution. Diagonal logits propose
    # almost certain self-bonds; these must contribute exactly zero.
    p = torch.tensor([.7, .1, .2], dtype=torch.double)
    diagonal = torch.tensor([.001, .001, .998], dtype=torch.double)
    logits = torch.stack([diagonal, p, p, diagonal]).view(1, 2, 2, 3).log()
    occupancy = torch.tensor([[.5, .8]], dtype=torch.double)
    values = torch.tensor([[[1.], [3.]]], dtype=torch.double)
    output = layer(values, logits, occupancy)
    pair_mass = .5 * .8
    weighted = pair_mass * (.1 * 2 + .2 * 5) / (1 + pair_mass * .3)**.5
    torch.testing.assert_close(output, values.flip(1) * weighted)
    torch.testing.assert_close(layer(values, logits, occupancy * torch.tensor([1., 0.])),
                               torch.zeros_like(output))


def test_weak_bond_can_strengthen_by_gradient_through_the_messages():
    layer = BondMessageAggregation(width=1, bond_types=2).double()
    with torch.no_grad():
        layer.values.weight.fill_(1.)
    # Enforce one shared variable for both pair directions; argmax is NO_BOND.
    edge_logit = torch.tensor(-9., dtype=torch.double, requires_grad=True)
    occupied = torch.tensor([[.02, 1.]], dtype=torch.double, requires_grad=True)
    values = torch.tensor([[[0.], [2.]]], dtype=torch.double)

    def incoming(logit):
        logits = torch.stack((torch.zeros_like(logit), logit)).expand(1, 2, 2, -1)
        return layer(values, logits, occupied)[0, 0, 0]

    signal = incoming(edge_logit)
    edge_grad, occupancy_grad = torch.autograd.grad(signal, (edge_logit, occupied))
    assert edge_grad.isfinite() and edge_grad > 0 and (occupancy_grad > 0).all()
    delta = 1e-5
    central = (incoming(edge_logit + delta) - incoming(edge_logit - delta)) / (2 * delta)
    torch.testing.assert_close(edge_grad, central, rtol=1e-6, atol=1e-13)
    assert incoming(edge_logit + .1 * edge_grad.sign()) > signal


@pytest.mark.parametrize(("variant", "decoder_parameters"), [
    ("bond_message", 190736), ("triangle_bond_message", 196928)])
def test_all_soft_bond_predictions_influence_atom_gradient_and_preserve_schema(tmp_path, variant, decoder_parameters):
    torch.manual_seed(21)
    model = tiny_model(variant=variant).eval()
    assert [sum(p.numel() for p in part.parameters()) for part in (model.encoder, model.decoder)] == [238360, decoder_parameters]
    assert len(model.decoder.rounds) == 4
    spec, cfg = model.spec, model.config
    z = torch.randn(2, cfg.latent, requires_grad=True)
    heads = []
    handle = model.decoder.bond_head.register_forward_hook(lambda _, inputs, bonds: heads.append(bonds))
    try:
        logits, features = model.decode_with_features(z)
    finally:
        handle.remove()
    assert len(heads) == 5 and features.pair_hidden.shape == (2, 50, 50, cfg.pair_hidden)
    graph = logits.probabilities()
    atom_objective = graph.atoms[0, 0, 1] - graph.charges[1, 1, 0]
    grads = torch.autograd.grad(atom_objective, [z, *heads[:-1]])
    assert all(g.isfinite().all() and g.abs().sum() > 0 for g in grads)
    for bonds in heads:
        torch.testing.assert_close(bonds, bonds.transpose(1, 2))
    torch.testing.assert_close(graph.bonds.sum(-1), torch.ones_like(graph.bonds[..., 0]))
    assert torch.all(graph.bonds[..., 0].diagonal(dim1=1, dim2=2) == 1)

    path = tmp_path / "bond-message.pt"
    torch.save(dict(format_version=1, graph_spec=spec.to_dict(), model_config=asdict(cfg),
                    model_state=model.state_dict()), path)
    restored, _ = load_model(path, "cpu", spec)
    assert restored.config == cfg
    for field in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(restored.decode(z), field), getattr(logits, field), rtol=0, atol=0)

    # The ordinary training entry point consumes verified hard canonical graphs
    # and still computes a differentiable posterior draw with this decoder.
    encoded = logits.harden(spec)
    result = restored(encoded, sample_latent=True, from_canonical_cache=True)
    assert result.mu.shape == result.logvar.shape == result.z.shape == (2, cfg.latent)
    assert isinstance(result.decoded.harden(spec), HardGraphBatch)
    assert torch.autograd.grad(result.decoded.bonds[..., 1].mean(), restored.encoder.mu.weight)[0].norm() > 0


def test_hybrid_starts_as_matched_message_control_and_learns_through_every_triangle():
    torch.manual_seed(43)
    control = tiny_model(nodes=6).eval()
    hybrid = tiny_model(nodes=6, variant="triangle_bond_message").eval()
    state = hybrid.state_dict()
    added = state.keys() - control.state_dict().keys()
    assert added and all(".triangle." in k for k in added)
    # Paired initialization copies all common tensors, including the four message rounds.
    hybrid.load_state_dict({**state, **control.state_dict()}, strict=True)
    z = torch.randn(2, hybrid.config.latent)
    reference, initial = control.decode(z), hybrid.decode(z)
    for field in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(initial, field), getattr(reference, field), rtol=0, atol=0)
    rounds = hybrid.decoder.rounds
    assert all(layer.bond_messages is not None and not hasattr(layer, "query") for layer in rounds)
    # Only update the added branch. With its zero output projection, one gradient
    # step awakens its internal two-edge projections without altering the control.
    optimizer = torch.optim.SGD([p for layer in rounds for p in layer.triangle.parameters()], lr=.1)
    readout = lambda logits: logits.atoms[0, 0, 1] + logits.bonds[1, 0, 3, 1]
    readout(initial).backward()
    assert all(torch.isfinite(layer.triangle.output.weight.grad).all()
               and layer.triangle.output.weight.grad.norm() > 0 for layer in rounds)
    optimizer.step()
    changed = hybrid.decode(z)
    assert not torch.equal(changed.bonds, initial.bonds)
    grad = torch.autograd.grad(readout(changed), [layer.triangle.left.weight for layer in rounds])
    assert all(g.isfinite().all() and g.norm() > 0 for g in grad)


def test_bond_round_commutes_with_a_permutation_of_all_atom_slots():
    torch.manual_seed(23)
    decoder = tiny_model(nodes=5).decoder
    layer = decoder.rounds[0]
    perm = torch.tensor([4, 2, 0, 3, 1])
    h, context = torch.randn(2, 1, 5, 24)
    occupancy = torch.rand(1, 5)
    edge = torch.randn(1, 5, 5, 18)
    edge = (edge + edge.transpose(1, 2)) * .5
    out_h, out_e = layer(h, edge, context, occupancy, bond_head=decoder.bond_head)
    perm_h, perm_e = layer(h[:, perm], edge[:, perm][:, :, perm], context[:, perm], occupancy[:, perm],
                           bond_head=decoder.bond_head)
    torch.testing.assert_close(perm_h, out_h[:, perm])
    torch.testing.assert_close(perm_e, out_e[:, perm][:, :, perm])


def test_config_requires_a_real_refinement_round_and_allows_non_attention_widths():
    cfg = tiny_model(nodes=2).config
    with pytest.raises(ValueError, match="at least one"):
        replace(cfg, refinement_rounds=0)
    model = GraphAutoencoder(GraphSpec(max_nodes=2), replace(cfg, hidden=19))
    assert model.decode(torch.randn(1, cfg.latent)).atoms.shape[:2] == (1, 2)
