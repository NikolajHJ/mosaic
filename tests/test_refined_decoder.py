"""New decoder contracts; run these numerical checks only on DTU HPC."""

import pytest
import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec


def config(variant: str, *, small: bool = True) -> ModelConfig:
    if small:
        return ModelConfig(hidden=8, latent=12, layers=1, decoder_variant=variant,
                           global_hidden=16, pair_hidden=6, attention_heads=2,
                           refinement_rounds=2 if variant == "pair_refine" else 0)
    return ModelConfig(hidden=96, latent=256, layers=3, decoder_variant=variant,
                       global_hidden=320, pair_hidden=48, attention_heads=4,
                       refinement_rounds=2 if variant == "pair_refine" else 0)


@pytest.mark.parametrize("variant", ["residual", "pair_refine"])
def test_refined_outputs_keep_graph_contract_and_latent_gradient(variant):
    torch.manual_seed(23)
    spec = GraphSpec(max_nodes=4)
    model = GraphAutoencoder(spec, config(variant))
    z = torch.randn(2, 12, requires_grad=True)
    logits = model.decode(z)
    assert logits.atoms.shape == (2, 4, len(spec.atomic_numbers))
    assert logits.charges.shape == (2, 4, len(spec.formal_charges))
    assert logits.bonds.shape == (2, 4, 4, len(spec.bond_names))
    assert torch.allclose(logits.bonds, logits.bonds.transpose(1, 2), atol=1e-6)
    graph = logits.probabilities()
    assert torch.allclose(graph.bonds.sum(-1), torch.ones(2, 4, 4))
    assert torch.equal(graph.bonds[:, range(4), range(4), 0], torch.ones(2, 4))
    objective = (graph.expected_size.sum() + graph.expected_charge(spec).sum()
                 + graph.bonds[..., 1:].sum())
    gradient = torch.autograd.grad(objective, z)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_soft_atom_charge_pair_feedback_reaches_earlier_heads():
    torch.manual_seed(31)
    model = GraphAutoencoder(GraphSpec(max_nodes=4), config("pair_refine"))
    logits = model.decode(torch.randn(2, 12))
    charge_to_atom = torch.autograd.grad(logits.charges[..., 1].sum(),
                                         model.decoder.atom_head.weight, retain_graph=True)[0]
    bond_to_charge = torch.autograd.grad(logits.bonds[..., 1].sum(),
                                         model.decoder.charge_head.weight, retain_graph=True)[0]
    atom_to_charge = torch.autograd.grad(logits.atoms[..., 1].sum(),
                                         model.decoder.charge_head.weight, retain_graph=True)[0]
    atom_to_pair = torch.autograd.grad(logits.atoms[..., 1].sum(),
                                       model.decoder.rounds[0].edge_bias.weight)[0]
    for gradient in (charge_to_atom, bond_to_charge, atom_to_charge, atom_to_pair):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_parameter_budget_and_control_are_explicit():
    spec = GraphSpec()
    legacy = GraphAutoencoder(spec, ModelConfig()).decoder
    control = GraphAutoencoder(spec, config("residual", small=False)).decoder
    paired = GraphAutoencoder(spec, config("pair_refine", small=False)).decoder
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert count(legacy) == 1_746_452
    assert count(control) == 1_694_964
    assert count(paired) == 1_872_476
    assert count(control) < count(paired) <= int(1.5 * count(legacy))


def test_residual_control_declares_zero_rounds_and_global_width_covers_latent():
    with pytest.raises(ValueError, match="zero"):
        ModelConfig(decoder_variant="residual", global_hidden=256)
    with pytest.raises(ValueError, match="global width"):
        ModelConfig(hidden=96, latent=256, decoder_variant="pair_refine",
                    global_hidden=128, pair_hidden=48, refinement_rounds=2)


@pytest.mark.parametrize("variant", ["deep_pair_refine", "global_bond"])
def test_deep_trunk_variants_preserve_graph_contract_and_budget(variant):
    torch.manual_seed(47)
    spec = GraphSpec(max_nodes=4)
    config = ModelConfig(hidden=8, latent=12, layers=1, decoder_variant=variant,
                         global_hidden=16, trunk_hidden=24, pair_hidden=6,
                         attention_heads=2, refinement_rounds=2)
    decoder = GraphAutoencoder(spec, config).decoder
    z = torch.randn(2, 12, requires_grad=True)
    logits = decoder(z)
    assert logits.atoms.shape == (2, 4, len(spec.atomic_numbers))
    assert logits.charges.shape == (2, 4, len(spec.formal_charges))
    assert logits.bonds.shape == (2, 4, 4, len(spec.bond_names))
    assert torch.allclose(logits.bonds, logits.bonds.transpose(1, 2), atol=1e-6)
    assert decoder.nodes[-1].out_features == (5 if variant == "global_bond" else 4) * 8
    gradient = torch.autograd.grad(logits.bonds[..., 1].sum(), z,
                                   retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    if variant == "global_bond":
        context_gradient = torch.autograd.grad(
            logits.bonds[..., 1].sum(), decoder.bond_context_norm.weight,
            retain_graph=True)[0]
        assert torch.isfinite(context_gradient).all() and context_gradient.abs().sum() > 0
        atom_to_context = torch.autograd.grad(
            logits.atoms[..., 1].sum(), decoder.bond_context_norm.weight,
            allow_unused=True)[0]
        assert atom_to_context is None


def test_deep_trunk_context_ablation_stays_under_original_decoder_cap():
    spec = GraphSpec()
    standard = GraphAutoencoder(spec, config("pair_refine", small=False)).decoder
    common = dict(hidden=96, latent=256, layers=3, global_hidden=320,
                  trunk_hidden=512, pair_hidden=48, attention_heads=4,
                  refinement_rounds=2)
    deep = GraphAutoencoder(spec, ModelConfig(decoder_variant="deep_pair_refine",
                                               **common)).decoder
    global_bond = GraphAutoencoder(spec, ModelConfig(decoder_variant="global_bond",
                                                      **common)).decoder
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert count(standard) < count(deep) < count(global_bond)
    assert count(global_bond) <= int(1.5 * 1_746_452)


def test_stable_deep_refinement_keeps_checkpoint_keys_and_finite_gradients():
    """The new normalization boundaries do not add weights or break transfer."""
    torch.manual_seed(59)
    spec = GraphSpec(max_nodes=4)
    common = dict(hidden=8, latent=12, layers=1, global_hidden=16,
                  trunk_hidden=24, pair_hidden=6, attention_heads=2,
                  refinement_rounds=2)
    historical = GraphAutoencoder(
        spec, ModelConfig(decoder_variant="deep_pair_refine", **common))
    stable = GraphAutoencoder(
        spec, ModelConfig(decoder_variant="stable_deep_pair_refine", **common))
    assert historical.state_dict().keys() == stable.state_dict().keys()
    stable.load_state_dict(historical.state_dict(), strict=True)
    assert sum(p.numel() for p in historical.parameters()) == sum(
        p.numel() for p in stable.parameters())

    z = torch.randn(2, 12, requires_grad=True)
    logits = stable.decode(z)
    assert logits.atoms.shape == (2, 4, len(spec.atomic_numbers))
    assert logits.charges.shape == (2, 4, len(spec.formal_charges))
    assert logits.bonds.shape == (2, 4, 4, len(spec.bond_names))
    assert all(torch.isfinite(value).all() for value in
               (logits.atoms, logits.charges, logits.bonds))
    assert torch.allclose(logits.bonds, logits.bonds.transpose(1, 2), atol=1e-6)
    for index, objective in enumerate((logits.atoms[..., 1].sum(),
                                       logits.charges[..., 1].sum(),
                                       logits.bonds[..., 1].sum())):
        gradient = torch.autograd.grad(objective, z, retain_graph=index < 2)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_global_bond_config_requires_deep_trunk():
    with pytest.raises(ValueError, match="trunk width"):
        ModelConfig(hidden=96, latent=256, decoder_variant="global_bond",
                    global_hidden=320, pair_hidden=48, refinement_rounds=2)
    with pytest.raises(ValueError, match="Legacy"):
        ModelConfig(decoder_variant="legacy", trunk_hidden=512)


def test_wide_global_bond_keeps_512_width_and_short_latent_gradient_paths():
    torch.manual_seed(53)
    spec = GraphSpec(max_nodes=4)
    config = ModelConfig(hidden=8, latent=12, layers=1,
                         decoder_variant="wide_global_bond", global_hidden=16,
                         trunk_hidden=24, pair_hidden=6, attention_heads=2,
                         refinement_rounds=2)
    decoder = GraphAutoencoder(spec, config).decoder
    z = torch.randn(2, 12, requires_grad=True)
    logits = decoder(z)
    assert decoder.nodes.output.in_features == 24
    assert decoder.nodes.output.out_features == 5 * 8
    assert logits.atoms.shape == (2, 4, len(spec.atomic_numbers))
    assert logits.charges.shape == (2, 4, len(spec.formal_charges))
    assert logits.bonds.shape == (2, 4, 4, len(spec.bond_names))
    assert torch.allclose(logits.bonds, logits.bonds.transpose(1, 2), atol=1e-6)
    atom_objective = logits.atoms[..., 1].sum()
    bond_objective = logits.bonds[..., 1].sum()
    atom_skip = torch.autograd.grad(
        atom_objective, decoder.nodes.latent_skip.weight, retain_graph=True)[0]
    bond_skip = torch.autograd.grad(
        bond_objective, decoder.bond_context_direct.weight, retain_graph=True)[0]
    bond_mlp = torch.autograd.grad(
        bond_objective, decoder.bond_context_update[0].weight,
        retain_graph=True)[0]
    bond_z = torch.autograd.grad(bond_objective, z, retain_graph=True)[0]
    atom_to_bond_skip = torch.autograd.grad(
        atom_objective, decoder.bond_context_direct.weight,
        allow_unused=True)[0]
    for gradient in (atom_skip, bond_skip, bond_mlp, bond_z):
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert atom_to_bond_skip is None


def test_wide_global_bond_parameter_growth_is_explicit():
    spec = GraphSpec()
    common = dict(hidden=96, latent=256, layers=3, global_hidden=320,
                  trunk_hidden=512, pair_hidden=48, attention_heads=4,
                  refinement_rounds=2)
    narrower = GraphAutoencoder(spec, ModelConfig(decoder_variant="global_bond",
                                                   **common)).decoder
    wide = GraphAutoencoder(spec, ModelConfig(decoder_variant="wide_global_bond",
                                               **common)).decoder
    count = lambda module: sum(parameter.numel() for parameter in module.parameters())
    assert wide.nodes.output.in_features == 512
    assert wide.nodes.output.out_features == 51 * 96
    assert count(narrower) == 2_242_044
    assert count(wide) == 3_191_548
    # A direct 512-to-4,896 layer supersedes the earlier 150% parameter cap.
    assert count(wide) > int(1.5 * 1_746_452)
