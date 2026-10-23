"""Checkpoint-preserving hidden-state capture; numerical execution is HPC-only."""

from io import BytesIO

import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.models.refined_decoder import symmetric_pair_features
from latentmol.representation import GraphSpec


def triangle_model():
    """Use small tensors while retaining the production triangle architecture."""
    config = ModelConfig(
        hidden=8, latent=12, layers=1, decoder_variant="triangle_refine",
        global_hidden=16, pair_hidden=6, refinement_rounds=2,
        attention_heads=2, triangle_hidden=3, encoder_residual_layers=1)
    return GraphAutoencoder(GraphSpec(max_nodes=4), config).eval()


def assert_same_logits(left, right):
    for expected, actual in ((left.atoms, right.atoms),
                             (left.charges, right.charges),
                             (left.bonds, right.bonds)):
        assert torch.equal(expected, actual)


def test_capture_keeps_default_decoder_outputs_identical():
    torch.manual_seed(137)
    model = triangle_model()
    z = torch.randn(2, model.config.latent)
    with torch.no_grad():
        ordinary = model.decode(z)
        captured, features = model.decode_with_features(z)
        direct, direct_features = model.decoder(z, return_features=True)
    assert_same_logits(ordinary, captured)
    assert_same_logits(ordinary, direct)
    for field in ("atom_hidden", "pair_hidden", "node_context", "charge_input", "bond_input"):
        assert torch.equal(getattr(features, field), getattr(direct_features, field))


def test_capture_keeps_checkpoint_keys_and_strict_roundtrip_without_cached_state():
    """Capturing an example must neither persist features nor mutate model state."""
    torch.manual_seed(139)
    model = triangle_model()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    parameter_names = tuple(name for name, _ in model.named_parameters())
    checkpoint = BytesIO()
    torch.save(before, checkpoint)

    z = torch.randn(2, model.config.latent)
    ordinary, _ = model.decode_with_features(z)
    after = model.state_dict()
    assert tuple(before) == tuple(after)
    assert parameter_names == tuple(name for name, _ in model.named_parameters())
    assert all(torch.equal(value, after[key]) for key, value in before.items())

    checkpoint.seek(0)
    restored = triangle_model()
    status = restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    assert not status.missing_keys and not status.unexpected_keys
    assert tuple(restored.state_dict()) == tuple(before)
    with torch.no_grad():
        assert_same_logits(ordinary, restored.decode(z))


def test_final_features_recompose_all_original_heads():
    """Capture the actual last head inputs, including atom/charge feedback."""
    torch.manual_seed(149)
    model = triangle_model()
    z = torch.randn(2, model.config.latent)
    decoded, features = model.decode_with_features(z)
    decoder = model.decoder

    atoms = decoder.atom_head(features.atom_hidden)
    atom_context = decoder.atom_feedback(atoms.softmax(dim=-1))
    charge_input = torch.cat((features.atom_hidden, atom_context), dim=-1)
    charges = decoder.charge_head(charge_input)
    charge_context = decoder.charge_feedback(charges.softmax(dim=-1))
    node_context = decoder.context_norm(features.atom_hidden + atom_context + charge_context)
    bond_input = torch.cat((features.pair_hidden,
                            symmetric_pair_features(node_context)), dim=-1)
    assert torch.equal(features.charge_input, charge_input)
    assert torch.equal(features.node_context, node_context)
    assert torch.equal(features.bond_input, bond_input)
    assert torch.equal(decoded.atoms, atoms)
    assert torch.equal(decoded.charges, charges)
    assert torch.equal(decoded.bonds, decoder.bond_head(bond_input))


def test_frozen_capture_preserves_latent_gradient_and_frozen_weights():
    torch.manual_seed(151)
    model = triangle_model().requires_grad_(False)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    z_plain = torch.randn(2, model.config.latent, requires_grad=True)
    z_captured = z_plain.detach().clone().requires_grad_(True)
    plain = model.decode(z_plain)
    captured, features = model.decode_with_features(z_captured)

    def objective(decoded):
        # Nonuniform channel weights avoid a normalized-distribution sum that
        # can make a severed latent derivative look like an expected zero.
        return (decoded.atoms[..., 1].square().mean()
                + 0.7 * decoded.charges[..., 2].mean()
                + 0.3 * decoded.bonds[:, 0, 1, 1].sum())

    plain_gradient = torch.autograd.grad(objective(plain), z_plain)[0]
    captured_gradient = torch.autograd.grad(
        objective(captured), z_captured, retain_graph=True)[0]
    assert torch.isfinite(plain_gradient).all() and plain_gradient.abs().sum() > 0
    assert torch.equal(plain_gradient, captured_gradient)

    # The exported feature tensors themselves must retain the path to z after
    # VAE freezing; equality of ordinary outputs alone cannot establish that.
    for index, tensor in enumerate((features.atom_hidden, features.pair_hidden,
                                    features.node_context, features.charge_input,
                                    features.bond_input)):
        gradient = torch.autograd.grad(
            tensor[..., 0].square().sum(), z_captured,
            retain_graph=index < 4)[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_triangle_features_keep_shapes_undirected_pairs_and_zero_pair_diagonal():
    torch.manual_seed(157)
    model = triangle_model()
    decoded, features = model.decode_with_features(torch.randn(2, model.config.latent))
    assert features.atom_hidden.shape == (2, 4, 8)
    assert features.pair_hidden.shape == (2, 4, 4, 6)
    assert features.node_context.shape == (2, 4, 8)
    assert features.charge_input.shape == (2, 4, 16)
    assert features.bond_input.shape == (2, 4, 4, 30)
    for pair_tensor in (features.pair_hidden, features.bond_input, decoded.bonds):
        assert torch.equal(pair_tensor, pair_tensor.transpose(1, 2))
    diagonal = torch.arange(4)
    assert torch.equal(features.pair_hidden[:, diagonal, diagonal], torch.zeros(2, 4, 6))
    assert all(torch.isfinite(getattr(features, field)).all() for field in
               ("atom_hidden", "pair_hidden", "node_context", "charge_input", "bond_input"))
