"""Native projection and hidden bridge contracts; run numerically on DTU HPC."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.boltz.embedding_bridge import (
    BridgeTargets, HiddenEmbeddingBridge, NativeProjectionBank,
)
from latentmol.representation import GraphSpec, HardGraphBatch


def assert_permutation_equivariant(actual, expected):
    # Reordering GEMM rows can choose a different summation kernel. Eight dtype
    # epsilons allow its rounding, while meaningful feature changes still fail.
    epsilon = torch.finfo(actual.dtype).eps
    torch.testing.assert_close(actual, expected, rtol=8 * epsilon, atol=8 * epsilon)


def native_fixture(*, confidence=False, typed=True, no_chars=True, width=8):
    """Only projection-bearing native attributes, never a mock score model."""
    model = nn.Module()
    model.input_embedder = nn.Module()
    encoder = model.input_embedder.atom_encoder = nn.Module()
    encoder.use_no_atom_char = no_chars
    encoder.use_atom_backbone_feat = encoder.use_residue_feats_atoms = False
    encoder.embed_atom_features = nn.Linear(132 if no_chars else 388, width)
    model.token_bonds = nn.Linear(1, width, bias=False)
    model.bond_type_feature = typed
    if typed:
        model.token_bonds_type = nn.Embedding(7, width)
    model.confidence_prediction = confidence
    if confidence:
        module = model.confidence_module = nn.Module()
        module.add_z_input_to_z = True
        module.token_bonds = nn.Linear(1, width, bias=False)
        module.bond_type_feature = typed
        if typed:
            module.token_bonds_type = nn.Embedding(7, width)
    return model.eval().requires_grad_(False)


def hard_fixture():
    spec = GraphSpec(max_nodes=4)
    hard = HardGraphBatch(torch.tensor([[1, 3, 0, 2]]),
                         torch.tensor([[2, 1, 2, 3]]),
                         torch.zeros(1, 4, 4, dtype=torch.long))
    hard.bonds[0, 0, 1] = hard.bonds[0, 1, 0] = 2
    return spec, hard


def bank_fixture():
    return NativeProjectionBank.from_models({"affinity": native_fixture(),
                                            "structure": native_fixture(confidence=True)})


def test_extraction_has_checkpoint_specific_keys_and_no_native_module_ownership():
    affinity, structure = native_fixture(), native_fixture(confidence=True)
    bank = NativeProjectionBank.from_models({"affinity": affinity, "structure": structure},
                                          checkpoint_sha256={"affinity": "a" * 64, "structure": "b" * 64})
    assert len(bank.dimensions) == 10
    assert "structure.confidence_type" in bank.dimensions
    assert "affinity.confidence_type" not in bank.dimensions
    assert len(list(bank.parameters())) == 0
    assert len(list(bank.children())) == 0
    assert all(key.startswith("projection_") for key in bank.state_dict())
    original = bank.projection("affinity.atom_element").clone()
    with torch.no_grad():
        affinity.input_embedder.atom_encoder.embed_atom_features.weight.fill_(0)
    assert torch.equal(bank.projection("affinity.atom_element"), original)
    assert not torch.equal(original, bank.projection("structure.atom_element"))


@pytest.mark.parametrize("no_chars", [True, False])
def test_atom_targets_reproduce_exact_native_chemical_contribution(no_chars):
    model = native_fixture(no_chars=no_chars)
    bank = NativeProjectionBank.from_models({"affinity": model})
    spec, hard = hard_fixture()
    targets = bank.targets(hard, spec)
    atom = model.input_embedder.atom_encoder.embed_atom_features
    numbers = torch.tensor(spec.atomic_numbers)[hard.atoms]
    charges = torch.tensor(spec.formal_charges)[hard.charges].float()
    features = torch.randn(1, 4, atom.in_features)
    context = features.clone()
    context[..., 3:132] = 0
    features[..., 3] = charges
    features[..., 4:132] = F.one_hot(numbers, 128).float()
    prediction = atom(context) + targets.values["affinity.atom_element"] + targets.values["affinity.atom_charge"]
    torch.testing.assert_close(prediction[targets.atom_mask], atom(features)[targets.atom_mask],
                               rtol=2e-6, atol=2e-6)
    assert targets.values["affinity.atom_element"][0, 2].count_nonzero() == 0
    assert targets.values["affinity.atom_charge"][0, 2].count_nonzero() == 0


def test_bond_targets_exact_native_connectivity_and_none_embedding_on_diagonal():
    model = native_fixture(confidence=True)
    bank = NativeProjectionBank.from_models({"structure": model})
    spec, hard = hard_fixture()
    targets = bank.targets(hard, spec)
    native_types = torch.tensor([0, 2, 3, 4, 5])[hard.bonds]
    for prefix, module in (("trunk", model), ("confidence", model.confidence_module)):
        connectivity = module.token_bonds((hard.bonds != 0).float()[..., None])
        typed = module.token_bonds_type(native_types)
        for name, expected in (("connectivity", connectivity), ("type", typed)):
            expected = expected.masked_fill(~targets.pair_mask[..., None], 0)
            assert torch.equal(targets.values[f"structure.{prefix}_{name}"], expected)
        assert torch.equal(targets.values[f"structure.{prefix}_type"][0, 0, 0],
                           module.token_bonds_type.weight[0])
    assert targets.pair_mask.sum() == 9
    assert targets.bond_present.sum() == 2
    assert all(not value.requires_grad for value in targets.values.values())


def test_disabled_native_bond_channels_are_not_invented():
    model = native_fixture(confidence=True, typed=False)
    model.confidence_module.add_z_input_to_z = False
    bank = NativeProjectionBank.from_models({"structure": model})
    assert set(bank.dimensions) == {"structure.atom_element", "structure.atom_charge",
                                    "structure.trunk_connectivity"}


def test_affinity_stored_confidence_flag_does_not_make_unexecuted_targets_reachable():
    affinity, structure = native_fixture(confidence=True), native_fixture(confidence=True)
    bank = NativeProjectionBank.from_models({"affinity": affinity, "structure": structure})
    assert not any(key.startswith("affinity.confidence_") for key in bank.dimensions)
    assert "structure.confidence_type" in bank.dimensions
    assert affinity.confidence_prediction  # Stored checkpoint flags remain untouched.
    custom = NativeProjectionBank.from_models({"custom": affinity}, confidence_models={"custom"})
    assert "custom.confidence_type" in custom.dimensions
    with pytest.raises(ValueError, match="present"):
        NativeProjectionBank.from_models({"affinity": affinity}, confidence_models={"missing"})
    with pytest.raises(ValueError, match="trained"):
        NativeProjectionBank.from_models({"affinity": native_fixture()}, confidence_models={"affinity"})
    no_confidence = NativeProjectionBank.from_models({"structure": native_fixture()})
    assert not any(key.startswith("structure.confidence_") for key in no_confidence.dimensions)


def test_bank_serialization_roundtrip_uses_tensors_and_plain_metadata_only():
    bank = bank_fixture()
    restored = NativeProjectionBank.from_metadata(bank.metadata(), bank.state_dict())
    assert restored.metadata() == bank.metadata()
    for key in bank.dimensions:
        assert torch.equal(restored.projection(key), bank.projection(key))
    with pytest.raises(ValueError, match="state keys"):
        NativeProjectionBank.from_metadata(bank.metadata(), {})
    metadata = dict(bank.metadata(), version="unknown")
    with pytest.raises(ValueError, match="metadata"):
        NativeProjectionBank.from_metadata(metadata, bank.state_dict())


@pytest.mark.parametrize("mutation", ["training", "unfrozen", "width", "backbone", "typed", "connectivity_bias"])
def test_native_extraction_fails_closed_for_unsupported_models(mutation):
    model = native_fixture()
    if mutation == "training":
        model.train()
    elif mutation == "unfrozen":
        model.input_embedder.atom_encoder.embed_atom_features.weight.requires_grad_(True)
    elif mutation == "width":
        model.input_embedder.atom_encoder.embed_atom_features = nn.Linear(131, 8).requires_grad_(False)
    elif mutation == "backbone":
        model.input_embedder.atom_encoder.use_atom_backbone_feat = True
    elif mutation == "typed":
        model.token_bonds_type = nn.Embedding(6, 8).requires_grad_(False)
    else:
        model.token_bonds = nn.Linear(1, 8).requires_grad_(False)
    with pytest.raises(ValueError):
        NativeProjectionBank.from_models({"affinity": model})


@pytest.mark.parametrize("mutation", ["asymmetric", "diagonal", "absent_bond", "absent_charge", "range", "dtype"])
def test_hard_target_generation_rejects_identity_schema_errors(mutation):
    bank = bank_fixture()
    spec, hard = hard_fixture()
    if mutation == "asymmetric":
        hard.bonds[0, 0, 1] = 1
    elif mutation == "diagonal":
        hard.bonds[0, 0, 0] = 1
    elif mutation == "absent_bond":
        hard.bonds[0, 0, 2] = hard.bonds[0, 2, 0] = 1
    elif mutation == "absent_charge":
        hard.charges[0, 2] = 0
    elif mutation == "range":
        hard.atoms[0, 0] = len(spec.atomic_numbers)
    else:
        hard.atoms = hard.atoms.float()
    with pytest.raises(ValueError):
        bank.targets(hard, spec)


def test_perfect_target_loss_identity_and_none_present_denominators():
    bank = bank_fixture()
    spec, hard = hard_fixture()
    targets = bank.targets(hard, spec)
    loss, metrics = bank.loss(targets.values, targets)
    assert loss.item() == 0
    assert metrics["affinity.atom_element/atom_count"].item() == 3
    assert metrics["affinity.trunk_type/none_count"].item() == 7
    assert metrics["affinity.trunk_type/present_count"].item() == 2
    identity = bank.identity_metrics(targets.values, hard, spec)
    assert all(value.item() == 1 for key, value in identity.items() if key.endswith("accuracy"))
    predictions = dict(targets.values)
    key = "affinity.trunk_type"
    predictions[key] = targets.values[key] + (~targets.bond_present & targets.pair_mask)[..., None] * 2
    # NONE row MSE = 4 and present row MSE = 0, equally weighted => 2.
    changed_loss, changed_metrics = bank.loss(predictions, targets)
    assert changed_loss.item() == 2
    assert changed_metrics[f"{key}/mse"].item() == 2


def test_empty_masks_are_zero_count_exclusions_without_nan():
    bank = bank_fixture()
    spec = GraphSpec(max_nodes=4)
    hard = HardGraphBatch(torch.zeros(2, 4, dtype=torch.long),
                         torch.full((2, 4), spec.formal_charges.index(0), dtype=torch.long),
                         torch.zeros(2, 4, 4, dtype=torch.long))
    targets = bank.targets(hard, spec)
    predictions = {key: torch.randn_like(value, requires_grad=True) for key, value in targets.values.items()}
    loss, metrics = bank.loss(predictions, targets)
    assert torch.isfinite(loss) and loss.item() == 0
    assert all(value.item() == 0 for key, value in metrics.items() if key.endswith("count"))
    loss.backward()
    assert all(value.grad.count_nonzero() == 0 for value in predictions.values())


def test_loss_rejects_trainable_targets_and_projection_key_mismatch():
    bank = bank_fixture()
    spec, hard = hard_fixture()
    targets = bank.targets(hard, spec)
    values = dict(targets.values)
    first = next(iter(values))
    values[first] = values[first].requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        bank.loss(targets.values, replace(targets, values=values))
    with pytest.raises(ValueError, match="keys"):
        bank.loss({}, targets)


def hidden_fixture(*, requires_grad=False):
    return (torch.randn(2, 4, 6, requires_grad=requires_grad),
            torch.randn(2, 4, 4, 9, requires_grad=requires_grad),
            torch.tensor([[True, False, True, True], [False, True, True, False]]),
            torch.randn(2, 4, 12, requires_grad=requires_grad))


def test_hidden_bridge_masks_symmetry_and_slot_permutation_equivariance():
    bank = bank_fixture()
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=6, pair_input_dim=9,
                                   charge_input_dim=12, hidden_dim=16)
    atom, pair, active, charge = hidden_fixture()
    output = bridge(atom, pair, active, charge)
    assert set(output) == set(bank.dimensions)
    for descriptor in bank.descriptors:
        value = output[descriptor.key]
        if descriptor.kind.startswith("atom_"):
            assert value.shape == (2, 4, descriptor.width)
            assert value[~active].count_nonzero() == 0
        else:
            assert value.shape == (2, 4, 4, descriptor.width)
            assert torch.equal(value, value.transpose(1, 2))
            assert value[~(active[:, :, None] & active[:, None, :])].count_nonzero() == 0
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = bridge(atom[:, permutation], pair[:, permutation][:, :, permutation],
                      active[:, permutation], charge[:, permutation])
    for descriptor in bank.descriptors:
        expected = output[descriptor.key][:, permutation]
        if not descriptor.kind.startswith("atom_"):
            expected = expected[:, :, permutation]
        assert_permutation_equivariant(permuted[descriptor.key], expected)


def test_permutation_comparison_accepts_rounding_and_rejects_material_error():
    expected = torch.tensor([1., 0.], dtype=torch.float32)
    rounded = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    assert_permutation_equivariant(rounded, expected)
    with pytest.raises(AssertionError):
        assert_permutation_equivariant(expected + 1e-3, expected)


def test_frozen_bridge_still_supplies_input_gradients_without_training_native_bank():
    bank = bank_fixture()
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=6, pair_input_dim=9,
                                   charge_input_dim=12, hidden_dim=16).eval().requires_grad_(False)
    atom, pair, active, charge = hidden_fixture(requires_grad=True)
    output = bridge(atom, pair, active, charge)
    scalar = sum(value.square().sum() for value in output.values())
    gradients = torch.autograd.grad(scalar, (atom, pair, charge))
    assert all(torch.isfinite(value).all() and value.norm() > 0 for value in gradients)
    assert gradients[0][~active].count_nonzero() == 0
    assert gradients[2][~active].count_nonzero() == 0
    assert gradients[1][~(active[:, :, None] & active[:, None, :])].count_nonzero() == 0
    assert all(value.grad is None and not value.requires_grad for value in bank.buffers())
    assert all(value.grad is None for value in bridge.parameters())
    assert not any("projection_" in key or "input_embedder" in key for key in bridge.state_dict())


def test_only_bridge_parameters_update_on_supervised_training_step():
    bank = bank_fixture()
    spec, hard = hard_fixture()
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=6, pair_input_dim=9,
                                   charge_input_dim=12, hidden_dim=16)
    atom, pair, _, charge = hidden_fixture()
    targets = bank.targets(hard, spec)
    native_before = {key: value.clone() for key, value in bank.state_dict().items()}
    before = {key: value.clone() for key, value in bridge.state_dict().items()}
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3)
    output = bridge(atom[:1], pair[:1], hard.atoms != 0, charge[:1])
    loss, _ = bank.loss(output, targets)
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(value, before[key]) for key, value in bridge.state_dict().items())
    assert all(torch.equal(value, native_before[key]) for key, value in bank.state_dict().items())


@pytest.mark.parametrize("mutation", ["atom", "pair", "active_shape", "active_dtype", "charge"])
def test_bridge_rejects_misaligned_hidden_state_inputs(mutation):
    bridge = HiddenEmbeddingBridge(bank_fixture(), atom_input_dim=6, pair_input_dim=9,
                                   charge_input_dim=12, hidden_dim=16)
    atom, pair, active, charge = hidden_fixture()
    if mutation == "atom":
        atom = atom[..., :5]
    elif mutation == "pair":
        pair = pair[:, :, :3]
    elif mutation == "active_shape":
        active = active[:, :3]
    elif mutation == "active_dtype":
        active = active.float()
    else:
        charge = None
    with pytest.raises(ValueError):
        bridge(atom, pair, active, charge)
