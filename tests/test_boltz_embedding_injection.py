"""Scoped ligand embedding injection contracts, numerically executed on HPC.

Toy modules contain native-shaped trained linear projections only. These tests
verify injection algebra, alignment, gradients and cleanup, never native Boltz
scientific fidelity or useful molecular optimization.
"""

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from latentmol.boltz.embedding_bridge import NativeProjectionBank
from latentmol.boltz.embedding_injection import (
    any_injected_requires_grad, bond_type_embedding, inject_native_embeddings,
    native_embedding_models,
)
from test_boltz_adapter import prepared_fixture
from test_boltz_embedding_bridge import native_fixture


def fixture(*, dual=True):
    prepared = prepared_fixture()
    scorer = nn.Module()
    if dual:
        scorer.affinity = nn.Module()
        scorer.affinity.model = native_fixture(no_chars=False)
        scorer.structure = native_fixture(confidence=True, no_chars=False)
    else:
        scorer.model = native_fixture(no_chars=False)
    scorer.eval().requires_grad_(False)
    bank = NativeProjectionBank.from_models(native_embedding_models(scorer))
    predictions = bank.targets(prepared.seed, prepared.spec).values
    raw = torch.cat((prepared.features["ref_pos"],
                     prepared.features["ref_charge"][..., None],
                     prepared.features["ref_element"].float(), torch.randn(1, 5, 256)), -1)
    native_types = F.one_hot(prepared.features["type_bonds"], 7).float()
    return scorer, prepared, bank, predictions, raw, native_types


def outputs(scorer, prepared, raw, native_types, *, repeats=2, keyword=False,
            connectivity=None):
    values = {}
    if connectivity is None:
        connectivity = (1 - native_types[..., 0])[..., None]
    for name, model in native_embedding_models(scorer).items():
        atom = model.input_embedder.atom_encoder.embed_atom_features
        for repeat in range(repeats):
            values[f"{name}.atom_{repeat}"] = atom(input=raw) if keyword else atom(raw)
        values[f"{name}.trunk_connectivity"] = (model.token_bonds(input=connectivity)
                                                if keyword else model.token_bonds(connectivity))
        values[f"{name}.trunk_type"] = bond_type_embedding(model, native_types)
        if name == "structure" and getattr(model, "confidence_prediction", False):
            confidence = model.confidence_module
            values[f"{name}.confidence_connectivity"] = confidence.token_bonds(connectivity)
            values[f"{name}.confidence_type"] = bond_type_embedding(confidence, native_types)
    return values


def assert_clean(scorer):
    assert not any_injected_requires_grad()
    for model in native_embedding_models(scorer).values():
        for module in model.modules():
            assert not module._forward_hooks


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
def test_native_oracle_projection_values_repeated_channels_and_cleanup(dual, keyword):
    scorer, prepared, _, predictions, raw, native_types = fixture(dual=dual)
    before = {key: value.clone() for key, value in scorer.state_dict().items()}
    baseline = outputs(scorer, prepared, raw, native_types, keyword=keyword)
    with inject_native_embeddings(scorer, prepared, predictions) as receipt:
        injected = outputs(scorer, prepared, raw, native_types, keyword=keyword)
    for key in baseline:
        # Atom subtraction/readdition is not bitwise associative in FP32.
        torch.testing.assert_close(injected[key], baseline[key], rtol=2e-6, atol=2e-6)
    assert receipt.record()["all_channels_consumed"]
    assert receipt.visited["affinity.atom_element"] == 2
    assert receipt.visited["affinity.atom_charge"] == 2
    assert receipt.visited["affinity.trunk_type"] == 1
    if dual:
        assert receipt.visited["structure.confidence_type"] == 1
    assert_clean(scorer)
    restored = outputs(scorer, prepared, raw, native_types, keyword=keyword)
    assert all(torch.equal(restored[key], baseline[key]) for key in baseline)
    assert all(torch.equal(value, before[key]) for key, value in scorer.state_dict().items())


def test_only_aligned_ligand_rows_change_and_native_diagonal_is_retained():
    scorer, prepared, _, predictions, raw, native_types = fixture()
    baseline = outputs(scorer, prepared, raw, native_types)
    active = prepared.active
    pair = active[:, None] & active[None, :]
    modified = {}
    for key, value in predictions.items():
        mask = active if key.endswith(("atom_element", "atom_charge")) else pair
        modified[key] = value + 0.75 * mask[None, ..., None]
    with inject_native_embeddings(scorer, prepared, modified):
        injected = outputs(scorer, prepared, raw, native_types)
    ai = prepared.slot_to_atom[active]
    ti = prepared.slot_to_token[active]
    atom_mask = torch.zeros(raw.shape[1], dtype=torch.bool)
    atom_mask[ai] = True
    pair_mask = torch.zeros(4, 4, dtype=torch.bool)
    pair_mask[ti[:, None], ti[None, :]] = True
    pair_mask.fill_diagonal_(False)
    for key in baseline:
        if ".atom_" in key:
            assert torch.equal(injected[key][:, ~atom_mask], baseline[key][:, ~atom_mask])
            torch.testing.assert_close(injected[key][:, atom_mask] - baseline[key][:, atom_mask],
                                       torch.full_like(injected[key][:, atom_mask], 1.5),
                                       rtol=2e-6, atol=2e-6)
        else:
            assert torch.equal(injected[key][:, ~pair_mask], baseline[key][:, ~pair_mask])
            torch.testing.assert_close(injected[key][:, pair_mask] - baseline[key][:, pair_mask],
                                       torch.full_like(injected[key][:, pair_mask], 0.75),
                                       rtol=2e-6, atol=2e-6)


def test_affinity_checkpoint_confidence_flags_do_not_add_unreachable_injection_channels():
    scorer, prepared, _, _, raw, native_types = fixture()
    scorer.affinity.model = native_fixture(confidence=True, no_chars=False)
    bank = NativeProjectionBank.from_models(native_embedding_models(scorer), confidence_models={"structure"})
    predictions = bank.targets(prepared.seed, prepared.spec).values
    assert not any(key.startswith("affinity.confidence_") for key in predictions)
    with inject_native_embeddings(scorer, prepared, predictions) as receipt:
        outputs(scorer, prepared, raw, native_types)
    assert receipt.record()["all_channels_consumed"]
    assert scorer.affinity.model.confidence_prediction
    assert_clean(scorer)


def differentiable_oracle(bank, prepared, raw, native_types):
    """Independent native linear algebra, retaining the same input derivatives."""
    active = prepared.active
    slots = active.nonzero().flatten()
    ai = prepared.slot_to_atom[active]
    ti = prepared.slot_to_token[active]
    nodes = prepared.spec.max_nodes
    predictions = {}
    for descriptor in bank.descriptors:
        projection = bank.projection(descriptor.key)
        if descriptor.kind == "atom_element":
            chemical = raw[..., 4:132] @ projection
            predictions[descriptor.key] = chemical.new_zeros(1, nodes, descriptor.width).index_copy(
                1, slots, chemical.index_select(1, ai))
        elif descriptor.kind == "atom_charge":
            chemical = raw[..., 3:4] @ projection
            predictions[descriptor.key] = chemical.new_zeros(1, nodes, descriptor.width).index_copy(
                1, slots, chemical.index_select(1, ai))
        else:
            if descriptor.kind == "connectivity":
                chemical = (1 - native_types[..., 0:1]) @ projection[1:2]
            else:
                chemical = native_types @ projection
            selected = chemical.index_select(1, ti).index_select(2, ti)
            flat = (slots[:, None] * nodes + slots[None, :]).flatten()
            predictions[descriptor.key] = chemical.new_zeros(1, nodes * nodes, descriptor.width).index_copy(
                1, flat, selected.reshape(1, -1, descriptor.width)).reshape(1, nodes, nodes, descriptor.width)
    return predictions


def test_native_oracle_input_gradient_matches_independent_untouched_projection_path():
    scorer, prepared, bank, _, raw, _ = fixture()
    raw = raw.requires_grad_(True)
    logits = torch.randn(1, 4, 4, 7, requires_grad=True)
    symmetric = ((logits + logits.transpose(1, 2)) * 0.5).softmax(-1)
    diagonal = torch.eye(4, dtype=torch.bool)[None, :, :, None]
    none = F.one_hot(torch.zeros(1, 4, 4, dtype=torch.long), 7).float()
    native_types = torch.where(diagonal, none, symmetric)
    reference = outputs(scorer, prepared, raw, native_types)
    reference_loss = sum(value.square().sum() for value in reference.values())
    reference_gradients = torch.autograd.grad(reference_loss, (raw, logits), retain_graph=True)
    predictions = differentiable_oracle(bank, prepared, raw, native_types)
    with inject_native_embeddings(scorer, prepared, predictions):
        injected = outputs(scorer, prepared, raw, native_types)
        loss = sum(value.square().sum() for value in injected.values())
        gradients = torch.autograd.grad(loss, (raw, logits))
    for key in reference:
        torch.testing.assert_close(injected[key], reference[key], rtol=3e-6, atol=3e-6)
    for gradient, expected in zip(gradients, reference_gradients, strict=True):
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        torch.testing.assert_close(gradient, expected, rtol=2e-5, atol=2e-5)
    assert_clean(scorer)


def test_bridge_leaf_gradients_cover_every_channel_but_exclude_absent_and_pair_diagonal():
    scorer, prepared, _, predictions, raw, native_types = fixture()
    predictions = {key: value.clone().requires_grad_(True) for key, value in predictions.items()}
    with inject_native_embeddings(scorer, prepared, predictions):
        assert any_injected_requires_grad()
        actual = outputs(scorer, prepared, raw, native_types)
        loss = sum(value.sum() for value in actual.values())
        gradients = torch.autograd.grad(loss, tuple(predictions.values()))
    active = prepared.active
    pair = active[:, None] & active[None, :]
    pair.fill_diagonal_(False)
    for (key, _), gradient in zip(predictions.items(), gradients, strict=True):
        mask = active if key.endswith(("atom_element", "atom_charge")) else pair
        assert torch.isfinite(gradient).all() and gradient[:, mask].norm() > 0
        assert gradient[:, ~mask].count_nonzero() == 0
    assert_clean(scorer)


def test_context_survives_nonreentrant_projection_recomputation_during_backward():
    scorer, prepared, _, predictions, raw, native_types = fixture(dual=False)
    predictions = {key: value.clone().requires_grad_(True) for key, value in predictions.items()}
    raw = raw.requires_grad_(True)

    def forward(value):
        projected = outputs(scorer, prepared, value, native_types)
        return sum(tensor.square().sum() for tensor in projected.values())

    with inject_native_embeddings(scorer, prepared, predictions) as receipt:
        value = checkpoint(forward, raw, use_reentrant=False)
        value.backward()
        assert any_injected_requires_grad()
    assert receipt.visited["affinity.atom_element"] >= 2
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for value in predictions.values())
    assert_clean(scorer)


def test_missing_channel_consumption_fails_after_restoring_every_hook():
    scorer, prepared, _, predictions, _, _ = fixture()
    with pytest.raises(ValueError, match="not consumed"):
        with inject_native_embeddings(scorer, prepared, predictions):
            pass
    assert_clean(scorer)


def test_original_exception_preserved_and_following_native_forward_is_unchanged():
    scorer, prepared, _, predictions, raw, native_types = fixture()
    baseline = outputs(scorer, prepared, raw, native_types)
    with pytest.raises(RuntimeError, match="deliberate original failure"):
        with inject_native_embeddings(scorer, prepared, predictions):
            raise RuntimeError("deliberate original failure")
    assert_clean(scorer)
    restored = outputs(scorer, prepared, raw, native_types)
    assert all(torch.equal(restored[key], baseline[key]) for key in baseline)


def test_nested_context_rejected_without_damaging_outer_context():
    scorer, prepared, _, predictions, raw, native_types = fixture()
    other, other_prepared, _, other_predictions, _, _ = fixture(dual=False)
    with inject_native_embeddings(scorer, prepared, predictions):
        with pytest.raises(ValueError, match="Nested"):
            with inject_native_embeddings(other, other_prepared, other_predictions):
                pass
        outputs(scorer, prepared, raw, native_types)
    assert_clean(scorer)
    assert_clean(other)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "integer", "nonfinite", "absent", "asymmetric"])
def test_injection_rejects_projection_key_shape_mask_and_value_errors(mutation):
    scorer, prepared, _, predictions, _, _ = fixture()
    predictions = {key: value.clone() for key, value in predictions.items()}
    atom_key, pair_key = "affinity.atom_element", "affinity.trunk_type"
    if mutation == "missing":
        predictions.pop(atom_key)
    elif mutation == "extra":
        predictions["invented"] = predictions[atom_key]
    elif mutation == "shape":
        predictions[atom_key] = predictions[atom_key][:, :3]
    elif mutation == "integer":
        predictions[atom_key] = predictions[atom_key].long()
    elif mutation == "nonfinite":
        predictions[atom_key][0, 0, 0] = float("nan")
    elif mutation == "absent":
        predictions[atom_key][0, 1, 0] = 1
    else:
        predictions[pair_key][0, 0, 2, 0] += 1
    with pytest.raises(ValueError):
        with inject_native_embeddings(scorer, prepared, predictions):
            pass
    assert_clean(scorer)


@pytest.mark.parametrize("channel", ["connectivity", "type", "atom_shape", "unknown_owner"])
def test_runtime_native_invocation_guards_restore_context(channel):
    scorer, prepared, _, predictions, raw, native_types = fixture(dual=False)
    model = scorer.model
    with pytest.raises(ValueError):
        with inject_native_embeddings(scorer, prepared, predictions):
            if channel == "connectivity":
                malformed = (1 - native_types[..., 0:1]).clone()
                malformed[0, 0, 0] = 1
                model.token_bonds(malformed)
            elif channel == "type":
                malformed = native_types.clone()
                malformed[0, 0, 0] = F.one_hot(torch.tensor(2), 7).float()
                bond_type_embedding(model, malformed)
            elif channel == "atom_shape":
                model.input_embedder.atom_encoder.embed_atom_features(raw[:, :4])
            else:
                bond_type_embedding(native_fixture(), native_types)
    assert_clean(scorer)


def test_model_resolution_rejects_unknown_or_aliased_checkpoint_owners():
    with pytest.raises(ValueError, match="requires"):
        native_embedding_models(nn.Module())
    scorer = nn.Module()
    scorer.affinity = nn.Module()
    scorer.affinity.model = native_fixture()
    scorer.structure = scorer.affinity.model
    with pytest.raises(ValueError, match="distinct"):
        native_embedding_models(scorer)
