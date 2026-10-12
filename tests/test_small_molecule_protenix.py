"""Resident/cheap-path contract spies; native checkpoint gates are separate.

No test double here can establish real Tiny gradient fidelity. These numerical
tests are authored for DTU HPC and have not been run locally.
"""

import sys
import types
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule import features as feature_module
from mosaic.small_molecule import protenix as adapter
from mosaic.small_molecule.features import MODEL_ALIAS
from mosaic.small_molecule.losses import DistogramDiagnosticLoss
from mosaic.small_molecule.native_hook import GeometryMaskedAtomInput
from mosaic.small_molecule.optimizers import evaluate_graph
from mosaic.small_molecule.state import mix_with_uniform
from test_small_molecule_features import synthetic_context


class _Trunk(eqx.Module):
    s: jax.Array
    z: jax.Array


class _Model(eqx.Module):
    def distogram_head(self, z):
        return z
    def sample_structures(self, **kwargs):
        raise AssertionError("Stage 0 must never call structure sampling")
    def confidence_metrics(self, **kwargs):
        raise AssertionError("Stage 0 must never call confidence")


def _fake_trunk(model, features, key):
    atoms = jnp.sum(features["ref_element"] * jnp.arange(128), axis=-1)
    token_count = features["restype"].shape[0]
    tokens = jnp.zeros(token_count).at[features["atom_to_token_idx"]].add(atoms)
    identity = tokens[:, None] + tokens[None, :]
    z = identity[..., None] + features["token_bonds"][..., None] * jnp.arange(64)
    return None, _Trunk(tokens[:, None], z)


def test_cheap_graph_forward_keeps_native_outputs_and_never_calls_structure_or_confidence(monkeypatch):
    monkeypatch.setattr(adapter, "_native_trunk", _fake_trunk)
    context, graph = synthetic_context()
    model = adapter.FrozenProtenixTiny(_Model(), _Model(), {"model_alias": "test-double"})
    graph = mix_with_uniform(graph, 0.2)
    output = model.forward_trunk(context=context, graph=graph, key=jax.random.key(0))
    assert output.distogram_logits.shape == (5, 5, 64)
    assert output.ligand_token_indices == (3, 1, 4)
    assert output.protein_token_indices == (0, 2)
    assert not hasattr(output, "pae") and not hasattr(output, "structure_coordinates")
    assert np.isfinite(output.distogram_logits).all()
    assert np.isinf(output.distogram_bins[-1])
    expected = np.concatenate((np.linspace(2.3125, 21.6875, 63), [np.inf]))
    np.testing.assert_allclose(output.distogram_bins, expected, rtol=0, atol=2e-6)
    # Generated outputs participate in the strict evaluator; the intentional
    # infinite bin edge is metadata and must not be mixed into finite aux.
    objective = model.build_loss(context=context, loss=DistogramDiagnosticLoss(bin_index=1))
    evaluation = evaluate_graph(objective, graph, jax.random.key(0))
    assert np.isfinite(evaluation.value)
    assert np.isfinite(evaluation.gradients.X).all()
    assert np.isfinite(evaluation.gradients.E_upper).all()
    assert "distogram_logits" in evaluation.aux
    assert "trunk_s" in evaluation.aux and "trunk_z" in evaluation.aux
    assert "distogram_bins" not in evaluation.aux


def test_model_spy_a_b_a_recomputes_graph_inputs_instead_of_caching_embeddings(monkeypatch):
    monkeypatch.setattr(adapter, "_native_trunk", _fake_trunk)
    context, graph = synthetic_context()
    graph = mix_with_uniform(graph, 0.3)
    model = adapter.FrozenProtenixTiny(_Model(), _Model(), {"model_alias": "test-double"})
    moved = eqx.tree_at(lambda g: g.X, graph, graph.X.at[0, 0].add(-0.03).at[0, 1].add(0.03))
    key = jax.random.key(3)
    a0 = model.forward_trunk(context=context, graph=graph, key=key).distogram_logits
    b = model.forward_trunk(context=context, graph=moved, key=key).distogram_logits
    a1 = model.forward_trunk(context=context, graph=graph, key=key).distogram_logits
    assert not np.array_equal(a0, b)
    np.testing.assert_array_equal(a0, a1)


def test_composed_native_losses_keep_nested_mosaic_auxiliary_tree(monkeypatch):
    monkeypatch.setattr(adapter, "_native_trunk", _fake_trunk)
    context, graph = synthetic_context()
    model = adapter.FrozenProtenixTiny(_Model(), _Model(), {})
    combination = 0.4 * DistogramDiagnosticLoss(bin_index=0) + 0.6 * DistogramDiagnosticLoss(bin_index=1)
    evaluation = evaluate_graph(model.build_loss(context=context, loss=combination), mix_with_uniform(graph, 0.2), jax.random.key(0))
    assert isinstance(evaluation.aux["objective"], list)
    assert len(evaluation.aux["objective"]) == 2
    assert evaluation.aux["distogram_logits"].shape == (5, 5, 64)


@pytest.mark.parametrize("cycles", [0, 2, -1, True, 1.0])
def test_unsupported_recycling_is_rejected_before_native_work(cycles):
    context, graph = synthetic_context()
    model = adapter.FrozenProtenixTiny(_Model(), _Model(), {})
    with pytest.raises(ValueError, match="must be 1"):
        model.forward_trunk(context=context, graph=graph, key=jax.random.key(0), recycling_steps=cycles)


def test_offline_loader_fails_before_backend_import_if_a_pinned_asset_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_module, "verify_native_source", lambda: {})
    imported = []
    import builtins
    original_import = builtins.__import__
    def tracked_import(name, *args, **kwargs):
        imported.append(name)
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", tracked_import)
    with pytest.raises(FileNotFoundError, match="Pre-stage"):
        adapter.FrozenProtenixTiny.load(cache_dir=tmp_path)
    assert "protenix.backend" not in imported


class _MSA(eqx.Module):
    training: bool = True


class _OriginalEncoder(eqx.Module):
    has_coords: bool = False
    def __call__(self, input_feature_dict):
        value = input_feature_dict["ref_element"]
        return value, value, value, value


class _Input(eqx.Module):
    atom_attention_encoder: Any


class _Loadable(eqx.Module):
    input_embedder: _Input
    msa_module: _MSA
    weight: jax.Array


def test_loader_is_resident_and_hook_shares_weights_without_mutating_native_model(monkeypatch, tmp_path):
    calls = []
    original = _Loadable(_Input(_OriginalEncoder()), _MSA(), jnp.array([1.0, 2.0]))
    backend = types.ModuleType("protenix.backend")
    backend.load_model = lambda path: calls.append(path) or original
    parent = types.ModuleType("protenix")
    parent.backend = backend
    parent.__path__ = []
    monkeypatch.setitem(sys.modules, "protenix", parent)
    monkeypatch.setitem(sys.modules, "protenix.backend", backend)
    monkeypatch.setattr(adapter, "bootstrap_native_assets", lambda *a, **k: {})
    resident = adapter.FrozenProtenixTiny.load(cache_dir=tmp_path)
    assert calls == [str(tmp_path / MODEL_ALIAS)]
    assert original.msa_module.training  # eval-mode change is functional
    assert not resident.native_model.msa_module.training
    assert not resident.model.msa_module.training
    assert resident.native_model.weight is resident.model.weight
    assert isinstance(resident.native_model.input_embedder.atom_attention_encoder, _OriginalEncoder)
    assert isinstance(resident.model.input_embedder.atom_attention_encoder, GeometryMaskedAtomInput)
    assert resident.model.input_embedder.atom_attention_encoder.native is resident.native_model.input_embedder.atom_attention_encoder
    # Building objectives does not trigger a second loader or weight allocation.
    context, _ = synthetic_context()
    resident.build_loss(context=context, loss=DistogramDiagnosticLoss())
    resident.build_loss(context=context, loss=DistogramDiagnosticLoss())
    assert len(calls) == 1


def test_protein_only_hook_preserves_the_exact_native_encoder_path():
    encoder = _OriginalEncoder()
    features = {"ref_element": jnp.array([[0.1, 0.9]])}
    original = encoder(features)
    hooked = GeometryMaskedAtomInput(encoder)(features)
    for actual, expected in zip(hooked, original):
        np.testing.assert_array_equal(actual, expected)


def test_native_hard_baseline_and_unsupported_pae_boundary_are_explicit(monkeypatch):
    monkeypatch.setattr(adapter, "_native_trunk", _fake_trunk)
    context, graph = synthetic_context()
    model = adapter.FrozenProtenixTiny(_Model(), _Model(), {})
    native = model.forward_native_trunk(context=context, key=jax.random.key(0))
    assert np.isfinite(native.distogram_logits).all()
    prerequisites = adapter.ligand_pae_gate_prerequisites(context)
    assert not prerequisites["supported"] and prerequisites["valid_ligand_frame_count"] == 0
    with pytest.raises(NotImplementedError, match="no valid ligand reference frames"):
        model.confidence_output(context=context, graph=graph, key=jax.random.key(0))
