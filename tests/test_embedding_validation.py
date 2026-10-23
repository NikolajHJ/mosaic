"""Allocated-HPC fixtures for native injection validation and score diagnostics.

Small deterministic projections are engineering fixtures, never substitutes for
real frozen Boltz native parity or useful molecular-gradient evidence.
"""

import json

import pytest
import torch
from torch import nn

from latentmol.boltz.adapter import adapt_graph
from latentmol.boltz.embedding_bridge import HiddenEmbeddingBridge, NativeProjectionBank
from latentmol.boltz.embedding_injection import bond_type_embedding
from latentmol.boltz.embedding_validation import (
    graph_embedding_predictions, native_gate, score_learned_bridge)
from latentmol.models import GraphAutoencoder, ModelConfig
from test_boltz_adapter import prepared_fixture


class NativeFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedder = nn.Module()
        self.input_embedder.atom_encoder = nn.Module()
        encoder = self.input_embedder.atom_encoder
        encoder.use_no_atom_char = True
        encoder.use_atom_backbone_feat = encoder.use_residue_feats_atoms = False
        encoder.embed_atom_features = nn.Linear(132, 3)
        self.token_bonds = nn.Linear(1, 3, bias=False)
        self.token_bonds_type = nn.Embedding(7, 3)
        self.bond_type_feature = True
        self.confidence_prediction = False


class ScorerFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = NativeFixture()
        self.reference_offset = 0.
        self.eval().requires_grad_(False)

    def atom(self, features):
        inputs = torch.cat((features["ref_pos"], features["ref_charge"][..., None],
                            features["ref_element"].float()), -1)
        return self.model.input_embedder.atom_encoder.embed_atom_features(inputs)

    @staticmethod
    def reduce(atom, connectivity, typed):
        # Smooth nonlinear chemistry response tests both values and chain rule.
        return {"dbmol_total_loss": atom.square().mean() + connectivity.square().mean()
                + .3 * typed.square().mean() + atom.mean() * typed.mean()}

    def forward(self, features, bonds, coordinates):
        return self.reduce(self.atom(features), self.model.token_bonds(features["token_bonds"]),
                           bond_type_embedding(self.model, bonds))

    def discrete_reference(self, features, coordinates):
        result = self.reduce(self.atom(features), self.model.token_bonds(features["token_bonds"]),
                             self.model.token_bonds_type(features["type_bonds"]))
        return {key: value + self.reference_offset for key, value in result.items()}


def frozen_models():
    torch.manual_seed(801)
    prepared = prepared_fixture()
    scorer = ScorerFixture()
    bank = NativeProjectionBank.from_models({"affinity": scorer.model}).eval()
    vae = GraphAutoencoder(prepared.spec, ModelConfig(hidden=8, latent=4, layers=1,
        decoder_variant="triangle_refine", global_hidden=8, pair_hidden=6,
        attention_heads=2, refinement_rounds=1, triangle_hidden=4)).eval().requires_grad_(False)
    return prepared, scorer, bank, vae


def test_independent_graph_projection_values_and_gradients():
    prepared, _, bank, _ = frozen_models()
    graph = prepared.seed.to_relaxed(prepared.spec)
    for value in (graph.atoms, graph.charges, graph.bonds):
        value.requires_grad_(True)
    independent = graph_embedding_predictions(graph, prepared, bank)
    expected = bank.targets(prepared.seed, prepared.spec).values
    assert set(independent) == set(expected)
    for key in expected:
        torch.testing.assert_close(independent[key], expected[key], rtol=0, atol=1e-7)
    gradient = torch.autograd.grad(sum(value.sum() for value in independent.values()),
                                   (graph.atoms, graph.charges, graph.bonds))
    assert all(bool(value.isfinite().all()) for value in gradient)
    assert all(value[:, ~prepared.active].count_nonzero() == 0 for value in gradient)


def test_native_oracle_gate_persists_complete_fresh_evidence(tmp_path):
    prepared, scorer, bank, vae = frozen_models()
    path = tmp_path / "native-gate.json"
    report = native_gate(scorer, prepared, bank, vae, path)
    saved = json.loads(path.read_text())
    assert saved == report
    assert report["passed"] and report["status"] == "passed_native_oracle_only"
    assert report["existing_graph_vs_native"]["passed"]
    assert report["exact_injection_vs_native"]["passed"]
    assert report["projection_gradient_parity"]["passed"]
    assert report["derivative_injection_receipt"]["all_channels_consumed"]
    assert report["frozen_modules_unchanged"] and report["context_unchanged"]
    assert len(report["finite_difference_observations"]) == 4
    assert report["finite_differences_are_diagnostic"]
    assert report["calls"]["backward_attempted"] == 2
    assert not report["learned_bridge_validated"]
    with pytest.raises(FileExistsError):
        native_gate(scorer, prepared, bank, vae, path)


def test_native_oracle_failure_is_written_before_raising(tmp_path):
    prepared, scorer, bank, vae = frozen_models()
    scorer.reference_offset = 10.
    path = tmp_path / "native-gate.json"
    with pytest.raises(ValueError, match="native scalar parity"):
        native_gate(scorer, prepared, bank, vae, path)
    saved = json.loads(path.read_text())
    assert not saved["passed"] and saved["status"] == "failed_not_passed"
    assert saved["stage"] == "native_seed_parity"
    assert not saved["existing_graph_vs_native"]["passed"]
    assert saved["frozen_modules_unchanged"] and saved["context_unchanged"]


def test_learned_score_finishes_backward_inside_context_and_classifies_hard_graph():
    prepared, scorer, bank, vae = frozen_models()
    z = torch.randn(1, vae.config.latent)
    _, features = vae.decode_with_features(z)
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=features.atom_hidden.shape[-1],
        pair_input_dim=features.bond_input.shape[-1], charge_input_dim=features.charge_input.shape[-1],
        hidden_dim=8).eval().requires_grad_(False)
    result = score_learned_bridge(scorer, prepared, bridge, vae, z, require_gradient=True)
    assert result["gradient"].shape == z.shape and not result["gradient"].requires_grad
    assert bool(result["gradient"].isfinite().all()) and result["gradient_norm"] > 0
    assert result["injection_receipt"]["all_channels_consumed"]
    assert "strict_teacher_valid" in result["hard_classification"]
    assert result["canonical_smiles"] == result["hard_classification"]["smiles"]
    assert result["graph_policy_difference"]["conditioned_occupied_slots"] == int(prepared.active.sum())
    assert "hard_fixed_context_scores" in result
    assert result["invalid_hard_scores_retained_as_tensor_diagnostics"]
    assert result["frozen_modules_unchanged"] and result["context_unchanged"]
    assert not result["learned_bridge_validated"]
    # A subsequent ordinary call proves the injected hooks were removed.
    f, b = adapt_graph(prepared.seed.to_relaxed(prepared.spec), prepared)
    old = scorer(f, b, prepared.coordinates)
    reference = scorer.discrete_reference(prepared.features, prepared.coordinates)
    torch.testing.assert_close(old["dbmol_total_loss"], reference["dbmol_total_loss"], rtol=0, atol=0)
