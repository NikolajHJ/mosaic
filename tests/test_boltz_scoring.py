"""Facade/direct-call equivalence without native Boltz assets; HPC execution only."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.boltz.adapter import adapt_graph
from latentmol.boltz.contracts import BOLTZ_COMMIT, PreparedComplex
from latentmol.boltz.scoring import BoltzGraphBranch, boltz_branch_manifest
from latentmol.boltz.provenance import context_digest
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def prepared_fixture():
    spec = GraphSpec(max_nodes=2)
    bonds = torch.tensor([[[0, 1], [1, 0]]])
    seed = HardGraphBatch(torch.tensor([[1, 2]]), torch.tensor([[2, 2]]), bonds)
    pool = torch.eye(2)[None]
    features = {
        "ref_element": F.one_hot(torch.tensor([[6, 7]]), 128),
        "ref_charge": torch.zeros(1, 2), "token_pad_mask": torch.ones(1, 2),
        "atom_pad_mask": torch.ones(1, 2), "affinity_token_mask": torch.ones(1, 2),
        "mol_type": torch.full((1, 2), 3), "type_bonds": bonds * 2,
        "token_bonds": bonds.float()[..., None], "atom_to_token": pool,
        "token_to_rep_atom": pool, "ref_pos": torch.zeros(1, 2, 3),
    }
    return PreparedComplex(features, torch.zeros(1, 1, 2, 3), torch.arange(2),
                           torch.arange(2), seed, spec, {"boltz_commit": BOLTZ_COMMIT})


class MockNative(nn.Module):
    """Exercise native-input plumbing, never claim this fixture predicts affinity."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.recycling_steps = 0

    def forward(self, features, types, coordinates):
        assert coordinates.shape == (1, 1, 2, 3)
        value = (features["ref_element"] * torch.arange(128.)).sum((1, 2))
        value = value + features["ref_charge"].sum(1) + features["token_bonds"].sum((1, 2, 3))
        value = value + (types * torch.arange(7.)).sum((1, 2, 3))
        return {"affinity_pred_value": self.weight * value,
                "affinity_probability_binary": (value / 100).sigmoid()}


def test_facade_matches_direct_native_inputs_outputs_and_input_gradients():
    prepared, scorer = prepared_fixture(), MockNative()
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256="a" * 64, recycling_steps=0)
    branch = BoltzGraphBranch(scorer, prepared, manifest, allow_lossy_charge=True)
    assert manifest.capabilities.gradient_mode == "partial"
    decoded = DecoderOutput(torch.randn(1, 2, 10, requires_grad=True),
                            torch.randn(1, 2, 5, requires_grad=True),
                            torch.randn(1, 2, 2, 5, requires_grad=True))
    graph = graph_from_decoder(decoded, prepared.spec, fixed_active=prepared.active)
    before = context_digest(prepared)
    features, types = adapt_graph(graph, prepared)
    direct = scorer(features, types, prepared.coordinates)
    wrapped = branch(graph)
    assert set(direct) == set(wrapped)
    for name in direct:
        torch.testing.assert_close(wrapped[name], direct[name], rtol=0, atol=0)
    inputs = (decoded.atoms, decoded.charges, decoded.bonds)
    direct_grad = torch.autograd.grad(direct["affinity_pred_value"].sum(), inputs, retain_graph=True)
    wrapped_grad = torch.autograd.grad(wrapped["affinity_pred_value"].sum(), inputs)
    for left, right in zip(direct_grad, wrapped_grad):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        assert left.isfinite().all() and left.norm() > 0
    assert context_digest(prepared) == before
    assert scorer.weight.grad is None and not scorer.weight.requires_grad


def test_soft_charge_distribution_is_explicitly_lossy_even_when_its_mean_is_native():
    prepared, scorer = prepared_fixture(), MockNative()
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256="a" * 64, recycling_steps=0)
    branch = BoltzGraphBranch(scorer, prepared, manifest)
    hard = prepared.seed.to_relaxed(prepared.spec)
    mixture = prepared.seed.to_relaxed(prepared.spec)
    mixture.charges[0, 0] = torch.tensor([0., 0.5, 0., 0.5, 0.])
    assert branch.assess(hard).status == "exact"
    assert branch.assess(mixture).status == "lossy"
    # Different categorical identities have the same expected charge: the loss
    # is real, not a label assigned merely because the tensors are floating.
    left, _ = adapt_graph(hard, prepared)
    right, _ = adapt_graph(mixture, prepared)
    torch.testing.assert_close(left["ref_charge"], right["ref_charge"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="allow_lossy_charge"):
        branch(mixture)
    enabled = BoltzGraphBranch(scorer, prepared, manifest, allow_lossy_charge=True)
    for name, value in enabled(hard).items():
        torch.testing.assert_close(value, enabled(mixture)[name], rtol=0, atol=0)


def test_facade_rejects_changed_context_configuration_soft_occupancy_or_training():
    prepared, scorer = prepared_fixture(), MockNative()
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256="a" * 64, recycling_steps=0)
    with pytest.raises(ValueError, match="manifest differs"):
        BoltzGraphBranch(scorer, prepared, replace(manifest, context_sha256="0" * 64))
    with pytest.raises(ValueError, match="manifest differs"):
        BoltzGraphBranch(scorer, prepared, replace(manifest, settings=(("recycling_steps", "2"),)))
    branch = BoltzGraphBranch(scorer, prepared, manifest)
    graph = prepared.seed.to_relaxed(prepared.spec)
    graph.atoms[0, 0] = 0
    graph.atoms[0, 0, 0] = graph.atoms[0, 0, 1] = 0.5
    assert branch.assess(graph).status == "unsupported"
    with pytest.raises(ValueError, match="occupancy"):
        branch(graph)
    branch.train()
    with pytest.raises(ValueError, match="eval mode"):
        branch(prepared.seed.to_relaxed(prepared.spec))
    scorer.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        BoltzGraphBranch(scorer, prepared, manifest)


def test_context_mutation_is_detected_at_explicit_run_boundary():
    prepared, scorer = prepared_fixture(), MockNative()
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256="a" * 64, recycling_steps=0)
    branch = BoltzGraphBranch(scorer, prepared, manifest)
    assert branch.verify_context_unchanged() == manifest.context_sha256
    prepared.coordinates.add_(0.5)
    with pytest.raises(ValueError, match="context changed"):
        branch.verify_context_unchanged()


def test_bond_mask_allows_active_pair_but_rejects_non_none_diagonal():
    prepared, scorer = prepared_fixture(), MockNative()
    manifest = boltz_branch_manifest(prepared, adapter_source_sha256="a" * 64, recycling_steps=0)
    branch = BoltzGraphBranch(scorer, prepared, manifest)
    graph = prepared.seed.to_relaxed(prepared.spec)
    # The fixture already has a SINGLE bond between its two active slots.
    assert branch.assess(graph).status == "exact"
    graph.bonds[0, 0, 0] = 0
    graph.bonds[0, 0, 0, 1] = 1
    support = branch.assess(graph)
    assert support.status == "unsupported"
    assert "NONE on absent/diagonal pairs" in support.reasons[0]
