"""Small local tests of invariance, target alignment and live loss gradients."""
from dataclasses import asdict, replace
import itertools

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph
from latentmol.training.losses import LossConfig, reconstruction_losses
from latentmol.training.matching import (
    MatchingConfig, align_targets, matched_reconstruction_losses, numpy_assignment,
)


def graph(smiles, spec):
    row = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.tensor(row[key], dtype=torch.long)[None]
                            for key in ("atom_types", "charges", "bond_types")))


def permute(value, order):
    return HardGraphBatch(value.atoms[:, order], value.charges[:, order], value.bonds[:, order][:, :, order])


@pytest.mark.parametrize("size", [1, 3, 5])
def test_numpy_assignment_matches_brute_force(size):
    cost = np.random.default_rng(17).normal(size=(size, size))
    answer = numpy_assignment(cost)
    best = min(sum(cost[i, j] for i, j in enumerate(p)) for p in itertools.permutations(range(size)))
    assert cost[np.arange(size), answer].sum() == pytest.approx(best)
    with pytest.raises(ValueError):
        numpy_assignment(np.array([[np.nan]]))


@pytest.mark.parametrize("structural", [False, True])
def test_invariant_encoder_hard_and_soft_graphs_including_scattered_padding(structural):
    torch.manual_seed(17)
    spec = GraphSpec(max_nodes=8)
    model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=2, encoder_pooling="invariant",
        encoder_aggregation="sum" if structural else "mean", encoder_degree_features=structural))
    hard = graph("CC(=O)N", spec)
    order = torch.tensor([6, 2, 0, 7, 4, 3, 5, 1])
    for value in [hard.to_relaxed(spec), RelaxedGraph(torch.rand(1, 8, 10).softmax(-1),
            torch.rand(1, 8, 5).softmax(-1), torch.rand(1, 8, 8, 5).softmax(-1))]:
        rearranged = RelaxedGraph(value.atoms[:, order], value.charges[:, order], value.bonds[:, order][:, :, order])
        mu, logvar = model.encode(value)
        moved_mu, moved_logvar = model.encode(rearranged)
        torch.testing.assert_close(mu, moved_mu, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(logvar, moved_logvar, rtol=1e-5, atol=1e-6)
    assert model.encoder.position is None
    assert model.encoder.pool[1].in_features == 2*model.config.hidden+1


def test_mean_collapses_carbon_topologies_but_structural_encoder_distinguishes_them():
    """Same element/count is insufficient: preserve chain/branch/ring information."""
    spec = GraphSpec(max_nodes=8)
    graphs = [graph(s, spec).to_relaxed(spec) for s in ("CCCC", "CC(C)C", "C1CCC1")]
    for aggregation, degrees in (("mean", False), ("sum", False), ("sum", True)):
        torch.manual_seed(31)
        model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=4,
            encoder_pooling="invariant", encoder_aggregation=aggregation, encoder_degree_features=degrees))
        codes = torch.cat([model.encode(g)[0] for g in graphs])
        distances = torch.pdist(codes)
        if aggregation == "mean":
            assert float(distances.max()) < 1e-6
        else:
            assert float(distances.min()) > 1e-4
        if degrees:
            (codes-torch.randn_like(codes)).square().mean().backward()
            grad = model.encoder.degree_projection.weight.grad
            assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 0


def test_legacy_model_config_and_state_still_load_strictly():
    spec = GraphSpec(max_nodes=8)
    original = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=1))
    old_config = {key: value for key, value in asdict(original.config).items() if not key.startswith("encoder_")}
    loaded = GraphAutoencoder(spec, ModelConfig(**old_config))
    loaded.load_state_dict(original.state_dict(), strict=True)
    z = torch.randn(2, 8)
    assert torch.equal(original.decode(z).atoms, loaded.decode(z).atoms)
    assert "encoder.position" in loaded.state_dict()


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "[NH3+]CC(=O)[O-]"])
def test_exact_permuted_graphs_recover_the_same_joint_loss(smiles):
    spec = GraphSpec(max_nodes=8)
    target = graph(smiles, spec)
    logits = DecoderOutput(*(F.one_hot(getattr(target, key), classes).float()*12
        for key, classes in (("atoms", 10), ("charges", 5), ("bonds", 5))))
    shuffled = permute(target, torch.tensor([7, 3, 0, 5, 1, 6, 2, 4]))
    expected = reconstruction_losses(logits, target, LossConfig())["reconstruction"]
    actual, receipt = matched_reconstruction_losses(logits, shuffled, LossConfig(),
        MatchingConfig(assignment_backend="numpy"))
    torch.testing.assert_close(actual["reconstruction"], expected, rtol=1e-4, atol=1e-6)
    assert receipt["isomorphism_witnesses"] == 1


def test_matching_never_worsens_actual_batch_loss_and_preserves_gradients():
    torch.manual_seed(9)
    spec = GraphSpec(max_nodes=8)
    first, second = graph("CCO", spec), graph("CCCCN", spec)
    target = HardGraphBatch(*(torch.cat([getattr(first, k), getattr(second, k)]) for k in ("atoms", "charges", "bonds")))
    decoded = DecoderOutput(torch.randn(2, 8, 10, requires_grad=True),
        torch.randn(2, 8, 5, requires_grad=True), torch.randn(2, 8, 8, 5, requires_grad=True))
    config = LossConfig()
    before = reconstruction_losses(decoded, target, config)["reconstruction"]
    losses, receipt = matched_reconstruction_losses(decoded, target, config, MatchingConfig(assignment_backend="numpy"))
    assert float(losses["reconstruction"]) <= float(before)+1e-6
    assert receipt["objective_after_without_size"] <= receipt["objective_before_without_size"]+1e-10
    losses["reconstruction"].backward()
    assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in (decoded.atoms, decoded.charges, decoded.bonds))


@pytest.mark.parametrize("smiles", ["c1ccccc1", "[NH3+]CC(=O)[O-]"])
def test_graph_order_real_loss_gradients_match_with_permuted_predictions(smiles):
    """One graph including internal padding keeps every head's CE and derivative."""
    from latentmol.chemistry_continuation import reconstruction_chunk, reconstruction_denominators

    spec = GraphSpec(max_nodes=8)
    target = graph(smiles, spec)
    order = torch.tensor([7, 3, 0, 5, 1, 6, 2, 4])
    moved = permute(target, order)
    torch.manual_seed(14)
    logits = []
    for key, classes in (("atoms", 10), ("charges", 5), ("bonds", 5)):
        labels = F.one_hot(getattr(moved, key), classes).float()
        logits.append((3 * labels + torch.rand_like(labels)).requires_grad_())
    predicted = DecoderOutput(*logits)
    config = LossConfig(atom_order="graph")
    expected = reconstruction_losses(predicted, moved, replace(config, atom_order="ordered"))
    observed = reconstruction_losses(predicted, target, config)
    for key in expected:
        torch.testing.assert_close(observed[key], expected[key])
    chunk = reconstruction_chunk(predicted, target, config, reconstruction_denominators(target, config))
    torch.testing.assert_close(chunk, expected["reconstruction"])
    reference_gradients = torch.autograd.grad(expected["reconstruction"], logits)
    for actual, reference in zip(torch.autograd.grad(chunk, logits), reference_gradients):
        torch.testing.assert_close(actual, reference)
        assert bool(torch.isfinite(actual).all()) and float(actual.abs().sum()) > 0


def test_exact_fallback_resolves_symmetric_bonds_without_accepting_a_different_topology():
    # A six-cycle and two triangles have the same local node and bond inventory.
    # Force the short bounded search to stop; fallback must verify full topology.
    spec = GraphSpec(max_nodes=8)
    target = graph("C1CCCCC1", spec)
    order = torch.tensor([7, 3, 0, 5, 1, 6, 2, 4])
    shuffled = permute(target, order)
    config = LossConfig(atom_order="graph")
    for isomorphic in (True, False):
        hard = HardGraphBatch(shuffled.atoms, shuffled.charges, shuffled.bonds.clone())
        if not isomorphic:
            active = torch.nonzero(hard.atoms[0]).flatten().tolist()
            hard.bonds.zero_()
            for x, y, z in (active[:3], active[3:]):
                for i, j in ((x, y), (y, z), (z, x)):
                    hard.bonds[0, i, j] = hard.bonds[0, j, i] = 1
        logits = DecoderOutput(*(F.one_hot(getattr(hard, key), n).float()*3
                                for key, n in (("atoms", 10), ("charges", 5), ("bonds", 5))))
        aligned, report = align_targets(logits, target, config,
            MatchingConfig(iterations=0, isomorphism_states=1), preserve_exact=True)
        assert report["isomorphism_budget_exhaustions"] == 1
        assert report["isomorphism_witnesses"] == int(isomorphic)
        assert torch.equal(aligned.bonds, hard.bonds) == isomorphic


def test_inexact_graph_matching_across_accumulated_chunks_uses_logical_batch_weights():
    from latentmol.chemistry_continuation import reconstruction_chunk, reconstruction_denominators

    spec = GraphSpec(max_nodes=8)
    graphs = [graph(s, spec) for s in ("CC", "CCOCC", "C1CCCCC1")]
    heads = ("atoms", "charges", "bonds")
    targets = HardGraphBatch(*(torch.cat([getattr(g, key) for g in graphs]) for key in heads))
    rng = torch.Generator().manual_seed(53)
    scores = [torch.randn(shape, generator=rng, requires_grad=True)
              for shape in ((3, 8, 10), (3, 8, 5), (3, 8, 8, 5))]
    cfg = LossConfig(atom_order="graph")
    expected = reconstruction_losses(DecoderOutput(*scores), targets, cfg)["reconstruction"]
    denominators = reconstruction_denominators(targets, cfg)
    actual = sum(reconstruction_chunk(DecoderOutput(*(h[i:i+1] for h in scores)),
                                      graph_i, cfg, denominators)
                 for i, graph_i in enumerate(graphs))
    torch.testing.assert_close(actual, expected)
    for given, wanted in zip(torch.autograd.grad(actual, scores), torch.autograd.grad(expected, scores)):
        torch.testing.assert_close(given, wanted)
