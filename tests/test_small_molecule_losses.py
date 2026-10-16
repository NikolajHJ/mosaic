"""Exact reductions and explicit mappings, independent of a model download."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import MolecularTrunkOutput
from mosaic.small_molecule.losses import (
    DBMolContactLoss,
    DistogramDiagnosticLoss,
    NativeContactLoss,
)
from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard, mix_with_uniform


def _output():
    # Noncontiguous, interleaved ligand/target tokens; reverse contacts differ.
    probabilities = np.tile(
        np.array([0.05, 0.05, 0.4, 0.5], dtype=np.float32), (5, 5, 1)
    )
    for i, j, contact in ((3, 4, 0.4), (3, 0, 0.8), (1, 4, 0.7), (1, 0, 0.6)):
        probabilities[i, j] = (
            contact / 2,
            contact / 2,
            (1 - contact) / 2,
            (1 - contact) / 2,
        )
    return MolecularTrunkOutput(
        distogram_logits=jnp.log(jnp.asarray(probabilities)),
        distogram_bins=jnp.asarray([4.0, 7.0, 9.0, jnp.inf]),
        ligand_token_indices=(3, 1),
        protein_token_indices=(4, 0),
    )


def test_contact_uses_mean_ligand_best_target_and_conservative_native_upper_edges():
    graph = graph_from_hard([0, 1], [1])
    loss, aux = NativeContactLoss()(graph, _output(), key=jax.random.key(0))
    np.testing.assert_allclose(loss, -0.75, rtol=0, atol=1e-6)
    np.testing.assert_allclose(aux["contact_score"], 0.75, rtol=0, atol=1e-6)
    assert aux["conservative_upper_edge_bins"]
    assert not aux["diagnostic_only"]
    # A bin whose upper edge equals the cutoff is excluded by strict '<'.
    edge_loss, _ = NativeContactLoss(cutoff_angstrom=7)(
        graph, _output(), key=jax.random.key(0)
    )
    np.testing.assert_allclose(edge_loss, -0.375, rtol=0, atol=1e-6)


def test_contact_requires_target_instead_of_inventing_ligand_only_score():
    output = _output()
    output = MolecularTrunkOutput(
        output.distogram_logits, output.distogram_bins, (3, 1), ()
    )
    with pytest.raises(ValueError, match="nonempty explicit ligand and protein"):
        NativeContactLoss()(graph_from_hard([0, 1], [0]), output, key=jax.random.key(0))


def test_diagnostic_addresses_explicit_offdiagonal_ligand_pair_only():
    output = _output()
    value, aux = DistogramDiagnosticLoss(bin_index=2)(
        graph_from_hard([0, 1], [0]), output, key=jax.random.key(0)
    )
    np.testing.assert_array_equal(value, output.distogram_logits[3, 1, 2])
    assert aux["diagnostic_only"]
    with pytest.raises(ValueError, match="distinct"):
        DistogramDiagnosticLoss(ligand_pair=(0, 0))
    with pytest.raises(ValueError, match="outside"):
        DistogramDiagnosticLoss(ligand_pair=(0, 2))(
            graph_from_hard([0, 1], [0]), output, key=jax.random.key(0)
        )


def test_contact_raw_derivative_matches_probability_transfer_without_max_tie():
    output = _output()

    def scalar(logits):
        changed = MolecularTrunkOutput(
            logits,
            output.distogram_bins,
            output.ligand_token_indices,
            output.protein_token_indices,
        )
        return NativeContactLoss()(
            graph_from_hard([0, 1], [0]), changed, key=jax.random.key(0)
        )[0]

    direction = (
        jnp.zeros_like(output.distogram_logits).at[3, 0, 0].set(1).at[3, 0, 3].set(-1)
    )
    analytic = jnp.sum(jax.grad(scalar)(output.distogram_logits) * direction)
    epsilon = 1e-3
    difference = (
        scalar(output.distogram_logits + epsilon * direction)
        - scalar(output.distogram_logits - epsilon * direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, difference, rtol=0.002, atol=2e-5)


def test_native_graph_losses_reuse_mosaic_weighted_composition():
    graph = graph_from_hard([0, 1], [0])
    first = DistogramDiagnosticLoss(bin_index=0)
    second = DistogramDiagnosticLoss(bin_index=1)
    value, auxiliary = (0.25 * first + 0.75 * second)(
        graph, _output(), key=jax.random.key(0)
    )
    expected = 0.25 * first(graph, _output(), key=jax.random.key(0))[0]
    expected += 0.75 * second(graph, _output(), key=jax.random.key(0))[0]
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-6)
    assert isinstance(auxiliary, list) and len(auxiliary) == 2


def _dbmol_output():
    output = _output()
    # Pocket-first direction differs deliberately from the reverse sentinels.
    # Rows are native protein IDs (4, 0); columns are ligand IDs (3, 1).
    probabilities = jnp.exp(output.distogram_logits)
    for i, j, contact in ((4, 3, 0.2), (4, 1, 0.7), (0, 3, 0.8), (0, 1, 0.6)):
        probabilities = probabilities.at[i, j].set(
            jnp.asarray([contact / 2] * 2 + [(1 - contact) / 2] * 2)
        )
    return MolecularTrunkOutput(
        jnp.log(probabilities),
        output.distogram_bins,
        output.ligand_token_indices,
        output.protein_token_indices,
    )


def test_dbmol_matches_manual_pocket_mean_then_anchor_and_explicit_native_maps():
    graph = graph_from_hard([0, 1], [0])
    objective = DBMolContactLoss(pocket_token_indices=(4, 0))
    value, aux = objective(graph, _dbmol_output(), key=jax.random.key(0))
    # Mean over pocket residues, selecting each residue's strongest ligand atom.
    pocket = np.mean(-np.log(np.asarray([0.7, 0.8]) + 1e-8))
    anchor = -np.log(0.8 + 1e-8)
    np.testing.assert_allclose(aux["pocket_contact_loss"], pocket, rtol=0, atol=1e-6)
    np.testing.assert_allclose(aux["anchor_contact_loss"], anchor, rtol=0, atol=1e-6)
    np.testing.assert_allclose(value, 0.5 * (pocket + anchor), rtol=0, atol=1e-6)
    assert aux["pocket_token_count"] == 2
    assert aux["conservative_upper_edge_bins"]
    assert not aux["diagnostic_only"]
    # Reordering fixed pocket IDs preserves the loss; a subset changes coverage.
    reordered = DBMolContactLoss(pocket_token_indices=(0, 4))(
        graph, _dbmol_output(), key=jax.random.key(0)
    )[0]
    subset = DBMolContactLoss(pocket_token_indices=(4,))(
        graph, _dbmol_output(), key=jax.random.key(0)
    )[0]
    np.testing.assert_allclose(reordered, value, rtol=0, atol=1e-6)
    np.testing.assert_allclose(subset, -np.log(0.7 + 1e-8), rtol=0, atol=1e-6)


def test_dbmol_conservative_cutoff_and_additive_log_epsilon_are_explicit():
    graph = graph_from_hard([0, 1], [0])
    output = _dbmol_output()
    value, _ = DBMolContactLoss(pocket_token_indices=(4, 0), cutoff_angstrom=7)(
        graph, output, key=jax.random.key(0)
    )
    expected = 0.5 * (
        np.mean(-np.log(np.asarray([0.35, 0.4]) + 1e-8)) - np.log(0.4 + 1e-8)
    )
    np.testing.assert_allclose(value, expected, rtol=0, atol=1e-6)
    empty_bins, _ = DBMolContactLoss(
        pocket_token_indices=(4,), cutoff_angstrom=2, log_epsilon=1e-4
    )(graph, output, key=jax.random.key(0))
    np.testing.assert_allclose(empty_bins, -np.log(1e-4), rtol=0, atol=1e-6)


def test_dbmol_loss_has_independent_element_and_bond_gradients_without_min_ties():
    # Synthetic differentiable feature channels isolate the reduction plumbing.
    # This does not establish native Protenix or chemistry gradient validity.
    base = mix_with_uniform(graph_from_hard([0, 1], [0]), 0.15)
    output = _dbmol_output()
    objective = DBMolContactLoss(pocket_token_indices=(4, 0))

    def scalar(graph):
        x_shift = (graph.X - base.X)[0, 0] - (graph.X - base.X)[0, 1]
        e_shift = (graph.E_upper - base.E_upper)[0, 0] - (graph.E_upper - base.E_upper)[
            0, 1
        ]
        logits = output.distogram_logits.at[4, 1, 0].add(x_shift)
        logits = logits.at[0, 3, 0].add(e_shift)
        changed = MolecularTrunkOutput(
            logits,
            output.distogram_bins,
            output.ligand_token_indices,
            output.protein_token_indices,
        )
        return objective(graph, changed, key=jax.random.key(0))[0]

    gradients = jax.grad(scalar)(base)
    epsilon = 1e-3
    for leaf_name in ("X", "E_upper"):
        leaf = getattr(base, leaf_name)
        direction = jnp.zeros_like(leaf).at[0, 0].set(1).at[0, 1].set(-1)
        analytic = jnp.sum(getattr(gradients, leaf_name) * direction)
        if leaf_name == "X":
            plus = RelaxedGraph(base.X + epsilon * direction, base.E_upper)
            minus = RelaxedGraph(base.X - epsilon * direction, base.E_upper)
        else:
            plus = RelaxedGraph(base.X, base.E_upper + epsilon * direction)
            minus = RelaxedGraph(base.X, base.E_upper - epsilon * direction)
        finite_difference = (scalar(plus) - scalar(minus)) / (2 * epsilon)
        assert np.isfinite(np.asarray(getattr(gradients, leaf_name))).all()
        assert float(analytic) < 0  # Increasing either selected contact lowers loss.
        np.testing.assert_allclose(analytic, finite_difference, rtol=0.003, atol=2e-5)


@pytest.mark.parametrize("pocket", [(), (4, 4), (-1,), (True,), (1.0,), ("4",), [4]])
def test_dbmol_rejects_malformed_or_mutable_pocket_ids(pocket):
    with pytest.raises(ValueError, match="Pocket token IDs"):
        DBMolContactLoss(pocket_token_indices=pocket)


def test_dbmol_rejects_nonprotein_pockets_and_invalid_native_token_maps():
    graph = graph_from_hard([0, 1], [0])
    output = _dbmol_output()
    for pocket in ((3,), (2,), (5,)):
        with pytest.raises(ValueError, match="protein map"):
            DBMolContactLoss(pocket_token_indices=pocket)(
                graph, output, key=jax.random.key(0)
            )
    malformed = MolecularTrunkOutput(
        output.distogram_logits, output.distogram_bins, (3, 1), (4, 7)
    )
    with pytest.raises(ValueError, match="outside the native distogram"):
        DBMolContactLoss(pocket_token_indices=(7,))(
            graph, malformed, key=jax.random.key(0)
        )
    overlap = MolecularTrunkOutput(
        output.distogram_logits, output.distogram_bins, (3, 1), (4, 3)
    )
    with pytest.raises(ValueError, match="overlap"):
        DBMolContactLoss(pocket_token_indices=(3,))(
            graph, overlap, key=jax.random.key(0)
        )


@pytest.mark.parametrize("epsilon", [0, -1e-8, 1, np.inf, np.nan])
def test_dbmol_rejects_invalid_log_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        DBMolContactLoss(pocket_token_indices=(4,), log_epsilon=epsilon)
