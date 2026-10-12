"""Exact reductions and explicit mappings, independent of a model download."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import MolecularTrunkOutput
from mosaic.small_molecule.losses import DistogramDiagnosticLoss, NativeContactLoss
from mosaic.small_molecule.state import graph_from_hard


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
