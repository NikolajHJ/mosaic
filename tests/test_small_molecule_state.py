"""Weight-free graph contracts; numerical execution is deferred to DTU HPC."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.state import (
    BOND_STATES,
    ELEMENTS,
    RelaxedGraph,
    bond_existence,
    full_bonds,
    graph_diagnostics,
    graph_from_hard,
    initialize_graph,
    mix_with_uniform,
    pair_indices,
    validate_graph,
)


def test_contract_vocabulary_and_exactly_two_differentiated_leaves():
    graph = graph_from_hard([0, 2], [1])
    assert ELEMENTS == ("C", "N", "O", "S", "F", "Cl")
    assert BOND_STATES == ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    leaves = jax.tree_util.tree_leaves(graph)
    assert len(leaves) == 2
    assert leaves[0] is graph.X
    assert leaves[1] is graph.E_upper
    assert graph.n_atoms == 2


def test_pair_order_preserves_identifiable_unordered_edges_and_diagonal():
    graph = graph_from_hard([0, 1, 2, 3], [0, 1, 2, 3, 4, 1])
    i, j = pair_indices(4)
    assert list(zip(np.asarray(i), np.asarray(j))) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]
    bonds = np.asarray(full_bonds(graph))
    assignments = bonds.argmax(axis=-1)
    np.testing.assert_array_equal(
        assignments,
        [[0, 0, 1, 2], [0, 0, 3, 4], [1, 3, 0, 1], [2, 4, 1, 0]],
    )
    np.testing.assert_array_equal(bonds, bonds.swapaxes(0, 1))
    np.testing.assert_array_equal(
        bonds[np.arange(4), np.arange(4)], [[1, 0, 0, 0, 0]] * 4
    )
    np.testing.assert_array_equal(
        np.diag(np.asarray(bond_existence(graph))), np.zeros(4)
    )


def test_reconstruction_is_jittable_and_accumulates_both_directed_gradients():
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.3)
    np.testing.assert_array_equal(jax.jit(full_bonds)(graph), full_bonds(graph))
    # The two directed entries have deliberately different coefficients. A
    # detached mirror or a first-direction-only adapter cannot produce -5.
    weights = jnp.array([[0.0, 2.0], [3.0, 0.0]])
    gradients = jax.grad(lambda g: jnp.sum(weights * bond_existence(g)))(graph)
    np.testing.assert_array_equal(gradients.E_upper, [[-5, 0, 0, 0, 0]])
    np.testing.assert_array_equal(gradients.X, np.zeros((2, 6)))


def test_present_bond_order_exchange_is_invisible_to_existence():
    graph = mix_with_uniform(graph_from_hard([0, 1, 2], [1, 2, 4]), 0.4)
    changed = RelaxedGraph(
        graph.X, graph.E_upper.at[1, 1].add(0.03).at[1, 4].add(-0.03)
    )
    validate_graph(changed)
    np.testing.assert_array_equal(bond_existence(graph), bond_existence(changed))
    assert not np.array_equal(full_bonds(graph), full_bonds(changed))


def test_uniform_mix_is_interior_and_retains_fixed_slots():
    hard = graph_from_hard([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4] * 3)
    interior = mix_with_uniform(hard, 0.2)
    validate_graph(interior)
    assert interior.X.shape == hard.X.shape == (6, 6)
    assert interior.E_upper.shape == hard.E_upper.shape == (15, 5)
    assert np.all(np.asarray(interior.X) > 0)
    assert np.all(np.asarray(interior.E_upper) > 0)
    np.testing.assert_array_equal(mix_with_uniform(hard, 0).X, hard.X)
    uniform = mix_with_uniform(hard, 1)
    np.testing.assert_allclose(uniform.X, np.full((6, 6), 1 / 6))
    np.testing.assert_allclose(uniform.E_upper, np.full((15, 5), 1 / 5))


@pytest.mark.parametrize("mixing", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_mix_is_rejected(mixing):
    with pytest.raises(ValueError, match="mixing"):
        mix_with_uniform(graph_from_hard([0, 1], [0]), mixing)


@pytest.mark.parametrize("n_atoms", [0, -1, 2.5, True])
def test_invalid_atom_count_is_rejected(n_atoms):
    with pytest.raises((ValueError, TypeError)):
        pair_indices(n_atoms)


@pytest.mark.parametrize(
    "elements,bonds,error",
    [
        ([0, 1], [], ValueError),
        ([0, 6], [1], ValueError),
        ([0, 1], [5], ValueError),
        ([0.0, 1.0], [1], TypeError),
        ([0, 1], [True], TypeError),
        ([[0, 1]], [0], ValueError),
    ],
)
def test_hard_assignments_reject_shape_dtype_or_unsupported_categories(
    elements, bonds, error
):
    with pytest.raises(error):
        graph_from_hard(elements, bonds)


@pytest.mark.parametrize(
    "graph,error",
    [
        (RelaxedGraph(jnp.ones((2, 5)), jnp.ones((1, 5)) / 5), ValueError),
        (RelaxedGraph(jnp.ones((2, 6)) / 6, jnp.ones((2, 5)) / 5), ValueError),
        (
            RelaxedGraph(jnp.ones((2, 6), dtype=jnp.int32), jnp.ones((1, 5)) / 5),
            TypeError,
        ),
        (RelaxedGraph(jnp.full((2, 6), jnp.nan), jnp.ones((1, 5)) / 5), ValueError),
        (RelaxedGraph(jnp.ones((2, 6)), jnp.ones((1, 5)) / 5), ValueError),
        (
            RelaxedGraph(
                jnp.array([[-0.1, 0.22, 0.22, 0.22, 0.22, 0.22]] * 2),
                jnp.ones((1, 5)) / 5,
            ),
            ValueError,
        ),
    ],
)
def test_validation_rejects_malformed_graphs(graph, error):
    with pytest.raises(error):
        validate_graph(graph)


def test_validation_proposal_mode_checks_finiteness_but_allows_off_simplex():
    proposal = RelaxedGraph(jnp.full((2, 6), -2.0), jnp.full((1, 5), 3.0))
    validate_graph(proposal, require_simplex=False)
    with pytest.raises(ValueError):
        validate_graph(proposal)
    with pytest.raises(ValueError, match="atol"):
        validate_graph(proposal, require_simplex=False, atol=np.nan)


def test_initializer_baseline_has_requested_degree_and_carbon_mass():
    graph = initialize_graph(
        5,
        jax.random.key(3),
        carbon_probability=0.8,
        expected_degree=2.0,
        atom_noise=0,
        bond_noise=0,
    )
    validate_graph(graph)
    np.testing.assert_allclose(graph.X[:, 0], 0.8, atol=1e-7)
    np.testing.assert_allclose(graph.X[:, 1:], 0.04, atol=1e-7)
    np.testing.assert_allclose(graph.E_upper[:, 0], 0.5, atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(bond_existence(graph)).sum(axis=-1), 2, atol=1e-6
    )
    assert graph_diagnostics(graph)["expected_degree"] == pytest.approx(2)


def test_initializer_independent_upper_pair_noise_reproducible_and_projected():
    settings = dict(
        carbon_probability=0.8, expected_degree=1.5, atom_noise=0.15, bond_noise=0.15
    )
    graph = initialize_graph(5, jax.random.key(9), **settings)
    replay = initialize_graph(5, jax.random.key(9), **settings)
    other = initialize_graph(5, jax.random.key(10), **settings)
    validate_graph(graph)
    np.testing.assert_array_equal(graph.X, replay.X)
    np.testing.assert_array_equal(graph.E_upper, replay.E_upper)
    assert not np.array_equal(graph.E_upper, other.E_upper)
    assert len(np.unique(np.asarray(graph.E_upper), axis=0)) == 10
    np.testing.assert_array_equal(
        full_bonds(graph), np.asarray(full_bonds(graph)).swapaxes(0, 1)
    )


@pytest.mark.parametrize(
    "settings",
    [
        {"carbon_probability": 1.1},
        {"expected_degree": 3.1},
        {"expected_degree": -1},
        {"atom_noise": -1},
        {"bond_noise": np.inf},
        {"dtype": jnp.int32},
    ],
)
def test_initializer_rejects_invalid_configuration(settings):
    with pytest.raises((ValueError, TypeError)):
        initialize_graph(4, jax.random.key(0), **settings)


def test_single_atom_has_no_pairs_no_self_bonds_and_defined_diagnostics():
    graph = initialize_graph(
        1, jax.random.key(0), expected_degree=0, atom_noise=0, bond_noise=0
    )
    hard = graph_from_hard([5], [])
    validate_graph(hard)
    assert graph.E_upper.shape == (0, 5)
    np.testing.assert_array_equal(bond_existence(graph), [[0]])
    diagnostics = graph_diagnostics(graph)
    assert diagnostics["n_atoms"] == 1
    assert diagnostics["bond_entropy"] == 0
    assert diagnostics["expected_degree"] == 0
