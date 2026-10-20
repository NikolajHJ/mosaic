"""Allocated research-probe checks against independent categorical enumeration."""

import importlib.util
import itertools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "compatibility_probe",
    Path(__file__).resolve().parents[1] / "scripts/chemistry_compatibility_probe.py",
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Chemistry probe tests require allocated DTU HPC")


def fixture():
    # Hard argmax is the valid Cl-C-C chain; probabilities remain uncertain.
    atoms = jnp.zeros((3, 6), jnp.float32).at[0, 5].set(2).at[1:, 0].set(2)
    bonds = (
        jnp.zeros((3, 5), jnp.float32).at[0, 1].set(2).at[1, 0].set(2).at[2, 1].set(2)
    )
    return atoms, bonds


def softmax(values):
    values = np.exp(
        np.asarray(values, np.float64) - np.asarray(values).max(-1, keepdims=True)
    )
    return values / values.sum(-1, keepdims=True)


def test_three_atom_loss_matches_exhaustive_global_edge_assignments():
    atoms, bonds = fixture()
    pa, pe = softmax(atoms), softmax(bonds)
    masses = np.zeros(3)
    for labels in itertools.product(range(5), repeat=3):
        probability = np.prod([pe[e, category] for e, category in enumerate(labels)])
        orders = [0, 1, 2, 3, 1]
        a, b, c = [orders[label] for label in labels]
        for node, valence in enumerate((a + b, a + c, b + c)):
            masses[node] += probability * sum(
                pa[node, element]
                for element, cap in enumerate(PROBE.CAPS)
                if valence <= cap
            )
    np.testing.assert_allclose(
        float(PROBE.loss(atoms, bonds)), -np.log(masses).mean(), rtol=2e-6, atol=2e-6
    )


@pytest.mark.parametrize("field", [0, 1])
def test_live_atom_and_unordered_bond_derivatives_match_independent_primal(field):
    atoms, bonds = fixture()
    directions = [
        jnp.zeros_like(atoms).at[0, 0].set(1).at[0, 5].set(-1),
        jnp.zeros_like(bonds).at[1, 0].set(-1).at[1, 1].set(1),
    ]
    gradients = jax.grad(PROBE.loss, argnums=(0, 1))(atoms, bonds)
    analytic = float(jnp.sum(gradients[field] * directions[field]))
    assert abs(analytic) > 1e-4
    for epsilon in (0.003, 0.001):
        endpoints = []
        for sign in (-1, 1):
            point = [atoms, bonds]
            point[field] = point[field] + sign * epsilon * directions[field]
            endpoints.append(float(PROBE.loss(*point)))
        np.testing.assert_allclose(
            (endpoints[1] - endpoints[0]) / (2 * epsilon),
            analytic,
            rtol=0.01,
            atol=2e-4,
        )


def test_node_permutation_and_independent_row_logit_shifts_preserve_loss():
    atoms, bonds = fixture()
    permutation, pair_index = (2, 0, 1), {(0, 1): 0, (0, 2): 1, (1, 2): 2}
    permuted_bonds = jnp.stack(
        [
            bonds[pair_index[tuple(sorted((permutation[i], permutation[j])))]]
            for i, j in pair_index
        ]
    )
    baseline = float(PROBE.loss(atoms, bonds))
    np.testing.assert_allclose(
        float(PROBE.loss(atoms[jnp.asarray(permutation)], permuted_bonds)),
        baseline,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        float(
            PROBE.loss(
                atoms + jnp.asarray([3, -2, 1])[:, None],
                bonds + jnp.asarray([-2, 5, 1])[:, None],
            )
        ),
        baseline,
        atol=2e-6,
    )


def test_probability_weighted_caps_have_atom_gradient_where_hard_caps_have_none():
    atoms, bonds = fixture()
    labels = np.asarray([5, 0, 0], np.int32)
    soft = jax.grad(PROBE.loss)(atoms, bonds)
    hard = jax.grad(lambda a: PROBE.hard_cap_loss(a, bonds, labels))(atoms)
    assert (
        np.isfinite(np.asarray(soft)).all() and np.linalg.norm(np.asarray(soft)) > 1e-4
    )
    np.testing.assert_array_equal(hard, np.zeros_like(hard))
    changed = atoms.at[0, 5].add(0.1)
    assert np.argmax(atoms[0]) == np.argmax(changed[0])
    assert float(PROBE.loss(atoms, bonds)) != float(PROBE.loss(changed, bonds))


def test_disconnected_and_aromatic_counterexamples_expose_proxy_limits():
    atoms = jnp.full((2, 6), -30.0, jnp.float32).at[:, 5].set(30)
    none = jnp.full((1, 5), -30.0, jnp.float32).at[:, 0].set(30)
    single = none.at[0, 0].set(-30).at[0, 1].set(30)
    aromatic = none.at[0, 0].set(-30).at[0, 4].set(30)
    assert (
        abs(float(PROBE.loss(atoms, none))) < 1e-6
    )  # Disconnected atoms satisfy local caps.
    np.testing.assert_allclose(
        float(PROBE.loss(atoms, aromatic)), float(PROBE.loss(atoms, single)), atol=1e-6
    )
    gradients = jax.grad(PROBE.loss, argnums=(0, 1))(atoms, aromatic)
    assert all(
        np.isfinite(np.asarray(g)).all() for g in gradients
    )  # Unreachable DP states stay safe.


def test_high_cap_sulfur_can_reduce_risk_without_repairing_dense_bonds():
    bonds = jnp.full((3, 5), -5.0, jnp.float32).at[:, 1].set(5)
    chlorine = jnp.full((3, 6), -5.0, jnp.float32).at[:, 5].set(5)
    sulfur = chlorine.at[:, 5].set(-5).at[:, 3].set(5)
    assert float(PROBE.loss(sulfur, bonds)) < float(PROBE.loss(chlorine, bonds))


def test_valid_margin_stops_sharpening_after_margin_but_uncertainty_has_gradient():
    atoms, bonds = fixture()
    a, e = np.asarray([5, 0, 0]), np.asarray([1, 0, 1])

    def objective(x, y):
        return PROBE.valid_margin_loss(x, y, a, e)

    assert float(objective(atoms, bonds)) == 0.0
    assert all(
        np.count_nonzero(np.asarray(g)) == 0
        for g in jax.grad(objective, argnums=(0, 1))(atoms, bonds)
    )
    uncertain = jnp.zeros_like(atoms), jnp.zeros_like(bonds)
    assert float(objective(*uncertain)) > 0
    assert all(
        np.linalg.norm(np.asarray(g)) > 0
        for g in jax.grad(objective, argnums=(0, 1))(*uncertain)
    )


@pytest.mark.parametrize(
    "labels", [np.asarray([5.0, 0.0, 0.0]), np.asarray([6, 0, 0]), np.asarray([5, 0])]
)
def test_frozen_labels_are_validated_on_host(labels):
    with pytest.raises(ValueError):
        PROBE.validate_labels(labels, n_atoms=3)
