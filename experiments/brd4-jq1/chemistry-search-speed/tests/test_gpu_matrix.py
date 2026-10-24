"""Allocated numerical fixtures for experimental matrix telemetry/local-risk DP."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_matrix
from mosaic.small_molecule.chemistry import loss
from mosaic.small_molecule.chemistry.state import DecoderLogits
from mosaic.small_molecule.chemistry._reference.representation import GraphSpec


TOLERANCE = 2e-5
INDICES = (0, 1, 2, 3, 4)


def graph_fixture():
    spec = GraphSpec(
        max_nodes=4, atomic_numbers=(0, 6, 8, 9), formal_charges=(-1, 0, 1)
    )
    atoms = np.asarray([[1, 1, 2, 0], [1, 1, 1, 1]], dtype=np.int32)
    charges = np.ones_like(atoms)
    bonds = np.zeros((2, 4, 4), dtype=np.int32)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    for i in range(3):
        bonds[1, i, i + 1] = bonds[1, i + 1, i] = 1
    return spec, atoms, charges, bonds


def risk_fixture(n=6):
    rng = np.random.default_rng(230025)
    logits = jnp.asarray(rng.normal(size=(2, n, n, 5)), dtype=jnp.float32)
    log_bonds = jax.nn.log_softmax(logits, -1)
    active = jnp.asarray([[True] * n, [True] * (n - 2) + [False] * 2])
    caps = jnp.asarray([[1] * n, [2] * (n - 2) + [-1] * 2], dtype=jnp.int32)
    return log_bonds, active, caps


def test_integer_components_and_counters_match_expected():
    spec, atoms, charges, bonds = graph_fixture()
    result = gpu_matrix.matrix_features(atoms, charges, bonds, spec)
    np.testing.assert_array_equal(result["component_count"], [2, 1])
    np.testing.assert_array_equal(result["active_counts"], [3, 4])
    np.testing.assert_array_equal(result["component_representatives"], [[0, 0, 2, -1], [0, 0, 0, 0]])
    np.testing.assert_array_equal(result["component_sizes_at_representatives"], [[2, 0, 1, 0], [4, 0, 0, 0]])
    np.testing.assert_array_equal(result["nonaromatic_bond_order_sum"], [[1, 1, 0, 0], [1, 2, 2, 1]])
    assert bool(result["reachability"][0, 0, 1])
    assert not bool(result["reachability"][0, 0, 2])
    assert not bool(result["reachability"][0, 3, 3])
    for key in (
        "atom_category_ok", "active_charge_category_ok", "active_bond_category_ok",
        "active_bond_symmetry_ok", "active_diagonal_zero", "padding_charge_neutral",
        "padding_bonds_zero",
    ):
        assert bool(jnp.all(result[key]))
    assert "valid" not in result


def test_asymmetry_padding_and_category_errors_are_flags_not_validity():
    spec, atoms, charges, bonds = graph_fixture()
    atoms[1, 2] = 88
    charges[0, 3] = 0
    bonds[0, 0, 1] = 2
    bonds[0, 0, 0] = 1
    bonds[0, 3, 1] = bonds[0, 1, 3] = 1
    bonds[1, 0, 1] = 99
    result = gpu_matrix.matrix_features(atoms, charges, bonds, spec)
    assert not bool(result["atom_category_ok"][1])
    assert not bool(result["active_bond_category_ok"][1])
    assert not bool(result["active_bond_symmetry_ok"][0])
    assert not bool(result["active_diagonal_zero"][0])
    assert not bool(result["padding_charge_neutral"][0])
    assert not bool(result["padding_bonds_zero"][0])
    assert not bool(result["valence_counter_supported"][1, 2])


def test_aromatic_order_is_explicitly_unsupported_numeric_counter():
    spec, atoms, charges, bonds = graph_fixture()
    bonds[0, 0, 1] = bonds[0, 1, 0] = 4
    result = gpu_matrix.matrix_features(atoms, charges, bonds, spec)
    np.testing.assert_array_equal(result["aromatic_incident_count"][0], [1, 1, 0, 0])
    np.testing.assert_array_equal(result["nonaromatic_bond_order_sum"][0], [0, 0, 0, 0])
    assert not bool(result["valence_counter_supported"][0, 0])
    assert int(result["declared_cap_excess"][0, 0]) == -1


@pytest.mark.parametrize("bad", ("shape", "dtype", "empty"))
def test_matrix_inputs_reject_wrong_structure(bad):
    spec, atoms, charges, bonds = graph_fixture()
    if bad == "shape":
        bonds = bonds[:, :-1]
    elif bad == "dtype":
        atoms = atoms.astype(np.float32)
    else:
        atoms, charges, bonds = atoms[:0], charges[:0], bonds[:0]
    with pytest.raises(ValueError):
        gpu_matrix.matrix_features(atoms, charges, bonds, spec)


def test_gpu_requirement_refuses_cpu_fallback(monkeypatch):
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match="observed cpu"):
        gpu_matrix.require_backend("gpu")
    spec, atoms, charges, bonds = graph_fixture()
    with pytest.raises(RuntimeError, match="observed cpu"):
        gpu_matrix.matrix_features(atoms, charges, bonds, spec, require_gpu=True)


@pytest.mark.parametrize("method", ("scan", "matrix"))
@pytest.mark.parametrize("n", (4, 6, 25))
def test_local_risk_values_and_gradients_keep_original_tolerance(method, n):
    log_bonds, active, caps = risk_fixture(n)
    candidate = getattr(gpu_matrix, "local_valence_" + method)
    original_value, original_gradient = jax.value_and_grad(
        lambda value: jnp.sum(loss._local(value, active, caps, INDICES))
    )(log_bonds)
    value, gradient = jax.value_and_grad(
        lambda value: jnp.sum(candidate(value, active, caps, INDICES))
    )(log_bonds)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    np.testing.assert_allclose(value, original_value, rtol=TOLERANCE, atol=TOLERANCE)
    np.testing.assert_allclose(gradient, original_gradient, rtol=TOLERANCE, atol=TOLERANCE)


@pytest.mark.parametrize("method", ("scan", "matrix"))
def test_local_gradient_matches_independent_central_difference(method):
    log_bonds, active, caps = risk_fixture(4)
    function = getattr(gpu_matrix, "local_valence_" + method)
    scalar = lambda value: jnp.sum(function(value, active, caps, INDICES))
    direction = jnp.zeros_like(log_bonds).at[0, 0, 1, 3].set(1.0)
    analytic = jnp.sum(jax.grad(scalar)(log_bonds) * direction)
    # Diagnostic direct-coordinate FD does not certify native-model derivatives.
    epsilon = 0.003
    finite_difference = (
        scalar(log_bonds + epsilon * direction) - scalar(log_bonds - epsilon * direction)
    ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, rtol=0.002, atol=0.0002)


@pytest.mark.parametrize("method", ("scan", "matrix"))
def test_install_retraces_only_local_kernel_and_restores_on_exception(method):
    old_local, old_compiled = loss._local, loss._compiled_terms
    with pytest.raises(RuntimeError, match="fixture stop"):
        with gpu_matrix.install_matrix_loss(method) as receipt:
            assert loss._local is getattr(gpu_matrix, "local_valence_" + method)
            assert loss._compiled_terms is not old_compiled
            assert receipt["native_oracle_unchanged"] is True
            with pytest.raises(RuntimeError, match="nested"):
                with gpu_matrix.install_matrix_loss(method):
                    pass
            raise RuntimeError("fixture stop")
    assert loss._local is old_local
    assert loss._compiled_terms is old_compiled


def test_policy_does_not_claim_native_validity_or_measured_gpu_speed():
    receipt = gpu_matrix.policy()
    assert receipt["version"] == "chemistry-matrix-acceleration-v1"
    assert receipt["native_oracle_replaced"] is False
    assert receipt["search_pruning_authorized"] is False
    assert receipt["GPU_speedup_established"] is False
    assert receipt["value_gradient_parity_tolerance"] == TOLERANCE
    with pytest.raises(ValueError):
        with gpu_matrix.install_matrix_loss("invalid"):
            pass


@pytest.mark.parametrize("method", ("scan", "matrix"))
def test_actual_loss_terms_trace_selected_kernel_after_original_was_warm(method, monkeypatch):
    n, batch = 4, 1
    decoded = DecoderLogits(
        jnp.zeros((batch, n, 4), dtype=jnp.float32),
        jnp.zeros((batch, n, 3), dtype=jnp.float32),
        jnp.zeros((batch, n, n, 5), dtype=jnp.float32),
    )
    arrays = {
        "active": jnp.ones((batch, n), dtype=bool),
        "route": jnp.full((batch,), 4, dtype=jnp.int32),
        "target_atoms": jnp.ones((batch, n), dtype=jnp.int32),
        "target_charges": jnp.ones((batch, n), dtype=jnp.int32),
        "target_bonds": jnp.zeros((batch, n, n), dtype=jnp.int32),
        "changed": jnp.zeros((batch, n, n), dtype=bool),
        "caps": jnp.ones((batch, n), dtype=jnp.int32),
        "cuts": jnp.zeros((batch, n, n * (n - 1) // 2), dtype=bool),
        "cut_active": jnp.zeros((batch, n), dtype=bool),
    }
    weights = loss.FailureGuidanceConfig().weights(jnp.float32)
    original = loss._compiled_terms(decoded, arrays, weights, INDICES)
    jax.block_until_ready(original)
    calls = []
    function_name = "local_valence_" + method
    selected = getattr(gpu_matrix, function_name)

    def observed(*args):
        calls.append("traced_selected_kernel")
        return selected(*args)

    monkeypatch.setattr(gpu_matrix, function_name, observed)
    with gpu_matrix.install_matrix_loss(method):
        result = loss._compiled_terms(decoded, arrays, weights, INDICES)
        jax.block_until_ready(result)
    assert calls == ["traced_selected_kernel"]
    np.testing.assert_allclose(result[0], original[0], rtol=TOLERANCE, atol=TOLERANCE)
    for key in original[1]:
        np.testing.assert_allclose(result[1][key], original[1][key], rtol=TOLERANCE, atol=TOLERANCE)
