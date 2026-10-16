"""Exact uniform starts and explicit replay permissions must stay auditable."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

from mosaic.small_molecule.state import (
    bond_existence,
    graph_from_hard,
    mix_with_uniform,
    validate_graph,
)


def _module():
    path = (
        Path(__file__).resolve().parents[1] / "examples/protenix_relaxed_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("uniform_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config():
    import json

    path = (
        Path(__file__).resolve().parents[1] / "examples/configs/brd4_uniform_5000.json"
    )
    return json.loads(path.read_text())


def test_uniform_control_is_exact_and_has_no_seed_noise_or_chain_edges():
    module = _module()
    (case,) = module._case_specs(_config())
    hard = graph_from_hard([0] * 6, [1] * 15)
    graph = module._initial_graph(hard, case, jax.random.key(17))
    other = module._initial_graph(hard, case, jax.random.key(99))
    np.testing.assert_array_equal(graph.X, np.full((6, 6), 1 / 6, dtype=np.float32))
    np.testing.assert_array_equal(
        graph.E_upper, np.tile([0.5, 0.125, 0.125, 0.125, 0.125], (15, 1))
    )
    np.testing.assert_array_equal(graph.X, other.X)
    np.testing.assert_array_equal(graph.E_upper, other.E_upper)
    np.testing.assert_array_equal(
        bond_existence(graph), 0.5 * (np.ones((6, 6)) - np.eye(6))
    )


def test_old_mixture_initialization_is_unchanged():
    module = _module()
    hard = graph_from_hard([0, 1], [1])
    (case,) = module._case_specs({"mixing_values": [0.3]})
    graph = module._initial_graph(hard, case, jax.random.key(17))
    expected = mix_with_uniform(hard, 0.3)
    np.testing.assert_array_equal(graph.X, expected.X)
    np.testing.assert_array_equal(graph.E_upper, expected.E_upper)


def test_initialization_replacement_requires_explicit_policy():
    module = _module()
    config = _config()
    previous = deepcopy(config)
    previous.pop("initialization")
    previous["mixing_values"] = [0.15, 0.3, 0.5]
    previous["checks"]["steps"] = 3
    with pytest.raises(ValueError):
        module._validate_prepared_config(
            config, previous, allow_update_count_change=True
        )
    module._validate_prepared_config(
        config,
        previous,
        allow_update_count_change=True,
        allow_initialization_change=True,
    )
    for field, value in [
        ("seed", 99),
        ("objective", "dbmol_pocket"),
        ("matmul_precision", "default"),
    ]:
        changed = deepcopy(config)
        changed[field] = value
        with pytest.raises(ValueError):
            module._validate_prepared_config(
                changed,
                previous,
                allow_update_count_change=True,
                allow_initialization_change=True,
            )
    changed = deepcopy(config)
    changed["checks"]["initial_step_size"] = 0.09
    with pytest.raises(ValueError):
        module._validate_prepared_config(
            changed,
            previous,
            allow_update_count_change=True,
            allow_initialization_change=True,
        )


def test_uniform_initializer_rejects_ambiguous_or_impossible_requests():
    module = _module()
    config = _config()
    with pytest.raises(ValueError, match="not both"):
        module._case_specs({**config, "mixing_values": [1.0]})
    config["initialization"]["bond_existence_numerator"] = 6
    (case,) = module._case_specs(config)
    with pytest.raises(ValueError, match="in \\(0, 1\\)"):
        module._initial_graph(
            graph_from_hard([0] * 6, [0] * 15), case, jax.random.key(17)
        )


def _noisy_config():
    import json

    path = (
        Path(__file__).resolve().parents[1]
        / "examples/configs/brd4_noisy_uniform_5000.json"
    )
    return json.loads(path.read_text())


def _reference_projection(rows):
    """Independent NumPy row simplex projection for the declared noise recipe."""
    projected = []
    for row in np.asarray(rows, dtype=np.float64):
        ordered = np.sort(row)[::-1]
        offsets = (np.cumsum(ordered) - 1) / np.arange(1, len(row) + 1)
        last = np.flatnonzero(ordered > offsets)[-1]
        projected.append(np.maximum(row - offsets[last], 0))
    return np.asarray(projected)


def test_noisy_control_matches_projected_gaussian_mixture_and_keeps_floor():
    module = _module()
    (case,) = module._case_specs(_noisy_config())
    key = jax.random.key(17)
    hard = graph_from_hard([0] * 6, [1] * 15)
    graph = module._initial_graph(hard, case, key)
    atom_key, bond_key = jax.random.split(key)
    atom_noise = np.asarray(jax.random.normal(atom_key, (6, 6)))
    bond_noise = np.asarray(jax.random.normal(bond_key, (15, 5)))
    atom_expected = 0.5 / 6 + 0.5 * _reference_projection(atom_noise)
    bond_floor = 0.5 * np.tile([0.5, 0.125, 0.125, 0.125, 0.125], (15, 1))
    bond_expected = bond_floor + 0.5 * _reference_projection(bond_noise)
    assert (atom_noise < 0).any() and (bond_noise < 0).any()
    np.testing.assert_allclose(graph.X, atom_expected, atol=2e-7)
    np.testing.assert_allclose(graph.E_upper, bond_expected, atol=2e-7)
    assert np.all(np.asarray(graph.X) >= 0.5 / 6 - 1e-7)
    assert np.all(np.asarray(graph.E_upper) >= bond_floor - 1e-7)
    # The unchanged largest central probe is 0.01, and order-null move is 0.005.
    assert float(np.min(graph.X)) > 0.01
    assert float(np.min(graph.E_upper)) > 0.01
    validate_graph(graph)
    np.testing.assert_array_equal(np.diag(bond_existence(graph)), np.zeros(6))
    assert not np.allclose(graph.X[0], graph.X[1])
    assert not np.allclose(bond_existence(graph)[0, 1:], 0.5)


def test_noisy_control_is_seeded_and_independent_of_hard_scaffold_labels():
    module = _module()
    (case,) = module._case_specs(_noisy_config())
    hard = graph_from_hard([0] * 6, [1] * 15)
    changed = graph_from_hard([2] * 6, [0] * 15)
    graph = module._initial_graph(hard, case, jax.random.key(17))
    repeat = module._initial_graph(changed, case, jax.random.key(17))
    different = module._initial_graph(hard, case, jax.random.key(99))
    np.testing.assert_array_equal(graph.X, repeat.X)
    np.testing.assert_array_equal(graph.E_upper, repeat.E_upper)
    assert not np.array_equal(graph.X, different.X)
    assert not np.array_equal(graph.E_upper, different.E_upper)


@pytest.mark.parametrize("weight", [0, 1, -0.5, float("nan"), True, "0.5"])
def test_noisy_control_rejects_invalid_weights(weight):
    config = _noisy_config()
    config["initialization"]["noise_weight"] = weight
    with pytest.raises(ValueError, match="Noise weight"):
        _module()._case_specs(config)


def test_noisy_replay_requires_initialization_permission_and_keeps_rate():
    module = _module()
    config, previous = _noisy_config(), _config()
    with pytest.raises(ValueError):
        module._validate_prepared_config(config, previous)
    module._validate_prepared_config(config, previous, allow_initialization_change=True)
    config["checks"]["initial_step_size"] = 0.09
    with pytest.raises(ValueError):
        module._validate_prepared_config(
            config, previous, allow_initialization_change=True
        )
