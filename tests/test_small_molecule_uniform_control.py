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
