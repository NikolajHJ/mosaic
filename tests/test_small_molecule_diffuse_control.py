"""Section 4.2 probability-space initialization with the user's fixed N=6."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from mosaic.small_molecule.state import (
    bond_existence,
    full_bonds,
    graph_from_hard,
    validate_graph,
)


def _setup():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "diffuse_example", root / "examples/protenix_relaxed_gradients.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(
        (root / "examples/configs/brd4_canonical_diffuse_5000.json").read_text()
    )
    return module, config


def _project(rows):
    result = []
    for row in np.asarray(rows, dtype=np.float64):
        ordered = np.sort(row)[::-1]
        threshold = (ordered.cumsum() - 1) / np.arange(1, len(row) + 1)
        last = np.flatnonzero(ordered > threshold)[-1]
        result.append(np.maximum(row - threshold[last], 0))
    return np.asarray(result)


def test_canonical_recipe_adds_small_noise_before_projection_and_keeps_six_slots():
    module, config = _setup()
    (case,) = module._case_specs(config)
    graph = module._initial_graph(
        graph_from_hard([0] * 6, [1] * 15), case, jax.random.key(17)
    )
    atom_key, bond_key = jax.random.split(jax.random.key(17))
    X = np.tile([0.75] + [0.05] * 5, (6, 1)) + 0.005 * np.asarray(
        jax.random.normal(atom_key, (6, 6))
    )
    E = np.tile([0.55] + [0.1125] * 4, (15, 1)) + 0.005 * np.asarray(
        jax.random.normal(bond_key, (15, 5))
    )
    np.testing.assert_allclose(graph.X, _project(X), atol=2e-7)
    np.testing.assert_allclose(graph.E_upper, _project(E), atol=2e-7)
    validate_graph(graph)
    assert graph.n_atoms == 6
    assert graph.E_upper.shape == (15, 5)
    assert len(np.unique(np.asarray(graph.X), axis=0)) == 6
    assert len(np.unique(np.asarray(graph.E_upper), axis=0)) == 15
    np.testing.assert_array_equal(
        full_bonds(graph), np.asarray(full_bonds(graph)).swapaxes(0, 1)
    )
    np.testing.assert_array_equal(np.diag(bond_existence(graph)), np.zeros(6))
    assert abs(float(np.asarray(bond_existence(graph)).sum(axis=1).mean()) - 2.25) < 0.1
    # The actual predeclared seed supports unchanged central/null probes.
    assert float(np.min(graph.X)) > 0.01
    assert float(np.min(graph.E_upper)) > 0.01
    fixture = json.loads(
        (Path(__file__).parent / "data/brd4_3mxf/fixture.json").read_text()
    )
    assert fixture["n_heavy_atoms"] == graph.n_atoms


def test_canonical_start_is_seeded_and_does_not_inherit_scaffold_topology():
    module, config = _setup()
    (case,) = module._case_specs(config)
    graph = module._initial_graph(
        graph_from_hard([0] * 6, [1] * 15), case, jax.random.key(17)
    )
    other_labels = module._initial_graph(
        graph_from_hard([2] * 6, [0] * 15), case, jax.random.key(17)
    )
    other_seed = module._initial_graph(
        graph_from_hard([0] * 6, [0] * 15), case, jax.random.key(99)
    )
    np.testing.assert_array_equal(graph.X, other_labels.X)
    np.testing.assert_array_equal(graph.E_upper, other_labels.E_upper)
    assert not np.array_equal(graph.X, other_seed.X)
    assert not np.array_equal(graph.E_upper, other_seed.E_upper)
    assert np.all(np.asarray(graph.E_upper).argmax(-1) == 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("carbon_probability", 1),
        ("carbon_probability", 1 / 6),
        ("expected_degree", 0),
        ("atom_noise", -0.01),
        ("bond_noise", 0),
        ("atom_noise", True),
        ("bond_noise", float("nan")),
    ],
)
def test_canonical_recipe_rejects_invalid_or_missing_settings(field, value):
    module, config = _setup()
    config["initialization"][field] = value
    with pytest.raises(ValueError):
        module._case_specs(config)


def test_canonical_replay_changes_initialization_only():
    module, config = _setup()
    previous = deepcopy(config)
    previous.pop("initialization")
    previous["mixing_values"] = [0.15, 0.3, 0.5]
    previous["checks"]["steps"] = 3
    module._validate_prepared_config(
        config,
        previous,
        allow_initialization_change=True,
        allow_update_count_change=True,
    )
    config["fixture"] = "different-size-fixture.json"
    with pytest.raises(ValueError):
        module._validate_prepared_config(
            config,
            previous,
            allow_initialization_change=True,
            allow_update_count_change=True,
        )
