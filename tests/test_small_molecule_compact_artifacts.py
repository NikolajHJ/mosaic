"""Artifact compression must preserve numerical behavior and finite checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.gradient_checks import check_graph_gradients, _value
from mosaic.small_molecule.state import (
    graph_from_hard,
    mix_with_uniform,
    bond_existence,
)


def loss(graph, *, key):
    del key
    value = jnp.sum(graph.X[:, 1]) + 0.5 * jnp.sum(bond_existence(graph))
    return value, {
        "distogram_logits": jnp.full((2, 2, 4), value),
        "trunk_z": jnp.ones((2, 2, 8)),
    }


def test_compact_preserves_trajectory_and_raw_gradients(tmp_path):
    graph = mix_with_uniform(graph_from_hard([0, 0], [1]), 0.5)
    reports = []
    for policy in ("full", "compact"):
        reports.append(
            check_graph_gradients(
                loss,
                graph,
                key=jax.random.key(17),
                output_directory=tmp_path / policy,
                steps=2,
                record_readouts=True,
                primal_every_step=True,
                artifact_policy=policy,
            )
        )
    assert reports[0]["states"] == reports[1]["states"]
    with (
        np.load(tmp_path / "full/step-0-trial-0.npz") as full,
        np.load(tmp_path / "compact/step-0-trial-0.npz") as compact,
    ):
        assert "trunk_z" in full and "trunk_z" not in compact
        assert "distogram_logits" not in compact
        for name in compact.files:
            np.testing.assert_array_equal(compact[name], full[name])
    with np.load(tmp_path / "compact/initial.npz") as initial:
        assert "distogram_logits" in initial and "gradient_X" in initial
        assert "trunk_z" not in initial


def test_compact_still_rejects_nonfinite_omitted_tensor(tmp_path):
    def broken(graph, *, key):
        value, aux = loss(graph, key=key)
        return value, {**aux, "trunk_z": jnp.full((2, 2, 8), jnp.nan)}

    with pytest.raises(FloatingPointError, match="Nonfinite"):
        _value(
            broken,
            mix_with_uniform(graph_from_hard([0, 0], [1]), 0.5),
            jax.random.key(17),
            tmp_path / "failed.npz",
            compact=True,
            logits=False,
        )
