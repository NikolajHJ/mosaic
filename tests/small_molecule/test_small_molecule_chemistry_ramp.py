"""Allocated CPU: schedule boundaries, shared random graphs and trial freezing."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import (
    FailureGuidanceConfig,
    GraphLogits,
    mosaic_spec,
    prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry.initialization import initialize_graph
from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
from mosaic.small_molecule.chemistry.trajectory import (
    evaluate_combined,
    run_chemical_updates,
)
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform


def test_schedule_endpoints_update_index_and_geometric_midpoint():
    schedule = RepairWeightSchedule(0.001, 50.0, 5000, 10000)
    assert schedule.for_update(1) == schedule.for_update(5000) == 0.001
    assert 0.001 < schedule.for_update(5001) < 0.002
    assert schedule.for_update(7500) == pytest.approx(np.sqrt(0.001 * 50.0))
    assert schedule.for_update(10000) == 50.0
    assert schedule.for_state(4999) == 0.001
    assert schedule.for_state(5000) == schedule.for_update(5001)
    assert schedule.for_state(9999) == schedule.for_state(10000) == 50.0
    assert all(
        schedule.for_update(i) <= schedule.for_update(i + 1) for i in range(1, 10000)
    )
    for invalid in (0, 10001, True, 1.5):
        with pytest.raises(ValueError):
            schedule.for_update(invalid)


@pytest.mark.parametrize(
    "args",
    [
        (0, 50, 5, 10),
        (0.001, float("inf"), 5, 10),
        (1, 0.5, 5, 10),
        (0.001, 50, 10, 10),
    ],
)
def test_invalid_schedules_fail(args):
    with pytest.raises(ValueError):
        RepairWeightSchedule(*args)


def test_seeded_random_initialization_and_chain_recipe_preserve_schema():
    hard = graph_from_hard(
        [0] * 6, [1 if j == i + 1 else 0 for i in range(6) for j in range(i + 1, 6)]
    )
    recipe = {"kind": "dirichlet", "concentration": 1.0, "seed": 20261019}
    first, second = initialize_graph(hard, recipe), initialize_graph(hard, recipe)
    changed = initialize_graph(hard, {**recipe, "seed": 20261020})
    assert first.X.shape == (6, 6) and first.E_upper.shape == (15, 5)
    for a, b in ((first.X, second.X), (first.E_upper, second.E_upper)):
        np.testing.assert_array_equal(a, b)
        assert np.min(a) > 0
        np.testing.assert_allclose(np.sum(a, -1), 1.0, atol=2e-7)
    assert not np.array_equal(first.X, changed.X)
    np.testing.assert_allclose(
        GraphLogits.from_graph(first).graph().X, first.X, atol=2e-7
    )
    chain = initialize_graph(hard, {"kind": "chain_uniform", "mixing": 0.5})
    np.testing.assert_array_equal(chain.X, mix_with_uniform(hard, 0.5).X)


def test_weight_changes_recompute_scalar_and_gradient_reusing_only_native():
    initial = GraphLogits.from_graph(
        mix_with_uniform(graph_from_hard([4, 4], [3], dtype=jnp.float64), 0.5)
    )
    spec = mosaic_spec(2)
    prepared = prepare_failure_guidance(initial.decoded(), spec)
    calls = []

    def native(graph, key):
        calls.append(True)
        return jnp.sum(graph.X * jnp.arange(6)) * 0.01, {}

    options = {
        "loss": FailureGuidanceConfig(),
        "repair_weight": 0.001,
        "element_js_weight": 0.1,
    }
    key = jax.random.key(17)
    low = evaluate_combined(
        native, initial, prepared, spec, jnp.ones(6), key=key, config=options
    )
    count = len(calls)
    high = evaluate_combined(
        native,
        initial,
        prepared,
        spec,
        jnp.ones(6),
        key=key,
        config={**options, "repair_weight": 50.0},
        native=low.native,
    )
    assert len(calls) == count and high.repair_weight == 50.0
    assert high.primal - low.primal == pytest.approx(
        (50.0 - 0.001) * low.terms["chemistry_loss"]
    )
    assert not np.allclose(low.gradients.bonds, high.gradients.bonds)
    # AD/central FD with one frozen reference at the large coefficient.
    delta = eqx.tree_at(lambda v: v.bonds, initial, jnp.ones_like(initial.bonds) * 0.03)
    direction = delta.bonds.at[:, 0].add(0.1)
    eps = 1e-4
    values = []
    for sign in (-1, 1):
        trial = eqx.tree_at(
            lambda v: v.bonds, initial, initial.bonds + sign * eps * direction
        )
        values.append(
            evaluate_combined(
                native,
                trial,
                prepared,
                spec,
                jnp.ones(6),
                key=key,
                config={**options, "repair_weight": 50.0},
            ).primal
        )
    assert (values[1] - values[0]) / (2 * eps) == pytest.approx(
        float(jnp.sum(high.gradients.bonds * direction)), rel=1e-4, abs=1e-5
    )


def test_trial_sets_freeze_weight_and_refresh_only_after_acceptance(
    tmp_path, monkeypatch
):
    import mosaic.small_molecule.chemistry.trajectory as module

    monkeypatch.setattr(module, "check_combined_directions", lambda *a, **k: [])
    original = module.evaluate_combined
    seen = []

    def observe(*args, **kwargs):
        seen.append(kwargs["config"]["repair_weight"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_combined", observe)
    initial = GraphLogits.from_graph(
        mix_with_uniform(graph_from_hard([0, 0], [1]), 0.5)
    )
    spec = mosaic_spec(2)
    schedule = RepairWeightSchedule(0.001, 50.0, 1, 3)
    report = run_chemical_updates(
        lambda graph, key: (jnp.asarray(0.0, graph.X.dtype), {}),
        initial,
        prepare=lambda logits, update: prepare_failure_guidance(
            logits.decoded(), spec, update=update
        ),
        spec=spec,
        reference=jnp.ones(6),
        key=jax.random.key(17),
        directory=tmp_path / "run",
        steps=3,
        step_size=0.03,
        max_backtracks=10,
        config={
            "loss": FailureGuidanceConfig(),
            "repair_weight": 0.001,
            "element_js_weight": 0.0,
        },
        schedule=schedule,
    )
    assert seen == [0.001] + [0.001] * 10 + [schedule.for_state(1)] + [
        schedule.for_state(1)
    ] * 10 + [50.0] + [50.0] * 10 + [50.0]
    assert report["counts"]["applied"] == report["counts"]["no_op"] == 3
    for update in report["updates"]:
        assert all(
            a["repair_weight"] == update["repair_weight"] for a in update["attempts"]
        )
    assert [s["repair_weight"] for s in report["states"]] == [
        0.001,
        schedule.for_state(1),
        50.0,
        50.0,
    ]
    assert (
        json.loads((tmp_path / "run/report.json").read_text())["repair_weight_schedule"]
        == schedule.record()
    )
    assert Path(tmp_path / "run/trials/step-0002.npz").is_file()
