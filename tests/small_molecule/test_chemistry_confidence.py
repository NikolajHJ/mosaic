"""Allocated CPU checks of optional valid-source bounded-slope guidance."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, prepare_failure_guidance
from mosaic.small_molecule.chemistry.confidence import (
    ConfidenceMarginConfig, _row_hinge, valid_confidence_margin,
)
from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig, failure_aware_repair_loss
from mosaic.small_molecule.chemistry.trajectory import evaluate_combined, run_chemical_updates


@pytest.fixture(autouse=True)
def allocated():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Scientific confidence fixtures require allocated DTU HPC")


def chain(dtype=jnp.float64, boost=1.5):
    atoms = jnp.zeros((3, 6), dtype).at[:, 0].set(boost)
    bonds = jnp.zeros((3, 5), dtype)
    bonds = bonds.at[jnp.arange(3), jnp.asarray([1, 0, 1])].set(boost)
    return GraphLogits(atoms, bonds)


def prepare(state, update=0):
    return prepare_failure_guidance(state.decoded(), mosaic_spec(3), update=update)


def independent(state, prep, config):
    """Host FP64 log-odds and piecewise Huber primal; no JAX loss reuse."""
    if int(prep.arrays["route"][0]) != 0:
        return 0.0
    aa = np.asarray(prep.arrays["target_atoms"])[0] - 1
    bb = np.asarray(prep.arrays["target_bonds"])[0][np.triu_indices(3, 1)]
    terms = []
    for field, labels in ((state.atoms, aa), (state.bonds, bb)):
        values = np.asarray(field, np.float64)
        rows = []
        for logits, label in zip(values, labels):
            rivals = np.delete(logits, label)
            lse = np.max(rivals) + np.log(np.sum(np.exp(rivals - np.max(rivals))))
            t = max(config.margin + lse - logits[label], 0.0)
            rows.append(t*t/2 if t <= config.slope_cap
                        else config.slope_cap * (t-config.slope_cap/2))
        terms.append(np.mean(rows))
    return float(np.mean(terms))


@pytest.mark.parametrize("name,value", [
    ("weight", -1), ("weight", True), ("weight", float("nan")),
    ("weight", float("inf")), ("margin", -0.1), ("margin", True),
    ("margin", float("inf")), ("slope_cap", 0), ("slope_cap", -1),
    ("slope_cap", True), ("slope_cap", float("nan")), ("version", "unknown"),
])
def test_invalid_policies_rejected(name, value):
    with pytest.raises(ValueError):
        ConfidenceMarginConfig(**{name: value})


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("field", ["atoms", "bonds"])
def test_independent_primal_and_two_adjacent_fd_scales(dtype, field):
    state, spec, config = chain(dtype), mosaic_spec(3), ConfidenceMarginConfig()
    prep = prepare(state)
    assert int(prep.arrays["route"][0]) == 0
    fn = lambda v: valid_confidence_margin(v.decoded(), prep, spec, config)[0]
    value, grad = eqx.filter_value_and_grad(fn)(state)
    np.testing.assert_allclose(float(value), independent(state, prep, config), rtol=2e-6)
    direction = jnp.zeros_like(getattr(state, field)).at[0, 0].set(-1).at[0, 1].set(1)
    analytic = float(jnp.sum(getattr(grad, field) * direction))
    assert abs(analytic) > 1e-5
    passes = []
    for eps in (.01, .003, .001):
        points = [eqx.tree_at(lambda v: getattr(v, field), state,
                             getattr(state, field) + sign*eps*direction) for sign in (-1, 1)]
        fd = (independent(points[1], prep, config)-independent(points[0], prep, config))/(2*eps)
        passes.append(abs(fd-analytic) <= 1e-5 + .01*max(abs(fd), abs(analytic)))
    assert any(a and b for a, b in zip(passes, passes[1:])), (analytic, passes)
    assert abs(2*analytic-analytic) > 1e-5  # nonzero probe detects a doubled/wrong pullback


def test_confident_valid_graph_and_all_its_gradients_are_exactly_zero():
    state, spec = chain(boost=8), mosaic_spec(3)
    prep = prepare(state)
    value, grad = eqx.filter_value_and_grad(
        lambda x: valid_confidence_margin(x.decoded(), prep, spec)[0]
    )(state)
    assert float(value) == 0
    assert all(np.array_equal(np.asarray(x), np.zeros_like(x)) for x in (grad.atoms, grad.bonds))


def test_invalid_source_cannot_acquire_margin_by_becoming_a_valid_trial():
    state, spec = chain(), mosaic_spec(3)
    # F with two single bonds at the middle atom is invalid.
    invalid = eqx.tree_at(lambda v: v.atoms, state, state.atoms.at[1].set(0).at[1, 4].set(1.5))
    prep = prepare(invalid)
    assert int(prep.arrays["route"][0]) != 0
    for point in (invalid, state):
        value, grad = eqx.filter_value_and_grad(
            lambda x: valid_confidence_margin(x.decoded(), prep, spec)[0]
        )(point)
        assert float(value) == 0
        assert all(np.all(np.asarray(x) == 0) for x in (grad.atoms, grad.bonds))


@pytest.mark.parametrize("deficit", [-10., 0., .5, 1., 2., 10000.])
def test_row_slope_cap_and_categorical_shift_null(deficit):
    config = ConfidenceMarginConfig()
    # Two categories give exact target-minus-rival gap, including Huber joins.
    logits = jnp.asarray([config.margin-deficit, 0.], jnp.float64)
    fn = lambda x: _row_hinge(x[None], jnp.asarray([0]), config.margin, config.slope_cap)[0].sum()
    gradient = np.asarray(jax.grad(fn)(logits))
    assert np.isfinite(gradient).all() and np.max(np.abs(gradient)) <= config.slope_cap + 1e-12
    assert abs(gradient.sum()) < 1e-12
    assert np.allclose(gradient, [-min(max(deficit, 0), config.slope_cap),
                                   min(max(deficit, 0), config.slope_cap)])


def test_valid_source_targets_remain_frozen_after_a_trial_changes_identity():
    state, spec = chain(), mosaic_spec(3)
    prep = prepare(state)
    old = prep.record()
    trial = eqx.tree_at(lambda v: v.atoms, state, state.atoms.at[1, 4].set(7))
    assert int(jnp.argmax(trial.atoms[1])) == 4
    assert float(valid_confidence_margin(trial.decoded(), prep, spec)[0]) > 0
    assert prep.record() == old
    refreshed = prepare(trial, 1)
    assert int(refreshed.arrays["route"][0]) != 0
    assert float(valid_confidence_margin(trial.decoded(), refreshed, spec)[0]) == 0


def test_preparation_tampering_is_rejected():
    state, spec = chain(), mosaic_spec(3)
    prep = prepare(state)
    changed = replace(prep, arrays={**prep.arrays, "target_atoms": prep.arrays["target_atoms"].at[0, 0].set(4)})
    with pytest.raises(ValueError, match="identity"):
        valid_confidence_margin(state.decoded(), changed, spec)


def test_mixed_batch_invalid_rows_are_zero_and_do_not_change_field_scaling():
    state, spec = chain(), mosaic_spec(3)
    invalid = eqx.tree_at(lambda v: v.atoms, state, state.atoms.at[1].set(0).at[1, 4].set(1.5))
    decoded = jax.tree.map(lambda a, b: jnp.concatenate((a, b), 0), state.decoded(), invalid.decoded())
    prep = prepare_failure_guidance(decoded, spec)
    value, gradient = eqx.filter_value_and_grad(lambda x: valid_confidence_margin(x, prep, spec)[0])(decoded)
    single = valid_confidence_margin(state.decoded(), prepare(state), spec)[0]
    np.testing.assert_allclose(value, single/2, rtol=1e-12)
    assert np.all(np.asarray(gradient.atoms[1]) == 0)
    assert np.all(np.asarray(gradient.bonds[1]) == 0)


def native(graph, *, key):
    return jnp.sum(graph.X*jnp.arange(6))*0.01 + jnp.sum(graph.E_upper**2)*0.2, {}


def options(**extra):
    return {"loss": FailureGuidanceConfig(), "repair_weight": .001,
            "element_js_weight": 0., **extra}


@pytest.mark.parametrize("invalid", [False, True])
def test_absent_opt_in_matches_archived_prechange_trajectory_bitwise(invalid):
    control_path = Path(os.environ["MOSAIC_CONTROL_TRAJECTORY"])
    module_name = "mosaic.small_molecule.chemistry._confidence_control_trajectory"
    loader = importlib.util.spec_from_file_location(module_name, control_path)
    control = importlib.util.module_from_spec(loader)
    sys.modules[module_name] = control; loader.loader.exec_module(control)
    state, spec = chain(), mosaic_spec(3)
    if invalid:
        state = eqx.tree_at(lambda v: v.atoms, state, state.atoms.at[1].set(0).at[1, 4].set(1.5))
    args = (native, state, prepare(state), spec, jnp.ones(6))
    base = control.evaluate_combined(*args, key=jax.random.key(17), config=options())
    new = evaluate_combined(*args, key=jax.random.key(17), config=options())
    assert base.value == new.value and base.primal == new.primal and base.terms == new.terms
    assert np.array_equal(base.gradients.atoms, new.gradients.atoms)
    assert np.array_equal(base.gradients.bonds, new.gradients.bonds)


def test_disabled_path_and_zero_weight_preserve_base_values_and_gradients():
    state, spec, prep = chain(), mosaic_spec(3), prepare(chain())
    args = (native, state, prep, spec, jnp.ones(6))
    base = evaluate_combined(*args, key=jax.random.key(17), config=options())
    zero = evaluate_combined(*args, key=jax.random.key(17),
                            config=options(confidence_margin=ConfidenceMarginConfig(weight=0).record()))
    assert base.value == zero.value and base.primal == zero.primal
    assert not any(k.startswith("confidence_") for k in base.terms)
    for a, b in ((base.gradients.atoms, zero.gradients.atoms), (base.gradients.bonds, zero.gradients.bonds)):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("repair_weight", [.001, 100.])
def test_combined_gradient_is_native_plus_repair_plus_independent_margin(repair_weight):
    state, spec, prep = chain(), mosaic_spec(3), prepare(chain())
    config = ConfidenceMarginConfig()
    actual = evaluate_combined(native, state, prep, spec, jnp.ones(6), key=jax.random.key(17),
                              config=options(repair_weight=repair_weight, confidence_margin=config.record()))
    def direct(x):
        binding = native(x.graph(), key=jax.random.key(17))[0]
        repair = failure_aware_repair_loss(x.decoded(), prep, spec)[0]
        margin = valid_confidence_margin(x.decoded(), prep, spec, config)[0]
        return binding + repair_weight*repair + config.weight*margin
    expected = eqx.filter_grad(direct)(state)
    np.testing.assert_allclose(actual.gradients.atoms, expected.atoms, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.gradients.bonds, expected.bonds, rtol=1e-12, atol=1e-12)
    assert actual.terms["chemistry_loss"] == 0 and actual.terms["confidence_weighted_loss"] > 0
    np.testing.assert_allclose(actual.terms["confidence_weighted_loss"], config.weight*independent(state, prep, config))


@pytest.mark.parametrize("extra", [{"element_js_weight": .1}, {"objective_mode": "native_only"}])
def test_margin_rejects_composition_or_native_only_mode(extra):
    state = chain()
    with pytest.raises(ValueError):
        evaluate_combined(native, state, prepare(state), mosaic_spec(3), jnp.ones(6),
                          key=jax.random.key(17), config=options(confidence_margin=ConfidenceMarginConfig().record(), **extra))


def test_actual_fixed_step_loop_refreshes_targets_only_between_steps_and_can_change_atoms(tmp_path):
    state, spec, seen = chain(), mosaic_spec(3), []
    def teacher(point, update):
        result = prepare(point, update)
        seen.append(result)
        return result
    def strong_identity_signal(graph, *, key):
        return -25*graph.X[0, 2] + jnp.sum(graph.E_upper**2)*.2, {}
    report = run_chemical_updates(
        strong_identity_signal, state, prepare=teacher, spec=spec, reference=jnp.ones(6),
        key=jax.random.key(17), directory=tmp_path/"run", steps=2, step_size=.3,
        max_backtracks=1, acceptance_policy="fixed_step",
        config=options(confidence_margin=ConfidenceMarginConfig().record()),
    )
    assert len(seen) == 3 and report["status"] == "passed"
    for step, update in enumerate(report["updates"]):
        assert update["preparation_identity"] == seen[step].identity
        assert len(update["attempts"]) == 1 and update["selected_trial"] == 0
    # The weak teacher is a differentiable preference, never an atom freeze.
    final = np.load(tmp_path/"run"/"snapshots"/"step-0002.npz")
    assert np.argmax(final["X"][0]) == 2
    assert int(seen[0].arrays["target_atoms"][0, 0]) == 1
    assert int(seen[-1].arrays["target_atoms"][0, 0]) == 3
