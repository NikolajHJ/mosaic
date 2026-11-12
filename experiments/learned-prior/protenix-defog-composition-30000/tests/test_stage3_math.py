"""Tests for intervention meaning rather than native-model efficacy."""

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph
from mosaic.small_molecule.Stage2 import operators as op


spec = importlib.util.spec_from_file_location("stage3_math", Path(__file__).parents[1] / "scripts" / "stage3_math.py")
maths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maths)


def graph():
    return RelaxedGraph(jnp.array([[.7, .3, 0., 0., 0., 0.], [.1, .9, 0., 0., 0., 0.]]),
                        jnp.array([[.5, .5, 0., 0., 0.]]))


def test_schedule_update_endpoints_and_delayed_ramp():
    linear = {"kind": "linear", "start": .2, "end": .95}
    assert [maths.schedule_value(linear, i) for i in (0, 499, 500)] == pytest.approx([.2, .95, .95])
    delayed = linear | {"begin": 250}
    assert maths.schedule_value(delayed, 249) == .2
    assert maths.schedule_value(delayed, 499) == .95
    cyclic = {"kind": "cosine", "low": 0, "high": .1, "period": 100}
    assert [maths.schedule_value(cyclic, i) for i in (0, 50, 100)] == pytest.approx([0, .1, 0])
    for invalid in ({"kind": "linear", "start": 0, "end": 1, "finish": 0}, {"kind": "cosine", "low": 1, "high": 0, "period": 2}, float("nan")):
        with pytest.raises(ValueError):
            maths.schedule_value(invalid, 0)


def test_temperature_one_is_identity_in_forward_and_gradient_even_at_zero():
    g = graph()
    out = maths.cool_graph(g, 1.)
    np.testing.assert_array_equal(out.X, g.X)
    np.testing.assert_array_equal(out.E_upper, g.E_upper)
    weights = jnp.arange(12, dtype=g.X.dtype).reshape(2, 6)
    grad = jax.grad(lambda x: jnp.sum(maths.cool_graph(RelaxedGraph(x, g.E_upper), 1).X * weights))(g.X)
    np.testing.assert_array_equal(grad, weights)


def test_cooling_accepts_dynamic_temperature_and_reuses_jit_contract():
    g = graph()
    evaluate = jax.jit(jax.value_and_grad(lambda current, temperature: jnp.sum(maths.cool_graph(current, temperature).X ** 2)))
    for temperature in (1., .5, .25):
        value, grad = evaluate(g, jnp.array(temperature))
        assert np.isfinite(value) and np.isfinite(grad.X).all()
    _, identity_grad = evaluate(g, jnp.array(1.))
    np.testing.assert_allclose(identity_grad.X, 2 * g.X)


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_cooling_rejects_invalid_host_temperatures(temperature):
    with pytest.raises(ValueError, match="temperature"):
        maths.cool_graph(graph(), temperature)


def test_cooling_sharpens_preserves_zeros_and_has_finite_exact_gradients():
    g = graph()
    cooled = maths.cool_graph(g, .25)
    validate_graph(cooled)
    assert float(cooled.X[0, 0]) > float(g.X[0, 0])
    np.testing.assert_array_equal(np.asarray(cooled.X)[:, 2:], 0)
    grad = jax.grad(lambda x: jnp.sum(maths.cool_graph(RelaxedGraph(x, g.E_upper), .25).X ** 2))(g.X)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(np.asarray(grad)[:, 2:], 0)
    assert np.linalg.norm(grad) > 0


def test_ste_same_forward_but_passthrough_gradient():
    g = graph()
    exact = maths.cool_graph(g, .5)
    ste = maths.cool_graph(g, .5, "ste")
    np.testing.assert_allclose(ste.X, exact.X)
    grad = jax.grad(lambda x: maths.cool_graph(RelaxedGraph(x, g.E_upper), .5, "ste").X.sum())(g.X)
    np.testing.assert_array_equal(grad, np.ones_like(g.X))


def test_incremental_state_cooling_matches_one_absolute_transform_without_updates():
    g = graph()
    out = maths.cool_state_graph(g, 1., .8)
    out = maths.cool_state_graph(out, .8, .5)
    out = maths.cool_state_graph(out, .5, .25)
    expected = maths.cool_graph(g, .25)
    np.testing.assert_allclose(out.X, expected.X, atol=1e-7)
    np.testing.assert_allclose(out.E_upper, expected.E_upper, atol=1e-7)
    np.testing.assert_array_equal(maths.cool_state_graph(out, .25, .25).X, out.X)


@pytest.mark.parametrize("seed", [17, 23, 42, 71, 101])
def test_initial_families_match_atoms_and_have_intended_hard_connectivity(seed):
    sparse = maths.make_initial_graph(seed, "sparse")
    dense = maths.make_initial_graph(seed, "dense")
    fragmented = maths.make_initial_graph(seed, "fragmented")
    np.testing.assert_array_equal(sparse.X, dense.X)
    np.testing.assert_array_equal(sparse.X, fragmented.X)
    np.testing.assert_array_equal(sparse.E_upper, maths.make_initial_graph(seed, "sparse").E_upper)
    assert len(np.unique(op.components(25, sparse.E_upper))) == 1
    assert len(np.unique(op.components(25, fragmented.E_upper))) == 5
    assert op.argmax_valence_violations(sparse.X, sparse.E_upper) == (0, 0.)
    assert np.sum(np.asarray(sparse.E_upper).argmax(-1) != 0) == 24
    assert np.sum(np.asarray(dense.E_upper).argmax(-1) != 0) > 100


def test_one_atom_initialization_has_empty_bond_rows():
    for family in ("sparse", "dense", "fragmented"):
        g = maths.make_initial_graph(17, family, n_atoms=1)
        validate_graph(g)
        assert g.E_upper.shape == (0, 5)


def test_mse_matches_independent_head_derivative_and_stops_target():
    g = graph()
    target = RelaxedGraph(g.X * .5 + .5 / 6, g.E_upper * .5 + .5 / 5)
    dx = jax.grad(lambda student: maths.teacher_distance(student, target))(g)
    np.testing.assert_allclose(dx.X, 2 * (g.X - target.X) / len(g.X), atol=1e-7)
    np.testing.assert_allclose(dx.E_upper, 2 * (g.E_upper - target.E_upper) / len(g.E_upper), atol=1e-7)
    dt = jax.grad(lambda t: maths.teacher_distance(g, t))(target)
    np.testing.assert_array_equal(dt.X, np.zeros_like(g.X))
    np.testing.assert_array_equal(dt.E_upper, np.zeros_like(g.E_upper))


def test_kl_has_finite_zero_entries_and_empty_bond_head():
    g = RelaxedGraph(graph().X[:1], jnp.zeros((0, 5), jnp.float32))
    target = RelaxedGraph(g.X * .5 + .5 / 6, g.E_upper)
    value, derivative = jax.value_and_grad(lambda student: maths.teacher_distance(student, target, kind="kl"))(g)
    assert np.isfinite(value) and np.isfinite(derivative.X).all()
    assert derivative.E_upper.shape == (0, 5)


def test_projection_schedule_bridge_only_and_reports():
    g = maths.make_initial_graph(17, "fragmented")
    config = {"mode": "bridge_only", "every": 5, "start_update": 2}
    unchanged, skipped = maths.apply_projection(g, config, 1)
    np.testing.assert_array_equal(unchanged.E_upper, g.E_upper)
    assert not skipped["applied"]
    connected, note = maths.apply_projection(g, config, 2)
    assert note["components_before"] == 5 and note["components_after"] == 1
    assert [item["operator"] for item in note["steps"]] == ["bridge"]
    assert note["bridges_surviving_final"] == 4
    validate_graph(connected)
    assert not maths.apply_projection(g, config, 3)[1]["applied"]
    assert maths.apply_projection(g, config, 7)[1]["applied"]


def test_normal_operator_composition_matches_public_implementation():
    g = maths.make_initial_graph(17, "dense")
    config = {"mode": "sparse", "readout_caps": True, "fluidity_threshold": .95, "fluidity_strength": .05,
              "bridge_retry": True, "bridge_probability": .6}
    expected_x, expected_e, _ = op.apply_operators(np.asarray(g.X), np.asarray(g.E_upper), op.Stage2Config(**config))
    actual, note = maths.apply_projection(g, config, 1)
    np.testing.assert_allclose(actual.X, expected_x, atol=1e-7)
    np.testing.assert_allclose(actual.E_upper, expected_e, atol=1e-7)
    assert note["steps"][0]["operator"] == "fluidity"
    assert note["steps"][-1]["operator"] == "post_shrink_valence"


def test_no_final_valence_is_explicit_diagnostic_ablation():
    g = maths.make_initial_graph(23, "fragmented")
    _, note = maths.apply_projection(g, {"mode": "connected", "final_valence": False}, 1)
    assert [item["operator"] for item in note["steps"]] == ["initial_valence", "bridge"]


def test_simple_chemistry_loss_attached_and_zero_when_no_excess():
    g = maths.make_initial_graph(17, "dense")
    value, grad = jax.value_and_grad(maths.simple_soft_valence_loss)(g)
    assert float(value) > 0 and np.isfinite(grad.X).all() and np.linalg.norm(grad.X) > 0
    assert np.isfinite(grad.E_upper).all() and np.linalg.norm(grad.E_upper) > 0
    empty = RelaxedGraph(graph().X, jnp.array([[1., 0., 0., 0., 0.]]))
    assert float(maths.simple_soft_valence_loss(empty)) == 0.
