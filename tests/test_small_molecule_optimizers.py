"""Analytic strict-evaluator/update contracts, authored for allocated HPC."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import RawEvaluation
from mosaic.small_molecule.optimizers import (
    NonfiniteEvaluationError,
    evaluate_and_project,
    evaluate_graph,
    project_graph,
    projected_step,
)
from mosaic.small_molecule.state import (
    RelaxedGraph,
    graph_from_hard,
    mix_with_uniform,
    validate_graph,
)


def _interior():
    return mix_with_uniform(graph_from_hard([0, 1], [1]), 1.0)


def test_graph_only_raw_derivatives_and_separate_analytic_steps():
    graph = _interior()
    target = graph_from_hard([2, 5], [3])

    def loss(g, *, key):
        del key
        value = jnp.sum((g.X - target.X) ** 2) + 3 * jnp.sum((g.E_upper - target.E_upper) ** 2)
        return value, {"raw_output": g.X @ jnp.arange(6.0), "label": "analytic"}

    key = jax.random.key(13)
    step = evaluate_and_project(loss, graph, key, eta_X=0.1, eta_E=0.05)
    validate_graph(step.next_graph)
    np.testing.assert_allclose(step.evaluation.gradients.X, 2 * (graph.X - target.X))
    np.testing.assert_allclose(step.evaluation.gradients.E_upper, 6 * (graph.E_upper - target.E_upper))
    # These settings stay interior, so an analytic convex combination specifies
    # the result without reproducing the projection algorithm under test.
    np.testing.assert_allclose(step.next_graph.X, 0.8 * graph.X + 0.2 * target.X, atol=1e-7)
    np.testing.assert_allclose(step.next_graph.E_upper, 0.7 * graph.E_upper + 0.3 * target.E_upper, atol=1e-7)
    assert step.eta_X == 0.1
    assert step.eta_E == 0.05
    assert step.evaluation.graph is graph
    np.testing.assert_array_equal(jax.random.key_data(step.evaluation.key), jax.random.key_data(key))
    next_evaluation = evaluate_graph(loss, step.next_graph, key)
    assert float(next_evaluation.value) < float(step.evaluation.value)
    # The old loss stays attached to the old graph, never to next_graph.
    np.testing.assert_allclose(step.evaluation.value, loss(graph, key=key)[0])


def test_raw_gradients_are_not_centered_or_clipped_even_when_projection_cancels_them():
    graph = _interior()

    def loss(g, *, key):
        del key
        return 50 * jnp.sum(g.X) + 70 * jnp.sum(g.E_upper), {}

    evaluation = evaluate_graph(loss, graph, jax.random.key(0))
    np.testing.assert_array_equal(evaluation.gradients.X, np.full((2, 6), 50))
    np.testing.assert_array_equal(evaluation.gradients.E_upper, np.full((1, 5), 70))
    step = projected_step(evaluation, eta_X=0.02, eta_E=0.01)
    np.testing.assert_allclose(step.next_graph.X, graph.X, atol=1e-7)
    np.testing.assert_allclose(step.next_graph.E_upper, graph.E_upper, atol=1e-7)
    assert step.projection_residuals["X"] > 0
    assert step.projection_residuals["E_upper"] > 0


def test_projection_can_hit_boundary_without_deleting_an_atom_slot():
    graph = _interior()

    def loss(g, *, key):
        del key
        return -jnp.sum(g.X[:, 0]) - 2 * jnp.sum(g.E_upper[:, 0]), {}

    step = evaluate_and_project(loss, graph, jax.random.key(0), eta_X=1, eta_E=1)
    validate_graph(step.next_graph)
    np.testing.assert_array_equal(step.next_graph.X, [[1, 0, 0, 0, 0, 0]] * 2)
    np.testing.assert_array_equal(step.next_graph.E_upper, [[1, 0, 0, 0, 0]])
    assert step.next_graph.n_atoms == graph.n_atoms
    assert all(value > 0 for value in step.projection_residuals.values())


@pytest.mark.parametrize("location", ["value", "aux", "gradient"])
def test_nonfinite_raw_quantities_fail_without_sanitation_and_are_retained(location):
    graph = graph_from_hard([0, 1], [1])
    key = jax.random.key(7)

    def loss(g, *, key):
        del key
        if location == "value":
            return jnp.sum(g.X) * jnp.inf, {"label": "deliberate failure"}
        if location == "aux":
            return jnp.sum(g.X), {"output": jnp.array([0.0, jnp.nan])}
        # sqrt(0) has a finite value and an infinite derivative: inspecting the
        # score alone cannot establish a safe update.
        return jnp.sqrt(g.X[0, 1]), {"label": "finite score, bad derivative"}

    with pytest.raises(NonfiniteEvaluationError) as captured:
        evaluate_and_project(loss, graph, key, eta_X=0.1, eta_E=0.1)
    failure = captured.value
    assert failure.graph is graph
    assert failure.gradients is not None
    assert failure.nonfinite_paths
    np.testing.assert_array_equal(jax.random.key_data(failure.key), jax.random.key_data(key))
    if location == "value":
        assert np.isinf(failure.value)
    elif location == "aux":
        assert np.isnan(failure.aux["output"][1])
        assert np.isfinite(failure.value)
    else:
        assert np.isfinite(failure.value)
        assert not np.isfinite(failure.gradients.X).all()


def test_nonfinite_input_is_preserved_before_forward_is_called():
    graph = RelaxedGraph(jnp.full((2, 6), jnp.nan), jnp.full((1, 5), 0.2))

    def forbidden_forward(g, *, key):
        raise AssertionError("invalid graph must never enter a model forward")

    with pytest.raises(NonfiniteEvaluationError) as captured:
        evaluate_graph(forbidden_forward, graph, jax.random.key(0))
    assert captured.value.graph is graph
    assert captured.value.value is None
    assert captured.value.gradients is None


@pytest.mark.parametrize("field,value", [("eta_X", -1), ("eta_E", np.inf), ("eta_X", np.nan)])
def test_invalid_step_sizes_are_rejected(field, value):
    graph = _interior()
    evaluation = RawEvaluation(graph, jnp.array(0.0), {},
                               RelaxedGraph(jnp.zeros_like(graph.X), jnp.zeros_like(graph.E_upper)),
                               jax.random.key(0))
    settings = {"eta_X": 0.1, "eta_E": 0.1, field: value}
    with pytest.raises(ValueError, match=field):
        projected_step(evaluation, **settings)


def test_manually_supplied_nonfinite_gradients_cannot_bypass_strict_update():
    graph = _interior()
    evaluation = RawEvaluation(graph, jnp.array(0.0), {},
                               RelaxedGraph(jnp.full_like(graph.X, jnp.nan), jnp.zeros_like(graph.E_upper)),
                               jax.random.key(0))
    with pytest.raises(NonfiniteEvaluationError):
        projected_step(evaluation, eta_X=0.1, eta_E=0.1)


@pytest.mark.parametrize("value", [jnp.array([0.0, 1.0]), jnp.array(1), jnp.array(1j)])
def test_manually_supplied_non_scalar_or_non_real_score_is_rejected(value):
    graph = _interior()
    evaluation = RawEvaluation(
        graph, value, {},
        RelaxedGraph(jnp.zeros_like(graph.X), jnp.zeros_like(graph.E_upper)),
        jax.random.key(0),
    )
    with pytest.raises(TypeError, match="real floating scalar"):
        projected_step(evaluation, eta_X=0.1, eta_E=0.1)


def test_project_graph_rejects_nonfinite_proposals_instead_of_replacing_them():
    graph = RelaxedGraph(jnp.full((2, 6), jnp.inf), jnp.full((1, 5), 0.2))
    with pytest.raises(ValueError, match="nonfinite"):
        project_graph(graph)


def test_finite_raw_evaluation_that_overflows_proposal_retains_both_states():
    graph = _interior()

    def loss(g, *, key):
        del key
        return 1e20 * jnp.sum(g.X), {}

    evaluation = evaluate_graph(loss, graph, jax.random.key(0))
    assert np.isfinite(evaluation.value)
    assert np.isfinite(evaluation.gradients.X).all()
    with pytest.raises(NonfiniteEvaluationError) as captured:
        projected_step(evaluation, eta_X=1e20, eta_E=0)
    failure = captured.value
    assert failure.graph is graph
    assert failure.proposal is not None
    assert np.isinf(failure.proposal.X).any()
    assert np.isfinite(failure.graph.X).all()
    assert any("proposal" in path for path in failure.nonfinite_paths)


def test_empty_pair_leaf_projects_and_differentiates_without_special_occupancy_masks():
    graph = graph_from_hard([3], [])

    def loss(g, *, key):
        del key
        return jnp.sum(g.X * jnp.arange(6.0)), {"empty_edges": g.E_upper}

    step = evaluate_and_project(loss, graph, jax.random.key(0), eta_X=0.05, eta_E=0.2)
    validate_graph(step.next_graph)
    assert step.evaluation.gradients.E_upper.shape == (0, 5)
    assert step.next_graph.n_atoms == 1
    assert step.projection_residuals["E_upper"] == 0
