"""Strict graph-only derivatives and detached projected-probability updates.

This module accepts a graph loss callable and has no Protenix/model dependency.
Unlike the existing sequence evaluator it preserves every raw derivative and
rejects nonfinite results before projection, centering or any other transform.
"""

from collections.abc import Callable
from numbers import Number
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.contracts import ProjectedStep, RawEvaluation
from mosaic.small_molecule.state import RelaxedGraph, validate_graph

GraphLoss = Callable[..., tuple[jax.Array, Any]]


class NonfiniteEvaluationError(FloatingPointError):
    """Retain raw failure quantities for the integration owner's artifacts."""

    def __init__(
        self,
        *,
        graph: RelaxedGraph,
        value,
        aux,
        gradients,
        key,
        nonfinite_paths: tuple[str, ...],
        proposal: RelaxedGraph | None = None,
    ):
        self.graph = graph
        self.value = value
        self.aux = aux
        self.gradients = gradients
        self.key = key
        self.nonfinite_paths = nonfinite_paths
        self.proposal = proposal
        super().__init__("nonfinite raw evaluation at " + ", ".join(nonfinite_paths))


def _nonfinite_paths(tree) -> tuple[str, ...]:
    """Inspect numerical output leaves, including arrays inside native results."""
    failures = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if isinstance(leaf, (jax.Array, np.ndarray)):
            # Typed keys may be retained as metadata; np.asarray(key) is not
            # legal and RNG bits are not a numerical model output.
            if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
                continue
            array = np.asarray(leaf)
            if array.dtype.kind not in "biufc":
                continue
        elif isinstance(leaf, Number):
            array = np.asarray(leaf)
        else:
            continue
        if not np.isfinite(array).all():
            failures.append(jax.tree_util.keystr(path))
    return tuple(failures)


def _validate_gradients(graph: RelaxedGraph, gradients: RelaxedGraph) -> None:
    if not isinstance(gradients, RelaxedGraph):
        raise TypeError("graph derivatives must retain the RelaxedGraph structure")
    for name in ("X", "E_upper"):
        leaf = getattr(gradients, name)
        source = getattr(graph, name)
        if leaf is None or leaf.shape != source.shape:
            raise ValueError(f"raw {name} gradient must match its graph leaf")
        if not np.issubdtype(np.asarray(leaf).dtype, np.floating):
            raise TypeError(f"raw {name} gradient must be real floating")


def _validate_value(value) -> None:
    value_array = np.asarray(value)
    if value_array.shape != () or not np.issubdtype(value_array.dtype, np.floating):
        raise TypeError("the graph minimization loss must be a real floating scalar")


def evaluate_graph(
    loss: GraphLoss,
    graph: RelaxedGraph,
    key: jax.Array,
    *,
    atol: float = 1e-6,
) -> RawEvaluation:
    """Evaluate ``loss(graph, key=key)`` and differentiate only its graph.

    The supplied key is used exactly as provided, without splitting, so the
    caller can implement a fixed-key derivative check or checkpoint its next
    unused trajectory key. Model and context captured by ``loss`` are frozen
    arguments, never differentiated variables. Include raw model outputs in
    ``aux`` if their finiteness must be checked alongside loss and gradients.
    """
    initial_failures = _nonfinite_paths({"graph": graph})
    if initial_failures:
        raise NonfiniteEvaluationError(
            graph=graph,
            value=None,
            aux=None,
            gradients=None,
            key=key,
            nonfinite_paths=initial_failures,
        )
    validate_graph(graph, atol=atol)
    # Validates key representation without consuming or reseeding it.
    jax.random.key_data(key)
    (value, aux), gradients = eqx.filter_value_and_grad(loss, has_aux=True)(
        graph, key=key
    )
    _validate_value(value)
    _validate_gradients(graph, gradients)
    failures = _nonfinite_paths({"value": value, "aux": aux, "gradients": gradients})
    if failures:
        raise NonfiniteEvaluationError(
            graph=graph,
            value=value,
            aux=aux,
            gradients=gradients,
            key=key,
            nonfinite_paths=failures,
        )
    return RawEvaluation(
        graph=graph, value=value, aux=aux, gradients=gradients, key=key
    )


def project_graph(graph: RelaxedGraph, *, atol: float = 1e-6) -> RelaxedGraph:
    """Project each leaf row onto the unit simplex outside the model forward."""
    validate_graph(graph, atol=atol, require_simplex=False)
    # Reuse Mosaic's exact Euclidean projection without importing the sequence
    # optimizer (or its scipy extras) during ordinary graph-module collection.
    from mosaic.optimizers import projection_simplex

    projected = RelaxedGraph(
        X=jnp.asarray(projection_simplex(graph.X, z=1), dtype=graph.X.dtype),
        E_upper=jnp.asarray(
            projection_simplex(graph.E_upper, z=1), dtype=graph.E_upper.dtype
        ),
    )
    validate_graph(projected, atol=atol)
    return projected


def projected_step(
    evaluation: RawEvaluation,
    *,
    eta_X: float,
    eta_E: float,
    atol: float = 1e-6,
) -> ProjectedStep:
    """Propose one update while retaining the exact state that was evaluated.

    Raw gradients are applied without centering or clipping. ``evaluation``
    owns the score of its own graph; ``next_graph`` has no score until a new
    evaluation. Projection residuals are the separate Euclidean correction
    norms between the unprojected proposal and the projected next graph.
    """
    for name, value in (("eta_X", eta_X), ("eta_E", eta_E)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    validate_graph(evaluation.graph, atol=atol)
    _validate_value(evaluation.value)
    _validate_gradients(evaluation.graph, evaluation.gradients)
    failures = _nonfinite_paths(
        {
            "value": evaluation.value,
            "aux": evaluation.aux,
            "gradients": evaluation.gradients,
        }
    )
    if failures:
        raise NonfiniteEvaluationError(
            graph=evaluation.graph,
            value=evaluation.value,
            aux=evaluation.aux,
            gradients=evaluation.gradients,
            key=evaluation.key,
            nonfinite_paths=failures,
        )
    proposal = RelaxedGraph(
        X=evaluation.graph.X - eta_X * evaluation.gradients.X,
        E_upper=evaluation.graph.E_upper - eta_E * evaluation.gradients.E_upper,
    )
    proposal_failures = _nonfinite_paths({"proposal": proposal})
    if proposal_failures:
        raise NonfiniteEvaluationError(
            graph=evaluation.graph,
            value=evaluation.value,
            aux=evaluation.aux,
            gradients=evaluation.gradients,
            key=evaluation.key,
            nonfinite_paths=proposal_failures,
            proposal=proposal,
        )
    next_graph = project_graph(proposal, atol=atol)
    projection_residuals = {
        name: float(
            np.linalg.norm(np.asarray(getattr(next_graph, name) - getattr(proposal, name)))
        )
        for name in ("X", "E_upper")
    }
    return ProjectedStep(
        evaluation=evaluation,
        next_graph=next_graph,
        eta_X=float(eta_X),
        eta_E=float(eta_E),
        projection_residuals=projection_residuals,
    )


def evaluate_and_project(
    loss: GraphLoss,
    graph: RelaxedGraph,
    key: jax.Array,
    *,
    eta_X: float,
    eta_E: float,
    atol: float = 1e-6,
) -> ProjectedStep:
    """One strict evaluation and proposal; trajectory/RNG ownership is external."""
    evaluation = evaluate_graph(loss, graph, key, atol=atol)
    return projected_step(evaluation, eta_X=eta_X, eta_E=eta_E, atol=atol)
