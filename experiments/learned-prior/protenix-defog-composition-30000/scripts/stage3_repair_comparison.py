"""Use the historical Mosaic repair loss on a Stage 3 probability graph.

Target discovery is a detached host operation. The loss differentiates through
log(p) on the current occupied neutral graph, using the original categorical
likelihood and failure risks. The caller supplies the same strictly positive
simplex convention to both pilot arms. This adapter adds no floor. Its
derivatives are with respect to probabilities, unlike the historical logit
optimizer. A projected update can leave a previously valid molecule.
"""

from collections import Counter
from importlib.util import find_spec
import sys


def make_repair_evaluator(n_atoms=None):
    """Return a per-trajectory evaluator with history and route_counts."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.core import state

    # The unmodified research chemistry package imports the pre-reorganization
    # name. The current core implements the same occupied six-element graph.
    if find_spec("mosaic.small_molecule.state") is None:
        sys.modules["mosaic.small_molecule.state"] = state
    from mosaic.small_molecule.chemistry import (
        FailureGuidanceConfig, GraphLogits, StructuralRepairConfig,
        failure_aware_repair_loss, mosaic_spec, prepare_failure_guidance,
    )
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.chemistry._reference.quality import graph_identity

    # Match the Protenix structural-repair screen. The probability floor belongs
    # to the detached search's edit ordering and is absent from the live loss.
    search = ProbabilityRepairConfig(max_edits=3, checks_per_depth=32, beam_width=4,
                                     probability_weight=1.0, probability_floor=1e-12)
    progress = ProgressSearchConfig(max_edits=6, checks_per_depth=32, beam_width=4,
                                    probability_weight=1.0, allow_atom_edits=False,
                                    atom_edit_penalty=3.0, max_atom_edits=1)
    structural = StructuralRepairConfig(max_candidates=6, max_bond_operations=1024)
    loss_config = FailureGuidanceConfig()
    history, route_counts = [], Counter()

    def decoded(graph):
        return GraphLogits(jnp.log(graph.X), jnp.log(graph.E_upper)).decoded()

    def prepare(graph, spec):
        return prepare_failure_guidance(decoded(graph), spec, update=len(history),
                                        search=search, progress_search=progress,
                                        structural_repair=structural)

    def evaluate(graph, temperature=1.0, placement="view", gradient="exact"):
        if float(temperature) != 1.0 or placement != "view" or gradient != "exact":
            raise ValueError("The matched repair pilot requires an uncooled graph with exact gradients")
        state.validate_graph(graph)
        x, e = np.asarray(graph.X), np.asarray(graph.E_upper)
        if np.any(x <= 0) or np.any(e <= 0):
            raise ValueError("The shared probability graph must be strictly positive for log(p)")
        if n_atoms is not None and graph.n_atoms != n_atoms:
            raise ValueError("Unexpected number of occupied atom slots for this repair arm")
        spec = mosaic_spec(graph.n_atoms)
        # The host preparation reconstitutes probabilities from logits. Check
        # that rounding there preserved the exact hard graph used for scoring.
        ii, jj = np.triu_indices(graph.n_atoms, 1)
        bonds = np.zeros((graph.n_atoms, graph.n_atoms), np.int64)
        bonds[ii, jj] = bonds[jj, ii] = e.argmax(-1)
        expected = graph_identity(x.argmax(-1) + 1, np.zeros(graph.n_atoms, np.int64), bonds, spec)
        prepared = prepare(graph, spec)
        prep_precision = str(x.dtype)
        prep_seconds = prepared.preparation_seconds
        if prepared.receipt["rows"][0]["raw_sha256"] != expected and x.dtype == np.float32:
            # A close FP32 winner can round into a tie under log then softmax.
            # Recompute only detached witness discovery, leaving the native
            # graph, live loss precision and per-update derivatives intact.
            enable_x64 = getattr(jax, "enable_x64", None)
            if enable_x64 is None:
                from jax.experimental import enable_x64
            with enable_x64():
                high = state.RelaxedGraph(jnp.asarray(x, jnp.float64), jnp.asarray(e, jnp.float64))
                prepared = prepare(high, spec)
            prep_seconds += prepared.preparation_seconds
            prep_precision = "float64_retry"
        receipt = prepared.receipt
        row = receipt["rows"][0]
        if row["raw_sha256"] != expected:
            raise ValueError("Repair preparation and Stage 3 readout selected different argmax graphs")

        def loss(current):
            return failure_aware_repair_loss(decoded(current), prepared, spec, loss_config)

        (value, terms), grad = eqx.filter_value_and_grad(loss, has_aux=True)(graph)
        route = receipt["routes"][str(row["route"])]
        history.append({
            "update": len(history), "route": route, "raw_sha256": expected,
            "prepared_sha256": prepared.identity, "raw_components": row["component_count"],
            "structural_status": row["structural"]["status"],
            "structural_checks": receipt["structural_checks"], "beam_checks": receipt["beam_checks"],
            "additional_checks": receipt["additional_checks"],
            "preparation_seconds": prep_seconds, "preparation_precision": prep_precision,
            "terms": {k: float(v) for k, v in terms.items()},
        })
        route_counts[route] += 1
        return value, grad

    evaluate.history = history
    evaluate.route_counts = route_counts
    return evaluate
