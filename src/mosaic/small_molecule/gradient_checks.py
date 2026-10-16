"""Bounded frozen-model graph checks, sharing one explicit loss throughout.

This is an allocated-HPC engineering experiment, not a molecular design score.
RDKit preparation/readout remains outside differentiation. The only derivative
arguments are the shared graph's atom and unordered-bond probability arrays.
"""

import hashlib
from pathlib import Path

import equinox as eqx
import jax
import numpy as np

from .io import hard_readout, save_graph, write_json
from .optimizers import evaluate_graph, projected_step
from .state import bond_existence, graph_diagnostics, validate_graph


def array_digest(tree):
    """Hash resident numerical bytes for before/after immutability evidence."""
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        if not eqx.is_array(leaf):
            continue
        array = np.asarray(leaf)
        digest.update(str(path).encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_evaluation(path, evaluation, *, compact=False, logits=True):
    arrays = {
        "X": evaluation.graph.X,
        "E_upper": evaluation.graph.E_upper,
        "gradient_X": evaluation.gradients.X,
        "gradient_E_upper": evaluation.gradients.E_upper,
        "loss": evaluation.value,
        "key_data": jax.random.key_data(evaluation.key),
    }
    # Preserve the generated trunk and scalar objective evidence checked by the
    # strict evaluator alongside the raw probability derivatives.
    arrays.update(_artifact_aux(evaluation.aux, compact=compact, logits=logits))
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def _artifact_aux(aux, *, compact=False, logits=True):
    """Compact evidence keeps scalars and selected logits, never trunk tensors.

    This controls serialization only. The strict evaluator and primal helper
    still check every returned tensor for finiteness before any acceptance.
    Graphs, raw derivatives, RNG and losses are retained in both policies.
    """
    return {
        name: value
        for name, value in aux.items()
        if eqx.is_array(value)
        and (not compact or value.ndim == 0 or (logits and name == "distogram_logits"))
    }


def _move(graph, field, row, source, destination, epsilon):
    values = getattr(graph, field)
    moved = values.at[row, source].add(-epsilon).at[row, destination].add(epsilon)
    result = eqx.tree_at(lambda g: getattr(g, field), graph, moved)
    validate_graph(result)
    return result


def _value(loss, graph, key, path, *, return_aux=False, compact=False, logits=True):
    value, aux = loss(graph, key=key)
    # Keep the actual perturbed forward before checking it, including failures.
    arrays = {
        "loss": value,
        "X": graph.X,
        "E_upper": graph.E_upper,
        "key_data": jax.random.key_data(key),
    }
    arrays.update(_artifact_aux(aux, compact=compact, logits=logits))
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    if any(not np.isfinite(np.asarray(v)).all() for v in arrays.values()) or any(
        not np.isfinite(np.asarray(v)).all() for v in aux.values() if eqx.is_array(v)
    ):
        raise FloatingPointError("Nonfinite perturbed native forward")
    return (float(value), aux) if return_aux else float(value)


def _snapshot(directory, step, evaluation, primal_loss):
    """Detached, unrepaired chemistry and exact graph bytes for a time series."""
    snapshots = directory / "snapshots"
    snapshots.mkdir(exist_ok=True)
    path = snapshots / f"step-{step:04d}.npz"
    np.savez_compressed(
        path,
        X=np.asarray(evaluation.graph.X),
        E_upper=np.asarray(evaluation.graph.E_upper),
        gradient_X=np.asarray(evaluation.gradients.X),
        gradient_E_upper=np.asarray(evaluation.gradients.E_upper),
        autodiff_loss=np.asarray(evaluation.value),
        primal_loss=np.asarray(primal_loss),
    )
    return {
        "step": step,
        "autodiff_loss": float(evaluation.value),
        "primal_loss": float(primal_loss),
        "snapshot": path.relative_to(directory).as_posix(),
        "graph": graph_diagnostics(evaluation.graph),
        "readout": hard_readout(evaluation.graph),
    }


def check_graph_gradients(
    loss,
    graph,
    *,
    key,
    output_directory,
    epsilons=(0.01, 0.003, 0.001),
    derivative_atol=0.002,
    derivative_rtol=0.15,
    minimum_tangent=1e-7,
    steps=3,
    initial_step_size=0.01,
    max_backtracks=10,
    record_readouts=False,
    primal_every_step=False,
    artifact_policy="full",
):
    """Require separate atom/bond FD agreement and measured projected descent.

    A maximum-range categorical contrast is chosen deterministically in each
    leaf from its initial raw gradient. All epsilon probes use the SAME loss,
    context and RNG key. At least two adjacent predeclared epsilons must agree;
    failures are retained, never replaced by a favorable scale or a surrogate.
    Backtracking accepts an actual lower frozen-model loss, not a linearized
    estimate. Present-bond-order redistribution must remain a null direction
    under the currently declared existence-only native feature map.
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "directions": [], "updates": []}
    evaluation = None
    try:
        if artifact_policy not in ("full", "compact"):
            raise ValueError("Unknown artifact policy")
        compact = artifact_policy == "compact"
        if type(record_readouts) is not bool or type(primal_every_step) is not bool:
            raise ValueError("Readout and primal-step options must be boolean")
        report.update(
            requested_updates=steps,
            record_readouts=record_readouts,
            primal_every_step=primal_every_step,
            artifact_policy=artifact_policy,
            artifact_scope=(
                "all graphs/raw gradients/scalars/RNG; logits at initial/final/probes; no persisted trunk tensors"
                if compact
                else "all raw returned tensors"
            ),
        )
        if (
            type(steps) is not int
            or type(max_backtracks) is not int
            or steps < 1
            or max_backtracks < 1
            or not np.isfinite(initial_step_size)
            or initial_step_size <= 0
        ):
            raise ValueError("Require positive bounded update settings")
        if (
            len(epsilons) < 2
            or len(set(epsilons)) != len(epsilons)
            or any(not np.isfinite(e) or e <= 0 for e in epsilons)
            or not all(a > b for a, b in zip(epsilons, epsilons[1:]))
        ):
            raise ValueError(
                "Require distinct positive finite epsilons in decreasing order"
            )
        if (
            not np.isfinite(derivative_atol)
            or derivative_atol < 0
            or not np.isfinite(derivative_rtol)
            or not 0 < derivative_rtol < 1
            or not np.isfinite(minimum_tangent)
            or minimum_tangent <= 0
        ):
            raise ValueError(
                "Require finite derivative tolerances and a positive signal floor"
            )
        report["finite_difference_policy"] = {
            "absolute_tolerance": derivative_atol,
            "relative_tolerance": derivative_rtol,
            "minimum_tangent": minimum_tangent,
            "acceptance": (
                "same sign, relative error <= rtol, and absolute+relative "
                "tolerance; two adjacent scales"
            ),
            "relative_error_denominator": "max(abs(FD), abs(autodiff))",
            "epsilons": list(epsilons),
        }
        evaluation = evaluate_graph(loss, graph, key)
        _save_evaluation(directory / "initial.npz", evaluation, compact=compact)
        report["initial_loss"] = float(evaluation.value)
        # Native JITs transformed by reverse-mode AD may compile a different
        # arithmetic schedule from a standalone primal. Diagnose that difference
        # separately; a feature-null probe must compare two identical primal
        # execution paths rather than conflate compilation with input changes.
        primal_value, primal_aux = _value(
            loss,
            graph,
            key,
            directory / "baseline-primal.npz",
            return_aux=True,
            compact=compact,
        )
        report["initial_primal_loss"] = primal_value
        primal_scalar_error = abs(primal_value - float(evaluation.value))
        primal_output_error = float(
            np.max(
                np.abs(
                    np.asarray(primal_aux["distogram_logits"], dtype=np.float64)
                    - np.asarray(evaluation.aux["distogram_logits"], dtype=np.float64)
                )
            )
        )
        report["primal_autodiff_parity"] = {
            "primal_loss": primal_value,
            "autodiff_loss": float(evaluation.value),
            "absolute_scalar_difference": primal_scalar_error,
            "scalar_atol": 2e-5,
            "max_logit_difference": primal_output_error,
            "logit_interpretation": (
                "compilation-path diagnostic; null invariance compares "
                "standalone primal with standalone primal"
            ),
            "raw_artifact": "baseline-primal.npz",
        }
        if not np.isfinite(primal_scalar_error) or primal_scalar_error > 2e-5:
            raise AssertionError("Primal and autodiff scalar losses disagree")
        current_primal = primal_value
        if record_readouts:
            report["states"] = [_snapshot(directory, 0, evaluation, primal_value)]
            write_json(directory / "report.json", report)
        for field in ("X", "E_upper"):
            # Accumulate host diagnostics/contrasts in float64 without changing
            # the model's stored float32 graph or derivatives.
            gradients = np.asarray(
                getattr(evaluation.gradients, field), dtype=np.float64
            )
            ranges = np.ptp(gradients, axis=-1)
            row = int(np.argmax(ranges))
            source = int(np.argmin(gradients[row]))
            destination = int(np.argmax(gradients[row]))
            tangent = float(gradients[row, destination] - gradients[row, source])
            record = {
                "field": field,
                "row": row,
                "source": source,
                "destination": destination,
                "autodiff": tangent,
                "gradient_norm": float(np.linalg.norm(gradients)),
                "sweep": [],
            }
            report["directions"].append(record)
            if not np.isfinite(tangent) or tangent <= minimum_tangent:
                raise AssertionError(f"No nonzero simplex tangent in {field}")
            values = np.asarray(getattr(graph, field))
            if max(epsilons) >= min(values[row, source], values[row, destination]):
                raise ValueError("Predeclared epsilon leaves the simplex interior")
            for index, epsilon in enumerate(epsilons):
                plus = _move(graph, field, row, source, destination, epsilon)
                minus = _move(graph, field, row, source, destination, -epsilon)
                p = _value(
                    loss,
                    plus,
                    key,
                    directory / f"{field}-{index}-plus.npz",
                    compact=compact,
                )
                m = _value(
                    loss,
                    minus,
                    key,
                    directory / f"{field}-{index}-minus.npz",
                    compact=compact,
                )
                finite_difference = (p - m) / (2 * epsilon)
                absolute_error = abs(finite_difference - tangent)
                relative_error = absolute_error / max(
                    abs(finite_difference), abs(tangent)
                )
                same_sign = bool(finite_difference * tangent > 0)
                # Absolute tolerance alone can accept FD=0 for a small nonzero
                # tangent. Require resolved, correctly signed relative
                # agreement too; cancellation is a failed derivative gate.
                agrees = bool(
                    np.isfinite(finite_difference)
                    and same_sign
                    and relative_error <= derivative_rtol
                    and np.isclose(
                        finite_difference,
                        tangent,
                        atol=derivative_atol,
                        rtol=derivative_rtol,
                    )
                )
                record["sweep"].append(
                    {
                        "epsilon": epsilon,
                        "finite_difference": finite_difference,
                        "plus_loss": p,
                        "minus_loss": m,
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "same_sign": same_sign,
                        "agrees": agrees,
                    }
                )
            agrees = [entry["agrees"] for entry in record["sweep"]]
            if not any(a and b for a, b in zip(agrees, agrees[1:])):
                raise AssertionError(f"No adjacent-scale FD agreement for {field}")

        # All four present-bond categories feed the identical existence scalar.
        order_tangent = np.asarray(evaluation.gradients.E_upper)
        order_error = float(
            np.max(np.abs(order_tangent[:, 1:] - order_tangent[:, 1:2]))
        )
        report["bond_order_null_max_gradient_difference"] = order_error
        if order_error > minimum_tangent:
            raise AssertionError(
                "Existence-only features acquired bond-order gradients"
            )
        order = _move(graph, "E_upper", 0, 1, 2, 0.005)
        # The native contract observes X and existence only. Assert the exact
        # shared input identity before attributing any output difference to a
        # dependence on the deliberately absent bond-order feature channel.
        np.testing.assert_array_equal(order.X, graph.X)
        np.testing.assert_array_equal(bond_existence(order), bond_existence(graph))
        report["bond_order_null_same_atom_and_existence_inputs"] = True
        order_value, order_aux = _value(
            loss,
            order,
            key,
            directory / "bond-order-null.npz",
            return_aux=True,
            compact=compact,
        )
        order_output_error = float(
            np.max(
                np.abs(
                    np.asarray(order_aux["distogram_logits"], dtype=np.float64)
                    - np.asarray(primal_aux["distogram_logits"], dtype=np.float64)
                )
            )
        )
        report["bond_order_null_max_output_difference"] = order_output_error
        report["bond_order_null_output_atol"] = 2e-5
        report["bond_order_null_comparison"] = (
            "standalone primal versus standalone primal"
        )
        if not np.isfinite(order_output_error) or order_output_error > 2e-5:
            raise AssertionError("Bond-order null perturbation changed native outputs")
        if not np.isfinite(float(order_value)):
            raise FloatingPointError("Nonfinite bond-order null loss")
        order_scalar_error = abs(order_value - primal_value)
        report["bond_order_null_absolute_scalar_difference"] = order_scalar_error
        report["bond_order_null_scalar_atol"] = 2e-5
        if order_scalar_error > 2e-5:
            raise AssertionError("Bond-order null perturbation changed scalar loss")

        for step in range(steps):
            attempts = []
            record = {
                "step": step,
                "before": float(evaluation.value),
                "attempts": attempts,
            }
            report["updates"].append(record)
            for backtrack in range(max_backtracks):
                rate = initial_step_size * 0.5**backtrack
                proposal = projected_step(evaluation, eta_X=rate, eta_E=rate)
                candidate = evaluate_graph(loss, proposal.next_graph, key)
                _save_evaluation(
                    directory / f"step-{step}-trial-{backtrack}.npz",
                    candidate,
                    compact=compact,
                    logits=not compact,
                )
                after = float(candidate.value)
                candidate_primal = None
                if primal_every_step:
                    candidate_primal = _value(
                        loss,
                        candidate.graph,
                        key,
                        directory / f"primal-step-{step}-trial-{backtrack}.npz",
                        compact=compact,
                        logits=not compact,
                    )
                attempts.append(
                    {
                        "step_size": rate,
                        "loss": after,
                        "projection_residuals": proposal.projection_residuals,
                        "primal_loss": candidate_primal,
                    }
                )
                if after < float(evaluation.value) - 1e-7 and (
                    not primal_every_step or candidate_primal < current_primal - 1e-7
                ):
                    record.update(
                        after=after,
                        accepted_step_size=rate,
                        delta_X=float(
                            np.linalg.norm(
                                np.asarray(candidate.graph.X - evaluation.graph.X)
                            )
                        ),
                        delta_E_upper=float(
                            np.linalg.norm(
                                np.asarray(
                                    candidate.graph.E_upper - evaluation.graph.E_upper
                                )
                            )
                        ),
                    )
                    evaluation = candidate
                    if primal_every_step:
                        current_primal = candidate_primal
                    elif record_readouts:
                        current_primal = _value(
                            loss,
                            candidate.graph,
                            key,
                            directory / f"primal-step-{step}-accepted.npz",
                            compact=compact,
                            logits=not compact,
                        )
                    if record_readouts:
                        report["states"].append(
                            _snapshot(directory, step + 1, evaluation, current_primal)
                        )
                        write_json(directory / "report.json", report)
                    print(
                        f"accepted update {step + 1}/{steps}: loss={after:.9g}",
                        flush=True,
                    )
                    break
            else:
                raise AssertionError("No measured loss-decreasing projected step")
        save_graph(directory / "final_graph.npz", evaluation.graph)
        _save_evaluation(directory / "final.npz", evaluation, compact=compact)
        final_primal_value = _value(
            loss, evaluation.graph, key, directory / "final_primal.npz", compact=compact
        )
        report["final_primal_loss"] = final_primal_value
        report["primal_descent_minimum_decrease"] = 1e-7
        if final_primal_value >= primal_value - 1e-7:
            raise AssertionError("No measured standalone-primal loss decrease")
        report.update(status="passed", final_loss=float(evaluation.value))
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        # Retain strict-evaluator evidence even if a nonfinite input/proposal or
        # generated output fails before usable derivatives exist.
        if hasattr(error, "nonfinite_paths"):
            arrays = {
                "X": error.graph.X,
                "E_upper": error.graph.E_upper,
                "key_data": jax.random.key_data(error.key),
            }
            if error.value is not None:
                arrays["loss"] = error.value
            if error.gradients is not None:
                arrays.update(
                    gradient_X=error.gradients.X,
                    gradient_E_upper=error.gradients.E_upper,
                )
            if error.proposal is not None:
                arrays.update(
                    proposal_X=error.proposal.X,
                    proposal_E_upper=error.proposal.E_upper,
                )
            aux_paths = {}
            for path, leaf in jax.tree_util.tree_leaves_with_path(error.aux):
                if eqx.is_array(leaf):
                    name = f"aux_{len(aux_paths):04d}"
                    arrays[name] = leaf
                    aux_paths[name] = jax.tree_util.keystr(path)
            np.savez_compressed(
                directory / "nonfinite_failure.npz",
                **{name: np.asarray(value) for name, value in arrays.items()},
            )
            report.update(
                nonfinite_paths=error.nonfinite_paths,
                failure_aux_paths=aux_paths,
                failure_artifact="nonfinite_failure.npz",
                key_impl=str(jax.random.key_impl(error.key)),
            )
        raise
    finally:
        write_json(directory / "report.json", report)
    return report
