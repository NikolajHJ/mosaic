"""Authenticated continuation of saved, fixed-key graph optimization studies.

An exhausted strict line search is the only accepted failed parent. Its initial
native FD/null gates are inherited with their exact evidence; a boundary graph
is replayed, never moved into the simplex interior to repeat those probes.
"""

from copy import deepcopy
import json
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from .gradient_checks import _save_evaluation, _value
from .gradient_trajectory import run_updates
from .io import file_sha256, save_graph, write_json
from .optimizers import evaluate_graph
from .state import RelaxedGraph, validate_graph


STALL = "AssertionError: No measured loss-decreasing projected step"


def assert_identical_contexts(reference, restored):
    """Require exact decoded arrays AND static maps, independent of ZIP labels.

    save_context assigns array labels in dictionary insertion order. Restoring
    its sorted JSON and saving again can renumber those labels without changing
    a feature. Each container must first pass its own pinned file hashes.
    """
    left, left_tree = jax.tree_util.tree_flatten(reference)
    right, right_tree = jax.tree_util.tree_flatten(restored)
    if left_tree != right_tree:
        raise ValueError("Prepared context structure or static identities changed")
    for expected, actual in zip(left, right, strict=True):
        a, b = np.asarray(expected), np.asarray(actual)
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError("Prepared context array bytes or dtype changed")


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Resume artifact escaped or is missing")
    return path


def validate_parent_report(report):
    """Authenticate eligibility, without treating a failed parent as passed."""
    if report.get("status") != "failed" or report.get("error") != STALL:
        raise ValueError("Only a strict descent stall is eligible for continuation")
    if not report.get("primal_every_step") or not report.get("record_readouts"):
        raise ValueError("Parent needs saved independent primal measurements")
    if {d["field"] for d in report["directions"]} != {"X", "E_upper"}:
        raise ValueError("Missing inherited native derivative checks")
    for direction in report["directions"]:
        agrees = [s["agrees"] for s in direction["sweep"]]
        if not any(a and b for a, b in zip(agrees, agrees[1:])):
            raise ValueError("Parent derivative gate did not pass")
    for value, limit in (
        ("bond_order_null_max_output_difference", "bond_order_null_output_atol"),
        ("bond_order_null_absolute_scalar_difference", "bond_order_null_scalar_atol"),
    ):
        if not np.isfinite(report[value]) or report[value] > report[limit]:
            raise ValueError("Parent null gate did not pass")
    if not report["bond_order_null_same_atom_and_existence_inputs"]:
        raise ValueError("Parent null input identity missing")
    if (
        report["bond_order_null_max_gradient_difference"]
        > report["finite_difference_policy"]["minimum_tangent"]
    ):
        raise ValueError("Parent bond-order derivative null gate failed")
    parity = report["primal_autodiff_parity"]
    if parity["absolute_scalar_difference"] > parity["scalar_atol"]:
        raise ValueError("Parent primal/AD parity failed")
    states = report["states"]
    if not states or [s["step"] for s in states] != list(range(len(states))):
        raise ValueError("Parent trajectory is not a contiguous prefix")
    last = len(states) - 1
    if (
        len(report["updates"]) != last + 1
        or "accepted_step_size" in report["updates"][-1]
    ):
        raise ValueError("Parent does not end in a rejected update")
    for i, update in enumerate(report["updates"][:-1]):
        if update["step"] != i or update["after"] >= update["before"] - 1e-7:
            raise ValueError("Invalid accepted parent prefix")
        if states[i + 1]["primal_loss"] >= states[i]["primal_loss"] - 1e-7:
            raise ValueError("Parent primal prefix did not decrease")
    return last


def continue_case(loss, *, entry, key, output_directory, checks):
    """Restore exact probabilities/key, replay the boundary, then add updates."""
    parent = Path(entry["directory"])
    for name, digest in entry["files"].items():
        if file_sha256(checked_path(parent, name)) != digest:
            raise ValueError("Parent case artifact identity mismatch: " + name)
    original = json.loads((parent / "report.json").read_text())
    start = validate_parent_report(original)
    if start != entry["last_step"] or not start < checks["steps"]:
        raise ValueError("Continuation endpoint changed or already reached")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = deepcopy(original)
    report.pop("error", None)
    report.update(
        status="running",
        acceptance_policy="best_primal",
        requested_updates=checks["steps"],
        updates=report["updates"][:start],
        policy_segments=[
            {"first_update": 1, "last_update": start, "policy": "strict"},
            {
                "first_update": start + 1,
                "last_update": checks["steps"],
                "policy": "best_primal",
            },
        ],
        continuation={
            "parent": str(parent),
            "parent_report_sha256": entry["files"]["report.json"],
            "resumed_from_step": start,
            "inherited_initial_native_checks": True,
            "optimizer_state": "stateless projected gradient; same fixed key",
        },
    )
    for name in entry["files"]:
        target = directory / ("parent_report.json" if name == "report.json" else name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checked_path(parent, name), target)
    try:
        with np.load(parent / "initial.npz", allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["key_data"], jax.random.key_data(key))
        with np.load(
            checked_path(parent, original["states"][-1]["snapshot"]), allow_pickle=False
        ) as arrays:
            graph = RelaxedGraph(
                jnp.asarray(arrays["X"]), jnp.asarray(arrays["E_upper"])
            )
            if (
                np.dtype(graph.X.dtype) != arrays["X"].dtype
                or np.dtype(graph.E_upper.dtype) != arrays["E_upper"].dtype
            ):
                raise ValueError("Resume would change graph precision")
            saved_gradients = (
                arrays["gradient_X"].copy(),
                arrays["gradient_E_upper"].copy(),
            )
        validate_graph(graph)
        evaluation = evaluate_graph(loss, graph, key)
        _save_evaluation(directory / "resume-replay.npz", evaluation, compact=True)
        primal = _value(loss, graph, key, directory / "resume-primal.npz", compact=True)
        previous = original["states"][-1]
        if (
            abs(float(evaluation.value) - previous["autodiff_loss"]) > 2e-5
            or abs(primal - previous["primal_loss"]) > 2e-5
            or abs(primal - float(evaluation.value)) > 2e-5
        ):
            raise ValueError("Resume scalar replay/parity mismatch")
        for fresh, saved in zip(
            (evaluation.gradients.X, evaluation.gradients.E_upper),
            saved_gradients,
            strict=True,
        ):
            np.testing.assert_allclose(fresh, saved, atol=2e-5, rtol=1e-4)
        report["continuation"]["replay"] = {
            "autodiff_loss": float(evaluation.value),
            "primal_loss": primal,
            "scalar_atol": 2e-5,
            "gradient_atol": 2e-5,
            "gradient_rtol": 1e-4,
        }
        evaluation, primal = run_updates(
            loss,
            evaluation,
            key=key,
            directory=directory,
            report=report,
            total_steps=checks["steps"],
            initial_step_size=checks["initial_step_size"],
            max_backtracks=checks["max_backtracks"],
            current_primal=primal,
            compact=True,
            record_readouts=True,
            primal_every_step=True,
            acceptance_policy="best_primal",
            start_step=start,
        )
        save_graph(directory / "final_graph.npz", evaluation.graph)
        _save_evaluation(directory / "final.npz", evaluation, compact=True)
        final = _value(
            loss, evaluation.graph, key, directory / "final_primal.npz", compact=True
        )
        report.update(
            status="passed",
            final_loss=float(evaluation.value),
            final_primal_loss=final,
            primal_descent_minimum_decrease=None,
            completion_scope="requested exploratory updates; no monotonicity or convergence guarantee",
        )
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(directory / "report.json", report)
    return report
