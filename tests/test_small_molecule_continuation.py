"""Allocated CPU tests for explicit fallback selection and authenticated resume."""

from copy import deepcopy
import json

import jax
import numpy as np
import pytest

from mosaic.small_molecule import gradient_checks as checks
from mosaic.small_molecule.gradient_resume import (
    STALL,
    assert_identical_contexts,
    continue_case,
    validate_parent_report,
)
from mosaic.small_molecule.gradient_trajectory import run_updates
from mosaic.small_molecule.io import file_sha256, write_json
from mosaic.small_molecule.optimizers import evaluate_graph
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform
from test_small_molecule_gradient_checks import _loss


def test_context_container_renumbering_keeps_exact_decoded_identity(tmp_path):
    from mosaic.small_molecule.io import load_context, save_context
    from test_small_molecule_features import synthetic_context

    context, _ = synthetic_context()
    first = save_context(tmp_path / "first", context)
    restored = load_context(tmp_path / "first", first)
    second = save_context(tmp_path / "second", restored)
    assert_identical_contexts(context, load_context(tmp_path / "second", second))


def test_context_comparison_rejects_array_and_static_metadata_changes():
    from dataclasses import replace
    import equinox as eqx
    from test_small_molecule_features import synthetic_context

    context, _ = synthetic_context()
    altered = eqx.tree_at(
        lambda c: c.reference_geometry_mask,
        context,
        context.reference_geometry_mask.at[0].add(1),
    )
    with pytest.raises(ValueError):
        assert_identical_contexts(context, altered)
    with pytest.raises(ValueError):
        assert_identical_contexts(
            context, replace(context, atom_names=tuple(reversed(context.atom_names)))
        )


@pytest.mark.parametrize(
    "scores,selected,increased",
    [([2.0, 1.0, 3.0], 1, True), ([0.9 - 5e-8] * 3, 0, False)],
)
def test_fallback_selects_lowest_independent_primal_including_increases(
    tmp_path, monkeypatch, scores, selected, increased
):
    loss = _loss()
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    evaluation = evaluate_graph(loss, graph, jax.random.key(17))
    original = checks._value
    measured = iter(scores)

    def primal(*args, **kwargs):
        original(*args, **kwargs)
        return next(measured)

    monkeypatch.setattr(checks, "_value", primal)
    report = {"updates": [], "states": []}
    _, value = run_updates(
        loss,
        evaluation,
        key=jax.random.key(17),
        directory=tmp_path,
        report=report,
        total_steps=1,
        initial_step_size=0.03,
        max_backtracks=3,
        current_primal=0.9,
        compact=True,
        record_readouts=True,
        primal_every_step=True,
        acceptance_policy="best_primal",
    )
    update = report["updates"][0]
    assert update["selected_trial"] == selected
    assert update["fallback_used"] and len(update["attempts"]) == 3
    assert update["primal_increased"] is increased
    assert value == min(scores)
    assert report["states"][0]["primal_loss"] == value


def test_fallback_still_rejects_nonfinite_primal(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, "_value", lambda *a, **k: float("nan"))
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        run_updates(
            _loss(),
            evaluate_graph(_loss(), graph, jax.random.key(17)),
            key=jax.random.key(17),
            directory=tmp_path,
            report={"updates": []},
            total_steps=1,
            initial_step_size=0.03,
            max_backtracks=3,
            current_primal=0.0,
            compact=True,
            record_readouts=True,
            primal_every_step=True,
            acceptance_policy="best_primal",
        )


def _parent(tmp_path):
    parent = tmp_path / "parent"
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    d = checks.check_graph_gradients(
        _loss(),
        graph,
        key=jax.random.key(17),
        output_directory=parent,
        steps=1,
        record_readouts=True,
        primal_every_step=True,
    )
    # Engineering fixture: preserve a real evaluated prefix and attach the
    # declared strict-stall termination. This is not a native experiment.
    d.update(status="failed", error=STALL, requested_updates=3)
    d["updates"].append({"step": 1, "before": d["final_loss"], "attempts": [{}] * 10})
    write_json(parent / "report.json", d)
    files = {
        p.relative_to(parent).as_posix(): file_sha256(p)
        for p in parent.rglob("*")
        if p.is_file()
    }
    return {"directory": str(parent), "last_step": 1, "files": files}, d


def test_resume_keeps_prefix_and_matches_uninterrupted_graph(tmp_path):
    entry, parent = _parent(tmp_path)
    old_bytes = (tmp_path / "parent/report.json").read_bytes()
    out = tmp_path / "continuation"
    report = continue_case(
        _loss(),
        entry=entry,
        key=jax.random.key(17),
        output_directory=out,
        checks={"steps": 3, "initial_step_size": 0.01, "max_backtracks": 10},
    )
    assert report["status"] == "passed"
    assert [s["step"] for s in report["states"]] == [0, 1, 2, 3]
    # Compare persisted metadata: JSON encodes the readout's tuple pairs as
    # lists. The saved prefix, rather than its pre-serialization container
    # types, is the continuation contract.
    saved_parent = json.loads((tmp_path / "parent/report.json").read_text())
    assert report["states"][:2] == saved_parent["states"]
    assert report["continuation"]["resumed_from_step"] == 1
    assert (tmp_path / "parent/report.json").read_bytes() == old_bytes
    with (
        np.load(out / "snapshots/step-0001.npz") as resumed,
        np.load(tmp_path / "parent/snapshots/step-0001.npz") as original,
    ):
        np.testing.assert_array_equal(resumed["X"], original["X"])
    full = checks.check_graph_gradients(
        _loss(),
        mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5),
        key=jax.random.key(17),
        output_directory=tmp_path / "full",
        steps=3,
        record_readouts=True,
        primal_every_step=True,
        acceptance_policy="best_primal",
    )
    assert report["final_loss"] == full["final_loss"]


@pytest.mark.parametrize("change", ["hash", "gap", "gate", "other_failure", "key"])
def test_resume_rejects_changed_inputs_and_failed_native_gates(tmp_path, change):
    entry, d = _parent(tmp_path)
    if change == "hash":
        (tmp_path / "parent/report.json").write_text("{}")
    elif change == "key":
        pass
    else:
        altered = deepcopy(d)
        if change == "gap":
            altered["states"][-1]["step"] = 5
        if change == "gate":
            altered["directions"][0]["sweep"] = [{"agrees": False}] * 3
        if change == "other_failure":
            altered["error"] = "FloatingPointError: nonfinite"
        with pytest.raises(ValueError):
            validate_parent_report(altered)
        return
    with pytest.raises((ValueError, AssertionError)):
        continue_case(
            _loss(),
            entry=entry,
            key=jax.random.key(99 if change == "key" else 17),
            output_directory=tmp_path / "bad",
            checks={"steps": 3, "initial_step_size": 0.01, "max_backtracks": 10},
        )
