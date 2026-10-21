"""Regression checks for primal/AD evidence, authored for allocated DTU HPC."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule import gradient_checks as checks
from mosaic.small_molecule.state import (
    bond_existence,
    graph_from_hard,
    mix_with_uniform,
)


def _loss(*, leak_order=False):
    """Existence-only scalar; optional output leak must be caught separately."""

    def loss(graph, *, key):
        del key
        value = jnp.sum(graph.X[:, 1]) + 0.5 * jnp.sum(bond_existence(graph))
        output = value + (graph.E_upper[0, 1] if leak_order else 0)
        return value, {"distogram_logits": jnp.full((2, 2, 4), output)}

    return loss


def _shift_only_autodiff_aux(monkeypatch):
    """Simulate distinct AD arithmetic without changing the standalone path."""
    strict_evaluator = checks.evaluate_graph

    def shifted(loss, graph, key, **kwargs):
        evaluation = strict_evaluator(loss, graph, key, **kwargs)
        return replace(
            evaluation,
            aux=evaluation.aux
            | {"distogram_logits": evaluation.aux["distogram_logits"] + 0.001},
        )

    monkeypatch.setattr(checks, "evaluate_graph", shifted)


def _run(loss, directory):
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    return checks.check_graph_gradients(
        loss,
        graph,
        key=jax.random.key(17),
        output_directory=directory,
        steps=1,
    )


def test_null_probe_compares_primal_paths_and_reports_ad_aux_difference(
    monkeypatch, tmp_path
):
    _shift_only_autodiff_aux(monkeypatch)
    report = _run(_loss(), tmp_path / "correct-null")
    assert report["status"] == "passed"
    assert report["primal_autodiff_parity"]["max_logit_difference"] > 2e-5
    assert report["primal_autodiff_parity"]["absolute_scalar_difference"] == 0
    assert report["bond_order_null_max_gradient_difference"] == 0
    assert report["bond_order_null_same_atom_and_existence_inputs"]
    assert report["bond_order_null_max_output_difference"] == 0
    assert report["bond_order_null_output_atol"] == 2e-5
    assert report["final_primal_loss"] < report["initial_primal_loss"] - 1e-7
    with np.load(tmp_path / "correct-null" / "baseline-primal.npz") as baseline:
        with np.load(tmp_path / "correct-null" / "bond-order-null.npz") as null:
            np.testing.assert_array_equal(
                baseline["distogram_logits"], null["distogram_logits"]
            )
        with np.load(tmp_path / "correct-null" / "initial.npz") as autodiff:
            assert (
                np.max(
                    np.abs(autodiff["distogram_logits"] - baseline["distogram_logits"])
                )
                > 2e-5
            )
    assert (tmp_path / "correct-null" / "final_primal.npz").is_file()


def test_true_bond_order_output_leak_is_still_rejected(monkeypatch, tmp_path):
    _shift_only_autodiff_aux(monkeypatch)
    directory = tmp_path / "leaking-null"
    with pytest.raises(
        AssertionError, match="null perturbation changed native outputs"
    ):
        _run(_loss(leak_order=True), directory)
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed"
    assert report["bond_order_null_max_gradient_difference"] == 0
    assert report["bond_order_null_same_atom_and_existence_inputs"]
    assert report["bond_order_null_max_output_difference"] > 2e-5
    assert report["bond_order_null_output_atol"] == 2e-5
    assert (directory / "baseline-primal.npz").is_file()
    assert (directory / "bond-order-null.npz").is_file()


def test_recorded_trajectory_has_one_snapshot_per_accepted_update(tmp_path):
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    report = checks.check_graph_gradients(
        _loss(),
        graph,
        key=jax.random.key(17),
        output_directory=tmp_path / "trace",
        steps=3,
        record_readouts=True,
        primal_every_step=True,
    )
    assert report["requested_updates"] == 3
    assert [state["step"] for state in report["states"]] == [0, 1, 2, 3]
    losses = [state["primal_loss"] for state in report["states"]]
    assert all(b < a - 1e-7 for a, b in zip(losses, losses[1:]))
    for state in report["states"]:
        assert (
            state["readout"]["policy"] == "raw argmax; no repair, no fragment removal"
        )
        with np.load(tmp_path / "trace" / state["snapshot"]) as arrays:
            assert float(arrays["primal_loss"]) == state["primal_loss"]


def test_primal_step_guard_rejects_unconfirmed_ad_improvement(monkeypatch, tmp_path):
    original = checks._value

    def inconsistent_primal(loss, graph, key, path, **kwargs):
        value = original(loss, graph, key, path, **kwargs)
        # Inject a disagreeing candidate-only forward measurement. Initial
        # gradient/FD/null checks retain their true native-equivalent behavior.
        return value + 10 if path.name.startswith("primal-step-") else value

    monkeypatch.setattr(checks, "_value", inconsistent_primal)
    graph = mix_with_uniform(graph_from_hard([0, 1], [1]), 0.5)
    directory = tmp_path / "reject-primal"
    with pytest.raises(
        AssertionError, match="No measured loss-decreasing projected step"
    ):
        checks.check_graph_gradients(
            _loss(),
            graph,
            key=jax.random.key(17),
            output_directory=directory,
            steps=3,
            max_backtracks=2,
            record_readouts=True,
            primal_every_step=True,
        )
    report = json.loads((directory / "report.json").read_text())
    assert report["status"] == "failed"
    assert len(report["states"]) == 1
    assert len(report["updates"][0]["attempts"]) == 2


def test_25_slot_driver_native_check_measures_candidate_primal_and_retains_gates(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples/brd4_25_atoms.py"
    spec = importlib.util.spec_from_file_location("driver25_primal_contract", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    graph = mix_with_uniform(graph_from_hard([0] * 25, [int(j == i + 1) for i in range(25) for j in range(i + 1, 25)]), .5)
    directory = tmp_path / "driver-native"
    report = driver.check_protenix_raw_gradients(_loss(), graph, key=jax.random.key(17), directory=directory)
    assert report["status"] == "passed"
    assert report["primal_every_step"] and report["acceptance_policy"] == "best_primal"
    assert {d["field"] for d in report["directions"]} == {"X", "E_upper"}
    assert all(all(p["agrees"] for p in d["sweep"]) for d in report["directions"])
    assert report["bond_order_null_max_output_difference"] == 0
    update = report["updates"][0]
    assert update["primal_after"] < update["primal_before"]
    assert all(t["primal_loss"] is not None for t in update["attempts"])
    assert (directory / "primal-step-0-trial-0.npz").is_file()
    with np.load(directory / "primal-step-0-trial-0.npz") as saved:
        assert float(saved["loss"]) == update["attempts"][0]["primal_loss"]
