"""Authored engineering checks. Execute only on allocated DTU HPC."""

import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.io import TrajectoryConfig, hard_readout, run_trajectory
from mosaic.small_molecule.optimizers import NonfiniteEvaluationError
from mosaic.small_molecule.state import graph_from_hard, mix_with_uniform


def initial_graph():
    return mix_with_uniform(graph_from_hard([0, 1], [1]), 0.4)


def loss(graph, *, key):
    # RNG influences gradients so a missing next-key receipt changes replay.
    noise = jax.random.uniform(key, graph.X.shape, minval=0.1, maxval=0.9)
    return jnp.sum(graph.X * noise) + jnp.sum(graph.E_upper**2), {
        "raw": graph.X + noise
    }


def rows(directory):
    return [
        json.loads(line)
        for line in (directory / "trace.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("policy", ["fixed", "split"])
def test_checkpoint_resume_matches_and_final_state_is_evaluated(tmp_path, policy):
    config = TrajectoryConfig(steps=5, eta_X=0.1, eta_E=0.03, rng_policy=policy)
    key = jax.random.key(17)
    whole = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "whole",
        config=config,
        identities={"source": "fixture-source", "context": "fixture-context"},
    )
    pause = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "split",
        config=config,
        identities={"source": "fixture-source", "context": "fixture-context"},
        max_updates=2,
    )
    assert not pause.complete and pause.update_step == 2
    resumed = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "split",
        config=config,
        identities={"source": "fixture-source", "context": "fixture-context"},
        resume=True,
    )
    np.testing.assert_allclose(whole.graph.X, resumed.graph.X, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        whole.graph.E_upper, resumed.graph.E_upper, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        jax.random.key_data(whole.next_key), jax.random.key_data(resumed.next_key)
    )
    assert rows(tmp_path / "whole") == rows(tmp_path / "split")
    assert len(rows(tmp_path / "whole")) == 6
    assert rows(tmp_path / "whole")[-1]["evaluated_step"] == 5
    assert rows(tmp_path / "whole")[-1]["final"]
    with np.load(tmp_path / "whole" / "evaluations" / "step_000005.npz") as arrays:
        np.testing.assert_array_equal(arrays["X"], whole.graph.X)
        np.testing.assert_array_equal(arrays["E_upper"], whole.graph.E_upper)
    # Immutable step+evaluation filenames include both pre-final and final commits.
    assert len(list((tmp_path / "whole" / "checkpoints").glob("*.npz"))) == 7


def test_fresh_process_resume(tmp_path):
    config = TrajectoryConfig(steps=3, rng_policy="split")
    key = jax.random.key(17)
    whole = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "whole",
        config=config,
        identities={},
    )
    run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "split",
        config=config,
        identities={},
        max_updates=1,
    )
    script = """
from pathlib import Path
import sys
import jax
from test_small_molecule_io import loss, initial_graph
from mosaic.small_molecule.io import run_trajectory, TrajectoryConfig
run_trajectory(loss, initial_graph(), jax.random.key(17), directory=Path(sys.argv[1]),
               config=TrajectoryConfig(steps=3, rng_policy='split'), identities={}, resume=True)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), environment.get("PYTHONPATH", "")]
    )
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "split")],
        check=True,
        env=environment,
        timeout=180,
    )
    receipt = json.loads((tmp_path / "split" / "latest_checkpoint.json").read_text())
    with np.load(tmp_path / "split" / receipt["path"]) as arrays:
        np.testing.assert_allclose(arrays["X"], whole.graph.X, atol=1e-6, rtol=1e-6)
    assert rows(tmp_path / "split") == rows(tmp_path / "whole")


@pytest.mark.parametrize("change", ["config", "context", "graph", "key"])
def test_resume_rejects_incompatible_inputs(tmp_path, change):
    config = TrajectoryConfig(steps=2)
    graph = initial_graph()
    key = jax.random.key(17)
    identities = {"context": "one"}
    run_trajectory(
        loss,
        graph,
        key,
        directory=tmp_path,
        config=config,
        identities=identities,
        max_updates=1,
    )
    if change == "config":
        config = TrajectoryConfig(steps=3)
    elif change == "context":
        identities = {"context": "two"}
    elif change == "graph":
        graph = mix_with_uniform(graph, 0.1)
    else:
        key = jax.random.key(18)
    with pytest.raises(ValueError, match="identity mismatch"):
        run_trajectory(
            loss,
            graph,
            key,
            directory=tmp_path,
            config=config,
            identities=identities,
            resume=True,
        )


def test_nonfinite_failure_retains_raw_arrays_and_disables_resume(tmp_path):
    def invalid(graph, *, key):
        return jnp.sum(graph.X), {"raw": jnp.array([jnp.nan, jnp.inf])}

    with pytest.raises(NonfiniteEvaluationError):
        run_trajectory(
            invalid,
            initial_graph(),
            jax.random.key(0),
            directory=tmp_path,
            config=TrajectoryConfig(),
            identities={},
        )
    receipt = json.loads((tmp_path / "failure.json").read_text())
    with np.load(tmp_path / receipt["raw_artifact"]) as arrays:
        assert any(
            np.any(~np.isfinite(arrays[name]))
            for name in arrays.files
            if arrays[name].dtype.kind == "f"
        )
        assert "gradient_X" in arrays
    with pytest.raises(ValueError, match="failed trajectory"):
        run_trajectory(
            loss,
            initial_graph(),
            jax.random.key(0),
            directory=tmp_path,
            config=TrajectoryConfig(),
            identities={},
            resume=True,
        )


def test_readout_failure_does_not_change_continuation(tmp_path):
    def broken(graph):
        raise RuntimeError("deliberate detached conversion failure")

    config = TrajectoryConfig(steps=2)
    key = jax.random.key(0)
    one = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "normal",
        config=config,
        identities={},
    )
    two = run_trajectory(
        loss,
        initial_graph(),
        key,
        directory=tmp_path / "broken",
        config=config,
        identities={},
        readout=broken,
    )
    np.testing.assert_array_equal(one.graph.X, two.graph.X)
    np.testing.assert_array_equal(one.graph.E_upper, two.graph.E_upper)
    assert (
        "deliberate"
        in json.loads(
            (tmp_path / "broken" / "readouts" / "step_000000.json").read_text()
        )["conversion_error"]
    )


def test_raw_readout_keeps_disconnected_and_invalid_assignments():
    disconnected = hard_readout(graph_from_hard([0, 0], [0]))
    assert disconnected["bond_upper_indices"] == [0]
    assert disconnected["disconnected"] is True
    # Neutral fluorine with a triple bond is deliberately not repaired.
    invalid = hard_readout(graph_from_hard([4, 4], [3]))
    assert invalid["bond_upper_indices"] == [3]
    assert not invalid["sanitized"] and invalid["canonical_smiles"] is None
    assert invalid["conversion_error"]


def test_trace_orphan_is_ignored_after_committed_prefix(tmp_path):
    config = TrajectoryConfig(steps=2)
    run_trajectory(
        loss,
        initial_graph(),
        jax.random.key(0),
        directory=tmp_path,
        config=config,
        identities={},
        max_updates=1,
    )
    with (tmp_path / "trace.jsonl").open("a") as handle:
        handle.write('{"uncommitted":true}\n')
    run_trajectory(
        loss,
        initial_graph(),
        jax.random.key(0),
        directory=tmp_path,
        config=config,
        identities={},
        resume=True,
    )
    assert len(rows(tmp_path)) == 3
    assert not any("uncommitted" in row for row in rows(tmp_path))


def test_artifact_write_failure_retains_receipt(tmp_path, monkeypatch):
    from mosaic.small_molecule import io

    original = io._write_npz

    def fail_evaluation(path, arrays):
        if path.parent.name == "evaluations":
            raise OSError("deliberate storage failure")
        return original(path, arrays)

    monkeypatch.setattr(io, "_write_npz", fail_evaluation)
    with pytest.raises(OSError, match="storage failure"):
        run_trajectory(
            loss,
            initial_graph(),
            jax.random.key(0),
            directory=tmp_path,
            config=TrajectoryConfig(),
            identities={},
        )
    assert json.loads((tmp_path / "failure.json").read_text())["type"] == "OSError"
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "failed"


def test_sparse_readout_schedule_always_includes_final_state(tmp_path):
    run_trajectory(
        loss,
        initial_graph(),
        jax.random.key(0),
        directory=tmp_path,
        config=TrajectoryConfig(steps=3, readout_every=2),
        identities={},
    )
    assert sorted(path.name for path in (tmp_path / "readouts").glob("*.json")) == [
        "step_000000.json",
        "step_000002.json",
        "step_000003.json",
    ]
    assert rows(tmp_path)[1]["readout"] is None
