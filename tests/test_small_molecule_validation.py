"""Weight-free receipt/failure contracts; real gates stay in integration tests."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.contracts import PreparedContext
from mosaic.small_molecule.io import load_context, save_context
from mosaic.small_molecule.state import graph_from_hard
from mosaic.small_molecule.validation import _failure, validate_pipeline


def _context():
    features = {
        "ref_element": jnp.zeros((2, 128)),
        "restype": jnp.zeros((2, 32)),
        "atom_to_token_idx": jnp.array([0, 1]),
        "ref_mask": jnp.ones(2),
        "ref_charge": jnp.zeros(2),
        "ref_pos": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "token_bonds": jnp.array([[0.0, 1.0], [1.0, 0.0]]),
        "has_frame": jnp.zeros(2),
        "frame_atom_index": -jnp.ones((2, 3), dtype=jnp.int32),
    }
    return PreparedContext(
        base_features=features,
        native_features=dict(features),
        ligand_atom_indices=(0, 1),
        ligand_token_indices=(0, 1),
        protein_token_indices=(),
        reference_geometry_mask=jnp.zeros(2),
        atom_names=("Q000", "Q001"),
        identities={"input": "synthetic receipt only"},
    )


def test_context_sealing_roundtrip_preserves_exact_native_conformer_bytes(tmp_path):
    context = _context()
    receipt = save_context(tmp_path, context)
    restored = load_context(tmp_path, receipt)
    assert restored.identities == context.identities
    assert restored.ligand_atom_indices == context.ligand_atom_indices
    for kind in ("base_features", "native_features"):
        for name, values in getattr(context, kind).items():
            actual = getattr(restored, kind)[name]
            assert actual.dtype == values.dtype
            np.testing.assert_array_equal(actual, values)


def test_context_tampering_cannot_regenerate_or_replace_native_inputs(tmp_path):
    receipt = save_context(tmp_path, _context())
    with (tmp_path / "prepared_context.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_context(tmp_path, receipt)


def test_failed_prerequisite_writes_failure_and_leaves_all_other_gates_pending(
    tmp_path,
):
    context = _context()
    context = PreparedContext(
        base_features=context.base_features,
        native_features=context.native_features,
        ligand_atom_indices=context.ligand_atom_indices,
        ligand_token_indices=context.ligand_token_indices,
        protein_token_indices=(),
        reference_geometry_mask=jnp.ones(2),
        atom_names=context.atom_names,
        identities=context.identities,
    )
    model = SimpleNamespace(identities={"model": "must not be loaded or called"})
    directory = tmp_path / "failed-validation"
    report = validate_pipeline(
        model,
        context,
        graph_from_hard([0, 0], [1]),
        key=jax.random.key(0),
        output_directory=directory,
        settings={},
    )
    assert report["status"] == "failed"
    assert report["gates"]["A"]["status"] == "failed"
    assert all(report["gates"][letter]["status"] == "pending" for letter in "BCDEFG")
    persisted = json.loads((directory / "validation.json").read_text())
    assert persisted == report
    assert json.loads((directory / "failure.json").read_text())["gate"] == "A"


def test_nonfinite_failure_receipt_keeps_raw_arrays_instead_of_sanitizing(tmp_path):
    from mosaic.small_molecule.optimizers import NonfiniteEvaluationError
    from mosaic.small_molecule.state import RelaxedGraph

    graph = graph_from_hard([0, 1], [1])
    gradients = RelaxedGraph(
        jnp.full_like(graph.X, jnp.nan), jnp.zeros_like(graph.E_upper)
    )
    error = NonfiniteEvaluationError(
        graph=graph,
        value=jnp.array(jnp.inf),
        aux={"distogram_logits": jnp.array([jnp.nan])},
        gradients=gradients,
        key=jax.random.key(17),
        nonfinite_paths=("gradient_X", "value"),
    )
    _failure(tmp_path, error, "D")
    with np.load(tmp_path / "failed_evaluation.npz", allow_pickle=False) as arrays:
        assert np.isnan(arrays["gradient_X"]).all()
        assert np.isinf(arrays["value"])
        assert np.isnan(arrays["aux_0000"]).all()
        np.testing.assert_array_equal(arrays["X"], graph.X)
    receipt = json.loads((tmp_path / "failure.json").read_text())
    assert receipt["raw_artifact"] == "failed_evaluation.npz"
    assert receipt["nonfinite_paths"] == ["gradient_X", "value"]


def test_worker_failure_keeps_existing_trajectory_counter_and_raw_receipt(tmp_path):
    previous = {"update_step": 3, "raw_artifact": "evaluations/failed_000003.npz"}
    (tmp_path / "failure.json").write_text(json.dumps(previous), encoding="utf-8")
    _failure(
        tmp_path, RuntimeError("worker continuation failed"), "fresh-process resume"
    )
    receipt = json.loads((tmp_path / "failure.json").read_text())
    assert receipt["previous_failure_receipt"] == previous
    assert receipt["gate"] == "fresh-process resume"
