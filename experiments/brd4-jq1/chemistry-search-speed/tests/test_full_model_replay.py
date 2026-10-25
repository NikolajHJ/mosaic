"""Allocated-only exact saved-pilot replay contracts.

CPU contracts do not establish a GPU/native pass. Every actual low/high pilot
boundary also executes the same roundtrip canary on its original logit class,
device and snapshot before paired native evaluations can run.
"""

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import full_model_pilot as pilot


def _allocation():
    assert os.environ.get("LSB_JOBID", "").isdigit(), "Numerical fixtures require allocated DTU HPC"
    assert os.environ.get("LSB_QUEUE") in ("hpc", "gpul40s")


def _protenix_logit_type():
    """Use the authenticated original class, never a fixture approximation."""
    source = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/"
                  "protenix-extended-release-03/source")
    manifest_path = source / "source_manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562")
    relative = "scripts/protenix_initialization.py"
    path = source / relative
    manifest = json.loads(manifest_path.read_bytes())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][relative]
    sys.path.insert(0, str(source / "scripts"))
    module = importlib.import_module("protenix_initialization")
    assert Path(module.__file__).resolve() == path.resolve()
    return module.ProtenixReplayLogits


@pytest.fixture
def saved_point(tmp_path):
    _allocation()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry.state import GraphLogits

    atoms = np.arange(150, dtype=np.float32).reshape(25, 6) / np.float32(101)
    bonds = np.arange(1500, dtype=np.float32).reshape(300, 5) / np.float32(1009)
    logits = GraphLogits(jnp.asarray(atoms), jnp.asarray(bonds))
    key = jax.random.key(17)

    def write(value=logits, changes=None):
        graph = value.graph()
        arrays = {"atom_logits": np.asarray(value.atoms), "bond_logits": np.asarray(value.bonds),
                  "X": np.asarray(graph.X), "E_upper": np.asarray(graph.E_upper),
                  "key_data": np.asarray(jax.random.key_data(key))}
        arrays.update(changes or {})
        path = tmp_path / ("snapshot-" + str(len(list(tmp_path.glob("snapshot-*")))) + ".npz")
        np.savez_compressed(path, **arrays)
        placement = {"atom_logits": pilot.replay_leaf_metadata(value.atoms),
                     "bond_logits": pilot.replay_leaf_metadata(value.bonds)}
        return pilot._binding(path), placement

    snapshot, placement = write()
    return logits, key, snapshot, placement, write


def test_original_graph_logits_roundtrip_is_exact_for_all_five_saved_fields(saved_point):
    logits, key, snapshot, placement, _ = saved_point
    restored, report = pilot.restore_saved_logits(snapshot, type(logits), key, placement)
    assert type(restored) is type(logits)
    assert report["status"] == "passed_exact_five_field_replay"
    assert set(report["field_bytes_sha256"]) == {"X", "E_upper", "atom_logits", "bond_logits", "key_data"}
    assert report["probabilities_recomputed_by_original_logit_class"] is True
    assert report["stored_probability_substitution"] is False
    assert report["equality_tolerance"] is None


def test_authenticated_original_protenix_class_roundtrip_and_updated_point(saved_point):
    import jax.numpy as jnp

    logits, key, _, _, write = saved_point
    original = _protenix_logit_type()
    for value in (original(logits.atoms, logits.bonds),
                  original(logits.atoms + jnp.float32(.001), logits.bonds - jnp.float32(.002))):
        snapshot, placement = write(value)
        restored, report = pilot.restore_saved_logits(snapshot, original, key, placement)
        assert type(restored) is original
        assert report["status"] == "passed_exact_five_field_replay"


@pytest.mark.parametrize("field", ("X", "E_upper", "atom_logits", "bond_logits", "key_data"))
def test_changed_saved_field_cannot_pass_exact_replay(saved_point, field):
    import numpy as np

    logits, key, snapshot, placement, write = saved_point
    with np.load(snapshot["path"], allow_pickle=False) as stored:
        changed = stored[field].copy()
    changed.flat[0] += 1
    altered, _ = write(changes={field: changed})
    with pytest.raises((AssertionError, ValueError)):
        pilot.restore_saved_logits(altered, type(logits), key, placement)


@pytest.mark.parametrize("field", ("X", "E_upper", "atom_logits", "bond_logits", "key_data"))
def test_changed_saved_precision_is_not_a_tolerance_or_cast_escape(saved_point, field):
    import numpy as np

    logits, key, snapshot, placement, write = saved_point
    with np.load(snapshot["path"], allow_pickle=False) as stored:
        changed = stored[field].astype(np.uint64 if field == "key_data" else np.float64)
    altered, _ = write(changes={field: changed})
    with pytest.raises(ValueError, match="precision|FP32"):
        pilot.restore_saved_logits(altered, type(logits), key, placement)


@pytest.mark.parametrize("field,value", (
    ("version", "other"), ("platform", "tpu"), ("device_id", True),
    ("process_index", -1), ("sharding_type", "NamedSharding"),
    ("dtype", "float64"), ("shape", [25, 7]), ("weak_type", True),
    ("source_committed", "false"), ("memory_kind", 7),
))
def test_malformed_original_placement_blocks_before_replay(saved_point, field, value):
    logits, key, snapshot, placement, _ = saved_point
    changed = deepcopy(placement)
    changed["atom_logits"][field] = value
    with pytest.raises(ValueError, match="placement is malformed"):
        pilot.restore_saved_logits(snapshot, type(logits), key, changed)


@pytest.mark.parametrize("field", ("atom_logits", "bond_logits"))
def test_unavailable_original_leaf_device_cannot_fall_back_to_default(saved_point, field):
    logits, key, snapshot, placement, _ = saved_point
    changed = deepcopy(placement)
    changed[field]["device_id"] = 999999
    with pytest.raises(ValueError, match="device is unavailable"):
        pilot.restore_saved_logits(snapshot, type(logits), key, changed)


@pytest.mark.parametrize("placement", (None, {}, {"atom_logits": {}},
                                      {"atom_logits": {}, "bond_logits": {}, "default": {}}))
def test_missing_original_leaf_placement_is_not_inferred_from_process_default(saved_point, placement):
    logits, key, snapshot, _, _ = saved_point
    with pytest.raises(ValueError, match="placement is mandatory"):
        pilot.restore_saved_logits(snapshot, type(logits), key, placement)


def test_snapshot_binding_is_authenticated_before_any_array_replay(saved_point):
    logits, key, snapshot, placement, _ = saved_point
    wrong = {**snapshot, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Input file identity differs"):
        pilot.restore_saved_logits(wrong, type(logits), key, placement)


def test_failed_exact_replay_prevents_all_paired_native_evaluations(saved_point, monkeypatch, tmp_path):
    from mosaic.small_molecule.chemistry import trajectory

    logits, key, snapshot, _, _ = saved_point
    reference = tmp_path / "reference.json"
    reference.write_text("{}")
    points = [{"label": label, "snapshot": snapshot, "reference": pilot._binding(reference),
               "logit_class": "mosaic.small_molecule.chemistry.state.GraphLogits"}
              for label in ("early-0", "early-10", "high-0", "high-10")]
    calls = []
    monkeypatch.setattr(trajectory, "evaluate_combined", lambda *a, **k: calls.append(True))
    with pytest.raises(ValueError, match="placement is mandatory"):
        pilot.paired_saved_checks(points, None, None, None, key, tmp_path / "paired", (None, None))
    assert calls == []
    assessment = json.loads((tmp_path / "paired/assessment.json").read_text())
    assert assessment["status"] == "failed" and assessment["points"] == []
