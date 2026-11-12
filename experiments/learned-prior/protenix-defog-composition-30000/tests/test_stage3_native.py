"""Archived context restoration without loading either pretrained model."""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.core.contracts import contract_manifest

SPEC = importlib.util.spec_from_file_location("stage3_native", Path(__file__).parents[1] / "scripts/stage3_native.py")
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def context_fixture(path, dtype=np.float32):
    np.savez_compressed(path / "prepared_context.npz",
        restype=np.zeros((2, 32), dtype=dtype), atom_to_token=np.array([0, 1], np.int32),
        mask=np.ones(2, dtype=dtype), geometry=np.array([0, 1], dtype=dtype))
    base = {"dict": {
        "restype": {"array": "restype"}, "atom_to_token_idx": {"array": "atom_to_token"},
        "atom_pad_mask": {"array": "mask"}, "token_pad_mask": {"array": "mask"},
        "ref_mask": {"array": "mask"},
    }}
    metadata = {
        "contract": contract_manifest(), "base_features": base,
        "native_features": {"scalar": None}, "reference_geometry_mask": {"array": "geometry"},
        "ligand_atom_indices": [0], "ligand_token_indices": [0], "protein_token_indices": [1],
        "atom_names": ["C1"], "geometry_mode": "off", "identities": {"fixture": "test_only"},
    }
    (path / "prepared_context.json").write_text(json.dumps(metadata))
    return {name: native.file_sha(path / name) for name in ("prepared_context.json", "prepared_context.npz")}


def test_safe_archived_context_codec_restores_identical_arrays(tmp_path):
    expected = context_fixture(tmp_path)
    context = native.load_prepared_context(tmp_path, expected)
    assert context.n_atoms == 1 and context.n_tokens == 2
    assert context.ligand_token_indices == (0,)
    np.testing.assert_array_equal(context.reference_geometry_mask, [0, 1])
    before = native.array_digest(context)
    assert native.array_digest(context) == before


def test_context_hash_and_contract_mismatch_fail_before_scoring(tmp_path):
    expected = context_fixture(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        native.load_prepared_context(tmp_path, expected | {"prepared_context.json": "wrong"})
    path = tmp_path / "prepared_context.json"
    data = json.loads(path.read_text())
    data["contract"]["formal_charge"] = 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="contract"):
        native.load_prepared_context(tmp_path, expected | {"prepared_context.json": native.file_sha(path)})


def test_context_does_not_silently_downcast_saved_precision(tmp_path):
    expected = context_fixture(tmp_path, np.float64)
    import jax
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="precision"):
            native.load_prepared_context(tmp_path, expected)
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_numerical_digest_detects_changed_resident_arrays():
    first = {"frozen": jnp.array([1., 2.])}
    changed = {"frozen": jnp.array([1., 3.])}
    assert native.array_digest(first) != native.array_digest(changed)
