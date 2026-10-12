"""Context/artifact boundary checks authored for later allocated HPC execution."""

import json

import equinox as eqx
import jax
import numpy as np
import pytest

from mosaic.small_molecule.contracts import validate_context
from mosaic.small_molecule.io import load_context, save_context
from mosaic.small_molecule.features import inject_graph, NATIVE_ELEMENT_INDICES
from mosaic.small_molecule.state import graph_from_hard
from test_small_molecule_features import synthetic_context


def test_context_roundtrip_restores_both_native_and_geometry_off_inputs(tmp_path):
    context, _ = synthetic_context()
    identity = save_context(tmp_path, context)
    restored = load_context(tmp_path, identity)
    assert restored.identities == context.identities
    assert restored.ligand_atom_indices == context.ligand_atom_indices
    assert restored.ligand_token_indices == context.ligand_token_indices
    assert restored.protein_token_indices == context.protein_token_indices
    assert restored.atom_names == context.atom_names
    for expected, actual in zip(
        jax.tree_util.tree_leaves(context),
        jax.tree_util.tree_leaves(restored),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype


def test_context_tampering_is_rejected_before_restore(tmp_path):
    context, _ = synthetic_context()
    identity = save_context(tmp_path, context)
    path = tmp_path / "prepared_context.json"
    metadata = json.loads(path.read_text())
    metadata["geometry_mode"] = "on"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity mismatch"):
        load_context(tmp_path, identity)


@pytest.mark.parametrize(
    "field", ["ref_mask", "atom_pad_mask", "token_pad_mask", "ref_charge"]
)
def test_ligand_occupancy_and_charge_are_guards_independent_of_producer(field):
    context, _ = synthetic_context()
    index = (
        context.ligand_token_indices[0]
        if field == "token_pad_mask"
        else context.ligand_atom_indices[0]
    )
    changed = eqx.tree_at(
        lambda value: value.base_features[field],
        context,
        context.base_features[field].at[index].set(1 if field == "ref_charge" else 0),
    )
    with pytest.raises(ValueError):
        validate_context(changed)


def test_ligand_geometry_mask_cannot_disable_native_protein_geometry():
    context, _ = synthetic_context()
    changed = eqx.tree_at(
        lambda value: value.reference_geometry_mask,
        context,
        context.reference_geometry_mask.at[1].set(0),
    )
    with pytest.raises(ValueError, match="protein reference geometry"):
        validate_context(changed)


def test_duplicate_or_overlapping_token_maps_are_rejected():
    context, _ = synthetic_context()
    changed = type(context)(
        base_features=context.base_features,
        ligand_atom_indices=context.ligand_atom_indices,
        ligand_token_indices=(3, 1, 1),
        protein_token_indices=context.protein_token_indices,
        reference_geometry_mask=context.reference_geometry_mask,
        atom_names=context.atom_names,
    )
    with pytest.raises(ValueError, match="duplicated"):
        validate_context(changed)


@pytest.mark.parametrize("element", range(6))
def test_all_element_categories_share_native_float_injection(element):
    context, _ = synthetic_context()
    graph = graph_from_hard([element] * 3, [0, 1, 4])
    injected = np.asarray(inject_graph(context, graph)["ref_element"])[
        list(context.ligand_atom_indices)
    ]
    expected = np.zeros((3, 128), dtype=np.float32)
    expected[:, NATIVE_ELEMENT_INDICES[element]] = 1
    np.testing.assert_array_equal(injected, expected)
