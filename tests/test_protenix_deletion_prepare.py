"""Small feature-axis fixtures; run numerically only on allocated DTU HPC.

These do not import native Protenix, read molecules or load model weights.
Actual original/compact/ordinary endpoint preparation remains a separate gate.
"""

from copy import deepcopy

import pytest
import torch

from latentmol.protenix_occupancy.deletion_prepare import (
    FEATURE_AXES, _select_axes, audit_feature_schema, remap_pairs,
    native_file_atom_names, validate_hydrogen_replacement,
)


def test_native_file_atom_names_preserve_mixed_element_file_order():
    assert native_file_atom_names(["C", "N", "C", "Br", "O", "C"]) == [
        "C1", "N1", "C2", "BR1", "O1", "C3"]


def test_hydrogen_replacement_accepts_exact_halogen_to_h_endpoint():
    validate_hydrogen_replacement({"C1": 0, "C2": 3, "I1": 0},
        {"C1": 1, "C2": 3}, "I1", "C1")


@pytest.mark.parametrize("deleted", [
    {"C1": 0, "C2": 3}, {"C1": 2, "C2": 3}, {"C1": 1, "C2": 2},
    {"C1": 1}, {"C1": 1, "C2": 3, "I1": 0}, {"C1": 1, "C2": 3, "C3": 0},
    {"C1": 1., "C2": 3}, {"C1": True, "C2": 3}, {"C1": 1, "C2": -1},
])
def test_hydrogen_replacement_rejects_wrong_counts_or_changed_atom_set(deleted):
    with pytest.raises(ValueError):
        validate_hydrogen_replacement({"C1": 0, "C2": 3, "I1": 0}, deleted, "I1", "C1")


def feature_fixture(n_atom=4, n_token=3):
    sizes = {"A": n_atom, "T": n_token}
    features = {key: torch.zeros(tuple(sizes.get(axis, axis) for axis in axes))
                for key, axes in FEATURE_AXES.items()}
    features["constraint_feature"] = {}
    features["token_index"] = torch.arange(n_token)
    features["atom_to_token_idx"] = torch.tensor([0, 0, 1, 2])
    features["frame_atom_index"] = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 2, 3]])
    features["has_frame"] = torch.ones(n_token, dtype=torch.long)
    features["distogram_rep_atom_mask"] = torch.tensor([1, 0, 1, 1])
    return features


def test_closed_schema_accepts_complete_small_raw_native_features():
    report = audit_feature_schema(feature_fixture())
    assert report["atom_count"] == 4
    assert report["token_count"] == 3
    assert report["unknown_fields"] == []
    assert set(report["fields"]) == set(FEATURE_AXES)


@pytest.mark.parametrize("name", ["relp", "d_lm", "v_lm", "pad_info", "new_native_field"])
def test_schema_rejects_unknown_fields_and_precomputed_native_caches(name):
    features = feature_fixture()
    features[name] = torch.tensor(0)
    with pytest.raises(ValueError, match="Unsupported native feature fields"):
        audit_feature_schema(features)


@pytest.mark.parametrize("key", sorted(FEATURE_AXES))
def test_schema_requires_every_declared_axis_field(key):
    features = feature_fixture()
    del features[key]
    with pytest.raises(ValueError, match="Unsupported native feature fields"):
        audit_feature_schema(features)


@pytest.mark.parametrize("key", ["ref_element", "ref_pos", "token_bonds", "msa", "bond_mask",
                                "template_all_atom_positions", "template_restype"])
def test_schema_rejects_wrong_dimensions_without_guessing_axes(key):
    features = feature_fixture()
    features[key] = features[key].unsqueeze(0)
    with pytest.raises(ValueError, match="shape/device/graph"):
        audit_feature_schema(features)


@pytest.mark.parametrize("key", ["ref_pos", "ref_element", "token_bonds", "msa"])
def test_schema_rejects_nonfinite_values(key):
    features = feature_fixture()
    features[key].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_feature_schema(features)


def test_schema_rejects_live_autograd_input():
    features = feature_fixture()
    features["ref_element"].requires_grad_()
    with pytest.raises(ValueError, match="shape/device/graph"):
        audit_feature_schema(features)


@pytest.mark.parametrize("bad", [-1, 3])
def test_schema_rejects_outside_atom_to_token_indices(bad):
    features = feature_fixture()
    features["atom_to_token_idx"][0] = bad
    with pytest.raises(ValueError, match="Atom-to-token"):
        audit_feature_schema(features)


def test_schema_requires_one_representative_per_token():
    features = feature_fixture()
    features["distogram_rep_atom_mask"] = torch.tensor([1, 1, 0, 1])
    with pytest.raises(ValueError, match="Distogram representatives"):
        audit_feature_schema(features)


def test_schema_rejects_noncontiguous_token_numbering():
    features = feature_fixture()
    features["token_index"] = torch.tensor([0, 2, 3])
    with pytest.raises(ValueError, match="contiguous"):
        audit_feature_schema(features)


def test_schema_rejects_frame_referencing_removed_atom():
    features = feature_fixture()
    features["frame_atom_index"][0, 0] = 4
    with pytest.raises(ValueError, match="references an absent atom"):
        audit_feature_schema(features)


def test_schema_allows_negative_frame_sentinel_only_when_frame_invalid():
    features = feature_fixture()
    features["frame_atom_index"][0, 0] = -1
    with pytest.raises(ValueError, match="Valid native frame"):
        audit_feature_schema(features)
    features["has_frame"][0] = 0
    audit_feature_schema(features)


def test_axis_selection_compacts_atom_token_pair_msa_and_template_axes():
    features = feature_fixture()
    for key in ("ref_pos", "token_bonds", "bond_mask", "msa", "template_all_atom_positions"):
        features[key] = torch.arange(features[key].numel()).reshape(features[key].shape)
    before = deepcopy(features)
    atoms, tokens = torch.tensor([0, 1, 3]), torch.tensor([0, 2])
    result = _select_axes(features, atoms, tokens)
    assert torch.equal(result["ref_pos"], features["ref_pos"][atoms])
    assert torch.equal(result["token_bonds"], features["token_bonds"][tokens][:, tokens])
    assert torch.equal(result["bond_mask"], features["bond_mask"][atoms][:, atoms])
    assert torch.equal(result["msa"], features["msa"][:, tokens])
    assert torch.equal(result["template_all_atom_positions"], features["template_all_atom_positions"][:, tokens])
    result["ref_pos"][0, 0] = -99
    assert all(torch.equal(features[key], before[key]) for key in FEATURE_AXES)


def test_remap_readout_preserves_row_order_and_denominator():
    pairs = torch.tensor([[0, 4], [1, 4], [0, 3], [1, 3]])
    mapping = torch.tensor([0, 1, -1, 2, 3])
    mapped = remap_pairs(pairs, mapping)
    assert torch.equal(mapped, torch.tensor([[0, 3], [1, 3], [0, 2], [1, 2]]))
    assert len(mapped) == len(pairs)
    assert torch.equal(pairs, torch.tensor([[0, 4], [1, 4], [0, 3], [1, 3]]))


def test_remap_readout_rejects_deleted_target_instead_of_dropping_a_row():
    with pytest.raises(ValueError, match="deleted or unmapped"):
        remap_pairs(torch.tensor([[0, 2]]), torch.tensor([0, 1, -1]))


def test_remap_readout_rejects_nonbijective_collapsing_map():
    with pytest.raises(ValueError, match="collapsed"):
        remap_pairs(torch.tensor([[0, 2], [1, 2]]), torch.tensor([0, 0, 1]))


@pytest.mark.parametrize("pairs,mapping", [
    (torch.tensor([[0., 1.]]), torch.tensor([0, 1])),
    (torch.tensor([0, 1]), torch.tensor([0, 1])),
    (torch.tensor([[0, 1]]), torch.tensor([[0, 1]])),
    (torch.tensor([[0, 1]]), torch.tensor([0., 1.])),
])
def test_remap_readout_requires_explicit_integer_pair_and_map_shapes(pairs, mapping):
    with pytest.raises(ValueError, match="integer pairs"):
        remap_pairs(pairs, mapping)


def test_remap_readout_rejects_out_of_range_original_index():
    with pytest.raises(ValueError, match="outside"):
        remap_pairs(torch.tensor([[0, 3]]), torch.tensor([0, 1, 2]))


