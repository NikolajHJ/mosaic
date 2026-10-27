"""Allocated pocket identity/geometry tests; extraction uses pinned PDB bytes."""
from copy import deepcopy

import pytest

from latentmol.diffsbdd.pocket import PDB_SHA256, VERSION, audit_pose, derive_definition, validate_definition


def _definition():
    sequence = "A" * 127
    residues = [{"author_residue": position + 41, "sequence_position": position,
                 "insertion_code": "", "residue_name": "ALA",
                 "minimum_jq1_distance_angstrom": 1.0}
                for position in (40, 56, 99)]
    return {"version": VERSION, "pdb_sha256": PDB_SHA256,
            "protein_chain": "A", "protein_sequence": sequence,
            "reference_ligand": "JQ1", "reference_ligand_heavy_atoms": 31,
            "radius_angstrom": 6.0, "anchor_author_residues": [81, 140],
            "anchor_sequence_positions": [40, 99], "residues": residues,
            "native_pocket": {"atom_rows": [
                {"sequence_position": 40, "coordinates": [0., 0., 0.]},
                {"sequence_position": 56, "coordinates": [1., 0., 0.]},
                {"sequence_position": 99, "coordinates": [2., 0., 0.]},
            ]}}


def test_author_and_sequence_anchors_are_not_interchangeable():
    definition = _definition()
    assert validate_definition(definition) is definition
    definition["anchor_sequence_positions"] = [81, 140]
    with pytest.raises(ValueError, match="protocol"):
        validate_definition(definition)


def test_omitted_sequence56_is_rejected():
    definition = _definition()
    definition["residues"] = [row for row in definition["residues"]
                              if row["sequence_position"] != 56]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_definition(definition)


def test_changed_pdb_bytes_fail_before_geometry_or_model_loading(tmp_path):
    path = tmp_path / "fake-3mxf.pdb"
    path.write_text("ATOM fake\n")
    with pytest.raises(ValueError, match="PDB bytes"):
        derive_definition(path)


def test_pose_audit_rejects_wrong_global_frame_without_redocking():
    definition = _definition()
    good = audit_pose([[1., 1., 0.], [1., 2., 0.]], definition)
    bad = audit_pose([[101., 1., 0.], [101., 2., 0.]], definition)
    assert good["passed"]
    assert not bad["passed"]
    assert bad["ligand_fraction_within_6A_of_pocket"] == 0.0


def test_pose_audit_requires_both_anchors_and_majority_of_ligand():
    definition = _definition()
    result = audit_pose([[0., 0., 0.], [100., 100., 100.], [101., 101., 101.]], definition)
    assert not result["passed"]
    assert result["anchor_minimum_distances_angstrom"]["40"] == 0
    assert result["ligand_fraction_within_6A_of_pocket"] == pytest.approx(1 / 3)


def test_bad_residue_mapping_and_missing_native_atom_rows_are_hard_errors():
    bad = deepcopy(_definition())
    bad["residues"][0]["author_residue"] = 82
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_definition(bad)
    bad = deepcopy(_definition())
    bad["native_pocket"]["atom_rows"] = bad["native_pocket"]["atom_rows"][:2]
    with pytest.raises(ValueError, match="no coordinates"):
        audit_pose([[0., 0., 0.]], bad)
