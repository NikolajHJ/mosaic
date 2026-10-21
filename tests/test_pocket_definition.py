"""Allocated crystallographic identity and selection contracts."""

from copy import deepcopy
from pathlib import Path

import pytest

from mosaic.small_molecule.pocket import derive_definition, validate_definition, boltz_constraints

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pocket():
    return derive_definition(ROOT / "tests/data/brd4_3mxf/3MXF.pdb", ROOT / "tests/data/brd4_3mxf/fixture_25.json")


def test_experimental_jq1_not_solvent_defines_a_mapped_pocket(pocket):
    assert pocket["reference_ligand_heavy_atoms"] == 31
    rows = {r["author_residue"]: r for r in pocket["residues"]}
    assert rows[140]["sequence_position"] == 99 and rows[140]["residue_name"] == "ASN"
    assert {81, 82, 83, 97, 140}.issubset(rows)
    assert all(r["minimum_jq1_distance_angstrom"] <= 6 for r in rows.values())


def test_changed_crystal_bytes_fail_before_geometry(tmp_path):
    changed = tmp_path / "changed.pdb"
    changed.write_text("ATOM changed\n")
    with pytest.raises(ValueError, match="bytes"):
        derive_definition(changed, ROOT / "tests/data/brd4_3mxf/fixture_25.json")


@pytest.mark.parametrize("key,value", [("radius_angstrom", 8), ("anchor_sequence_positions", [81, 140]),
    ("pdb_sha256", "0" * 64), ("protein_chain", "B"), ("version", "new"), ("status", "passed")])
def test_no_silent_protocol_or_index_space_change(pocket, key, value):
    changed = deepcopy(pocket)
    changed[key] = value
    with pytest.raises(ValueError):
        validate_definition(changed)


def test_wrong_amino_acid_or_duplicate_residue_rejected(pocket):
    for mode in ("amino", "duplicate"):
        changed = deepcopy(pocket)
        if mode == "amino":
            changed["residues"][0]["residue_name"] = "XXX"
        else:
            changed["residues"].append(changed["residues"][0])
        with pytest.raises(ValueError):
            validate_definition(changed)


def test_boltz_native_constraints_use_one_based_supplied_sequence(pocket):
    assert boltz_constraints(pocket) == [{"pocket": {"binder": "L", "contacts": [["A", 40], ["A", 99]],
        "max_distance": 6.0, "force": True}}]
