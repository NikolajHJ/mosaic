"""Real-target residue selection must preserve the exact deposited construct."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_known_ligand_fixture.py"
    )
    spec = importlib.util.spec_from_file_location("known_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pdb():
    return (Path(__file__).parent / "data/t4_lysozyme_181l/181L.pdb").read_text()


def test_deposited_construct_and_size_matched_neutral_reference():
    fixture = _module().prepare(_pdb())
    assert len(fixture["protein_sequence"]) == 164
    assert fixture["protein_sequence"][53] == "T"
    assert fixture["protein_sequence"][96:99] == "AAA"
    assert 99 in fixture["pocket_author_residues"]
    assert fixture["n_heavy_atoms"] == 6
    assert len(fixture["reference_bond_assignments"]) == 6


def test_missing_reference_is_rejected():
    text = "\n".join(line for line in _pdb().splitlines() if line[17:20] != "BNZ")
    with pytest.raises(ValueError, match="six-atom"):
        _module().prepare(text)


def test_mismatched_author_residue_is_rejected():
    text = _pdb().replace("ATOM      1  N   MET A   1", "ATOM      1  N   ALA A   1")
    with pytest.raises(ValueError, match="mapping"):
        _module().prepare(text)
