"""Supplied-pose construction/gradient contracts; numerical fixtures are HPC-only."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmol.diffsbdd import scorer_context as context
from latentmol.diffsbdd.scorer import finite_difference_certificate


def pdb_atom(serial, name, residue, number, xyz, *, chain="A", insertion="", altloc="", occupancy=1.0):
    element = name[0]
    return (f"ATOM  {serial:5d} {name:^4s}{altloc:1s}{residue:3s} {chain:1s}{number:4d}{insertion:1s}   "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{occupancy:6.2f}{20.0:6.2f}          {element:>2s}")


def test_crystal_parser_retains_author_insertions_coordinates_and_alternates(tmp_path):
    path = tmp_path / "receptor.pdb"
    path.write_text("\n".join([
        pdb_atom(1, "N", "ALA", 42, (1, 2, 3)),
        pdb_atom(2, "CA", "ALA", 42, (2, 2, 3), altloc="A", occupancy=.4),
        pdb_atom(3, "CA", "ALA", 42, (3, 2, 3), altloc="B", occupancy=.6),
        pdb_atom(4, "CA", "GLY", 42, (4, 2, 3), insertion="A"),
        pdb_atom(5, "CA", "VAL", 2, (9, 9, 9), chain="B")]) + "\n")
    parsed = context.read_receptor_pdb(path)
    assert parsed["sequence"] == "AG"
    assert list(parsed["residues"]) == [("A", 42, ""), ("A", 42, "A")]
    chosen = parsed["residues"][("A", 42, "")]["atoms"]["CA"]
    assert chosen["altloc"] == "B"
    np.testing.assert_array_equal(chosen["coordinates"], [3, 2, 3])


def test_parser_rejects_duplicate_atoms_and_unselected_multiple_models(tmp_path):
    line = pdb_atom(1, "CA", "ALA", 1, (0, 0, 0))
    path = tmp_path / "bad.pdb"
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        context.read_receptor_pdb(path)
    path.write_text("MODEL        1\n" + line + "\nENDMDL\nMODEL        2\n" + line + "\nENDMDL\n")
    with pytest.raises(ValueError, match="single PDB model"):
        context.read_receptor_pdb(path)


def test_explicit_author_pocket_requires_chain_number_and_insertion():
    good = [{"chain": "A", "residue_number": 42, "insertion_code": "A"}]
    assert context.normalize_author_residues(good) == (("A", 42, "A"),)
    with pytest.raises(ValueError, match="explicit"):
        context.normalize_author_residues([{"chain": "A", "residue_number": 42}])
    with pytest.raises(ValueError, match="unique"):
        context.normalize_author_residues(good + good)
    with pytest.raises(ValueError, match="explicit"):
        context.normalize_author_residues([dict(good[0], chain="B")])


def test_fresh_structure_binding_keeps_reference_conformers_separate():
    atoms = np.zeros(5, dtype=[("name", "U4"), ("coords", "f4", (3,)), ("is_present", "?")])
    atoms["name"] = ["N", "CA", "C", "C1", "O2"]
    residues = np.zeros(2, dtype=[("name", "U5"), ("res_idx", "i4"),
        ("atom_idx", "i4"), ("atom_num", "i4"), ("is_present", "?")])
    residues[0] = ("ALA", 0, 0, 3, True)
    residues[1] = ("LIG0", 0, 3, 2, True)
    chains = np.zeros(2, dtype=[("name", "U5"), ("asym_id", "i4"), ("res_idx", "i4"),
        ("res_num", "i4"), ("atom_idx", "i4"), ("atom_num", "i4")])
    chains[0], chains[1] = ("A", 0, 0, 1, 0, 3), ("L", 1, 1, 1, 3, 2)
    coords = np.zeros(5, dtype=[("coords", "f4", (3,))])
    reference = np.arange(15).reshape(5, 3).copy()
    target = SimpleNamespace(structure=SimpleNamespace(atoms=atoms, residues=residues,
        chains=chains, coords=coords), reference_conformer=reference)
    receptor = {"residues": {("A", 42, ""): {"name": "ALA", "atoms": {
        "N": {"coordinates": np.array([1, 2, 3]), "altloc": ""},
        "CA": {"coordinates": np.array([2, 2, 3]), "altloc": ""}}}}}
    supplied = np.zeros((8, 3), dtype=np.float32)
    supplied[4], supplied[1] = [8, 9, 10], [11, 12, 13]
    mapping, _ = context._bind_structure(target, receptor, {"C1": 4, "O2": 1}, supplied, "A")
    assert mapping[("A", 42, "")] == (0, 0)
    np.testing.assert_array_equal(atoms["coords"][3], supplied[4])
    np.testing.assert_array_equal(coords["coords"][4], supplied[1])
    assert not atoms["is_present"][2]  # Missing receptor C remains unresolved.
    np.testing.assert_array_equal(target.reference_conformer, np.arange(15).reshape(5, 3))


def test_a3m_query_is_explicit_and_cannot_silently_drop_gaps(tmp_path):
    path = tmp_path / "query.a3m"
    path.write_text(">query\nACD\nEF\n>other\nACDEF\n")
    assert context.msa_query(path) == "ACDEF"
    path.write_text(">query\nAC-DEF\n")
    with pytest.raises(ValueError, match="ungapped uppercase"):
        context.msa_query(path)


def test_resolved_adjacent_fd_contract_accepts_correct_and_rejects_wrong_gradient():
    point = torch.tensor([[0.7, -0.2]], dtype=torch.float64)

    def objective(value):
        return {"total": value.square().sum() + 0.3 * value.sum()}

    correct = 2 * point + .3
    directions = {"one": torch.tensor([[.8, .6]], dtype=torch.float64)}
    passed = finite_difference_certificate(objective, point, {"total": correct}, directions)
    assert passed["passed"] and all(row["wrong_gradient_control_rejected"] for row in passed["rows"])
    failed = finite_difference_certificate(objective, point, {"total": -correct}, directions)
    assert not failed["passed"]


def test_null_or_forward_jitter_cannot_be_called_a_gradient_pass():
    point = torch.ones((1, 2), dtype=torch.float64)
    null = {"null": torch.zeros_like(point)}
    result = finite_difference_certificate(lambda value: {"total": value.sum()}, point,
        {"total": torch.ones_like(point)}, null)
    assert not result["passed"] and all(row["null_direction_cannot_establish_nonzero_gradient"] for row in result["rows"])
    calls = 0

    def noisy(value):
        nonlocal calls
        calls += 1
        return {"total": value.sum() + (1.0 if calls % 2 else 0.0)}

    result = finite_difference_certificate(noisy, point, {"total": torch.ones_like(point)},
        {"one": torch.ones_like(point)})
    assert not result["passed"] and all(not row["resolved"] for row in result["rows"])
