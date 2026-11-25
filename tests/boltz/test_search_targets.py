import hashlib

import numpy as np
import pytest
from rdkit import Chem

from latentmol.boltz.search_targets import seed_in_ccd_order, selected_label_chains, verify_native_pocket, verify_pose_distances
import torch
from latentmol.representation import GraphSpec


def test_seed_keeps_ccd_order_and_names_instead_of_canonical_reordering():
    molecule = Chem.RenumberAtoms(Chem.MolFromSmiles("CCO"), [2, 1, 0])
    for i, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("name", f"X{i}")
    _, seed, names, smiles = seed_in_ccd_order(molecule, GraphSpec())
    assert names == ["X0", "X1", "X2"]
    assert smiles == "CCO"
    assert seed.atoms[0, :3].tolist() == [3, 1, 1]
    assert seed.bonds[0, 0, 1] == seed.bonds[0, 1, 2] == 1
    molecule.GetAtomWithIdx(1).SetProp("name", "X0")
    with pytest.raises(ValueError, match="duplicated"):
        seed_in_ccd_order(molecule, GraphSpec())


def test_explicit_aromatic_h_is_an_explicit_reference_fixture_exception():
    molecule = Chem.MolFromSmiles("c1cc[nH]c1")
    for i, atom in enumerate(molecule.GetAtoms()):
        atom.SetProp("name", f"X{i}")
    with pytest.raises(ValueError, match="not representable"):
        seed_in_ccd_order(molecule, GraphSpec())
    _, seed, _, canonical = seed_in_ccd_order(molecule, GraphSpec(), allow_reference_roundtrip_failure=True)
    assert "[nH]" in canonical and int((seed.atoms != 0).sum()) == 5


def test_exact_author_locator_resolves_a_different_label_subchain(tmp_path):
    pytest.importorskip("gemmi")
    cif = tmp_path / "input.cif"
    cif.write_text("""data_test
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.pdb_strand_id
X A
Y B
loop_
_atom_site.label_asym_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.auth_comp_id
L A 600 IOK
M B 600 IOK
N A 601 IOK
""")
    target = dict(pocket=dict(mmcif_sha256=hashlib.sha256(cif.read_bytes()).hexdigest(),
        protein_chain="A", reference_ligand_locator=dict(chain="A", residue=600, component="IOK")))
    assert selected_label_chains(cif, target) == ("X", "L")
    target["pocket"]["reference_ligand_locator"]["residue"] = 602
    with pytest.raises(ValueError, match="missing or ambiguous"):
        selected_label_chains(cif, target)


def test_pocket_uses_sequence_positions_and_refuses_missing_crop_residues():
    tokens = np.array([(0, 128, "ASP", 0, True), (0, 129, "PHE", 0, True)],
        dtype=[("asym_id", "i4"), ("res_idx", "i4"), ("res_name", "U4"),
               ("mol_type", "i4"), ("resolved_mask", "?")])
    target = dict(pocket=dict(residues=[dict(sequence_position=129, author_residue=127, residue_name="ASP")]))
    assert verify_native_pocket(tokens, target, 0) == [dict(asym_id=0, residue_index=128)]
    with pytest.raises(ValueError, match="after native crop"):
        verify_native_pocket(tokens[1:], target, 0)
    tokens[0]["resolved_mask"] = False
    with pytest.raises(ValueError, match="resolved crystal"):
        verify_native_pocket(tokens, target, 0)


def test_pose_guard_accepts_global_translation_but_rejects_ligand_only_shift():
    ligand = torch.randn(32, 3) + 200
    pocket = torch.randn(5, 3) + 201
    audit = verify_pose_distances(ligand-100, ligand, pocket-100, pocket)
    assert audit["ligand_pocket_max_distance_error"] < 1e-4
    with pytest.raises(AssertionError):
        verify_pose_distances(ligand-100, ligand, pocket-95, pocket)
