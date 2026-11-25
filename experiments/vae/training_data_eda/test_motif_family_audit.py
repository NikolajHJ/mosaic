"""Small reference structures for ring participation and aromatic carbonyl forms."""

import pytest
from rdkit import Chem

from experiments.vae.training_data_eda.motif_family_audit import can_complete, flags, urea_centers


@pytest.mark.parametrize(("smiles", "cyclic", "pendant", "lactam"), [
    ("O=C1NCCN1", 1, 0, False),                    # Ethylene urea.
    ("O=C1NCC(=O)N1", 1, 0, True),                 # Hydantoin has a second carbonyl.
    ("O=c1cc[nH]c(=O)[nH]1", 1, 0, True),          # Diketo uracil, aromatic RDKit graph.
    ("Cn1ccc(=O)n(C)c1=O", 1, 0, True),            # N-methylation preserves the motif.
    ("NC(=O)N1CCCCC1", 0, 1, False),                # Ring nitrogen, external urea carbonyl.
    ("O=C(N1CCCCC1)N1CCCCC1", 0, 1, False),        # Both nitrogens belong to separate rings.
    ("NC(=O)NC1CN(C(=O)N1)C", 1, 1, False),       # Both kinds at different centers.
    ("O=C1CCCN1", 0, 0, True),                     # Lactam with only one flanking nitrogen.
    ("O=C1OCCN1", 0, 0, False),                    # Cyclic carbamate.
    ("NC(=N)N", 0, 0, False),                      # Guanidine lacks the carbonyl.
])
def test_urea_ring_membership(smiles, cyclic, pendant, lactam):
    mol = Chem.MolFromSmiles(smiles)
    ring, external = urea_centers(mol)
    assert (len(ring), len(external)) == (cyclic, pendant)
    assert ring.isdisjoint(external)
    categories = flags(mol)
    assert categories["cyclic_urea"] == bool(cyclic)
    assert categories["noncyclic_urea"] == bool(pendant)
    assert categories["urea"] == bool(cyclic + pendant)
    assert categories["lactam"] == lactam


@pytest.mark.parametrize(("smiles", "can_oxidize"), [
    ("OC1NCCN1", True),          # C(OH)H between the two ring nitrogens.
    ("OC(N)N1CCCCC1", False),    # Carbon outside the ring.
    ("COC1NCCN1", False),        # Methoxy oxygen cannot make that edit.
    ("OC1(C)NCCN1", False),     # Carbon already carries an extra substituent.
    ("OC1NCCO1", False),         # Carbamate precursor, one ring nitrogen.
])
def test_one_edit_cyclic_urea_precursors(smiles, can_oxidize):
    assert can_complete(Chem.MolFromSmiles(smiles)) == can_oxidize


@pytest.mark.parametrize(("smiles", "expected"), [
    ("O=C1OCCN1", True), ("O=c1[nH]cco1", True), ("O=C1OCCO1", False),
])
def test_carbamate(smiles, expected):
    assert flags(Chem.MolFromSmiles(smiles))["carbamate"] == expected


@pytest.mark.parametrize(("smiles", "expected"), [
    ("O=C1NCCN1", True), ("O=C1NCCCN1", True), ("O=C1NCCCCN1", False),
    ("NC(=O)N1CCCCC1", False),
])
def test_five_six_member_cyclic_urea(smiles, expected):
    assert flags(Chem.MolFromSmiles(smiles))["cyclic_urea_5_or_6_membered"] == expected
