"""Portable typed-identity fixtures; execute numerical checks only on DTU HPC.

No pretrained model, Boltz import, native pickle or network is used here. These
fixtures establish helper behavior, not validation of the real native workflow.
"""
import numpy as np
import pytest
from rdkit import Chem

from latentmol.boltz.processed_identity import (
    ArtifactCorruption, IdentityMismatch, check_crop, check_features, check_target,
    compare_molecules, _msa_query, _protein_sequence, audit_processed,
)


def named_molecule(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    for atom in molecule.GetAtoms():
        atom.SetProp('name', f'A{atom.GetIdx()}')
    return molecule


def native_fixture(molecule, order=None):
    """Native feature rows intentionally need not follow RDKit atom order."""
    if order is None:
        order = list(range(molecule.GetNumAtoms()))
    n = len(order)
    elements = np.zeros((1, n, 128))
    charges = np.zeros((1, n))
    pairs = np.zeros((1, n, n), dtype=np.int64)
    inverse = {original: row for row, original in enumerate(order)}
    for row, index in enumerate(order):
        atom = molecule.GetAtomWithIdx(index)
        elements[0, row, atom.GetAtomicNum()] = 1
        charges[0, row] = atom.GetFormalCharge()
    types = {'SINGLE': 2, 'DOUBLE': 3, 'TRIPLE': 4, 'AROMATIC': 5}
    for bond in molecule.GetBonds():
        a, b = inverse[bond.GetBeginAtomIdx()], inverse[bond.GetEndAtomIdx()]
        pairs[0, a, b] = pairs[0, b, a] = types[str(bond.GetBondType()).upper()]
    names = [molecule.GetAtomWithIdx(index).GetProp('name') for index in order]
    return names, {'ref_element': elements, 'ref_charge': charges,
                   'type_bonds': pairs, 'token_bonds': (pairs > 0)[..., None]}


def test_reordered_charged_molecule_retains_typed_identity_and_features():
    expected = '[NH3+]CC[O-]'
    molecule = named_molecule(expected)
    reordered = Chem.RenumberAtoms(molecule, [3, 1, 0, 2])
    chemical = compare_molecules(expected, reordered)
    assert chemical['same_molecular_identity']
    assert chemical['processed_net_formal_charge'] == 0
    names, features = native_fixture(molecule, [3, 1, 0, 2])
    assert check_features(reordered, names, list(range(4)), list(range(4)), features)[
        'native_chemical_feature_parity']


def test_net_charge_alone_does_not_establish_molecular_identity():
    original = named_molecule('[NH3+]CC[O-]')
    neutral = named_molecule('NCCO')
    assert sum(a.GetFormalCharge() for a in original.GetAtoms()) == 0
    assert sum(a.GetFormalCharge() for a in neutral.GetAtoms()) == 0
    assert not compare_molecules('[NH3+]CC[O-]', neutral)['same_molecular_identity']


def test_local_charge_feature_redistribution_is_detected_even_at_equal_net_charge():
    molecule = named_molecule('[NH3+]CC[O-]')
    names, features = native_fixture(molecule)
    features['ref_charge'][0, [0, 3]] = [-1, 1]
    assert features['ref_charge'].sum() == 0
    with pytest.raises(ArtifactCorruption, match='formal-charge'):
        check_features(molecule, names, list(range(4)), list(range(4)), features)


def test_changed_bond_order_is_an_identity_change_and_feature_mismatch():
    assert not compare_molecules('CCO', named_molecule('C=CO'))['same_molecular_identity']
    molecule = named_molecule('CCO')
    names, features = native_fixture(molecule)
    features['type_bonds'][0, 0, 1] = features['type_bonds'][0, 1, 0] = 3
    with pytest.raises(ArtifactCorruption, match='typed-bond'):
        check_features(molecule, names, list(range(3)), list(range(3)), features)


def test_aromatic_bonds_are_preserved_in_native_typed_features():
    molecule = named_molecule('c1ccncc1')
    names, features = native_fixture(molecule)
    assert check_features(molecule, names, list(range(6)), list(range(6)), features)[
        'native_chemical_feature_parity']
    features['type_bonds'][0, 0, 1] = features['type_bonds'][0, 1, 0] = 2
    with pytest.raises(ArtifactCorruption):
        check_features(molecule, names, list(range(6)), list(range(6)), features)


def test_crop_requires_exact_full_heavy_atom_coverage():
    assert check_crop(['A0', 'A1'], ['A1', 'A0'], ['A0', 'A1'])['complete_ligand_coverage']
    with pytest.raises(IdentityMismatch, match='loses/replaces'):
        check_crop(['A0', 'A1'], ['A0', 'A1'], ['A0'])
    with pytest.raises(ArtifactCorruption, match='Duplicate'):
        check_crop(['A0', 'A1'], ['A0', 'A0'], ['A0', 'A1'])


@pytest.mark.parametrize('observations', [('AC', 'AC', 'AD'), ('AC', 'AD', 'AC'), ('AD', 'AC', 'AC')])
def test_changed_processed_target_pose_or_msa_is_ineligible(observations):
    with pytest.raises(IdentityMismatch, match='full target'):
        check_target('AC', *observations)


def test_target_and_query_are_reconstructed_from_actual_native_tables():
    from types import SimpleNamespace
    chains = np.array([(0, 0, 2)], dtype=[('asym_id', 'i4'), ('res_idx', 'i4'), ('res_num', 'i4')])
    residues = np.array([('ALA', True), ('CYS', True)], dtype=[('name', 'U4'), ('is_standard', '?')])
    structure = SimpleNamespace(chains=chains, residues=residues)
    msa = SimpleNamespace(sequences=np.array([(0, 2)], dtype=[('res_start', 'i4'), ('res_end', 'i4')]),
                          residues=np.array([(1,), (2,)], dtype=[('res_type', 'i4')]))
    mapping = {'ALA': 'A', 'CYS': 'C'}
    sequence = _protein_sequence(structure, 0, mapping)
    query = _msa_query(msa, ['<pad>', 'ALA', 'CYS'], mapping)
    assert sequence == query == 'AC'
    assert check_target('AC', sequence, sequence, query)['processed_msa_query_sha256']


def test_audit_cannot_accidentally_load_native_artifacts_outside_hpc(tmp_path, monkeypatch):
    monkeypatch.delenv('LSB_JOBID', raising=False)
    monkeypatch.delenv('LATENTMOL_SOURCE_SHA256', raising=False)
    with pytest.raises(ValueError, match='requires HPC'):
        audit_processed(tmp_path, tmp_path, {})
