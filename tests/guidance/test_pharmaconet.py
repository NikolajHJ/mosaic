import math

import pytest
import torch
from rdkit import Chem

from latentmol.design.boltz_direct_graph import DirectGraphLogits
from latentmol.guidance.pharmaconet import (PocketFeatureLoss, _relaxed_features, hard_feature_atoms,
                                            hard_feature_groups, registered_reference)
from latentmol.representation import GraphSpec, HardGraphBatch


def example(smiles):
    mol = Chem.MolFromSmiles(smiles)
    spec = GraphSpec(max_nodes=mol.GetNumAtoms())
    atoms = torch.tensor([[spec.atomic_numbers.index(atom.GetAtomicNum()) for atom in mol.GetAtoms()]])
    charges = torch.tensor([[spec.formal_charges.index(atom.GetFormalCharge()) for atom in mol.GetAtoms()]])
    bonds = torch.zeros((1, spec.max_nodes, spec.max_nodes), dtype=torch.long)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[0, i, j] = bonds[0, j, i] = spec.bond_names.index(str(bond.GetBondType()))
    return spec, HardGraphBatch(atoms, charges, bonds), mol


@pytest.mark.parametrize("smiles", ["CC(=O)NC", "CCO", "COC", "CC#N", "c1ccncc1", "C[NH3+]", "CCN", "CCCl"])
def test_hard_features_match_one_hot_local_roles(smiles):
    spec, hard, mol = example(smiles)
    expected = hard_feature_atoms(mol)
    active = hard.atoms[0] != 0
    features = _relaxed_features(hard.to_relaxed(spec), active, spec)
    for i, role in enumerate(("acceptor", "donor", "hydrophobic")):
        assert set(features[i].nonzero().flatten().tolist()) == expected[role]
        assert torch.all((features[i] == 0) | (features[i] == 1))


def test_registered_coordinates_and_live_gradient_from_typed_spatial_match():
    spec, hard, _ = example("CC(=O)NC")
    xyz = torch.tensor([[1., 2., 0.], [2., 1., 1.], [2., 0., 1.], [0., 1., 1.], [1., 1., 1.]])
    names = list("ABCDE")
    metadata = dict(target=dict(pocket=dict(pdb_id="EX", protein_chain="A")),
                    slot_map=[dict(slot=i, atom_name=n) for i,n in enumerate(names)])
    source = dict(pdb_id="EX", chain="A", reference_atoms=dict(zip(names, xyz.tolist(), strict=True)))
    native = xyz.roll(1, dims=1) + 43.
    assert registered_reference(source, metadata, list(range(5)), native)[2] < .00001
    with pytest.raises(ValueError, match="registration failed"):
        registered_reference(source, metadata, list(range(5)), native * 1.2)
    hot = dict(id="site", feature="acceptor", center=xyz[2].tolist(), radius=.6, score=.94)
    loss = PocketFeatureLoss([hot], xyz)
    hard_score, matches = loss(hard.to_relaxed(spec), hard.atoms[0] != 0, spec)
    assert hard_score.abs() < 1.e-6 and matches["site"] == 1.
    model = DirectGraphLogits(hard, spec, 13)
    value, _ = loss(model()[1], model.active, spec)
    grads = torch.autograd.grad(value, tuple(model.parameters()))
    assert all(g.isfinite().all() and g.norm() > .01 for g in grads)
    # Increasing the strongest site's carbonyl oxygen and its double bond
    # improves the match for this soft graph and registered hotspot.
    assert grads[0][2, spec.atomic_numbers.index(8) - 1] < 0
    edge = (model.left == 1) & (model.right == 2)
    assert grads[2][edge, spec.bond_names.index("DOUBLE")] < 0


@pytest.mark.parametrize("smiles", ["c1ccccc1", "c1occc1"])
def test_ring_center_and_kekule_patterns_require_a_complete_aromatic_cycle(smiles):
    spec, hard, mol = example(smiles)
    active = hard.atoms[0] != 0
    size = len(active)
    angle = torch.arange(size) * 2 * math.pi / size
    radius = 1.4 / (2 * math.sin(math.pi / size))
    xyz = torch.stack((radius * angle.cos(), radius * angle.sin(), angle * 0), dim=1)
    hot = dict(id="ring", feature="aromatic", center=[0., 0., 0.], radius=.6, score=.96)
    loss = PocketFeatureLoss([hot], xyz)
    assert len(hard_feature_groups(mol)["aromatic"]) == 1
    assert loss(hard.to_relaxed(spec), active, spec)[1]["ring"] == pytest.approx(1.)
    for normal, expected in (([0., 0., 1.], 0.), ([1., 0., 0.], 1.)):
        oriented = hot | dict(interaction_type="PiStacking_T", protein_normal=normal)
        assert PocketFeatureLoss([oriented], xyz)(hard.to_relaxed(spec), active, spec)[1]["ring"] == pytest.approx(expected)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        hard.bonds[0, i, j] = hard.bonds[0, j, i] = spec.bond_names.index(str(bond.GetBondType()))
    assert loss(hard.to_relaxed(spec), active, spec)[1]["ring"] == pytest.approx(1.)
    hard.bonds[0, size - 1, 0] = hard.bonds[0, 0, size - 1] = 0
    assert loss(hard.to_relaxed(spec), active, spec)[1]["ring"] == 0.
    model = DirectGraphLogits(hard, spec, 139)
    score, _ = loss(model()[1], active, spec)
    assert all(g.isfinite().all() and g.norm() > 0 for g in torch.autograd.grad(score, tuple(model.parameters())))
    hot2 = hot | dict(id="h1", feature="hydrophobic")
    h3 = hot2 | dict(id="h2", score=.48)
    assert PocketFeatureLoss([hot, hot2, h3], xyz).weights.tolist() == pytest.approx([.5, 1/3, 1/6])
