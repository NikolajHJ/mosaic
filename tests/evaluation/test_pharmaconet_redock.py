import numpy as np
import pytest
from rdkit import Chem

from latentmol.evaluation.pharmaconet_redock import compare, measure, summarize


def test_same_hotspot_and_donor_angle_survive_permutation_but_not_displacement():
    mol = Chem.MolFromSmiles("CC=O")
    xyz = np.array([[-2.,0.,0.], [-1.,0.,0.], [0.,0.,0.]])
    site = dict(id="accept", feature="acceptor", center=[0.,0.,0.], radius=1.,
                protein_position=[3.,0.,0.], donor_hydrogens=[[2.,0.,0.]])
    before = measure(mol, xyz, [site])
    assert before["accept"]["hits"][0]["best_donor_H_acceptor_angle_deg"] == 180.
    order = [2, 1, 0]
    after = measure(Chem.RenumberAtoms(mol, order), xyz[order], [site])
    recorded = compare(before, after)
    assert recorded["retained_with_equivalent_ligand_atom"] == ["accept"]
    assert compare(before, measure(mol, xyz + 5., [site]))["lost_contacts"] == ["accept"]
    wrong_angle = measure(mol, xyz, [site | {"donor_hydrogens":[[4.,0.,0.]]}])
    assert wrong_angle["accept"]["in_hotspot"] and not wrong_angle["accept"]["contact"]
    counts = summarize([dict(status="evaluated", before=before, comparison=recorded),
                        dict(status="missing_pose", before=before)], 75, {"accept":"acceptor"})
    assert counts["with_designed_contact"] == 2 and counts["with_retained_contact"] == 1
    assert counts["missing_pose"] == 1 and counts["contact_retention_fraction"] == 1.
    assert counts["by_family"]["acceptor"]["retained_contacts"] == 1


def test_aromatic_centroid_parallel_vs_perpendicular_and_hydrophobic():
    mol = Chem.MolFromSmiles("c1ccccc1")
    phi = np.arange(6) * (np.pi/3)
    xy = np.column_stack([np.cos(phi), np.sin(phi), np.zeros(6)])
    xyz = xy + [0,0,3.5]
    pi = dict(id="ring", feature="aromatic", center=[0,0,3.5], radius=.2,
              protein_ring_coordinates=xy.tolist(), protein_position=[0.,0.,0.], interaction_type="PiStacking_P")
    carbon = dict(id="c", feature="hydrophobic", center=xyz[0].tolist(), radius=.1, protein_position=xy[0].tolist())
    measured = measure(mol, xyz, [pi, carbon])
    assert measured["ring"]["contact"] and measured["c"]["contact"]
    incorrect = measure(mol, xyz, [pi | dict(interaction_type="PiStacking_T")])
    assert incorrect["ring"]["in_hotspot"] and not incorrect["ring"]["contact"]
    assert measured["ring"]["hits"][0]["plane_angle_deg"] == pytest.approx(0.)


def test_partner_geometry_survives_outside_map_radius():
    mol = Chem.MolFromSmiles("CC=O")
    xyz = np.array([[-2., 0., 0.], [-1., 0., 0.], [0., 0., 0.]])
    site = dict(id="accept", feature="acceptor", center=[0., 0., 0.], radius=1.,
                protein_atom="ASN A140 ND2", protein_position=[3., 0., 0.],
                donor_hydrogens=[[2., 0., 0.]])
    before = measure(mol, xyz, [site])
    after = measure(mol, xyz + [0., 1.1, 0.], [site])
    assert not after["accept"]["in_hotspot"] and after["accept"]["partner_contact"]
    recorded = compare(before, after)
    assert recorded["retained_contacts"] == [] and recorded["retained_partner_contacts"] == ["accept"]
    assert recorded["retained_protein_partners"][0]["protein_atom"] == "ASN A140 ND2"
