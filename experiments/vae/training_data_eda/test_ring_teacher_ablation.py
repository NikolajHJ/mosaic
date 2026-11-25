import numpy as np
from rdkit import Chem
import torch
from torch.nn import functional as F

from latentmol.guidance.ring_teacher import conditional_ring_teacher, ring_single_mask, ring_teacher_loss
from latentmol.chemistry import _label_molecule, encode_smiles
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.prior_repair import conditional_parent_loss


def test_actual_aromatic_site_and_kekule_share_mask_and_keep_other_teacher_labels():
    spec = GraphSpec(max_nodes=10)
    graph = encode_smiles("Cc1ccccc1", spec)
    a,c,b = [graph[k] for k in ("atom_types","charges","bond_types")]
    aromatic_edges = list(zip(*np.where(np.triu(b == 4,1))))
    ref = np.where(b != 0,1,0)
    for (i,j), label in zip(aromatic_edges[:2],(0,2)):
        ref[i,j] = ref[j,i] = label
    hard = HardGraphBatch(*(torch.tensor(t).long().unsqueeze(0) for t in (a,c,ref)))
    valid = torch.tensor([True])
    expected = {(0,i,j) for i,j in aromatic_edges[2:]}
    slots = np.flatnonzero(a)
    mol = _label_molecule(a,c,b[np.ix_(slots,slots)],spec,slots)
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol,clearAromaticFlags=True)
    alternating = b.copy()
    for bond in mol.GetBonds():
        i,j = slots[bond.GetBeginAtomIdx()], slots[bond.GetEndAtomIdx()]
        alternating[i,j] = alternating[j,i] = spec.bond_names.index(str(bond.GetBondType()))
    for raw in (b,alternating):
        out = DecoderOutput(*(5. * F.one_hot(torch.tensor(t).long().unsqueeze(0),len(v)).double() for t,v in
                              zip((a,c,raw),(spec.atomic_numbers,spec.formal_charges,spec.bond_names))))
        masked, counts = ring_single_mask(out,hard,valid,spec)
        assert set(masked) == expected and counts["masked_observed"] == 4
        loss, metrics = conditional_ring_teacher(out,hard,valid,spec)
        assert metrics["masked_pairs"] == 4
        torch.testing.assert_close(loss,ring_teacher_loss(out,hard,valid,masked),rtol=0,atol=0)
        assert ring_single_mask(out,hard,~valid,spec)[0] == []
        assert torch.equal(ring_teacher_loss(out,hard,valid,[]),conditional_parent_loss(out,hard,valid))
