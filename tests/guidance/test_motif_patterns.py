from itertools import product

import numpy as np
from rdkit import Chem
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule, encode_smiles
from latentmol.guidance.motif_patterns import _partial_probability, _pattern_sites, _selected_ce, pattern_completion_loss
from latentmol.representation import DecoderOutput, GraphSpec


SPEC = GraphSpec(max_nodes=14)


def arrays(smiles):
    graphs = [encode_smiles(s, SPEC) for s in smiles]
    return [np.stack([g[key] for g in graphs]) for key in ("atom_types", "charges", "bond_types")]


def logits(arr):
    return DecoderOutput(*(F.one_hot(torch.tensor(a).long(), len(labels)).float().mul(5).requires_grad_()
                         for a, labels in zip(arr, (SPEC.atomic_numbers, SPEC.formal_charges, SPEC.bond_names))))


def test_coherence_probability_and_gradient_at_boundaries():
    p = torch.tensor([[.2, .5, .8], [0., 0., 0.], [0., 1., 1.]], requires_grad=True)
    mixed = sum(torch.where(torch.tensor(labels), p, 1-p).prod(-1)
                for labels in product((False, True), repeat=3) if len(set(labels)) > 1)
    partial = _partial_probability(p)
    torch.testing.assert_close(mixed, partial)
    g, = torch.autograd.grad(partial.sum(), p)
    assert torch.isfinite(g).all() and g[2, 0] < 0 and (g[1] > 0).all()


def test_protects_every_fused_bond_including_kekule_and_unselected_candidates():
    a, c, b = arrays(["c1ccc2ccccc2c1", "C1CCCCC1"])
    ring = b[0] == SPEC.bond_names.index("AROMATIC")
    for kekule in (False, True):
        if kekule:
            slots = np.flatnonzero(a[0])
            mol = _label_molecule(a[0], c[0], b[0][np.ix_(slots, slots)], SPEC, slots)
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            for bond in mol.GetBonds():
                i, j = slots[bond.GetBeginAtomIdx()], slots[bond.GetEndAtomIdx()]
                b[0,i,j] = b[0,j,i] = SPEC.bond_names.index(str(bond.GetBondType()))
        valid, offsite, _, rings, sites, _ = _pattern_sites((a,c,b), SPEC, False)
        assert valid == 2 and not offsite[0][ring].any() and len(sites) == 1
        assert len(rings[6]) == 3
        assert all(not offsite[r,i,j] for r, i, j in sites[0])
        assert [weight for row, _, weight in rings[6] if row == 0] == [.5, .5]


def test_selects_only_lowest_nll_and_normalizes_by_all_rows():
    z = torch.zeros((4, 3, 3, 5))
    z[2, 0, 1, 4] = 2.
    z.requires_grad_()
    loss, selected = _selected_ce(z, [[(0,0,1)], [(2,0,1)]], 4, 1)
    expected = F.cross_entropy(z[2,0,1,1:].unsqueeze(0), torch.tensor([3])) / 4
    torch.testing.assert_close(loss, expected)
    assert selected == 1
    gradient, = torch.autograd.grad(loss, z)
    assert gradient[2,0,1,4] < 0 and not gradient[0].count_nonzero() and not gradient[...,0].count_nonzero()


def test_impossible_co_and_offsite_aromatic_push_down_and_skip_invalid_rows():
    arr = arrays(["COC", "CC"])
    for tensor in arr:
        tensor[1] = 0
    decoded = logits(arr)
    loss, metrics = pattern_completion_loss(decoded, SPEC, aromatic_needed=0, amide_needed=0)
    assert metrics["pattern_valid_rows"] == 1 and metrics["pattern_blocked_co_pairs"] == 2
    total, = torch.autograd.grad(loss, decoded.bonds)
    assert total[...,4].sum() > 0 and total[...,2].sum() > 0  # SGD decreases both unwanted labels
    assert total[1].count_nonzero() == 0
    torch.testing.assert_close(total, total.transpose(1, 2))
