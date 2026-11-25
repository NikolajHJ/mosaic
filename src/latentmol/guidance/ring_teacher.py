"""Selective parent bond-type supervision on verified six-membered C/N ring edges."""

from collections import Counter

import numpy as np
from rdkit import rdBase
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule
from latentmol.guidance.motif_completion import _completion_targets, _roundtrips, _valid
from latentmol.training.prior_repair import conditional_parent_loss


def ring_single_mask(decoded, hard, valid, spec):
    """Return unique upper-triangle indices. Use every observed aromatic six-C/N ring and at most one near miss."""
    with torch.no_grad(), rdBase.BlockLogs():
        student = decoded.harden(spec)
        a, c, b, ref_atoms, ref_bonds, ok = [x.cpu().numpy() for x in (
            student.atoms, student.charges, student.bonds, hard.atoms, hard.bonds, valid)]
        chosen = []
        counts = Counter()
        for row in np.flatnonzero(ok):
            slots = np.flatnonzero(a[row])
            if not len(slots):
                continue
            mol = _label_molecule(a[row], c[row], b[row][np.ix_(slots, slots)], spec, slots)
            if not _valid(mol) or not _roundtrips(mol):
                continue
            counts["joint_strict_valid"] += 1
            edges = {}
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) != 6 or any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6,7) for i in ring):
                    continue
                cycle = list(zip(ring, ring[1:] + ring[:1]))
                if all(mol.GetBondBetweenAtoms(i,j).GetIsAromatic() for i,j in cycle):
                    counts["observed_rings"] += 1
                    for i,j in cycle:
                        edges[tuple(sorted((int(slots[i]),int(slots[j]))))] = "observed"
            proposed, _ = _completion_targets(a[row], c[row], b[row], spec, True, False)
            if proposed:
                counts["verified_candidate_rings"] += 1
                for i,j,_ in proposed:
                    edges.setdefault(tuple(sorted((i,j))), "candidate")
            masked_row = False
            for (i,j), kind in edges.items():
                if not (ref_atoms[row,i] and ref_atoms[row,j]):
                    counts["unoccupied_teacher_pair"] += 1
                    continue
                label = spec.bond_names[ref_bonds[row,i,j]]
                counts["teacher_" + label] += 1
                if label == "SINGLE":
                    masked_row = True
                    chosen.append((int(row), i, j))
                    counts["masked_" + kind] += 1
            counts["affected_rows"] += masked_row
        return chosen, dict(counts)


def ring_teacher_loss(decoded, hard, valid, masked):
    """Keep native CE, subtract only present-type CE at verified conflicting SINGLE targets."""
    original = conditional_parent_loss(decoded, hard, valid)
    if not masked:
        return original
    row, i, j = torch.as_tensor(masked, device=decoded.bonds.device).T
    selected = decoded.symmetric_bond_logits[row,i,j,1:]
    loss_to_remove = F.cross_entropy(selected, hard.bonds[row,i,j] - 1, reduction="sum")
    active = valid[:,None] & hard.atoms.ne(0)
    pairs = (active[:,:,None] & active[:,None,:]).triu(1)
    targets = hard.bonds[pairs]
    weights = torch.where(targets != 0, selected.new_ones(()), selected.new_tensor(.1))
    return original - 2 * valid.to(selected.dtype).mean() * loss_to_remove / weights.sum()


def conditional_ring_teacher(decoded, hard, valid, spec):
    """Apply the selective ring teacher ablation using the original eligibility and normalization."""
    masked, counts = ring_single_mask(decoded, hard, valid, spec)
    return ring_teacher_loss(decoded, hard, valid, masked), dict(counts, masked_pairs=len(masked))
