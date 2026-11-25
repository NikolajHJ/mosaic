"""Batch-limited completion targets and expected bond-pattern errors on fixed prior skeletons."""

from collections import defaultdict
import math

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule
from latentmol.guidance.motif_completion import _completion_targets, _roundtrips, _valid


def _edge(i, j):
    return (i, j) if i < j else (j, i)


def _pattern_sites(arrays, spec, need_amide):
    atoms, charges, bonds = arrays
    offsite = np.zeros_like(bonds, dtype=bool)
    blocked_co = np.zeros_like(offsite)
    positive_rings, positive_amides = [], []
    rings_by_length = defaultdict(list)
    valid_rows = 0
    for row, (a, c, b) in enumerate(zip(atoms, charges, bonds)):
        slots = np.flatnonzero(a)
        if not len(slots):
            continue
        mol = _label_molecule(a, c, b[np.ix_(slots, slots)], spec, slots)
        if not _valid(mol) or not _roundtrips(mol):
            continue
        valid_rows += 1
        offsite[row] = np.triu((a != 0)[:, None] & (a != 0)[None, :], 1)
        rings = set()
        for ring in mol.GetRingInfo().AtomRings():
            edges = list(zip(ring, ring[1:] + ring[:1]))
            if all(mol.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in edges):
                rings.add(tuple(sorted(_edge(int(slots[i]), int(slots[j])) for i, j in edges)))
        for bond in mol.GetBonds():
            left, right = bond.GetBeginAtom(), bond.GetEndAtom()
            i, j = _edge(int(slots[left.GetIdx()]), int(slots[right.GetIdx()]))
            # Protect every perceived aromatic edge, including fused bonds and
            # aromaticity inferred by RDKit from alternating single/double bonds.
            if bond.GetIsAromatic():
                offsite[row, i, j] = False
            if (bond.GetBondType() == Chem.BondType.SINGLE and not left.GetFormalCharge()
                    and not right.GetFormalCharge() and {left.GetAtomicNum(), right.GetAtomicNum()} == {6, 8}):
                if any(atom.GetExplicitValence() >= (4 if atom.GetAtomicNum() == 6 else 2)
                       for atom in (left, right)):
                    blocked_co[row, i, j] = True
        ring, amide = _completion_targets(a, c, b, spec, True, need_amide)
        if ring:
            ring = tuple(sorted(_edge(i, j) for i, j, _ in ring))
            rings.add(ring)
            positive_rings.append([(row, i, j) for i, j in ring])
        if amide:
            positive_amides.append([(row, i, j) for i, j, _ in amide])
        for ring in sorted(rings):
            for i, j in ring:
                offsite[row, i, j] = False
            rings_by_length[len(ring)].append((row, ring, 1. / len(rings)))
    return valid_rows, offsite, blocked_co, rings_by_length, positive_rings, positive_amides


def _selected_ce(logits, sites, desired, needed):
    """Choose the lowest mean live conditional NLL with a detached site decision."""
    n = min(needed, len(sites))
    if not n:
        return logits[0, 0, 0, 0] * 0, 0
    row, i, j = torch.as_tensor(sites, device=logits.device).unbind(-1)
    conditional = F.log_softmax(logits[row, i, j, 1:], dim=-1)
    site_nll = -conditional[..., desired - 1].mean(-1)
    keep = site_nll.detach().argsort(stable=True)[:n]
    return site_nll[keep].sum() / logits.shape[0], n


def _partial_probability(p):
    """Probability independent edge labels contain both aromatic and nonaromatic bonds."""
    return (1 - p.prod(-1) - (1 - p).prod(-1)).clamp_min(0)


def _pattern_objective(logits, sites, spec, aromatic_needed, amide_needed, amide_relative_weight):
    valid, offsite, blocked, rings_by_length, positive_rings, positive_amides = sites
    aro, double = (spec.bond_names.index(name) for name in ("AROMATIC", "DOUBLE"))
    batch = len(logits)
    aromatic_ce, n_rings = _selected_ce(logits, positive_rings, aro, aromatic_needed)
    amide_ce, n_amides = _selected_ce(logits, positive_amides, double, amide_needed)
    p = logits.softmax(-1)   # Unconditional bond probabilities, including NO_BOND.
    masks = [torch.as_tensor(mask, device=p.device) for mask in (offsite, blocked)]
    offsite_expectation = p[..., aro][masks[0]].sum() / batch
    blocked_double = p[..., double][masks[1]].sum() / batch
    partial = logits[0, 0, 0, 0] * 0
    for rings in rings_by_length.values():
        indices = [[(row, i, j) for i, j in ring] for row, ring, _ in rings]
        row, i, j = torch.as_tensor(indices, device=p.device).unbind(-1)
        weights = p.new_tensor([weight for _, _, weight in rings])
        partial = partial + (_partial_probability(p[row, i, j, aro]) * weights).sum() / batch
    objective = (.5 * (aromatic_ce + amide_relative_weight * amide_ce) + .5 * offsite_expectation
                 + .25 * partial + .10 * blocked_double)
    metrics = dict(pattern_valid_rows=valid, pattern_aromatic_candidates=len(positive_rings),
                   pattern_aromatic_selected=n_rings, pattern_amide_candidates=len(positive_amides),
                   pattern_amide_selected=n_amides, pattern_protected_rings=sum(map(len, rings_by_length.values())),
                   pattern_offsite_pairs=int(offsite.sum()), pattern_blocked_co_pairs=int(blocked.sum()),
                   pattern_aromatic_positive_ce=float(aromatic_ce.detach()), pattern_amide_positive_ce=float(amide_ce.detach()),
                   pattern_offsite_aromatic=float(offsite_expectation.detach()),
                   pattern_partial_aromatic=float(partial.detach()), pattern_blocked_co_double=float(blocked_double.detach()))
    return objective, metrics


def pattern_completion_loss(decoded, spec, *, aromatic_needed: int, amide_needed: int, amide_relative_weight=.10):
    """Teach up to the batch's missing sites and penalize stray aromatic and impossible C–O labels.

    The hard graph fixes eligible sites, occupied slots, and strict chemical readouts.
    Each positive site contributes one mean CE divided by the full input batch. Each ring
    coherence cost is averaged over its graph's protected rings and then over the input
    batch. Unreadable skeletons contribute zero to all these sums.
    """
    if any(type(n) is not int or n < 0 for n in (aromatic_needed, amide_needed)):
        raise ValueError("Pattern deficits must be nonnegative integer hit counts")
    if not math.isfinite(amide_relative_weight) or amide_relative_weight < 0:
        raise ValueError("Amide weight must be finite and nonnegative")
    if len(decoded.bonds) == 0 or spec.bond_names[0] != "NONE":
        raise ValueError("Expected a nonempty batch with NONE as bond class zero")
    with torch.no_grad(), rdBase.BlockLogs():
        hard = decoded.harden(spec)
        arrays = [t.cpu().numpy() for t in (hard.atoms, hard.charges, hard.bonds)]
        sites = _pattern_sites(arrays, spec, amide_needed > 0)
    logits = decoded.symmetric_bond_logits
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return _pattern_objective(logits, sites, spec, aromatic_needed, amide_needed, amide_relative_weight)
