"""Optional context-paired bond-type supervision for verified motif completions."""

from collections import defaultdict
import math

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule
from latentmol.guidance.motif_completion import _completion_targets, _valid


def _negatives(atoms, charges, bonds, spec, need_ring, need_amide):
    """Present, neutral single bonds where the suggested new label is chemically incompatible."""
    slots = np.flatnonzero(atoms)
    if not len(slots):
        return []
    mol = _label_molecule(atoms, charges, bonds[np.ix_(slots, slots)], spec, slots)
    if not _valid(mol):
        return []
    result = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        ends = (bond.GetBeginAtom(), bond.GetEndAtom())
        if any(a.GetFormalCharge() for a in ends):
            continue
        elements = tuple(sorted(a.GetAtomicNum() for a in ends))
        if need_ring and elements in ((6, 6), (6, 7)) and not bond.IsInRing():
            group = "aromatic"
        elif need_amide and elements == (6, 8) and any(
                a.GetExplicitValence() >= (4 if a.GetAtomicNum() == 6 else 2) for a in ends):
            # A neutral four-valent carbon or two-valent oxygen cannot add another bond order
            # here without a second graph edit. Implicit hydrogens may adjust on other sites.
            group = "amide"
        else:
            continue
        result.append((group, elements, int(slots[ends[0].GetIdx()]), int(slots[ends[1].GetIdx()])))
    return result


def _contrast_sites(arrays, spec, need_ring, need_amide):
    candidates = {"aromatic": [], "amide": []}
    paired = {"aromatic": [], "amide": []}
    negative_by_element = defaultdict(list)
    local = defaultdict(list)
    with rdBase.BlockLogs():
        for row, (atoms, charges, bonds) in enumerate(zip(*arrays)):
            ring, amide = _completion_targets(atoms, charges, bonds, spec, need_ring, need_amide)
            for name, site in (("aromatic", ring), ("amide", amide)):
                if site:
                    candidates[name].append((row, site))
            for name, elements, i, j in _negatives(atoms, charges, bonds, spec, need_ring, need_amide):
                edge = (row, i, j)
                local[row, name, elements].append(edge)
                negative_by_element[name, elements].append(edge)
    stats = {}
    for name, sites in candidates.items():
        local_pairs = 0
        for row, site in sites:
            edges = []
            for k, (i, j, _) in enumerate(site):
                elements = tuple(sorted(spec.atomic_numbers[arrays[0][row, n]] for n in (i, j)))
                pool = local[row, name, elements] or negative_by_element[name, elements]
                if not pool:
                    break
                # Use distinct positions across a ring when multiple controls exist. Pairing
                # relies only on the detached hard graph and does not consume the training RNG.
                neg = pool[(17 * row + k) % len(pool)]
                edges.append((row, i, j, *neg))
            if len(edges) == len(site):
                paired[name].extend(edges)
                local_pairs += sum(p[0] == p[3] for p in edges)
        stats[f"contrast_{name}_eligible_sites"] = len(sites)
        stats[f"contrast_{name}_sites"] = len(paired[name]) // (6 if name == "aromatic" else 1)
        stats[f"contrast_{name}_local_pairs"] = local_pairs
        stats[f"contrast_{name}_pairs"] = len(paired[name])
        stats[f"contrast_{name}_available_negatives"] = sum(len(pool) for (group, _), pool in
                                                            negative_by_element.items() if group == name)
    return paired, stats


def _paired_loss(decoded, spec, paired, *, amide_relative_weight, margin, absolute_fraction):
    total = decoded.bonds[0, 0, 0, 0] * 0
    observations = {}
    for name, label, weight in (("aromatic", "AROMATIC", 1.), ("amide", "DOUBLE", amide_relative_weight)):
        if not paired[name]:
            continue
        r, i, j, n, a, b = torch.tensor(paired[name], device=decoded.bonds.device).T
        pos = .5 * (decoded.bonds[r, i, j] + decoded.bonds[r, j, i])
        neg = .5 * (decoded.bonds[n, a, b] + decoded.bonds[n, b, a])
        wanted = spec.bond_names.index(label)
        single = spec.bond_names.index("SINGLE")
        pos_gap = pos[:, wanted] - pos[:, single]
        neg_gap = neg[:, wanted] - neg[:, single]
        relative = pos_gap - neg_gap
        rank = F.softplus(margin - relative).mean()
        if absolute_fraction:
            # Conditional target CE intentionally reintroduces absolute class pressure.
            target = torch.full_like(r, wanted - 1)
            rank = (1 - absolute_fraction) * rank + absolute_fraction * F.cross_entropy(pos[:, 1:], target)
        total = total + .5 * weight * rank
        observations[f"contrast_{name}_mean_relative_gap"] = float(relative.detach().mean())
    return total, observations


def contrast_completion_loss(decoded, spec, *, need_ring, need_amide, amide_relative_weight=1.,
                             margin=1., absolute_fraction=0.):
    """Rank verified positive-vs-single gaps above matched negative-vs-single gaps.

    A constant offset to any bond category cancels exactly in the pure ranking term.
    The shared network can still generalize to other positions through its feature weights.
    A ring requires a matched control for every constituent edge. A batch without a
    match contributes an attached zero. Each motif's weight follows completion_loss.
    """
    if (not math.isfinite(amide_relative_weight) or amide_relative_weight < 0 or not math.isfinite(margin)
            or margin < 0 or not 0 <= absolute_fraction <= 1):
        raise ValueError("Invalid motif ranking margin or strength")
    if not need_ring and not need_amide:
        return decoded.bonds[0, 0, 0, 0] * 0, {}
    with torch.no_grad():
        hard = decoded.harden(spec)
        arrays = [t.cpu().numpy() for t in (hard.atoms, hard.charges, hard.bonds)]
        paired, stats = _contrast_sites(arrays, spec, need_ring, need_amide)
    objective, observations = _paired_loss(decoded, spec, paired, amide_relative_weight=amide_relative_weight,
                                          margin=margin, absolute_fraction=absolute_fraction)
    return objective, {**stats, **observations}
