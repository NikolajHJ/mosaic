"""Pilot decoder of one joint six-bond aromatic assignment on a verified cycle.

Eligibility uses the realized graph and is discrete. Given that graph, the head
reads element, charge and substituent information; it never sees the chosen
ring's six bond orders. Raw invalid graphs and fused/spiro rings are ineligible.
The relaxed output is a conditional surrogate for a fixed selected cycle.
"""

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule
from latentmol.guidance.motif_completion import _roundtrips, _valid
from latentmol.representation import HardGraphBatch


@dataclass(frozen=True)
class RingSite:
    row: int
    slots: tuple[int, ...]
    aromatic: bool

    @property
    def saturated(self):
        return not self.aromatic


def _verified_order(atoms, charges, edges, spec, occupied, ring, aromatic):
    """Inspect the actual raw slot reassignment, including all original off-ring labels."""
    labels = edges.copy()
    for i, j in zip(ring, ring[1:] + ring[:1]):
        labels[i, j] = labels[j, i] = spec.bond_names.index("AROMATIC" if aromatic else "SINGLE")
    edited = _label_molecule(atoms, charges, labels, spec, occupied)
    if not _valid(edited) or not _roundtrips(edited):
        return False
    return all(edited.GetBondBetweenAtoms(i, j).GetIsAromatic() == aromatic
               for i, j in zip(ring, ring[1:] + ring[:1]))


def _topology_ranks(atoms, charges, edges, spec, occupied):
    """Canonical symmetry ranks on a bond-order-blind graph for site selection."""
    skeleton = _label_molecule(atoms, charges, np.where(edges != 0, spec.bond_names.index("SINGLE"), 0),
                               spec, occupied)
    skeleton.UpdatePropertyCache(strict=False)
    return Chem.CanonicalRankAtoms(skeleton, breakTies=False, includeChirality=False)


def find_ring_sites(hard, spec):
    """Select at most one isolated, singly bonded or wholly aromatic six-C/N ring per row.

    Both complete labels are verified on each ring. Multiple candidates are
    ranked using only topology and atom labels; skip an unresolved symmetry tie.
    """
    sites = []
    arrays = [getattr(hard, field).detach().cpu().numpy() for field in ("atoms", "charges", "bonds")]
    with rdBase.BlockLogs():
        for row, (atoms, charges, bonds) in enumerate(zip(*arrays, strict=True)):
            occupied = np.flatnonzero(atoms)
            if len(occupied) < 6:
                continue
            edges = bonds[np.ix_(occupied, occupied)]
            if not np.array_equal(edges, edges.T) or np.any(np.diag(edges)):
                raise ValueError("Ring gate requires undirected hard bonds without self loops")
            mol = _label_molecule(atoms, charges, edges, spec, occupied)
            if not _valid(mol) or not _roundtrips(mol):
                continue
            ri = mol.GetRingInfo()
            candidates = []
            for ring in ri.AtomRings():
                if len(ring) != 6 or any(ri.NumAtomRings(i) != 1 for i in ring):
                    continue
                members = [mol.GetAtomWithIdx(i) for i in ring]
                if any(a.GetAtomicNum() not in (6, 7) or a.GetFormalCharge() for a in members):
                    continue
                perimeter = [mol.GetBondBetweenAtoms(i, j) for i, j in zip(ring, ring[1:] + ring[:1])]
                aromatic = all(b.GetIsAromatic() for b in perimeter)
                if not aromatic and not all(b.GetBondType() == Chem.BondType.SINGLE for b in perimeter):
                    continue
                # The starting endpoint was verified above. Applying the other
                # label to the raw graph must also produce the expected endpoint.
                if _verified_order(atoms, charges, edges, spec, occupied, ring, not aromatic):
                    candidates.append((ring, aromatic))
            if not candidates:
                continue
            if len(candidates) > 1:
                ranks = _topology_ranks(atoms, charges, edges, spec, occupied)
                keys = [tuple(sorted(ranks[i] for i in ring)) for ring, _ in candidates]
                best = min(keys)
                if keys.count(best) > 1:
                    continue
                ring, aromatic = candidates[keys.index(best)]
            else:
                ring, aromatic = candidates[0]
            sites.append(RingSite(row, tuple(int(occupied[i]) for i in ring), aromatic))
    return sites


class SharedRingGate(nn.Module):
    """Permutation-invariant readout of six atoms and their off-ring substituents."""

    def __init__(self, spec):
        super().__init__()
        width = 2 * len(spec.atomic_numbers) + len(spec.formal_charges) + len(spec.bond_names) - 1
        self.per_atom = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, graph, sites):
        """Read only contextual features. The six ring-edge labels are masked even for positives."""
        row, left, _ = _indices(sites, graph.atoms.device)
        b = row[:, None]
        atom_features = graph.atoms[b, left]
        charge_features = graph.charges[b, left]
        # Discard all six ring atoms as substituents; this excludes both directions
        # of every perimeter bond even if a caller supplies arbitrary masked labels.
        external = torch.ones((len(sites), graph.atoms.size(1)), dtype=torch.bool, device=left.device)
        external.scatter_(1, left, False)
        types = graph.bonds[b, left, :, 1:] * external[:, None, :, None]
        external_atoms = types.sum(-1) @ graph.atoms[row]
        h = self.per_atom(torch.cat([atom_features, charge_features, types.sum(2), external_atoms], dim=-1))
        return self.readout(torch.cat([h.mean(1), h.amax(1)], dim=-1)).squeeze(-1)


def _indices(sites, device):
    row = torch.tensor([s.row for s in sites], device=device, dtype=torch.long)
    left = torch.tensor([s.slots for s in sites], device=device, dtype=torch.long)
    return row, left, left.roll(-1, 1)


def gate_training_loss(gate, hard, spec):
    """Class-balanced BCE on real graphs. An absent class supplies zero for its half.

    These scores are decision scores under balanced training priors. Evaluate
    on both natural positives and saturated negatives before interpreting yields.
    """
    sites = find_ring_sites(hard, spec)
    positives = sum(s.aromatic for s in sites)
    stats = dict(rows=len(hard.atoms), sites=len(sites), positive=positives, negative=len(sites) - positives)
    if not sites:
        return next(gate.parameters()).reshape(-1)[0] * 0, stats
    graph = hard.to(next(gate.parameters()).device).to_relaxed(spec)
    logits = gate(graph, sites)
    targets = logits.new_tensor([s.aromatic for s in sites])
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets.bool()
    pos_loss = (losses * positive).sum() / max(positives, 1)
    neg_loss = (losses * ~positive).sum() / max(len(sites) - positives, 1)
    return 0.5 * (pos_loss + neg_loss), stats


@torch.no_grad()
def apply_ring_gate(gate, hard, spec, *, decoded=None, generator=None, include_ceiling=False):
    """Choose the base or verified aromatic product using one decision per complete ring.

    A categorical caller can pass an independent generator; None chooses the
    aromatic endpoint at probability >=0.5. Preserve inputs and slot numbering.
    """
    sites = find_ring_sites(hard, spec)
    stats = {key: [0] * len(hard.atoms) for key in (
        "eligible_by_row", "saturated_by_row", "fired_by_row", "changed_by_row")}
    stats["probability_by_row"] = [None] * len(hard.atoms)
    if not sites:
        if include_ceiling:
            stats["all_eligible"] = hard
        return hard, stats
    device = next(gate.parameters()).device
    graph = hard.to(device).to_relaxed(spec) if decoded is None else decoded.probabilities()
    scores = gate(graph, sites).sigmoid()
    if not torch.isfinite(scores).all():
        raise ValueError("Ring gate returned nonfinite probabilities")
    choose = scores >= .5 if generator is None else torch.rand(
        scores.shape, device=scores.device, generator=generator) < scores
    bonds = hard.bonds.clone()
    ceiling = hard.bonds.clone() if include_ceiling else None
    for site, score, chosen in zip(sites, scores.tolist(), choose.tolist(), strict=True):
        stats["eligible_by_row"][site.row] = 1
        stats["saturated_by_row"][site.row] = int(site.saturated)
        stats["probability_by_row"][site.row] = score
        stats["fired_by_row"][site.row] = int(chosen)
        if site.saturated and (chosen or include_ceiling):
            left = torch.as_tensor(site.slots, device=bonds.device)
            right = left.roll(-1)
            if include_ceiling:
                ceiling[site.row, left, right] = ceiling[site.row, right, left] = spec.bond_names.index("AROMATIC")
            if chosen:
                bonds[site.row, left, right] = bonds[site.row, right, left] = spec.bond_names.index("AROMATIC")
                stats["changed_by_row"][site.row] = 1
    if include_ceiling:
        stats["all_eligible"] = HardGraphBatch(hard.atoms, hard.charges, ceiling)
    return HardGraphBatch(hard.atoms, hard.charges, bonds), stats


def relaxed_ring_bonds(gate, hard, decoded, spec):
    """Return ring-mixture marginals for the fixed selected saturated cycle.

    Hold bond existence at its original probability. Blend conditional bond type
    including a shared aromatic choice. This is a local differentiable surrogate;
    discrete resampling of the starting graph can select a different cycle.
    """
    sites = [s for s in find_ring_sites(hard, spec) if s.saturated]
    graph = decoded.probabilities()
    if not sites:
        return graph.bonds
    prob = gate(graph, sites).sigmoid()[:, None, None]
    rows, left, right = _indices(sites, graph.bonds.device)
    b = rows[:, None]
    source = graph.bonds[b, left, right]
    target = torch.zeros_like(source)
    target[..., 0] = source[..., 0]
    target[..., spec.bond_names.index("AROMATIC")] = 1 - source[..., 0]
    blended = source + prob * (target - source)
    bonds = graph.bonds.clone()
    bonds[b, left, right] = blended
    bonds[b, right, left] = blended
    return bonds
