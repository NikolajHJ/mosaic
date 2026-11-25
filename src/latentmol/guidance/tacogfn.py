"""Relaxed TacoGFN guidance on the current valid molecular graph."""

from dataclasses import dataclass
import math
from itertools import chain

import numpy as np
from rdkit import Chem
import torch
from torch.nn import functional as F

from latentmol.chemistry import _label_molecule, inspect_graph


@dataclass
class _Topology:
    smiles: str
    slots: torch.Tensor
    edge_index: torch.Tensor
    aromatic: torch.Tensor
    hard_bonds: torch.Tensor


def _mapped_topology(smiles, labels, spec, active, topology, device):
    """Map canonical Open Babel features to live decoder slots via the hard graph."""
    atoms, charges, bonds = labels
    atom_ids = torch.as_tensor(topology["x"], dtype=torch.long)
    edges = torch.as_tensor(topology["edge_index"], dtype=torch.long)
    edge_types = torch.as_tensor(topology["edge_attr"], dtype=torch.long)
    n = spec.max_nodes
    if (atoms.shape != (n,) or charges.shape != (n,) or bonds.shape != (n, n)
            or active.ndim != 1 or len(active) < 2 or len(np.unique(active)) != len(active)
            or not np.array_equal(np.sort(active), np.flatnonzero(atoms))):
        raise ValueError("Inconsistent slots or label shapes")
    mol = _label_molecule(atoms, charges, bonds[np.ix_(active, active)], spec, active)
    Chem.SanitizeMol(mol)
    if Chem.MolToSmiles(mol, isomericSmiles=False) != smiles:
        raise ValueError("Slot labels do not reproduce the canonical SMILES")
    query = Chem.MolFromSmiles(smiles)
    order = mol.GetSubstructMatch(query)
    if len(set(order)) != len(active) or len(atom_ids) != len(active):
        raise ValueError("Incomplete canonical atom mapping")
    slots = active[list(order)]
    if list(atoms[slots] - 1) != atom_ids.tolist():
        raise ValueError("Upstream TacoGFN atom identities disagree with live slots")
    if (edges.shape != (2, mol.GetNumBonds()) or edge_types.shape != (mol.GetNumBonds(),)
            or edges.min() < 0 or edges.max() >= len(active)
            or len({tuple(sorted(pair)) for pair in edges.T.tolist()}) != mol.GetNumBonds()):
        raise ValueError("Invalid directed TacoGFN bond list")
    for (left, right), category in zip(edges.T.tolist(), edge_types.tolist(), strict=True):
        bond = mol.GetBondBetweenAtoms(order[left], order[right])
        if bond is None:
            raise ValueError("TacoGFN edge missing from live chemistry")
        expected = 3 if bond.GetIsAromatic() else int(bond.GetBondTypeAsDouble()) - 1
        if expected != category:
            raise ValueError("TacoGFN and sanitized bond order disagree")
    return _Topology(smiles, torch.as_tensor(slots, device=device), edges.to(device),
                     (edge_types == 3).to(device), F.one_hot(edge_types.to(device), 5).float())


def _label_key(labels):
    # Include slot order and formal charge, even when graph identity is unchanged.
    return tuple(a.astype(np.uint8, copy=False).tobytes() for a in labels)


class TacoGFNGuidance:
    """Add a docking gradient to optional base guidance, scaled in latent space.

    At each valid argmax, hold its edge list and perceived aromatic bonds fixed
    for that derivative. Update the topology whenever the hard graph changes.
    Nonaromatic bond order is conditional on presence. Invalid graphs continue
    ordinary optimization with zero docking gradient.
    """

    def __init__(self, head, founder_row, graph_spec, topology, *, cap_ratio=0.05,
                 norm_mode="cap", base_guidance=None):
        if (norm_mode not in ("cap", "match") or not math.isfinite(cap_ratio) or not 0 <= cap_ratio <= 1
                or graph_spec.atomic_numbers != (0, 6, 7, 8, 9, 15, 16, 17, 35, 53)
                or graph_spec.bond_names != ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")):
            raise ValueError("Incompatible TacoGFN vocabulary or latent gradient cap")
        if head.training or any(p.requires_grad or p.grad is not None for p in head.parameters()):
            raise ValueError("TacoGFN head must already be frozen and in evaluation mode")
        self.device = next(chain(head.parameters(), head.buffers())).device
        self.spec = graph_spec
        self.active_indices = np.asarray(founder_row["active_slots"], dtype=np.int64)
        labels = tuple(np.asarray(founder_row["labels"][key]) for key in ("atoms", "charges", "bonds"))
        initial = _mapped_topology(founder_row["smiles"], labels, graph_spec, self.active_indices,
                                   topology, self.device)
        self.founder_key = _label_key(labels)
        self.cache = {self.founder_key: (initial, None)}
        self.active = torch.zeros(graph_spec.max_nodes, dtype=torch.bool, device=self.device)
        self.active[initial.slots] = True
        self.head, self.base_guidance, self.cap_ratio = head, base_guidance, cap_ratio
        self.norm_mode = norm_mode
        self.requires_raw_valid = getattr(base_guidance, "requires_raw_valid", False)
        with torch.no_grad():
            ids = torch.as_tensor(topology["x"], dtype=torch.long, device=self.device)
            hard_score = head(F.one_hot(ids, 10).float(), initial.hard_bonds, initial.edge_index)
        if hard_score.ndim != 0 or not bool(hard_score.isfinite()):
            raise ValueError("Invalid hard founder score")
        if not math.isclose(float(hard_score), topology["hard_score"], abs_tol=0.0005, rel_tol=0):
            raise ValueError("Frozen TacoGFN head disagrees with upstream hard founder score")

    def _layout(self, labels):
        from .tacogfn_head import topology_from_smiles

        inspected = inspect_graph(*labels, self.spec)
        if inspected.smiles is None:
            return None, f"invalid:{inspected.failure_reason}"
        try:
            topology = topology_from_smiles(inspected.smiles)
            return _mapped_topology(inspected.smiles, labels, self.spec, self.active_indices,
                                    topology, self.device), None
        except (ValueError, OSError) as exc:
            return None, f"unrepresentable:{exc}"

    def evaluate(self, graph, active, point, binding_gradient, step, **validity):
        n = self.active.numel()
        if (active.dtype != torch.bool or not torch.equal(active, self.active)
                or graph.atoms.shape != (1, n, 10) or graph.bonds.shape != (1, n, n, 5)
                or graph.charges.shape != (1, n, len(self.spec.formal_charges))
                or any(not bool(v.isfinite().all()) for v in (graph.atoms, graph.bonds, graph.charges))
                or not torch.equal(graph.occupancy.detach(), self.active.to(graph.atoms)[None])
                or binding_gradient.shape != point.shape or not bool(binding_gradient.isfinite().all())
                or not point.requires_grad or not bool(point.isfinite().all())):
            raise ValueError("TacoGFN requires a finite, attached, fixed-occupancy founder branch")
        labels = tuple(getattr(graph, name).detach().argmax(-1)[0].byte().cpu().numpy()
                       for name in ("atoms", "charges", "bonds"))
        key = _label_key(labels)
        if step == 0 and key != self.founder_key:
            raise ValueError("First live TacoGFN graph does not replay its founder")
        if self.base_guidance is None:
            base, metadata = torch.zeros_like(point), {}
        else:
            base, metadata = self.base_guidance.evaluate(graph, active, point, binding_gradient, step, **validity)
        cached = key in self.cache
        if not cached:
            self.cache[key] = self._layout(labels)
        layout, reason = self.cache[key]
        reference = float(binding_gradient.detach().double().norm())
        detail = dict(active=layout is not None, skip_reason=reason, used_founder_graph=key == self.founder_key,
                      topology_cache_hit=cached, binding_gradient_norm=reference, norm_mode=self.norm_mode)
        detail["cap_ratio" if self.norm_mode == "cap" else "target_ratio"] = self.cap_ratio
        if layout is None:
            detail.update(score=None, raw_gradient_norm=0., applied_gradient_norm=0.,
                          cap_scale=None, actual_ratio=0. if reference else None)
            return base, {**metadata, "tacogfn": detail}
        atom_probs = F.pad(graph.atoms[0, layout.slots, 1:], (0, 1))
        left, right = layout.slots[layout.edge_index]
        positive = graph.bonds[0, left, right, 1:]
        mass = positive.sum(-1, keepdim=True)
        if not bool((mass > 0).all()):
            raise ValueError("Undefined bond-order distribution on a current hard edge")
        orders = F.pad(positive / mass, (0, 1))
        bond_probs = torch.where(layout.aromatic[:, None], layout.hard_bonds, orders)
        score = self.head(atom_probs, bond_probs, layout.edge_index)
        if score.ndim != 0 or not score.requires_grad or not bool(score.isfinite()):
            raise ValueError("Nonfinite or detached TacoGFN objective")
        raw, = torch.autograd.grad(score, point, retain_graph=True, allow_unused=True)
        if raw is None or not bool(raw.isfinite().all()):
            raise ValueError("Missing or nonfinite TacoGFN latent gradient")
        raw_norm = float(raw.double().norm())
        if self.norm_mode == "match":
            scale = self.cap_ratio * reference / raw_norm if raw_norm else 0.
            # A very small but finite derivative can require a large scale.
            applied = (raw.double() * scale).to(raw.dtype)
            if reference == 0 or raw_norm == 0:
                detail.update(active=False, skip_reason="zero_binding_gradient" if not reference
                              else "zero_tacogfn_gradient")
        else:
            scale = min(1.0, self.cap_ratio * reference / raw_norm) if raw_norm else 1.0
            applied = raw * scale
        if not bool(applied.isfinite().all()):
            raise ValueError("Nonfinite rescaled docking gradient")
        applied_norm = float(applied.double().norm())
        detail.update(score=float(score.detach()), raw_gradient_norm=raw_norm, applied_gradient_norm=applied_norm,
                      scale=scale, actual_ratio=applied_norm / reference if reference else None,
                      smiles=layout.smiles, edges=layout.edge_index.shape[1],
                      fixed_aromatic_edges=int(layout.aromatic.sum()))
        if self.norm_mode == "cap":
            detail["cap_scale"] = scale
        return base + applied, {**metadata, "tacogfn": detail}
