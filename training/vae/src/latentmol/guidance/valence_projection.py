"""Project over-valent occupied graphs and supervise only the bonds that changed."""

from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from latentmol.guidance.regularization import ALLOWED_VALENCES


@lru_cache(maxsize=16)
def _tables(spec):
    names = spec.bond_names
    if set(names) != {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"} or names[0] != "NONE":
        raise ValueError("Projection requires the five known conditional bond categories")
    cap = np.full((len(spec.atomic_numbers), len(spec.formal_charges)), np.inf)
    for i, element in enumerate(spec.atomic_numbers):
        for j, charge in enumerate(spec.formal_charges):
            allowed = ALLOWED_VALENCES.get(element, {}).get(charge)
            if allowed:
                cap[i, j] = max(allowed)
    lower = np.arange(len(names))
    for current, target in (("SINGLE", "NONE"), ("DOUBLE", "SINGLE"), ("TRIPLE", "DOUBLE")):
        lower[names.index(current)] = names.index(target)
    orders = np.array([{"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1}[n] for n in names])
    return cap, lower, orders, np.triu_indices(spec.max_nodes, 1)


def projection_targets(atoms, charges, logits, spec, max_edits):
    """Detached labels, indexed by unordered slot pairs. Rows can stop partially repaired."""
    caps, lower, orders, (left, right) = _tables(spec)
    batch, slots = atoms.shape
    if (slots != spec.max_nodes or charges.shape != atoms.shape or max_edits < 1
            or logits.shape != (batch, len(left), len(spec.bond_names)) or not np.isfinite(logits).all()):
        raise ValueError("Invalid conditional edge logits, atom/charge batch or projection budget")
    row, edge = np.ogrid[:batch, :len(left)]
    labels = logits.argmax(-1)
    labels[(atoms[:, left] == 0) | (atoms[:, right] == 0)] = 0
    original = labels.copy()
    cap = caps[atoms, charges].copy()
    aromatic_rows, aromatic_edges = np.where(labels == spec.bond_names.index("AROMATIC"))
    cap[aromatic_rows, left[aromatic_edges]] = np.inf
    cap[aromatic_rows, right[aromatic_edges]] = np.inf
    load = np.zeros_like(cap)
    np.add.at(load, (row, left), orders[labels])
    np.add.at(load, (row, right), orders[labels])
    defect = (load > cap).any(axis=1)
    protected = np.zeros_like(labels, dtype=bool)
    # Construct a maximum-confidence spanning forest only for graphs needing edits.
    for k in np.flatnonzero(defect):
        used = np.flatnonzero(labels[k])
        values = logits[k, used]
        confidence = 1. / np.exp(values - values.max(-1, keepdims=True)).sum(-1)
        roots = list(range(slots))
        def root(n):
            while roots[n] != n:
                roots[n] = roots[roots[n]]
                n = roots[n]
            return n
        # Kruskal over occupied edges. Stable order breaks probability ties by slot pair.
        for idx in used[np.argsort(-confidence, kind="stable")]:
            a, b = root(left[idx]), root(right[idx])
            if a != b:
                protected[k, idx] = True
                roots[b] = a
    for _ in range(max_edits):
        excess = load > cap
        gain = excess[:, left].astype(np.int8) + excess[:, right]
        target = lower[labels]
        eligible = (gain > 0) & (target != labels) & ((target != 0) | ~protected)
        chosen = np.flatnonzero(eligible.any(axis=1))
        if not len(chosen):
            break
        cost = (logits[row, edge, labels] - logits[row, edge, target]) / np.maximum(1, gain)
        edits = np.where(eligible, cost, np.inf)[chosen].argmin(axis=1)
        labels[chosen, edits] = target[chosen, edits]
        load[chosen, left[edits]] -= 1
        load[chosen, right[edits]] -= 1
    return original, labels, {"over_valent": int(defect.sum()), "still_over_valent": int((load > cap).any(1).sum())}


def projection_loss(decoded, spec, max_edits=8):
    """Average changed-edge CE within graph, then over every sampled prior."""
    i, j = torch.triu_indices(spec.max_nodes, spec.max_nodes, offset=1, device=decoded.bonds.device)
    logits = (decoded.bonds[:, i, j] + decoded.bonds[:, j, i]) * .5
    with torch.no_grad():
        if not (decoded.atoms.isfinite().all() & decoded.charges.isfinite().all()):
            raise FloatingPointError("Nonfinite atom or charge logits in projection")
        original, target, stats = projection_targets(decoded.atoms.argmax(-1).cpu().numpy(),
            decoded.charges.argmax(-1).cpu().numpy(), logits.detach().float().cpu().numpy(), spec, max_edits)
    rows, edges = np.where(original != target)
    counts = np.bincount(rows, minlength=len(original))
    stats.update(changed_graphs=int((counts > 0).sum()), changed_edges=len(edges))
    if not len(edges):
        return logits.reshape(-1)[0] * 0., stats
    device = logits.device
    labels = torch.as_tensor(target[rows, edges], device=device)
    weight = torch.as_tensor(1. / counts[rows], device=device, dtype=logits.dtype)
    selected = logits[torch.as_tensor(rows, device=device), torch.as_tensor(edges, device=device)]
    return (F.cross_entropy(selected, labels, reduction="none") * weight).sum() / len(original), stats
