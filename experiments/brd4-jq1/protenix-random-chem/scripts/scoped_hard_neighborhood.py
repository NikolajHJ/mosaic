"""Bounded first-order move ranking for a connected hard Mosaic graph.

Candidate rankings use gradient differences between categorical channels.
They are a screening heuristic; chemical validity and exact forward scores
belong to the caller. No graph arrays are copied until a move is applied.
"""

from heapq import nsmallest
from typing import NamedTuple

import numpy as np


KINDS = ("atom", "bond", "reattach")
ATOM_CATEGORIES = 6
BOND_CATEGORIES = 5
NO_BOND = 0
SINGLE = 1
MAX_ATOMS = 25
MAX_LIMIT_PER_KIND = 192


class Move(NamedTuple):
    kind: str
    atom_edits: tuple[tuple[int, int], ...]
    bond_edits: tuple[tuple[int, int], ...]
    predicted_gain: float

    def apply(self, atoms, bonds):
        """Make independent slot-aligned copies with the stored edits applied."""
        new_atoms, new_bonds = np.array(atoms, copy=True), np.array(bonds, copy=True)
        for slot, label in self.atom_edits:
            new_atoms[slot] = label
        for row, label in self.bond_edits:
            new_bonds[row] = label
        return new_atoms, new_bonds


def _categorical_gains(labels, gradient, shape):
    gradient = np.asarray(gradient)
    if (gradient.shape != shape or not np.issubdtype(gradient.dtype, np.floating)
            or not np.isfinite(gradient).all()):
        raise ValueError("Gradient must be finite, floating and aligned with categorical rows")
    gradient = gradient.astype(np.float64, copy=False)
    with np.errstate(over="ignore", invalid="ignore"):
        gains = gradient - gradient[np.arange(len(labels)), labels, None]
    if not np.isfinite(gains).all():
        raise ValueError("Nonfinite categorical gain")
    return gains


def _reachable(adjacency, start, omit=None):
    seen = {start}
    queue = [start]
    for vertex in queue:
        for neighbor, row in adjacency[vertex]:
            if row != omit and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _rank(moves, limit):
    return tuple(nsmallest(
        limit, moves, key=lambda move: (-move.predicted_gain, move.atom_edits, move.bond_edits)
    ))


def rank_neighborhood(atoms, bonds, atom_grad, bond_grad, *, limit_per_kind=96):
    """Rank atom, bond and single-bond reattachment moves for score maximization.

    Atoms and bonds contain zero-based labels. Bonds follow the lexicographic
    upper-triangle row order. The gradient is with respect to one-hot entries
    and points toward higher score. All moves, including weaker directions,
    compete under the same deterministic ordering within their respective kind.
    Bond moves change existence because Protenix is insensitive to bond order.
    Reattachment preserves a shared endpoint and proposes only connected
    results, while final validity and admissibility remain caller checks.
    """
    if type(limit_per_kind) is not int or not 1 <= limit_per_kind <= MAX_LIMIT_PER_KIND:
        raise ValueError("Require a bounded positive per-kind shortlist")
    atoms, bonds = np.asarray(atoms), np.asarray(bonds)
    n = len(atoms) if atoms.ndim == 1 else 0
    if not 1 <= n <= MAX_ATOMS or bonds.shape != (n * (n - 1) // 2,):
        raise ValueError("Require fixed-occupancy Mosaic atom and upper-triangle labels")
    for labels, count in ((atoms, ATOM_CATEGORIES), (bonds, BOND_CATEGORIES)):
        if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= count)).any():
            raise ValueError("Categorical labels must be supported integer channels")

    atom_gains = _categorical_gains(atoms, atom_grad, (n, ATOM_CATEGORIES))
    bond_gains = _categorical_gains(bonds, bond_grad, (len(bonds), BOND_CATEGORIES))
    pairs = tuple(zip(*(values.tolist() for values in np.triu_indices(n, 1))))
    row_by_pair = {pair: row for row, pair in enumerate(pairs)}
    adjacency = [[] for _ in atoms]
    present = np.flatnonzero(bonds != NO_BOND).tolist()
    for row in present:
        i, j = pairs[row]
        adjacency[i].append((j, row))
        adjacency[j].append((i, row))
    if len(_reachable(adjacency, 0)) != n:
        raise ValueError("The hard neighborhood starts from a connected graph")

    # For a connected source, deleting one edge either leaves the entire graph
    # connected or partitions it into the reached component and its complement.
    splits = {}
    for row in present:
        i, j = pairs[row]
        component = _reachable(adjacency, i, omit=row)
        if j not in component:
            splits[row] = component

    atom_moves = (
        Move("atom", ((slot, label),), (), float(atom_gains[slot, label]))
        for slot, old in enumerate(atoms)
        for label in range(ATOM_CATEGORIES)
        if label != old
    )

    def bond_moves():
        for row, old in enumerate(bonds):
            if old == NO_BOND:
                label = SINGLE
            elif row in splits:
                continue
            else:
                label = NO_BOND
            yield Move("bond", (), ((row, label),), float(bond_gains[row, label]))

    def reattachments():
        for removed in present:
            if bonds[removed] != SINGLE:
                continue
            i, j = pairs[removed]
            component = splits.get(removed)
            for anchor in (i, j):
                for endpoint in range(n):
                    if endpoint == anchor:
                        continue
                    if component is not None and (anchor in component) == (endpoint in component):
                        continue
                    added = row_by_pair[(min(anchor, endpoint), max(anchor, endpoint))]
                    if bonds[added] != NO_BOND:
                        continue
                    gain = float(bond_gains[removed, NO_BOND]) + float(bond_gains[added, SINGLE])
                    if not np.isfinite(gain):
                        raise ValueError("Nonfinite combined reattachment gain")
                    edits = tuple(sorted(((removed, NO_BOND), (added, SINGLE))))
                    yield Move("reattach", (), edits, gain)

    return {
        "atom": _rank(atom_moves, limit_per_kind),
        "bond": _rank(bond_moves(), limit_per_kind),
        "reattach": _rank(reattachments(), limit_per_kind),
    }
