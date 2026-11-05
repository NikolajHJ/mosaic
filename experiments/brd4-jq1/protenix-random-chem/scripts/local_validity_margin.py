"""Detached, bond-local validity guard for the fixed-occupancy pilot."""

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from mosaic.small_molecule.chemistry._reference.chemistry import encode_smiles, inspect_graph


MARGIN = 0.2
SLOPE_CAP = 0.25
WEIGHT_PER_REPAIR_UNIT = 0.001
MAX_ROWS = 8


@jax.jit
def margin_value(logits, allowed, enabled):
    allowed, enabled = jax.lax.stop_gradient((allowed, enabled))
    good = jnp.where(allowed, logits, -jnp.inf)
    bad = jnp.where(allowed, -jnp.inf, logits)
    mass_gap = logsumexp(good, axis=-1) - logsumexp(bad, axis=-1)
    # Pooled safe mass alone can mask an unsafe argmax when split over labels.
    winner_gap = jnp.max(good, axis=-1) - jnp.max(bad, axis=-1)
    deficit = jax.nn.relu(MARGIN - jnp.minimum(mass_gap, winner_gap))
    quadratic = jnp.minimum(deficit, SLOPE_CAP)
    loss = 0.5 * quadratic**2 + SLOPE_CAP * (deficit - quadratic)
    return jnp.sum(jnp.where(enabled, loss, 0.0))


@dataclass(frozen=True)
class MarginStep:
    pairs: tuple
    allowed: tuple
    checked: int

    def value(self, decoded):
        if not self.pairs:
            return jnp.zeros((), dtype=decoded.bonds.dtype)
        n = len(self.pairs)
        # Both sides of every logsumexp remain finite on padded rows.
        padding = MAX_ROWS - n
        i, j = jnp.asarray(self.pairs + ((0, 1),) * padding).T
        safe = jnp.asarray(self.allowed + ((True, False, False, False, False),) * padding)
        logits = decoded.symmetric_bond_logits[0, i, j]
        return WEIGHT_PER_REPAIR_UNIT * margin_value(logits, safe, jnp.arange(MAX_ROWS) < n)

    def record(self):
        return {"checked_rows": self.checked, "pairs": self.pairs, "allowed": self.allowed}


@lru_cache(maxsize=512)
def allowed_labels(spec, atoms, charges, raw_bonds, left, right):
    n = spec.max_nodes
    bonds = np.asarray(raw_bonds, dtype=np.int64).reshape(n, n).copy()
    original = bonds[left, right]
    allowed = []
    for label in range(len(spec.bond_names)):
        if label == original:  # Already verified by the original preparation.
            allowed.append(True)
            continue
        bonds[left, right] = bonds[right, left] = label
        smiles = inspect_graph(atoms, charges, bonds, spec).smiles
        try:
            allowed.append(smiles is not None and encode_smiles(smiles, spec)["smiles"] == smiles)
        except ValueError:
            allowed.append(False)
    return tuple(allowed)


def prepare_margin(decoded, prepared, spec):
    """Freeze valid one-bond choices at the source, outside autodiff."""
    if prepared.arrays["route"].shape != (1,) or len(spec.bond_names) != 5:
        raise ValueError("The pilot requires a single molecule and the five bond labels")
    if int(prepared.arrays["route"][0]) != 0:
        return MarginStep((), (), 0)
    arrays = prepared.arrays
    atoms, charges, bonds = (np.asarray(arrays[name][0]) for name in
                             ("target_atoms", "target_charges", "target_bonds"))
    i, j = np.triu_indices(spec.max_nodes, 1)
    logits = np.asarray(decoded.symmetric_bond_logits)[0, i, j]
    if not np.isfinite(logits).all():
        raise FloatingPointError("Nonfinite source bond logits")
    winners = logits[np.arange(len(i)), bonds[i, j]]
    rivals = np.where(np.arange(logits.shape[-1]) != bonds[i, j, None], logits, -np.inf)
    worst_gap = winners - np.logaddexp.reduce(rivals, axis=-1)
    # Even if every rival is unsafe, confident rows have no margin penalty.
    # A small screening buffer also covers nearby trials using the fixed mask.
    eligible = np.flatnonzero((atoms[i] != 0) & (atoms[j] != 0) & (worst_gap <= MARGIN + 0.15))
    selected = eligible[np.argsort(worst_gap[eligible], kind="stable")[:MAX_ROWS]]
    atom_key, charge_key, bond_key = tuple(atoms.tolist()), tuple(charges.tolist()), tuple(bonds.ravel().tolist())
    pairs, masks = [], []
    for row in selected:
        pair = int(i[row]), int(j[row])
        mask = allowed_labels(spec, atom_key, charge_key, bond_key, *pair)
        if not all(mask):
            pairs.append(pair)
            masks.append(mask)
    return MarginStep(tuple(pairs), tuple(masks), len(selected))
