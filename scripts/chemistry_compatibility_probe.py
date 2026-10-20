"""Experimental saved-state chemistry probes; not a production loss or default.

All slots are occupied and neutral. Element order is C,N,O,S,F,Cl; unordered
bonds are NONE,SINGLE,DOUBLE,TRIPLE,AROMATIC. The local factorized surrogate
uses maximum heavy-atom valence (implicit H permitted), with aromatic counted
as order one. It does not establish connectivity or native chemical validity.
Numerical execution belongs on allocated DTU HPC. Nothing here loads a model.
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np


CAPS = (4, 3, 2, 6, 1, 1)
MARGIN = 0.2


def _inputs(atom_logits, bond_logits):
    if atom_logits.ndim != 2 or atom_logits.shape[1] != 6:
        raise ValueError("Require occupied neutral atom logits [N,6]")
    n = atom_logits.shape[0]
    if n < 2 or bond_logits.shape != (n * (n - 1) // 2, 5):
        raise ValueError("Require N>=2 and unordered bond logits [N*(N-1)/2,5]")
    if (
        atom_logits.dtype not in (jnp.float32, jnp.float64)
        or bond_logits.dtype != atom_logits.dtype
    ):
        raise TypeError("Require common FP32/FP64 logits; caller checks finite values")
    return n


def validate_labels(labels, *, n_atoms, bonds=False):
    """Validate concrete frozen targets on the host, outside differentiation.

    Comparator labels must be captured constants, not traced/differentiated
    inputs. No scorer may silently derive or replace them with an argmax.
    """
    labels = np.asarray(labels)
    shape = (n_atoms * (n_atoms - 1) // 2,) if bonds else (n_atoms,)
    categories = 5 if bonds else 6
    if labels.shape != shape or labels.dtype.kind not in "iu":
        raise ValueError("Frozen label shape/integer dtype differs")
    if np.any(labels < 0) or np.any(labels >= categories):
        raise ValueError("Frozen target outside categorical vocabulary")
    return np.array(labels, dtype=np.int32, copy=True)


def _safe_lse(terms, axis=-1):
    # Unreachable DP states have zero cotangent instead of NaN derivatives.
    supported = jnp.any(jnp.isfinite(terms), axis=axis, keepdims=True)
    value = logsumexp(jnp.where(supported, terms, 0.0), axis=axis)
    return jnp.where(jnp.squeeze(supported, axis), value, -jnp.inf)


def _valence_log_mass(bond_logits, n):
    lb = jax.nn.log_softmax(bond_logits, -1)
    orders = jnp.stack(
        (lb[:, 0], logsumexp(lb[:, jnp.asarray([1, 4])], -1), lb[:, 2], lb[:, 3]), -1
    )
    pairs = tuple((i, j) for i in range(n) for j in range(i + 1, n))
    masses = []
    for node in range(n):
        state = [jnp.asarray(0.0, lb.dtype)] + [jnp.asarray(-jnp.inf, lb.dtype)] * 6
        for edge, pair in enumerate(pairs):
            if node not in pair:
                continue
            state = [
                _safe_lse(
                    jnp.stack(
                        [state[v - k] + orders[edge, k] for k in range(min(3, v) + 1)]
                    )
                )
                for v in range(7)
            ]
        masses.append(jnp.stack(state))
    # Sums above six cannot satisfy any supported neutral cap; discard them.
    return jnp.stack(masses)


def _cap_log_mass(bond_logits, n):
    mass = _valence_log_mass(bond_logits, n)
    permitted = jnp.arange(7)[None, :] <= jnp.asarray(CAPS)[:, None]
    return _safe_lse(jnp.where(permitted[None], mass[:, None, :], -jnp.inf))


def _negative_log_probability(log_mass):
    # A probability cannot exceed one; discard only positive roundoff excursions.
    return jnp.where(log_mass > 0.0, 0.0, -log_mass)


def loss(atom_logits, bond_logits):
    """Mean -log(sum_a P(atom_i=a) P(incident order sum<=cap_a)).

    This is a local compatibility mass under independent categorical marginals,
    not the probability that the shared graph is globally chemically valid.
    It can favor high-cap atoms such as sulfur. All atom/bond logits remain live.
    """
    n = _inputs(atom_logits, bond_logits)
    local = _cap_log_mass(bond_logits, n)
    joint = _safe_lse(jax.nn.log_softmax(atom_logits, -1) + local)
    return jnp.mean(_negative_log_probability(joint))


def hard_cap_loss(atom_logits, bond_logits, hard_labels):
    """Comparator: identical DP, conditioned on frozen host atom labels.

    No direct atom derivative exists; atom_logits supplies shape/precision only.
    Captured labels must be validated host constants before calling grad/JIT.
    """
    n = _inputs(atom_logits, bond_logits)
    labels = validate_labels(hard_labels, n_atoms=n)
    local = _cap_log_mass(bond_logits, n)
    return jnp.mean(
        _negative_log_probability(local[jnp.arange(n), jnp.asarray(labels)])
    )


def valid_margin_loss(atom_logits, bond_logits, target_atoms, target_bonds):
    """Comparator: frozen valid-label margin, not full categorical sharpening.

    Mean atom and mean bond squared hinges receive equal weight. The rival is
    logsumexp of all non-target logits. This cannot determine target validity;
    callers must authenticate that separately. It can inhibit valid exploration.
    """
    n = _inputs(atom_logits, bond_logits)
    atoms = validate_labels(target_atoms, n_atoms=n)
    bonds = validate_labels(target_bonds, n_atoms=n, bonds=True)

    def penalty(logits, labels):
        labels = jnp.asarray(labels)
        target = jnp.take_along_axis(logits, labels[:, None], -1)[:, 0]
        rival = logsumexp(
            jnp.where(jnp.arange(logits.shape[1]) == labels[:, None], -jnp.inf, logits),
            -1,
        )
        return jnp.mean(jax.nn.relu(MARGIN + rival - target) ** 2)

    return 0.5 * (penalty(atom_logits, atoms) + penalty(bond_logits, bonds))
