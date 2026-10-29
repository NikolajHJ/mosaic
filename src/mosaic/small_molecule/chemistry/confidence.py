"""Opt-in confidence guidance for a detached, already-valid source graph.

This is not a validity oracle and does not replace failure-aware repair. Labels,
validity and occupancy come only from the authenticated step preparation; a
trial's argmax is never used to select its own target. No atom valence enters
this loss. The coefficient is independent of the repair schedule.
"""

from dataclasses import asdict, dataclass
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .state import validate_decoded_shape


@dataclass(frozen=True)
class ConfidenceMarginConfig:
    """Predeclared experimental policy, disabled unless explicitly supplied.

    The target-versus-all-alternatives log-odds margin is in logit units.
    A Huber hinge caps d(loss)/d(margin deficit) at ``slope_cap``. The
    nonnegative loss itself is linear, not bounded, at large deficits.
    """

    version: str = "valid-confidence-huber-v1"
    weight: float = 0.01
    margin: float = 0.2
    slope_cap: float = 1.0

    def __post_init__(self):
        if self.version != "valid-confidence-huber-v1":
            raise ValueError("Unknown confidence margin version")
        for name in ("weight", "margin", "slope_cap"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Confidence parameters must be finite real scalars")
            if value < 0 or (name == "slope_cap" and value == 0):
                raise ValueError("Require nonnegative weight/margin and positive slope cap")

    def record(self):
        return asdict(self)


def _row_hinge(logits, labels, margin, slope_cap):
    labels = jax.lax.stop_gradient(labels)
    target = jnp.take_along_axis(logits, labels[..., None], -1)[..., 0]
    other = jnp.arange(logits.shape[-1]) != labels[..., None]
    rival = logsumexp(jnp.where(other, logits, -jnp.inf), -1)
    deficit = jax.nn.relu(margin + rival - target)
    # This algebra avoids squaring an arbitrarily large deficit. It also has
    # the same derivative from both sides at deficit == slope_cap.
    quadratic = jnp.minimum(deficit, slope_cap)
    loss = 0.5 * quadratic**2 + slope_cap * (deficit - quadratic)
    return loss, deficit > 0


@jax.jit
def _terms(decoded, arrays, margin, slope_cap):
    valid = jax.lax.stop_gradient(arrays["route"] == 0)
    active = jax.lax.stop_gradient(arrays["active"])
    atom_mask = valid[:, None] & active
    n = active.shape[-1]
    i, j = jnp.triu_indices(n, 1)
    bond_mask = valid[:, None] & active[:, i] & active[:, j]
    atoms, atom_uncertain = _row_hinge(
        decoded.atoms, arrays["target_atoms"], margin, slope_cap
    )
    bonds, bond_uncertain = _row_hinge(
        decoded.symmetric_bond_logits[:, i, j],
        arrays["target_bonds"][:, i, j], margin, slope_cap,
    )

    def mean_rows(values, mask):
        # Average each field within each molecule, then across the batch.
        # Invalid source rows contribute zero, rather than changing scaling.
        count = jnp.maximum(jnp.sum(mask, -1), 1)
        return jnp.mean(jnp.sum(jnp.where(mask, values, 0.0), -1) / count)

    atom_loss, bond_loss = mean_rows(atoms, atom_mask), mean_rows(bonds, bond_mask)
    return 0.5 * (atom_loss + bond_loss), {
        "confidence_atom_loss": atom_loss,
        "confidence_bond_loss": bond_loss,
        "confidence_active_atom_fraction": mean_rows(atom_uncertain, atom_mask),
        "confidence_active_bond_fraction": mean_rows(bond_uncertain, bond_mask),
        "confidence_valid_source_fraction": jnp.mean(valid.astype(decoded.atoms.dtype)),
    }


def valid_confidence_margin(decoded, prepared, spec, config=ConfidenceMarginConfig()):
    """Return unweighted loss and diagnostics using fixed, authenticated targets.

    Already-valid SOURCE rows only; repaired witnesses are intentionally not
    eligible. Charge logits and inactive slots/edges receive no guidance. For
    each included categorical row, the absolute target derivative and sum of
    rival derivatives are each <= slope_cap before field/batch averaging.
    Frozen labels never constrain the optimizer's native atom/bond variables.
    """
    if not isinstance(config, ConfidenceMarginConfig):
        raise TypeError("Require explicit ConfidenceMarginConfig")
    prepared.verify(spec)
    validate_decoded_shape(decoded, spec, batch=prepared.arrays["route"].shape[0])
    decoded = decoded.math_precision()
    return _terms(decoded, prepared.arrays, config.margin, config.slope_cap)
