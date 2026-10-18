"""JAX port of fixed-target failure guidance; no chemistry conversion in AD."""

from dataclasses import asdict, dataclass
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from ._reference.chemistry_regularization import ALLOWED_VALENCES
from .state import validate_decoded_shape


@dataclass(frozen=True)
class FailureGuidanceConfig:
    version: str = "repair-failure-guidance-v1"
    extra_valid_weight: float = 1.0
    partial_weight: float = 1.0
    local_bond_weight: float = 0.1
    component_cut_weight: float = 0.1
    nonempty_weight: float = 0.1

    def __post_init__(self):
        if self.version != "repair-failure-guidance-v1":
            raise ValueError("Unknown failure guidance version")
        for k, v in asdict(self).items():
            if k != "version" and (
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
            ):
                raise ValueError("Invalid failure guidance coefficient")
        if (
            min(self.local_bond_weight, self.component_cut_weight, self.nonempty_weight)
            <= 0
        ):
            raise ValueError("Failure risks must be explicitly enabled")

    def weights(self, dtype=None):
        return jnp.asarray(
            [
                self.extra_valid_weight,
                self.partial_weight,
                self.local_bond_weight,
                self.component_cut_weight,
                self.nonempty_weight,
            ],
            dtype=dtype,
        )


def _safe_lse(terms, axis=-1):
    """Unreachable log-DP states have zero cotangent, as in the source."""
    supported = jnp.any(jnp.isfinite(terms), axis=axis, keepdims=True)
    value = logsumexp(jnp.where(supported, terms, 0.0), axis=axis)
    return jnp.where(jnp.squeeze(supported, axis), value, -jnp.inf)


def _log_any_present(absent, present):
    prefix = jnp.concatenate(
        (jnp.zeros_like(absent[..., :1]), jnp.cumsum(absent, -1)[..., :-1]), -1
    )
    return _safe_lse(prefix + present)


def _local(log_bonds, active, caps, order_indices):
    n = active.shape[-1]
    maximum = max(
        v
        for rule in ALLOWED_VALENCES.values()
        for allowed in rule.values()
        if allowed
        for v in allowed
    )
    state = [jnp.zeros_like(caps, dtype=log_bonds.dtype)] + [
        jnp.full(caps.shape, -jnp.inf, log_bonds.dtype) for _ in range(maximum)
    ]
    none, single, double, triple, aromatic = order_indices
    for j in range(n):
        edge = log_bonds[:, :, j]
        orders = (
            edge[..., none],
            logsumexp(edge[..., jnp.asarray([single, aromatic])], -1),
            edge[..., double],
            edge[..., triple],
        )
        include = active & active[:, j, None] & (jnp.arange(n) != j)[None]
        orders = tuple(
            jnp.where(include, v, 0.0 if k == 0 else -jnp.inf)
            for k, v in enumerate(orders)
        )
        state = [
            _safe_lse(
                jnp.stack([state[v - k] + orders[k] for k in range(min(3, v) + 1)], -1)
            )
            for v in range(maximum + 1)
        ]
    all_states = jnp.stack(state, -1)
    allowed = jnp.arange(maximum + 1) <= jnp.maximum(caps[..., None], 0)
    mass = _safe_lse(jnp.where(allowed, all_states, -jnp.inf))
    # Strict > preserves Torch clamp_max's boundary derivative, rather than
    # the half-gradient chosen by minimum at an exact equality.
    risk = jnp.where(active & (caps >= 0), jnp.where(mass > 0, 0.0, -mass), 0.0)
    return jnp.sum(risk, -1) / jnp.maximum(jnp.sum(active, -1), 1).astype(
        log_bonds.dtype
    )


def _terms(decoded, arrays, weights, order_indices):
    la, lq = (
        jax.nn.log_softmax(decoded.atoms, -1),
        jax.nn.log_softmax(decoded.charges, -1),
    )
    lb = jax.nn.log_softmax(decoded.symmetric_bond_logits, -1)
    active, route = arrays["active"], arrays["route"]
    n = active.shape[-1]
    i, j = jnp.triu_indices(n, 1)

    def get(p, labels):
        return jnp.take_along_axis(p, labels[..., None], -1)[..., 0]

    atom = jnp.sum(get(la, arrays["target_atoms"]), -1)
    charge = jnp.sum(jnp.where(active, get(lq, arrays["target_charges"]), 0.0), -1)
    bond = get(lb, arrays["target_bonds"])
    pair = active[:, i] & active[:, j]
    k = jnp.sum(active, -1)
    full = -(atom + charge + jnp.sum(jnp.where(pair, bond[:, i, j], 0.0), -1)) / (
        n + k + k * (k - 1) // 2
    ).astype(la.dtype)
    partial_mask = arrays["changed"][:, i, j]
    partial = -jnp.sum(jnp.where(partial_mask, bond[:, i, j], 0.0), -1) / jnp.maximum(
        jnp.sum(partial_mask, -1), 1
    ).astype(lb.dtype)
    fallback = (route == 3) | (route == 4)
    local = _local(lb, active, arrays["caps"], order_indices)
    # Cuts are frozen unordered edge sets, one per original component.
    edge = lb[:, i, j]
    none = order_indices[0]
    present_indices = jnp.asarray([v for v in range(lb.shape[-1]) if v != none])
    absent = jnp.where(arrays["cuts"], edge[:, None, :, none], 0.0)
    present = jnp.where(
        arrays["cuts"], logsumexp(edge[..., present_indices], -1)[:, None], -jnp.inf
    )
    cut = -_log_any_present(absent, present)
    cut = jnp.sum(jnp.where(arrays["cut_active"], cut, 0.0), -1) / jnp.maximum(
        jnp.sum(arrays["cut_active"], -1), 1
    ).astype(lb.dtype)
    nonempty = -_log_any_present(la[..., 0], logsumexp(la[..., 1:], -1))
    values = {
        "valid_full_nll": jnp.mean(jnp.where(route == 1, full, 0.0)),
        "extra_valid_nll": jnp.mean(jnp.where(route == 2, full, 0.0)),
        "partial_edited_nll": jnp.mean(jnp.where(route == 3, partial, 0.0)),
        "local_bond_risk": jnp.mean(jnp.where(fallback, local, 0.0)),
        "component_cut_risk": jnp.mean(jnp.where(fallback, cut, 0.0)),
        "nonempty_risk": jnp.mean(jnp.where(fallback & (k == 0), nonempty, 0.0)),
    }
    total = values["valid_full_nll"] + jnp.dot(
        weights,
        jnp.stack(
            [
                values[key]
                for key in (
                    "extra_valid_nll",
                    "partial_edited_nll",
                    "local_bond_risk",
                    "component_cut_risk",
                    "nonempty_risk",
                )
            ]
        ),
    )
    return total, values


_compiled_terms = jax.jit(_terms, static_argnames=("order_indices",))


def failure_aware_repair_loss(decoded, prepared, spec, config=None):
    """Unfloored log-softmax NLL and risks on one immutable prepared batch."""
    config = FailureGuidanceConfig() if config is None else config
    if not isinstance(config, FailureGuidanceConfig):
        raise TypeError("Require FailureGuidanceConfig")
    prepared.verify(spec)
    validate_decoded_shape(decoded, spec, batch=prepared.receipt["attempted_rows"])
    decoded = decoded.math_precision()
    if set(spec.bond_names) != {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Require the exact declared five bond categories")
    indices = tuple(
        spec.bond_names.index(k)
        for k in ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    )
    return _compiled_terms(
        decoded, prepared.arrays, config.weights(decoded.atoms.dtype), indices
    )


def population_js(counts, reference, smoothing=1e-8):
    """Source-equivalent aggregate-count JS, with declared probability smoothing."""
    if (
        counts.ndim != 1
        or counts.size < 2
        or not jnp.issubdtype(counts.dtype, jnp.floating)
    ):
        raise ValueError("Population counts require a floating category vector")
    if not math.isfinite(smoothing) or not 0 < smoothing < 1:
        raise ValueError("Require declared probability smoothing in (0, 1)")
    reference = jax.lax.stop_gradient(jnp.asarray(reference, dtype=counts.dtype))
    if reference.shape != counts.shape:
        raise ValueError("Composition reference vocabulary differs")
    total = jnp.sum(counts)
    p = counts / jnp.where(total > 0, total, 1.0)
    p = jnp.where(total > 0, p, jnp.ones_like(p) / p.size)
    q = reference / jnp.sum(reference)
    p, q = p + smoothing, q + smoothing
    p, q = p / jnp.sum(p), q / jnp.sum(q)
    middle = (p + q) * 0.5
    return 0.5 * (
        jnp.sum(p * (jnp.log(p) - jnp.log(middle)))
        + jnp.sum(q * (jnp.log(q) - jnp.log(middle)))
    )


def composition_statistics(decoded):
    decoded = decoded.math_precision()
    atoms, charges = (
        jax.nn.softmax(decoded.atoms, -1),
        jax.nn.softmax(decoded.charges, -1),
    )
    occupancy = jnp.sum(atoms[..., 1:], -1)
    return {
        "elements": jnp.sum(atoms[..., 1:], (0, 1)),
        "charges": jnp.sum(charges * occupancy[..., None], (0, 1)),
        "occupied_mass": jnp.sum(occupancy),
    }
