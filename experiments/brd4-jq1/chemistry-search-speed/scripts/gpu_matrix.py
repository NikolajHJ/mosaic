"""Separate JAX matrix/scan experiments; no native chemistry oracle replacement.

``chemistry-matrix-acceleration-v1`` has two distinct scientific contracts:

* Integer matrix features are exact graph bookkeeping. They do not establish
  RDKit sanitization, aromatic chemistry, schema roundtrip or molecule validity.
  They must not prune native checks or supply replacement repair targets.
* Local-risk scan/matrix kernels compute the existing frozen-cap probability
  dynamic program. Scan keeps each per-state reduction's term order. The
  log-semiring matrix variant uses balanced associative products and changes
  FP32 association. Both need the original predeclared 2e-5 value/gradient
  tolerance and independent finite differences on allocated hardware; neither
  is assumed equivalent or faster before those measurements.

Host/device transfers, compilation and synchronized warm execution must be
timed separately. ``require_backend('gpu')`` rejects CPU fallback. No existing
run/source is modified; the context manager is an isolated benchmark adapter.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from mosaic.small_molecule.chemistry._reference.chemistry_regularization import (
    ALLOWED_VALENCES,
)


VERSION = "chemistry-matrix-acceleration-v1"
PARITY_TOLERANCE = 2e-5
_INSTALLED = False
_MAXIMUM = max(
    value
    for rule in ALLOWED_VALENCES.values()
    for allowed in rule.values()
    if allowed
    for value in allowed
)


def policy():
    """Versioned interpretation; this is not a performance or validity result."""
    return {
        "version": VERSION,
        "integer_features": "exact_slot_graph_bookkeeping_not_native_validity",
        "native_oracle_replaced": False,
        "search_pruning_authorized": False,
        "local_scan": "same_DP_state_and_per_state_term_order",
        "local_matrix": "balanced_log_semiring_transition_products_FP32_reassociated",
        "value_gradient_parity_tolerance": PARITY_TOLERANCE,
        "finite_difference_gate_required": True,
        "GPU_speedup_established": False,
        "timing": "separate_transfer_compilation_and_synchronized_warm_execution",
    }


def require_backend(expected="gpu"):
    """Require an explicit actual JAX backend; never relabel CPU execution GPU."""
    if expected not in ("cpu", "gpu"):
        raise ValueError("Require explicit cpu or gpu backend")
    actual = jax.default_backend()
    if actual != expected:
        raise RuntimeError(f"Require JAX {expected} backend, observed {actual}")
    devices = jax.devices(expected)
    if not devices or any(device.platform != expected for device in devices):
        raise RuntimeError(f"Actual JAX devices do not establish {expected} execution")
    return actual


def _matrix_kernel(
    atoms, charges, bonds, *, atom_categories, charge_categories, bond_categories,
    neutral, order_values, aromatic_index, cap_rows,
):
    batch, n = atoms.shape
    active = atoms != 0
    pair = active[:, :, None] & active[:, None, :]
    identity = jnp.eye(n, dtype=bool)[None]
    # For component telemetry, match the original undirected component walk:
    # either directed entry suffices, while asymmetry is separately disclosed.
    adjacent = pair & ((bonds != 0) | (jnp.swapaxes(bonds, -1, -2) != 0))
    reach = adjacent | (identity & pair)

    def closure(k, value):
        return value | (value[:, :, k, None] & value[:, None, k, :])

    reach = jax.lax.fori_loop(0, n, closure, reach)
    slots = jnp.arange(n, dtype=jnp.int32)
    representative = jnp.min(jnp.where(reach, slots[None, None, :], n), axis=-1)
    is_representative = active & (representative == slots[None])
    sizes = jnp.sum(reach, axis=-1, dtype=jnp.int32)
    atom_ok = (atoms >= 0) & (atoms < atom_categories)
    charge_ok = (charges >= 0) & (charges < charge_categories)
    bond_ok = (bonds >= 0) & (bonds < bond_categories)
    order_table = jnp.asarray(order_values, dtype=jnp.int32)
    safe_bonds = jnp.clip(bonds, 0, bond_categories - 1)
    orders = order_table[safe_bonds]
    order_sum = jnp.sum(jnp.where(pair & bond_ok, orders, 0), axis=-1, dtype=jnp.int32)
    aromatic = pair & (bonds == aromatic_index)
    aromatic_count = jnp.sum(aromatic, axis=-1, dtype=jnp.int32)
    caps_table = jnp.asarray(cap_rows, dtype=jnp.int32)
    caps = caps_table[
        jnp.clip(atoms, 0, atom_categories - 1),
        jnp.clip(charges, 0, charge_categories - 1),
    ]
    cap_supported = active & atom_ok & charge_ok & (caps >= 0)
    valence_counter_supported = cap_supported & (aromatic_count == 0) & jnp.all(
        ~pair | bond_ok, axis=-1
    )
    return {
        "active": active,
        "active_pairs": pair,
        "active_counts": jnp.sum(active, -1, dtype=jnp.int32),
        "reachability": reach,
        "component_representatives": jnp.where(active, representative, -1),
        "component_count": jnp.sum(is_representative, -1, dtype=jnp.int32),
        "component_sizes_at_representatives": jnp.where(is_representative, sizes, 0),
        "atom_category_ok": jnp.all(atom_ok, axis=-1),
        "active_charge_category_ok": jnp.all(~active | charge_ok, axis=-1),
        "active_bond_category_ok": jnp.all(~pair | bond_ok, axis=(-2, -1)),
        "active_bond_symmetry_ok": jnp.all(
            ~pair | (bonds == jnp.swapaxes(bonds, -1, -2)), axis=(-2, -1)
        ),
        "active_diagonal_zero": jnp.all(~(identity & pair) | (bonds == 0), axis=(-2, -1)),
        "padding_charge_neutral": jnp.all(active | (charges == neutral), axis=-1),
        "padding_bonds_zero": jnp.all(pair | (bonds == 0), axis=(-2, -1)),
        "nonaromatic_bond_order_sum": order_sum,
        "aromatic_incident_count": aromatic_count,
        "declared_rule_caps": jnp.where(cap_supported, caps, -1),
        "valence_counter_supported": valence_counter_supported,
        "declared_cap_excess": jnp.where(
            valence_counter_supported, jnp.maximum(order_sum - caps, 0), -1
        ),
    }


_compiled_matrix = jax.jit(
    _matrix_kernel,
    static_argnames=(
        "atom_categories", "charge_categories", "bond_categories", "neutral",
        "order_values", "aromatic_index", "cap_rows",
    ),
)


def matrix_features(atoms, charges, bonds, spec, *, require_gpu=False):
    """Return batched integer/boolean telemetry; never a native-validity flag.

    Arrays must have shapes BxN, BxN and BxNxN with N=spec.max_nodes. Invalid
    categories are represented by flags rather than silently treated as valid.
    Aromatic bonds are counted separately and excluded from numeric order sums.
    Cap excess is -1 where this simple nonaromatic counter is unsupported.
    """
    if type(require_gpu) is not bool:
        raise TypeError("require_gpu must be an explicit boolean")
    if require_gpu:
        require_backend("gpu")
    atoms, charges, bonds = map(jnp.asarray, (atoms, charges, bonds))
    n = spec.max_nodes
    if (
        atoms.ndim != 2 or atoms.shape[0] < 1 or atoms.shape[1] != n
        or charges.shape != atoms.shape or bonds.shape != atoms.shape + (n,)
        or any(value.dtype.kind not in "iu" for value in (atoms, charges, bonds))
    ):
        raise ValueError("Require nonempty batched integer graphs matching the schema")
    if set(spec.bond_names) != {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Require the explicit five-category bond schema")
    if require_gpu:
        device = jax.devices("gpu")[0]
        atoms, charges, bonds = (
            jax.device_put(value, device) for value in (atoms, charges, bonds)
        )
    order_lookup = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 0}
    cap_rows = tuple(
        tuple(
            max(allowed) if allowed else -1
            for charge in spec.formal_charges
            for allowed in (ALLOWED_VALENCES.get(number, {}).get(charge),)
        )
        for number in spec.atomic_numbers
    )
    result = _compiled_matrix(
        atoms, charges, bonds,
        atom_categories=len(spec.atomic_numbers),
        charge_categories=len(spec.formal_charges),
        bond_categories=len(spec.bond_names),
        neutral=spec.formal_charges.index(0),
        order_values=tuple(order_lookup[name] for name in spec.bond_names),
        aromatic_index=spec.bond_names.index("AROMATIC"),
        cap_rows=cap_rows,
    )
    if require_gpu and any(
        not value.devices() or any(device.platform != "gpu" for device in value.devices())
        for value in jax.tree_util.tree_leaves(result)
    ):
        raise RuntimeError("Matrix features did not execute on actual GPU devices")
    return result


def _safe_lse(terms, axis=-1):
    # Exactly the original unreachable-state zero-cotangent convention.
    supported = jnp.any(jnp.isfinite(terms), axis=axis, keepdims=True)
    value = logsumexp(jnp.where(supported, terms, 0.0), axis=axis)
    return jnp.where(jnp.squeeze(supported, axis), value, -jnp.inf)


def _risk_from_states(all_states, log_bonds, active, caps):
    allowed = jnp.arange(_MAXIMUM + 1) <= jnp.maximum(caps[..., None], 0)
    mass = _safe_lse(jnp.where(allowed, all_states, -jnp.inf))
    risk = jnp.where(active & (caps >= 0), jnp.where(mass > 0, 0.0, -mass), 0.0)
    return jnp.sum(risk, -1) / jnp.maximum(jnp.sum(active, -1), 1).astype(log_bonds.dtype)


def local_valence_scan(log_bonds, active, caps, order_indices):
    """Existing local-risk DP in lax.scan, preserving per-state summation order."""
    n = active.shape[-1]
    initial = jnp.stack(
        [jnp.zeros_like(caps, dtype=log_bonds.dtype)]
        + [jnp.full(caps.shape, -jnp.inf, log_bonds.dtype) for _ in range(_MAXIMUM)],
        axis=0,
    )
    none, single, double, triple, aromatic = order_indices

    def advance(state, j):
        edge = log_bonds[:, :, j]
        orders = (
            edge[..., none],
            logsumexp(edge[..., jnp.asarray([single, aromatic])], -1),
            edge[..., double],
            edge[..., triple],
        )
        include = active & active[:, j, None] & (jnp.arange(n) != j)[None]
        orders = tuple(
            jnp.where(include, value, 0.0 if k == 0 else -jnp.inf)
            for k, value in enumerate(orders)
        )
        next_state = jnp.stack(
            [
                _safe_lse(jnp.stack(
                    [state[v - k] + orders[k] for k in range(min(3, v) + 1)], -1
                ))
                for v in range(_MAXIMUM + 1)
            ],
            axis=0,
        )
        return next_state, None

    state, _ = jax.lax.scan(advance, initial, jnp.arange(n))
    return _risk_from_states(jnp.moveaxis(state, 0, -1), log_bonds, active, caps)


def _semiring_product(left, right):
    """Log-semiring matrix product, with unreachable states' zero cotangents."""
    return _safe_lse(left[..., :, :, None] + right[..., None, :, :], axis=-2)


def local_valence_matrix(log_bonds, active, caps, order_indices):
    """Balanced log-semiring transition matrices; FP32 association differs.

    Each edge is a Toeplitz transition T[old,new] for bond orders 0..3.
    Matrix products combine independent edges before reading initial-state row0.
    This is an acceleration candidate, not an adopted chemistry-loss change.
    """
    n = active.shape[-1]
    none, single, double, triple, aromatic = order_indices
    # Edge axis first; remaining axes are batch, receiving atom and category.
    edges = jnp.moveaxis(log_bonds, 2, 0)
    orders = jnp.stack(
        (
            edges[..., none],
            logsumexp(edges[..., jnp.asarray([single, aromatic])], -1),
            edges[..., double],
            edges[..., triple],
        ),
        axis=-1,
    )
    slots = jnp.arange(n)
    include = (
        active[None, :, :]
        & jnp.swapaxes(active, 0, 1)[:, :, None]
        & (slots[:, None, None] != slots[None, None, :])
    )
    identity_orders = jnp.asarray([0.0, -jnp.inf, -jnp.inf, -jnp.inf], log_bonds.dtype)
    orders = jnp.where(include[..., None], orders, identity_orders)
    states = jnp.arange(_MAXIMUM + 1)
    delta = states[None, :] - states[:, None]
    transitions = jnp.take(orders, jnp.clip(delta, 0, 3), axis=-1)
    transitions = jnp.where((delta >= 0) & (delta <= 3), transitions, -jnp.inf)
    products = jax.lax.associative_scan(_semiring_product, transitions, axis=0)
    all_states = products[-1, ..., 0, :]
    return _risk_from_states(all_states, log_bonds, active, caps)


@contextmanager
def install_matrix_loss(method="scan"):
    """Temporarily replace only the local-risk implementation and retrace terms.

    Restoring `_local` alone cannot invalidate an existing compiled JAX trace;
    this context also binds a fresh JIT and restores both exact original objects.
    No RDKit/search/preparation function or other loss term is replaced.
    """
    global _INSTALLED
    if method not in ("scan", "matrix"):
        raise ValueError("Require scan or matrix local-risk method")
    if _INSTALLED:
        raise RuntimeError("Matrix-loss adapters cannot be nested")
    from mosaic.small_molecule.chemistry import loss

    old_local, old_compiled = loss._local, loss._compiled_terms
    _INSTALLED = True
    try:
        loss._local = local_valence_scan if method == "scan" else local_valence_matrix

        def selected_terms(decoded, arrays, weights, order_indices):
            # A NEW Python function identity is essential. Re-jitting the same
            # loss._terms object may hit a global cache of its old _local trace.
            return loss._terms(decoded, arrays, weights, order_indices)

        loss._compiled_terms = jax.jit(selected_terms, static_argnames=("order_indices",))
        yield {**policy(), "method": method, "native_oracle_unchanged": True}
    finally:
        loss._local, loss._compiled_terms = old_local, old_compiled
        _INSTALLED = False
