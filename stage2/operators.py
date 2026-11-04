"""Stage 2 chemistry operators: detached, explicit, applied after each update.

These are operators, not losses. The protocol's Stage 2 loop is

    AF3 gradient step -> simplex projection -> valence projection

with connectivity and sparsity conditional on a demonstrated failure. Nothing
here touches the objective, the model wrapper or the optimizer, and nothing
here performs RDKit edits: the optimization state stays continuous and hard
molecules remain readouts.

The valence projection, component bridging and weak-edge shrinkage are carried
over from ``src/mosaic/small_molecule/feasibility.py`` (local Stage 2 work,
17 October 2026). The dual solve is unchanged in substance. What is new here is
the composition into one ``post_step``, the invariant checks of protocol 5.1,
and the five individually toggleable improvements, all neutral by default.

Arrays are plain NumPy. ``atoms`` is ``X`` with shape ``[N, 6]`` over
(C, N, O, S, F, Cl); ``edges`` is ``E_upper`` with shape ``[N(N-1)/2, 5]`` over
(NO_BOND, SINGLE, DOUBLE, TRIPLE, AROMATIC) in lexicographic ``i < j`` order.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.optimize import minimize

ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
BOND_STATES = ("NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")

#: Neutral valence ceilings, protocol 5.2. Charge is fixed at zero in V0.
NEUTRAL_VALENCE = np.array([4.0, 3.0, 2.0, 2.0, 1.0, 1.0])

#: Operator sets. ``sparse_only`` is off the protocol's ladder on purpose: it
#: isolates weak-edge shrinkage from component bridging, which the ladder
#: cannot express because ``sparse`` includes both.
MODES = ("none", "valence", "connected", "sparse_only", "sparse")

#: Which operators each mode applies, after the fluidity floor and before the
#: final valence reprojection. Valence is in every mode but ``none``.
BRIDGING_MODES = frozenset({"connected", "sparse"})
SHRINKING_MODES = frozenset({"sparse_only", "sparse"})


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Stage2Config:
    """One experiment arm. Defaults are the protocol baseline, nothing added.

    ``mode`` selects the operator set and follows the protocol's experiment
    order: ``none`` is the frozen Stage 1 control, ``valence`` is Experiment A
    (5.2), ``connected`` is Experiment B (+5.3), ``sparse`` is Experiment C
    (+5.3 +5.4). ``sparse_only`` adds 5.4 without 5.3, which the protocol's
    ladder does not cover but which isolates shrinkage from bridging.
    Hardening (Experiment D) is a separate run, not an operator; see
    :mod:`run`.

    The five fields below ``sparsity_threshold`` are the suggested improvements.
    Every one of them is neutral at its default, so ``Stage2Config(mode="valence")``
    is exactly protocol 5.2 with no additions. Each can be switched on alone,
    which is how the improvement sweep in :mod:`experiments` compares them.
    """

    mode: str = "valence"

    # protocol parameters
    aromatic_order: float = 1.5
    bridge_probability: float = 0.6
    sparsity_amount: float = 0.01
    sparsity_threshold: float = 0.5

    # improvements, all neutral by default
    readout_caps: bool = False
    valence_damping: float = 1.0
    fluidity_threshold: float = 1.0
    fluidity_strength: float = 0.0
    bridge_retry: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.aromatic_order not in (1.0, 1.5):
            raise ValueError("supported aromatic feasibility weights are 1.0 and 1.5")
        if not 0 < self.valence_damping <= 1:
            raise ValueError("valence_damping must lie in (0, 1]")
        if not 0 < self.bridge_probability <= 1:
            raise ValueError("bridge_probability must lie in (0, 1]")
        if not 0 <= self.sparsity_amount <= 1:
            raise ValueError("sparsity_amount must lie in [0, 1]")
        if not 0 <= self.sparsity_threshold <= 1:
            raise ValueError("sparsity_threshold must lie in [0, 1]")
        if not 0 <= self.fluidity_strength <= 1:
            raise ValueError("fluidity_strength must lie in [0, 1]")
        if not 0 < self.fluidity_threshold <= 1:
            raise ValueError("fluidity_threshold must lie in (0, 1]")

    @property
    def improvements(self):
        """Names of the improvements that are actually active in this arm."""
        base = Stage2Config(mode=self.mode)
        names = ("readout_caps", "valence_damping", "bridge_retry")
        active = [n for n in names if getattr(self, n) != getattr(base, n)]
        if self.fluidity_strength > 0 and self.fluidity_threshold < 1:
            active.append("fluidity_floor")
        return tuple(active)

    def record(self):
        return {**asdict(self), "active_improvements": list(self.improvements)}


# --------------------------------------------------------------------------- #
# representation invariants (protocol 5.1)
# --------------------------------------------------------------------------- #


def pair_indices(n):
    """Lexicographic ``i < j`` indices, the one pair order every artifact uses."""
    return np.triu_indices(n, 1)


def n_atoms_from_pairs(pair_count):
    """Invert ``N(N-1)/2``; raises when ``pair_count`` is not triangular."""
    n = round((1 + np.sqrt(1 + 8 * pair_count)) / 2)
    if n * (n - 1) // 2 != pair_count:
        raise ValueError(f"{pair_count} is not N(N-1)/2 for any integer N")
    return n


def check_invariants(atoms, edges, *, atol=1e-6):
    """Assert the six protocol-5.1 invariants and return N.

    Symmetry and the NO_BOND diagonal are structural: only the upper triangle
    exists as a variable, so they cannot be violated and are checked by shape.
    """
    atoms = np.asarray(atoms)
    edges = np.asarray(edges)
    if atoms.ndim != 2 or atoms.shape[1] != len(ELEMENTS):
        raise ValueError(f"atoms must have shape [N, {len(ELEMENTS)}]")
    n = atoms.shape[0]
    if edges.shape != (n * (n - 1) // 2, len(BOND_STATES)):
        raise ValueError(
            f"edges must have shape [{n * (n - 1) // 2}, {len(BOND_STATES)}]"
        )
    for name, rows in (("atoms", atoms), ("edges", edges)):
        if not np.issubdtype(rows.dtype, np.floating):
            raise TypeError(f"{name} must be real floating")
        if not np.isfinite(rows).all():
            raise ValueError(f"{name} contains nonfinite values")
        if rows.size:
            if (rows < -atol).any() or (rows > 1 + atol).any():
                raise ValueError(f"{name} rows must hold probabilities in [0, 1]")
            if not np.allclose(rows.sum(-1), 1.0, atol=atol, rtol=0):
                raise ValueError(f"{name} rows must sum to one")
    return n


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def project_simplex_rows(values):
    """Euclidean projection of each row onto the probability simplex."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values
    ordered = np.sort(values, axis=-1)[:, ::-1]
    shifted = ordered.cumsum(-1) - 1
    ranks = np.arange(1, values.shape[-1] + 1)
    rho = (ordered - shifted / ranks > 0).sum(-1) - 1
    theta = shifted[np.arange(len(values)), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0.0)


def bond_orders(aromatic_order=1.5):
    return np.array([0.0, 1.0, 2.0, 3.0, aromatic_order])


def soft_valence(edges, n, *, aromatic_order=1.5):
    """Expected valence per atom, ``v_i = sum_{j != i} sum_b o_b E_ij,b``."""
    i, j = pair_indices(n)
    orders = np.asarray(edges) @ bond_orders(aromatic_order)
    return np.bincount(i, weights=orders, minlength=n) + np.bincount(
        j, weights=orders, minlength=n
    )


def valence_ceiling(atoms, *, readout_caps=False):
    """``v_max,i`` from the soft identity, optionally tightened to the argmax.

    The soft ceiling is a mixture over identities, so a state can satisfy it
    while its argmax molecule is over-valent. ``readout_caps`` closes that gap
    by also requiring the selected element's own ceiling; it only ever tightens.
    """
    atoms = np.asarray(atoms, dtype=np.float64)
    caps = atoms @ NEUTRAL_VALENCE
    if not readout_caps:
        return caps, 0
    conditional = NEUTRAL_VALENCE[atoms.argmax(-1)]
    return np.minimum(caps, conditional), int(np.count_nonzero(conditional < caps))


def components(n, edges):
    """Connected-component labels of the argmax graph, by union-find."""
    parent = np.arange(n)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (i, j), category in zip(zip(*pair_indices(n)), np.asarray(edges).argmax(-1)):
        if category:
            parent[find(i)] = find(j)
    return np.array([find(k) for k in range(n)])


# --------------------------------------------------------------------------- #
# 5.2 valence projection
# --------------------------------------------------------------------------- #


def project_valence(atoms, edges, *, config=None, tolerance=2e-6):
    """Project ``E`` onto the edge simplexes intersected with valence halfspaces.

    Solves the N-dimensional nonnegative dual rather than the
    ``5*N*(N-1)/2``-variable primal QP; each dual evaluation eliminates the edge
    rows by simplex projection. The primal is strictly convex, and feasibility
    plus complementary slackness are verified on every active result. ``X`` is
    held fixed by this operator.

    The projection removes only excess valence. It never pushes an atom up
    toward its ceiling, so an under-coordinated atom is left alone.

    With ``valence_damping`` below one the returned state is a convex
    combination of the input and its exact projection. Each row stays on its
    simplex, but the valence constraint is then only partially enforced and the
    residual excess is reported as ``damped_excess`` rather than raising.
    """
    config = config or Stage2Config()
    atoms = np.asarray(atoms, dtype=np.float64)
    source = np.asarray(edges)
    u = source.astype(np.float64)
    n = len(atoms)
    i, j = pair_indices(n)
    order = bond_orders(config.aromatic_order)
    caps, tightened = valence_ceiling(atoms, readout_caps=config.readout_caps)
    before = soft_valence(u, n, aromatic_order=config.aromatic_order)
    excess_before = float(max(0.0, np.max(before - caps) if n else 0.0))

    if excess_before <= tolerance:
        return source.copy(), {
            "active": False,
            "tightened_cap_atoms": tightened,
            "iterations": 0,
            "max_excess_before": excess_before,
            "max_excess_after": excess_before,
            "displacement_l2": 0.0,
        }

    def dual(lam):
        e = project_simplex_rows(u - (lam[i] + lam[j])[:, None] * order)
        v = soft_valence(e, n, aromatic_order=config.aromatic_order)
        q = 0.5 * np.square(e - u).sum() + lam @ (v - caps)
        return -q, caps - v

    result = minimize(
        dual,
        np.zeros(n),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * n,
        options={"ftol": 1e-15, "gtol": 1e-11, "maxiter": 500, "maxls": 40},
    )
    exact = project_simplex_rows(u - (result.x[i] + result.x[j])[:, None] * order)
    slack = caps - soft_valence(exact, n, aromatic_order=config.aromatic_order)
    excess = max(0.0, float(-slack.min()))
    complementarity = float(np.abs(result.x * slack).max())
    if excess > tolerance or complementarity > 2e-6:
        raise ArithmeticError(
            f"valence projection KKT failure: excess={excess}, "
            f"complementarity={complementarity}, solver={result.message}"
        )

    alpha = config.valence_damping
    out = (u + alpha * (exact - u)) if alpha < 1 else exact
    after = soft_valence(out, n, aromatic_order=config.aromatic_order)
    receipt = {
        "active": True,
        "tightened_cap_atoms": tightened,
        "iterations": int(result.nit),
        "solver_success": bool(result.success),
        "max_excess_before": excess_before,
        "max_excess_after": max(0.0, float(np.max(after - caps))),
        "complementarity": complementarity,
        "displacement_l2": float(np.linalg.norm(out - u)),
    }
    if alpha < 1:
        receipt["damping"] = alpha
        receipt["damped_excess"] = receipt["max_excess_after"]
    return out.astype(source.dtype), receipt


# --------------------------------------------------------------------------- #
# 5.3 connectivity
# --------------------------------------------------------------------------- #


def connect_components(atoms, edges, *, config=None):
    """Bridge argmax components with greedy highest-confidence SINGLE edges.

    Only inter-component candidates are considered, at most ``c-1`` edges are
    added, and hard valence capacity is respected. No chain, ring or tree
    topology is prescribed. A bridge can be removed again by the valence
    projection that follows; that is reported, never hidden.
    """
    config = config or Stage2Config()
    atoms = np.asarray(atoms)
    out = np.asarray(edges).copy()
    n = len(atoms)
    i, j = pair_indices(n)
    labels = components(n, out)
    count = len(np.unique(labels))
    order = bond_orders(config.aromatic_order)
    used_order = order[out.argmax(-1)]
    used = np.bincount(i, weights=used_order, minlength=n) + np.bincount(
        j, weights=used_order, minlength=n
    )
    caps = NEUTRAL_VALENCE[atoms.argmax(-1)]
    p = config.bridge_probability
    selected = []
    # Stable ordering makes ties reproducible; labels depend only on the graph.
    for k in np.argsort(-(1 - out[:, 0]), kind="stable"):
        a, b = int(i[k]), int(j[k])
        if labels[a] == labels[b] or used[a] + 1 > caps[a] or used[b] + 1 > caps[b]:
            continue
        labels[labels == labels[b]] = labels[a]
        out[k] = [1 - p, p, 0, 0, 0]
        used[[a, b]] += 1
        selected.append([a, b])
        if len(selected) == count - 1:
            break
    return out, {
        "components_before": int(count),
        "bridges": selected,
        "components_proposed": len(np.unique(components(n, out))),
    }


# --------------------------------------------------------------------------- #
# 5.4 sparsity
# --------------------------------------------------------------------------- #


def shrink_weak_edges(edges, *, config=None, protect=None):
    """Move a little low-confidence bond mass back toward NO_BOND.

    One interpretable mechanism, as the protocol requires: no stacked edge
    count, degree, L1 and bond-mass penalties. Conditional order mass is
    rescaled so each row stays on its simplex. ``protect`` exempts rows (used
    to spare freshly added bridges).
    """
    config = config or Stage2Config()
    out = np.asarray(edges).copy()
    existence = 1 - out[:, 0]
    weak = existence < config.sparsity_threshold
    if protect is not None:
        weak &= ~np.asarray(protect, dtype=bool)
    removed = np.where(weak, np.minimum(existence, config.sparsity_amount), 0.0)
    scale = np.divide(
        existence - removed, existence, out=np.ones_like(existence), where=existence > 0
    )
    out[:, 1:] *= scale[:, None]
    out[:, 0] += removed
    return out, {
        "removed_bond_mass": float(removed.sum()),
        "changed_pairs": int((removed > 0).sum()),
    }


# --------------------------------------------------------------------------- #
# improvement: fluidity floor
# --------------------------------------------------------------------------- #


def apply_fluidity_floor(rows, *, threshold, strength):
    """Mix near-one-hot rows back toward uniform, in probability space.

    Protocol 4.3 C.10 wants premature collapse addressed by reducing the update
    or projection aggressiveness before reaching for entropy control. This is
    the narrowest well-defined version of that control: a row whose largest
    probability exceeds ``threshold`` is replaced by
    ``(1-strength)*row + strength/K``. Rows below the threshold are untouched,
    so an ordinary trajectory is unaffected and the operator is not an
    entropy loss. Each row stays on its simplex by construction.
    """
    out = np.asarray(rows, dtype=np.float64).copy()
    if not out.size or strength <= 0 or threshold >= 1:
        return out.astype(np.asarray(rows).dtype), {"rows_relaxed": 0}
    k = out.shape[-1]
    collapsed = out.max(-1) > threshold
    if collapsed.any():
        out[collapsed] = (1 - strength) * out[collapsed] + strength / k
    return out.astype(np.asarray(rows).dtype), {
        "rows_relaxed": int(collapsed.sum()),
        "threshold": threshold,
        "strength": strength,
    }


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #


def apply_operators(atoms, edges, config):
    """Run one arm's operator set and return ``(atoms, edges, receipt)``.

    Order matters and is fixed:

    1. fluidity floor, if enabled, on both ``X`` and ``E``
    2. valence projection
    3. connectivity bridging, then valence projection again
    4. weak-edge shrinkage, then valence projection again

    Steps 3 and 4 are each selected by the mode, so ``sparse_only`` runs 4
    without 3 and nothing about the remaining steps changes.

    Valence is always last so the returned state is the feasible one. The
    fluidity floor runs first precisely because relaxing ``E`` can reintroduce
    valence excess, and because relaxing ``X`` changes the ceiling itself.
    """
    n = check_invariants(atoms, edges)
    atoms = np.asarray(atoms)
    edges = np.asarray(edges)
    receipt = {"mode": config.mode, "improvements": list(config.improvements)}

    if config.mode == "none":
        # The frozen Stage 1 control still reports the same diagnostics, so the
        # comparison against an operator arm reads off identical fields.
        violations, worst = argmax_valence_violations(
            atoms, edges, aromatic_order=config.aromatic_order
        )
        receipt["components_after"] = len(np.unique(components(n, edges)))
        receipt["argmax_valence_violations"] = violations
        receipt["argmax_valence_max_excess"] = worst
        check_invariants(atoms, edges)
        return atoms, edges, receipt

    if config.fluidity_strength > 0 and config.fluidity_threshold < 1:
        atoms, atom_note = apply_fluidity_floor(
            atoms,
            threshold=config.fluidity_threshold,
            strength=config.fluidity_strength,
        )
        edges, edge_note = apply_fluidity_floor(
            edges,
            threshold=config.fluidity_threshold,
            strength=config.fluidity_strength,
        )
        receipt["fluidity_floor"] = {"atoms": atom_note, "edges": edge_note}

    edges, receipt["valence"] = project_valence(atoms, edges, config=config)

    if config.mode in BRIDGING_MODES:
        edges, bridging = connect_components(atoms, edges, config=config)
        edges, bridging["valence_after"] = project_valence(atoms, edges, config=config)
        survived = _surviving_bridges(n, edges, bridging["bridges"])
        bridging["bridges_surviving"] = survived
        bridging["bridges_lost"] = len(bridging["bridges"]) - survived

        if config.bridge_retry and bridging["bridges_lost"]:
            # One bounded retry at a stronger bridge probability. Not a loop:
            # a bridge the valence projection still rejects stays rejected.
            stronger = replace(
                config, bridge_probability=min(1.0, config.bridge_probability + 0.3)
            )
            retried, retry = connect_components(atoms, edges, config=stronger)
            retried, retry["valence_after"] = project_valence(
                atoms, retried, config=config
            )
            retry["bridge_probability"] = stronger.bridge_probability
            retry["bridges_surviving"] = _surviving_bridges(
                n, retried, retry["bridges"]
            )
            if retry["bridges_surviving"] > survived:
                edges = retried
            else:
                retry["kept"] = False
            bridging["retry"] = retry
        receipt["connectivity"] = bridging

    if config.mode in SHRINKING_MODES:
        # Spare every pair a bridging attempt touched, including a kept retry.
        # Without bridging (``sparse_only``) there is nothing to protect.
        protect = np.zeros(len(edges), dtype=bool)
        bridging = receipt.get("connectivity")
        if bridging:
            bridged = list(bridging["bridges"])
            retry = bridging.get("retry")
            if retry and retry.get("kept", True):
                bridged += retry["bridges"]
            for a, b in bridged:
                protect[_pair_row(n, a, b)] = True
        edges, receipt["sparsity"] = shrink_weak_edges(
            edges, config=config, protect=protect
        )
        edges, receipt["sparsity"]["valence_after"] = project_valence(
            atoms, edges, config=config
        )

    violations, worst = argmax_valence_violations(
        atoms, edges, aromatic_order=config.aromatic_order
    )
    receipt["components_after"] = len(np.unique(components(n, edges)))
    receipt["argmax_valence_violations"] = violations
    receipt["argmax_valence_max_excess"] = worst
    check_invariants(atoms, edges)
    return atoms, edges, receipt


def _pair_row(n, a, b):
    """Row index of unordered pair ``(a, b)`` in lexicographic order."""
    i, j = (a, b) if a < b else (b, a)
    return int(i * (2 * n - i - 1) // 2 + (j - i - 1))


def _surviving_bridges(n, edges, bridges):
    argmax = np.asarray(edges).argmax(-1)
    return int(sum(1 for a, b in bridges if argmax[_pair_row(n, a, b)] != 0))


def argmax_valence_violations(atoms, edges, *, aromatic_order=1.5):
    """Count atoms whose *discrete* readout exceeds its element's ceiling.

    This is the number that matters for molecular validity, and it is not what
    the soft projection constrains. Reported as ``(count, max_excess)``.
    """
    atoms = np.asarray(atoms)
    n = len(atoms)
    hard = np.zeros_like(np.asarray(edges), dtype=np.float64)
    hard[np.arange(len(hard)), np.asarray(edges).argmax(-1)] = 1.0
    valence = soft_valence(hard, n, aromatic_order=aromatic_order)
    caps = NEUTRAL_VALENCE[atoms.argmax(-1)]
    excess = valence - caps
    return int(np.count_nonzero(excess > 1e-9)), float(max(0.0, excess.max()))


def post_step(config):
    """Build the ``post_step`` callable ``run_trajectory`` expects.

    Returns ``(graph, receipt)``. JAX is imported lazily so this module, the
    operators and their tests stay importable with NumPy and SciPy alone.
    """
    from mosaic.small_molecule.state import RelaxedGraph

    def operate(graph, step):
        import jax.numpy as jnp

        atoms, edges, receipt = apply_operators(
            np.asarray(graph.X), np.asarray(graph.E_upper), config
        )
        receipt["step"] = int(step)
        return (
            RelaxedGraph(
                jnp.asarray(atoms, dtype=graph.X.dtype),
                jnp.asarray(edges, dtype=graph.E_upper.dtype),
            ),
            receipt,
        )

    return operate
