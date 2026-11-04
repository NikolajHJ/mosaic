"""Detached, explicit chemistry operators for matched Stage 2 experiments.

These operate after an AF3 gradient/simplex step. They are not model losses,
RDKit repairs, or claims that soft feasibility implies hard molecular validity.
"""

import numpy as np
from scipy.optimize import minimize

NEUTRAL_VALENCE = np.array([4.0, 3.0, 2.0, 2.0, 1.0, 1.0])
BOND_ORDER = np.array([0.0, 1.0, 2.0, 3.0, 1.5])


def simplex_rows(values):
    values = np.asarray(values, dtype=np.float64)
    ordered = np.sort(values, axis=-1)[:, ::-1]
    shifted = ordered.cumsum(-1) - 1
    ranks = np.arange(1, values.shape[-1] + 1)
    rho = (ordered - shifted / ranks > 0).sum(-1) - 1
    theta = shifted[np.arange(len(values)), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0.0)


def soft_valence(edges, n, *, aromatic_order=1.5):
    i, j = np.triu_indices(n, 1)
    bond_order = np.array([0.0, 1.0, 2.0, 3.0, aromatic_order])
    orders = np.asarray(edges) @ bond_order
    return np.bincount(i, weights=orders, minlength=n) + np.bincount(
        j, weights=orders, minlength=n
    )


def project_valence_arrays(
    atoms, edges, *, tolerance=2e-6, aromatic_order=1.5, readout_caps=False
):
    """Euclidean projection onto edge simplexes and soft valence halfspaces.

    Solve the N-dimensional nonnegative dual instead of a 5*N*(N-1)/2-variable
    primal QP. Each dual evaluation eliminates edge rows by simplex projection.
    The primal is strictly convex; feasibility and complementary slackness are
    checked on every result. X is fixed during this operator.
    Aromatic weight 1.0 gives a loose coordination lower bound, not a complete
    chemical valence model; 1.5 preserves the original protocol surrogate.
    Optional readout caps only tighten the original ceiling to the selected
    element ceiling. This does not guarantee valid hard bond assignments.
    """
    if aromatic_order not in (1.0, 1.5):
        raise ValueError("Supported aromatic feasibility weights are 1.0 and 1.5")
    bond_order = np.array([0.0, 1.0, 2.0, 3.0, aromatic_order])
    x, u = np.asarray(atoms, dtype=np.float64), np.asarray(edges, dtype=np.float64)
    n = len(x)
    i, j = np.triu_indices(n, 1)
    caps = x @ NEUTRAL_VALENCE
    tightened = 0
    if readout_caps:
        conditional = NEUTRAL_VALENCE[x.argmax(-1)]
        tightened = int(np.count_nonzero(conditional < caps))
        caps = np.minimum(caps, conditional)
    before = soft_valence(u, n, aromatic_order=aromatic_order)
    if np.max(before - caps) <= tolerance:
        return np.asarray(edges).copy(), {
            "active": False,
            "tightened_cap_atoms": tightened,
            "iterations": 0,
            "max_excess": float(max(0.0, np.max(before - caps))),
            "displacement_l2": 0.0,
        }

    def dual(lam):
        e = simplex_rows(u - (lam[i] + lam[j])[:, None] * bond_order)
        v = soft_valence(e, n, aromatic_order=aromatic_order)
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
    e = simplex_rows(u - (result.x[i] + result.x[j])[:, None] * bond_order)
    out = e.astype(np.asarray(edges).dtype)
    slack = caps - soft_valence(out, n, aromatic_order=aromatic_order)
    excess = max(0.0, float(-slack.min()))
    complementarity = float(np.abs(result.x * slack).max())
    if excess > tolerance or complementarity > 2e-6:
        raise ArithmeticError(
            f"Valence projection KKT failure: excess={excess}, complementarity={complementarity}, solver={result.message}"
        )
    return out, {
        "active": True,
        "tightened_cap_atoms": tightened,
        "iterations": int(result.nit),
        "solver_success": bool(result.success),
        "max_excess": excess,
        "complementarity": complementarity,
        "displacement_l2": float(np.linalg.norm(out - u)),
    }


def components(atoms, edges):
    n = len(atoms)
    parent = np.arange(n)

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for (i, j), category in zip(
        zip(*np.triu_indices(n, 1)), np.asarray(edges).argmax(-1), strict=True
    ):
        if category:
            parent[find(i)] = find(j)
    return np.array([find(k) for k in range(n)])


def connect_components_arrays(
    atoms, edges, *, single_probability=0.6, aromatic_order=1.5
):
    """Greedy maximum-soft-confidence bridges, with hard valence capacity.

    Add only inter-component SINGLE candidates and at most c-1 edges. No global
    chain/ring/tree topology is prescribed. The subsequent soft-valence
    projection can remove a bridge; that failure is logged rather than hidden.
    """
    x, out = np.asarray(atoms), np.asarray(edges).copy()
    n = len(x)
    i, j = np.triu_indices(n, 1)
    labels = components(x, out)
    count = len(np.unique(labels))
    bond_order = np.array([0.0, 1.0, 2.0, 3.0, aromatic_order])
    hard_orders = bond_order[out.argmax(-1)]
    used = np.bincount(i, weights=hard_orders, minlength=n) + np.bincount(
        j, weights=hard_orders, minlength=n
    )
    caps = NEUTRAL_VALENCE[x.argmax(-1)]
    selected = []
    # Stable ordering makes ties reproducible; labels depend only on the graph.
    for k in np.argsort(-(1 - out[:, 0]), kind="stable"):
        a, b = i[k], j[k]
        if labels[a] == labels[b] or used[a] + 1 > caps[a] or used[b] + 1 > caps[b]:
            continue
        old = labels[b]
        labels[labels == old] = labels[a]
        out[k] = [1 - single_probability, single_probability, 0, 0, 0]
        used[[a, b]] += 1
        selected.append([int(a), int(b)])
        if len(selected) == count - 1:
            break
    return out, {
        "components_before": count,
        "bridges": selected,
        "components_proposed": len(np.unique(components(x, out))),
    }


def shrink_weak_edges(edges, *, amount=0.01, threshold=0.5):
    out = np.asarray(edges).copy()
    existence = 1 - out[:, 0]
    removed = np.where(existence < threshold, np.minimum(existence, amount), 0.0)
    scale = np.divide(
        existence - removed, existence, out=np.ones_like(existence), where=existence > 0
    )
    out[:, 1:] *= scale[:, None]
    out[:, 0] += removed
    return out, {
        "removed_bond_mass": float(removed.sum()),
        "changed_pairs": int((removed > 0).sum()),
    }
