"""Bounded detached graph matching for ordinary categorical reconstruction.

One bijection relabels atoms, charges and both bond axes. Identity is always an
incumbent for approximate proposals scored with the batch-normalized reconstruction
objective. Graph mode gives precedence to any exact hard-label bijection. Its
fallback matches colored vertices with RDKit without interpreting chemistry.
For nonisomorphic graphs the permutation search is approximate.
Gradients flow through the selected reconstruction loss, not the discrete map.
"""
from dataclasses import dataclass, replace
import time

import numpy as np
import torch

from latentmol.representation import HardGraphBatch
from latentmol.training.losses import reconstruction_losses


@dataclass(frozen=True)
class MatchingConfig:
    iterations: int = 2
    degree_weight: float = .1
    isomorphism_states: int = 512
    assignment_backend: str = "auto"

    def __post_init__(self):
        if (type(self.iterations) is not int or not 0 <= self.iterations <= 5 or
                not np.isfinite(self.degree_weight) or self.degree_weight < 0 or
                type(self.isomorphism_states) is not int or self.isomorphism_states < 1 or
                self.assignment_backend not in ("auto", "numpy")):
            raise ValueError("Invalid bounded graph-matching configuration")


def numpy_assignment(cost):
    """Square Hungarian assignment, O(n^3), with no optional dependency."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or not np.isfinite(cost).all():
        raise ValueError("Assignment requires a finite square cost matrix")
    n = len(cost)
    u, v = np.zeros(n+1), np.zeros(n+1)
    p, way = np.zeros(n+1, dtype=int), np.zeros(n+1, dtype=int)
    for i in range(1, n+1):
        p[0] = i
        j0 = 0
        minimum, used = np.full(n+1, np.inf), np.zeros(n+1, dtype=bool)
        while True:
            used[j0] = True
            current = cost[p[j0]-1] - u[p[j0]] - v[1:]
            improve = (~used[1:]) & (current < minimum[1:])
            minimum[1:][improve] = current[improve]
            way[1:][improve] = j0
            eligible = np.where(~used[1:], minimum[1:], np.inf)
            j1 = int(np.argmin(eligible))+1
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    answer = np.empty(n, dtype=int)
    answer[p[1:]-1] = np.arange(n)
    return answer


def _solver(config):
    if config.assignment_backend == "auto":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            pass
        else:
            def solve(cost):
                rows, columns = linear_sum_assignment(cost)
                answer = np.empty(len(cost), dtype=int)
                answer[rows] = columns
                return answer
            return solve, "scipy"
    return numpy_assignment, "numpy"


def graph_isomorphism(predicted, target, *, edge_classes, max_states):
    """Find a bounded exact hard-label witness, especially for symmetric rings.

    This only reindexes a known target. It never edits a predicted label or
    converts a relaxed model input to chemistry. Failure/budget exhaustion simply
    leaves the other approximate candidates available.
    """
    pa, pq, pb = predicted
    ta, tq, tb = target
    occupied_p, occupied_t = np.flatnonzero(pa), np.flatnonzero(ta)
    if len(occupied_p) != len(occupied_t):
        return None, 0
    def signature(a, q, b, i):
        return (int(a[i]), int(q[i]), *[int(np.sum(b[i] == e)) for e in range(1, edge_classes)])
    kp = {int(i): signature(pa, pq, pb, i) for i in occupied_p}
    kt = {int(i): signature(ta, tq, tb, i) for i in occupied_t}
    if sorted(kp.values()) != sorted(kt.values()):
        return None, 0
    domains = {i: [j for j in kt if kt[j] == key] for i, key in kp.items()}
    mapping, used, states = {}, set(), 0
    def visit():
        nonlocal states
        if len(mapping) == len(domains):
            return True
        options = []
        for i in domains:
            if i not in mapping:
                compatible = [j for j in domains[i] if j not in used and
                    all(pb[i, k] == tb[j, l] for k, l in mapping.items())]
                options.append((len(compatible), -sum(pb[i, k] != 0 for k in mapping), i, compatible))
        _, _, i, compatible = min(options)
        for j in compatible:
            states += 1
            if states > max_states:
                return False
            mapping[i] = j; used.add(j)
            if visit():
                return True
            del mapping[i]; used.remove(j)
        return False
    if not visit():
        return None, states
    answer = np.empty(len(pa), dtype=int)
    for i, j in mapping.items():
        answer[i] = j
    answer[np.flatnonzero(pa == 0)] = np.flatnonzero(ta == 0)
    return answer, states


def _complete_isomorphism(predicted, target, edge_classes, charge_classes):
    """Resolve a bounded search timeout with an exact labeled graph match.

    Use RDKit only as a combinatorial subgraph matcher here. Each occupied slot
    and each labeled bond is a colored vertex. Edges have one generic type.
    No sanitization, aromaticity changes or other chemistry interpretation can
    make two different categorical graphs match. The bounded search already
    established equal node and edge inventories before this fallback.
    """
    from rdkit import Chem

    def labeled_graph(labels):
        atoms, charges, bonds = labels
        active = np.flatnonzero(atoms)
        graph = Chem.RWMol()

        def vertex(color):
            atom = Chem.Atom(6)
            atom.SetIsotope(int(color))
            atom.SetNoImplicit(True)
            return graph.AddAtom(atom)

        for i in active:
            vertex(edge_classes + atoms[i] * charge_classes + charges[i])
        edges = bonds[np.ix_(active, active)]
        for i, j in zip(*np.triu(edges).nonzero()):
            middle = vertex(edges[i, j])
            graph.AddBond(int(i), middle, Chem.BondType.SINGLE)
            graph.AddBond(middle, int(j), Chem.BondType.SINGLE)
        return graph.GetMol(), active

    pred_graph, pred_active = labeled_graph(predicted)
    target_graph, target_active = labeled_graph(target)
    match = pred_graph.GetSubstructMatch(target_graph)
    if not match:
        return None
    permutation = np.empty(len(predicted[0]), dtype=int)
    permutation[pred_active[list(match[:len(target_active)])]] = target_active
    permutation[np.flatnonzero(predicted[0] == 0)] = np.flatnonzero(target[0] == 0)
    if (not np.array_equal(predicted[0], target[0][permutation])
            or not np.array_equal(predicted[1][pred_active], target[1][permutation[pred_active]])
            or not np.array_equal(predicted[2], target[2][np.ix_(permutation, permutation)])):
        raise RuntimeError("Graph-isomorphism fallback returned a non-label-preserving map")
    return permutation


@torch.no_grad()
def align_targets(decoded, target, loss_config, config=MatchingConfig(), *,
                  preserve_exact=False, denominators=None):
    """Propose maps, evaluate exact CE contributions, and return detached labels."""
    started = time.monotonic()
    batch, nodes = target.atoms.shape
    if (decoded.atoms.shape[:2] != (batch, nodes) or decoded.charges.shape[:2] != (batch, nodes) or
            target.charges.shape != (batch, nodes) or
            target.bonds.shape != (batch, nodes, nodes) or decoded.bonds.shape[:3] != (batch, nodes, nodes)):
        raise ValueError("Incompatible reconstruction graph shapes")
    names = ("atoms", "charges", "bonds")
    if any(not bool(torch.isfinite(getattr(decoded, name)).all()) for name in names):
        raise FloatingPointError("Nonfinite matching logits")
    log_a = decoded.atoms.log_softmax(-1).cpu().double().numpy()
    log_q = decoded.charges.log_softmax(-1).cpu().double().numpy()
    log_b = decoded.symmetric_bond_logits.log_softmax(-1).cpu().double().numpy()
    ta, tq, tb = (getattr(target, name).cpu().numpy() for name in names)
    if (not np.array_equal(tb, tb.transpose(0, 2, 1)) or
            np.any(np.diagonal(tb, axis1=1, axis2=2) != 0) or
            np.any(tb*((ta == 0)[:, :, None] | (ta == 0)[:, None, :]))):
        raise ValueError("Target bonds must be symmetric, diagonal NONE and absent at padding")
    if any(not np.isfinite(value).all() for value in (log_a, log_q, log_b)):
        raise FloatingPointError("Nonfinite matching log probabilities")
    for values, classes in ((ta, log_a.shape[-1]), (tq, log_q.shape[-1]), (tb, log_b.shape[-1])):
        if np.any(values < 0) or np.any(values >= classes):
            raise ValueError("Target categorical label outside decoder vocabulary")
    atom_weights, bond_weights = np.ones(log_a.shape[-1]), np.ones(log_b.shape[-1])
    atom_weights[0], bond_weights[0] = loss_config.padding_weight, loss_config.no_bond_weight
    occupied = ta != 0
    upper = np.triu(np.ones((nodes, nodes), dtype=bool), 1)
    pair_mask = occupied[:, :, None] & occupied[:, None, :] & upper
    da = max(float(atom_weights[ta].sum()), 1e-30)
    dq = max(int(occupied.sum()), 1)
    db = max(float(bond_weights[tb[pair_mask]].sum()), 1e-30)
    logical_rows = batch
    if denominators is not None:
        da, dq, db = (float(denominators[name]) for name in ("atoms", "charges", "bonds"))
        logical_rows = denominators["rows"]
        if logical_rows < batch or not all(np.isfinite(v) and v > 0 for v in (da, dq, db)):
            raise ValueError("Invalid logical reconstruction denominators")
    solve, backend = _solver(config)
    maps, before, after, assignments, witnesses = [], 0., 0., 0, 0
    search_states = budget_exhaustions = 0
    slots = np.arange(nodes)
    bond_classes = np.eye(log_b.shape[-1])
    for row in range(batch):
        a, q, b = ta[row], tq[row], tb[row]
        node_cost = -log_a[row][:, a]*atom_weights[a][None]/da
        node_cost += -loss_config.charge_weight*log_q[row][:, q]*(a != 0)[None]/dq
        def objective(permutation):
            ai, qi, bi = a[permutation], q[permutation], b[np.ix_(permutation, permutation)]
            active = ai != 0
            pairs = active[:, None] & active[None, :] & upper
            value = float(np.sum(-log_a[row][slots, ai]*atom_weights[ai])/da)
            value += float(np.sum(-log_q[row][slots, qi]*active)*loss_config.charge_weight/dq)
            value += float(np.sum(-log_b[row][slots[:, None], slots[None, :], bi][pairs]
                                  *bond_weights[bi[pairs]])*loss_config.bond_weight/db)
            return value
        identity = slots
        best_map, best_cost = identity, objective(identity)
        before += best_cost
        hard_a, hard_q = log_a[row].argmax(-1), log_q[row].argmax(-1)
        hard_b = log_b[row].argmax(-1)
        hard_b[(hard_a == 0)[:, None] | (hard_a == 0)[None, :]] = 0
        np.fill_diagonal(hard_b, 0)
        predicted_graph, target_graph = (hard_a, hard_q, hard_b), (a, q, b)
        if (preserve_exact and np.array_equal(hard_a, a)
                and np.array_equal(hard_q[a != 0], q[a != 0]) and np.array_equal(hard_b, b)):
            witness, states = identity, 0
        else:
            witness, states = graph_isomorphism(predicted_graph, target_graph,
                edge_classes=log_b.shape[-1], max_states=config.isomorphism_states)
        search_states += states
        budget_exhaustions += states > config.isomorphism_states
        if preserve_exact and witness is None and states > config.isomorphism_states:
            witness = _complete_isomorphism(predicted_graph, target_graph, log_b.shape[-1], log_q.shape[-1])
        if witness is not None:
            witnesses += 1
            cost = objective(witness)
            if preserve_exact:
                # Unequal category weights can otherwise give a wrong hard
                # graph a lower weighted CE than its correct witness.
                maps.append(witness)
                after += cost
                continue
            if cost < best_cost:
                best_map, best_cost = witness, cost
        probability_b = np.exp(log_b[row])
        probability_b[np.arange(nodes), np.arange(nodes)] = 0
        if log_b.shape[-1] > 1:
            degree_p = (probability_b[..., 1:]*(1-np.exp(log_a[row][:, 0]))[None, :, None]).sum(1)
            degree_t = np.stack([(b == e).sum(1) for e in range(1, log_b.shape[-1])], -1)
            descriptor = np.abs(degree_p[:, None]-degree_t[None]).mean(-1)*config.degree_weight/(logical_rows*nodes)
        else:
            descriptor = np.zeros((nodes, nodes))
        proposal = solve(node_cost+descriptor); assignments += 1
        negative_b = -log_b[row]*bond_weights[None, None, :]
        negative_b[np.arange(nodes), np.arange(nodes)] = 0
        for current in (identity, proposal):
            for iteration in range(config.iterations+1):
                cost = objective(current)
                if cost < best_cost:
                    best_map, best_cost = current.copy(), cost
                if iteration == config.iterations:
                    break
                context = bond_classes[b[:, current]]
                context *= ((a != 0)[:, None] & (a[current] != 0)[None, :])[..., None]
                edge_cost = np.einsum("ike,jke->ij", negative_b, context, optimize=True)
                edge_cost *= .5*loss_config.bond_weight/db
                current = solve(node_cost+edge_cost); assignments += 1
        maps.append(best_map); after += best_cost
    permutations = torch.as_tensor(np.stack(maps), dtype=torch.long, device=target.atoms.device)
    bonds = target.bonds.gather(1, permutations[:, :, None].expand(-1, -1, nodes))
    bonds = bonds.gather(2, permutations[:, None, :].expand(-1, nodes, -1))
    aligned = HardGraphBatch(target.atoms.gather(1, permutations), target.charges.gather(1, permutations), bonds)
    return aligned, dict(backend=backend, seconds=time.monotonic()-started, graphs=batch,
        assignment_calls=assignments, isomorphism_witnesses=witnesses,
        isomorphism_search_states=search_states, isomorphism_budget_exhaustions=budget_exhaustions,
        nonidentity_maps=sum(not np.array_equal(p, np.arange(nodes)) for p in maps),
        objective_before_without_size=before, objective_after_without_size=after,
        scope=("Exact map for isomorphic predictions; bounded approximate map for non-isomorphic predictions"
               if preserve_exact else "Bounded approximate map; identity incumbent; unchanged categorical CE weights and global batch denominators"))


def matched_reconstruction_losses(decoded, target, loss_config, config=MatchingConfig()):
    aligned, receipt = align_targets(decoded, target, loss_config, config)
    return reconstruction_losses(decoded, aligned, replace(loss_config, atom_order="ordered")), receipt


def align_reconstruction_targets(decoded, target, loss_config, *, denominators=None):
    """Return one detached slot map for the compact VAE's reconstruction CE.

    Return labels on prediction slots. If the hard prediction is the target
    graph under relabeling, every active atom, formal charge and undirected
    bond aligns exactly. Other graphs use the legacy Hungarian proposal with
    one edge-aware refinement and the original slot map as an incumbent.
    CE still measures prediction confidence; graph correctness is a separate
    hard readout. Pass logical batch denominators during gradient accumulation.
    """
    if denominators is None:
        # Use the same floating-point reduction as the accumulated trainer. Near
        # assignment ties, a second NumPy calculation can select a different map.
        from latentmol.chemistry_continuation import reconstruction_denominators
        denominators = reconstruction_denominators(target, loss_config, dtype=decoded.atoms.dtype)
    aligned, _ = align_targets(decoded, target, loss_config, MatchingConfig(iterations=1),
                               preserve_exact=True, denominators=denominators)
    return aligned
