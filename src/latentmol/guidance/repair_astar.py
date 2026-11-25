"""Probability-aware A* over fixed-atom, fixed-charge bond-label assignments.

Every bond changes at most once. Edge costs are the existing edit-count plus
nonnegative probability regret, using probabilities frozen for the whole search.
No heuristic/RDKit operation is differentiated; accepted targets use the existing
repair loss. Bounded searches report interruption rather than global optimality.
"""
from bisect import insort
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.atom_preserving_diagnostics import BondOnlyDiagnostic
from latentmol.atom_preserving_fast import BondSearchContext
from latentmol.chemistry import MoleculeRejected, encode_smiles
from latentmol.guidance.regularization import ALLOWED_VALENCES
from latentmol.probability_repair import measure_probability_edits, probability_identity, validate_probabilities
from latentmol.quality import graph_identity


@dataclass(frozen=True)
class AStarRepairConfig:
    version: str = "probability-astar-v1"
    max_edits: int | None = None  # None inherits the common beam edit radius.
    candidate_checks: int = 96
    max_generated: int = 50000
    max_frontier: int = 20000
    heuristic: str = "probability"  # zero is a Dijkstra diagnostic control.

    def __post_init__(self):
        if self.version != "probability-astar-v1" or self.heuristic not in ("probability", "zero"):
            raise ValueError("Unknown A* policy or heuristic")
        if self.max_edits is not None and (type(self.max_edits) is not int or not 1 <= self.max_edits <= 10000):
            raise ValueError("A* edit radius must be positive or inherited")
        for name in ("candidate_checks", "max_generated", "max_frontier"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 1000000:
                raise ValueError("Invalid A* budget: " + name)


def astar_policy(config, astar):
    return dict(version=astar.version, astar=asdict(astar),
        resolved_max_edits=config.max_edits if astar.max_edits is None else astar.max_edits,
        cost=config.policy()["cost"], probability_floor=config.probability_floor,
        probability_weight=config.probability_weight, allowed_edits=["bond"],
        atom_identity="fixed_exact_slot_labels", formal_charge="fixed", occupancy="fixed",
        alignment="shared_slots", already_valid="skip_without_self_target",
        queue="g_plus_h; then smaller_h; then depth_and_insertion_order",
        heuristic="sum_r_cheapest_distinct_remaining_bond_costs; r=max(components-1,ceil(excess/6))",
        aromatic_order="one_unit_optimistic_lower_bound", unknown_valence_rule="omit_valence_bound",
        budgets="candidate native checks exclude raw/final teacher checks; memory caps interrupt, never beam-prune",
        target_loss="existing_repair_loss_interface; discovery_does_not_change_loss_scaling",
        minimum_is_proven=False)


def required_edit_bound(excess, components):
    """One change removes at most 3 order units at each of its two endpoints.

    Aromatic order is one here, not 1.5: native-valid aromatic O/S rings must not
    acquire an over-valence lower bound. Unknown rules contribute zero excess.
    Connectivity and valence bounds are combined with max, not double-counted.
    """
    return max(max(0, int(components)-1), math.ceil(max(0, float(excess))/6))


def probability_lower_bound(required, ordered_costs, edited):
    """Relax chemistry: buy the cheapest r distinct not-yet-edited bonds.

    A replacement's probability cost may be low even if it cannot repair the
    current defect. This optimism is necessary for the lower-bound argument.
    At most one alternative per bond appears in ordered_costs. Empty availability
    means no completion in this once-per-bond edit space, not chemical impossibility.
    """
    if required == 0:
        return 0.
    costs = []
    for cost, index in ordered_costs:
        if index not in edited:
            costs.append(cost)
            if len(costs) == required:
                value = sum(costs)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite A* heuristic cost")
                return value
    return math.inf


class _Space:
    """Point-local fixed costs; topology is computed only for expanded states."""
    def __init__(self, raw, p, spec, config):
        self.raw, self.spec, self.config = raw, spec, config
        self.context = BondSearchContext(raw, p, spec, config)
        self.pairs = self.context.pairs
        self.left = np.array([i for i, _ in self.pairs], dtype=int)
        self.right = np.array([j for _, j in self.pairs], dtype=int)
        self.original = raw[2][self.left, self.right].astype(np.uint8)
        self.root = self.original.tobytes()
        self.orders = np.array([{"NONE": 0, "SINGLE": 1, "DOUBLE": 2,
                                 "TRIPLE": 3, "AROMATIC": 1}[name] for name in spec.bond_names])
        self.caps = np.full(spec.max_nodes, np.inf)
        for i in self.context.slots:
            allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[int(raw[0][i])], {}).get(
                spec.formal_charges[int(raw[1][i])])
            if allowed:
                self.caps[i] = max(allowed)
        self.regret = np.zeros((len(self.pairs), len(spec.bond_names)))
        ordered = []
        for index, (i, j) in enumerate(self.pairs):
            alternatives = []
            for label in range(len(spec.bond_names)):
                if label != self.original[index]:
                    self.regret[index, label] = self.context.regrets[i, j, label]
                    alternatives.append(1 + config.probability_weight*float(self.regret[index, label]))
            if not all(math.isfinite(value) and value >= 1 for value in alternatives):
                raise ValueError("Nonfinite A* edit cost")
            ordered.append((min(alternatives), index))
        self.ordered_costs = sorted(ordered)

    def cost(self, edits):
        # Canonical unordered-pair order matches the public witness cost;
        # edit path order does not change the duplicate-state cost.
        value = len(edits) + self.config.probability_weight*sum(float(self.regret[i, label]) for i, label in edits)
        if not math.isfinite(value):
            raise ValueError("Nonfinite A* accumulated cost")
        return value

    def bonds(self, key):
        bonds = self.raw[2].copy()
        values = np.frombuffer(key, dtype=np.uint8)
        bonds[self.left, self.right] = bonds[self.right, self.left] = values
        return bonds

    def topology(self, key):
        values = np.frombuffer(key, dtype=np.uint8)
        adjacency = {i: [] for i in self.context.slots}
        for index in np.flatnonzero(values):
            i, j = self.pairs[int(index)]
            adjacency[i].append((j, int(index))); adjacency[j].append((i, int(index)))
        discovered, low, labels, bridges = {}, {}, {}, set()
        def visit(i, parent_edge, component):
            discovered[i] = low[i] = len(discovered)
            labels[i] = component
            for j, edge in adjacency[i]:
                if edge == parent_edge:
                    continue
                if j not in discovered:
                    visit(j, edge, component); low[i] = min(low[i], low[j])
                    if low[j] > discovered[i]:
                        bridges.add(edge)
                else:
                    low[i] = min(low[i], discovered[j])
        components = 0
        for i in self.context.slots:
            if i not in discovered:
                visit(i, -1, components); components += 1
        valence = np.bincount(self.left, weights=self.orders[values], minlength=self.spec.max_nodes)
        valence += np.bincount(self.right, weights=self.orders[values], minlength=self.spec.max_nodes)
        return values, components, labels, bridges, valence, float(np.maximum(valence-self.caps, 0).sum())


def search_astar(raw, probabilities, spec, config, astar):
    if not isinstance(astar, AStarRepairConfig):
        raise TypeError("Require an explicit AStarRepairConfig")
    raw = tuple(np.array(v.detach().cpu().numpy() if hasattr(v, "detach") else v, copy=True) for v in raw)
    p = validate_probabilities(probabilities, spec)
    if set(spec.bond_names)-{"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Unsupported A* bond vocabulary")
    if len(spec.bond_names) < 2:
        raise ValueError("A* requires at least one non-NONE bond alternative")
    measure_probability_edits(raw, raw, p, spec, config)
    engine = BondOnlyDiagnostic(raw, spec)
    initial = engine.diagnose(raw[2])
    native_valid, roundtrip_error = initial["valid"], None
    if native_valid:
        try:
            if encode_smiles(initial["smiles"], spec)["smiles"] != initial["smiles"]:
                roundtrip_error = "Canonical identity differs"
        except MoleculeRejected as exc:
            roundtrip_error = str(exc)
    report = dict(policy=astar_policy(config, astar), raw_graph_id=graph_identity(*raw, spec),
        probability_sha256=probability_identity(p, spec), raw_rdkit_valid=native_valid,
        raw_roundtrip_error=roundtrip_error, raw_valid=native_valid and roundtrip_error is None,
        checks=0, generated_states=0, expanded_states=0, duplicate_states=0,
        edit_bound_pruned=0, frontier_peak=0, weighted_witness=None,
        best_observed_weighted_cost=None, distance_is_exact=False, accepted_source=None,
        roundtrip_rejected_candidates=0, interrupted=False, goal_popped=False,
        lower_bound_weighted_cost=None)
    if report["raw_valid"]:
        return dict(report, status="already_valid", termination="raw_valid", best_observed_weighted_cost=0.)
    if not initial.get("schema_supported") or not initial.get("heavy_atoms") or not initial.get("categorical_support_labels_ok"):
        return dict(report, status="unsupported", termination="unsupported_input")
    space = _Space(raw, p, spec, config)
    max_edits = min(len(space.pairs), config.max_edits if astar.max_edits is None else astar.max_edits)
    _, components, _, _, _, excess = space.topology(space.root)
    required = required_edit_bound(excess, components)
    if required > max_edits:
        return dict(report, status="censored", termination="edit_lower_bound_exceeds_radius",
                    lower_bound_weighted_cost=probability_lower_bound(required, space.ordered_costs, set()))
    h = probability_lower_bound(required, space.ordered_costs, set()) if astar.heuristic == "probability" else 0.
    queue, seen = [(h, h, 0, 0, space.root, ())], {space.root}
    report.update(generated_states=1, frontier_peak=1)
    while queue:
        f, h, depth, _, key, edits = heapq.heappop(queue)
        if key != space.root:
            if report["checks"] >= astar.candidate_checks:
                return dict(report, status="censored", termination="candidate_check_budget", interrupted=True,
                            lower_bound_weighted_cost=f)
            report["checks"] += 1
            bonds = space.bonds(key)
            native = engine.diagnose(bonds)
            if native["valid"]:
                try:
                    compatible = encode_smiles(native["smiles"], spec)["smiles"] == native["smiles"]
                except MoleculeRejected:
                    compatible = False
                if compatible:
                    if h > 1e-10:
                        raise RuntimeError("Valence heuristic assigns positive remaining cost to a native-valid goal")
                    graph = (raw[0], raw[1], bonds)
                    cost = measure_probability_edits(raw, graph, p, spec, config)
                    witness = dict(**cost, graph_id=graph_identity(*graph, spec), smiles=native["smiles"],
                        labels={name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), graph)})
                    return dict(report, status="witness_found", termination="goal_popped", goal_popped=True,
                        accepted_source="astar", weighted_witness=witness,
                        best_observed_weighted_cost=cost["weighted_cost"], lower_bound_weighted_cost=cost["weighted_cost"])
                report["roundtrip_rejected_candidates"] += 1
        if depth == max_edits:
            continue
        report["expanded_states"] += 1
        values, components, labels, bridges, valence, excess = space.topology(key)
        changed = {i for i, _ in edits}
        for index, (i, j) in enumerate(space.pairs):
            if index in changed:
                continue
            old = int(values[index])
            for new in range(len(spec.bond_names)):
                if new == old:
                    continue
                delta = int(space.orders[new]-space.orders[old])
                next_excess = excess
                for atom in (i, j):
                    next_excess += max(0., valence[atom]+delta-space.caps[atom])-max(0., valence[atom]-space.caps[atom])
                next_components = components
                if old == 0 and new != 0 and labels[i] != labels[j]:
                    next_components -= 1
                elif old != 0 and new == 0 and index in bridges:
                    next_components += 1
                required = required_edit_bound(next_excess, next_components)
                if depth+1+required > max_edits:
                    report["edit_bound_pruned"] += 1
                    continue
                next_key = bytearray(key); next_key[index] = new; next_key = bytes(next_key)
                if next_key in seen:
                    report["duplicate_states"] += 1
                    continue
                if report["generated_states"] >= astar.max_generated or len(queue) >= astar.max_frontier:
                    reason = "generated_state_budget" if report["generated_states"] >= astar.max_generated else "frontier_memory_budget"
                    # The current parent still has ungenerated children. Its f
                    # must remain in the lower bound; frontier minimum alone is
                    # not valid evidence after an interrupted expansion.
                    bound = min(f, queue[0][0]) if queue else f
                    return dict(report, status="censored", termination=reason, interrupted=True,
                                lower_bound_weighted_cost=bound)
                next_edits = list(edits); insort(next_edits, (index, new)); next_edits = tuple(next_edits)
                next_h = (probability_lower_bound(required, space.ordered_costs, changed | {index})
                          if astar.heuristic == "probability" else 0.)
                sequence = report["generated_states"]
                heapq.heappush(queue, (space.cost(next_edits)+next_h, next_h, depth+1,
                                      sequence, next_key, next_edits))
                seen.add(next_key); report["generated_states"] += 1
                report["frontier_peak"] = max(report["frontier_peak"], len(queue))
    return dict(report, status="censored", termination="declared_space_exhausted")


class AStarRepairTeacher(AtomPreservingRepairTeacher):
    """Explicit discovery swap; targets, native checks and NLL are inherited."""
    def __init__(self, spec, config, *, astar, **kwargs):
        if not isinstance(astar, AStarRepairConfig):
            raise TypeError("Require an explicit A* policy")
        if kwargs.get("backend", "optimized") != "optimized":
            raise ValueError("A* uses the optimized native diagnostic")
        kwargs.setdefault("backend", "optimized")
        self.astar = astar
        super().__init__(spec, config, **kwargs)

    def _policy(self):
        return astar_policy(self.config, self.astar)

    def _search(self, raw, probabilities):
        return search_astar(raw, probabilities, self.spec, self.config, self.astar)

    def _search_task(self, raw, probabilities):
        return (*super()._search_task(raw, probabilities), self.astar)

    def _map_search(self, tasks):
        return self._pool.map_astar(tasks)
