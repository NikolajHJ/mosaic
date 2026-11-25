"""Opt-in fixed-atom repair discovery; the accepted-target NLL is unchanged.

Small conditional bond samples and coordinated connectivity/valence proposals
precede the existing beam search. All witnesses pass native chemistry and the
shared-schema roundtrip. Search never enters autograd or replaces the graph in
a binding scorer. Censoring is not proof of infeasibility or nearest distance.
"""
from dataclasses import asdict, dataclass, replace
import hashlib

import numpy as np

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_policy
from latentmol.atom_preserving_diagnostics import BondOnlyDiagnostic
from latentmol.atom_preserving_fast import BondSearchContext, search_atom_preserving_repair_fast
from latentmol.chemistry import MoleculeRejected, encode_smiles, inspect_graph
from latentmol.guidance.regularization import ALLOWED_VALENCES
from latentmol.probability_repair import measure_probability_edits, probability_identity, validate_probabilities
from latentmol.quality import graph_identity


@dataclass(frozen=True)
class RepairProposalConfig:
    version: str = "fixed-atom-proposals-v1"
    sample_draws: int = 0
    sample_seed: int = 170101
    max_sample_edits: int = 8
    connect_components: bool = True
    prune_dense: bool = True
    max_bond_changes: int = 1225
    max_macro_steps: int = 2048
    candidate_checks: int = 96

    def __post_init__(self):
        if self.version != "fixed-atom-proposals-v1":
            raise ValueError("Unknown proposal policy")
        for name, lower, upper in (("sample_draws", 0, 64), ("sample_seed", 0, 2**63-1),
                ("max_sample_edits", 1, 10000), ("max_bond_changes", 1, 10000),
                ("max_macro_steps", 1, 10000), ("candidate_checks", 1, 100000)):
            if type(getattr(self, name)) is not int or not lower <= getattr(self, name) <= upper:
                raise ValueError("Invalid proposal budget: " + name)
        if type(self.connect_components) is not bool or type(self.prune_dense) is not bool:
            raise ValueError("Proposal toggles must be explicit booleans")


def proposal_policy(config, proposals):
    legacy = repair_policy(config)
    return dict(version=proposals.version, proposals=asdict(proposals), fallback_search=legacy,
        allowed_edits=["bond"], atom_identity="fixed_exact_slot_labels", formal_charge="fixed",
        occupancy="fixed", alignment="shared_slots", cost=legacy["cost"],
        already_valid="skip_without_self_target", minimum_is_proven=False,
        target_loss="existing_repair_loss_interface; discovery_does_not_change_loss_scaling",
        selection="first_native_valid_candidate_in_cost_ranked_proposals_then_bounded_beam",
        sampling="independent_conditional_bonds_once_per_unordered_occupied_pair",
        budget="candidate native checks shared by proposals and fallback; raw and final teacher checks additional")


def _capacities(raw, spec):
    """Optimistic proposal pruning only; the native verifier remains authoritative.

    Fixed identities prevent using a larger element valence as a substitution
    shortcut. This is not a new sulfur/environment rule or a training penalty.
    Unknown charge/element rules skip macros rather than invent a valence cap.
    """
    caps = np.zeros(spec.max_nodes, dtype=float)
    for i in np.flatnonzero(raw[0]):
        allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[int(raw[0][i])], {}).get(
            spec.formal_charges[int(raw[1][i])])
        if not allowed:
            return None
        caps[i] = max(allowed)
    return caps


def _backbone(raw, probabilities, spec, caps, allow_new):
    """Greedy degree-bounded spanning forest, not an optimal chemistry solver.

    Preserve existing edges first. Reject a premature component whose remaining
    degree capacity is zero, e.g. an isolated F-F pair in a larger graph. Failure
    to construct a tree is only a heuristic failure; beam search still follows.
    """
    slots = np.flatnonzero(raw[0]).tolist()
    parent = {i: i for i in slots}
    members = {i: {i} for i in slots}
    degree = {i: 0 for i in slots}
    single = spec.bond_names.index("SINGLE")
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    edges = []
    for offset, i in enumerate(slots):
        for j in slots[offset+1:]:
            old = int(raw[2][i, j])
            if old or allow_new:
                label = old or single
                edges.append((not bool(old), -float(probabilities[2][i, j, label]), i, j))
    chosen = set()
    for _, _, i, j in sorted(edges):
        left, right = find(i), find(j)
        if left == right or degree[i] + 1 > caps[i] or degree[j] + 1 > caps[j]:
            continue
        merged = members[left] | members[right]
        spare = sum(caps[k] - degree[k] for k in merged) - 2
        if len(merged) < len(slots) and spare < 1:
            continue
        parent[right] = left; members[left] = merged
        degree[i] += 1; degree[j] += 1
        chosen.add((i, j))
        if len(chosen) == len(slots) - 1:
            return chosen
    return chosen if len(slots) <= 1 else None


def _macro_candidate(raw, p, spec, config, context):
    caps = _capacities(raw, spec)
    slots = context.slots
    if caps is None or not slots or "SINGLE" not in spec.bond_names:
        return None, "unsupported_macro_rules"
    # Necessary for a connected graph with fixed atoms; not sufficient for
    # chemical feasibility. This never rejects ordinary sampling/beam search.
    if caps.sum() < 2 * (len(slots)-1):
        return None, "insufficient_total_valence_capacity"
    tree = _backbone(raw, p, spec, caps, config.connect_components)
    if tree is None:
        return None, "no_greedy_backbone"
    bonds = raw[2].copy()
    single = spec.bond_names.index("SINGLE")
    for i, j in tree:
        if not bonds[i, j]:
            bonds[i, j] = bonds[j, i] = single
    if not config.prune_dense:
        return bonds, "component_bridges"
    orders = np.array([{"NONE": 0., "SINGLE": 1., "DOUBLE": 2.,
                        "TRIPLE": 3., "AROMATIC": 1.5}[name] for name in spec.bond_names])
    ordinary = [i for i, name in enumerate(spec.bond_names) if name != "AROMATIC"]
    # The protected backbone makes every deletion connectivity-preserving;
    # lowering its bond orders is allowed, deleting its edges is not.
    for _ in range(config.max_macro_steps):
        excess = np.maximum(orders[bonds].sum(-1) - caps, 0)
        if not excess.any():
            return bonds, "connected_valence_pruning"
        options = []
        for i, j in context.pairs:
            old = int(bonds[i, j])
            if not old or not (excess[i] or excess[j]):
                continue
            for new in ordinary:
                drop = orders[old] - orders[new]
                if drop <= 0 or (new == 0 and (i, j) in tree):
                    continue
                relief = min(excess[i], drop) + min(excess[j], drop)
                if relief <= 0:
                    continue
                original = int(raw[2][i, j])
                old_cost = 0 if old == original else 1 + context.config.probability_weight * context.regrets[i, j, old]
                new_cost = 0 if new == original else 1 + context.config.probability_weight * context.regrets[i, j, new]
                options.append(((new_cost-old_cost)/relief, -relief, i, j, new))
        if not options:
            return None, "no_connectivity_preserving_reduction"
        _, _, i, j, new = min(options)
        bonds[i, j] = bonds[j, i] = new
    return None, "macro_step_budget_exhausted"


def search_with_proposals(raw, probabilities, spec, config, proposals):
    """Return a legacy-compatible target bundle with a distinct discovery policy.

    The RNG is local and bound to raw graph, probabilities and configured seed;
    batching, workers and evaluation never advance the training RNG. Occupancy,
    atom identities and charges cannot change, even in stochastic candidates.
    """
    if not isinstance(proposals, RepairProposalConfig):
        raise TypeError("Require an explicit RepairProposalConfig")
    raw = tuple(np.array(v.detach().cpu().numpy() if hasattr(v, "detach") else v, copy=True) for v in raw)
    p = validate_probabilities(probabilities, spec)
    if set(spec.bond_names) - {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Unsupported proposal bond vocabulary")
    raw_id = graph_identity(*raw, spec)
    measure_probability_edits(raw, raw, p, spec, config)
    probability_id = probability_identity(p, spec)
    diagnostic = BondOnlyDiagnostic(raw, spec).diagnose(raw[2])
    raw_rdkit_valid = diagnostic["valid"]
    raw_roundtrip_error = None
    if raw_rdkit_valid:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                raw_roundtrip_error = "Canonical identity differs"
        except MoleculeRejected as exc:
            raw_roundtrip_error = str(exc)
        if raw_roundtrip_error is not None:
            diagnostic = dict(diagnostic, valid=False)
    report = dict(policy=proposal_policy(config, proposals), raw_graph_id=raw_id,
        probability_sha256=probability_id, raw_valid=diagnostic["valid"], checks=0,
        raw_rdkit_valid=raw_rdkit_valid, raw_roundtrip_error=raw_roundtrip_error,
        roundtrip_rejected_candidates=0,
        weighted_witness=None, best_observed_weighted_cost=None, distance_is_exact=False,
        proposal_checks=0, generated_proposals=0, filtered_proposals=0,
        fallback_checks=0, accepted_source=None, macro_status="disabled")
    if diagnostic["valid"]:
        return dict(report, status="already_valid", best_observed_weighted_cost=0.)
    if not diagnostic.get("schema_supported") or not diagnostic.get("heavy_atoms") or not diagnostic.get("categorical_support_labels_ok"):
        return dict(report, status="unsupported", reason="Requires supported nonempty fixed slots")
    context = BondSearchContext(raw, p, spec, config)
    candidates = []
    if proposals.connect_components or proposals.prune_dense:
        macro, report["macro_status"] = _macro_candidate(raw, p, spec, proposals, context)
        if macro is not None:
            candidates.append(("macro", macro, proposals.max_bond_changes))
    seed = int.from_bytes(hashlib.sha256(f"{proposals.sample_seed}:{raw_id}:{probability_id}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    for _ in range(proposals.sample_draws):
        bonds = np.zeros_like(raw[2])
        for i, j in context.pairs:
            law = p[2][i, j]
            bonds[i, j] = bonds[j, i] = rng.choice(len(law), p=law/law.sum())
        candidates.append(("bond_sample", bonds, min(proposals.max_sample_edits, proposals.max_bond_changes)))
    report["generated_proposals"] = len(candidates)
    ranked, seen = [], {raw[2].tobytes()}
    for source, bonds, limit in candidates:
        key = bonds.tobytes()
        count, cost = context.cost(bonds)
        if key in seen or not 1 <= count <= limit:
            report["filtered_proposals"] += 1
            continue
        seen.add(key); ranked.append((cost, count, len(ranked), source, bonds))
    for _, _, _, source, bonds in sorted(ranked, key=lambda x: x[:3]):
        if report["checks"] >= proposals.candidate_checks:
            break
        report["checks"] += 1; report["proposal_checks"] += 1
        inspected = inspect_graph(raw[0], raw[1], bonds, spec)
        if inspected.smiles is None:
            continue
        try:
            compatible = encode_smiles(inspected.smiles, spec)["smiles"] == inspected.smiles
        except MoleculeRejected:
            compatible = False
        if not compatible:
            report["roundtrip_rejected_candidates"] += 1
            continue
        graph = (raw[0], raw[1], bonds)
        cost = measure_probability_edits(raw, graph, p, spec, config)
        witness = dict(**cost, graph_id=graph_identity(*graph, spec), smiles=inspected.smiles,
            labels={name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), graph)})
        return dict(report, status="witness_found", weighted_witness=witness,
                    best_observed_weighted_cost=cost["weighted_cost"], accepted_source=source)
    remaining = proposals.candidate_checks - report["checks"]
    if remaining:
        depths = min(config.max_edits, remaining, proposals.max_bond_changes)
        effective = replace(config, max_edits=depths,
                            checks_per_depth=min(config.checks_per_depth, max(1, remaining//depths)))
        result = search_atom_preserving_repair_fast(raw, p, spec, effective)
        report["fallback_checks"] = result["checks"]
        report["checks"] += result["checks"]
        report["effective_fallback_search"] = asdict(effective)
        report["fallback_roundtrip_rejected_candidates"] = result.get("roundtrip_rejected_candidates", 0)
        if result["status"] == "witness_found":
            return dict(report, status="witness_found", weighted_witness=result["weighted_witness"],
                best_observed_weighted_cost=result["best_observed_weighted_cost"], accepted_source="legacy_beam")
    return dict(report, status="censored", reason="No verified target within the combined candidate budget")


class ProposalRepairTeacher(AtomPreservingRepairTeacher):
    """Drop-in detached target builder for NEW experiments, not live adoption.

    Inherits the exact same frozen target checks, all-attempt NLL, cache,
    snapshot isolation, ordered async coordinator and classification machinery.
    Only explicit search tasks and their policy differ from the legacy teacher.
    """
    def __init__(self, spec, config, *, proposals, **kwargs):
        if not isinstance(proposals, RepairProposalConfig):
            raise TypeError("Require an explicit proposal policy")
        if kwargs.get("backend", "optimized") != "optimized":
            raise ValueError("Proposal discovery uses the optimized native repair backend")
        kwargs.setdefault("backend", "optimized")
        self.proposals = proposals
        super().__init__(spec, config, **kwargs)

    def _policy(self):
        return proposal_policy(self.config, self.proposals)

    def _search(self, raw, probabilities):
        return search_with_proposals(raw, probabilities, self.spec, self.config, self.proposals)

    def _search_task(self, raw, probabilities):
        return (*super()._search_task(raw, probabilities), self.proposals)

    def _map_search(self, tasks):
        return self._pool.map_proposals(tasks)
