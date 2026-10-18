"""Graded, hard-graph evaluation; never part of a differentiable scorer path.

The default v2 mixture gives equal weight to verified bounded-repair proximity,
RDKit local consistency and connectedness. It is not a validity probability or
binding estimate. Explicit valence excess remains supplementary because our rule
table does not cover every charge/aromatic environment. Diagnostic repair search
never substitutes its witness for the shared raw molecule or trains a model.
External projector reports remain a separate branch. See docs27 for limits.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem, rdBase

from .chemistry import inspect_graph
from .representation import GraphSpec
from .repair_search import RepairSearchConfig, search_repair


VERSION = "graded-chemistry-v2"
COMPONENTS = ("validity", "local_consistency", "connectedness",
              "repair_quality", "native_retention", "repair_proximity")


@dataclass(frozen=True)
class QualityScoreConfig:
    """Fixed bounded components; a missing positive-weight term blocks the score.

    The default three weights of one normalize to exactly equal thirds.
    Weights need not sum to one: normalization uses the configured sum, never
    the sum of the available terms. These defaults are declared heuristics,
    not fitted coefficients. Different profiles must not be ranked together.
    """

    validity: float = 0.0
    local_consistency: float = 1.0
    connectedness: float = 1.0
    repair_quality: float = 0.0
    native_retention: float = 0.0
    repair_proximity: float = 1.0

    def __post_init__(self):
        values = asdict(self).values()
        if any(not math.isfinite(x) or x < 0 for x in values) or sum(values) <= 0:
            raise ValueError("Quality weights must be finite, nonnegative and not all zero")

    @classmethod
    def from_dict(cls, value):
        """Explicit pre-v2 JSON profiles retain their original binary component."""
        return cls(**{**value, "repair_proximity": value.get("repair_proximity", 0.0)})


def combine_quality(components: Mapping[str, float | None],
                    config: QualityScoreConfig | None = None) -> dict:
    """Return the score on [0,100], every contribution, and missing-term reasons."""
    weights = asdict(config or QualityScoreConfig())
    unknown = set(components) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown quality components: {sorted(unknown)}")
    for name, value in components.items():
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"Component {name} must be finite in [0,1], or None")
    missing = [name for name, weight in weights.items()
               if weight > 0 and components.get(name) is None]
    total = sum(weights.values())
    contributions = {name: (100 * weight * components[name] / total
                            if weight and components.get(name) is not None else
                            0.0 if not weight else None)
                     for name, weight in weights.items()}
    # Preserve v1 profile hashes when loading its original explicit JSON file.
    identity_weights = weights if weights["repair_proximity"] else {k: v for k, v in weights.items() if k != "repair_proximity"}
    identity = {"version": VERSION if weights["repair_proximity"] else "graded-chemistry-v1", "weights": identity_weights,
                "missing_policy": "undefined_if_positive_weight_missing"}
    return {**identity, "profile_sha256": hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            "score": None if missing else sum(contributions.values()),
            "status": "unavailable" if missing else "available",
            "missing_components": missing,
            "components": {name: components.get(name) for name in COMPONENTS},
            "contributions": contributions,
            "interpretation": "Heuristic chemistry diagnostic; not overall model quality"}


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "iu":
        raise ValueError("Hard graph labels must be integers")
    return result.astype(np.int64, copy=True)


def _labels(atoms, charges, bonds, spec):
    """Validate shape and active labels, matching the existing occupancy contract."""
    a, q, b = map(_array, (atoms, charges, bonds))
    if a.ndim != 1 or q.shape != a.shape or b.shape != (len(a), len(a)) or not 0 < len(a) <= spec.max_nodes:
        raise ValueError("Hard graph shape differs from schema")
    if np.any(a < 0) or np.any(a >= len(spec.atomic_numbers)):
        raise ValueError("Atom category outside schema")
    active = a != 0
    pair = active[:, None] & active[None, :]
    if (np.any(q[active] < 0) or np.any(q[active] >= len(spec.formal_charges))
            or np.any(b[pair] < 0) or np.any(b[pair] >= len(spec.bond_names))):
        raise ValueError("Active charge/bond category outside schema")
    # Padding channels have no chemical identity in our decoder's sampling law.
    q[~active] = spec.formal_charges.index(0)
    b[~pair] = 0
    return a, q, b


def graph_identity(atoms, charges, bonds, spec) -> str:
    """Hash exact slot-ordered labels, including otherwise ignored padding."""
    _labels(atoms, charges, bonds, spec)
    labels = tuple(_array(value) for value in (atoms, charges, bonds))
    payload = {"schema": spec.fingerprint(), "labels": [x.tolist() for x in labels]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _components(active, bonds):
    remaining, sizes = set(np.flatnonzero(active).tolist()), []
    while remaining:
        todo, size = [remaining.pop()], 0
        while todo:
            node = todo.pop()
            size += 1
            neighbors = {j for j in remaining if bonds[node, j] != 0 or bonds[j, node] != 0}
            remaining -= neighbors
            todo.extend(neighbors)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def diagnose_graph(atoms, charges, bonds, spec: GraphSpec) -> dict:
    """Measure a single hard attempt without correcting or rejecting chemistry.

    local_consistency = 1/(1 + problem burden). Burden counts distinct atoms
    implicated by RDKit diagnostics (or one for a global problem), plus radicals
    and explicit label/symmetry defects. It is not a graph edit distance. Unknown
    construction failures are unavailable, never implicitly perfect. Low valence
    is not penalized: hydrogens are implicit in this representation.
    """
    inspected = inspect_graph(atoms, charges, bonds, spec)
    result = {"version": VERSION, "valid": inspected.smiles is not None,
              "smiles": inspected.smiles, "failure_reason": inspected.failure_reason,
              "failure_detail": inspected.detail, "schema_supported": False,
              "local_consistency": None, "connectedness": None}
    try:
        a, q, b = _labels(atoms, charges, bonds, spec)
    except (ValueError, TypeError, IndexError) as exc:
        return {**result, "diagnostic_unavailable_reason": str(exc)}
    active = np.flatnonzero(a).tolist()
    raw_q, raw_b = _array(charges), _array(bonds)
    padding = a == 0
    padding_charges = int(np.count_nonzero(raw_q[padding] != spec.formal_charges.index(0)))
    padding_bonds = int(np.count_nonzero(raw_b[padding[:, None] | padding[None, :]]))
    sizes = _components(a != 0, b)
    symmetry = int(np.count_nonzero(np.triu(b != b.T, 1)))
    self_bonds = int(np.count_nonzero(np.diag(b)))
    result.update(schema_supported=True, graph_id=graph_identity(atoms, charges, bonds, spec),
                  heavy_atoms=len(active), component_sizes=sizes, component_count=len(sizes),
                  total_formal_charge=sum(spec.formal_charges[int(q[i])] for i in active),
                  asymmetric_pairs=symmetry, self_bonds=self_bonds,
                  nonneutral_padding_slots=padding_charges, nonzero_padding_bond_entries=padding_bonds,
                  categorical_support_labels_ok=not (padding_charges or padding_bonds or symmetry or self_bonds),
                  connectedness=(1 / len(sizes) if sizes else 0.0))
    if not active:
        return {**result, "local_consistency": 0.0, "problem_burden": 1,
                "problems": [{"type": "empty", "slots": []}],
                "valence_checked_atoms": 0, "valence_coverage": None,
                "overvalence_excess": None}
    if symmetry or self_bonds:
        return {**result, "local_consistency": 1 / (1 + symmetry + self_bonds),
                "problem_burden": symmetry + self_bonds,
                "problems": [{"type": "structural_labels", "slots": []}],
                "valence_checked_atoms": 0, "valence_coverage": 0.0,
                "overvalence_excess": None}
    bond_types = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
                  "TRIPLE": Chem.BondType.TRIPLE, "AROMATIC": Chem.BondType.AROMATIC}
    try:
        rw = Chem.RWMol()
        for i in active:
            atom = Chem.Atom(spec.atomic_numbers[int(a[i])])
            atom.SetFormalCharge(spec.formal_charges[int(q[i])])
            rw.AddAtom(atom)
        for left, i in enumerate(active):
            for right in range(left + 1, len(active)):
                name = spec.bond_names[int(b[i, active[right]])]
                if name == "NONE":
                    continue
                rw.AddBond(left, right, bond_types[name])
                if name == "AROMATIC":
                    rw.GetAtomWithIdx(left).SetIsAromatic(True)
                    rw.GetAtomWithIdx(right).SetIsAromatic(True)
        mol = rw.GetMol()
        with rdBase.BlockLogs():
            detected = Chem.DetectChemistryProblems(mol)
        problems, implicated, global_count = [], set(), 0
        for problem in detected:
            ids = ([problem.GetAtomIdx()] if hasattr(problem, "GetAtomIdx") else
                   list(problem.GetAtomIndices()) if hasattr(problem, "GetAtomIndices") else [])
            slots = [active[i] for i in ids]
            implicated.update(slots)
            global_count += int(not slots)
            problems.append({"type": problem.GetType(), "slots": slots})
        # A sanitized graph may still contain radicals, which our schema rejects.
        with rdBase.BlockLogs():
            sanitized = Chem.Mol(mol)
            flag = Chem.SanitizeMol(sanitized, catchErrors=True)
        if int(flag) == 0:
            radical_slots = [active[x.GetIdx()] for x in sanitized.GetAtoms()
                             if x.GetNumRadicalElectrons()]
            implicated.update(radical_slots)
            if radical_slots:
                problems.append({"type": "radical", "slots": radical_slots})
        elif not detected:
            # Keep unknown sanitize failures as one observable problem.
            global_count += 1
            problems.append({"type": "sanitization_other", "slots": []})
        burden = len(implicated) + global_count
        result.update(problems=problems, problem_burden=burden,
                      local_consistency=1 / (1 + burden))
    except (ValueError, RuntimeError, KeyError, OverflowError) as exc:
        result.update(diagnostic_unavailable_reason=type(exc).__name__)
    # Supplementary severity only: neutral elements use RDKit's finite allowed
    # valences. A small explicit charged table and no aromatic approximation.
    charged_caps = {(7, 1): 4, (7, -1): 2, (8, 1): 3, (8, -1): 1,
                    (6, 1): 3, (6, -1): 3}
    orders = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}
    checks = []
    for i in active:
        names = [spec.bond_names[int(b[i, j])] for j in active]
        if any(name not in orders for name in names):
            continue
        element, charge = spec.atomic_numbers[int(a[i])], spec.formal_charges[int(q[i])]
        try:
            allowed = list(Chem.GetPeriodicTable().GetValenceList(element)) if charge == 0 else []
        except (ValueError, RuntimeError, OverflowError):
            continue
        cap = max(allowed) if allowed and min(allowed) >= 0 else charged_caps.get((element, charge))
        if cap is None:
            continue
        valence = sum(orders[name] for name in names)
        checks.append({"slot": i, "bond_order_sum": valence, "cap": cap,
                       "excess": max(0, valence - cap)})
    result.update(valence_checks=checks, valence_checked_atoms=len(checks),
                  valence_coverage=len(checks) / len(active),
                  overvalence_excess=sum(x["excess"] for x in checks) if checks else None)
    return result


class GradedEvaluator:
    """Evaluation-only diagnosis plus bounded repair search, cached per graph.

    Cache hits retain separate attempt rows and denominators. The shared search
    budget applies per unique graph; actual candidate checks and cache hits are
    available for compute accounting. Training probes do not instantiate this.
    """

    def __init__(self, spec: GraphSpec, config: QualityScoreConfig | None = None,
                 search_config: RepairSearchConfig | None = None):
        self.spec = spec
        self.config = config or QualityScoreConfig()
        self.search_config = search_config or RepairSearchConfig()
        self.cache = OrderedDict()
        self.cache_hits = self.physical_candidate_checks = 0

    def diagnose(self, atoms, charges, bonds):
        try:
            key = graph_identity(atoms, charges, bonds, self.spec)
        except (ValueError, TypeError, IndexError):
            key = None
        if key is not None and key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return dict(self.cache[key])
        record = diagnose_graph(atoms, charges, bonds, self.spec)
        if self.config.repair_proximity > 0:
            search = search_repair((atoms, charges, bonds), self.spec, record,
                                   diagnose_graph, self.search_config)
            record["repair_search"] = search
            record["repair_proximity"] = search["proximity"]
            self.physical_candidate_checks += search["checks"]
        if key is not None:
            self.cache[key] = record
            if len(self.cache) > 512:
                self.cache.popitem(last=False)
        return dict(record)

    def compute(self):
        return {"physical_candidate_checks": self.physical_candidate_checks,
                "cache_hits": self.cache_hits, "cache_capacity": 512,
                "search_policy": self.search_config.policy() if self.config.repair_proximity else None}


def graph_edit_cost(raw, corrected, spec: GraphSpec, *, corrected_to_raw=None) -> dict:
    """Count label edits with an explicit native-to-shared slot permutation.

    Each input is a tuple (atoms, charges, bonds). A None mapping asserts that
    both already use the same slots, including padding. No canonical SMILES
    reorder or inferred atom matching is performed. Insert/delete changes count
    as occupancy edits, not atom substitutions; incident bonds count separately.
    """
    left, right = _labels(*raw, spec), _labels(*corrected, spec)
    n = len(left[0])
    if len(right[0]) != n:
        raise ValueError("Repair outputs must be padded to the shared slot count")
    if corrected_to_raw is not None:
        mapping = _array(corrected_to_raw)
        if mapping.shape != (n,) or sorted(mapping.tolist()) != list(range(n)):
            raise ValueError("Alignment must be a permutation of all shared slots")
        inverse = np.argsort(mapping)
        right = (right[0][inverse], right[1][inverse], right[2][np.ix_(inverse, inverse)])
    for _, _, bonds in (left, right):
        if not np.array_equal(bonds, bonds.T) or np.any(np.diag(bonds)):
            raise ValueError("Edit metric requires symmetric bonds without self loops")
    occupied_left, occupied_right = left[0] != 0, right[0] != 0
    both = occupied_left & occupied_right
    counts = {"occupancy": int(np.count_nonzero(occupied_left != occupied_right)),
              "atom": int(np.count_nonzero((left[0] != right[0]) & both)),
              "charge": int(np.count_nonzero((left[1] != right[1]) & both)),
              "bond": int(np.count_nonzero(np.triu(left[2] != right[2], 1)))}
    union_size = int(np.count_nonzero(occupied_left | occupied_right))
    return {"counts": counts, "total": sum(counts.values()),
            "per_union_atom": sum(counts.values()) / max(1, union_size),
            "weights": {key: 1 for key in counts}, "alignment": "shared_slots"}


def measure_repairs(raw, attempts: Sequence, spec: GraphSpec, *, budget: dict,
                    mappings=None, edit_scale: float = 3.0, fixed_occupancy=False) -> dict:
    """Measure supplied external outputs; None retains a failed attempt.

    Required budget keys identify method/checkpoint and exact attempt/transition
    allocation. No calls are made here. A censored repair gets zero *observed
    repair utility*, but its unknown edit distance stays None. Not-run repair
    should be represented by absence of this report, never by an empty budget.
    """
    required = {"projector", "checkpoint_sha256", "attempts", "transitions_per_attempt", "seed", "policy"}
    if (not required <= budget.keys() or not budget["projector"] or not budget["checkpoint_sha256"]
            or any(type(budget[key]) is not int or budget[key] < 1
                   for key in ("attempts", "transitions_per_attempt"))
            or type(budget["seed"]) is not int or not isinstance(budget["policy"], dict) or not budget["policy"]
            or len(attempts) != budget["attempts"]):
        raise ValueError("Repair evidence must match a positive, identified fixed budget")
    if not math.isfinite(edit_scale) or edit_scale <= 0:
        raise ValueError("edit_scale must be finite and positive")
    mappings = [None] * len(attempts) if mappings is None else mappings
    if len(mappings) != len(attempts):
        raise ValueError("Every repair attempt needs an alignment entry")
    original = diagnose_graph(*raw, spec)
    if not original["schema_supported"]:
        raise ValueError("Raw repair input is outside the shared schema")
    if original["asymmetric_pairs"] or original["self_bonds"]:
        raise ValueError("Raw repair input must have symmetric bonds without self loops")
    records, valid_costs = [], []
    for graph, mapping in zip(attempts, mappings):
        if graph is None:
            records.append({"status": "failed", "edit_cost": None})
            continue
        diagnostics = diagnose_graph(*graph, spec)
        if not diagnostics["schema_supported"]:
            records.append({"status": "unsupported", "diagnostics": diagnostics, "edit_cost": None})
            continue
        if diagnostics["asymmetric_pairs"] or diagnostics["self_bonds"]:
            records.append({"status": "structural_unsupported", "diagnostics": diagnostics, "edit_cost": None})
            continue
        # Invalid native alignment is a contract error, never a chemical failure.
        cost = graph_edit_cost(raw, graph, spec, corrected_to_raw=mapping)
        supported = not fixed_occupancy or cost["counts"]["occupancy"] == 0
        status = "valid" if diagnostics["valid"] and supported else "invalid" if supported else "occupancy_unsupported"
        records.append({"status": status, "diagnostics": diagnostics, "edit_cost": cost,
                        "corrected_to_raw": None if mapping is None else list(map(int, mapping))})
        if status == "valid":
            valid_costs.append(cost["total"])
    best = 0 if original["valid"] else min(valid_costs) if valid_costs else None
    for index, record in enumerate(records):
        record["attempt_index"] = index
    return {"raw_graph_id": original["graph_id"], "budget": dict(budget),
            "edit_scale": edit_scale, "fixed_occupancy": fixed_occupancy,
            "status": "already_valid" if original["valid"] else "repaired" if valid_costs else "censored",
            "best_observed_edit_cost": best, "valid_output_count": len(valid_costs),
            "repair_quality": math.exp(-best / edit_scale) if best is not None else 0.0,
            "attempts": records}


def summarize_quality(records: Sequence[dict], config: QualityScoreConfig | None = None,
                      *, repairs: Sequence[dict] | None = None) -> dict:
    """Aggregate all attempts equally; undefined diagnostics keep their denominator.

    Argmax, categorical, prior and neighborhood cohorts must stay separate.
    Native retention is deliberately unavailable until a native scorer-specific
    normalization and identity contract is supplied by a later integration.
    """
    n = len(records)
    values = {"validity": sum(x["valid"] for x in records) / n if n else None}
    coverage = {"validity": n}
    for name in ("local_consistency", "connectedness", "repair_proximity"):
        observed = [x.get(name) for x in records if x.get(name) is not None]
        coverage[name] = len(observed)
        values[name] = sum(observed) / n if n and len(observed) == n else None
    values.update(repair_quality=None, native_retention=None)
    coverage.update(repair_quality=0, native_retention=0)
    repair_summary = {"status": "not_run", "reason": "No external repair evidence supplied"}
    if repairs is not None:
        if len(repairs) != n or not n:
            raise ValueError("Repair reports must match every attempted raw graph")
        if any(r["raw_graph_id"] != g.get("graph_id") for r, g in zip(repairs, records)):
            raise ValueError("Repair report belongs to a different raw graph")
        identity = lambda r: (r["budget"], r["edit_scale"], r["fixed_occupancy"])
        if any(identity(r) != identity(repairs[0]) for r in repairs):
            raise ValueError("Repair budgets and policies must match across attempts")
        values["repair_quality"] = sum(r["repair_quality"] for r in repairs) / n
        coverage["repair_quality"] = n
        repair_summary = {"status": "measured", "budget": repairs[0]["budget"],
                          "edit_scale": repairs[0]["edit_scale"],
                          "fixed_occupancy": repairs[0]["fixed_occupancy"],
                          "status_counts": dict(Counter(r["status"] for r in repairs))}
    supplementary = {}
    for name in ("problem_burden", "overvalence_excess", "valence_coverage", "total_formal_charge"):
        available = [x[name] for x in records if x.get(name) is not None]
        supplementary[name] = {"observations": len(available),
                               "observed_mean": sum(available) / len(available) if available else None}
    invalid = [x for x in records if not x["valid"]]
    invalid_local = [x["local_consistency"] for x in invalid if x.get("local_consistency") is not None]
    searches = [x["repair_search"] for x in records if "repair_search" in x]
    if searches and any(x["policy"] != searches[0]["policy"] for x in searches):
        raise ValueError("Repair-search policies must match across the cohort")
    mixture = combine_quality(values, config)
    bounds = None
    if searches and len(searches) == n:
        bounds = [sum(x["proximity_bounds"][i] for x in searches) / n for i in (0, 1)]
        mixture["score_interval"] = [combine_quality({**values, "repair_proximity": bound}, config)["score"]
                                     for bound in bounds]
        mixture["interval_interpretation"] = "Search-space distance bounds, not a statistical confidence interval"
    return {"attempts": n, "component_observations": coverage,
            "repair_search": {"status_counts": dict(Counter(x["status"] for x in searches)),
                              "policy": searches[0]["policy"] if searches else None,
                              "proximity_interval": bounds},
            "supplementary": supplementary,
            "invalid_attempts": len(invalid), "invalid_local_observations": len(invalid_local),
            "invalid_local_consistency_mean": sum(invalid_local) / len(invalid)
                    if invalid and len(invalid_local) == len(invalid) else None,
            "schema_unsupported": sum(not x["schema_supported"] for x in records),
            "failure_counts": dict(Counter(x["failure_reason"] for x in records if x["failure_reason"])),
            "mean_heavy_atoms": sum(x.get("heavy_atoms", 0) for x in records) / n
                                 if n and all("heavy_atoms" in x for x in records) else None,
            "repair": repair_summary, "native_retention_status": "not_run",
            "mixture": mixture}
