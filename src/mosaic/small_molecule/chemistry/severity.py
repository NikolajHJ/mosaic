"""Opt-in detached, severity-aware bond repair for far-from-valid graphs.

This proposal heuristic never differentiates through atom-dependent valence
caps and never edits an atom, charge or occupied slot. Aromatic order 1.5 is a
declared ranking approximation, not a chemical-validity certificate. Native
diagnostics guard every retained stage; complete witnesses require the same
sanitization/schema roundtrip as the original preparation.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from ._reference.chemistry import encode_smiles
from ._reference.chemistry_regularization import ALLOWED_VALENCES
from ._reference.probability_repair import (
    ProbabilityRepairConfig, measure_probability_edits, probability_identity,
    validate_probabilities,
)
from ._reference.quality import _components, diagnose_graph
from ._reference.repair_progress import (
    ProgressWitness, diagnostic_vector, strictly_improves,
)


@dataclass(frozen=True)
class SeverityProgressConfig:
    version: str = "severity-progress-v1"
    max_edits: int = 6
    max_checks: int = 12
    candidates_per_step: int = 32

    def __post_init__(self):
        if self.version != "severity-progress-v1":
            raise ValueError("Unknown severity progress policy")
        for name, maximum in (
            ("max_edits", 300), ("max_checks", 4096),
            ("candidates_per_step", 4096),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Invalid bounded severity budget: " + name)

    def record(self):
        return {
            **asdict(self), "atom_identity": "fixed_source_labels",
            "charges": "fixed", "occupancy": "fixed",
            "allowed_edits": "bond_order_reductions_only",
            "grading": "total_then_worst_positive_valence_excess_in_half_units",
            "aromatic_grading_order": 1.5,
            "aromatic_grading_is_not_native_validity": True,
            "safeguards": "no_worse_native_error_counts_or_component_count",
            "complete_validity": "native_sanitization_and_schema_roundtrip",
            "partial_is_valid": False, "minimum_is_proven": False,
            "replaces_legacy_progress_when_enabled": True,
            "max_checks_includes_source_assessment": True,
            "detached_edit_cost": "edit_count + probability_regret; floor1e-12",
        }


def severity_config(value):
    if isinstance(value, dict):
        value = SeverityProgressConfig(**value)
    if not isinstance(value, SeverityProgressConfig):
        raise TypeError("Require an explicit SeverityProgressConfig or its dict")
    return value


def _caps(graph, spec):
    caps = []
    for atom, charge in zip(graph[0], graph[1]):
        if not atom:
            caps.append(0)
            continue
        allowed = ALLOWED_VALENCES.get(spec.atomic_numbers[int(atom)], {}).get(
            spec.formal_charges[int(charge)]
        )
        if not allowed:
            return None
        caps.append(2 * max(allowed))
    return np.asarray(caps, dtype=np.int64)


def _valence(graph, orders):
    return np.sum(orders[graph[2]], axis=-1)


def _observe(graph, spec, caps, orders):
    diagnostic = diagnose_graph(*graph, spec)
    if diagnostic["valid"]:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                diagnostic = {**diagnostic, "valid": False}
        except ValueError:
            diagnostic = {**diagnostic, "valid": False}
    vector = diagnostic_vector(diagnostic)
    if vector is not None:
        excess = np.maximum(_valence(graph, orders) - caps, 0)
        vector = {
            **vector, "severity_total_half_units": int(np.sum(excess)),
            "severity_worst_half_units": int(np.max(excess, initial=0)),
        }
    return diagnostic, vector


def _proposals(current, original, probabilities, caps, orders, spec):
    """Rank cheap integer changes before spending the native-check budget."""
    valence = _valence(current, orders)
    before = np.maximum(valence - caps, 0)
    active = np.flatnonzero(current[0])
    options = []
    for left, i in enumerate(active):
        for j in active[left + 1:]:
            old = int(current[2][i, j])
            if (not old or current[2][i, j] != original[2][i, j]
                    or not (before[i] or before[j])):
                continue
            for new, name in enumerate(spec.bond_names):
                if name == "AROMATIC" or orders[new] >= orders[old]:
                    continue
                after_valence = valence.copy()
                decrease = orders[old] - orders[new]
                after_valence[i] -= decrease
                after_valence[j] -= decrease
                after = np.maximum(after_valence - caps, 0)
                if np.sum(after) >= np.sum(before):
                    continue
                p = probabilities[2]
                regret = max(0.0, math.log(max(float(p[i, j, old]), 1e-12))
                             - math.log(max(float(p[i, j, new]), 1e-12)))
                rank = (int(np.sum(after)), int(np.max(after, initial=0)),
                        regret, int(i), int(j), new)
                options.append((rank, int(i), int(j), new))
    options.sort(key=lambda item: item[0])
    return options


def search_severity_progress(raw, probabilities, spec, config):
    """Return a bounded complete or honestly partial atom-preserving witness.

    Native error counts and connected-component count may not worsen. A
    decrease in represented valence severity counts as partial progress even
    when the same atoms still have native valence errors. Targets belong only
    to this source snapshot and are frozen by the ordinary preparation hash.
    """
    config = severity_config(config)
    original = tuple(np.asarray(value).copy() for value in raw)
    p = validate_probabilities(probabilities, spec)
    cost_config = ProbabilityRepairConfig(max_edits=config.max_edits)
    measure_probability_edits(original, original, p, spec, cost_config)
    if set(spec.bond_names) != {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
        raise ValueError("Severity search requires the declared five bond categories")
    order_map = {"NONE": 0, "SINGLE": 2, "DOUBLE": 4, "TRIPLE": 6, "AROMATIC": 3}
    orders = np.asarray([order_map[name] for name in spec.bond_names], dtype=np.int64)
    caps = _caps(original, spec)
    result = {
        "policy": config.record(), "status": "no_progress", "checks": 0,
        "before": None, "witness": None, "valid_witnesses": 0,
        "partial_witnesses": 0, "connectivity_rejections": 0,
        "native_guard_rejections": 0, "structural_candidates": 0,
        "source_probability_sha256": probability_identity(p, spec),
    }
    if caps is None:
        return {**result, "status": "unsupported", "reason": "Unknown frozen valence rule"}
    initial, before = _observe(original, spec, caps, orders)
    result["checks"] = 1
    result["before"] = before
    if initial["valid"]:
        return {**result, "status": "already_valid"}
    if before is None:
        return {**result, "status": "unsupported", "reason": "Unavailable native diagnostics"}
    current, current_vector = original, before
    best = None
    for _ in range(config.max_edits):
        retained = None
        candidates = _proposals(current, original, p, caps, orders, spec)
        for _, i, j, new in candidates[:config.candidates_per_step]:
            if result["checks"] >= config.max_checks:
                break
            result["structural_candidates"] += 1
            candidate = tuple(value.copy() for value in current)
            candidate[2][i, j] = candidate[2][j, i] = new
            components = len(_components(candidate[0] != 0, candidate[2]))
            if components > current_vector.get("extra_components", 0) + 1:
                result["connectivity_rejections"] += 1
                continue
            diagnostic, after = _observe(candidate, spec, caps, orders)
            result["checks"] += 1
            if after is None or not strictly_improves(current_vector, after):
                result["native_guard_rejections"] += 1
                continue
            measured = measure_probability_edits(original, candidate, p, spec, cost_config)
            if measured["edit_counts"]["atom"] or measured["edit_counts"]["charge"]:
                raise RuntimeError("Severity search changed frozen identities")
            kind = "valid" if diagnostic["valid"] else "partial"
            witness = ProgressWitness.create(
                kind, original, candidate, measured["weighted_cost"], before, after,
            )
            witness.verify(spec)
            result[kind + "_witnesses"] += 1
            retained = candidate, after, witness
            break
        if retained is None:
            break
        current, current_vector, best = retained
        if best.kind == "valid":
            break
    if best is not None:
        result.update(status=best.kind, witness=best)
    result["budget_exhausted"] = result["checks"] >= config.max_checks
    return result
