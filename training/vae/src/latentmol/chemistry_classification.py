"""Detached native chemistry stages, kept separate from representation policy.

This module diagnoses saved/hard graphs only; it never repairs labels or
participates in autograd. A failed canonical/schema roundtrip is distinct
from failed RDKit chemistry. ``raw_chemical_valid`` follows the project's
existing nonempty, sanitized, connected, radical-free graph rule BEFORE that
roundtrip. ``graph_sanitization_valid`` is the narrower sanitization result;
a sanitized disconnected graph is not raw-chemical-valid under this rule.
Neither is a statement of stability, synthesis feasibility or useful binding.
Numerical chemistry fixtures must run on allocated DTU HPC.
"""
from __future__ import annotations

from copy import deepcopy
from collections import Counter

from rdkit import Chem, rdBase

from .chemistry import MoleculeRejected, encode_smiles, inspect_graph


VERSION = "repair-chemistry-classification-v1"
STAGES = ("graph_sanitization_valid", "raw_chemical_valid", "smiles_parse_valid",
          "schema_roundtrip_valid", "strict_teacher_valid")
KINDS = ("none", "chemical", "representation", "input")
_INPUT_REASONS = frozenset(("shape", "atom_category", "charge_category",
    "bond_labels_or_symmetry", "unsupported_bond", "input_or_construction"))
_CHEMICAL_REASONS = frozenset(("empty", "sanitization", "disconnected", "radical"))


def classify_repair_graph(atoms, charges, bonds, spec):
    """Return explicit native-stage outcomes without changing the input graph.

    ``None`` means a later stage was not evaluated, never that it passed.
    Chemistry uses the same graph construction as ``inspect_graph``. Expected
    source-schema rejections are classified; unexpected encoder/runtime
    exceptions propagate instead of being hidden as ordinary invalidity.
    ``smiles_parse_valid`` additionally detects a serialized-SMILES failure.
    """
    inspected = inspect_graph(atoms, charges, bonds, spec)
    record = dict(version=VERSION, graph_sanitization_valid=None,
        raw_chemical_valid=None, smiles_parse_valid=None, schema_roundtrip_valid=None,
        strict_teacher_valid=False, failure_kind="none", reason=None, detail=inspected.detail,
        smiles=inspected.smiles, representation_smiles=None)
    reason = inspected.failure_reason
    if inspected.smiles is None:
        if reason in _INPUT_REASONS:
            record.update(failure_kind="input", reason=reason)
        elif reason in _CHEMICAL_REASONS:
            record.update(failure_kind="chemical", reason=reason, raw_chemical_valid=False,
                graph_sanitization_valid=(True if reason in ("disconnected", "radical") else
                                          False if reason == "sanitization" else None))
        else:
            raise ValueError("Unknown native graph-classification failure: " + str(reason))
        return record
    if reason is not None:
        raise ValueError("Native graph inspection returned both SMILES and failure")
    record.update(graph_sanitization_valid=True, raw_chemical_valid=True)
    with rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(inspected.smiles)
    record["smiles_parse_valid"] = parsed is not None
    if parsed is None:
        return {**record, "failure_kind": "representation", "reason": "serialized_smiles_parse"}
    try:
        represented = encode_smiles(inspected.smiles, spec)["smiles"]
    except MoleculeRejected as error:
        return {**record, "schema_roundtrip_valid": False, "failure_kind": "representation",
                "reason": error.reason, "detail": str(error)}
    record["representation_smiles"] = represented
    record["schema_roundtrip_valid"] = represented == inspected.smiles
    if not record["schema_roundtrip_valid"]:
        return {**record, "failure_kind": "representation", "reason": "canonical_identity_mismatch"}
    record["strict_teacher_valid"] = True
    return record


def _validate_limit(value):
    if type(value) is not int or not 0 <= value <= 32:
        raise ValueError("Classification examples require an integer bound in [0,32]")


def classification_summary(records, *, max_examples_per_failure=4):
    """Summarize all attempts, retaining bounded original failure examples.

    Input records contain ``classification`` plus optional ``candidate_id``,
    ``raw_sha256`` and ``labels`` context. No record is treated as a valid-only
    denominator. Examples are capped separately for chemical, representation
    and input failures and remain detached JSON data. Counter semantics are
    suitable for deterministic microbatch/epoch aggregation.
    """
    _validate_limit(max_examples_per_failure)
    counts = Counter({stage: 0 for stage in STAGES})
    counts.update(chemical_failures=0, representation_failures=0, input_failures=0)
    assessments = {stage: Counter(passed=0, failed=0, not_evaluated=0) for stage in STAGES}
    reasons, examples = {}, {}
    attempts = 0
    for context in records:
        if type(context) is not dict or type(context.get("classification")) is not dict:
            raise ValueError("Classification summary requires explicit per-attempt records")
        item = context["classification"]
        if (item.get("version") != VERSION or item.get("failure_kind") not in KINDS
                or not set(STAGES).issubset(item)):
            raise ValueError("Unknown native chemistry classification version/kind")
        attempts += 1
        for stage in STAGES:
            value = item.get(stage)
            if value is not None and type(value) is not bool:
                raise ValueError("Classification stages require bool or None")
            assessments[stage]["not_evaluated" if value is None else "passed" if value else "failed"] += 1
            counts[stage] += int(value is True)
        kind = item["failure_kind"]
        if kind == "none":
            if item.get("strict_teacher_valid") is not True:
                raise ValueError("Failure-free classification must pass the strict teacher rule")
            continue
        if (item.get("strict_teacher_valid") is not False or type(item.get("reason")) is not str
                or not item["reason"]):
            raise ValueError("Failed classification requires an explicit reason and strict failure")
        counts[kind + "_failures"] += 1
        reasons.setdefault(kind, Counter())[item["reason"]] += 1
        retained = examples.setdefault(kind, [])
        if len(retained) < max_examples_per_failure:
            retained.append(deepcopy(context))
    return dict(version=VERSION, attempts=attempts, counts=dict(counts),
        assessment_counts={stage: dict(value) for stage, value in assessments.items()},
        failure_counts={kind: dict(value) for kind, value in reasons.items()},
        failure_examples=examples, max_examples_per_failure=max_examples_per_failure,
        scope="All attempted hard graphs; raw chemistry, serialization and schema roundtrip remain separate")


def merge_classification_summaries(summaries, *, max_examples_per_failure=4):
    """Add saved microbatch counters without duplicating unbounded examples."""
    result = classification_summary([], max_examples_per_failure=max_examples_per_failure)
    for value in summaries:
        if (type(value) is not dict or value.get("version") != VERSION
                or type(value.get("attempts")) is not int or value["attempts"] < 0
                or set(value.get("counts", {})) != set(result["counts"])
                or set(value.get("assessment_counts", {})) != set(STAGES)):
            raise ValueError("Incompatible chemistry-classification summary")
        result["attempts"] += value["attempts"]
        for name, count in value["counts"].items():
            if type(count) is not int or not 0 <= count <= value["attempts"]:
                raise ValueError("Classification counter is outside its all-attempt denominator")
            result["counts"][name] += count
        if (value["counts"]["strict_teacher_valid"] + sum(value["counts"][kind + "_failures"]
                for kind in KINDS[1:]) != value["attempts"]
                or any(value["counts"][later] > value["counts"][earlier] for earlier, later in
                       zip(STAGES, STAGES[1:]))):
            raise ValueError("Classification stages/failure groups disagree with all-attempt counts")
        for stage in STAGES:
            parts = value["assessment_counts"][stage]
            if (type(parts) is not dict or set(parts) != {"passed", "failed", "not_evaluated"}
                    or any(type(count) is not int or count < 0 for count in parts.values())
                    or sum(parts.values()) != value["attempts"]
                    or parts["passed"] != value["counts"][stage]):
                raise ValueError("Classification assessment denominators disagree")
            for status, count in parts.items():
                result["assessment_counts"][stage][status] += count
        for kind, reasons in value["failure_counts"].items():
            if (kind not in KINDS[1:] or type(reasons) is not dict
                    or any(type(reason) is not str or not reason or type(n) is not int or n < 0 for reason, n in reasons.items())):
                raise ValueError("Invalid classification failure counters")
            aggregate = result["failure_counts"].setdefault(kind, {})
            for reason, count in reasons.items():
                aggregate[reason] = aggregate.get(reason, 0) + count
        if any(sum(value["failure_counts"].get(kind, {}).values()) != value["counts"][kind + "_failures"]
               for kind in KINDS[1:]):
            raise ValueError("Classification failure reasons lost their group denominator")
        for kind, examples in value["failure_examples"].items():
            if kind not in KINDS[1:] or type(examples) is not list:
                raise ValueError("Invalid classification example group")
            retained = result["failure_examples"].setdefault(kind, [])
            retained.extend(deepcopy(examples[:max(0, max_examples_per_failure - len(retained))]))
    return result
