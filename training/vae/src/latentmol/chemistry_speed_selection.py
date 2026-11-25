"""Metadata-only selection of a fully gated chemistry execution candidate.

Ten hours is the preference threshold for retaining 1:3, not permission to
weaken any scientific check. If no healthy 1:3 candidate meets it, the suite
evaluates 1:2 and selects its fastest healthy candidate even if still slower
than ten hours. No passing fallback means no training submission.
"""
from __future__ import annotations

import math
import re


REQUIRED_GATES = frozenset((
    "finite", "size_retention", "nonempty_retention", "sulfur_no_inflation",
    "composition_no_deterioration", "charge_no_deterioration",
    "reconstruction_retention", "valid_reconstruction_retention", "exact_ratio",
    "exact_native_resume", "prior_target_coverage", "actual_repair_gradient",
    "decoder_gradient", "encoder_gradient", "memory", "complete_epoch_fits",
    "no_search_errors", "exact_reference_execution",
))
REQUIRED_PARITY_CHECKS = frozenset((
    "full_logical_batch", "metrics_exact", "ordered_teacher_receipt_chain_exact",
    "gradients_exact", "reference_gradients_exact", "model_exact", "reference_exact",
    "optimizer_exact", "rng_exact", "model_modes_exact", "reference_modes_exact",
    "reference_unchanged", "repair_target_observed",
))


def _variant(variant):
    if (type(variant) is not dict or type(variant.get("id")) is not str
            or re.fullmatch(r"[a-z0-9_]{1,80}", variant["id"]) is None
            or type(variant.get("prior_ratio")) is not int or variant["prior_ratio"] not in (2, 3)
            or type(variant.get("benchmark_updates")) is not int or variant["benchmark_updates"] < 1):
        raise ValueError("Malformed declared speed candidate")


def summarize_candidate(variant, gate, returncode, *, training_molecules, real_batch_size=256):
    """Require all native gates and conservatively rank complete update walls.

The native timing includes data collation and synchronizes CUDA. The maximum
of the declared timed updates avoids selecting a candidate on a lucky median.
The existing 600-second evaluation allowance remains an estimate, not an
observed full epoch or a confidence interval.
"""
    _variant(variant)
    if any(type(value) is not int or value < 1 for value in (training_molecules, real_batch_size)):
        raise ValueError("Positive integer training count and real batch size required")
    row = {"id": variant["id"], "prior_ratio": variant["prior_ratio"], "returncode": returncode,
           "eligible": False, "selection_epoch_seconds": None}
    try:
        if type(returncode) is not int or returncode != 0 or type(gate) is not dict or gate.get("status") != "passed":
            raise ValueError("Native benchmark did not complete successfully")
        gates = gate.get("gates", {})
        if type(gates) is not dict or set(gates) != REQUIRED_GATES or not all(type(value) is bool and value for value in gates.values()):
            raise ValueError("All eighteen original native gates are mandatory")
        proof = gate.get("native_execution_parity", {})
        if (type(proof) is not dict or proof.get("status") != "passed" or proof.get("passed") is not True
                or type(proof.get("restoration")) is not dict or proof["restoration"].get("passed") is not True
                or type(proof.get("checks")) is not dict or set(proof["checks"]) != REQUIRED_PARITY_CHECKS
                or not all(value is True for value in proof["checks"].values())):
            raise ValueError("Native full-update/reference/restoration parity is missing")
        updates = gate.get("updates", [])
        if type(updates) is not list or len(updates) != variant["benchmark_updates"]:
            raise ValueError("Timed native update coverage differs from the declared candidate")
        walls = []
        for update in updates:
            if type(update) is not dict:
                raise ValueError("Malformed timed update")
            value = update.get("update_seconds")
            if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                    or type(update.get("real_points")) is not int or type(update.get("fresh_prior_points")) is not int
                    or update.get("real_points") != real_batch_size
                    or update.get("fresh_prior_points") != variant["prior_ratio"] * real_batch_size):
                raise ValueError("Native update timing/sample count differs")
            walls.append(value)
        projected = gate.get("projected_epoch_seconds")
        if type(projected) not in (int, float) or not math.isfinite(projected) or projected <= 0:
            raise ValueError("Native epoch projection is missing")
        conservative = max(projected, max(walls) * training_molecules / real_batch_size + 600.)
        row.update(eligible=True, selection_epoch_seconds=conservative,
                   native_projected_epoch_seconds=projected, timed_update_seconds=walls,
                   native_gate_plan_sha256=gate.get("plan_sha256"), gates=gates)
    except (ValueError, TypeError, KeyError) as exc:
        row["rejection"] = f"{type(exc).__name__}: {exc}"
    return row


def choose_candidate(rows, *, preferred_epoch_seconds=36000., fallback_evaluated=False):
    """Deterministic preference: qualifying 1:3, otherwise fastest healthy 1:2.

Tie-break by canonical candidate ID. A ratio-2 fallback is explicitly a changed
sampling protocol; it is never represented as a bitwise-equivalent 1:3 run.
"""
    if (type(preferred_epoch_seconds) not in (int, float)
            or not math.isfinite(preferred_epoch_seconds) or preferred_epoch_seconds <= 0):
        raise ValueError("Require a positive finite preference threshold")
    if type(fallback_evaluated) is not bool or type(rows) is not list:
        raise ValueError("Require explicit fallback state and ordered candidate rows")
    for row in rows:
        if (type(row) is not dict or type(row.get("id")) is not str
                or re.fullmatch(r"[a-z0-9_]{1,80}", row["id"]) is None
                or type(row.get("prior_ratio")) is not int or row["prior_ratio"] not in (2, 3)
                or type(row.get("eligible")) is not bool):
            raise ValueError("Malformed candidate observation")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate observation")
    passing = [row for row in rows if row.get("eligible") is True]
    for row in passing:
        value = row.get("selection_epoch_seconds")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Eligible candidate has invalid duration")
    preferred = [row for row in passing if row["prior_ratio"] == 3
                 and row["selection_epoch_seconds"] < preferred_epoch_seconds]
    if preferred:
        chosen, reason = min(preferred, key=lambda row: (row["selection_epoch_seconds"], row["id"])), "passing_1to3_below_preference_threshold"
    elif not fallback_evaluated:
        return {"status": "fallback_required", "winner": None, "reason": "no_passing_1to3_below_preference_threshold"}
    else:
        fallback = [row for row in passing if row["prior_ratio"] == 2]
        if not fallback:
            return {"status": "stopped", "winner": None, "reason": "no_passing_1to2_candidate"}
        chosen, reason = min(fallback, key=lambda row: (row["selection_epoch_seconds"], row["id"])), "fastest_passing_1to2_fallback"
    return {"status": "selected_not_training_started", "winner": chosen["id"], "reason": reason,
            "prior_ratio": chosen["prior_ratio"], "selection_epoch_seconds": chosen["selection_epoch_seconds"],
            "below_preference_threshold": chosen["selection_epoch_seconds"] < preferred_epoch_seconds,
            "preference_threshold_seconds": preferred_epoch_seconds}
