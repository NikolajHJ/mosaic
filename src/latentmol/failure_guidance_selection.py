"""Metadata-only hard boundary for new-objective readiness and promotion.

Keep the original eighteen native gates intact. The additional receipt is
separately hash-bound and required by BOTH scheduler and numerical trainer.
No model, NumPy, Torch, chemistry engine or scheduler is imported here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .chemistry_speed_selection import REQUIRED_GATES, summarize_candidate
from .chemistry_runtime import epoch_fit_seconds


VERSION = "failed-repair-native-readiness-v1"
RETENTION_GATES = frozenset(("finite", "size_retention", "nonempty_retention", "sulfur_no_inflation",
    "composition_no_deterioration", "charge_no_deterioration", "reconstruction_retention",
    "valid_reconstruction_retention", "raw_prior_validity_retention", "all_original_native_gates",
    "conservative_complete_epoch_fits"))
EXTRA_GATES = frozenset(("failed_rows_observed", "failed_gradient_coverage", "zero_guidance_reduced",
    "native_extra_decoder_vjp", "frozen_target_directional_derivative", "no_atom_edit_shortcut",
    "production_extra_gradient"))
SELF_GATES = frozenset(("valid_self_rows_observed", "valid_self_gradient_coverage",
    "valid_self_native_decoder_vjp", "valid_self_directional_derivative", "valid_self_production_gradient"))
WEIGHTED_TARGET_GATES = frozenset(("weighted_repair_native_decoder_vjp",
    "weighted_repair_directional_derivative", "weighted_target_policy_bound"))
REFERENCE_GATES = frozenset(("reference_initial_step_parity", "reference_invalid_live_gradient",
    "reference_directional_derivative", "reference_anchor_unchanged"))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate readiness key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def authenticate_readiness(plan, expected_sha):
    """Require the actual completed same-source native benchmark and receipt."""
    if plan.get("valid_reference_adoption") is not None:
        from .chemistry_valid_reference_adoption import authenticate_adoption
        return authenticate_adoption(plan, expected_sha)
    if plan.get("runtime_adoption") is not None:
        # An explicitly authorized runtime-only successor authenticates the
        # original saved scientific evidence and a separate additive decision.
        # It does not relabel that historical failed benchmark as successful.
        from .chemistry_runtime_adoption import authenticate_adoption
        return authenticate_adoption(plan, expected_sha)
    root = Path(plan["campaign"]).resolve(strict=True)
    path = root / "failure_readiness.json"
    if (type(expected_sha) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
            or _sha(path) != expected_sha):
        raise ValueError("Failure-guidance readiness identity changed or is missing")
    receipt = _read(path)
    failure = plan["protocol"].get("failure_guidance") is not None
    policy = plan["protocol"].get("failure_guidance")
    if failure and policy.get("progress_search") is not None and policy["progress_search"].get("allow_atom_edits") is not False:
        raise ValueError("Atom-edit search is diagnostic-only in readiness v1")
    required = RETENTION_GATES | (EXTRA_GATES if failure else frozenset())
    from .chemistry_short_continuation import validate_valid_prior_self_guidance
    if validate_valid_prior_self_guidance(plan["protocol"].get("valid_prior_self_guidance")) is not None:
        required |= SELF_GATES
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    if validate_target_weighting_test(plan) is not None:
        required |= WEIGHTED_TARGET_GATES
        if receipt.get("chemistry_target_weighting") != plan["protocol"]["chemistry_target_weighting"]:
            raise ValueError("Native readiness changed its weighted chemistry target objective")
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    reference_test=validate_valid_reference_test(plan)
    if reference_test is not None:
        required |= REFERENCE_GATES
        if (receipt.get("valid_reference_test") != reference_test
                or receipt.get("valid_prior_reference_guidance") != plan["protocol"]["valid_prior_reference_guidance"]):
            raise ValueError("Native readiness changed its frozen valid-reference objective")
    gates = receipt.get("gates")
    if (receipt.get("version") != VERSION or receipt.get("status") != "passed"
            or receipt.get("plan_sha256") != plan["_plan_sha256"]
            or receipt.get("archive_sha256") != plan["archive_sha256"]
            or receipt.get("manifest_sha256") != plan["manifest_sha256"]
            or type(gates) is not dict or set(gates) != required
            or not all(type(value) is bool and value for value in gates.values())
            or receipt.get("scientific_gate_schema") != "separate-readiness-v1-keeps-original-eighteen"
            or receipt.get("benchmark_weights_discarded") is not True):
        raise ValueError("New-objective readiness did not pass every declared exact-source gate")
    native_path = root / "benchmark_gate.json"
    if _sha(native_path) != receipt.get("native_benchmark_sha256"):
        raise ValueError("Supplementary receipt changed its native benchmark binding")
    native = _read(native_path)
    if (native.get("status") != "passed" or native.get("plan_sha256") != plan["_plan_sha256"]
            or type(native.get("gates")) is not dict or set(native["gates"]) != REQUIRED_GATES
            or not all(type(value) is bool and value for value in native["gates"].values())):
        raise ValueError("Original eighteen native gates are mandatory")
    summary = summarize_candidate(dict(id=plan.get("failure_variant_id", "failure_guidance"),
        prior_ratio=plan["protocol"]["prior_ratio"], benchmark_updates=plan["protocol"]["benchmark_updates"]),
        native, 0, training_molecules=plan["training_molecules"])
    if (not summary["eligible"] or receipt.get("conservative_epoch_seconds") != summary["selection_epoch_seconds"]
            or epoch_fit_seconds(plan, summary["selection_epoch_seconds"]) >= plan["protocol"]["segment_wall_seconds"]):
        raise ValueError("Full-update timing, exact native parity/restoration or conservative epoch fit failed")
    run = Path(native["run_dir"]).resolve(strict=True)
    record = _read(run / "run.json")
    if (not run.is_relative_to(root / "benchmark")
            or record.get("status") != "completed" or record.get("returncode", 0) != 0
            or record.get("environment", {}).get("job_id") != receipt.get("job_id")
            or record.get("source_archive_sha256") != plan["archive_sha256"]
            or record.get("source", {}).get("sha256") != plan["manifest_sha256"]
            or _sha(run / "gate.json") != _sha(native_path)):
        raise ValueError("Readiness is not bound to its completed native allocation")
    panel = receipt.get("native_panel", {})
    expected_epoch = plan.get("starting_checkpoint", {}).get("epoch", plan["parent_epoch"])
    if (panel.get("status") != "observed" or panel.get("plan_sha256") != plan["_plan_sha256"]
            or panel.get("job_id") != receipt.get("job_id") or panel.get("completed_epoch") != expected_epoch
            or panel.get("archive_sha256") != plan["archive_sha256"]
            or panel.get("manifest_sha256") != plan["manifest_sha256"]):
        raise ValueError("Readiness panel changed its native checkpoint/source identity")
    if reference_test is not None:
        from .chemistry_valid_reference_anchor import validate_anchor_identity
        from .chemistry_valid_reference_validation import require_reference_update
        reference=panel.get("valid_prior_reference_guidance",{})
        anchor=validate_anchor_identity(reference.get("reference_before"))
        restoration=reference.get("restoration",{})
        names={"model","reference","optimizer","gradients","reference_gradients","model_modes","reference_modes","rng"}
        if (reference.get("status") != "passed" or reference.get("plan_sha256") != plan["_plan_sha256"]
                or reference.get("archive_sha256") != plan["archive_sha256"]
                or reference.get("manifest_sha256") != plan["manifest_sha256"]
                or reference.get("valid_reference_test") != reference_test
                or reference.get("reference_after") != anchor
                or set(reference.get("gates",{})) != REFERENCE_GATES
                or not all(type(v) is bool and v for v in reference["gates"].values())
                or set(restoration) != names or not all(type(v) is bool and v for v in restoration.values())
                or native.get("valid_reference_anchor_before") != anchor
                or native.get("valid_reference_anchor_after") != anchor
                or native.get("valid_prior_reference_guidance") != plan["protocol"]["valid_prior_reference_guidance"]):
            raise ValueError("Frozen-reference native mechanism, restoration or anchor binding failed")
        for row in native["updates"]:
            require_reference_update(plan,row,anchor)
    return receipt


def select_candidate(rows, *, control_id="legacy3"):
    """Prefer additional complete repairs, then failed-row signal, then runtime.

    Atom-edited policies are intentionally diagnostic-only in this campaign.
    An unsuccessful candidate remains in the result; an incomplete suite stops.
    """
    expected = (control_id, "fallback", "bond6")
    if type(rows) is not list or tuple(row.get("id") for row in rows) != expected:
        raise ValueError("Incomplete or reordered declared repair comparison")
    control = rows[0]
    if not control.get("eligible"):
        return dict(status="stopped", winner=None, reason="matched_native_control_failed")
    eligible = []
    control_panel = control["readiness"]["native_panel"]
    for row in rows[1:]:
        if not row.get("eligible"):
            continue
        panel = row["readiness"]["native_panel"]
        if panel["source_snapshot_chain_sha256"] != control_panel["source_snapshot_chain_sha256"]:
            raise ValueError("Comparison did not use identical native prior graph/probability snapshots")
        for key in ("attempts", "original_accepted", "already_valid", "original_censored", "unsupported"):
            if panel["counts"][key] != control_panel["counts"][key]:
                raise ValueError("Comparison changed original teacher coverage")
        eligible.append(row)
    if not eligible:
        return dict(status="stopped", winner=None, reason="no_passing_new_objective")
    winner = min(eligible, key=lambda row: (-row["readiness"]["native_panel"]["counts"]["extra_valid"],
        -row["readiness"]["native_panel"]["counts"]["failed_with_nonzero_gradient"],
        row["selection_epoch_seconds"], row["id"]))
    return dict(status="selected_not_training_submitted", winner=winner["id"],
        reason="complete_valid_coverage_then_failed_row_signal_then_runtime", prior_ratio=2,
        selection_epoch_seconds=winner["selection_epoch_seconds"])


def summarize_plan(plan, returncode):
    """Preserve failed evidence without silently relaxing either gate family."""
    root = Path(plan["campaign"])
    row = dict(id=plan["failure_variant_id"], eligible=False, returncode=returncode)
    try:
        readiness_path = root / "failure_readiness.json"
        readiness = authenticate_readiness(plan, _sha(readiness_path))
        native = _read(root / "benchmark_gate.json")
        original = summarize_candidate(dict(id=row["id"], prior_ratio=2,
            benchmark_updates=plan["protocol"]["benchmark_updates"]), native, returncode,
            training_molecules=plan["training_molecules"])
        row.update(original)
        row.update(readiness=readiness, readiness_sha256=_sha(readiness_path),
                   plan_sha256=plan["_plan_sha256"], plan_path=plan["_plan_path"])
    except (ValueError, KeyError, TypeError, OSError) as error:
        row.update(rejection=f"{type(error).__name__}: {error}")
    return row
