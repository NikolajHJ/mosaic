"""Metadata-only gate for the corrected chemistry objective and guarded restart.

The short native benchmark remains mandatory. A second, longer disposable
trajectory must pass the same fixed-panel checks before either the scheduler
or numerical trainer may start a continuation. No scientific library is
imported here, so both boundaries authenticate the same saved evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


VERSION = "guarded-chemistry-restart-v1"
GATE_VERSION = "guarded-chemistry-stability-v1"
GATES = frozenset(("all_native_readiness", "exact_starting_checkpoint", "full_update_budget",
    "all_early_panels_passed", "exact_panel_schedule", "finite_updates", "exact_sampling_ratio",
    "no_search_errors", "encoder_decoder_gradients", "memory", "complete_epoch_fits",
    "validation_weights_discarded", "classified_failure_counts"))


def _read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate guarded-restart metadata key")
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def readiness_science_sha256(receipt):
    """Avoid a circular digest between the native and longer stability gates.

    Supplementary baseline/stability bindings are authenticated separately;
    all original native scientific observations and checks remain in this hash.
    """
    native = {k: v for k, v in receipt.items()
              if k not in ("early_health_baseline_sha256", "stability_gate_sha256")}
    return hashlib.sha256(json.dumps(native, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_guarded_restart(plan):
    """Absence preserves v1; the corrected v2 objective requires its new gate."""
    corrected = (plan.get("protocol", {}).get("failure_guidance") or {}).get("loss", {}).get("version") == "repair-failure-guidance-v2"
    from .chemistry_short_continuation import validate_short_continuation
    short = validate_short_continuation(plan)
    if "guarded_restart" not in plan:
        if corrected:
            raise ValueError("Corrected failure guidance requires explicit guarded restart validation")
        return None
    value = plan["guarded_restart"]
    if ((not corrected and short is None) or type(value) is not dict
            or set(value) != {"version", "validation_updates", "validation_wall_seconds"}
            or value["version"] != VERSION
            or type(value["validation_updates"]) is not int or not 2400 <= value["validation_updates"] <= 6500
            or type(value["validation_wall_seconds"]) is not int
            or value["validation_wall_seconds"] != 82800
            or plan.get("protocol", {}).get("early_health") is None
            or plan.get("starting_checkpoint", {}).get("epoch") != 36
            or plan.get("runtime_adoption") is not None):
        raise ValueError("Require the explicit epoch36 corrected-objective stability protocol")
    return dict(value)


def panel_schedule(policy, total):
    """Relative update indices shared by validation, its verifier and tests."""
    first, interval = policy["first_check_updates"], policy["interval_updates"]
    return sorted({first, *range(interval, total + 1, interval), total})


def _positive_seconds(value, *, zero=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value >= 0 if zero else value > 0))


def _authenticate_worker_startup(plan, gate, run_path, rows, native, expected_schedule):
    """Authenticate an opt-in cold-pool receipt and recompute every warm wall.

    Legacy plans do not enter this path. The shared projection helper imports
    no scientific packages, so scheduler and numerical boundaries use precisely
    the same arithmetic without trusting a caller's claimed complete-epoch fit.
    Worker startup is charged once; every measured recurring update and all
    observed panel timings remain in the conservative projection.
    """
    from .chemistry_worker_startup import (NATIVE_MODULES, VERSION as STARTUP_VERSION,
                                          project_warmed_epoch, validate_worker_startup)
    if validate_worker_startup(plan.get("worker_startup")) is None:
        return
    binding = gate.get("worker_startup")
    if (type(binding) is not dict or set(binding) != {"path", "sha256", "elapsed_seconds", "status"}
            or binding["status"] != "passed" or type(binding["path"]) is not str
            or not _positive_seconds(binding["elapsed_seconds"])):
        raise ValueError("Warmed stability proof lacks its exact worker startup binding")
    path = Path(binding["path"]).resolve(strict=True)
    if path != run_path / "worker-startup.json" or _sha(path) != binding["sha256"]:
        raise ValueError("Worker startup escaped its proof run or changed")
    receipt = _read(path)
    if type(receipt) is not dict:
        raise ValueError("Worker startup receipt must be an object")
    start = plan["starting_checkpoint"]
    restoration_names = {"model", "reference", "optimizer", "gradients", "reference_gradients",
                         "model_modes", "reference_modes", "rng"}
    restoration = receipt.get("restoration", {})
    restored = restoration.get("checks")
    identities = {"plan_sha256": plan["_plan_sha256"], "archive_sha256": plan["archive_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "dataset_sha256": plan["dataset_sha256"],
        "starting_checkpoint_sha256": start["sha256"], "model_sha256": gate.get("starting_model_sha256")}
    zero_work = ("model_forwards", "optimizer_steps", "random_samples", "target_searches", "target_cache_entries")
    pids = receipt.get("observed_task_pids")
    execution = plan.get("execution", {})
    if (execution.get("backend") != "optimized" or execution.get("workers") != 12
            or execution.get("torch_threads") != 4 or execution.get("gpu_cpus") != 16
            or receipt.get("version") != STARTUP_VERSION or receipt.get("status") != "passed"
            or receipt.get("passed") is not True or receipt.get("job_id") != gate.get("job_id")
            or type(receipt.get("job_id")) is not str or not receipt["job_id"].isdecimal()
            or int(receipt["job_id"]) < 1
            or any(type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None
                   or receipt.get(key) != expected for key, expected in identities.items())
            or type(receipt.get("warmup_real_points")) is not int or receipt["warmup_real_points"] != 256
            or type(receipt.get("initialized_workers")) is not int or receipt["initialized_workers"] != 12
            or receipt.get("worker_initialization_barrier_passed") is not True
            or receipt.get("modules") != list(NATIVE_MODULES)
            or any(type(receipt.get(key)) is not int or receipt[key] != 0 for key in zero_work)
            or type(pids) is not list or not 1 <= len(pids) <= 12
            or any(type(pid) is not int or pid < 1 for pid in pids) or len(set(pids)) != len(pids)
            or not _positive_seconds(receipt.get("elapsed_seconds"))
            or receipt["elapsed_seconds"] != binding["elapsed_seconds"]
            or not _positive_seconds(receipt.get("worker_start_and_classification_seconds"))
            or receipt["worker_start_and_classification_seconds"] > receipt["elapsed_seconds"]
            or restoration.get("passed") is not True or type(restored) is not dict
            or set(restored) != restoration_names or not all(type(value) is bool and value for value in restored.values())):
        raise ValueError("Worker startup did not preserve the exact zero-update scientific state")
    for key in ("real_row_ids_sha256",):
        if type(receipt.get(key)) is not str or re.fullmatch(r"[0-9a-f]{64}", receipt[key]) is None:
            raise ValueError("Worker startup lost authenticated real-row identity")
    hard = receipt.get("hard_graphs")
    classification = receipt.get("classification")
    if (type(hard) is not dict or type(classification) is not dict
            or type(hard.get("sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", hard["sha256"]) is None
            or type(hard.get("named_entries")) is not int or hard["named_entries"] != 3
            or type(hard.get("nonmissing_entries")) is not int or hard["nonmissing_entries"] != 3
            or hard.get("missing_names") != [] or type(classification.get("attempts")) is not int
            or classification["attempts"] != 256):
        raise ValueError("Worker startup did not classify all three heads of 256 real graphs")
    panels = gate.get("fixed_panel_seconds")
    if (type(panels) is not list or len(panels) != len(expected_schedule)
            or not all(_positive_seconds(value) for value in panels)):
        raise ValueError("Warmed runtime lost actual fixed-panel timings")
    training = plan.get("training_molecules")
    real_count = plan["protocol"].get("real_batch_size")
    if (type(training) is not int or training < 1 or type(real_count) is not int or real_count != 256):
        raise ValueError("Warmed runtime lacks the exact full training epoch size")
    epoch_updates = math.ceil(training / real_count)
    panels_per_epoch = math.ceil(epoch_updates / plan["protocol"]["early_health"]["interval_updates"]) + 1
    projection = project_warmed_epoch([row["update_seconds"] for row in rows], training, real_count,
        panel_seconds=max(panels) * panels_per_epoch, startup_seconds=receipt["elapsed_seconds"])
    actual = gate.get("warmed_runtime_projection")
    # Require exact producer types as well as values: bool/1 must not substitute
    # for a measured floating wall or an integer number of updates.
    if (type(actual) is not dict or set(actual) != set(projection)
            or any(type(actual[key]) is not type(value) or actual[key] != value for key, value in projection.items())
            or gate.get("mean_update_seconds") != projection["mean_update_seconds"]
            or gate.get("slowest_update_seconds") != projection["slowest_update_seconds"]
            or gate.get("projected_mean_epoch_seconds") != projection["projected_mean_epoch_seconds"]):
        raise ValueError("Warmed runtime projection differs from all actual update/startup/panel walls")
    native_conservative = native.get("conservative_epoch_seconds")
    if (not _positive_seconds(native_conservative)
            or gate.get("conservative_epoch_seconds") != max(native_conservative, projection["conservative_epoch_seconds"])):
        raise ValueError("Warmed runtime discarded the native or observed conservative estimate")


def authenticate_stability(plan):
    """Authenticate a full, same-source disposable proof; never promote its z/weights."""
    if plan.get("valid_reference_adoption") is not None:
        from .chemistry_valid_reference_adoption import authenticate_adoption
        return authenticate_adoption(plan, stability=True)
    policy = validate_guarded_restart(plan)
    if policy is None:
        return None
    root = Path(plan["campaign"]).resolve(strict=True)
    native = _read(root / "failure_readiness.json")
    path = root / "stability_gate.json"
    if _sha(path) != native.get("stability_gate_sha256"):
        raise ValueError("Longer chemistry stability gate is absent or changed")
    gate = _read(path)
    from .chemistry_short_continuation import validate_short_continuation
    short = validate_short_continuation(plan)
    if short is not None and gate.get("short_continuation") != short:
        raise ValueError("Short stability proof lost its bounded training/test purpose")
    from .chemistry_self_guidance_iteration import validate_self_guidance_iteration
    iteration = validate_self_guidance_iteration(plan)
    if ((iteration is None and "self_guidance_iteration" in gate)
            or (iteration is not None and gate.get("self_guidance_iteration") != iteration)):
        raise ValueError("Stability proof changed its declared self-guidance iteration")
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    weighting_test = validate_target_weighting_test(plan)
    if (gate.get("target_weighting_test") != weighting_test
            or gate.get("chemistry_target_weighting") != plan["protocol"].get("chemistry_target_weighting")):
        raise ValueError("Stability proof changed its declared chemistry target loss")
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    reference_test = validate_valid_reference_test(plan)
    if (gate.get("valid_reference_test") != reference_test
            or gate.get("valid_prior_reference_guidance") != plan["protocol"].get("valid_prior_reference_guidance")):
        raise ValueError("Stability proof changed its frozen valid-reference objective")
    if reference_test is not None:
        before,after = gate.get("valid_reference_anchor_before"),gate.get("valid_reference_anchor_after")
        if not before or before != after or gate.get("valid_reference_anchor_unchanged") is not True:
            raise ValueError("Stability proof lost the unchanged frozen original36 anchor")
        from .chemistry_valid_reference_anchor import validate_anchor_identity
        validate_anchor_identity(before,expected_model_sha=gate.get("starting_model_sha256"))
        if before["model_sha256"] != gate.get("starting_model_sha256"):
            raise ValueError("Stability anchor does not match the original36 starting tensors")
    from .chemistry_optimizer_step import validate_optimizer_step_receipt
    step_control = plan.get("optimizer_step_control")
    benchmark = _read(root / "benchmark_gate.json")
    if reference_test is not None and benchmark.get("valid_reference_anchor_before") != before:
        raise ValueError("Stability anchor differs from the authenticated native original36 anchor")
    if (gate.get("optimizer_step_control") != step_control
            or benchmark.get("optimizer_step_control") != step_control):
        raise ValueError("Stability/native proof changed its optimizer-step policy")
    if step_control is not None:
        if (benchmark.get("native_resume_probe", {}).get("optimizer_step_control") != step_control
                or benchmark.get("native_execution_parity", {}).get("optimizer_step_control") != step_control
                or len(benchmark.get("updates", [])) != plan["protocol"]["benchmark_updates"]):
            raise ValueError("Native optimizer-step policy lacks exact resume/parity/update evidence")
        for row in benchmark["updates"]:
            validate_optimizer_step_receipt(row.get("optimizer_step_control"), step_control)
    checks = gate.get("gates")
    start = plan["starting_checkpoint"]
    expected_schedule = panel_schedule(plan["protocol"]["early_health"], policy["validation_updates"])
    if (gate.get("version") != GATE_VERSION or gate.get("status") != "passed"
            or gate.get("plan_sha256") != plan["_plan_sha256"]
            or gate.get("archive_sha256") != plan["archive_sha256"]
            or gate.get("manifest_sha256") != plan["manifest_sha256"]
            or gate.get("starting_checkpoint_sha256") != start["sha256"]
            or gate.get("starting_epoch") != start["epoch"]
            or gate.get("starting_update") != start["updates"]
            or gate.get("completed_updates") != policy["validation_updates"]
            or gate.get("last_update") != start["updates"] + policy["validation_updates"]
            or gate.get("validation_weights_discarded") is not True
            or gate.get("epoch_complete") is not False
            or type(checks) is not dict or set(checks) != GATES
            or not all(type(v) is bool and v for v in checks.values())
            or gate.get("native_readiness_science_sha256") != readiness_science_sha256(native)
            or gate.get("native_benchmark_sha256") != _sha(root / "benchmark_gate.json")
            or gate.get("early_health_baseline_sha256") != native.get("early_health_baseline_sha256")
            or gate.get("observed_panel_schedule") != expected_schedule):
        raise ValueError("Corrected objective did not pass all declared stability/identity checks")
    baseline = root / "early-health-baseline.json"
    if _sha(baseline) != gate["early_health_baseline_sha256"]:
        raise ValueError("Starting fixed-panel baseline changed")
    run_path = Path(gate["run_dir"]).resolve(strict=True)
    record = _read(run_path / "run.json")
    if (not run_path.is_relative_to(root / "stability")
            or record.get("status") != "completed"
            or record.get("environment", {}).get("job_id") != gate.get("job_id")
            or record.get("source_archive_sha256") != plan["archive_sha256"]
            or record.get("source", {}).get("sha256") != plan["manifest_sha256"]
            or _sha(run_path / "gate.json") != _sha(path)):
        raise ValueError("Stability proof is not bound to its completed allocated run")
    panels = gate.get("panels")
    if type(panels) is not list or [r.get("relative_update") for r in panels] != expected_schedule:
        raise ValueError("Longer stability proof has missing or duplicate panel records")
    for panel in panels:
        target = Path(panel["path"]).resolve(strict=True)
        if not target.is_relative_to(run_path) or _sha(target) != panel["sha256"]:
            raise ValueError("Saved early fixed panel escaped its run or changed")
        evidence = _read(target)
        if (evidence.get("status") != "passed" or not evidence.get("checks")
                or not all(type(v) is bool and v for v in evidence["checks"].values())
                or evidence.get("plan_sha256") != plan["_plan_sha256"]
                or evidence.get("archive_sha256") != plan["archive_sha256"]
                or evidence.get("manifest_sha256") != plan["manifest_sha256"]
                or evidence.get("starting_checkpoint_sha256") != start["sha256"]
                or evidence.get("job_id") != gate.get("job_id")
                or evidence.get("baseline_sha256") != gate["early_health_baseline_sha256"]
                or evidence.get("updates") != start["updates"] + panel["relative_update"]):
            raise ValueError("A longer-proof fixed panel did not pass")
    updates_path = Path(gate.get("updates_path", "")).resolve(strict=True)
    if not updates_path.is_relative_to(run_path) or _sha(updates_path) != gate.get("updates_sha256"):
        raise ValueError("Disposable-update evidence escaped its run or changed")
    rows = [_read_line(line) for line in updates_path.read_text().splitlines() if line.strip()]
    if len(rows) != policy["validation_updates"]:
        raise ValueError("Disposable-update proof is partial")
    for relative, row in enumerate(rows, 1):
        if reference_test is not None:
            from .chemistry_valid_reference_validation import require_reference_update
            require_reference_update(plan,row,before)
        if (row.get("relative_update") != relative or row.get("update") != start["updates"] + relative
                or row.get("real_points") != 256 or row.get("fresh_prior_points") != 512
                or any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key])
                       for key in ("total", "reconstruction", "repair_real", "repair_prior", "update_seconds"))
                or row["update_seconds"] <= 0):
            raise ValueError("Disposable-update state, ratio or finite evidence differs")
        validate_optimizer_step_receipt(row.get("optimizer_step_control"), step_control)
        if short is not None:
            gradients = row.get("group_gradient_norms", {})
            statuses = row.get("repair_statuses", {})
            classified = row.get("teacher_classification", {})
            if (not _positive_seconds(row.get("pre_clip_gradient_norm"),zero=True)
                    or type(gradients) is not dict or type(statuses) is not dict or type(classified) is not dict
                    or any(not _positive_seconds(gradients.get(key)) for key in ("encoder","decoder"))
                    or any(type(statuses.get(branch)) is not dict
                           or any(type(count) is not int or count < 0 for count in statuses[branch].values())
                           or sum(statuses[branch].values()) != count
                           or type(statuses[branch].get("rejected_or_error",0)) is not int
                           or statuses[branch].get("rejected_or_error",0) != 0
                           for branch,count in (("real",256),("prior",512)))
                    or any(type(classified.get(branch)) is not dict
                           or type(classified[branch].get("attempts")) is not int
                           or classified[branch]["attempts"] != count
                           for branch,count in (("real",256),("prior",512)))):
                raise ValueError("Short stability update lost finite gradients/error-free classified coverage")
    # Only the explicit new startup protocol changes this metadata contract.
    # Its final estimate still passes through the original epoch-fit guard.
    if plan.get("worker_startup") is not None:
        _authenticate_worker_startup(plan, gate, run_path, rows, native, expected_schedule)
    seconds = gate.get("conservative_epoch_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Stability proof lacks a finite complete-epoch runtime estimate")
    from .chemistry_runtime import epoch_fit_seconds
    if epoch_fit_seconds(plan, seconds) >= plan["protocol"]["segment_wall_seconds"]:
        raise ValueError("Longer observed chemistry runtime does not safely fit a full epoch")
    return gate


def _read_line(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
