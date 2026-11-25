"""Disposable exact full-update parity for faster bond-only repair execution.

Both executions consume the same real rows, posterior/prior random draws,
frozen reference, loss, search budgets and optimizer lineage. Only the repair
execution backend changes. Timing fields are observations, never parity inputs;
targets, search decisions, receipt order, losses and complete gradients remain
mandatory exact comparisons. The original parent, existing gradients, modes and
every RNG stream are restored even after failure. Neither update is training.
All numerical execution belongs on allocated DTU HPC.
"""
from copy import deepcopy
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re
import time

import torch

from .atom_preserving_chemistry import AtomPreservingRepairTeacher
from .chemistry_runtime import EXECUTION_WORKERS, composition_prior_weight
from .data import collate_graphs
from .experiments import write_json
from .train import capture_rng_state, restore_rng_state


VERSION = "exact-chemistry-execution-full-update-v1"
EXCLUDED_TIMINGS = ("search_seconds", "prepass_seconds", "teacher_phase_seconds")


def _gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()}


def _snapshot(model, reference, optimizer, generators):
    return {"model": deepcopy(model.state_dict()), "reference": deepcopy(reference.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()), "gradients": _gradients(model),
            "reference_gradients": _gradients(reference),
            "model_modes": {name: module.training for name, module in model.named_modules()},
            "reference_modes": {name: module.training for name, module in reference.named_modules()},
            "rng": capture_rng_state(generators, include_cuda=True)}


def _restore(state, model, reference, optimizer, generators):
    model.load_state_dict(state["model"], strict=True)
    reference.load_state_dict(state["reference"], strict=True)
    # Optimizer loading may reuse same-device tensor storage from its argument.
    # A subsequent AdamW step would then mutate the supposedly pristine saved
    # moments/counters, corrupting both the second arm's start and restoration.
    # Give each live optimizer its own copy; never hand it the retained snapshot.
    optimizer.load_state_dict(deepcopy(state["optimizer"]))
    for key, module in (("gradients", model), ("reference_gradients", reference)):
        for name, parameter in module.named_parameters():
            value = state[key][name]
            parameter.grad = None if value is None else value.to(parameter).clone()
    for key, module in (("model_modes", model), ("reference_modes", reference)):
        for name, child in module.named_modules():
            child.training = state[key][name]
    # Setup/restoration may allocate tensors. Restore all streams last, before
    # any following posterior epsilon, prior point or DataLoader iterator.
    restore_rng_state(state["rng"], generators)


def _tensor_summary(values):
    """Bind all named gradients without exporting millions of entries."""
    digest, count, missing = hashlib.sha256(), 0, []
    for name, value in sorted(values.items()):
        if value is None:
            digest.update((name + "|None").encode())
            missing.append(name)
        else:
            raw = value.detach().cpu().contiguous()
            digest.update(f"{name}|{raw.dtype}|{tuple(raw.shape)}|".encode())
            digest.update(raw.numpy().tobytes())
            count += 1
    return {"sha256": digest.hexdigest(), "named_entries": len(values),
            "nonmissing_entries": count, "missing_names": missing}


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(part) for part in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(part) for part in value)
    return not isinstance(value, float) or math.isfinite(value)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def native_execution_parity(model, reference, optimizer, data, cohorts, config, protocol,
                            composition, optimized_teacher, generators, epoch, updates, output, *, overlap=False,
                            phase_start_epoch=None, step_control=None):
    """Compare the same full logical update, then discard both executions.

    The optimized pool may remain warm, but its target cache is cleared before
    and after the disposable comparison. A legacy serial teacher is constructed
    with the same search configuration and cache capacity, then always closed.
    The caller must retain this passing receipt before timing native throughput.
    """
    # The continuation module imports this helper only at the benchmark boundary.
    # Resolve its exact current update/comparator lazily to avoid an import cycle.
    from . import chemistry_continuation as continuation

    output = Path(output)
    if output.exists():
        raise FileExistsError("Retain the prior execution parity result; use a new immutable output")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "status": "running", "passed": False,
              "disposable_updates": True, "training_updates_promoted": 0,
              "excluded_metric_fields": list(EXCLUDED_TIMINGS), "checks": {},
              "execution": {}, "restoration": {"passed": False}}
    if step_control is not None:
        from .chemistry_optimizer_step import validate_optimizer_step_control
        step_control = validate_optimizer_step_control(step_control)
        report["optimizer_step_control"] = dict(step_control)
    write_json(output, report)
    initial, reference_teacher, active_error = None, None, None
    try:
        initial = _snapshot(model, reference, optimizer, generators)
        device = next(model.parameters()).device
        prior_weight = composition_prior_weight(protocol)
        prior_count = 256 * protocol["prior_ratio"]
        if (config.batch_size != 256 or protocol.get("real_batch_size") != 256
                or protocol.get("microbatch_size") not in (128, 256) or torch.get_num_threads() != 4
                or optimized_teacher.backend != "optimized" or type(optimized_teacher.workers) is not int
                or optimized_teacher.workers not in EXECUTION_WORKERS
                or type(overlap) is not bool
                or optimized_teacher.spec != model.spec
                or reference.spec != model.spec or any(parameter.requires_grad for parameter in reference.parameters())):
            raise ValueError("Full execution parity requires batch256, declared ratio/microbatch, Torch4 and supported optimized workers")
        rows = list(cohorts["train"][:256])
        if (len(rows) != 256 or any(isinstance(row, bool) or not isinstance(row, Integral)
                                   or not 0 <= row < len(data) for row in rows)
                or len(set(rows)) != 256):
            raise ValueError("Execution parity requires the first 256 unique authenticated training rows")
        hard = collate_graphs([data[int(row)] for row in rows])
        if protocol.get("failure_guidance") is None:
            if protocol.get("baseline_continuation") is not None:
                from .failure_guidance_training import build_repair_teacher
                reference_teacher = build_repair_teacher(model.spec, protocol,
                    {"backend": "reference", "workers": 0}, cache_capacity=optimized_teacher.cache_capacity)
            else:
                reference_teacher = AtomPreservingRepairTeacher(model.spec, optimized_teacher.config,
                    cache_capacity=optimized_teacher.cache_capacity, backend="reference", workers=0)
        else:
            from .failure_guidance_training import FailureAwareRepairTeacher
            reference_teacher = FailureAwareRepairTeacher(model.spec, optimized_teacher.config,
                failure_guidance=protocol["failure_guidance"],
                cache_capacity=optimized_teacher.cache_capacity, backend="reference", workers=0)
        optimized_teacher.drain()
        optimized_teacher.clear_cache()
        report["execution"] = {"reference_backend": "reference", "reference_workers": 0,
            "optimized_backend": "optimized", "optimized_workers": optimized_teacher.workers,
            "main_torch_threads": torch.get_num_threads(), "device": str(device),
            "parameter_dtype": str(next(model.parameters()).dtype),
            "real_points": 256, "fresh_prior_points": prior_count, "epoch": epoch + 1,
            "prior_ratio": protocol["prior_ratio"], "composition_prior_weight": prior_weight,
            "microbatch_size": protocol["microbatch_size"], "optimized_overlap": overlap,
            "absolute_update": updates, "training_rows_sha256": hashlib.sha256(
                json.dumps([int(row) for row in rows], separators=(",", ":")).encode()).hexdigest()}
        write_json(output, report)
        observed = {}
        for name, teacher in (("reference", reference_teacher), ("optimized", optimized_teacher)):
            teacher.drain()
            _restore(initial, model, reference, optimizer, generators)
            start_state = _snapshot(model, reference, optimizer, generators)
            if not all(continuation.nested_equal(initial[key], start_state[key]) for key in initial):
                raise ValueError("Disposable backend did not start from the exact preserved parent")
            del start_state
            teacher.clear_cache()
            _sync(device)
            started = time.perf_counter()
            metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol,
                composition, teacher, generators, updates, epoch + 1,
                epoch if phase_start_epoch is None else phase_start_epoch,
                overlap=overlap if name == "optimized" else False,
                **({"step_control": step_control} if step_control is not None else {}))
            if step_control is not None:
                from .chemistry_optimizer_step import validate_optimizer_step_receipt
                validate_optimizer_step_receipt(metrics.get("optimizer_step_control"), step_control)
            teacher.drain()
            _sync(device)
            seconds = time.perf_counter() - started
            if not isinstance(metrics, dict) or not _finite(metrics) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Disposable update returned nonfinite metrics or timing")
            observed[name] = {"metrics": deepcopy(metrics), "state": _snapshot(model, reference, optimizer, generators)}
            report[name] = {"seconds": seconds, "metrics": metrics,
                "named_gradients": _tensor_summary(observed[name]["state"]["gradients"]),
                "reference_named_gradients": _tensor_summary(observed[name]["state"]["reference_gradients"])}
            write_json(output, report)
        left, right = observed["reference"], observed["optimized"]
        semantic = lambda value: {key: part for key, part in value.items() if key not in EXCLUDED_TIMINGS}
        chain = left["metrics"].get("ordered_teacher_receipt_chain_sha256")
        checks = {"full_logical_batch": all(value["metrics"].get("real_points") == 256
                    and value["metrics"].get("fresh_prior_points") == prior_count for value in observed.values()),
            "metrics_exact": continuation.nested_equal(semantic(left["metrics"]), semantic(right["metrics"])),
            "ordered_teacher_receipt_chain_exact": isinstance(chain, str) and re.fullmatch(r"[0-9a-f]{64}", chain) is not None
                and chain == right["metrics"].get("ordered_teacher_receipt_chain_sha256")}
        checks.update({key + "_exact": continuation.nested_equal(left["state"][key], right["state"][key])
            for key in ("gradients", "reference_gradients", "model", "reference", "optimizer", "rng", "model_modes", "reference_modes")})
        checks["reference_unchanged"] = all(continuation.nested_equal(value["state"][key], initial[key])
            for value in observed.values() for key in ("reference", "reference_gradients", "reference_modes"))
        def accepted(value):
            targets = value["metrics"].get("repair_targets")
            return (isinstance(targets, dict) and set(targets) == {"real", "prior"}
                    and all(type(count) is int and count >= 0 for count in targets.values())
                    and sum(targets.values()) > 0)
        checks["repair_target_observed"] = all(accepted(value) for value in observed.values())
        report.update(checks=checks, speedup=report["reference"]["seconds"] / report["optimized"]["seconds"],
            mismatched_metric_fields=sorted(key for key in set(semantic(left["metrics"])) | set(semantic(right["metrics"]))
                if key not in left["metrics"] or key not in right["metrics"]
                or not continuation.nested_equal(left["metrics"][key], right["metrics"][key])),
            mismatched_gradient_names=sorted(key for key in left["state"]["gradients"]
                if not continuation.nested_equal(left["state"]["gradients"][key], right["state"]["gradients"][key])))
        if not all(checks.values()):
            raise ValueError("Exact full-update repair execution parity failed: " + "; ".join(key for key, passed in checks.items() if not passed))
        report.update(status="passed", passed=True)
    except BaseException as error:
        active_error = error
        report.update(status="failed", passed=False, error=str(error), error_type=type(error).__name__)
        raise
    finally:
        cleanup_errors = []
        for name, action in (("reference_teacher_close", lambda: reference_teacher.close() if reference_teacher is not None else None),
                             ("optimized_drain", optimized_teacher.drain),
                             ("optimized_cache_clear", optimized_teacher.clear_cache)):
            try:
                action()
            except Exception as error:
                cleanup_errors.append(name + ": " + type(error).__name__ + ": " + str(error))
        if initial is not None:
            try:
                _restore(initial, model, reference, optimizer, generators)
                restored = _snapshot(model, reference, optimizer, generators)
                checks = {key: continuation.nested_equal(value, restored[key]) for key, value in initial.items()}
                report["restoration"] = {"passed": all(checks.values()), "checks": checks,
                    "optimized_cache_cleared": not bool(optimized_teacher._cache),
                    "reference_teacher_closed": reference_teacher is not None}
                if not all(checks.values()) or optimized_teacher._cache:
                    cleanup_errors.append("Exact preserved parent/gradient/RNG restoration or cache clearing failed")
            except Exception as error:
                cleanup_errors.append("parent_restore: " + type(error).__name__ + ": " + str(error))
        if cleanup_errors:
            report.update(status="failed", passed=False, cleanup_errors=cleanup_errors)
        write_json(output, report)
        if cleanup_errors:
            raise RuntimeError("Disposable chemistry execution cleanup failed: " + "; ".join(cleanup_errors)) from active_error
    return report
