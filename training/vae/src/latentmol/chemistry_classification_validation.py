"""Matched disposable updates isolating classification scheduling, not search.

Both arms use optimized search, the same 12 CPU workers, overlap setting and
exact model/AdamW/gradient/RNG state. The classifier's execution location alone
changes. All numerical execution belongs on allocated DTU HPC.
"""
from copy import deepcopy
import math
from pathlib import Path
import re
import time

import torch

from .chemistry_execution_validation import (EXCLUDED_TIMINGS, _finite, _restore, _snapshot,
                                              _sync, _tensor_summary)
from .experiments import write_json
from .failure_guidance_training import build_repair_teacher
from .repair_worker_pool import CLASSIFICATION_EXECUTION_VERSION, RepairWorkerPool


VERSION = "exact-chemistry-classification-execution-v1"


def classification_execution_parity(model, reference, optimizer, hard, config, protocol, composition,
                                    generators, epoch, updates, phase_start_epoch, output, *, overlap=False):
    """Restore every scientific state even when classification, update or cleanup fails.

    Each pool is warmed using native classifications of the supplied real hard
    rows, with no model forward, random sample or optimizer update. Those warmup
    outputs are discarded and separately counted. Pools close sequentially, so
    this helper never opens 24 workers inside a 16-slot allocation. The caller
    must drain and close any existing repair teacher/pool before invoking it;
    caller-owned workers cannot be discovered or closed safely by this helper.
    """
    from . import chemistry_continuation as continuation

    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior classification parity evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(version=VERSION, status="running", passed=False, checks={},
        disposable_updates=True, training_updates_promoted=0,
        excluded_metric_fields=list(EXCLUDED_TIMINGS), restoration={"passed": False},
        timing_scope="Matched full-update wall after native hard-data classifier warmup; worker sums are not critical-path time")
    write_json(output, report)
    initial, active_error, current_teacher = None, None, None
    try:
        initial = _snapshot(model, reference, optimizer, generators)
        if (config.batch_size != 256 or len(hard.atoms) != 256
                or protocol.get("real_batch_size") != 256 or protocol.get("microbatch_size") not in (128, 256)
                or type(overlap) is not bool or torch.get_num_threads() != 4
                or not protocol.get("failure_guidance") or reference.spec != model.spec
                or any(p.requires_grad for p in reference.parameters())):
            raise ValueError("Classification comparison requires the same full native 256-point update and frozen reference")
        device = next(model.parameters()).device
        execution = dict(backend="optimized", workers=12, torch_threads=4, gpu_cpus=16, overlap=overlap)
        pool_policy = {"version": CLASSIFICATION_EXECUTION_VERSION}
        report["execution"] = dict(backend="optimized", workers=12, torch_threads=4, overlap=overlap,
            device=str(device), parameter_dtype=str(next(model.parameters()).dtype),
            real_points=256, fresh_prior_points=256 * protocol["prior_ratio"],
            absolute_update=updates, epoch=epoch + 1, phase_start_epoch=phase_start_epoch,
            hard_batch=_tensor_summary({"atoms": hard.atoms, "charges": hard.charges, "bonds": hard.bonds}))
        raw_cpu = tuple(value.detach().cpu() for value in (hard.atoms, hard.charges, hard.bonds))
        warm_tasks = [(tuple(value[row].numpy().copy() for value in raw_cpu), model.spec) for row in range(256)]
        observed = {}
        for name, policy in (("serial_classification", None), ("pooled_classification", pool_policy)):
            current_teacher = build_repair_teacher(model.spec, protocol, execution,
                                                    classification_execution=policy)
            try:
                if current_teacher._pool is None:
                    current_teacher._pool = RepairWorkerPool(12)
                warm_started = time.perf_counter()
                warm_results = current_teacher._pool.map_classification(warm_tasks)
                if len(warm_results) != 256:
                    raise ValueError("Classification worker warmup lost a supplied hard row")
                warm_seconds = time.perf_counter() - warm_started
                current_teacher.clear_cache()
                _restore(initial, model, reference, optimizer, generators)
                start = _snapshot(model, reference, optimizer, generators)
                if not all(continuation.nested_equal(initial[key], start[key]) for key in initial):
                    raise ValueError("Classification arm did not restore the exact initial state")
                _sync(device)
                tick = time.perf_counter()
                metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol,
                    composition, current_teacher, generators, updates, epoch + 1, phase_start_epoch, overlap=overlap)
                current_teacher.drain()
                _sync(device)
                seconds = time.perf_counter() - tick
                if not isinstance(metrics, dict) or not _finite(metrics) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Classification arm returned nonfinite observations")
                observed[name] = dict(metrics=deepcopy(metrics), state=_snapshot(model, reference, optimizer, generators))
                report[name] = dict(seconds=seconds, metrics=metrics, warmup_hard_points=256,
                    warmup_seconds=warm_seconds, classification_execution=policy,
                    named_gradients=_tensor_summary(observed[name]["state"]["gradients"]))
                write_json(output, report)
            finally:
                current_teacher.close()
                current_teacher = None
        left, right = observed["serial_classification"], observed["pooled_classification"]
        semantic = lambda metrics: {key: value for key, value in metrics.items() if key not in EXCLUDED_TIMINGS}
        chain = left["metrics"].get("ordered_teacher_receipt_chain_sha256")
        checks = dict(full_logical_batch=all(value["metrics"].get("real_points") == 256 and
            value["metrics"].get("fresh_prior_points") == 256 * protocol["prior_ratio"] for value in observed.values()),
            metrics_exact=continuation.nested_equal(semantic(left["metrics"]), semantic(right["metrics"])),
            ordered_teacher_receipt_chain_exact=isinstance(chain, str) and re.fullmatch(r"[0-9a-f]{64}", chain) is not None
                and chain == right["metrics"].get("ordered_teacher_receipt_chain_sha256"))
        checks.update({key + "_exact": continuation.nested_equal(left["state"][key], right["state"][key])
                       for key in initial})
        checks["reference_unchanged"] = all(continuation.nested_equal(value["state"][key], initial[key])
            for value in observed.values() for key in ("reference", "reference_gradients", "reference_modes"))
        report.update(checks=checks, speedup=report["serial_classification"]["seconds"] / report["pooled_classification"]["seconds"])
        if not all(checks.values()):
            raise ValueError("Matched classification execution differs: " + ", ".join(k for k, v in checks.items() if not v))
        report.update(status="passed", passed=True)
    except BaseException as exc:
        active_error = exc
        report.update(status="failed", passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        errors = []
        if current_teacher is not None:
            try:
                current_teacher.close()
            except Exception as exc:
                errors.append("teacher_close: " + str(exc))
        if initial is not None:
            try:
                _restore(initial, model, reference, optimizer, generators)
                restored = _snapshot(model, reference, optimizer, generators)
                checks = {key: continuation.nested_equal(initial[key], restored[key]) for key in initial}
                report["restoration"] = dict(passed=all(checks.values()), checks=checks)
                if not all(checks.values()): errors.append("Exact initial state restoration failed")
            except Exception as exc:
                errors.append("state_restore: " + str(exc))
        if errors: report.update(status="failed", passed=False, cleanup_errors=errors)
        write_json(output, report)
        if errors:
            raise RuntimeError("Classification comparison cleanup failed: " + "; ".join(errors)) from active_error
    return report
