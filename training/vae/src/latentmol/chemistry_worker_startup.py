"""Explicit one-time CPU-worker startup, separate from recurring update cost.

Only authenticated real training graphs cross this boundary. No decoder call,
latent sample, target search, optimizer step or target-cache insertion occurs.
The existing bounded worker pool/scheduling/error contract remains in use.
All numerical execution and fixtures belong on allocated DTU HPC.
"""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import multiprocessing
from numbers import Integral
import os
from pathlib import Path
import time
import re

VERSION = "chemistry-worker-startup-v1"
NATIVE_MODULES = ("latentmol.atom_preserving_fast", "latentmol.repair_progress",
                  "latentmol.chemistry_classification")
_STARTUP_PROOF = None


def validate_worker_startup(value):
    """Metadata-only explicit policy; omitted plans retain their historical path."""
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"version"} or value["version"] != VERSION:
        raise ValueError("Unsupported explicit chemistry worker startup policy")
    return dict(value)


def _initialize_warm_worker(barrier):
    """Every allocated worker imports all native task modules before any task.

    The initializer barrier forces all pool processes to finish their imports;
    a few fast workers cannot consume all tiny warmup tasks before others start.
    Spawn transfers this barrier through initializer arguments, never a task.
    """
    global _STARTUP_PROOF
    _STARTUP_PROOF = None
    for name in NATIVE_MODULES:
        importlib.import_module(name)
    barrier.wait(timeout=180.)
    _STARTUP_PROOF = dict(version=VERSION, pid=os.getpid(), modules=list(NATIVE_MODULES),
                          initialized_workers=barrier.parties, initializer_barrier_passed=True)


def _warm_task(index, task):
    from .repair_worker_pool import _run_classification_task
    if not isinstance(_STARTUP_PROOF, dict) or not _STARTUP_PROOF.get("initializer_barrier_passed"):
        raise RuntimeError("Native worker did not complete explicit startup")
    receipt = _run_classification_task(index, task)
    receipt["result"]["startup"] = dict(_STARTUP_PROOF)
    return receipt


def _install_startup_executor(pool):
    """Configure only a pristine existing pool; never replace running workers."""
    if pool.started:
        raise ValueError("Explicit startup requires the pristine cold repair pool")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(pool.workers)
    # Retain the original pool's executor ownership/thread initializer through
    # its explicit public startup hook; never replace private executor state.
    pool.start(after_initialize=_initialize_warm_worker, initargs=(barrier,))
    return barrier


def _positive(value, name, *, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)):
        raise ValueError(name + " requires finite " + ("nonnegative" if zero else "positive") + " seconds")
    return float(value)


def project_warmed_epoch(update_walls, training_molecules, real_batch_size, *,
                         panel_seconds=0., startup_seconds=0., base_overhead_seconds=600.):
    """Charge observed startup once; retain every recurring slow update.

    The unchanged outer epoch-fit factor/reserve applies to these estimates.
    Nothing is trimmed from measured update walls, and no mean substitutes for
    their maximum. Call only after explicit startup completed before timing.
    """
    if (type(training_molecules) is not int or training_molecules < 1
            or type(real_batch_size) is not int or real_batch_size < 1):
        raise ValueError("Epoch projection requires positive integer data/batch counts")
    walls = [_positive(value, "update wall") for value in update_walls]
    if not walls:
        raise ValueError("Epoch projection needs actual timed updates")
    panels = _positive(panel_seconds, "fixed-panel overhead", zero=True)
    startup = _positive(startup_seconds, "one-time startup", zero=True)
    base = _positive(base_overhead_seconds, "base overhead", zero=True)
    count = math.ceil(training_molecules / real_batch_size)
    overhead = panels + startup + base
    mean = sum(walls) / len(walls) * count + overhead
    conservative = max(walls) * count + overhead
    if not math.isfinite(mean) or not math.isfinite(conservative):
        raise ValueError("Epoch projection overflow")
    return dict(version=VERSION, epoch_updates=count, mean_update_seconds=sum(walls)/len(walls),
        slowest_update_seconds=max(walls), one_time_startup_seconds=startup,
        fixed_panel_seconds=panels, base_overhead_seconds=base,
        projected_mean_epoch_seconds=mean, conservative_epoch_seconds=conservative,
        scope="Measured warm recurring updates plus one actual startup and declared evaluation overhead; outer runtime policy unchanged")


def warm_repair_workers(plan, teacher, model, reference, optimizer, data, cohorts, generators, output):
    """Warm the current cold12 teacher and preserve every scientific state.

    Invoke after setup and before timed stability/epoch updates. Keep the
    allocation start clock running; this measured startup is not free time.
    The caller must retain the passing plan-bound receipt and include its
    elapsed_seconds once in epoch projection/one-time allocation overhead.
    """
    from . import chemistry_continuation as continuation
    from .chemistry_classification import VERSION as CLASSIFIER_VERSION, classification_summary
    from .chemistry_execution_validation import _restore, _snapshot, _tensor_summary
    from .data import collate_graphs
    from .experiments import write_json
    from .quality import graph_identity
    from .repair_worker_pool import RepairWorkerPool, _validate_classification_task
    from .train import state_digest

    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior worker-startup receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    report = dict(version=VERSION, status="running", passed=False,
        utc=datetime.now(timezone.utc).isoformat(), job_id=os.environ.get("LSB_JOBID"),
        plan_sha256=plan.get("_plan_sha256"), archive_sha256=plan.get("archive_sha256"),
        manifest_sha256=plan.get("manifest_sha256"), dataset_sha256=plan.get("dataset_sha256"),
        starting_checkpoint_sha256=plan.get("starting_checkpoint", {}).get("sha256", plan.get("parent_checkpoint_sha256")),
        model_forwards=0, optimizer_steps=0, random_samples=0, target_searches=0,
        target_cache_entries=0, modules=list(NATIVE_MODULES), restoration={"passed": False})
    write_json(output, report)
    initial, pool, active_error = None, None, None
    try:
        if validate_worker_startup(plan.get("worker_startup")) is None:
            raise ValueError("Worker startup requires an explicit immutable policy")
        if (not str(report["job_id"] or "").isdecimal() or any(
                not isinstance(report[key], str) or re.fullmatch(r"[0-9a-f]{64}", report[key]) is None
                for key in ("plan_sha256", "archive_sha256", "manifest_sha256", "dataset_sha256",
                            "starting_checkpoint_sha256"))):
            raise ValueError("Worker startup needs allocated job and full immutable plan/source/data/checkpoint identities")
        execution = plan.get("execution", {})
        if (execution.get("backend") != "optimized" or execution.get("workers") != 12
                or execution.get("torch_threads") != 4 or execution.get("gpu_cpus") != 16
                or teacher.backend != "optimized" or teacher.workers != 12
                or teacher.spec != model.spec or reference.spec != model.spec):
            raise ValueError("Worker startup requires the existing optimized12/16-slot teacher")
        initial = _snapshot(model, reference, optimizer, generators)
        report["model_sha256"] = state_digest(initial["model"])
        teacher.drain()
        if teacher._cache or (teacher._pool is not None and teacher._pool.started):
            raise ValueError("Worker startup cannot discard cache entries or replace an existing executor")
        rows = list(cohorts["train"][:256])
        if (len(rows) != 256 or len(set(rows)) != 256
                or any(isinstance(row, bool) or not isinstance(row, Integral) or not 0 <= row < len(data) for row in rows)):
            raise ValueError("Worker startup requires 256 unique authenticated real training rows")
        rows = [int(row) for row in rows]
        hard = collate_graphs([data[row] for row in rows])
        raw = tuple(value.detach().cpu() for value in (hard.atoms, hard.charges, hard.bonds))
        tasks = [(tuple(value[row].numpy().copy() for value in raw), model.spec) for row in range(256)]
        if teacher._pool is None:
            teacher._pool = RepairWorkerPool(12)
        pool = teacher._pool
        barrier = _install_startup_executor(pool)
        tick = time.perf_counter()
        results = pool._map(tasks, _validate_classification_task, _warm_task)
        worker_wall = time.perf_counter() - tick
        if len(results) != 256 or barrier.broken or barrier.n_waiting:
            raise ValueError("Worker startup did not complete all rows/initializers")
        pids, classifications = set(), []
        for index, item in enumerate(results):
            proof = item.get("startup", {})
            if (set(proof) != {"version", "pid", "modules", "initialized_workers", "initializer_barrier_passed"}
                    or proof["version"] != VERSION or type(proof["pid"]) is not int or proof["pid"] <= 0
                    or proof["modules"] != list(NATIVE_MODULES) or proof["initialized_workers"] != 12
                    or proof["initializer_barrier_passed"] is not True
                    or item.get("raw_sha256") != graph_identity(*tasks[index][0], model.spec)
                    or item.get("schema_sha256") != model.spec.fingerprint()
                    or item.get("classification", {}).get("version") != CLASSIFIER_VERSION):
                raise ValueError("Worker startup lost input/schema/initializer provenance")
            pids.add(proof["pid"])
            classifications.append({"classification": item["classification"]})
        if teacher._cache:
            raise ValueError("Warmup unexpectedly inserted scientific targets")
        report.update(status="passed", passed=True, warmup_real_points=256, initialized_workers=12,
            worker_initialization_barrier_passed=True, observed_task_pids=sorted(pids),
            worker_start_and_classification_seconds=worker_wall,
            real_row_ids_sha256=hashlib.sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest(),
            hard_graphs=_tensor_summary(dict(atoms=hard.atoms, charges=hard.charges, bonds=hard.bonds)),
            classification=classification_summary(classifications,max_examples_per_failure=0))
    except BaseException as exc:
        active_error = exc
        report.update(status="failed", passed=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        errors = []
        if not report["passed"] and pool is not None:
            try:
                pool.close()
            except Exception as exc:
                errors.append("worker_close: " + str(exc))
        if initial is not None:
            try:
                _restore(initial, model, reference, optimizer, generators)
                restored = _snapshot(model, reference, optimizer, generators)
                checks = {key: continuation.nested_equal(initial[key], restored[key]) for key in initial}
                report["restoration"] = dict(passed=all(checks.values()),checks=checks)
                if not all(checks.values()):errors.append("Exact startup state restoration failed")
            except Exception as exc:
                errors.append("state_restore: " + str(exc))
        if errors:
            report.update(status="failed",passed=False,cleanup_errors=errors)
            if pool is not None:
                try:
                    pool.close()
                except Exception as exc:
                    errors.append("worker_close_after_restore: " + str(exc))
        report["elapsed_seconds"] = time.perf_counter() - start
        write_json(output, report)
        if errors:raise RuntimeError("Worker startup cleanup failed: " + "; ".join(errors)) from active_error
    return report
