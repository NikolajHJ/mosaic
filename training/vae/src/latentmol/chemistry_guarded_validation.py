"""Allocated native benchmark and a longer, disposable epoch36 stability test.

The same production update and fixed-panel observer serve validation and later
training. Validation uses the real first-epoch shuffled data stream, inherited
AdamW moments and RNG; all 2,500 updates are discarded before a separately
allocated continuation restores epoch36. Failure preserves nonpromotable state.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from . import chemistry_continuation as continuation
from .chemistry_early_health import (exclusive_json, materialize_early_health_baseline,
    save_failed_early_health_snapshot, validate_early_health_policy)
from .chemistry_restart_policy import (GATES, GATE_VERSION, panel_schedule,
    readiness_science_sha256, validate_guarded_restart)
from .chemistry_runtime import epoch_fit_seconds
from .data import collate_graphs
from .experiments import ExperimentRun, file_sha256, write_json
from .failure_guidance_benchmark import benchmark_candidate
from .failure_guidance_selection import authenticate_readiness
from .train import learning_rate_at_epoch, state_digest


def _compact_metrics(value):
    """Keep every scalar/count; bounded raw examples live in fixed-panel files.

    Per-update teacher receipt identities remain recorded. Avoid repeating
    hundreds of millions of graph labels in a long scalar log.
    """
    if isinstance(value, dict):
        return {key: _compact_metrics(item) for key, item in value.items() if key != "failure_examples"}
    if isinstance(value, list):
        return [_compact_metrics(item) for item in value]
    return value


def _observe_failed_generation_safely(plan, model, generators, run, report, *, live_state_observable=True):
    """Save a separate failure-end observation; never replace the primary error/gate.

    A model that failed before the starting observation has no paired endpoint.
    Observation/serialization errors are secondary evidence, not a successful
    scientific result and not a reason to overwrite or mask the original failure.
    """
    if (plan.get("short_continuation") is None or model is None or generators is None
            or "generation_start" not in report):
        return
    path = None
    try:
        path = Path(plan["campaign"]) / "failed-generation-end.json"
        primary = run.path / "gate.json"
        binding = dict(path=str(primary), sha256=file_sha256(primary))
        if not live_state_observable:
            write_json(path, dict(status="unavailable_observation",
                reason="The failed update did not reach its durable completed-step counter; live weights are not bound to the last recorded point",
                failure_gate=binding, completed_disposable_updates=report["completed_updates"],
                last_recorded_update=report["last_update"], model_observation_performed=False,
                promotable=False, primary_failure_preserved=True))
            return
        if report.get("completed_updates", 0) <= 0:
            return
        from .chemistry_generation_comparison import observe_failed_generation
        observed = observe_failed_generation(plan, model, generators, run.path / "generation-failed-end",
            completed_updates=report["completed_updates"], failure_gate=binding)
        write_json(path, dict(status="observed_failed_endpoint", generation=observed,
            failure_gate=binding, completed_disposable_updates=report["completed_updates"], promotable=False))
    except Exception as error:
        secondary = dict(status="failed_observation", error=f"{type(error).__name__}: {error}",
            completed_disposable_updates=report["completed_updates"], promotable=False,
            primary_failure_preserved=True)
        try:
            if path is None:
                raise ValueError("Failed generation sidecar path was unavailable")
            write_json(path, secondary)
        except Exception as save_error:
            # Even exhausted quota during reporting must preserve the original
            # training/health exception. The allocation log retains this note.
            print("Failed generation sidecar could not be saved: " + str(save_error), flush=True)


def run_stability(plan, *, allocation_started=None):
    policy = validate_guarded_restart(plan)
    early = validate_early_health_policy(plan["protocol"])
    root, protocol = Path(plan["campaign"]), plan["protocol"]
    output = root / "stability_gate.json"
    if output.exists():
        raise FileExistsError("Preserve prior stability result; use a new immutable attempt")
    native_path = root / "failure_readiness.json"
    native = json.loads(native_path.read_text())
    native_sha = file_sha256(root / "benchmark_gate.json")
    plan = dict(plan, _benchmark_gate_sha256=native_sha)
    # This authenticates the short native proof; the later stability gate is
    # deliberately not required until it has actually been produced.
    authenticate_readiness(plan, file_sha256(native_path))
    report = dict(version=GATE_VERSION, status="failed", plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        job_id=os.environ.get("LSB_JOBID"), starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        starting_epoch=36, starting_update=plan["starting_checkpoint"]["updates"], completed_updates=0,
        last_update=plan["starting_checkpoint"]["updates"], epoch_complete=False,
        validation_weights_discarded=True, native_benchmark_sha256=native_sha,
        native_readiness_science_sha256=readiness_science_sha256(native), panels=[], observed_panel_schedule=[])
    if plan.get("short_continuation") is not None:
        report["short_continuation"] = dict(plan["short_continuation"])
    if plan.get("self_guidance_iteration") is not None:
        report["self_guidance_iteration"] = dict(plan["self_guidance_iteration"])
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    weighting_test = validate_target_weighting_test(plan)
    if weighting_test is not None:
        report["target_weighting_test"] = weighting_test
        report["chemistry_target_weighting"] = dict(plan["protocol"]["chemistry_target_weighting"])
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    reference_test = validate_valid_reference_test(plan)
    if reference_test is not None:
        report["valid_reference_test"] = reference_test
        report["valid_prior_reference_guidance"] = dict(plan["protocol"]["valid_prior_reference_guidance"])
    if plan.get("optimizer_step_control") is not None:
        report["optimizer_step_control"] = dict(plan["optimizer_step_control"])
    phase_started = time.monotonic()
    started = phase_started if allocation_started is None else allocation_started
    if (type(started) not in (int, float) or not math.isfinite(started)
            or not 0 <= started <= phase_started):
        raise ValueError("Invalid shared native/stability allocation clock")
    # Native readiness and the stability proof share one allocation. The
    # internal budget must not start over after the short native benchmark.
    report["allocation_seconds_before_stability"] = phase_started - started
    model = optimizer = config = generators = run = None
    live_state_observable = False
    with ExitStack() as resources, ExperimentRun(root / "stability", "guarded-chemistry-stability",
                                                plan, Path(plan["release"])) as run:
        report["run_dir"] = str(run.path)
        try:
            parent_proof = {}
            model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, _, _ = continuation.setup(
                plan, parent_authentication=parent_proof)
            resources.callback(teacher.close)
            if reference_test is not None:
                report["valid_reference_anchor_before"] = teacher.valid_reference_anchor.assert_unchanged()
            if epoch != 36 or updates != report["starting_update"]:
                raise ValueError("Stability test did not restore the exact completed epoch36")
            checkpoint = torch.load(plan["starting_checkpoint"]["path"], map_location="cpu", weights_only=True)
            model_sha = state_digest(model.state_dict())
            if model_sha != state_digest(checkpoint["model_state"]):
                raise ValueError("Restored model differs from the authenticated epoch36 tensor payload")
            del checkpoint
            native_benchmark = json.loads((root / "benchmark_gate.json").read_text())
            continuation.authenticated(root / "training_composition.json", native_benchmark["composition_sha256"])
            composition = json.loads((root / "training_composition.json").read_text())
            initial = continuation.observe_early_health(model, data, config, protocol, composition, generators)
            baseline, baseline_sha = materialize_early_health_baseline(plan, initial,
                completed_epoch=epoch, updates=updates, model_sha256=model_sha)
            # Bind the classified starting panel without changing any original
            # native gate, scientific observation or historical receipt.
            native["early_health_baseline_sha256"] = baseline_sha
            write_json(native_path, native)
            plan["_early_health_baseline_sha256"] = baseline_sha
            plan["_failure_readiness_sha256"] = file_sha256(native_path)
            report["early_health_baseline_sha256"] = baseline_sha
            reference_observation = continuation.observe_early_health(reference, data, config, protocol, composition, generators)
            initial_assessment = continuation.assess_early_health(initial,
                reference_observation["population"], reference_observation["reconstruction"],
                initial["population"], initial["reconstruction"], protocol,
                starting_classification=initial["classification"])
            exclusive_json(run.path / "starting-observation.json", dict(observation=initial,
                reference_observation=reference_observation, assessment=initial_assessment,
                source_checkpoint_sha256=plan["starting_checkpoint"]["sha256"], model_sha256=model_sha))
            if initial_assessment["status"] != "passed":
                raise RuntimeError("The authenticated starting checkpoint fails the unchanged health limits")
            if plan.get("short_continuation") is not None:
                from .chemistry_generation_comparison import observe_generation
                report["generation_start"] = observe_generation(plan, model, generators,
                    run.path / "generation-start", completed_updates=0)
            live_state_observable = True
            startup = None
            if plan.get("worker_startup") is not None:
                from .chemistry_worker_startup import warm_repair_workers
                startup_path = run.path / "worker-startup.json"
                startup = warm_repair_workers(plan, teacher, model, reference, optimizer,
                    data["train"], cohorts, generators, startup_path)
                report["worker_startup"] = dict(path=str(startup_path), sha256=file_sha256(startup_path),
                    elapsed_seconds=startup["elapsed_seconds"], status=startup["status"])
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_epoch(config, epoch + 1)
            # This is the same shuffled epoch36->37 stream as production.
            loader = DataLoader(Subset(data["train"], cohorts["train"]), batch_size=config.batch_size,
                shuffle=True, generator=generators["train"], collate_fn=collate_graphs, num_workers=0)
            schedule = panel_schedule(early, policy["validation_updates"])
            records, walls, panel_walls = [], [], []
            overlap = bool(continuation.execution_settings(plan).get("overlap", False))
            torch.cuda.reset_peak_memory_stats()
            batches = iter(loader)
            log_path = run.path / "stability-updates.jsonl"
            with log_path.open("x") as stream:
                for relative in range(1, policy["validation_updates"] + 1):
                    if time.monotonic() > started + policy["validation_wall_seconds"] - 900:
                        raise TimeoutError("Longer stability test exhausted its 23-hour internal budget")
                    torch.cuda.synchronize()
                    tick = time.monotonic()
                    hard = next(batches)
                    live_state_observable = False
                    metrics = continuation.chemistry_update(model, reference, optimizer, hard, config, protocol,
                        composition, teacher, generators, updates, epoch + 1, plan["parent_epoch"],
                        deadline=started + policy["validation_wall_seconds"] - 900, overlap=overlap,
                        **({"step_control": plan["optimizer_step_control"]}
                           if plan.get("optimizer_step_control") is not None else {}))
                    torch.cuda.synchronize()
                    wall = time.monotonic() - tick
                    updates += 1
                    metrics = _compact_metrics(metrics)
                    metrics.update(update=updates, relative_update=relative, update_seconds=wall,
                                   failure_examples_scope="Raw bounded examples are saved by the fixed-panel observer")
                    stream.write(json.dumps(metrics, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(metrics)
                    walls.append(wall)
                    report.update(completed_updates=relative, last_update=updates)
                    live_state_observable = True
                    if relative == 1 or relative % 100 == 0:
                        run.log(dict(event="disposable_validation_update", **metrics))
                    if relative in schedule:
                        tick = time.monotonic()
                        continuation.run_early_health_check(plan, run, model, optimizer, config, data, composition, generators,
                            completed_epoch=epoch, epoch=epoch + 1, updates=updates,
                            reference_population=reference_observation["population"],
                            reference_reconstruction=reference_observation["reconstruction"],
                            starting_population=initial["population"], starting_reconstruction=initial["reconstruction"],
                            baseline=baseline)
                        panel_walls.append(time.monotonic() - tick)
                        path = run.path / "early_health" / f"update_{updates:09d}.json"
                        report["panels"].append(dict(relative_update=relative, path=str(path), sha256=file_sha256(path)))
                        report["observed_panel_schedule"].append(relative)
                        write_json(root / "stability-progress.json", dict(report,
                            status="running_not_passed", elapsed_seconds=time.monotonic()-started))
            if plan.get("short_continuation") is not None:
                report["generation_end"] = observe_generation(plan, model, generators,
                    run.path / "generation-end", completed_updates=policy["validation_updates"])
            epoch_updates = math.ceil(plan["training_molecules"] / protocol["real_batch_size"])
            panels_per_epoch = math.ceil(epoch_updates / early["interval_updates"]) + 1
            panel_seconds = max(panel_walls) * panels_per_epoch
            conservative = max(native["conservative_epoch_seconds"], max(walls) * epoch_updates + panel_seconds + 600.)
            projection = None
            if startup is not None:
                from .chemistry_worker_startup import project_warmed_epoch
                projection = project_warmed_epoch(walls, plan["training_molecules"], protocol["real_batch_size"],
                    panel_seconds=panel_seconds, startup_seconds=startup["elapsed_seconds"])
                conservative = max(native["conservative_epoch_seconds"], projection["conservative_epoch_seconds"])
            peak = torch.cuda.max_memory_reserved()
            finite = all(math.isfinite(row[key]) for row in records
                         for key in ("total", "reconstruction", "repair_real", "repair_prior", "pre_clip_gradient_norm", "update_seconds"))
            classified = all(row.get("teacher_classification", {}).get(branch, {}).get("attempts") == row[count_key]
                for row in records for branch, count_key in (("real", "real_points"), ("prior", "fresh_prior_points")))
            gates = dict(all_native_readiness=native["status"] == "passed" and all(native["gates"].values()),
                exact_starting_checkpoint=bool(parent_proof.get("objective_change_handoff", {}).get("passed")),
                full_update_budget=len(records) == policy["validation_updates"],
                all_early_panels_passed=len(report["panels"]) == len(schedule),
                exact_panel_schedule=report["observed_panel_schedule"] == schedule,
                finite_updates=finite,
                exact_sampling_ratio=all(r["real_points"] == 256 and r["fresh_prior_points"] == 512 for r in records),
                no_search_errors=all(not r["repair_statuses"][b].get("rejected_or_error", 0) for r in records for b in ("real", "prior")),
                encoder_decoder_gradients=all(r["group_gradient_norms"][k] > 0 for r in records for k in ("encoder", "decoder")),
                memory=peak < protocol["vram_fraction_limit"] * torch.cuda.get_device_properties(0).total_memory,
                complete_epoch_fits=epoch_fit_seconds(plan, conservative) < protocol["segment_wall_seconds"],
                validation_weights_discarded=True, classified_failure_counts=classified)
            if set(gates) != GATES:
                raise ValueError("Declared stability gate schema changed")
            if reference_test is not None:
                report["valid_reference_anchor_after"] = teacher.valid_reference_anchor.assert_unchanged()
                report["valid_reference_anchor_unchanged"] = report["valid_reference_anchor_before"] == report["valid_reference_anchor_after"]
                if not report["valid_reference_anchor_unchanged"]:
                    raise ValueError("Original36 reference anchor changed during disposable stability proof")
            report.update(status="passed" if all(gates.values()) else "failed", gates=gates,
                conservative_epoch_seconds=conservative, mean_update_seconds=sum(walls)/len(walls),
                slowest_update_seconds=max(walls), fixed_panel_seconds=panel_walls,
                projected_mean_epoch_seconds=sum(walls)/len(walls)*epoch_updates + panel_seconds + 600.,
                peak_reserved_vram_bytes=peak, elapsed_seconds=time.monotonic()-started,
                updates_path=str(log_path), updates_sha256=file_sha256(log_path),
                starting_model_sha256=model_sha, final_model_sha256=state_digest(model.state_dict()),
                scope="Disposable real-epoch prefix with inherited state; not a completed epoch or training checkpoint")
            if projection is not None:
                report["warmed_runtime_projection"] = projection
                report["projected_mean_epoch_seconds"] = projection["projected_mean_epoch_seconds"]
            write_json(run.path / "gate.json", report)
            write_json(output, report)
            if report["status"] != "passed":
                raise RuntimeError("Corrected chemistry stability, classification or runtime checks failed")
            run.annotate(gate_status="passed", completed_disposable_updates=len(records),
                         validation_weights_discarded=True, epoch_complete=False)
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic()-started)
            if reference_test is not None and "valid_reference_anchor_before" in report:
                try:
                    report["valid_reference_anchor_after"] = teacher.valid_reference_anchor.assert_unchanged()
                    report["valid_reference_anchor_unchanged"] = report["valid_reference_anchor_before"] == report["valid_reference_anchor_after"]
                except Exception as anchor_error:
                    report["valid_reference_anchor_error"] = f"{type(anchor_error).__name__}: {anchor_error}"
            write_json(run.path / "gate.json", report)
            write_json(output, report)
            if model is not None and not list((run.path / "early_health").glob("*.diagnostic.pt")):
                try:
                    snapshot = run.path / "failed-stability.diagnostic.pt"
                    digest = save_failed_early_health_snapshot(snapshot, model, optimizer, config, plan, run,
                        completed_epoch=36, partial_epoch=37, updates=report["last_update"], generators=generators,
                        health_sha256=file_sha256(output))
                    exclusive_json(run.path / "failed-stability-diagnostic.json", dict(path=str(snapshot), sha256=digest,
                        epoch_complete=False, promotable=False))
                except Exception as diagnostic_error:
                    exclusive_json(run.path / "diagnostic-error.json", dict(error=str(diagnostic_error), promotable=False))
            _observe_failed_generation_safely(plan, model, generators, run, report,
                                            live_state_observable=live_state_observable)
            raise
    # The successful completed-run record exists before binding this proof.
    native = json.loads(native_path.read_text())
    native["stability_gate_sha256"] = file_sha256(output)
    write_json(native_path, native)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan = continuation.read_plan(args.plan, args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    validate_guarded_restart(plan)
    allocation_started = time.monotonic()
    benchmark_candidate(plan)
    run_stability(plan, allocation_started=allocation_started)


if __name__ == "__main__":
    main()
