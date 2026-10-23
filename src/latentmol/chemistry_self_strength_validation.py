"""Native checks, a restored100-update coefficient probe, then2,500-step proof.

Only the declared valid-output self-NLL coefficient changes. Full logical-batch
component VJPs are observed at four probe updates, never on held-out panels.
Probe measurements cannot pollute the recurring-update runtime estimate: the
unchanged native/stability loops measure normal production updates separately.
One allocation clock includes all native, probe and stability work. Every probe
state is discarded and original epoch36 is restored before the longer proof.
There is no coefficient selection, training callback or checkpoint promotion.
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

from . import chemistry_continuation as c
from .chemistry_execution_validation import _snapshot, _restore
from .chemistry_generation_comparison import observe_strength_generation
from .chemistry_guarded_validation import _compact_metrics, run_stability
from .chemistry_objective_gradient_diagnostics import (ObjectiveGradientDiagnostics,
    VERSION as GRADIENT_VERSION)
from .chemistry_recovery_diagnostic import _loader
from .chemistry_self_guidance_iteration import validate_self_guidance_iteration
from .experiments import ExperimentRun, file_sha256, write_json
from .failure_guidance_benchmark import benchmark_candidate
from .failure_guidance_selection import authenticate_readiness
from .train import learning_rate_at_epoch, state_digest


VERSION = "valid-self-strength-probe-v1"
POLICY = dict(version=VERSION, updates=100, gradient_updates=[1,39,96,100])


def validate_probe(plan):
    """Closed experimental scope; marker alone can never authorize training."""
    if "valid_reference_test" in plan:
        from .chemistry_valid_reference_validation import validate_probe as validate_reference_probe
        return validate_reference_probe(plan)
    if "target_weighting_test" in plan:
        from .chemistry_target_weighting_validation import validate_probe as validate_weighting_probe
        return validate_weighting_probe(plan)
    iteration = validate_self_guidance_iteration(plan)
    value = plan.get("self_strength_probe")
    if (iteration is None or type(value) is not dict
            or json.dumps(value,sort_keys=True,allow_nan=False) != json.dumps(POLICY,sort_keys=True)
            or plan.get("starting_checkpoint",{}).get("epoch") != 36
            or plan.get("short_continuation",{}).get("purpose") != "self_guidance_test"):
        raise ValueError("Require the explicit nonpromotable100-update original36 coefficient probe")
    return dict(POLICY,gradient_updates=list(POLICY["gradient_updates"]))


def _probe_layout(plan):
    """Share execution mechanics; each immutable experiment owns its contract."""
    if "valid_reference_test" in plan:
        from .chemistry_valid_reference_validation import probe_layout
        return probe_layout(plan)
    if "target_weighting_test" in plan:
        from .chemistry_target_weighting_validation import probe_layout
        return probe_layout(plan)
    return dict(version=VERSION, policy=dict(POLICY,gradient_updates=list(POLICY["gradient_updates"])),
        panels=[0,32,100], result_name="self_strength_probe.json",
        run_folder="self_strength_probe",run_name="valid-self-coefficient-probe")


def _probe_extension(plan):
    if "valid_reference_test" in plan:
        from . import chemistry_valid_reference_validation
        return chemistry_valid_reference_validation
    if "target_weighting_test" in plan:
        from . import chemistry_target_weighting_validation
        return chemistry_target_weighting_validation
    return None


def _clock_deadline(plan, allocation_started):
    now = time.monotonic()
    if (type(allocation_started) not in (int,float) or not math.isfinite(allocation_started)
            or not 0 <= allocation_started <= now):
        raise ValueError("Require the shared native/probe/stability allocation clock")
    # Probe has at most110 minutes; reserve15 minutes for safe cleanup inside
    # the SAME23-hour allocation cap used by the native and longer proof.
    return min(now+6600,allocation_started+plan["guarded_restart"]["validation_wall_seconds"]-900)


def require_probe_completion(plan, report):
    """A returned partial/cleanup-failed probe cannot launch the longer proof."""
    layout = _probe_layout(plan)
    policy = layout["policy"]
    count = policy["updates"]
    identity_field = layout.get("identity_field","self_guidance_iteration")
    required_terms = layout.get("weighted_terms",["composition","kl","occupancy","reconstruction","repair","valid_self"])
    names = {"model","reference","optimizer","gradients","reference_gradients",
             "model_modes","reference_modes","rng"}
    restoration = report.get("restoration",{})
    checks = restoration.get("checks",{})
    gradients = report.get("gradient_observations",[])
    if (report.get("version") != layout["version"] or report.get("status") != "passed_disposable_probe_not_training"
            or report.get("diagnostic_completed") is not True
            or report.get(identity_field) != plan[identity_field]
            or report.get("plan_sha256") != plan["_plan_sha256"]
            or report.get("archive_sha256") != plan["archive_sha256"]
            or report.get("manifest_sha256") != plan["manifest_sha256"]
            or report.get("starting_checkpoint_sha256") != plan["starting_checkpoint"]["sha256"]
            or report.get("optimizer_step_control") != plan["optimizer_step_control"]
            or report.get("policy") != policy
            or type(report.get("completed_updates")) is not int or report["completed_updates"] != count
            or report.get("promotable") is not False or report.get("epoch_complete") is not False
            or type(report.get("training_updates_promoted")) is not int or report["training_updates_promoted"] != 0
            or restoration.get("passed") is not True or set(checks) != names
            or any(type(value) is not bool or not value for value in checks.values())
            or [row.get("relative_update") for row in report.get("panels",[])] != layout["panels"]
            or any(row.get("status") != "passed" for row in report.get("panels",[]))
            or [row.get("relative_update") for row in gradients] != policy["gradient_updates"]
            or [row.get("relative_update") for row in report.get("updates",[])] != list(range(1,count+1))):
        raise ValueError("Probe did not complete its exact update/health/restoration contract")
    for observation in gradients:
        receipt = observation.get("receipt",{})
        if (receipt.get("version") != GRADIENT_VERSION
                or receipt.get("status") != "observed_not_readiness"
                or receipt.get("complete_logical_update") is not True
                or receipt.get("component_sum_matches_native_backward") is not True
                or type(receipt.get("real_points")) is not int or receipt["real_points"] != 256
                or type(receipt.get("prior_points")) is not int or receipt["prior_points"] != 512
                or type(receipt.get("microbatch_size")) is not int or receipt["microbatch_size"] != 128
                or receipt.get("observed_microbatches") != {"real":2,"prior":4}
                or receipt.get("weighted_terms") != required_terms
                or any(type(receipt.get(key)) is not int or receipt[key] != 0 for key in
                    ("observer_model_forwards","observer_optimizer_steps","observer_random_samples"))):
            raise ValueError("Coefficient probe lost a complete logical component-gradient receipt")
    extension = _probe_extension(plan)
    if extension is not None:
        extension.require_probe_bindings(plan,report)
    return report


def run_probe(plan, *, allocation_started):
    policy = validate_probe(plan)
    layout = _probe_layout(plan)
    extension = _probe_extension(plan)
    observe_generation = observe_strength_generation
    if extension is not None:
        observe_generation = extension.observe_probe_generation
    root = Path(plan["campaign"])
    output = root/layout["result_name"]
    if output.exists():
        raise FileExistsError("Preserve the old coefficient diagnostic; use a new immutable attempt")
    deadline = _clock_deadline(plan,allocation_started)
    started = time.monotonic()
    native_path = root/"failure_readiness.json"
    plan = dict(plan,_benchmark_gate_sha256=file_sha256(root/"benchmark_gate.json"))
    authenticate_readiness(plan,file_sha256(native_path))
    report = dict(version=layout["version"],status="running",plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        optimizer_step_control=plan["optimizer_step_control"],
        job_id=os.environ.get("LSB_JOBID"),policy=policy,completed_updates=0,
        training_updates_promoted=0,promotable=False,epoch_complete=False,
        allocation_seconds_before_probe=started-allocation_started,panels=[],updates=[],gradient_observations=[])
    identity_field = layout.get("identity_field","self_guidance_iteration")
    report[identity_field] = plan[identity_field]
    if extension is not None:
        report.update(extension.receipt_bindings(plan))
        report["last_attempted_relative_update"] = 0
    initial = None
    anchor_before = None
    with ExitStack() as resources, ExperimentRun(root/layout["run_folder"],layout["run_name"],
            plan,Path(plan["release"])) as run:
        report["run_dir"] = str(run.path)
        try:
            parent_proof = {}
            model,reference,optimizer,config,data,cohorts,generators,teacher,epoch,updates,_,_ = c.setup(
                plan,parent_authentication=parent_proof)
            resources.callback(teacher.close)
            if epoch != 36 or updates != plan["starting_checkpoint"]["updates"]:
                raise ValueError("Coefficient probe did not restore original epoch36")
            initial = _snapshot(model,reference,optimizer,generators)
            report.update(parent_checkpoint_authentication=parent_proof,starting_model_sha256=state_digest(model.state_dict()))
            if "valid_reference_test" in plan:
                anchor_before = extension.observe_anchor_identity(plan,teacher)
                report["valid_reference_anchor"] = dict(before=anchor_before)
            native = json.loads((root/"benchmark_gate.json").read_text())
            c.authenticated(root/"training_composition.json",native["composition_sha256"])
            composition = json.loads((root/"training_composition.json").read_text())
            baseline = c.observe_early_health(model,data,config,plan["protocol"],composition,generators)
            reference_observation = c.observe_early_health(reference,data,config,plan["protocol"],composition,generators)

            def observe(relative):
                value = baseline if relative == 0 else c.observe_early_health(
                    model,data,config,plan["protocol"],composition,generators)
                assessment = c.assess_early_health(value,reference_observation["population"],
                    reference_observation["reconstruction"],baseline["population"],baseline["reconstruction"],
                    plan["protocol"],starting_classification=baseline["classification"])
                path = run.path/f"panel-{relative:04d}.json"
                write_json(path,dict(relative_update=relative,observation=value,assessment=assessment,
                    diagnostic_only=True,promotable=False,epoch_complete=False))
                report["panels"].append(dict(relative_update=relative,path=str(path),sha256=file_sha256(path),
                    status=assessment["status"]))
                return assessment

            if observe(0)["status"] != "passed":
                raise RuntimeError("Authenticated epoch36 failed initial unchanged health checks")
            report["generation_start"] = observe_generation(plan,model,generators,
                run.path/"generation-start",completed_updates=0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_epoch(config,epoch+1)
            batches = iter(_loader(data,cohorts,config,generators,"production",plan["seed"]))
            overlap = bool(c.execution_settings(plan).get("overlap",False))
            for relative in range(1,policy["updates"]+1):
                if time.monotonic() > deadline:
                    raise TimeoutError("Disposable coefficient probe exhausted its shared runtime")
                hard = next(batches)
                if extension is not None:
                    report["last_attempted_relative_update"] = relative
                kwargs = dict(deadline=deadline,overlap=overlap,step_control=plan["optimizer_step_control"])
                if relative in policy["gradient_updates"]:
                    with ObjectiveGradientDiagnostics(model,real_points=256,prior_points=512,
                            microbatch_size=plan["protocol"]["microbatch_size"]) as observer:
                        metrics = c.chemistry_update(model,reference,optimizer,hard,config,plan["protocol"],
                            composition,teacher,generators,updates+relative-1,epoch+1,plan["parent_epoch"],
                            component_observer=observer,**kwargs)
                    report["gradient_observations"].append(dict(relative_update=relative,receipt=observer.receipt()))
                else:
                    metrics = c.chemistry_update(model,reference,optimizer,hard,config,plan["protocol"],
                        composition,teacher,generators,updates+relative-1,epoch+1,plan["parent_epoch"],**kwargs)
                report["completed_updates"] = relative
                report["updates"].append(dict(relative_update=relative,metrics=_compact_metrics(metrics)))
                if relative in layout["panels"][1:]:
                    assessment = observe(relative)
                    if assessment["status"] != "passed":
                        # The scientific health failure is primary even when a
                        # secondary saved-output observer fails or fills quota.
                        report["health_failure"] = dict(relative_update=relative,assessment=assessment)
                        write_json(output,report)
                        try:
                            report["generation_end"] = observe_generation(plan,model,generators,
                                run.path/"generation-failed-end",completed_updates=relative)
                        except Exception as error:
                            report["generation_end_error"] = f"{type(error).__name__}: {error}"
                        raise RuntimeError("Coefficient probe failed unchanged fixed-panel health; no longer proof or training")
                write_json(output,report)
            report["generation_end"] = observe_generation(plan,model,generators,
                run.path/"generation-end",completed_updates=policy["updates"])
            report.update(status="passed_disposable_probe_not_training",diagnostic_completed=True)
        except BaseException as error:
            report.update(status="failed",diagnostic_completed=False,error=f"{type(error).__name__}: {error}")
            if extension is not None and initial is not None:
                # Preserve a new failed state before full restoration, without
                # changing the historical coefficient-probe path or promoting
                # this partial DataLoader stream to a resumable checkpoint.
                write_json(output,report)
                try:
                    report["failed_diagnostic"] = extension.save_failed_probe_snapshot(plan,run,model,optimizer,
                        config,generators,last_completed_update=updates+report["completed_updates"],
                        attempted_update=updates+report["last_attempted_relative_update"],health_path=output)
                except BaseException as diagnostic_error:
                    report["failed_diagnostic_error"] = f"{type(diagnostic_error).__name__}: {diagnostic_error}"
            raise
        finally:
            errors = []
            if initial is not None:
                for action in (teacher.drain,teacher.clear_cache):
                    try: action()
                    except BaseException as error: errors.append(f"{type(error).__name__}: {error}")
                try:
                    _restore(initial,model,reference,optimizer,generators)
                    restored = _snapshot(model,reference,optimizer,generators)
                    checks = {key:c.nested_equal(value,restored[key]) for key,value in initial.items()}
                    report["restoration"] = dict(passed=all(checks.values()),checks=checks)
                    if not all(checks.values()):errors.append("Full original36 restoration failed")
                except BaseException as error:errors.append(f"{type(error).__name__}: {error}")
                if "valid_reference_test" in plan and anchor_before is not None:
                    try:
                        anchor_after = extension.observe_anchor_identity(plan,teacher)
                        unchanged = c.nested_equal(anchor_before,anchor_after)
                        report["valid_reference_anchor"] = dict(before=anchor_before,after=anchor_after,unchanged=unchanged)
                        if not unchanged: errors.append("Frozen epoch36 validity reference changed")
                    except BaseException as error: errors.append(f"{type(error).__name__}: {error}")
            report["elapsed_seconds"] = time.monotonic()-started
            if errors:report.update(status="failed",diagnostic_completed=False,cleanup_errors=errors)
            write_json(output,report);write_json(run.path/"diagnostic.json",report)
            run.annotate(promotable=False,epoch_complete=False,training_updates_promoted=0,diagnostic_status=report["status"])
    return require_probe_completion(plan,report)


def run_validation(plan):
    """Keep one clock; never use a probe endpoint as the longer proof's start."""
    validate_probe(plan)
    allocation_started = time.monotonic()
    benchmark_candidate(plan)
    probe = run_probe(plan,allocation_started=allocation_started)
    require_probe_completion(plan,probe)
    return run_stability(plan,allocation_started=allocation_started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    args = parser.parse_args()
    plan = c.read_plan(args.plan,args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    run_validation(plan)


if __name__ == "__main__":main()
