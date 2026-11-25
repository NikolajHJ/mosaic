"""Disposable bond-target weighting probe, then the unchanged longer proof.

The shared coefficient-probe loop owns model/data/teacher execution and complete
restoration. This module declares the separate 200-update experiment and labels
its saved observations honestly; it neither selects an objective nor promotes a
checkpoint. Primitive finite-difference fixtures and the full component VJPs are
different evidence: component receipts do not isolate a NONE-bond causal effect.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


VERSION = "chemistry-target-weighting-probe-v1"
GENERATION_VERSION = "chemistry-fixed-generation-target-weighting-diagnostic-v1"
POLICY = dict(version=VERSION, updates=200, gradient_updates=[1,39,96,100,199,200])
PANELS = [0,32,100,200]


def validate_probe(plan):
    """Validate the closed test contract without importing numerical modules."""
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    marker = validate_target_weighting_test(plan)
    if (marker is None or "self_strength_probe" in plan
            or json.dumps(plan.get("target_weighting_probe"),sort_keys=True,allow_nan=False)
               != json.dumps(POLICY,sort_keys=True)
            or plan.get("starting_checkpoint",{}).get("epoch") != 36
            or plan.get("short_continuation",{}).get("purpose") != "self_guidance_test"):
        raise ValueError("Require the explicit nonpromotable200-update original36 target-weighting probe")
    return dict(POLICY,gradient_updates=list(POLICY["gradient_updates"]))


def probe_layout(plan):
    return dict(version=VERSION,policy=validate_probe(plan),panels=list(PANELS),
        result_name="target_weighting_probe.json",run_folder="target_weighting_probe",
        run_name="chemistry-target-weighting-probe")


def receipt_bindings(plan):
    validate_probe(plan)
    return dict(target_weighting_test=dict(plan["target_weighting_test"]),
        chemistry_target_weighting=dict(plan["protocol"]["chemistry_target_weighting"]),
        evidence_scope="Weighted primitive FD fixtures and full logical weighted component VJPs; "
                       "no matched NONE-only optimizer counterfactual or causal certificate")


def require_probe_bindings(plan, report):
    """Additional identities must not be borrowed from a historical100 probe."""
    bindings = receipt_bindings(plan)
    if (any(report.get(key) != value for key,value in bindings.items())
            or json.dumps(report.get("policy"),sort_keys=True,allow_nan=False)
               != json.dumps(POLICY,sort_keys=True)
            or any(type(row.get("relative_update")) is not int
                   for key in ("panels","updates","gradient_observations") for row in report.get(key,[]))
            or "failed_diagnostic" in report or "failed_diagnostic_error" in report):
        raise ValueError("Target-weighting probe lost its explicit objective/evidence identity")
    for key,relative in (("generation_start",0),("generation_end",200)):
        receipt = report.get(key)
        if type(receipt) is not dict or set(receipt) != {"path","sha256"}:
            raise ValueError("Target-weighting probe lacks its saved unfiltered generation endpoint")
        from .experiments import file_sha256
        path = Path(receipt["path"]).resolve(strict=True)
        root = Path(report["run_dir"]).resolve(strict=True)
        if (not path.is_relative_to(root) or path.name != "metrics.json"
                or file_sha256(path) != receipt["sha256"]):
            raise ValueError("Target-weighting generation endpoint escaped or changed")
        value = json.loads(path.read_text())
        if (value.get("version") != GENERATION_VERSION
                or value.get("target_weighting_test") != plan["target_weighting_test"]
                or value.get("chemistry_target_weighting") != plan["protocol"]["chemistry_target_weighting"]
                or value.get("self_guidance_iteration") != plan["self_guidance_iteration"]
                or value.get("plan_sha256") != plan["_plan_sha256"]
                or value.get("archive_sha256") != plan["archive_sha256"]
                or value.get("manifest_sha256") != plan["manifest_sha256"]
                or value.get("starting_checkpoint_sha256") != plan["starting_checkpoint"]["sha256"]
                or type(value.get("completed_disposable_updates")) is not int
                or value["completed_disposable_updates"] != relative
                or value.get("diagnostic_only") is not True
                or value.get("promotable") is not False or value.get("epoch_complete") is not False):
            raise ValueError("Target-weighting generation endpoint lost its experiment identity")


def observe_probe_generation(plan, model, generators, output, *, completed_updates):
    """Reuse the existing saved-raw evaluator, never a new sampling procedure."""
    validate_probe(plan)
    if type(completed_updates) is not int or completed_updates not in PANELS:
        raise ValueError("Target-weighting generation requires an exact declared probe point")
    from .chemistry_generation_comparison import _observe
    from .experiments import file_sha256, write_json
    receipt = _observe(plan,model,generators,output,completed_updates=completed_updates)
    path = Path(receipt["path"])
    result = json.loads(path.read_text())
    result.update(version=GENERATION_VERSION,target_weighting_test=dict(plan["target_weighting_test"]),
        chemistry_target_weighting=dict(plan["protocol"]["chemistry_target_weighting"]),
        self_guidance_iteration=dict(plan["self_guidance_iteration"]),
        diagnostic_only=True,promotable=False,epoch_complete=False,
        scope="Unfiltered fixed-seed200-update target-weighting diagnostic; not a2500-update proof or training")
    write_json(path,result)
    return dict(path=str(path),sha256=file_sha256(path))


def save_failed_probe_snapshot(plan, run, model, optimizer, config, generators, *,
                               last_completed_update, attempted_update, health_path):
    """Retain failure state; its counter is the last returned complete update.

    A failed update can have partially mutated live state. The saved diagnostic
    payload's ``updates`` counter therefore does not certify that all tensors
    represent precisely that completed optimizer boundary; it cannot resume.
    """
    from .chemistry_early_health import exclusive_json, save_failed_early_health_snapshot
    from .experiments import file_sha256
    snapshot = run.path/"failed-target-weighting-probe.diagnostic.pt"
    digest = save_failed_early_health_snapshot(snapshot,model,optimizer,config,plan,run,
        completed_epoch=36,partial_epoch=37,updates=last_completed_update,generators=generators,
        health_sha256=file_sha256(health_path))
    receipt = dict(path=str(snapshot),sha256=digest,epoch_complete=False,promotable=False,
        last_completed_update=last_completed_update,attempted_update=attempted_update,
        state_may_include_partial_attempted_update=attempted_update>last_completed_update,
        snapshot_updates_field_scope="Last successfully returned update; not a partial-state completion certificate",
        scope="Failed disposable probe; explicitly rejected as a training resume")
    exclusive_json(run.path/"failed-probe-diagnostic.json",receipt)
    return receipt


def run_probe(plan, *, allocation_started):
    validate_probe(plan)
    from .chemistry_self_strength_validation import run_probe as shared_probe
    return shared_probe(plan,allocation_started=allocation_started)


def require_probe_completion(plan, report):
    validate_probe(plan)
    from .chemistry_self_strength_validation import require_probe_completion as shared_completion
    return shared_completion(plan,report)


def run_validation(plan):
    """Native,200 disposable updates, restore, then fresh2500; one23h clock."""
    validate_probe(plan)
    from .failure_guidance_benchmark import benchmark_candidate
    from .chemistry_guarded_validation import run_stability
    allocation_started = time.monotonic()
    benchmark_candidate(plan)
    report = run_probe(plan,allocation_started=allocation_started)
    require_probe_completion(plan,report)
    return run_stability(plan,allocation_started=allocation_started)


def main():
    from . import chemistry_continuation as c
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--plan-sha256",required=True)
    args = parser.parse_args()
    plan = c.read_plan(args.plan,args.plan_sha256)
    plan["_plan_sha256"] = args.plan_sha256
    run_validation(plan)


if __name__ == "__main__":
    main()
