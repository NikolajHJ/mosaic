"""Restored200-update frozen-valid-reference test, then fresh2500 proof.

Reuse the existing numerical loop and unfiltered generation evaluator. The new
contract replaces the moving self target; it does not select a training state,
reset inherited moments, or identify the optimizer as a cause of collapse.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from .chemistry_valid_reference_policy import PROBE_VERSION as VERSION, PROBE_POLICY

GENERATION_VERSION = "chemistry-fixed-generation-valid-reference-diagnostic-v1"
POLICY = deepcopy(PROBE_POLICY)
PANELS = [0,32,100,200]
WEIGHTED_TERMS = ["composition","kl","occupancy","reconstruction","repair","valid_reference"]
RETIRED_FIELDS = {"self_guidance_iteration","self_strength_probe","target_weighting_test","target_weighting_probe",
                  "chemistry_target_weighting","valid_prior_self_guidance"}


def validate_probe(plan):
    """Closed metadata-only test scope; successful diagnostics cannot train."""
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    marker = validate_valid_reference_test(plan)
    if (marker is None or type(plan.get("valid_reference_probe")) is not dict
            or json.dumps(plan["valid_reference_probe"],sort_keys=True,allow_nan=False)
               != json.dumps(POLICY,sort_keys=True)
            or type(plan.get("starting_checkpoint",{}).get("epoch")) is not int
            or plan["starting_checkpoint"]["epoch"] != 36
            or plan["starting_checkpoint"].get("sha256") != marker["reference_sha256"]
            or plan.get("short_continuation",{}).get("purpose") != "self_guidance_test"):
        raise ValueError("Require the explicit nonpromotable200-update original36 valid-reference probe")
    return dict(POLICY,gradient_updates=list(POLICY["gradient_updates"]))


def probe_layout(plan):
    return dict(version=VERSION,policy=validate_probe(plan),panels=list(PANELS),
        identity_field="valid_reference_test",weighted_terms=list(WEIGHTED_TERMS),
        result_name="valid_reference_probe.json",run_folder="valid_reference_probe",
        run_name="chemistry-valid-reference-probe")


def receipt_bindings(plan):
    validate_probe(plan)
    return dict(valid_reference_test=deepcopy(plan["valid_reference_test"]),
        valid_prior_reference_guidance=deepcopy(plan["protocol"]["valid_prior_reference_guidance"]),
        evidence_scope="Frozen original36 strict-valid targets at identical fresh prior coordinates; "
                       "full logical component observations, not optimizer-causality proof")


def _anchor_identity(plan, value):
    from .chemistry_valid_reference_anchor import validate_anchor_identity
    value = validate_anchor_identity(value)
    if value["reference_sha256"] != plan["valid_reference_test"]["reference_sha256"]:
        raise ValueError("Probe requires the authenticated separate frozen epoch36 validity reference")
    return value


def observe_anchor_identity(plan, teacher):
    """Ask the public anchor boundary to verify actual immutable model state."""
    validate_probe(plan)
    anchor = getattr(teacher,"valid_reference_anchor",None)
    if anchor is None or not callable(getattr(anchor,"assert_unchanged",None)):
        raise ValueError("Valid-reference probe has no public authenticated anchor")
    return _anchor_identity(plan,anchor.assert_unchanged())


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def require_reference_update(plan, metrics, anchor):
    """Metadata-only common reader for actual native/probe/stability updates.

    Require the same frozen identity, all scheduled rows and both target chains;
    no scientific runtime is imported and no missing scalar is invented.
    """
    anchor = _anchor_identity(plan,anchor)
    if type(metrics) is not dict:
        raise ValueError("Probe update metrics must be an actual metadata object")
    reference = dict(epoch=36,checkpoint_sha256=anchor["reference_sha256"],model_sha256=anchor["model_sha256"])
    receipt = metrics.get("valid_prior_reference_guidance",{})
    if (type(receipt) is not dict or set(receipt) != {"policy","reference_identity","attempted","eligible",
            "excluded","reference_valid_live_invalid","target_receipt_chain_sha256","scope"}
            or receipt.get("policy") != plan["protocol"]["valid_prior_reference_guidance"]
            or receipt.get("reference_identity") != reference
            or type(receipt.get("reference_identity",{}).get("epoch")) is not int
            or any(type(receipt.get(key)) is not int for key in
                ("attempted","eligible","excluded","reference_valid_live_invalid"))
            or receipt["attempted"] != 512 or not 0 <= receipt["eligible"] <= 512
            or receipt["excluded"] != 512-receipt["eligible"]
            or not 0 <= receipt["reference_valid_live_invalid"] <= receipt["eligible"]
            or not _sha(receipt.get("target_receipt_chain_sha256"))
            or not _sha(metrics.get("ordered_teacher_receipt_chain_sha256"))
            or type(receipt.get("scope")) is not str or not receipt["scope"]
            or any(type(metrics.get(key)) not in (int,float) or not math.isfinite(metrics[key]) or metrics[key] < 0
                   for key in ("valid_prior_reference_nll","valid_prior_reference_weighted"))
            or RETIRED_FIELDS.intersection(metrics)):
        raise ValueError("Probe update lost its frozen same-z target/eligibility receipt chain")


def require_probe_bindings(plan, report):
    bindings = receipt_bindings(plan)
    if (any(report.get(k) != v for k,v in bindings.items()) or RETIRED_FIELDS.intersection(report)
            or json.dumps(report.get("policy"),sort_keys=True,allow_nan=False) != json.dumps(POLICY,sort_keys=True)
            or any(type(row.get("relative_update")) is not int
                   for key in ("panels","updates","gradient_observations") for row in report.get(key,[]))
            or "failed_diagnostic" in report or "failed_diagnostic_error" in report):
        raise ValueError("Valid-reference probe lost its separate objective/evidence identity")
    anchor = report.get("valid_reference_anchor",{})
    before = _anchor_identity(plan,anchor.get("before"))
    after = _anchor_identity(plan,anchor.get("after"))
    if (set(anchor) != {"before","after","unchanged"} or before != after or anchor.get("unchanged") is not True
            or before["model_sha256"] != report.get("starting_model_sha256")):
        raise ValueError("Frozen epoch36 validity reference changed during the disposable probe")
    for row in report["updates"]:
        require_reference_update(plan,row.get("metrics",{}),before)
    from .experiments import file_sha256
    root = Path(report["run_dir"]).resolve(strict=True)
    for key,relative in (("generation_start",0),("generation_end",200)):
        receipt = report.get(key)
        if type(receipt) is not dict or set(receipt) != {"path","sha256"}:
            raise ValueError("Valid-reference probe lacks a saved unfiltered generation endpoint")
        path = Path(receipt["path"]).resolve(strict=True)
        if (not path.is_relative_to(root) or path.name != "metrics.json"
                or file_sha256(path) != receipt["sha256"]):
            raise ValueError("Valid-reference generation endpoint escaped or changed")
        value = json.loads(path.read_text())
        if (value.get("version") != GENERATION_VERSION or RETIRED_FIELDS.intersection(value)
                or any(value.get(k) != bindings[k] for k in ("valid_reference_test","valid_prior_reference_guidance"))
                or value.get("plan_sha256") != plan["_plan_sha256"]
                or value.get("archive_sha256") != plan["archive_sha256"]
                or value.get("manifest_sha256") != plan["manifest_sha256"]
                or value.get("starting_checkpoint_sha256") != plan["starting_checkpoint"]["sha256"]
                or value.get("valid_reference_anchor") != before
                or type(value.get("completed_disposable_updates")) is not int
                or value["completed_disposable_updates"] != relative
                or value.get("prior_count") != 1024 or value.get("categorical_samples_per_latent") != 4
                or value.get("evaluation_seed") != plan["protocol"]["evaluation_seed"]
                or value.get("diagnostic_only") is not True
                or value.get("promotable") is not False or value.get("epoch_complete") is not False):
            raise ValueError("Valid-reference generation endpoint lost its exact experiment identity")


def observe_probe_generation(plan, model, generators, output, *, completed_updates):
    """Label only new observations from the existing fixed-seed raw evaluator."""
    validate_probe(plan)
    if type(completed_updates) is not int or completed_updates not in PANELS:
        raise ValueError("Valid-reference generation requires an exact declared probe point")
    from .chemistry_generation_comparison import _observe
    from .experiments import file_sha256, write_json
    receipt = _observe(plan,model,generators,output,completed_updates=completed_updates)
    path = Path(receipt["path"])
    result = json.loads(path.read_text())
    result.update(version=GENERATION_VERSION,**receipt_bindings(plan),diagnostic_only=True,
        promotable=False,epoch_complete=False,
        scope="Unfiltered fixed-seed200-update valid-reference diagnostic; not a2500-update proof or training")
    write_json(path,result)
    return dict(path=str(path),sha256=file_sha256(path))


def save_failed_probe_snapshot(plan, run, model, optimizer, config, generators, *,
                               last_completed_update, attempted_update, health_path):
    """Keep failed state with truthful counters in the rejected-resume format."""
    from .chemistry_early_health import exclusive_json, save_failed_early_health_snapshot
    from .experiments import file_sha256
    snapshot = run.path/"failed-valid-reference-probe.diagnostic.pt"
    digest = save_failed_early_health_snapshot(snapshot,model,optimizer,config,plan,run,
        completed_epoch=36,partial_epoch=37,updates=last_completed_update,generators=generators,
        health_sha256=file_sha256(health_path))
    receipt = dict(path=str(snapshot),sha256=digest,epoch_complete=False,promotable=False,
        last_completed_update=last_completed_update,attempted_update=attempted_update,
        state_may_include_partial_attempted_update=attempted_update>last_completed_update,
        snapshot_updates_field_scope="Last successfully returned update; not a partial-state completion certificate",
        valid_reference_test=deepcopy(plan["valid_reference_test"]),
        scope="Failed disposable valid-reference probe; explicitly rejected as a training resume")
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
    """Native→200 probe→eight restore plus anchor invariance→fresh2500 proof."""
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
