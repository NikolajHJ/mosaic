"""Authenticated 500-update control/guided comparison in one serial allocation.

New long-run engineering protocol; a completed 25-update pilot is its input,
not a certificate for the unobserved 500-update trajectory. Native gates remain
mandatory. Run only on allocated DTU HPC. Allocated HPC validation is pending.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time


CASE_IDS = ("boltz_only", "boltz_defog")
STEPS = 500
AUDIT_INTERVAL = 5
CHECKPOINTS = (0, 25, 50, 100, 250, 500)
HARD_SAMPLES = 8
PREDICTION_PRIORITY = tuple((step,role,seed) for seed in (17,23)
    for step in (0,500,25,50,100,250) for role in ("raw","repaired"))
POLICY = "pointwise_native_ad_v4"
SCHEDULE = {"version": "linear_updates_v1", "start_weight": .001,
            "end_weight": .1, "start_update": 12, "end_update": 24}
BUDGETS = {"allocation_seconds": 172800, "internal_wall_seconds": 165600,
           "optimization_timeout_seconds": {"boltz_only": 54000, "boltz_defog": 82800},
           "followup_timeout_seconds": 7200, "startup_report_reserve_seconds": 14400,
           "maximum_boltz_calls_per_arm": 20000,
           "maximum_fresh_predictions_per_arm": 24, "fresh_prediction_timeout_seconds": 1800}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ref(path):
    from latentmol.experiments import file_sha256
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}


def _write(path, value):
    from latentmol.experiments import write_json
    write_json(Path(path), value)


def long_case_record(reference, case_id, output_root, *, step_control_version="best_trial_v2"):
    """Build future runs with best-trial selection; read old policy explicitly.

    Source-bound historical plans retain their recorded Armijo policy. This
    builder changes the default only for newly prepared immutable plans.
    """
    from dataclasses import asdict
    from latentmol.optimization import StepControl
    if case_id not in CASE_IDS:
        raise ValueError("Unknown long-comparison arm")
    raw = deepcopy(reference)
    raw["step_control"] = asdict(StepControl.from_mapping({
        **raw.get("step_control", {}), "version": step_control_version}))
    raw.update(name="long-500-" + case_id, output_root=str(output_root),
               gradient_audit_interval=AUDIT_INTERVAL)
    raw["optimizer"]["steps"] = STEPS
    raw["diagnostics"]["hard_samples"] = HARD_SAMPLES
    raw["execution_budget"] = {"version": "tracked_native_v1",
        "wall_seconds": BUDGETS["optimization_timeout_seconds"][case_id],
        "max_boltz_calls": BUDGETS["maximum_boltz_calls_per_arm"]}
    repair = raw["repair_guidance"]
    if case_id == "boltz_only":
        repair["weight"] = 0.0
        repair.pop("weight_schedule", None)
    else:
        repair["weight"] = .001
        repair["weight_schedule"] = deepcopy(SCHEDULE)
    return raw


def validate_plan_shape(plan):
    """Reject changed membership, allocation budgets and scientific checkpoints."""
    fields = {"version", "name", "output_root", "reference_plan", "reference_pilot",
              "reference_artifacts", "cases", "native_template", "cache", "repair_search",
              "checkpoints", "pose_seeds", "prediction_priority", "source_manifest_sha256", "source_archive_sha256",
              *BUDGETS}
    if not isinstance(plan, dict) or set(plan) != fields or type(plan["version"]) is not int or plan["version"] != 1:
        raise ValueError("Require the exact long-comparison version-1 plan")
    if (plan["checkpoints"] != list(CHECKPOINTS) or plan["pose_seeds"] != [17, 23]
            or plan["prediction_priority"] != [list(v) for v in PREDICTION_PRIORITY]):
        raise ValueError("Changed checkpoint or matched fresh-pose seed budget")
    if any(json.dumps(plan[key],sort_keys=True) != json.dumps(value,sort_keys=True) for key, value in BUDGETS.items()):
        raise ValueError("Changed declared long-run time/call budget")
    if ([c.get("id") for c in plan["cases"]] != list(CASE_IDS)
            or any(set(c) != {"id", "config"} for c in plan["cases"])):
        raise ValueError("Require control then guided, once each")
    for key in ("output_root", "cache"):
        if not isinstance(plan[key], str) or not Path(plan[key]).is_absolute():
            raise ValueError("Require absolute " + key)
    if (sum(plan["optimization_timeout_seconds"].values())
            + 2 * plan["followup_timeout_seconds"] + plan["startup_report_reserve_seconds"]
            != plan["internal_wall_seconds"]
            or plan["internal_wall_seconds"] >= plan["allocation_seconds"]):
        raise ValueError("Complete pair cannot fit its declared internal reservation")
    for ref in [plan["reference_plan"], plan["reference_pilot"], plan["native_template"],
                *plan["reference_artifacts"], *(c["config"] for c in plan["cases"])]:
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not Path(ref["path"]).is_absolute()
                or not isinstance(ref["sha256"], str) or len(ref["sha256"]) != 64
                or any(v not in "0123456789abcdef" for v in ref["sha256"])):
            raise ValueError("Malformed immutable input binding")
    if not plan["reference_artifacts"] or len({r["path"] for r in plan["reference_artifacts"]}) != len(plan["reference_artifacts"]):
        raise ValueError("Require unique authenticated historical evidence")


def _historical_optimization_view(row, summary, trajectory, config, child):
    """Recover optimization completion independently of controller acceptance.

    A successful pilot controller saves ``status=passed`` AFTER checking its
    optimizer's ``completed`` result. Those statuses have different meanings.
    Keep the controller row untouched, authenticate the full native terminal
    receipts, and derive a separate view for the unchanged scientific reader.
    A verified stall or partial trajectory cannot be this historical baseline.
    """
    from latentmol.boltz.trajectory_terminal import read_arm_terminal, terminal_evidence_paths
    from latentmol.config_identity import canonical_config_json

    if row.get("status") != "passed" or summary.get("frozen_state_unchanged") is not True:
        raise ValueError("Historical controller acceptance/frozen-state certification is missing")
    arms = summary.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"fixed_initial_mask"}:
        raise ValueError("Historical summary changes the native optimization arm")
    folder = Path(child) / "fixed_initial_mask"
    arm_path = folder / "summary.json"
    native_config = config.record()
    if not isinstance(native_config.get("step_control"), dict):
        raise ValueError("Historical native terminal lacks its declared bounded-step policy")
    paths = [arm_path, *terminal_evidence_paths(folder, native_config)]
    if (not folder.resolve().is_relative_to(Path(child).resolve())
            or any(path.is_symlink() or not path.is_file()
                   or not path.resolve().is_relative_to(folder.resolve()) for path in paths)):
        raise ValueError("Historical native terminal evidence is missing or escapes its child")
    arm = _read(arm_path)
    if canonical_config_json(arm) != canonical_config_json(arms["fixed_initial_mask"]):
        raise ValueError("Historical native arm summary differs from its completed child summary")
    terminal = read_arm_terminal(folder, native_config, arm)
    required = {"kind": "completed_budget", "requested_updates": 25, "accepted_updates": 25,
                "attempted_updates": 25, "endpoint_step": 25, "completed_budget": True, "stalled": False}
    if any(type(terminal.get(key)) is not type(value) or terminal.get(key) != value
           for key, value in required.items()):
        raise ValueError("Historical native terminal did not complete the exact 25-update budget")
    expected_trajectory = {"optimization_status": terminal["kind"],
        "requested_steps": terminal["requested_updates"], "accepted_updates": terminal["accepted_updates"],
        "completed_requested_budget": terminal["completed_budget"]}
    actual_trajectory = {key: trajectory.get(key) for key in expected_trajectory}
    costs = [value.get("oracle_cost") for value in (row, summary, trajectory)]
    if (canonical_config_json(actual_trajectory) != canonical_config_json(expected_trajectory)
            or type(row.get("accepted_updates")) is not int or row["accepted_updates"] != terminal["accepted_updates"]
            or any(not isinstance(value, dict) or not value for value in costs)
            or len({canonical_config_json(value) for value in costs}) != 1):
        raise ValueError("Historical controller/trajectory differs from the authenticated native terminal")
    return {**row, "status": "completed", "accepted_updates": terminal["accepted_updates"]}, paths


def _reference_evidence(plan_ref, pilot_ref):
    """Authenticate the completed short pair without changing its certificate."""
    from latentmol.boltz.active_set import ActiveSetConfig
    from latentmol.boltz.stability_study import verify_asset
    from latentmol.config_identity import canonical_config_json, verify_config_identity
    from latentmol.repair_guidance.matched_pilot import validate_case, validate_prior_run
    prior = _read(verify_asset(plan_ref)); pilot_path = verify_asset(pilot_ref); pilot = _read(pilot_path)
    if (prior.get("version") != 4 or prior.get("derivative_contract") != POLICY
            or pilot.get("status") != "passed" or pilot.get("engineering_passed") is not True
            or pilot.get("matched_initial_z") is not True
            or [r.get("id") for r in pilot.get("cases", [])] != list(CASE_IDS)):
        raise ValueError("Require the successful authenticated source109 pointwise pilot")
    parent_path = pilot_path.parent / "run.json"; parent = _read(parent_path)
    verify_config_identity(parent)
    if (parent.get("status") != "completed" or parent["config"].get("plan_sha256") != plan_ref["sha256"]
            or parent.get("source_archive_sha256") != prior["source_archive_sha256"]
            or parent.get("source", {}).get("sha256") != prior["source_manifest_sha256"]):
        raise ValueError("Historical pilot controller has inconsistent provenance")
    refs = [_ref(parent_path)]; configs = {}
    for declaration, row in zip(prior["cases"], pilot["cases"]):
        case_id = declaration["id"]
        if row.get("status") != "passed" or row.get("outcome", {}).get("passed") is not True:
            raise ValueError("Historical arm did not pass: " + case_id)
        config_path = verify_asset(declaration["config"]); cfg = ActiveSetConfig.load(config_path)
        child = Path(row["child_run"])
        if not child.resolve().is_relative_to((Path(prior["output_root"]) / case_id / "optimization").resolve()):
            raise ValueError("Historical child escaped the declared case")
        record_path = child / "run.json"; record = _read(record_path); verify_config_identity(record)
        validate_prior_run(cfg.record(),record,base_config=declaration["config"],prior_run_path=str(record_path),child_run=str(child))
        if (record.get("status") != "completed"
                or record["config"].get("config_file_sha256") != declaration["config"]["sha256"]
                or record.get("source_archive_sha256") != prior["source_archive_sha256"]
                or record.get("source", {}).get("sha256") != prior["source_manifest_sha256"]):
            raise ValueError("Historical optimization source/configuration differs")
        summary_path = verify_asset(row["summary"]); summary = _read(summary_path)
        if summary_path.resolve() != (child / "summary.json").resolve():
            raise ValueError("Historical summary reference is not its authenticated optimization child")
        trajectory_path = verify_asset(row["trajectory"]); trajectory = _read(trajectory_path)
        if trajectory_path.resolve() != (pilot_path.parent / (case_id + "-trajectory.json")).resolve():
            raise ValueError("Historical trajectory reference is not its authenticated pilot case")
        gates_path = child / "fixed_initial_mask/gates.json"; gates = _read(gates_path)
        step_paths = [child / "fixed_initial_mask" / f"step-control-{s:04d}.json" for s in range(25)]
        optimization, terminal_paths = _historical_optimization_view(row, summary, trajectory, cfg, child)
        outcome = validate_case(optimization, trajectory, gates, guided=case_id == "boltz_defog", config=cfg,
                                step_reports={s: _read(p) for s, p in enumerate(step_paths)})
        if outcome["passed"] is not True or summary.get("frozen_state_unchanged") is not True:
            raise ValueError("Historical scientific receipts do not pass their original contract")
        if canonical_config_json(outcome.get("checks")) != canonical_config_json(row.get("outcome", {}).get("checks")):
            raise ValueError("Historical scientific check coverage differs from its original acceptance")
        if (cfg.optimizer.steps != 25 or cfg.fd_policy != POLICY or cfg.arms != ("fixed_initial_mask",)
                or cfg.gradient_audit_interval != 5 or cfg.repair_guidance.refresh_interval != 5
                or cfg.repair_guidance.attempts_per_refresh != 4 or cfg.diagnostics.hard_samples != 8):
            raise ValueError("Historical configuration is not the validated 25-update protocol")
        paths = dict.fromkeys([config_path, record_path, summary_path, trajectory_path, gates_path,
                              *step_paths, *terminal_paths])
        refs.extend([_ref(p) for p in paths])
        configs[case_id] = cfg
    a,b=(configs[c] for c in CASE_IDS)
    for key in ("decoder", "affinity_checkpoint", "initial_latent", "context_manifest", "dbmol", "start", "optimizer", "step_control"):
        if getattr(a,key) != getattr(b,key):
            raise ValueError("Historical pair changes shared " + key)
    return prior, configs, refs


def prepare(reference_plan, reference_pilot, prepared_root, output_root):
    """Allocated CPU metadata preparation; no optimization or model creation."""
    import yaml
    from latentmol.boltz.active_set import ActiveSetConfig
    from latentmol.experiments import source_manifest
    from latentmol.repair_guidance.study import require_hpc
    require_hpc()
    prior, references, artifacts = _reference_evidence(reference_plan, reference_pilot)
    prepared_root=Path(prepared_root).resolve(); output_root=Path(output_root).resolve()
    prepared_root.mkdir(parents=True, exist_ok=False)
    cases=[]
    for case_id in CASE_IDS:
        raw=long_case_record(references[case_id].record(),case_id,output_root / case_id / "optimization")
        path=prepared_root / (case_id + ".yaml")
        file_record=deepcopy(raw);file_record["version"]=1;file_record.pop("context_identity",None)
        if file_record.get("context_manifest") is not None:file_record.pop("contexts",None)
        else:file_record.pop("context_manifest",None)
        path.write_text(yaml.safe_dump(file_record,sort_keys=False),encoding="utf-8")
        parsed=ActiveSetConfig.load(path)
        if parsed.repair_guidance.weight_at(25,STEPS) != (.1 if case_id=="boltz_defog" else 0.):
            raise ValueError("Absolute repair ramp does not preserve the short-pilot schedule")
        cases.append({"id":case_id,"config":_ref(path)})
    plan={"version":1,"name":"boltz-defog-long-500-comparison","output_root":str(output_root),
          "reference_plan":reference_plan,"reference_pilot":reference_pilot,"reference_artifacts":artifacts,
          "cases":cases,"native_template":prior["native_template"],"cache":prior["cache"],
          "repair_search":prior["repair_search"],"checkpoints":list(CHECKPOINTS),"pose_seeds":[17,23],
          "prediction_priority":[list(v) for v in PREDICTION_PRIORITY],
          "source_manifest_sha256":source_manifest(Path.cwd())["sha256"],
          "source_archive_sha256":os.environ["LATENTMOL_SOURCE_SHA256"],**deepcopy(BUDGETS)}
    validate_plan_shape(plan); _write(prepared_root / "plan.json",plan)
    return prepared_root / "plan.json"


def load_plan(path,expected_sha):
    from latentmol.boltz.active_set import ActiveSetConfig
    from latentmol.boltz.stability_study import verify_asset
    from latentmol.experiments import source_manifest
    from latentmol.config_identity import canonical_config_json
    from latentmol.repair_guidance.study import require_hpc
    require_hpc(); plan=_read(verify_asset({"path":str(Path(path).resolve()),"sha256":expected_sha}));validate_plan_shape(plan)
    identity={"source_manifest_sha256":source_manifest(Path.cwd())["sha256"],"source_archive_sha256":os.environ["LATENTMOL_SOURCE_SHA256"]}
    if any(plan[k]!=v for k,v in identity.items()):raise ValueError("Long plan belongs to another source release")
    for ref in plan["reference_artifacts"]:verify_asset(ref)
    prior, reference, artifacts=_reference_evidence(plan["reference_plan"],plan["reference_pilot"])
    if artifacts!=plan["reference_artifacts"]:raise ValueError("Historical evidence membership or digest changed")
    if any(plan[k]!=prior[k] for k in ("native_template","cache","repair_search")):
        raise ValueError("Native evaluation or bounded-repair policy changed")
    configs={}
    for case in plan["cases"]:
        case_id=case["id"];cfg=ActiveSetConfig.load(verify_asset(case["config"]))
        expected=long_case_record(reference[case_id].record(),case_id,Path(plan["output_root"]) / case_id / "optimization",
                                  step_control_version=cfg.step_control.version)
        if canonical_config_json(cfg.record())!=canonical_config_json(expected):raise ValueError("Long config changes undeclared science: "+case_id)
        configs[case_id]=cfg
    if len({cfg.step_control.version for cfg in configs.values()}) != 1:
        raise ValueError("Matched arms must use the same recorded step-control policy")
    verify_asset(plan["native_template"])
    return plan,configs,identity


def budget_receipt_passed(receipt,expected,oracle_cost):
    """A configured cap is not evidence that the completed child respected it."""
    elapsed=receipt.get("elapsed_seconds");calls=receipt.get("tracked_boltz_calls")
    rows=oracle_cost.get("boltz",{}) if isinstance(oracle_cost,dict) else {}
    counts=[row.get(key,0) for row in rows.values() if isinstance(row,dict) for key in ("forward_attempted","native_forward_attempted")]
    recomposed=(bool(rows) and len(counts)==2*len(rows) and all(type(v) is int and v>=0 for v in counts) and sum(counts)==calls)
    return (recomposed and receipt.get("version")=="tracked_native_v1" and receipt.get("status")=="completed"
        and receipt.get("within_limits") is True and receipt.get("limits")==expected
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<expected["wall_seconds"]
        and type(calls) is int and 0<calls<=expected["max_boltz_calls"])


def validate_case(result,trajectory,gates,config,step_reports,*,budget_receipt=None,terminal=None):
    """Validate the actual terminal prefix; a healthy stall is not 500 completion.

    Production callers supply ``read_arm_terminal``'s independently validated
    receipt, including the full final rejected-trial schedule on a stall.
    """
    from latentmol.boltz.additive_contract import profile_record
    from latentmol.repair_guidance.integration_smoke import full_gate_passed
    from latentmol.repair_guidance.matched_pilot import measured_step_check_name, measured_step_passed
    guided=config.repair_guidance.provider_activation_weight>0
    trace=trajectory.get("trace",[]);updates=[r for r in trace if r.get("accepted") is True]
    if terminal is None:
        terminal={"kind":"completed_budget","requested_updates":STEPS,"accepted_updates":STEPS,
            "attempted_updates":STEPS,"endpoint_step":STEPS,"completed_budget":True,"stalled":False}
    endpoint=terminal.get("endpoint_step");stalled=terminal.get("stalled") is True
    if type(endpoint) is not int or not 0<=endpoint<=STEPS:return {"passed":False,"checks":{"declared_terminal":False}}
    attempts=endpoint+int(stalled);refreshes=endpoint//5+1 if stalled else (endpoint+4)//5
    evidence=result.get("repair_evidence",{});labels={f"step-{s:04d}" for s in {*range(0,endpoint+1,AUDIT_INTERVAL),endpoint}}
    contexts={g.get("context_sha256") for g in gates.values()}
    def gate_pass(label,g):
        if config.fd_policy!=POLICY or g.get("derivative_contract",{}).get("version")!=POLICY:return False
        step=int(label.removeprefix("step-"));weight=config.repair_guidance.weight_at(step,STEPS) if guided else None
        return full_gate_passed(g,require_repair=guided,expected_profile=profile_record(config.dbmol,config.objective,repair_weight=weight))
    complete=not stalled and endpoint==STEPS
    checks={"declared_terminal":terminal.get("requested_updates")==STEPS and terminal.get("accepted_updates")==endpoint
                and terminal.get("attempted_updates")==attempts and terminal.get("completed_budget") is complete
                and terminal.get("kind")==("stalled" if stalled else "completed_budget")
                and result.get("status")==("stalled" if stalled else "completed") and result.get("accepted_updates")==endpoint
                and trajectory.get("requested_steps")==STEPS and trajectory.get("completed_requested_budget") is complete
                and (stalled and endpoint<STEPS or complete)
                and (config.step_control.version != "best_trial_v2" or complete),
            "all_state_receipts":[r.get("step") for r in trace]==list(range(endpoint+1)),
            "fixed_occupancy":trajectory.get("fixed_counts")==[22],
            "raw_sampling_budget":trajectory.get("prescribed_candidate_count")== (endpoint+1)*(HARD_SAMPLES+1),
            "full_guards":set(gates)==labels and all(gate_pass(label,g) for label,g in gates.items()),
            "fixed_context":len(contexts)==1 and None not in contexts,
            "native_execution_budget":budget_receipt_passed(budget_receipt or {},config.execution_budget.record(),result.get("oracle_cost")),
            measured_step_check_name(config):set(step_reports)==set(range(attempts)) and [r.get("step") for r in updates]==list(range(endpoint))
                and len(contexts)==1 and None not in contexts and all(measured_step_passed(
                    step_reports[r["step"]],r,config,next(iter(contexts)),requested_updates=STEPS) for r in updates)}
    if guided:
        checks.update(native_targets=evidence.get("accepted_targets")==refreshes and evidence.get("refreshes")==refreshes
            and evidence.get("accepted_native_targets",0)>=refreshes and evidence.get("native_attempts",0)>0,
            actual_guidance=(endpoint==0 or evidence.get("repair_validation_passed") is True) and evidence.get("guided_accepted_updates")==endpoint
                and evidence.get("nonzero_guided_updates")==endpoint and evidence.get("target_only_pointwise_ad_passes")==len(labels)
                and evidence.get("target_only_derivative_passes")==len(labels) and evidence.get("target_only_fd_passes")==0,
            both_gradients_reach_z=len(updates)==endpoint and all(r.get("used_target") is True and r.get("gradient_measured") is True
                and all(type(v) in (int,float) and math.isfinite(v) and v>1e-12
                    for v in (r.get("gradient_diagnostics",{}).get("binding_norm"),r.get("gradient_diagnostics",{}).get("weighted_repair_norm"))) for r in updates))
    else:
        checks["no_native_repair_or_guidance"]=all(evidence.get(k)==0 for k in ("native_attempts","accepted_native_targets","accepted_targets","guided_accepted_updates","nonzero_guided_updates","target_only_fd_passes","target_only_pointwise_ad_passes","target_only_derivative_passes")) and all(r.get("used_target") is False and r.get("applied_weight")==0 for r in trace)
        checks["disabled_wrapper_refreshes"]=evidence.get("refreshes")==refreshes
    return {"passed":all(checks.values()),"checks":checks,"terminal":terminal,
        "full_500_certificate":all(checks.values()) and complete,"full_500_passed":all(checks.values()) and complete,
        "healthy_terminal_passed":all(checks.values()),"completed_requested_budget":complete,
        "scope":("Actual engineering full-budget trajectory; selected finite fallback steps are not monotone-descent or convergence proof."
                 if config.step_control.version == "best_trial_v2" else
                 "Actual engineering terminal prefix; a stall is not convergence or 500-update completion")}


def _discover_partial(execution,config,identity,config_sha256):
    """Keep an authenticated terminal/partial child even when its process failed."""
    from latentmol.config_identity import verify_config_identity
    matches=[]
    for value in execution.get("run_paths",[]):
        child=Path(value)
        if not child.resolve().is_relative_to(config.output_root.resolve()) or not (child / "run.json").is_file():continue
        record=_read(child / "run.json");verify_config_identity(record)
        if (record.get("source",{}).get("sha256")==identity["source_manifest_sha256"]
                and record.get("source_archive_sha256")==identity["source_archive_sha256"]
                and record.get("config",{}).get("config_file_sha256")==config_sha256):matches.append(child)
    if len(set(matches))!=1:raise ValueError("Cannot uniquely authenticate the failed optimization child")
    return matches[0]


def followup_completed(execution,receipt,*,case_id,child,plan_sha256,parent_sha256):
    """A completed JSON cannot override a failed, timed-out or unrelated worker."""
    expected={(s,role,seed) for s in CHECKPOINTS for role in ("raw","repaired") for seed in (17,23)}
    rows=receipt.get("memberships",[])
    keys=[(r.get("step"),r.get("role"),r.get("seed")) for r in rows]
    return (execution.get("returncode")==0 and execution.get("timed_out") is False
        and execution.get("failure") is None and execution.get("interrupted") is False
        and receipt.get("status")=="completed" and receipt.get("case_id")==case_id
        and receipt.get("child_run")==str(child) and receipt.get("plan_sha256")==plan_sha256
        and receipt.get("child_run_sha256")==parent_sha256 and len(keys)==len(expected) and set(keys)==expected
        and all(r.get("status") in ("completed","invalid_raw","no_witness","not_reached") for r in rows)
        and type(receipt.get("prediction_attempts")) is int and 0<=receipt["prediction_attempts"]<=24)


def _reports(children,output,*,require_matched=False):
    """Presentation uses saved JSON only and cannot approve scientific failure."""
    from latentmol.reporting.optimization import render_report
    receipts=[]
    for label,child in children:
        try:receipts.append({"id":label,"status":"generated","path":str(render_report([child],output / label))})
        except Exception as error:receipts.append({"id":label,"status":"failed","error":str(error),"error_type":type(error).__name__})
    if children:
        try:receipts.append({"id":"combined","status":"generated","comparison":"matched_identity_checks" if require_matched else "descriptive_partial","path":str(render_report([p for _,p in children],output / "combined",labels=[k for k,_ in children],require_matched=require_matched,allow_guidance_comparison=True))})
        except Exception as error:receipts.append({"id":"combined","status":"failed","error":str(error),"error_type":type(error).__name__})
    return receipts


def run(path,expected_sha):
    from latentmol.boltz.stability_study import verified_child
    from latentmol.boltz.workflow_process import run_logged
    from latentmol.experiments import ExperimentRun
    from latentmol.repair_guidance.matched_pilot import verify_start
    from latentmol.repair_guidance.random_study import _optimize
    from latentmol.boltz.trajectory_terminal import read_arm_terminal
    plan,configs,identity=load_plan(path,expected_sha);started=time.monotonic();deadline=started+plan["internal_wall_seconds"]
    root=Path(plan["output_root"]);children=[]
    with ExperimentRun(root / "controller",plan["name"],{"plan":str(path),"plan_sha256":expected_sha,**identity},Path.cwd()) as experiment:
        report={"status":"running","engineering_passed":False,"cases":[{"id":c,"status":"not_attempted","followups":{"status":"not_attempted","memberships":[{"step":s,"role":role,"seed":seed,"status":"not_attempted"} for s in CHECKPOINTS for role in ("raw","repaired") for seed in (17,23)]}} for c in CASE_IDS],"checkpoints":list(CHECKPOINTS),"claim_limit":"One initialization, no replication; engineering and chemical results separate", "wall_budget":deepcopy(BUDGETS)}
        output=experiment.path / "comparison.json";_write(output,report)
        try:
            probe=experiment.path / "gpu-process.json"
            result=run_logged([sys.executable,"-m","latentmol.gpu_process_probe","--worker-python",configs["boltz_defog"].repair_guidance.provider["python"],"--output",str(probe)],experiment.path / "gpu-process.log",timeout_seconds=180)
            if result["timed_out"] or result["returncode"]!=0 or _read(probe).get("status")!="passed":raise RuntimeError("GPU-process coexistence check failed")
            for index,(case,row) in enumerate(zip(plan["cases"],report["cases"])):
                case_id=case["id"];cfg=configs[case_id]
                reservation=sum(plan["optimization_timeout_seconds"][c] + plan["followup_timeout_seconds"] for c in CASE_IDS[index:])+600
                if deadline-time.monotonic()<reservation:raise TimeoutError("Cannot reserve the complete remaining comparison")
                row["status"]="running";_write(output,report)
                try:
                    row.update(_optimize(case,experiment.path,case_id,identity,plan["optimization_timeout_seconds"][case_id]))
                    child=Path(row["child_run"]);children.append((case_id,child));row["initial_z_sha256"]=verify_start(child,cfg.initial_latent)
                    trajectory=_read(row["trajectory"]["path"]);gates=_read(child / "fixed_initial_mask/gates.json")
                    summary=_read(row["summary"]["path"]);terminal=read_arm_terminal(child / "fixed_initial_mask",cfg.record(),summary["arms"]["fixed_initial_mask"])
                    step_reports={s:_read(child / "fixed_initial_mask" / f"step-control-{s:04d}.json") for s in range(terminal["attempted_updates"])}
                    budget_receipt=_read(child / "execution-budget.json")
                    if summary.get("execution_budget")!=budget_receipt:raise ValueError("Execution budget summary differs from its terminal receipt")
                    row["execution_budget"]=budget_receipt
                    row["outcome"]=validate_case(row,trajectory,gates,cfg,step_reports,budget_receipt=budget_receipt,terminal=terminal)
                    if row["outcome"]["passed"] is not True:raise RuntimeError("Long optimization failed its declared health/derivative/descent checks")
                    row["status"]="stalled" if terminal["stalled"] else "passed"
                except Exception as error:
                    execution_path=experiment.path / (case_id+"-execution.json")
                    if execution_path.is_file() and not any(k==case_id for k,_ in children):
                        try:child=_discover_partial(_read(execution_path),cfg,identity,case["config"]["sha256"]);children.append((case_id,child));row["child_run"]=str(child)
                        except Exception as discovery:row["partial_discovery_error"]=str(discovery)
                    row.update(status="failed",error=str(error),error_type=type(error).__name__);_write(output,report)
                    raise RuntimeError("Stop comparison on failed optimization health; no replacement run") from error
                followup_path=experiment.path / (case_id+"-followups.json")
                command=[sys.executable,str(Path(__file__).with_name("evaluate_boltz_defog_long_checkpoints.py")),"--plan",str(path),"--sha256",expected_sha,"--case",case_id,"--child",str(child),"--output",str(followup_path)]
                execution=run_logged(command,experiment.path / (case_id+"-followups.log"),timeout_seconds=plan["followup_timeout_seconds"])
                row["followup_execution"]=execution
                row["followups"]=_read(followup_path) if followup_path.is_file() else {"status":"failed","error":"Checkpoint worker produced no final report"}
                from latentmol.experiments import file_sha256
                row["followup_completed"]=followup_completed(execution,row["followups"],case_id=case_id,child=child,
                    plan_sha256=expected_sha,parent_sha256=file_sha256(child / "run.json"))
                _write(output,report)
            report["matched_initial_z"]=len({r.get("initial_z_sha256") for r in report["cases"]})==1 and all(r.get("initial_z_sha256") for r in report["cases"])
            report["engineering_passed"]=all(r["status"]=="passed" for r in report["cases"]) and report["matched_initial_z"]
            report["healthy_terminal_comparison"]=all(r["status"] in ("passed","stalled") for r in report["cases"]) and report["matched_initial_z"]
            report["full_500_passed"]=report["engineering_passed"]
            report["healthy_terminal_passed"]=report["healthy_terminal_comparison"]
            report["evaluation_complete"]=all(r.get("followup_completed") is True for r in report["cases"])
            report["status"]=("completed" if report["engineering_passed"] else "stalled_observed") if report["healthy_terminal_comparison"] and report["evaluation_complete"] else "incomplete_evaluation"
            if not report["healthy_terminal_comparison"]:raise RuntimeError("Long pair lacks healthy matched terminal proof")
        except Exception as error:
            report.update(status="failed",engineering_passed=False,error=str(error),error_type=type(error).__name__)
            raise
        finally:
            report["seconds"]=time.monotonic()-started;_write(output,report)
            report["interactive_reports"]=_reports(children,root / "reports",require_matched=report["engineering_passed"]);_write(output,report)
            experiment.annotate(engineering_passed=report["engineering_passed"],evaluation_complete=report.get("evaluation_complete",False))
    return experiment.path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("prepare");p.add_argument("--reference-plan",type=Path,required=True);p.add_argument("--reference-plan-sha256",required=True);p.add_argument("--reference-pilot",type=Path,required=True);p.add_argument("--reference-pilot-sha256",required=True);p.add_argument("--prepared-root",type=Path,required=True);p.add_argument("--output-root",type=Path,required=True)
    for name in ("validate","run"):
        p=commands.add_parser(name);p.add_argument("--plan",type=Path,required=True);p.add_argument("--sha256",required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare":print(prepare({"path":str(args.reference_plan.resolve()),"sha256":args.reference_plan_sha256},{"path":str(args.reference_pilot.resolve()),"sha256":args.reference_pilot_sha256},args.prepared_root,args.output_root),flush=True)
    elif args.command=="validate":load_plan(args.plan,args.sha256);print("Authenticated long-run configuration; no model execution",flush=True)
    else:print(run(args.plan,args.sha256),flush=True)


if __name__=="__main__":main()
