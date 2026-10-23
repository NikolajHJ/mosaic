"""Allocated fixtures for long-budget provenance and fail-closed paired outcomes.

No native model weights are needed; full native validation remains a later HPC
outcome. These tests must also execute on DTU HPC under the project policy.
"""
from copy import deepcopy
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("run_boltz_defog_long_comparison")


def record():
    return {"name":"matched-pilot","output_root":"/old","gradient_audit_interval":5,
        "optimizer":{"steps":25,"learning_rate":400.,"trust_radius":3.,"max_gradient_norm":.000125},
        "diagnostics":{"hard_samples":8},"fd_policy":"pointwise_native_ad_v4",
        "decoder":{"path":"/decoder","sha256":"a"*64},"initial_latent":{"path":"/z","sha256":"b"*64},
        "dbmol":{"contact_weight":2.5},"repair_guidance":{"weight":.001,"refresh_interval":5,
            "attempts_per_refresh":4,"provider":{"model":"defog_guacamol"},
            "weight_schedule":{"version":"linear_v1","start_weight":.001,"end_weight":.1,"start_fraction":.5,"end_fraction":1.}}}


def test_long_configuration_changes_only_declared_fields(runner,tmp_path):
    parent=record();saved=deepcopy(parent)
    control=runner.long_case_record(parent,"boltz_only",tmp_path / "control")
    guided=runner.long_case_record(parent,"boltz_defog",tmp_path / "guided")
    assert parent==saved
    for cfg in (control,guided):
        assert cfg["optimizer"]=={**parent["optimizer"],"steps":500}
        assert cfg["decoder"]==parent["decoder"] and cfg["initial_latent"]==parent["initial_latent"]
        assert cfg["dbmol"]==parent["dbmol"] and cfg["repair_guidance"]["provider"]==parent["repair_guidance"]["provider"]
        assert cfg["execution_budget"]["max_boltz_calls"]==20000
        assert cfg["gradient_audit_interval"]==5
        assert cfg["step_control"]["version"] == "best_trial_v2"
    assert control["repair_guidance"]["weight"]==0. and "weight_schedule" not in control["repair_guidance"]
    assert guided["repair_guidance"]["weight_schedule"]==runner.SCHEDULE
    assert guided["execution_budget"]["wall_seconds"]==82800
    assert control["execution_budget"]["wall_seconds"]==54000


def test_historical_long_builder_validation_retains_explicit_armijo_policy(runner, tmp_path):
    parent = record()
    parent["step_control"] = {"version": "armijo_v1", "max_trials": 8,
                              "armijo": .001, "minimum_improvement_ulps": 2.}
    old = runner.long_case_record(parent, "boltz_only", tmp_path,
                                  step_control_version="armijo_v1")
    assert old["step_control"] == parent["step_control"]
    new = runner.long_case_record(parent, "boltz_only", tmp_path)
    assert new["step_control"] == {**parent["step_control"], "version": "best_trial_v2"}


def plan_fixture(runner,tmp_path):
    def ref(name):return {"path":str(tmp_path / name),"sha256":"a"*64}
    return {"version":1,"name":"long","output_root":str(tmp_path / "out"),
        "reference_plan":ref("oldplan"),"reference_pilot":ref("oldpilot"),"reference_artifacts":[ref("oldrun")],
        "cases":[{"id":c,"config":ref(c)} for c in runner.CASE_IDS],"native_template":ref("template"),
        "cache":str(tmp_path / "cache"),"repair_search":{},"checkpoints":list(runner.CHECKPOINTS),"pose_seeds":[17,23],
        "prediction_priority":[list(v) for v in runner.PREDICTION_PRIORITY],
        "source_manifest_sha256":"b"*64,"source_archive_sha256":"c"*64,**deepcopy(runner.BUDGETS)}


def test_complete_reservation_is_explicit(runner,tmp_path):
    plan=plan_fixture(runner,tmp_path);runner.validate_plan_shape(plan)
    assert sum(plan["optimization_timeout_seconds"].values())+2*plan["followup_timeout_seconds"]+plan["startup_report_reserve_seconds"]==165600
    assert plan["allocation_seconds"]==172800


@pytest.mark.parametrize("change",["steps","seeds","priority","membership","budget","budget_type","reference","duplicate","extra"])
def test_plan_rejects_silent_changes(runner,tmp_path,change):
    p=plan_fixture(runner,tmp_path)
    if change=="steps":p["checkpoints"].pop()
    elif change=="seeds":p["pose_seeds"]=[17]
    elif change=="priority":p["prediction_priority"].reverse()
    elif change=="membership":p["cases"].reverse()
    elif change=="budget":p["maximum_boltz_calls_per_arm"]+=1
    elif change=="budget_type":p["allocation_seconds"]=float(p["allocation_seconds"])
    elif change=="reference":p["reference_artifacts"][0]["sha256"]="bad"
    elif change=="duplicate":p["reference_artifacts"]*=2
    else:p["undocumented"]=True
    with pytest.raises(ValueError):runner.validate_plan_shape(p)


def case_fixture(runner,guided,monkeypatch):
    import latentmol.boltz.additive_contract as profiles
    import latentmol.repair_guidance.integration_smoke as gates_module
    import latentmol.repair_guidance.matched_pilot as pilot
    monkeypatch.setattr(profiles,"profile_record",lambda *a,**k:{"repair_weight":k.get("repair_weight")})
    monkeypatch.setattr(gates_module,"full_gate_passed",lambda g,**k:g.get("passed") is True)
    calls=[]
    def measured(report,trace,cfg,context,*,requested_updates):
        calls.append((trace["step"],requested_updates));return report.get("passed") is True
    monkeypatch.setattr(pilot,"measured_descent_passed",measured)
    limits={"version":"tracked_native_v1","wall_seconds":82800 if guided else 54000,"max_boltz_calls":20000}
    from latentmol.optimization import StepControl
    config=SimpleNamespace(fd_policy=runner.POLICY,dbmol=object(),objective=object(),
        step_control=StepControl(version="armijo_v1"),
        execution_budget=SimpleNamespace(record=lambda:deepcopy(limits)),
        repair_guidance=SimpleNamespace(provider_activation_weight=.1 if guided else 0.,weight_at=lambda s,n:.1 if guided else 0.))
    evidence={k:0 for k in ("native_attempts","accepted_native_targets","accepted_targets","guided_accepted_updates","nonzero_guided_updates","target_only_fd_passes","target_only_pointwise_ad_passes","target_only_derivative_passes")}
    evidence["refreshes"]=100
    if guided:evidence.update(native_attempts=400,accepted_native_targets=100,accepted_targets=100,guided_accepted_updates=500,nonzero_guided_updates=500,target_only_pointwise_ad_passes=101,target_only_derivative_passes=101,repair_validation_passed=True)
    result={"status":"completed","accepted_updates":500,"repair_evidence":evidence,"oracle_cost":{"boltz":{"all":{"forward_attempted":18000}}}}
    trajectory={"requested_steps":500,"completed_requested_budget":True,"fixed_counts":[22],"prescribed_candidate_count":4509,
        "trace":[{"step":i,"accepted":i<500,"used_target":guided,"applied_weight":.1 if guided else 0.,"gradient_measured":guided,
                  "gradient_diagnostics":{"binding_norm":1.,"weighted_repair_norm":.1}} for i in range(501)]}
    gates={f"step-{i:04d}":{"passed":True,"context_sha256":"a"*64,"derivative_contract":{"version":runner.POLICY}} for i in range(0,501,5)}
    reports={i:{"passed":True} for i in range(500)}
    receipt={"version":"tracked_native_v1","limits":limits,"status":"completed","within_limits":True,"elapsed_seconds":10.,"tracked_boltz_calls":18000}
    return result,trajectory,gates,config,reports,receipt,calls


@pytest.mark.parametrize("guided",[False,True])
def test_all_long_guards_are_required_without_raw_validity_claim(runner,monkeypatch,guided):
    result,trajectory,gates,cfg,reports,budget,calls=case_fixture(runner,guided,monkeypatch)
    result["raw_valid"]=0
    assert runner.validate_case(result,trajectory,gates,cfg,reports,budget_receipt=budget)["passed"]
    assert calls==[(i,500) for i in range(500)]


@pytest.mark.parametrize("change",["missing_audit","wrong_policy","missing_descent","descent_failed","missing_state","zero_repair_gradient","lost_refresh","budget_failed","budget_limit","budget_calls","budget_time","context"])
def test_long_guard_cannot_approve_missing_or_failed_evidence(runner,monkeypatch,change):
    result,t,g,cfg,reports,budget,_=case_fixture(runner,True,monkeypatch)
    if change=="missing_audit":g.pop("step-0250")
    elif change=="wrong_policy":g["step-0250"]["derivative_contract"]["version"]="branch_local_v2"
    elif change=="missing_descent":reports.pop(399)
    elif change=="descent_failed":reports[399]["passed"]=False
    elif change=="missing_state":t["trace"].pop(400)
    elif change=="zero_repair_gradient":t["trace"][399]["gradient_diagnostics"]["weighted_repair_norm"]=0.
    elif change=="lost_refresh":result["repair_evidence"]["accepted_targets"]=99
    elif change=="budget_failed":budget["status"]="exhausted"
    elif change=="budget_limit":budget["limits"]={**budget["limits"],"max_boltz_calls":20001}
    elif change=="budget_calls":budget["tracked_boltz_calls"]=20001
    elif change=="budget_time":budget["elapsed_seconds"]=82800.
    else:g["step-0250"]["context_sha256"]="b"*64
    assert not runner.validate_case(result,t,g,cfg,reports,budget_receipt=budget)["passed"]


def test_control_refresh_wrapper_does_not_allow_native_repair(runner,monkeypatch):
    result,t,g,cfg,reports,budget,_=case_fixture(runner,False,monkeypatch)
    result["repair_evidence"]["native_attempts"]=1
    assert not runner.validate_case(result,t,g,cfg,reports,budget_receipt=budget)["passed"]


def test_long_best_trial_policy_uses_honest_policy_check_and_requires_full_budget(runner, monkeypatch):
    import latentmol.repair_guidance.matched_pilot as pilot
    from latentmol.optimization import StepControl
    result, trajectory, gates, cfg, reports, budget, _ = case_fixture(runner, False, monkeypatch)
    cfg.step_control = StepControl(version="best_trial_v2")
    calls = []
    def measured(report, trace, config, context, *, requested_updates):
        calls.append(trace["step"])
        return report.get("passed") is True
    monkeypatch.setattr(pilot, "measured_step_passed", measured)
    outcome = runner.validate_case(result, trajectory, gates, cfg, reports, budget_receipt=budget)
    assert outcome["passed"] and outcome["full_500_passed"] and calls == list(range(500))
    assert "measured_native_step_policy" in outcome["checks"]
    assert "measured_native_descent" not in outcome["checks"]
    terminal = {"kind": "stalled", "requested_updates": 500, "accepted_updates": 499,
        "attempted_updates": 500, "endpoint_step": 499, "completed_budget": False, "stalled": True}
    result.update(status="stalled", accepted_updates=499)
    trajectory.update(completed_requested_budget=False)
    assert not runner.validate_case(result, trajectory, gates, cfg, reports,
        budget_receipt=budget, terminal=terminal)["checks"]["declared_terminal"]


@pytest.mark.parametrize("guided",[False,True])
@pytest.mark.parametrize("endpoint",[0,7,25])
def test_verified_stall_is_healthy_but_never_full_500(runner,monkeypatch,guided,endpoint):
    result,t,g,cfg,reports,budget,calls=case_fixture(runner,guided,monkeypatch)
    terminal={"kind":"stalled","requested_updates":500,"accepted_updates":endpoint,"attempted_updates":endpoint+1,
        "endpoint_step":endpoint,"completed_budget":False,"stalled":True}
    result.update(status="stalled",accepted_updates=endpoint)
    t.update(completed_requested_budget=False,prescribed_candidate_count=(endpoint+1)*9)
    t["trace"]=t["trace"][:endpoint+1];t["trace"][-1]["accepted"]=False
    labels={f"step-{s:04d}" for s in {*range(0,endpoint+1,5),endpoint}}
    g={label:{"passed":True,"context_sha256":"a"*64,"derivative_contract":{"version":runner.POLICY}} for label in labels}
    reports={s:{"passed":True} for s in range(endpoint+1)}
    evidence=result["repair_evidence"];evidence["refreshes"]=endpoint//5+1
    if guided:evidence.update(accepted_targets=evidence["refreshes"],accepted_native_targets=evidence["refreshes"],
        guided_accepted_updates=endpoint,nonzero_guided_updates=endpoint,target_only_pointwise_ad_passes=len(labels),target_only_derivative_passes=len(labels),repair_validation_passed=endpoint>0)
    outcome=runner.validate_case(result,t,g,cfg,reports,budget_receipt=budget,terminal=terminal)
    assert outcome["passed"] and outcome["healthy_terminal_passed"]
    assert not outcome["full_500_passed"] and not outcome["completed_requested_budget"]
    assert calls==[(i,500) for i in range(endpoint)]
    g.pop(f"step-{endpoint:04d}")
    assert not runner.validate_case(result,t,g,cfg,reports,budget_receipt=budget,terminal=terminal)["passed"]


def test_followup_process_and_memberships_are_authenticated(runner):
    execution={"returncode":0,"timed_out":False,"failure":None,"interrupted":False}
    receipt={"status":"completed","case_id":"boltz_only","child_run":"/child","plan_sha256":"a"*64,"child_run_sha256":"b"*64,"prediction_attempts":0,
        "memberships":[{"step":s,"role":r,"seed":seed,"status":"invalid_raw" if r=="raw" else "no_witness"} for s in runner.CHECKPOINTS for r in ("raw","repaired") for seed in (17,23)]}
    kwargs={"case_id":"boltz_only","child":Path("/child"),"plan_sha256":"a"*64,"parent_sha256":"b"*64}
    assert runner.followup_completed(execution,receipt,**kwargs)
    for key,value in [("returncode",1),("timed_out",True),("failure",{"error":"x"}),("interrupted",True)]:
        assert not runner.followup_completed({**execution,key:value},receipt,**kwargs)
    for key,value in [("case_id","boltz_defog"),("plan_sha256","c"*64),("child_run_sha256","c"*64)]:
        assert not runner.followup_completed(execution,{**receipt,key:value},**kwargs)
    receipt["memberships"][0]["status"]="budget_excluded"
    assert not runner.followup_completed(execution,receipt,**kwargs)


def test_checkpoint_selection_preserves_raw_invalidity_and_separate_repair(runner,monkeypatch):
    module=importlib.import_module("evaluate_boltz_defog_long_checkpoints")
    candidates=[{"step":s,"draw":0,"smiles":None,"failure":"invalid","candidate_id":str(s),"graph_sha256":"a"*64} for s in runner.CHECKPOINTS]
    repairs=[{"step":s,"arm":"fixed_initial_mask","status":"witness_found","repair":{"status":"witness_found","weighted_witness":{"smiles":"CC","edit_count":1,"weighted_cost":.3}}} for s in runner.CHECKPOINTS]
    chosen=module.selected_molecules(candidates,repairs)
    assert len(chosen)==6 and all(c["raw"]["status"]=="invalid_raw" for c in chosen)
    assert all(c["raw"]["smiles"] is None and c["repaired"]["smiles"]=="CC" for c in chosen)
    candidates[0]["smiles"]="C";repairs[0]["repair"]={"status":"already_valid"}
    assert module.selected_molecules(candidates,repairs)[0]["repaired"]["edit_count"]==0
    with pytest.raises(ValueError):module.selected_molecules(candidates+[candidates[0]],repairs)
    early=module.selected_molecules(candidates[:2],repairs,terminal_step=25)
    assert all(r["raw"]["status"]==r["repaired"]["status"]=="not_reached" for r in early[2:])
    with pytest.raises(ValueError):module.selected_molecules(candidates,repairs,terminal_step=25)


def historical_fixture(tmp_path, *, accepted=25, stalled=False):
    """Actual controller/native terminal schema; no model arithmetic or scoring.

    The real pilot controller overwrites its optimization result status with
    ``passed``. Native receipts retain ``completed_budget``, all25 accepted
    attempts, their measured trial outcomes, and a separately saved endpoint.
    """
    from latentmol.optimization import StepControl
    child=tmp_path / "historical-child";folder=child / "fixed_initial_mask"
    folder.mkdir(parents=True)
    control=StepControl(version="armijo_v1",max_trials=2)
    native_config={"optimizer":{"steps":25},"step_control":asdict(control)}
    cfg=SimpleNamespace(record=lambda:deepcopy(native_config),step_control=control,
        optimizer=SimpleNamespace(steps=25),fd_policy="pointwise_native_ad_v4",dbmol=object(),objective=object(),
        repair_guidance=SimpleNamespace(provider_activation_weight=0.,weight_at=lambda s,n:0.))
    def write(path,value):path.write_text(json.dumps(value),encoding="utf-8")
    metrics=[{"step":s,"score":{"loss":30.-s},"selection":{"count":22},
              **({"update_status":"accepted"} if s<accepted else {"update_status":"stalled"} if stalled else {})}
             for s in range(accepted+1)]
    for row in metrics:write(folder / f"metrics-{row['step']:04d}.json",row)
    for s in range(accepted):
        trial={"index":0,"scale":1.,"oracle_called":True,"accepted":True,"reason":"accepted",
            "actual_loss_improvement":.1,"observed_improvement_guard":1e-8,"required_armijo_improvement":1e-4,
            "strictly_decreased":True,"informative_decrease":True,"armijo_satisfied":True}
        report={"policy":control.record(),"status":"accepted","accepted":True,"stalled":False,"trials":[trial],
            "oracle_calls_attempted":1,"oracle_calls_completed":1,"oracle_calls_nonfinite":0,
            "oracle_calls_unexpected_errors":0,"accepted_trial":0,"accepted_scale":1.,"actual_loss_improvement":.1}
        write(folder / f"step-control-{s:04d}.json",report)
    if stalled:
        write(folder / f"step-control-{accepted:04d}.json",{
            "policy":control.record(),"status":"stalled","accepted":False,"stalled":True,
            "stall_reason":"trial_budget_exhausted","actual_loss_improvement":0.,"actual_displacement_norm":0.,
            "oracle_calls_attempted":2,"oracle_calls_completed":2,"oracle_calls_nonfinite":0,"oracle_calls_unexpected_errors":0,
            "trials":[{"index":i,"scale":scale,"oracle_called":True,"accepted":False,"reason":"no_strict_decrease"}
                      for i,scale in enumerate(control.record()["trial_scales"])]})
    arm={"steps":accepted,"requested_steps":25,"accepted_updates":accepted,"attempted_updates":accepted+int(stalled),
        "stalled":stalled,"completed_requested_budget":not stalled,
        "optimization_status":"stalled" if stalled else "completed_budget","step_control":control.record(),"endpoint":metrics[-1]}
    summary={"arms":{"fixed_initial_mask":deepcopy(arm)},"frozen_state_unchanged":True,"oracle_cost":{"boltz":{"forward":125}}}
    write(folder / "summary.json",arm);write(child / "summary.json",summary)
    evidence={key:0 for key in ("native_attempts","accepted_targets","guided_accepted_updates","nonzero_guided_updates","target_only_fd_passes")}
    row={"id":"boltz_only","status":"passed","accepted_updates":accepted,"repair_evidence":evidence,
         "oracle_cost":deepcopy(summary["oracle_cost"]),"outcome":{"passed":True}}
    trajectory={"optimization_status":arm["optimization_status"],"accepted_updates":accepted,"requested_steps":25,
        "completed_requested_budget":not stalled,"fixed_counts":[22],"prescribed_candidate_count":(accepted+1)*9,
        "oracle_cost":deepcopy(summary["oracle_cost"]),
        "trace":[{"step":s,"accepted":s<accepted,"used_target":False,"applied_weight":0.} for s in range(accepted+1)]}
    return child,cfg,row,summary,trajectory


def test_historical_passed_controller_uses_native_completion_without_mutating_acceptance(runner,tmp_path,monkeypatch):
    import latentmol.boltz.additive_contract as profiles
    import latentmol.repair_guidance.matched_pilot as pilot
    child,cfg,row,summary,trajectory=historical_fixture(tmp_path)
    before=deepcopy((row,summary,trajectory))
    monkeypatch.setattr(profiles,"profile_record",lambda *a,**k:{"repair_weight":k.get("repair_weight")})
    monkeypatch.setattr(pilot,"full_gate_passed",lambda g,**k:g.get("passed") is True)
    monkeypatch.setattr(pilot,"measured_descent_passed",lambda report,*args,**kwargs:report.get("accepted") is True)
    gates={f"step-{i:04d}":{"passed":True,"context_sha256":"a"*64,
                           "derivative_contract":{"version":runner.POLICY}} for i in range(0,26,5)}
    reports={i:json.loads((child / "fixed_initial_mask" / f"step-control-{i:04d}.json").read_text()) for i in range(25)}
    # Regression reproducer: the unchanged reader must continue rejecting a
    # controller's acceptance label as an optimizer completion status.
    old=pilot.validate_case(row,trajectory,gates,guided=False,config=cfg,step_reports=reports)
    assert old["checks"]["complete_budget"] is False and not old["passed"]
    view,paths=runner._historical_optimization_view(row,summary,trajectory,cfg,child)
    actual=pilot.validate_case(view,trajectory,gates,guided=False,config=cfg,step_reports=reports)
    assert actual["passed"] and all(actual["checks"].values())
    assert view["status"]=="completed" and row["status"]=="passed"
    assert (row,summary,trajectory)==before
    assert len(paths)==52 and len(set(paths))==52
    assert child / "fixed_initial_mask/summary.json" in paths
    assert sum(p.name.startswith("metrics-") for p in paths)==26
    assert sum(p.name.startswith("step-control-") for p in paths)==25
    gates["step-0010"]["passed"]=False
    assert not pilot.validate_case(view,trajectory,gates,guided=False,config=cfg,step_reports=reports)["passed"]


def test_historical_valid_stall_is_not_relabelled_completed(runner,tmp_path):
    child,cfg,row,summary,trajectory=historical_fixture(tmp_path,accepted=24,stalled=True)
    with pytest.raises(ValueError,match="exact 25-update budget"):
        runner._historical_optimization_view(row,summary,trajectory,cfg,child)
    assert row["status"]=="passed" and summary["arms"]["fixed_initial_mask"]["stalled"] is True


@pytest.mark.parametrize("defect",["partial","frozen","controller_failed","controller_count","controller_oracle",
    "missing_oracle","trajectory_oracle","unbounded_policy","trajectory_status","trajectory_count","trajectory_requested","trajectory_budget","arm_mismatch",
    "missing_metric","missing_step","endpoint_mismatch","failed_step","extra_step"])
def test_historical_completion_rejects_inconsistent_or_unproven_native_evidence(runner,tmp_path,defect):
    child,cfg,row,summary,trajectory=historical_fixture(tmp_path)
    folder=child / "fixed_initial_mask"
    if defect=="partial":
        arm=summary["arms"]["fixed_initial_mask"];arm["accepted_updates"]=24
        (folder / "summary.json").write_text(json.dumps(arm))
    elif defect=="frozen":summary["frozen_state_unchanged"]=False
    elif defect=="controller_failed":row["status"]="failed"
    elif defect=="controller_count":row["accepted_updates"]=24
    elif defect=="controller_oracle":row["oracle_cost"]["boltz"]["forward"]+=1
    elif defect=="missing_oracle":
        for value in (row,summary,trajectory):value.pop("oracle_cost")
    elif defect=="trajectory_oracle":trajectory["oracle_cost"]["boltz"]["forward"]+=1
    elif defect=="unbounded_policy":cfg.record=lambda:{"optimizer":{"steps":25}}
    elif defect=="trajectory_status":trajectory["optimization_status"]="stalled"
    elif defect=="trajectory_count":trajectory["accepted_updates"]=24
    elif defect=="trajectory_requested":trajectory["requested_steps"]=500
    elif defect=="trajectory_budget":trajectory["completed_requested_budget"]=False
    elif defect=="arm_mismatch":summary["arms"]["fixed_initial_mask"]["requested_steps"]=24
    elif defect=="missing_metric":(folder / "metrics-0010.json").unlink()
    elif defect=="missing_step":(folder / "step-control-0010.json").unlink()
    elif defect=="endpoint_mismatch":
        value=json.loads((folder / "metrics-0025.json").read_text());value["score"]["loss"]+=1
        (folder / "metrics-0025.json").write_text(json.dumps(value))
    elif defect=="failed_step":
        value=json.loads((folder / "step-control-0010.json").read_text());value["trials"][0]["actual_loss_improvement"]=-.1
        (folder / "step-control-0010.json").write_text(json.dumps(value))
    else:(folder / "step-control-0025.json").write_text((folder / "step-control-0024.json").read_text())
    with pytest.raises(ValueError):runner._historical_optimization_view(row,summary,trajectory,cfg,child)
