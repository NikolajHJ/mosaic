"""Allocated-only metadata contracts for bounded original-loss continuation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from latentmol import chemistry_short_continuation as short
from latentmol import chemistry_early_health as health
from latentmol import chemistry_restart_policy as restart
from latentmol.experiments import file_sha256, write_json


def baseline_plan(tmp_path=None, purpose="baseline_training"):
    files={"fixture.py":"f"*64}
    start=dict(path="/work3/s234818/latentmol/fixture/epoch_0036.pt",sha256=short.START_SHA,
        source_plan="/work3/s234818/latentmol/fixture/source121.json",source_plan_sha256=short.SOURCE_PLAN_SHA,
        completion="/work3/s234818/latentmol/fixture/source121-completion.json",
        completion_sha256=short.SOURCE_COMPLETION_SHA,epoch=36,updates=237456,
        algorithm=dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()))
    protocol=deepcopy(short.ORIGINAL_PROTOCOL)
    protocol.update(baseline_continuation={"version":short.BASELINE_VERSION},early_health=deepcopy(short.EARLY_POLICY))
    if purpose=="self_guidance_test":
        protocol["valid_prior_self_guidance"]={"version":short.SELF_VERSION,"weight":.01}
    return dict(short_continuation=dict(version=short.VERSION,start_epoch=36,end_epoch=39,purpose=purpose),
        protocol=protocol,starting_checkpoint=start,handoff_version="chemistry-objective-comparison-handoff-v1",
        guarded_restart=deepcopy(short.GUARDED_POLICY),execution=deepcopy(short.EXECUTION),
        campaign=str(tmp_path or "/work3/s234818/latentmol/runs/fixture"),
        _plan_sha256="a"*64,archive_sha256="b"*64,manifest_sha256="c"*64,**short.ORIGINAL_IDENTITIES)


def test_absent_policy_retains_historical_epoch100_and_no_new_guards():
    p=dict(protocol={"total_epochs":100})
    assert short.validate_short_continuation(p) is None
    assert short.require_training_branch(p) is None
    assert short.continuation_end_epoch(p)==100
    assert short.expected_stop_reason(p,99)=="wall_budget_epoch_boundary"
    assert short.expected_stop_reason(p,100)=="planned_horizon"
    assert restart.validate_guarded_restart(p) is None
    assert health.validate_early_health_policy({}) is None


@pytest.mark.parametrize("purpose",["baseline_training","self_guidance_test"])
def test_explicit_old_objective_uses_exact_same_monitoring_and_disposable_proof(tmp_path,purpose):
    p=baseline_plan(tmp_path,purpose)
    assert short.validate_short_continuation(p)==p["short_continuation"]
    assert short.continuation_end_epoch(p)==39 and p["protocol"]["total_epochs"]==100
    assert restart.validate_guarded_restart(p)==short.GUARDED_POLICY
    assert health.validate_early_health_policy(p["protocol"])==short.EARLY_POLICY
    parsed=short.validate_short_continuation(p);parsed["end_epoch"]=100
    assert p["short_continuation"]["end_epoch"]==39


def test_baseline_training_stops_at39_without_final100_and_weak_test_cannot_train(tmp_path):
    p=baseline_plan(tmp_path)
    assert short.require_training_branch(p)["purpose"]=="baseline_training"
    assert short.expected_stop_reason(p,37)==short.expected_stop_reason(p,38)=="wall_budget_epoch_boundary"
    assert short.expected_stop_reason(p,39)=="bounded_horizon"
    with pytest.raises(ValueError):short.expected_stop_reason(p,40)
    with pytest.raises(ValueError):short.expected_stop_reason(p,100)
    test=baseline_plan(tmp_path,"self_guidance_test")
    with pytest.raises(ValueError,match="disposable"):short.require_training_branch(test)
    with pytest.raises(ValueError,match="disposable"):short.expected_stop_reason(test,39)


@pytest.mark.parametrize("key,value",[("version","unknown"),("start_epoch",35),("start_epoch",36.),
    ("end_epoch",40),("end_epoch",True),("purpose","training"),("extra",True)])
def test_bounds_and_purpose_are_exact_not_coerced(tmp_path,key,value):
    p=baseline_plan(tmp_path);p["short_continuation"][key]=value
    with pytest.raises(ValueError):short.validate_short_continuation(p)


@pytest.mark.parametrize("change",[
    lambda p:p.pop("short_continuation"),
    lambda p:p.update(short_continuation=None),
    lambda p:p["protocol"].pop("baseline_continuation"),
    lambda p:p["protocol"].update(baseline_continuation=None),
    lambda p:p["protocol"].update(baseline_continuation={"version":"unknown"}),
    lambda p:p["protocol"].update(failure_guidance=None),
    lambda p:p["protocol"].update(failure_guidance={"loss":{"version":"repair-failure-guidance-v2"}}),
    lambda p:p["protocol"].update(prior_ratio=3),
    lambda p:p["protocol"].update(repair_weight=.1),
    lambda p:p["protocol"].update(repair_ramp_epochs=1),
    lambda p:p["protocol"].update(occupancy_anchor_weight=0.),
    lambda p:p["protocol"]["search"].update(max_edits=6),
    lambda p:p["protocol"].update(total_epochs=39),
    lambda p:p["protocol"].update(unknown_loss_weight=.1),
    lambda p:p["protocol"]["early_health"].update(first_check_updates=100),
    lambda p:p["protocol"]["early_health"].update(interval_updates=50),
    lambda p:p["protocol"]["early_health"].update(max_strict_validity_drop=.03),
    lambda p:p["guarded_restart"].update(validation_updates=2499),
    lambda p:p["guarded_restart"].update(validation_wall_seconds=86400),
    lambda p:p.update(runtime_adoption={"version":"reuse_failed"}),
])
def test_no_original_loss_sampling_search_or_proof_setting_can_change(tmp_path,change):
    p=baseline_plan(tmp_path);change(p)
    with pytest.raises(ValueError):short.validate_short_continuation(p)


@pytest.mark.parametrize("section,key,value",[("starting_checkpoint","epoch",37),
    ("starting_checkpoint","updates",237457),("starting_checkpoint","sha256","f"*64),
    ("starting_checkpoint","source_plan_sha256","f"*64),
    ("starting_checkpoint","completion_sha256","f"*64),
    ("execution","workers",20),("execution","overlap",False),
    (None,"parent_epoch",36),(None,"parent_checkpoint_sha256","f"*64),
    (None,"dataset_sha256","f"*64),(None,"config_sha256","f"*64),
    (None,"training_molecules",1688538)])
def test_actual36_and_frozen35_data_config_and_execution_are_bound(tmp_path,section,key,value):
    p=baseline_plan(tmp_path);(p[section] if section else p)[key]=value
    with pytest.raises(ValueError):short.validate_short_continuation(p)


def test_origin_algorithm_map_cannot_change_under_an_old_digest(tmp_path):
    p=baseline_plan(tmp_path);p["starting_checkpoint"]["algorithm"]["files"]["fixture.py"]="0"*64
    with pytest.raises(ValueError):short.validate_short_continuation(p)


@pytest.mark.parametrize("value",[None,{"version":short.SELF_VERSION,"weight":.02},
    {"version":short.SELF_VERSION,"weight":0.}, {"version":short.SELF_VERSION,"weight":True},
    {"version":short.SELF_VERSION,"weight":float("nan")},
    {"version":short.SELF_VERSION,"weight":.01,"extra":True}])
def test_weak_test_is_exact_point01_and_cannot_contaminate_baseline(tmp_path,value):
    p=baseline_plan(tmp_path,"self_guidance_test");p["protocol"]["valid_prior_self_guidance"]=value
    with pytest.raises(ValueError):short.validate_short_continuation(p)
    p=baseline_plan(tmp_path);p["protocol"]["valid_prior_self_guidance"]=value
    with pytest.raises(ValueError):short.validate_short_continuation(p)


@pytest.mark.parametrize("weight",[0.,.01,.02,1])
def test_generic_loss_schema_remains_configurable_but_does_not_authorize_a_plan(weight):
    value={"version":short.SELF_VERSION,"weight":weight}
    assert short.validate_valid_prior_self_guidance(value)==value
    parsed=short.validate_valid_prior_self_guidance(value);parsed["weight"]=999
    assert value["weight"]==weight


def _proof(tmp_path,purpose="baseline_training"):
    from test_chemistry_guarded_restart import _evidence, _bind
    old,gate,native,run=_evidence(tmp_path)
    p=baseline_plan(tmp_path,purpose)
    gate["starting_checkpoint_sha256"]=short.START_SHA
    gate["short_continuation"]=deepcopy(p["short_continuation"])
    for panel in gate["panels"]:
        path=Path(panel["path"]);value=json.loads(path.read_text());value["starting_checkpoint_sha256"]=short.START_SHA
        write_json(path,value);panel["sha256"]=file_sha256(path)
    rows=[]
    path=Path(gate["updates_path"])
    for text in path.read_text().splitlines():
        row=json.loads(text)
        row.update(pre_clip_gradient_norm=.1,group_gradient_norms={"encoder":.1,"decoder":.2},
            repair_statuses={"real":{"skipped_already_valid":256},"prior":{"accepted":200,"skipped_already_valid":312}},
            teacher_classification={"real":{"attempts":256},"prior":{"attempts":512}})
        rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows));gate["updates_sha256"]=file_sha256(path)
    _bind(tmp_path,run,gate,native)
    return p,gate,native,run,rows


@pytest.mark.parametrize("purpose",["baseline_training","self_guidance_test"])
def test_full2500_disposable_proof_remains_mandatory_for_both_purposes(tmp_path,purpose):
    p,gate,_,_,_=_proof(tmp_path,purpose)
    assert restart.authenticate_stability(p)==gate
    if purpose=="self_guidance_test":
        with pytest.raises(ValueError):short.require_training_branch(p)


@pytest.mark.parametrize("defect",["purpose","partial","missing_panel","classification","search_error","gradient"])
def test_rebound_hashes_cannot_hide_wrong_purpose_missing_rows_or_unhealthy_evidence(tmp_path,defect):
    from test_chemistry_guarded_restart import _bind
    p,gate,native,run,rows=_proof(tmp_path)
    if defect=="purpose":gate["short_continuation"]["purpose"]="self_guidance_test"
    elif defect=="partial":rows.pop()
    elif defect=="missing_panel":gate["panels"].pop()
    elif defect=="classification":rows[1]["teacher_classification"]["prior"]["attempts"]=511
    elif defect=="search_error":rows[1]["repair_statuses"]["prior"]={"rejected_or_error":1,"accepted":511}
    else:rows[1]["group_gradient_norms"]["decoder"]=0.
    path=Path(gate["updates_path"]);path.write_text("\n".join(json.dumps(row) for row in rows))
    gate["updates_sha256"]=file_sha256(path);_bind(tmp_path,run,gate,native)
    with pytest.raises(ValueError):restart.authenticate_stability(p)


@pytest.mark.parametrize("purpose",["baseline_training","self_guidance_test"])
def test_fixed_baseline_receipt_binds_short_purpose_objective_and_immutable36(tmp_path,monkeypatch,purpose):
    from test_chemistry_early_health import observation
    monkeypatch.setenv("LSB_JOBID","123456")
    p=baseline_plan(tmp_path,purpose)
    record,sha=health.materialize_early_health_baseline(p,observation(),completed_epoch=36,updates=237456,model_sha256="1"*64)
    assert record["short_continuation"]==p["short_continuation"]
    assert record["baseline_continuation"]==p["protocol"]["baseline_continuation"]
    assert health.require_early_health_baseline(p,sha)==record
    if purpose=="self_guidance_test":assert record["valid_prior_self_guidance"]["weight"]==.01
    p["short_continuation"]["purpose"]="self_guidance_test" if purpose=="baseline_training" else "baseline_training"
    with pytest.raises(ValueError):health.require_early_health_baseline(p,sha)


@pytest.mark.parametrize("purpose",["baseline_training","self_guidance_test"])
def test_optimizer_amendment_is_top_level_and_keeps_original_protocol_and_bounds(tmp_path,purpose):
    p=baseline_plan(tmp_path,purpose)
    old=deepcopy(p["protocol"])
    p["optimizer_step_control"]={"version":"adamw-displacement-scale-v1","factor":.25}
    assert short.validate_short_continuation(p)==p["short_continuation"]
    assert p["protocol"]==old and short.continuation_end_epoch(p)==39


def test_optimizer_amendment_cannot_escape_guarded36_or_use_an_implicit_null():
    with pytest.raises(ValueError):
        short.validate_short_continuation(dict(protocol={"total_epochs":100},
            optimizer_step_control={"version":"adamw-displacement-scale-v1","factor":.25}))
    p=baseline_plan();p["optimizer_step_control"]=None
    with pytest.raises(ValueError):short.validate_short_continuation(p)


def _quarter_proof(tmp_path):
    from test_chemistry_guarded_restart import _bind
    p,gate,native,run,rows=_proof(tmp_path)
    control={"version":"adamw-displacement-scale-v1","factor":.25}
    p["optimizer_step_control"]=control
    receipt=dict(version=control["version"],factor=.25,optimizer_steps=1,raw_displacement_norm=.1,
        applied_displacement_norm=.025,parameter_arithmetic="float64_delta_then_native_dtype",
        moment_state="one_native_adamw_step_retained")
    benchmark=dict(optimizer_step_control=control,
        native_resume_probe=dict(optimizer_step_control=control),
        native_execution_parity=dict(optimizer_step_control=control),
        updates=[dict(optimizer_step_control=deepcopy(receipt)) for _ in range(8)])
    write_json(tmp_path/"benchmark_gate.json",benchmark)
    gate.update(optimizer_step_control=control,native_benchmark_sha256=file_sha256(tmp_path/"benchmark_gate.json"))
    for row in rows:row["optimizer_step_control"]=deepcopy(receipt)
    path=Path(gate["updates_path"]);path.write_text("\n".join(json.dumps(row) for row in rows))
    gate["updates_sha256"]=file_sha256(path);_bind(tmp_path,run,gate,native)
    return p,gate,native,run,rows,benchmark


@pytest.mark.parametrize("defect",[None,"gate","benchmark","resume","parity","native_row","long_row","factor","steps"])
def test_guarded_restart_requires_same_quarter_policy_and_every_native_long_step_receipt(tmp_path,defect):
    from test_chemistry_guarded_restart import _bind
    p,gate,native,run,rows,benchmark=_quarter_proof(tmp_path)
    if defect=="gate":gate.pop("optimizer_step_control")
    elif defect=="benchmark":benchmark.pop("optimizer_step_control")
    elif defect=="resume":benchmark["native_resume_probe"].clear()
    elif defect=="parity":benchmark["native_execution_parity"].clear()
    elif defect=="native_row":benchmark["updates"][0].clear()
    elif defect=="long_row":rows[50].pop("optimizer_step_control")
    elif defect=="factor":rows[50]["optimizer_step_control"]["factor"]=1.
    elif defect=="steps":rows[50]["optimizer_step_control"]["optimizer_steps"]=2
    write_json(tmp_path/"benchmark_gate.json",benchmark)
    gate["native_benchmark_sha256"]=file_sha256(tmp_path/"benchmark_gate.json")
    path=Path(gate["updates_path"]);path.write_text("\n".join(json.dumps(row) for row in rows))
    gate["updates_sha256"]=file_sha256(path);_bind(tmp_path,run,gate,native)
    if defect is None:assert restart.authenticate_stability(p)==gate
    else:
        with pytest.raises(ValueError):restart.authenticate_stability(p)
