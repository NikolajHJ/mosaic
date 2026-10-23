"""Allocated orchestration contracts; no fake update is a molecular pass."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol import chemistry_valid_reference_validation as validation
from latentmol import chemistry_valid_reference_policy as policy
from latentmol import chemistry_self_strength_validation as shared
from latentmol.experiments import file_sha256, write_json
from test_chemistry_valid_reference_policy import valid_reference_plan
from test_chemistry_self_strength_validation import _plan as old_plan, _completed as old_completed
from test_chemistry_target_weighting_validation import _plan as weighted_plan, _completed as weighted_completed


def _plan(tmp_path):
    return dict(valid_reference_plan(tmp_path),release=str(tmp_path),seed=17)


def _anchor(plan):
    return dict(reference_epoch=36,reference_sha256=plan["starting_checkpoint"]["sha256"],
                model_sha256="1"*64,frozen=True,eval_mode=True)


def _gradient():
    return dict(version=shared.GRADIENT_VERSION,status="observed_not_readiness",
        complete_logical_update=True,component_sum_matches_native_backward=True,
        real_points=256,prior_points=512,microbatch_size=128,observed_microbatches={"real":2,"prior":4},
        weighted_terms=list(validation.WEIGHTED_TERMS),observer_model_forwards=0,
        observer_optimizer_steps=0,observer_random_samples=0)


def _metrics(plan):
    anchor=_anchor(plan)
    return dict(total=1.,valid_prior_reference_nll=.5,valid_prior_reference_weighted=.05,
        ordered_teacher_receipt_chain_sha256="3"*64,
        valid_prior_reference_guidance=dict(policy=deepcopy(plan["protocol"]["valid_prior_reference_guidance"]),
            reference_identity=dict(epoch=36,checkpoint_sha256=anchor["reference_sha256"],model_sha256=anchor["model_sha256"]),
            attempted=512,eligible=317,excluded=195,reference_valid_live_invalid=12,
            target_receipt_chain_sha256="4"*64,scope="Fixture target chain, not a molecular pass"))


def _generation(plan, root, relative):
    path=Path(root)/f"generation-{relative:04d}"/"metrics.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(version=validation.GENERATION_VERSION,**validation.receipt_bindings(plan),
        plan_sha256=plan["_plan_sha256"],archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"],starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        completed_disposable_updates=relative,prior_count=1024,categorical_samples_per_latent=4,
        evaluation_seed=plan["protocol"]["evaluation_seed"],valid_reference_anchor=_anchor(plan),
        diagnostic_only=True,promotable=False,epoch_complete=False)
    write_json(path,value)
    return dict(path=str(path),sha256=file_sha256(path))


def _completed(plan,tmp_path):
    report=dict(version=validation.VERSION,status="passed_disposable_probe_not_training",diagnostic_completed=True,
        **validation.receipt_bindings(plan),plan_sha256=plan["_plan_sha256"],archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"],starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        starting_model_sha256=_anchor(plan)["model_sha256"],
        optimizer_step_control=deepcopy(plan["optimizer_step_control"]),policy=deepcopy(validation.POLICY),
        completed_updates=200,promotable=False,epoch_complete=False,training_updates_promoted=0,
        restoration=dict(passed=True,checks={key:True for key in ("model","reference","optimizer","gradients",
            "reference_gradients","model_modes","reference_modes","rng")}),
        panels=[dict(relative_update=i,status="passed") for i in validation.PANELS],
        gradient_observations=[dict(relative_update=i,receipt=_gradient()) for i in validation.POLICY["gradient_updates"]],
        updates=[dict(relative_update=i,metrics=_metrics(plan)) for i in range(1,201)],run_dir=str(tmp_path/"probe"),
        valid_reference_anchor=dict(before=_anchor(plan),after=_anchor(plan),unchanged=True))
    report["generation_start"]=_generation(plan,report["run_dir"],0)
    report["generation_end"]=_generation(plan,report["run_dir"],200)
    return report


def test_closed_reference200_contract_and_historical100_and_weighted200_unchanged(tmp_path):
    plan=_plan(tmp_path)
    assert shared.validate_probe(plan)==validation.POLICY==policy.PROBE_POLICY
    assert shared.require_probe_completion(plan,_completed(plan,tmp_path))["completed_updates"]==200
    assert plan["guarded_restart"]["validation_updates"]==2500
    legacy=old_plan(tmp_path)
    assert shared.require_probe_completion(legacy,old_completed(legacy))["completed_updates"]==100
    weighted=weighted_plan(tmp_path)
    assert shared.require_probe_completion(weighted,weighted_completed(weighted,tmp_path))["completed_updates"]==200
    result=validation.validate_probe(plan); result["gradient_updates"].append(201)
    assert plan["valid_reference_probe"]==policy.PROBE_POLICY


@pytest.mark.parametrize("change", [
    lambda p:p.pop("valid_reference_test"),lambda p:p.update(valid_reference_test=None),
    lambda p:p["valid_reference_test"].update(reference_epoch=35),
    lambda p:p["valid_reference_test"].update(reference_sha256="9"*64),
    lambda p:p["valid_reference_probe"].update(updates=100),
    lambda p:p["valid_reference_probe"].update(updates=200.),
    lambda p:p["valid_reference_probe"].update(gradient_updates=[1,39,96,100]),
    lambda p:p["protocol"]["valid_prior_reference_guidance"].update(weight=.01),
    lambda p:p.update(self_guidance_iteration=None),lambda p:p.update(target_weighting_test=None),
    lambda p:p["protocol"].update(valid_prior_self_guidance=None),
    lambda p:p["protocol"].update(chemistry_target_weighting=None),
    lambda p:p["short_continuation"].update(purpose="baseline_training"),
    lambda p:p["starting_checkpoint"].update(sha256="9"*64),
    lambda p:p["starting_checkpoint"].update(epoch=37)])
def test_changed_or_mixed_contract_cannot_run(change,tmp_path):
    plan=_plan(tmp_path);change(plan)
    with pytest.raises(ValueError):validation.validate_probe(plan)


@pytest.mark.parametrize("change", [
    lambda r:r.update(status="failed"),lambda r:r.update(completed_updates=100),
    lambda r:r.update(epoch_complete=True),lambda r:r.update(training_updates_promoted=1),
    lambda r:r["restoration"]["checks"].pop("optimizer"),
    lambda r:r["restoration"]["checks"].update(rng=False),lambda r:r["panels"].pop(),
    lambda r:r["panels"][-1].update(status="failed"),lambda r:r["gradient_observations"].pop(),
    lambda r:r["gradient_observations"][-1]["receipt"].update(weighted_terms=["composition","kl","occupancy","reconstruction","repair","valid_self"]),
    lambda r:r["gradient_observations"][0]["receipt"].update(observer_model_forwards=1),
    lambda r:r["updates"][0].update(relative_update=True),lambda r:r["policy"].update(updates=200.),
    lambda r:r["valid_reference_anchor"]["after"].update(model_sha256="2"*64),
    lambda r:r.update(starting_model_sha256="2"*64),
    lambda r:r["valid_reference_anchor"]["after"].update(frozen=False),
    lambda r:r["valid_reference_anchor"].update(unchanged=False),
    lambda r:r["valid_reference_anchor"].pop("after"),lambda r:r.pop("valid_reference_test"),
    lambda r:r.update(self_guidance_iteration=None),lambda r:r.pop("generation_end"),
    lambda r:r.update(failed_diagnostic={"promotable":False})])
def test_partial_failed_or_wrong_reference_receipt_blocks_fullproof(change,tmp_path):
    plan=_plan(tmp_path);report=_completed(plan,tmp_path);change(report)
    with pytest.raises(ValueError):validation.require_probe_completion(plan,report)


@pytest.mark.parametrize("change", [lambda m:m.pop("valid_prior_reference_guidance"),
    lambda m:m["valid_prior_reference_guidance"].update(policy={"version":"wrong","weight":.1}),
    lambda m:m["valid_prior_reference_guidance"]["reference_identity"].update(epoch=35),
    lambda m:m["valid_prior_reference_guidance"]["reference_identity"].update(epoch=36.),
    lambda m:m["valid_prior_reference_guidance"]["reference_identity"].update(model_sha256="2"*64),
    lambda m:m["valid_prior_reference_guidance"].update(attempted=256),
    lambda m:m["valid_prior_reference_guidance"].update(eligible=True),
    lambda m:m["valid_prior_reference_guidance"].update(excluded=194),
    lambda m:m["valid_prior_reference_guidance"].update(reference_valid_live_invalid=318),
    lambda m:m["valid_prior_reference_guidance"].update(target_receipt_chain_sha256=None),
    lambda m:m.pop("ordered_teacher_receipt_chain_sha256"),
    lambda m:m.update(valid_prior_reference_nll=float("nan")),
    lambda m:m.update(valid_prior_reference_weighted=-1.),lambda m:m.update(valid_prior_self_guidance={})])
def test_every_update_requires_same_reference_policy_and_allrow_chain(change,tmp_path):
    plan=_plan(tmp_path);report=_completed(plan,tmp_path);change(report["updates"][198]["metrics"])
    with pytest.raises(ValueError):validation.require_probe_completion(plan,report)


def test_excluded_reference_invalid_rows_remain_explicit_not_missing_receipts(tmp_path):
    plan=_plan(tmp_path);report=_completed(plan,tmp_path)
    metrics=report["updates"][0]["metrics"]
    metrics["valid_prior_reference_guidance"].update(eligible=0,excluded=512,reference_valid_live_invalid=0)
    metrics.update(valid_prior_reference_nll=0.,valid_prior_reference_weighted=0.)
    assert validation.require_probe_completion(plan,report)["completed_updates"]==200


@pytest.mark.parametrize("failure",["none","anchor","counts","chain"])
def test_public_update_reader_reuses_exact_native_probe_and_stability_contract(tmp_path,failure):
    plan=_plan(tmp_path);metrics=_metrics(plan);anchor=_anchor(plan)
    if failure=="anchor":anchor["reference_sha256"]="2"*64
    elif failure=="counts":metrics["valid_prior_reference_guidance"]["attempted"]=511
    elif failure=="chain":metrics["valid_prior_reference_guidance"]["target_receipt_chain_sha256"]="invalid"
    if failure=="none":
        original=deepcopy(metrics)
        validation.require_reference_update(plan,metrics,anchor)
        assert metrics==original
    else:
        with pytest.raises(ValueError):validation.require_reference_update(plan,metrics,anchor)


@pytest.mark.parametrize("change", [lambda p:p.update(reference_epoch=35),
    lambda p:p.update(reference_sha256="2"*64),lambda p:p.update(model_sha256="not-a-sha"),
    lambda p:p.update(frozen=False),lambda p:p.update(eval_mode=False),lambda p:p.update(extra=True)])
def test_anchor_boundary_authenticates_exact_source_and_actual_frozen_state(change,tmp_path):
    plan=_plan(tmp_path);identity=_anchor(plan);change(identity)
    teacher=SimpleNamespace(valid_reference_anchor=SimpleNamespace(assert_unchanged=lambda:identity))
    with pytest.raises(ValueError):validation.observe_anchor_identity(plan,teacher)
    with pytest.raises(ValueError):validation.observe_anchor_identity(plan,SimpleNamespace())


@pytest.mark.parametrize("field,value", [("completed_disposable_updates",100),
    ("plan_sha256","2"*64),("valid_reference_test",dict(policy.POLICY,reference_epoch=35)),
    ("prior_count",256),("categorical_samples_per_latent",1),("evaluation_seed",1),
    ("diagnostic_only",False),("self_guidance_iteration",None),("valid_reference_anchor",{} )])
def test_even_rehashed_generation_must_bind_same_reference_and200_endpoint(field,value,tmp_path):
    plan=_plan(tmp_path);report=_completed(plan,tmp_path);r=report["generation_end"];path=Path(r["path"])
    saved=json.loads(path.read_text());saved[field]=value;write_json(path,saved);r["sha256"]=file_sha256(path)
    with pytest.raises(ValueError):validation.require_probe_completion(plan,report)


def test_native_probe_restore_and_fresh2500_share_one_clock(monkeypatch,tmp_path):
    import latentmol.failure_guidance_benchmark as native
    import latentmol.chemistry_guarded_validation as stability
    plan=_plan(tmp_path);events=[]
    monkeypatch.setattr(validation.time,"monotonic",lambda:123.)
    monkeypatch.setattr(native,"benchmark_candidate",lambda p:events.append(("native",p)))
    monkeypatch.setattr(validation,"run_probe",lambda p,*,allocation_started:
        events.append(("probe",allocation_started)) or _completed(p,tmp_path))
    monkeypatch.setattr(stability,"run_stability",lambda p,*,allocation_started:
        events.append(("stability",allocation_started)) or {"fixture":True})
    assert validation.run_validation(plan)=={"fixture":True}
    assert [e[0] for e in events]==["native","probe","stability"]
    assert events[1][1]==events[2][1]==123.


@pytest.mark.parametrize("stage",["native","probe","anchor"])
def test_failed_stage_never_runs_later_proof(monkeypatch,tmp_path,stage):
    import latentmol.failure_guidance_benchmark as native
    import latentmol.chemistry_guarded_validation as stability
    plan=_plan(tmp_path);events=[]
    def benchmark(p):
        events.append("native")
        if stage=="native":raise RuntimeError("native failure")
    def probe(p,**kwargs):
        events.append("probe")
        if stage=="probe":raise RuntimeError("health failure")
        result=_completed(p,tmp_path);result["valid_reference_anchor"]["unchanged"]=False
        return result
    monkeypatch.setattr(native,"benchmark_candidate",benchmark)
    monkeypatch.setattr(validation,"run_probe",probe)
    monkeypatch.setattr(stability,"run_stability",lambda *a,**k:events.append("stability"))
    with pytest.raises((RuntimeError,ValueError)):validation.run_validation(plan)
    assert "stability" not in events


@pytest.mark.parametrize("failure",[None,32,100,200,"anchor","update"])
def test_real_shared_loop_schedule_readiness_and_restoration_with_fake_updates(monkeypatch,tmp_path,failure):
    plan=_plan(tmp_path)
    for name,value in (("failure_readiness.json",{"fixture":"mocked authentication"}),
                       ("benchmark_gate.json",{"composition_sha256":"a"*64}),("training_composition.json",{})):
        write_json(tmp_path/name,value)
    events=[];model=SimpleNamespace(update=0,state_dict=lambda:{"fixture":True})
    calls=[0]
    def unchanged():
        calls[0]+=1;events.append(("anchor",calls[0]));result=_anchor(plan)
        if failure=="anchor" and calls[0]>1:result["model_sha256"]="2"*64
        return result
    teacher=SimpleNamespace(close=lambda:events.append("close"),drain=lambda:events.append("drain"),
        clear_cache=lambda:events.append("clear"),valid_reference_anchor=SimpleNamespace(assert_unchanged=unchanged))
    optimizer=SimpleNamespace(param_groups=[{"lr":.001}]);names=("model","reference","optimizer","gradients",
        "reference_gradients","model_modes","reference_modes","rng");snapshot={k:"unchanged" for k in names}
    class Run:
        def __init__(self,root,name,*a):self.path=Path(root)/"fixture-run";self.path.mkdir(parents=True)
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def annotate(self,**kwargs):events.append(("annotate",kwargs))
    class Observer:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def receipt(self):return _gradient()
    def setup(*a,**k):
        events.append("setup");return model,object(),optimizer,object(),{},None,{},teacher,36,237456,None,None
    def update(*a,**kwargs):
        model.update+=1;events.append(("update",model.update,"component_observer" in kwargs))
        if failure=="update" and model.update==39:raise RuntimeError("partial update fixture")
        return _metrics(plan)
    def assess(*a,**k):
        events.append(("panel",model.update));return {"status":"failed" if failure==model.update else "passed"}
    def restore(*a):events.append("restore");model.update=0
    def generation(p,m,g,o,*,completed_updates):return _generation(p,Path(o).parent,completed_updates)
    monkeypatch.setattr(shared,"ExperimentRun",Run)
    monkeypatch.setattr(shared,"authenticate_readiness",lambda *a:events.append("readiness"))
    monkeypatch.setattr(shared.c,"setup",setup)
    monkeypatch.setattr(shared.c,"authenticated",lambda *a:None)
    monkeypatch.setattr(shared.c,"observe_early_health",lambda *a:{"population":{},"reconstruction":{},"classification":{}})
    monkeypatch.setattr(shared.c,"assess_early_health",assess)
    monkeypatch.setattr(shared.c,"execution_settings",lambda p:{})
    monkeypatch.setattr(shared.c,"chemistry_update",update)
    monkeypatch.setattr(shared.c,"nested_equal",lambda a,b:a==b)
    monkeypatch.setattr(shared,"_snapshot",lambda *a:deepcopy(snapshot))
    monkeypatch.setattr(shared,"_restore",restore)
    monkeypatch.setattr(shared,"state_digest",lambda *a:"1"*64)
    monkeypatch.setattr(shared,"learning_rate_at_epoch",lambda *a:.001)
    monkeypatch.setattr(shared,"_loader",lambda *a:range(2500))
    monkeypatch.setattr(shared,"ObjectiveGradientDiagnostics",Observer)
    monkeypatch.setattr(validation,"observe_probe_generation",generation)
    monkeypatch.setattr(validation,"save_failed_probe_snapshot",lambda *a,**k:
        events.append(("failed_snapshot",k)) or {"promotable":False})
    if failure is None:
        report=validation.run_probe(plan,allocation_started=shared.time.monotonic())
        assert report["completed_updates"]==200 and report["valid_reference_anchor"]["unchanged"]
    else:
        with pytest.raises((RuntimeError,ValueError)):
            validation.run_probe(plan,allocation_started=shared.time.monotonic())
        report=json.loads((tmp_path/"valid_reference_probe.json").read_text())
        assert report["status"]=="failed"
        if failure!="anchor":
            saved=next(e for e in events if isinstance(e,tuple) and e[0]=="failed_snapshot")
            assert events.index(saved)<events.index("restore")
            if failure=="update":
                assert saved[1]["last_completed_update"]==237494 and saved[1]["attempted_update"]==237495
    assert events.index("readiness")<events.index("setup")
    assert model.update==0 and report["restoration"]["passed"] and events[-1]=="close"
    last=39 if failure=="update" else (failure if type(failure) is int else 200)
    records=[e for e in events if isinstance(e,tuple) and e[0]=="update"]
    assert [e[1] for e in records]==list(range(1,last+1))
    assert [e[1] for e in records if e[2]]==[i for i in validation.POLICY["gradient_updates"] if i<=last]
    panels=[i for i in validation.PANELS if i<=last and not(failure=="update" and i==last)]
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=="panel"]==panels


def test_generation_adapter_preserves_existing_raw_evaluation(monkeypatch,tmp_path):
    import latentmol.chemistry_generation_comparison as generation
    plan=_plan(tmp_path);calls=[];path=tmp_path/"metrics.json"
    def observe(p,m,g,o,*,completed_updates):
        calls.append(completed_updates);write_json(path,{"metrics":{"unfiltered_fixture":7},"raw_graphs":{"sha256":"d"*64}})
        return {"path":str(path),"sha256":file_sha256(path)}
    monkeypatch.setattr(generation,"_observe",observe)
    receipt=validation.observe_probe_generation(plan,None,None,tmp_path,completed_updates=200)
    result=json.loads(path.read_text())
    assert calls==[200] and receipt["sha256"]==file_sha256(path)
    assert result["metrics"]=={"unfiltered_fixture":7} and result["raw_graphs"]=={"sha256":"d"*64}
    assert result["version"]==validation.GENERATION_VERSION and result["diagnostic_only"]
    assert result["promotable"] is result["epoch_complete"] is False
    with pytest.raises(ValueError):validation.observe_probe_generation(plan,None,None,tmp_path,completed_updates=199)


def test_failed_diagnostic_counter_is_last_completed_and_reference_bound(monkeypatch,tmp_path):
    import latentmol.chemistry_early_health as health
    plan=_plan(tmp_path);path=tmp_path/"failure.json";write_json(path,{"status":"failed"});calls=[]
    def save(out,*a,**kwargs):calls.append(kwargs);Path(out).write_bytes(b"fake diagnostic");return file_sha256(out)
    monkeypatch.setattr(health,"save_failed_early_health_snapshot",save)
    receipt=validation.save_failed_probe_snapshot(plan,SimpleNamespace(path=tmp_path),None,None,None,{},
        last_completed_update=237494,attempted_update=237495,health_path=path)
    assert calls[0]["updates"]==receipt["last_completed_update"]==237494
    assert receipt["attempted_update"]==237495 and receipt["state_may_include_partial_attempted_update"]
    assert receipt["valid_reference_test"]==policy.POLICY and not receipt["promotable"] and not receipt["epoch_complete"]
