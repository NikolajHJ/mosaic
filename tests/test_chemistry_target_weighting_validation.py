"""Allocated-only contracts for a restored200-update target-weighting probe.

These orchestration fixtures use fake updates, not fabricated molecular passes.
The separate primitive/native CUDA fixtures establish numerical weighting.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol import chemistry_self_strength_validation as shared
from latentmol import chemistry_target_weighting_validation as validation
from latentmol import chemistry_target_weighting_policy as policy
from latentmol.experiments import file_sha256, write_json
from test_chemistry_self_guidance_iteration import iteration_plan
from test_chemistry_self_strength_validation import _completed as old_completed, _plan as old_plan


def _plan(tmp_path):
    result = iteration_plan(tmp_path)
    result.update(target_weighting_test=deepcopy(policy.POLICY),
                  target_weighting_probe=deepcopy(validation.POLICY))
    result["protocol"]["chemistry_target_weighting"] = deepcopy(policy.LOSS_POLICY)
    return result


def _generation(plan, run_dir, relative):
    path = Path(run_dir)/f"generation-{relative:04d}"/"metrics.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    value = dict(version=validation.GENERATION_VERSION,
        target_weighting_test=deepcopy(plan["target_weighting_test"]),
        chemistry_target_weighting=deepcopy(plan["protocol"]["chemistry_target_weighting"]),
        self_guidance_iteration=deepcopy(plan["self_guidance_iteration"]),
        plan_sha256=plan["_plan_sha256"],archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"],
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        completed_disposable_updates=relative,diagnostic_only=True,promotable=False,epoch_complete=False)
    write_json(path,value)
    return dict(path=str(path),sha256=file_sha256(path))


def _completed(plan, tmp_path):
    result = old_completed(plan)
    result.update(version=validation.VERSION,policy=deepcopy(validation.POLICY),completed_updates=200,
                  run_dir=str(tmp_path/"probe"),**validation.receipt_bindings(plan))
    gradient = result["gradient_observations"][0]["receipt"]
    result["gradient_observations"] = [dict(relative_update=i,receipt=deepcopy(gradient))
                                      for i in validation.POLICY["gradient_updates"]]
    result["updates"] = [dict(relative_update=i) for i in range(1,201)]
    result["panels"] = [dict(relative_update=i,status="passed") for i in validation.PANELS]
    result["generation_start"] = _generation(plan,result["run_dir"],0)
    result["generation_end"] = _generation(plan,result["run_dir"],200)
    return result


def test_new_contract_preserves_exact_historical100_and_full2500(tmp_path):
    old = old_plan(tmp_path)
    assert shared.validate_probe(old) == shared.POLICY
    assert shared.require_probe_completion(old,old_completed(old))["completed_updates"] == 100
    new = _plan(tmp_path)
    assert shared.validate_probe(new) == validation.POLICY
    assert new["guarded_restart"]["validation_updates"] == 2500
    assert validation.require_probe_completion(new,_completed(new,tmp_path))["completed_updates"] == 200
    copy = shared.validate_probe(new); copy["gradient_updates"].append(201)
    assert new["target_weighting_probe"] == validation.POLICY


@pytest.mark.parametrize("change", [
    lambda p:p.pop("target_weighting_test"),lambda p:p.update(target_weighting_test=None),
    lambda p:p["target_weighting_test"].update(version="unknown"),
    lambda p:p.update(self_strength_probe=deepcopy(shared.POLICY)),
    lambda p:p["target_weighting_probe"].update(updates=100),
    lambda p:p["target_weighting_probe"].update(updates=200.),
    lambda p:p["target_weighting_probe"].update(gradient_updates=[1,39,96,100]),
    lambda p:p["protocol"]["chemistry_target_weighting"].update(none_bond_weight=1.),
    lambda p:p["protocol"]["valid_prior_self_guidance"].update(weight=1.),
    lambda p:p["short_continuation"].update(purpose="baseline_training"),
    lambda p:p["starting_checkpoint"].update(epoch=37)])
def test_undeclared_legacy_mixed_or_changed_probe_is_rejected(change,tmp_path):
    plan=_plan(tmp_path); change(plan)
    with pytest.raises(ValueError): validation.validate_probe(plan)


@pytest.mark.parametrize("change", [
    lambda r:r.update(version=shared.VERSION),lambda r:r.update(completed_updates=100),
    lambda r:r.update(diagnostic_completed=False),lambda r:r.update(promotable=True),
    lambda r:r.update(epoch_complete=True),lambda r:r.update(training_updates_promoted=1),
    lambda r:r["restoration"]["checks"].update(rng=False),
    lambda r:r["restoration"]["checks"].pop("optimizer"),
    lambda r:r["panels"].pop(),lambda r:r["panels"][-1].update(status="failed"),
    lambda r:r["updates"].pop(),lambda r:r["gradient_observations"].pop(),
    lambda r:r["gradient_observations"][-1]["receipt"].update(component_sum_matches_native_backward=False),
    lambda r:r["gradient_observations"][-1]["receipt"].update(observer_optimizer_steps=1),
    lambda r:r["chemistry_target_weighting"].update(none_bond_weight=1.),
    lambda r:r["updates"][0].update(relative_update=True),
    lambda r:r["policy"].update(updates=200.),
    lambda r:r.update(failed_diagnostic={"promotable":False}),
    lambda r:r.pop("target_weighting_test"),lambda r:r.pop("generation_end")])
def test_partial_failed_or_unbound200_cannot_launch_longer_proof(change,tmp_path):
    plan=_plan(tmp_path); report=_completed(plan,tmp_path); change(report)
    with pytest.raises(ValueError): validation.require_probe_completion(plan,report)


@pytest.mark.parametrize("field,value", [("completed_disposable_updates",100),
    ("plan_sha256","9"*64),("diagnostic_only",False),("promotable",True),
    ("version","old"),("chemistry_target_weighting",policy.POLICY)])
def test_rehashed_generation_must_still_match_actual200_policy(field,value,tmp_path):
    plan=_plan(tmp_path); report=_completed(plan,tmp_path); receipt=report["generation_end"]
    path=Path(receipt["path"]); record=json.loads(path.read_text()); record[field]=value
    write_json(path,record); receipt["sha256"]=file_sha256(path)
    with pytest.raises(ValueError): validation.require_probe_completion(plan,report)


def test_native_probe_and_fresh2500_use_one_clock_and_never_promote(monkeypatch,tmp_path):
    import latentmol.failure_guidance_benchmark as native
    import latentmol.chemistry_guarded_validation as stability
    plan=_plan(tmp_path); events=[]
    monkeypatch.setattr(validation.time,"monotonic",lambda:123.)
    monkeypatch.setattr(native,"benchmark_candidate",lambda p:events.append(("native",p)))
    monkeypatch.setattr(validation,"run_probe",lambda p,*,allocation_started:
        events.append(("probe",allocation_started)) or _completed(p,tmp_path))
    monkeypatch.setattr(stability,"run_stability",lambda p,*,allocation_started:
        events.append(("stability",allocation_started)) or {"fixture":True})
    assert validation.run_validation(plan) == {"fixture":True}
    assert [row[0] for row in events] == ["native","probe","stability"]
    assert events[1][1] == events[2][1] == 123.


@pytest.mark.parametrize("stage", ["native","probe","restoration"])
def test_each_failed_stage_blocks_longer_proof(monkeypatch,tmp_path,stage):
    import latentmol.failure_guidance_benchmark as native
    import latentmol.chemistry_guarded_validation as stability
    plan=_plan(tmp_path); events=[]
    def benchmark(p):
        events.append("native")
        if stage=="native": raise RuntimeError("native failure")
    def probe(p,**kwargs):
        events.append("probe")
        if stage=="probe": raise RuntimeError("health failure")
        result=_completed(p,tmp_path); result["restoration"]["checks"]["rng"]=False
        return result
    monkeypatch.setattr(native,"benchmark_candidate",benchmark)
    monkeypatch.setattr(validation,"run_probe",probe)
    monkeypatch.setattr(stability,"run_stability",lambda *a,**k:events.append("stability"))
    with pytest.raises((RuntimeError,ValueError)): validation.run_validation(plan)
    assert "stability" not in events


@pytest.mark.parametrize("failure_at", [None,32,100,200])
def test_shared_loop_executes_declared_schedule_and_restores_before_return(monkeypatch,tmp_path,failure_at):
    plan=_plan(tmp_path); plan.update(release=str(tmp_path),seed=17)
    # The reader is deliberately mocked below, but its byte-hash argument is
    # evaluated first. This file is fixture input, never a readiness pass.
    write_json(tmp_path/"failure_readiness.json",{"fixture":"mocked authentication input"})
    write_json(tmp_path/"benchmark_gate.json",{"composition_sha256":"a"*64})
    write_json(tmp_path/"training_composition.json",{})
    events=[]; model=SimpleNamespace(update=0,state_dict=lambda:{"fixture":True})
    teacher=SimpleNamespace(close=lambda:events.append("close"),drain=lambda:events.append("drain"),
                            clear_cache=lambda:events.append("clear"))
    names=("model","reference","optimizer","gradients","reference_gradients","model_modes","reference_modes","rng")
    snapshot={key:"unchanged" for key in names}; optimizer=SimpleNamespace(param_groups=[{"lr":.1}])
    class Run:
        def __init__(self,root,name,*args): self.path=Path(root)/"fixture-run"; self.path.mkdir(parents=True)
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def annotate(self,**kwargs): events.append(("annotate",kwargs))
    class Observer:
        def __init__(self,*a,**k): pass
        def __enter__(self): return self
        def __exit__(self,*a): return False
        def receipt(self): return old_completed(plan)["gradient_observations"][0]["receipt"]
    def update(*a,**kwargs):
        model.update+=1
        events.append(("update",model.update,"component_observer" in kwargs))
        return {"total":1.}
    def assessment(*a,**k):
        events.append(("panel",model.update))
        return {"status":"failed" if model.update==failure_at else "passed","checks":{"fixture":True}}
    def observe(p,m,g,o,*,completed_updates):
        events.append(("generation",completed_updates)); return _generation(p,Path(o).parent,completed_updates)
    def restore(*a): events.append("restore"); model.update=0
    monkeypatch.setattr(shared,"ExperimentRun",Run)
    monkeypatch.setattr(shared,"authenticate_readiness",lambda *a:None)
    monkeypatch.setattr(shared.c,"setup",lambda *a,**k:(model,object(),optimizer,object(),{},None,{},teacher,36,237456,None,None))
    monkeypatch.setattr(shared.c,"authenticated",lambda *a:None)
    monkeypatch.setattr(shared.c,"observe_early_health",lambda *a:{"population":{},"reconstruction":{},"classification":{}})
    monkeypatch.setattr(shared.c,"assess_early_health",assessment)
    monkeypatch.setattr(shared.c,"execution_settings",lambda p:{})
    monkeypatch.setattr(shared.c,"chemistry_update",update)
    monkeypatch.setattr(shared.c,"nested_equal",lambda a,b:a==b)
    monkeypatch.setattr(shared,"_snapshot",lambda *a:deepcopy(snapshot))
    monkeypatch.setattr(shared,"_restore",restore)
    monkeypatch.setattr(shared,"state_digest",lambda *a:"1"*64)
    monkeypatch.setattr(shared,"learning_rate_at_epoch",lambda *a:.1)
    monkeypatch.setattr(shared,"_loader",lambda *a:range(2500))
    monkeypatch.setattr(shared,"ObjectiveGradientDiagnostics",Observer)
    monkeypatch.setattr(validation,"observe_probe_generation",observe)
    monkeypatch.setattr(validation,"save_failed_probe_snapshot",lambda *a,**k:
        events.append("failed_snapshot") or {"promotable":False})
    if failure_at is None:
        report=validation.run_probe(plan,allocation_started=shared.time.monotonic())
        assert report["completed_updates"]==200
    else:
        with pytest.raises(RuntimeError,match="health"):
            validation.run_probe(plan,allocation_started=shared.time.monotonic())
        report=json.loads((tmp_path/"target_weighting_probe.json").read_text())
        assert report["status"]=="failed" and report["failed_diagnostic"]["promotable"] is False
        assert events.index("failed_snapshot") < events.index("restore")
    assert model.update==0 and report["restoration"]["passed"]
    updates=[e for e in events if isinstance(e,tuple) and e[0]=="update"]
    last=200 if failure_at is None else failure_at
    assert [e[1] for e in updates]==list(range(1,last+1))
    assert [e[1] for e in updates if e[2]]==[i for i in validation.POLICY["gradient_updates"] if i<=last]
    assert [e[1] for e in events if isinstance(e,tuple) and e[0]=="panel"]==[i for i in validation.PANELS if i<=last]
    assert events[-1]=="close"


def test_generation_adapter_uses_one_existing_evaluation_and_preserves_raw_result(monkeypatch,tmp_path):
    import latentmol.chemistry_generation_comparison as generation
    plan=_plan(tmp_path); calls=[]; path=tmp_path/"metrics.json"
    def observe(p,m,g,o,*,completed_updates):
        calls.append(completed_updates)
        write_json(path,dict(plan_sha256=p["_plan_sha256"],metrics={"unfiltered_fixture":7},raw_graphs={"sha256":"d"*64}))
        return {"path":str(path),"sha256":file_sha256(path)}
    monkeypatch.setattr(generation,"_observe",observe)
    result=validation.observe_probe_generation(plan,None,None,tmp_path,completed_updates=200)
    value=json.loads(path.read_text())
    assert calls==[200] and result["sha256"]==file_sha256(path)
    assert value["metrics"]=={"unfiltered_fixture":7} and value["raw_graphs"]=={"sha256":"d"*64}
    assert value["version"]==validation.GENERATION_VERSION and value["diagnostic_only"]
    assert value["promotable"] is value["epoch_complete"] is False
    with pytest.raises(ValueError): validation.observe_probe_generation(plan,None,None,tmp_path,completed_updates=199)


@pytest.mark.parametrize("attempted", [237556,237557])
def test_failed_snapshot_counter_labels_last_completed_not_partial_state(monkeypatch,tmp_path,attempted):
    import latentmol.chemistry_early_health as health
    plan=_plan(tmp_path); calls=[]; health_path=tmp_path/"failed-report.json"
    write_json(health_path,{"status":"failed"})
    def save(path,*args,**kwargs):
        calls.append(kwargs); Path(path).write_bytes(b"fake nonpromotable diagnostic")
        return file_sha256(path)
    monkeypatch.setattr(health,"save_failed_early_health_snapshot",save)
    receipt=validation.save_failed_probe_snapshot(plan,SimpleNamespace(path=tmp_path),None,None,None,{},
        last_completed_update=237556,attempted_update=attempted,health_path=health_path)
    assert calls[0]["updates"]==receipt["last_completed_update"]==237556
    assert receipt["attempted_update"]==attempted
    assert receipt["state_may_include_partial_attempted_update"]==(attempted>237556)
    assert "not a partial-state completion certificate" in receipt["snapshot_updates_field_scope"]
    assert receipt["promotable"] is receipt["epoch_complete"] is False
