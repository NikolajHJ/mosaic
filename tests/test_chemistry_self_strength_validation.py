"""Allocated contracts for one-clock, disposable coefficient validation."""
from copy import deepcopy
import hashlib
import json

import pytest

from latentmol import chemistry_self_strength_validation as validation
from latentmol import chemistry_generation_comparison as generation
from test_chemistry_self_guidance_iteration import iteration_plan


def _plan(tmp_path=None):
    return dict(iteration_plan(tmp_path),self_strength_probe=deepcopy(validation.POLICY),
                _plan_sha256="f"*64)


def _completed(plan):
    gradient = dict(version=validation.GRADIENT_VERSION,status="observed_not_readiness",
        complete_logical_update=True,component_sum_matches_native_backward=True,
        real_points=256,prior_points=512,microbatch_size=128,observed_microbatches=dict(real=2,prior=4),
        weighted_terms=["composition","kl","occupancy","reconstruction","repair","valid_self"],
        observer_model_forwards=0,observer_optimizer_steps=0,observer_random_samples=0)
    return dict(version=validation.VERSION,status="passed_disposable_probe_not_training",diagnostic_completed=True,
        self_guidance_iteration=plan["self_guidance_iteration"],plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"],manifest_sha256=plan["manifest_sha256"],
        optimizer_step_control=deepcopy(plan["optimizer_step_control"]),policy=deepcopy(validation.POLICY),
        starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],completed_updates=100,
        promotable=False,epoch_complete=False,training_updates_promoted=0,
        restoration=dict(passed=True,checks={key:True for key in ("model","reference","optimizer","gradients",
             "reference_gradients","model_modes","reference_modes","rng")}),
        panels=[dict(relative_update=i,status="passed") for i in (0,32,100)],
        gradient_observations=[dict(relative_update=i,receipt=deepcopy(gradient)) for i in (1,39,96,100)],
        updates=[dict(relative_update=i) for i in range(1,101)])


def test_probe_scope_and_unchanged_full2500_budget(tmp_path):
    plan = _plan(tmp_path)
    value = validation.validate_probe(plan)
    value["gradient_updates"].append(9)
    assert plan["self_strength_probe"] == validation.POLICY
    assert plan["guarded_restart"]["validation_updates"] == 2500


@pytest.mark.parametrize("change", [lambda p:p.pop("self_guidance_iteration"),
    lambda p:p["short_continuation"].update(purpose="baseline_training"),
    lambda p:p["self_strength_probe"].update(updates=99),
    lambda p:p["self_strength_probe"].update(updates=True),
    lambda p:p["self_strength_probe"].update(gradient_updates=[1,39,96]),
    lambda p:p["self_strength_probe"].update(allow_train=False),
    lambda p:p["starting_checkpoint"].update(epoch=37)])
def test_probe_cannot_expand_or_silently_change_scope(change,tmp_path):
    plan=_plan(tmp_path);change(plan)
    with pytest.raises(ValueError):validation.validate_probe(plan)


@pytest.mark.parametrize("change", [lambda r:r.update(status="failed"),
    lambda r:r.update(completed_updates=99),lambda r:r.update(completed_updates=True),
    lambda r:r.update(diagnostic_completed=False),lambda r:r.update(promotable=True),
    lambda r:r.update(epoch_complete=True),lambda r:r.update(training_updates_promoted=1),
    lambda r:r.update(plan_sha256="a"*64),lambda r:r["restoration"].update(passed=False),
    lambda r:r["restoration"]["checks"].update(rng=False),
    lambda r:r["restoration"]["checks"].pop("optimizer"),
    lambda r:r["panels"][-1].update(status="failed"),
    lambda r:r["gradient_observations"].pop(),lambda r:r["updates"].pop()])
def test_partial_failed_or_changed_probe_never_allows_longer_proof(change,tmp_path):
    plan=_plan(tmp_path);report=_completed(plan);change(report)
    with pytest.raises(ValueError):validation.require_probe_completion(plan,report)


@pytest.mark.parametrize("defect",["absent","version","partial","backward","self","counts","forward","step","policy"])
def test_missing_or_unbound_component_receipts_block_longer_proof(defect,tmp_path):
    plan=_plan(tmp_path);report=_completed(plan);receipt=report["gradient_observations"][0]["receipt"]
    if defect=="absent":report["gradient_observations"][0].pop("receipt")
    elif defect=="version":receipt["version"]="unknown"
    elif defect=="partial":receipt["complete_logical_update"]=False
    elif defect=="backward":receipt["component_sum_matches_native_backward"]=False
    elif defect=="self":receipt["weighted_terms"].remove("valid_self")
    elif defect=="counts":receipt["observed_microbatches"]["prior"]=1
    elif defect=="forward":receipt["observer_model_forwards"]=1
    elif defect=="step":report["optimizer_step_control"]["factor"]=1.
    elif defect=="policy":report["policy"]["updates"]=99
    with pytest.raises(ValueError):validation.require_probe_completion(plan,report)


def test_native_probe_and_longer_proof_share_one_clock_and_fresh_restoration(monkeypatch,tmp_path):
    plan=_plan(tmp_path);events=[]
    monkeypatch.setattr(validation.time,"monotonic",lambda:123.)
    monkeypatch.setattr(validation,"benchmark_candidate",lambda p:events.append(("native",p)))
    def probe(p,*,allocation_started):
        events.append(("probe",allocation_started));return _completed(p)
    def proof(p,*,allocation_started):
        events.append(("stability",allocation_started));return {"status":"fixture"}
    monkeypatch.setattr(validation,"run_probe",probe);monkeypatch.setattr(validation,"run_stability",proof)
    assert validation.run_validation(plan)=={"status":"fixture"}
    assert [row[0] for row in events]==["native","probe","stability"]
    assert events[1][1]==events[2][1]==123.


@pytest.mark.parametrize("stage",["native","probe","invalid_return"])
def test_failed_stage_blocks_later_stages_without_training(monkeypatch,tmp_path,stage):
    plan=_plan(tmp_path);events=[]
    def native(p):
        events.append("native")
        if stage=="native":raise RuntimeError("native failed")
    def probe(p,**kwargs):
        events.append("probe")
        if stage=="probe":raise RuntimeError("probe failed")
        return dict(_completed(p),completed_updates=99)
    monkeypatch.setattr(validation,"benchmark_candidate",native)
    monkeypatch.setattr(validation,"run_probe",probe)
    monkeypatch.setattr(validation,"run_stability",lambda *a,**k:events.append("stability"))
    with pytest.raises((RuntimeError,ValueError)):validation.run_validation(plan)
    assert "stability" not in events


def test_probe_deadline_charges_all_earlier_allocation_work(monkeypatch,tmp_path):
    plan=_plan(tmp_path)
    monkeypatch.setattr(validation.time,"monotonic",lambda:80000.)
    assert validation._clock_deadline(plan,100.)==82000.
    with pytest.raises(ValueError):validation._clock_deadline(plan,80001.)


def test_probe_generation_reuses_evaluator_and_labels_actual_disposable_point(monkeypatch,tmp_path):
    plan=_plan(tmp_path);path=tmp_path/"metrics.json";calls=[]
    def observe(p,m,g,o,*,completed_updates):
        calls.append(completed_updates)
        path.write_text(json.dumps(dict(version=generation.VERSION,completed_disposable_updates=completed_updates,
            metrics=dict(categorical=dict(valid_count=2,attempts=4)),raw_graphs={"sha256":"d"*64})))
        return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(generation,"_observe",observe)
    receipt=generation.observe_strength_generation(plan,None,None,tmp_path,completed_updates=100)
    result=json.loads(path.read_text())
    assert calls==[100] and result["version"]==generation.DIAGNOSTIC_VERSION
    assert result["diagnostic_only"] and not result["promotable"] and not result["epoch_complete"]
    assert result["metrics"]["categorical"]["valid_count"]==2
    assert receipt["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):generation.observe_strength_generation(plan,None,None,tmp_path,completed_updates=99)
