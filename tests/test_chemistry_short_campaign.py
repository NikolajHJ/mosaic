"""Allocated-only controller fixtures: disposable weak arm never trains itself."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/hpc"))
import chemistry_short_campaign as controller
from test_chemistry_short_continuation import baseline_plan


def fixture(monkeypatch,tmp_path,*,change=None):
    work=tmp_path/"work";submissions=work/"submissions";submissions.mkdir(parents=True)
    release=work/"releases"/"fixture";release.mkdir(parents=True)
    plan=baseline_plan(work/"runs"/"baseline")
    plan.update(work_root=str(work),release=str(release),loss_v1_reference={"module":str(release/"reference.py"),"sha256":"e"*64})
    test=deepcopy(plan);test["campaign"]=str(work/"runs"/"self-test")
    test["short_continuation"]["purpose"]="self_guidance_test"
    test["protocol"]["valid_prior_self_guidance"]={"version":"valid-prior-self-nll-v1","weight":.01}
    if change:change(plan,test)
    testpath=submissions/"test.json";testpath.write_text(json.dumps(test))
    testsha=hashlib.sha256(testpath.read_bytes()).hexdigest()
    plan["self_test_plan"]={"path":str(testpath),"sha256":testsha}
    path=submissions/"baseline.json";path.write_text(json.dumps(plan))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    def parent_init(self,path,sha):
        self.path=Path(path);self.sha=sha;self.plan=json.loads(self.path.read_text())
        self.work=work;self.release=release;self.root=Path(self.plan["campaign"])
        self.root.mkdir(parents=True,exist_ok=True)
        self.receipts=self.root/"receipts";self.receipts.mkdir()
        self.identity={"plan_sha256":sha,"archive_sha256":plan["archive_sha256"],
            "manifest_sha256":plan["manifest_sha256"],"parent_sha256":plan["parent_checkpoint_sha256"]}
    monkeypatch.setattr(controller.GuardedRestart,"__init__",parent_init)
    campaign=controller.ShortCampaign(path,sha)
    events=[];queued=[]
    campaign.require_cpu=lambda:events.append("cpu")
    campaign.require_predecessor_stopped=lambda:events.append("predecessor")
    campaign.complete=lambda stage:events.append(("complete",stage)) or "1000"
    campaign.worker=lambda command,label:command
    def queue(stage,command,**options):
        events.append(("queue",stage));queued.append((stage,command,options));return str(2000+len(queued))
    campaign.queue=queue
    campaign.receipt_job=lambda stage:"2001" if stage=="self-test" else "1000"
    (campaign.root/"failure_readiness.json").write_text('{"fixture":"baseline"}')
    def native(plan,sha):
        assert plan["campaign"]==campaign.plan["campaign"]
        assert plan["short_continuation"]["purpose"]=="baseline_training"
        events.append("native");return {"status":"passed"}
    def stability(plan):
        assert plan["campaign"]==campaign.plan["campaign"]
        assert plan["starting_checkpoint"]["sha256"]==campaign.plan["starting_checkpoint"]["sha256"]
        events.append("stability");return {"status":"passed"}
    monkeypatch.setattr(controller,"authenticate_readiness",native)
    monkeypatch.setattr(controller,"authenticate_stability",stability)
    monkeypatch.setattr(controller,"quota_receipt",lambda minimum:{"minimum_gib":minimum,"fixture":"no live quota query"})
    return campaign,events,queued


def test_only_baseline_purpose_can_own_controller(monkeypatch,tmp_path):
    def change(plan,test):
        plan["short_continuation"]["purpose"]="self_guidance_test"
        plan["protocol"]["valid_prior_self_guidance"]={"version":"valid-prior-self-nll-v1","weight":.01}
    with pytest.raises(ValueError,match="Only the original-loss baseline"):
        fixture(monkeypatch,tmp_path,change=change)


@pytest.mark.parametrize("change",[
    lambda p,t:t["short_continuation"].update(purpose="baseline_training"),
    lambda p,t:t.update(campaign=p["campaign"]),
    lambda p,t:t.update(archive_sha256="e"*64),
    lambda p,t:t.update(manifest_sha256="e"*64),
    lambda p,t:t.update(release=str(Path(p["release"])/"other")),
    lambda p,t:t["protocol"].update(repair_ramp_epochs=1),
    lambda p,t:t["protocol"]["valid_prior_self_guidance"].update(weight=.02),
    lambda p,t:t["starting_checkpoint"].update(sha256="e"*64),
    lambda p,t:t.update(optimizer_step_control={"version":"adamw-displacement-scale-v1","factor":.25}),
])
def test_independent_test_plan_cannot_change_inputs_or_become_training(monkeypatch,tmp_path,change):
    with pytest.raises(ValueError):fixture(monkeypatch,tmp_path,change=change)


def test_baseline_all_checks_precede_separate_24hour_disposable_test(monkeypatch,tmp_path):
    c,events,queued=fixture(monkeypatch,tmp_path)
    result=c.advance("benchmark")
    assert events[:5]==["cpu","predecessor",("complete","benchmark"),"native","stability"]
    assert [r[0] for r in queued]==["self-test","after-self-test"]
    assert queued[0][2]=={"gpu":True,"limit":"24:00"}
    assert queued[1][2]=={"dependency":"ended(2001)"}
    native=queued[0][1][-1]
    assert str(c.test_path) in native and c.test_sha in native
    assert "chemistry_guarded_validation" in native and " train " not in native
    assert result["status"]=="self_test_submitted_not_passed" and result["test_weights_disposable"]
    assert c.advance("benchmark")==result and len(queued)==2


@pytest.mark.parametrize("gate",["cpu","predecessor","completion","native","stability"])
def test_any_baseline_failure_blocks_test_and_training_and_is_retained(monkeypatch,tmp_path,gate):
    c,events,queued=fixture(monkeypatch,tmp_path)
    def fail(*args,**kwargs):raise ValueError("Injected baseline "+gate)
    if gate=="cpu":c.require_cpu=fail
    elif gate=="predecessor":c.require_predecessor_stopped=fail
    elif gate=="completion":c.complete=fail
    elif gate=="native":monkeypatch.setattr(controller,"authenticate_readiness",fail)
    else:monkeypatch.setattr(controller,"authenticate_stability",fail)
    with pytest.raises(ValueError):c.advance("benchmark")
    assert not queued
    saved=json.loads((c.root/"self-test-decision.json").read_text())
    assert saved["status"]=="stopped" and "Injected baseline" in saved["error"]
    assert c.advance("benchmark")==saved and not queued


@pytest.mark.parametrize("block",["active","quota"])
def test_active_phase_or_quota_failure_blocks_the_disposable_test(monkeypatch,tmp_path,block):
    c,_,queued=fixture(monkeypatch,tmp_path)
    if block=="active":(c.root/"active-phase.json").write_text('{"status":"running"}')
    else:
        def no_quota(minimum):raise ValueError("Insufficient work3 quota fixture")
        monkeypatch.setattr(controller,"quota_receipt",no_quota)
    with pytest.raises(ValueError):c.advance("benchmark")
    assert not queued and not Path(c.test["campaign"]).exists()


def _terminal_fixture(monkeypatch,tmp_path,state="DONE",returncode=0,stderr=""):
    c,events,queued=fixture(monkeypatch,tmp_path)
    c.advance("benchmark")
    queries=[];promotions=[]
    def query(command,**kwargs):
        assert command==["bjobs","-a","-noheader","-o","stat","2001"]
        queries.append(command);return SimpleNamespace(returncode=returncode,stdout=state,stderr=stderr)
    monkeypatch.setattr(controller.subprocess,"run",query)
    monkeypatch.setattr(controller.time,"sleep",lambda seconds:None)
    def promote(self,stage):
        assert stage=="benchmark" and self is c
        assert self.plan["short_continuation"]["purpose"]=="baseline_training"
        assert "valid_prior_self_guidance" not in self.plan["protocol"]
        assert self.plan["starting_checkpoint"]["epoch"]==36
        promotions.append(stage);return {"status":"baseline_training_guard_evaluated"}
    monkeypatch.setattr(controller.GuardedRestart,"advance",promote)
    return c,queries,promotions


@pytest.mark.parametrize("state",["DONE","EXIT"])
def test_only_terminal_weak_test_allows_independent_original_baseline_promotion(monkeypatch,tmp_path,state):
    c,queries,promotions=_terminal_fixture(monkeypatch,tmp_path,state)
    testroot=Path(c.test["campaign"])
    (testroot/"failure_readiness.json").write_text(json.dumps({"status":"failed" if state=="EXIT" else "passed"}))
    result=c.after_self_test()
    assert result["status"]=="baseline_promotion_evaluated" and promotions==["benchmark"]
    assert result["self_test_terminal"]["state"]==state
    assert result["weak_loss_automatically_adopted"] is False
    assert result["self_test_results"]["failure_readiness.json"]["path"].startswith(str(testroot))
    assert c.after_self_test()==result and len(queries)==len(promotions)==1


@pytest.mark.parametrize("state,returncode,stderr",[("RUN",0,""),("PEND",0,""),
    ("",0,"Job <2001> is not found\n"),("",255,"Permission denied"),
    ("DONE",0,"Permission denied"),("EXIT",1,"")])
def test_active_agedout_or_uncertain_test_never_promotes_baseline(monkeypatch,tmp_path,state,returncode,stderr):
    c,queries,promotions=_terminal_fixture(monkeypatch,tmp_path,state,returncode,stderr)
    with pytest.raises(ValueError,match="not confirmed terminal"):c.after_self_test()
    assert not promotions and 1<=len(queries)<=4
    saved=json.loads((c.root/"after-self-test-decision.json").read_text())
    assert saved["status"]=="stopped" and saved["identity"]==c.identity


def test_nonbenchmark_segment_callback_reuses_original_serial_health_quota_chain(monkeypatch,tmp_path):
    c,_,queued=fixture(monkeypatch,tmp_path)
    called=[]
    monkeypatch.setattr(controller.GuardedRestart,"advance",lambda self,stage:called.append(stage) or {"status":"serial"})
    assert c.advance("segment-001")=={"status":"serial"}
    assert called==["segment-001"] and not queued


def test_existing_decision_from_different_source_cannot_be_reused(monkeypatch,tmp_path):
    c,_,queued=fixture(monkeypatch,tmp_path)
    (c.root/"self-test-decision.json").write_text(json.dumps({"identity":{"plan_sha256":"different"}}))
    with pytest.raises(ValueError,match="different sealed"):c.advance("benchmark")
    assert not queued


def _endpoint_fixture(monkeypatch,tmp_path,*,through=39,reason="bounded_horizon",purpose="baseline_training"):
    from test_chemistry_campaign import campaign_module, _campaign, _benchmark_gate, _segment, _json
    from latentmol import failure_guidance_selection, chemistry_restart_policy
    c=_campaign(tmp_path)
    p=baseline_plan(c.root,purpose)
    p.update(work_root=str(c.work),release=str(c.release),campaign=str(c.root))
    c.plan=p
    c.identity.update(parent_sha256=p["parent_checkpoint_sha256"])
    _benchmark_gate(c,gates={"exact_reference_execution":True},execution=p["execution"],
        native_execution_parity=dict(passed=True,status="passed",restoration={"passed":True},checks={"all":True}))
    # Build ordinary contiguous epoch metadata starting at the actual36 handoff;
    # only opaque temporary checkpoint bytes are used by controller fixtures.
    reference={key:p[key] for key in ("parent_epoch","parent_updates","parent_checkpoint_sha256")}
    p.update(parent_epoch=36,parent_updates=237456,parent_checkpoint_sha256=p["starting_checkpoint"]["sha256"])
    run,record,rows,completion=_segment(c,through=through)
    p.update(reference)
    completion["stop_reason"]=reason
    _json(c.root/"segment-001-completion.json",completion)
    _json(c.root/"failure_readiness.json",{"fixture":"independently authenticated in native fixtures"})
    _json(c.root/"stability_gate.json",{"fixture":"independently authenticated in2500 fixtures"})
    c.require_cpu=lambda:None;c.complete=lambda stage:"22";c.receipt_job=lambda stage:"22"
    monkeypatch.setattr(failure_guidance_selection,"authenticate_readiness",lambda *args:{"status":"passed"})
    monkeypatch.setattr(chemistry_restart_policy,"authenticate_stability",lambda *args:{"completed_updates":2500})
    monkeypatch.setattr(campaign_module,"quota_receipt",lambda minimum:{"minimum_gib":minimum})
    queued=[]
    c.queue=lambda stage,command,**options:queued.append((stage,command,options)) or str(3000+len(queued))
    return c,queued


def test_actual_serial_controller_finishes39_without_epoch40_or_final100(monkeypatch,tmp_path):
    c,queued=_endpoint_fixture(monkeypatch,tmp_path)
    result=c.advance("segment-001")
    assert result["status"]=="short_continuation_completed" and result["completed_epoch"]==39
    assert not queued
    terminal=json.loads((c.root/"bounded-branch-completion.json").read_text())
    assert terminal["status"]=="completed_bounded_branch" and terminal["global_epoch100_complete"] is False
    assert terminal["updates"]==237456+3*6596 and terminal["short_continuation"]["end_epoch"]==39
    assert c.advance("segment-001")==result and not queued


def test_actual_serial_controller_resumes37_in_one24hour_segment_with_same_parent(monkeypatch,tmp_path):
    c,queued=_endpoint_fixture(monkeypatch,tmp_path,through=37,reason="wall_budget_epoch_boundary")
    result=c.advance("segment-001")
    assert result["status"]=="training_submitted_not_passed"
    assert [row[0] for row in queued]==["segment-002","after-segment-002"]
    assert queued[0][2]["gpu"] is True and queued[0][2]["limit"]=="24:00"
    assert "--resume-sha256" in queued[0][1] and "--failure-readiness-sha256" in queued[0][1]
    assert queued[1][2]["dependency"]=="ended(3001)"


@pytest.mark.parametrize("through,reason,purpose",[(40,"bounded_horizon","baseline_training"),
    (39,"planned_horizon","baseline_training"),(39,"bounded_horizon","self_guidance_test")])
def test_real_serial_controller_rejects_overshoot_wrong_stop_and_test_training(monkeypatch,tmp_path,through,reason,purpose):
    c,queued=_endpoint_fixture(monkeypatch,tmp_path,through=through,reason=reason,purpose=purpose)
    with pytest.raises(ValueError):c.advance("segment-001")
    assert not queued and not (c.root/"bounded-branch-completion.json").exists()
