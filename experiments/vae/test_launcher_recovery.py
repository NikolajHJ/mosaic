"""Epoch43 recovery keeps the objective and blocks ambiguous scheduler state."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def campaign(tmp_path):
    file=Path(__file__).resolve().parents[2]/"scripts/hpc/vae_neighborhood_campaign.py"
    spec=importlib.util.spec_from_file_location("recovery_campaign_fixture",file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=module.Campaign.__new__(module.Campaign)
    c.root=tmp_path/"scratch";c.root.mkdir()
    c.control=tmp_path/"home";c.control.mkdir();c.receipts=c.control/"receipts";c.receipts.mkdir()
    c.release=tmp_path/"release";c.sha="a"*64;c.path=c.control/"study.json"
    c.python="/work3/s234818/latentmol/env/boltz311/bin/python"
    c.study=dict(launcher_recovery=dict(control_root=str(c.control)),archive_sha256="d"*64,
                 source_job="111",source_callback="112",base_plan_sha256="b"*64)
    return module,c


def test_recovery_launcher_and_callback_use_home_and_system_python(campaign):
    _,c=campaign
    cli=c.cli("after-probe")
    assert cli[:2]==["/usr/bin/python3",str(c.control/"vae_neighborhood_campaign.py")]
    worker=c.worker("probe","/checkpoint43.pt","c"*64)
    assert worker[:2]==["/usr/bin/python3",str(c.control/"hpc_launch.py")]
    assert worker[worker.index("--startup-seconds")+1]=="600"
    assert worker[worker.index("--receipt")+1]==str(c.control/"probe-launch.json")
    assert "OMP_NUM_THREADS=4" in worker


@pytest.mark.parametrize("terminal,accepted",[("Done successfully.",True),("Exited with exit code1.",False)])
def test_aged_out_scheduler_records_require_exact_job_terminal_proof(campaign,monkeypatch,terminal,accepted):
    m,_=campaign
    monkeypatch.setattr(m,"status",lambda job:("AGED_OUT",{}))
    monkeypatch.setattr(m,"command",lambda argv:dict(returncode=0,stdout="Job <111>\nOct31: "+terminal))
    if accepted:assert m.require_done("111")["returncode"]==0
    else:
        with pytest.raises(ValueError,match="terminal DONE"):
            m.require_done("111", attempts=1, delay=0)


def test_unrelated_done_record_cannot_qualify_aged_out_job(campaign,monkeypatch):
    m,_=campaign
    monkeypatch.setattr(m,"status",lambda job:("AGED_OUT",{}))
    monkeypatch.setattr(m,"command",lambda argv:dict(returncode=0,stdout="Job <999>\nOct31: Done successfully."))
    with pytest.raises(ValueError,match="terminal DONE"):
        m.require_done("111", attempts=1, delay=0)


def test_stale_running_state_accepts_exact_terminal_history_without_waiving_exit(campaign,monkeypatch):
    m,_=campaign
    monkeypatch.setattr(m,"status",lambda job:("RUN",dict(stdout="stale")))
    monkeypatch.setattr(m,"command",lambda argv:dict(returncode=0,stdout="Job <111>\nNov1: Done successfully.",stderr=""))
    result=m.require_done("111",attempts=1,delay=0)
    assert len(result['scheduler_evidence'])==2
    assert result['scheduler_evidence'][-1]['state']=='DONE'


def test_history_done_followed_by_restart_is_not_a_terminal_proof(campaign,monkeypatch):
    m,_=campaign
    monkeypatch.setattr(m,"status",lambda job:("RUN",{}))
    monkeypatch.setattr(m,"command",lambda argv:dict(returncode=0,stdout="Job <111>\nNov1: Done successfully.\nNov1: Starting (Pid 42);",stderr=""))
    with pytest.raises(ValueError,match='finish DONE'):
        m.require_done('111',attempts=1,delay=0)


def test_explicit_scheduler_exit_cannot_be_overridden_by_old_history(campaign,monkeypatch):
    m,_=campaign
    monkeypatch.setattr(m,"status",lambda job:("EXIT",{}))
    monkeypatch.setattr(m,"command",lambda argv:pytest.fail('EXIT is authoritative failure'))
    with pytest.raises(ValueError,match='finish DONE'):
        m.require_done('111',attempts=1,delay=0)


def test_repeated_running_state_waits_for_proof_and_retains_evidence(campaign,monkeypatch):
    m,_=campaign;states=iter([('RUN',{}),('RUN',{}),('DONE',{'returncode':0})])
    monkeypatch.setattr(m,'status',lambda job:next(states))
    monkeypatch.setattr(m,'command',lambda argv:dict(returncode=0,stdout='Job <111>\nNov1: Running;',stderr=''))
    assert m.require_done('111',attempts=3,delay=0)['scheduler_evidence'][-1]['state']=='DONE'


def test_structural_probe_queues_only_epoch45_without_touching_old_callback(campaign,monkeypatch):
    m,c=campaign
    config={'sample_draws':0};checkpoint={'path':'/epoch44.pt','sha256':'c'*64}
    c.study.update(structural_adoption={'proposal_config':config},probe_checkpoint=checkpoint,probe_updates=32)
    (c.receipts/'probe.json').write_text(json.dumps({'job_id':'300'}))
    probe=dict(status='passed_bounded_probe',study_sha256=c.sha,source_checkpoint_sha256='c'*64,
        source_completed_epoch=44,completed_updates=32,repair_discovery=config,frozen_anchors_unchanged=True,
        terminal_checks={'no_search_errors':True,'prior_guidance_or_validity':True})
    c.scratch=lambda path:probe
    c.structural_source=lambda:({}, {})
    monkeypatch.setattr(m,'require_done',lambda job:{'returncode':0})
    calls=[]
    c.submit=lambda stage,argv,**kw:calls.append((stage,kw)) or '301'
    result=c.after_structural_probe()
    assert result['status']=='structural_epoch45_submitted'
    assert [name for name,_ in calls]==['epoch45','after-epoch45']
    assert calls[1][1]['dependency']=='ended(301)'


def test_structural_probe_cannot_promote_wrong_checkpoint(campaign,monkeypatch):
    m,c=campaign
    c.study.update(structural_adoption={'proposal_config':{}},probe_checkpoint={'sha256':'c'*64},probe_updates=32)
    (c.receipts/'probe.json').write_text(json.dumps({'job_id':'300'}))
    c.scratch=lambda path:dict(status='passed_bounded_probe',study_sha256=c.sha,source_checkpoint_sha256='d'*64)
    monkeypatch.setattr(m,'require_done',lambda job:{'returncode':0})
    c.submit=lambda *a,**kw:pytest.fail('No training on wrong probe')
    with pytest.raises(ValueError,match='exact checkpoint'):
        c.after_structural_probe()


def test_failed_probe_preserves_checkpoint_and_does_not_resume_old_policy(campaign,monkeypatch):
    m,c=campaign
    (c.control/"handoff-armed.json").write_text("{}")
    c.scratch=lambda path:(_ for _ in ()).throw(FileNotFoundError())
    monkeypatch.setattr(m,"status",lambda job:("PSUSP",{}))
    monkeypatch.setattr(m,"command",lambda *a,**k:pytest.fail("No scheduler mutation on failed recovery probe"))
    result=c.after_probe()
    assert result["status"]=="probe_failed_epoch43_preserved"
    assert result["new_training_submitted"] is False


def test_passed_probe_avoids_dependency_on_aged_out_source_job(campaign,monkeypatch):
    m,c=campaign
    (c.control/"handoff-armed.json").write_text("{}")
    (c.receipts/"probe.json").write_text(json.dumps(dict(job_id="200")))
    c.study.update(probe_checkpoint=dict(path="/checkpoint43.pt",sha256="c"*64),probe_updates=100)
    c.scratch=lambda path:dict(status="passed_bounded_probe",study_sha256=c.sha,
        source_checkpoint_sha256="c"*64,source_completed_epoch=43,completed_updates=100,
        frozen_anchors_unchanged=True,terminal_checks=dict(no_search_errors=True,prior_guidance_or_validity=True))
    monkeypatch.setattr(m,"require_done",lambda job:dict(status="DONE"))
    monkeypatch.setattr(m,"status",lambda job:("PSUSP",{}))
    calls=[]
    c.submit=lambda stage,argv,**kw:calls.append((stage,kw)) or "201"
    monkeypatch.setattr(m,"command",lambda argv:dict(returncode=0))
    c.after_probe()
    assert calls[0]==("epoch44",dict(gpu=True,dependency=None))
    assert (c.control/"callback-retire-reply.json").exists()


def test_control_startup_does_not_read_scratch(campaign):
    m,c=campaign
    repo=Path(__file__).resolve().parents[2]
    changed={}
    for name in ("vae_neighborhood_campaign.py","hpc_launch.py"):
        raw=(repo/"scripts/hpc"/name).read_bytes()
        (c.control/name).write_bytes(raw)
        changed["scripts/hpc/"+name]=hashlib.sha256(raw).hexdigest()
    raw=b'{}';(c.control/"base-plan.json").write_bytes(raw)
    study=dict(c.study,campaign=str(c.root),release="/unavailable/scratch/release",
        base_plan="/unavailable/scratch/plan",base_plan_sha256=hashlib.sha256(raw).hexdigest(),changed_sources=changed)
    c.path.write_text(json.dumps(study));sha=hashlib.sha256(c.path.read_bytes()).hexdigest()
    initialized=m.Campaign(c.path,sha)
    assert initialized.control==c.control and initialized.base=={}
