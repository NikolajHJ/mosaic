"""Scheduler regressions use metadata fixtures, never real LSF operations."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def controller(tmp_path):
    spec=importlib.util.spec_from_file_location("neighborhood_campaign_fixture",
        Path(__file__).resolve().parents[1]/"scripts/hpc/vae_neighborhood_campaign.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    c=module.Campaign.__new__(module.Campaign)
    c.root=tmp_path; c.receipts=tmp_path/"receipts"; c.receipts.mkdir()
    c.sha="a"*64; c.path=tmp_path/"study.json"; c.release=tmp_path/"release"
    c.python="/work3/s234818/latentmol/env/boltz311/bin/python"
    c.study=dict(source_job="111",source_callback="112",base_plan_sha256="b"*64)
    return module,c


def test_gpu_launch_uses_existing_dtu_wrapper_and_explicit_threads(controller):
    _,c=controller
    cmd=c.worker("probe","/tmp/checkpoint.pt","c"*64)
    assert cmd[:4]==["env", "LATENTMOL_SOURCE_ARCHIVE="+str(c.release)+".tar.gz", "bash", "scripts/hpc/boltz_run.sh"]
    assert "OMP_NUM_THREADS=4" in cmd and "CUBLAS_WORKSPACE_CONFIG=:4096:8" in cmd
    assert cmd[cmd.index("--mode")+1]=="probe"


def test_arm_only_holds_exact_pending_callback_and_preserves_gpu(controller,monkeypatch):
    m,c=controller
    (c.receipts/"probe.json").write_text("{}")
    calls=[]
    states=iter([("RUN",{"stdout":"originalGPU"}),
        ("PEND",{"stdout":"after-segment-003 " + "b"*64}), ("PSUSP",{"stdout":"held"})])
    monkeypatch.setattr(m,"status",lambda job:next(states))
    monkeypatch.setattr(m,"command",lambda argv,**kw:calls.append(argv) or dict(returncode=0))
    c.arm()
    assert calls==[["bstop","112"]]
    c.arm() # Idempotent replay must not hold or cancel a second job.
    assert calls==[["bstop","112"]]


def test_arm_rejects_callback_that_already_started(controller,monkeypatch):
    m,c=controller
    (c.receipts/"probe.json").write_text("{}")
    monkeypatch.setattr(m,"status",lambda job:("RUN",{"stdout":"other"}))
    monkeypatch.setattr(m,"command",lambda *a,**k:pytest.fail("must not mutate scheduler"))
    with pytest.raises(ValueError,match="advanced"):
        c.arm()


def test_probe_failure_restores_only_held_predecessor(controller,monkeypatch):
    m,c=controller
    (c.root/"handoff-armed.json").write_text("{}")
    monkeypatch.setattr(m,"status",lambda job:("PSUSP",{}))
    calls=[]
    monkeypatch.setattr(m,"command",lambda argv,**kw:calls.append(argv) or dict(returncode=0))
    result=c.after_probe()
    assert calls==[["bresume","112"]] and "restored" in result["status"]


def test_probe_pass_queues_next_epoch_before_retiring_held_callback(controller,monkeypatch):
    m,c=controller
    (c.root/"handoff-armed.json").write_text("{}")
    (c.root/"probe").mkdir()
    (c.root/"probe/result.json").write_text(json.dumps(dict(status="passed_bounded_probe",study_sha256=c.sha)))
    (c.receipts/"probe.json").write_text(json.dumps(dict(job_id="100")))
    monkeypatch.setattr(m,"status",lambda job:("PSUSP",{}))
    monkeypatch.setattr(m,"require_done",lambda job:dict(state="DONE"))
    events=[]
    def submit(stage,argv,**kw):
        events.append((stage,kw)); return "200" if stage=="epoch44" else "201"
    c.submit=submit
    monkeypatch.setattr(m,"command",lambda argv,**kw:events.append(argv) or dict(returncode=0))
    result=c.after_probe()
    assert events[0]==("epoch44",dict(gpu=True,dependency="done(111)"))
    assert events[-1]==["bkill","112"] and result["job_id"]=="200"


def test_pending_submission_intent_without_receipt_is_not_retried(controller,monkeypatch):
    m,c=controller
    (c.receipts/"probe-intent.json").write_text("{}")
    monkeypatch.setattr(m,"command",lambda *a,**k:pytest.fail("must reconcile, not resubmit"))
    with pytest.raises(FileExistsError):
        c.submit("probe",["worker"],gpu=True)


def test_failed_epoch_does_not_schedule_successor(controller,monkeypatch):
    m,c=controller
    (c.receipts/"epoch44.json").write_text(json.dumps(dict(job_id="200")))
    monkeypatch.setattr(m,"require_done",lambda job:dict(state="DONE"))
    (c.root/"epoch0044").mkdir()
    (c.root/"epoch0044/result.json").write_text(json.dumps(dict(status="failed")))
    c.submit=lambda *a,**k:pytest.fail("failed epoch cannot advance")
    with pytest.raises(ValueError,match="result required"):
        c.advance(44)


def test_saved_success_cannot_promote_an_exited_allocation(controller,monkeypatch):
    m,c=controller
    monkeypatch.setattr(m,"status",lambda job:("EXIT",{}))
    with pytest.raises(ValueError,match="finish DONE"):
        m.require_done("100")


def test_interrupted_hold_recovers_without_second_scheduler_mutation(controller,monkeypatch):
    m,c=controller
    (c.receipts/"probe.json").write_text("{}")
    (c.root/"callback-hold-intent.json").write_text(json.dumps(dict(study_sha256=c.sha)))
    monkeypatch.setattr(m,"status",lambda job:("RUN",{}) if job=="111" else
        ("PSUSP",{"stdout":"after-segment-003 " + "b"*64}))
    monkeypatch.setattr(m,"command",lambda *a,**k:pytest.fail("already held, do not repeat mutation"))
    assert c.arm()["held_callback"]=="112"
