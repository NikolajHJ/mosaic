"""Portable actual adapter/source integration tests, no numerical libraries."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

OWNER=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def modules():
    a=load(OWNER/"scripts/storage_policy_adapter.py","_actual_new_storage_adapter")
    source=OWNER/"scripts/rolling_capacity.py"
    if not source.exists():
        source=Path((json.loads((OWNER/"plan.json").read_bytes()).get("capacity_science") or json.loads((OWNER/"plan.json").read_bytes())["capacity_source"])["path"])/"scripts/rolling_capacity.py"
    r=load(source,"_unchanged_rolling_for_metadata")
    return a,r

def test_actual_adapter_handles_real_source_and_does_not_fake_RUN(monkeypatch,tmp_path):
    a,r=modules();monkeypatch.setattr(r.c,"validate_ledger",lambda ledger:([],100000));a.install(r,{"storage_policy":{"version":a.policy.VERSION}},tmp_path)
    row={"JOBID":"123","STAT":"RUN","QUEUE":"gpul40s","COMMAND":"bash /external/new-diagnostic.sh"}
    seen,external=r.classify_jobs([row],[],set(),Path("/owned/source"))
    assert external==["123"] and row["STAT"]=="RUN" and not seen
    monkeypatch.setattr(a,"live_quota",lambda p:({"summary":{"available_bytes":30*a.policy.GIB,"available_files":2000000}},{"path":"quota"}))
    monkeypatch.setattr(r.c,"validate_ledger",lambda ledger:([],100000))
    result=r.c.assess({"goals":[{"id":"dbmol","saved_updates":7500,"next_updates":0}],"phase":"entry","source_manifest_sha256":"pin"},"display",40*a.policy.GIB,{"f_files":0,"f_ffree":0,"f_favail":0})
    assert result["unprofiled_writer_ids"]==["123"] and result["required_bytes"]==6*a.policy.GIB
    assert result["ledger"]["goals"][0]["target_updates"]==30000 and result["ledger"]["goals"][0]["later_updates_deferred"]==22500

def test_external_CURRENT_bytes_are_not_subtracted_again(monkeypatch,tmp_path):
    a,r=modules();a.install(r,{"storage_policy":{"version":a.policy.VERSION}},tmp_path)
    before={"available_bytes":1000,"required_bytes":300,"available_files":1000000,"required_files":100000,"external_remaining_budgets":[]}
    value=r.__dict__["_policy_augment"](before,{"actual_bytes":500,"remaining_bytes":200,"remaining_files":3},"vae")
    assert value["available_bytes"]==1000 and value["required_bytes"]==500 and value["required_files"]==100003

def test_authenticated_extra_writer_keeps_identity_and_remaining_forecast(monkeypatch,tmp_path):
    a,r=modules();observer=SimpleNamespace(observe=lambda p,rows:{"accounted_job_ids":["123"],"remaining_bytes":500,"remaining_files":1})
    monkeypatch.setitem(sys.modules,"qualification_writer_capacity",observer)
    a.install(r,{"storage_policy":{"version":a.policy.VERSION}},tmp_path,{"exact":"profile"})
    rows=r.parse_scheduler(json.dumps({"RECORDS":[{"JOBID":"123","STAT":"RUN","QUEUE":"gpuv100","COMMAND":"exact-source146"}],"JOBS":1}))
    assert r.classify_jobs(rows,[],set(),Path("/owned/source"))[1]==[] and rows[0]["STAT"]=="RUN"
