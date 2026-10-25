"""Allocated engineering contracts for retry02; no fixture is a native pass."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import full_model_ops as ops
import full_model_pilot as pilot


@pytest.fixture
def admission(tmp_path,monkeypatch):
    prior={"status":"passed","reserve_bytes":4*1024**3,"file_reserve":100000,
           "required_bytes":1000,"available_bytes":2000,
           "parent_gate":{"status":"passed","additional_allowance_bytes":
                          1024**3+ops.LIMIT+768*1024**2+512*1024**2}}
    monkeypatch.setattr(ops,"verify",lambda *a:{"files":dict.fromkeys(range(14))})
    calls=[]
    def storage(parent,kind):
        calls.append((parent,kind));return deepcopy(prior)
    monkeypatch.setattr(ops,"_module",lambda *a:SimpleNamespace(storage=storage))
    monkeypatch.setattr(ops,"own_bytes",lambda:1)
    return tmp_path,prior,calls


@pytest.mark.parametrize("phase,kind",[("preflight","preflight"),("cpu-entry","entry"),
    ("two-gpu-dispatch","dispatch"),("protenix-entry","entry"),("boltz-entry","entry")])
def test_current_full_guard_is_retained_without_duplicate_or_missing_chemistry_allowance(admission,phase,kind):
    parent,prior,calls=admission
    value=ops.storage(parent,phase)
    assert calls==[(parent,kind)] and value["whole_current_gate"]==prior
    assert value["status"]=="passed" and value["required_bytes"]==prior["required_bytes"]
    assert value["additional_allowance_bytes"]==0
    assert value["reserved_chemistry_validation_bytes"]==2*1024**3
    assert value["reserve_bytes"]==4*1024**3 and value["file_reserve"]==100000
    assert value["original_gate_changed"] is False


@pytest.mark.parametrize("fault",["prior_failed","retained_failed","reserve","files",
    "missing_reserved_allowance","capacity","cap","source_count"])
def test_current_admission_never_waives_failed_gate_or_drops_cap(admission,monkeypatch,fault):
    parent,prior,_=admission
    if fault=="prior_failed":prior["status"]="failed"
    elif fault=="retained_failed":prior["parent_gate"]["status"]="failed"
    elif fault=="reserve":prior["reserve_bytes"]-=1
    elif fault=="files":prior["file_reserve"]-=1
    elif fault=="missing_reserved_allowance":prior["parent_gate"]["additional_allowance_bytes"]-=1
    elif fault=="capacity":prior["available_bytes"]=999
    elif fault=="cap":monkeypatch.setattr(ops,"own_bytes",lambda:ops.LIMIT)
    else:monkeypatch.setattr(ops,"verify",lambda *a:{"files":dict.fromkeys(range(13))})
    with pytest.raises(ValueError):ops.storage(parent,"preflight")


@pytest.mark.parametrize("profile",["protenix-random50k","boltz-chem30k"])
def test_pilot_and_production_have_one_identical_adapter_activation(profile,tmp_path,monkeypatch):
    events=[]
    @contextmanager
    def install(name,**expected):
        events.append(("enter",name,expected))
        try:yield {"adapter":name}
        finally:events.append(("exit",name))
    monkeypatch.setitem(sys.modules,"hybrid_search",SimpleNamespace(
        install_hybrid=lambda **kw:install("hybrid",**kw)))
    monkeypatch.setitem(sys.modules,"gpu_matrix",SimpleNamespace(
        install_matrix_loss=lambda variant:install("matrix",variant=variant)))
    monkeypatch.setitem(sys.modules,"gpu_local_bridge",SimpleNamespace(
        install_offloaded_matrix_loss=lambda path:install("bridge",path=path)))
    with pilot.combined_adapters(profile,tmp_path) as value:
        assert value["hybrid_policy"]=={"adapter":"hybrid"}
        assert events[0]==("enter","hybrid",{"workers":4,"max_entries":8192})
        target="matrix" if profile=="protenix-random50k" else "bridge"
        assert value["matrix_policy"]=={"adapter":target}
    assert events[-2:]==[("exit",target),("exit","hybrid")]


def test_undeclared_combined_profile_fails_before_installation(tmp_path):
    with pytest.raises(ValueError,match="Undeclared"):
        with pilot.combined_adapters("changed-science",tmp_path):pass


@pytest.mark.parametrize("error",["RuntimeError: Native Boltz worker exited: None",
    "AssertionError: Arrays are not equal",None])
def test_first_failure_cannot_be_masked_as_absent_coupled_or_runtime_report(error):
    with pytest.raises(ValueError,match="Original validation failed"):
        pilot.require_driver_progress({"status":"failed","error":error})
    assert pilot.require_driver_progress({"status":"failed","error":pilot.RUNTIME_ERROR}) is None
    assert pilot.require_driver_progress({"status":"passed","error":None}) is None


@pytest.mark.parametrize("fault",["none","nonterminal","unbound"])
def test_previous_pilots_must_have_bound_actual_terminal_failures(tmp_path,monkeypatch,fault):
    monkeypatch.setattr(ops,"BASE",tmp_path)
    jobs={"protenix":"29811883","boltz":"29811884"};bindings={}
    for arm,job in jobs.items():
        parent=tmp_path/"chemistry-full-model-01"/arm
        (parent/"payload").mkdir(parents=True)
        assessment=parent/"payload/assessment.json"
        assessment.write_text(json.dumps({"status":"failed","job_id":job,"production_authorized":False,
            "source_manifest_sha256":"347e753d481ec3d584f3b0b05cc2801ee745036ee17638d7c3672fc2ca2cba16"}))
        log=parent/("stdout."+job+".log")
        state="Done" if fault=="nonterminal" else "Exited"
        log.write_text(f"Subject: Job {job}: <mosaic-chem-full01-{arm}> in cluster <dcc> {state}\n"
                       "Job executed in queue <gpul40s>\nExited with exit code 1.\n")
        bindings[str(assessment)]={};bindings[str(log)]={}
    if fault=="unbound":bindings.pop(str(log))
    if fault=="none":assert ops.previous_failures({"failed_predecessors":jobs},bindings)==jobs
    else:
        with pytest.raises(ValueError):ops.previous_failures({"failed_predecessors":jobs},bindings)


def test_previous_live_boltz_group_bytes_are_retained_under_same_joint_cap(tmp_path,monkeypatch):
    root=tmp_path/"new";source=tmp_path/"release/source";prior=tmp_path/"previous"
    root.mkdir();source.mkdir(parents=True);prior.mkdir()
    (root/"new.bin").write_bytes(b"1234");(prior/"previous.bin").write_bytes(b"123456")
    monkeypatch.setattr(ops,"ROOT",root);monkeypatch.setattr(ops,"SOURCE",source)
    monkeypatch.setattr(ops,"PRIOR_OWNED_ROOTS",(prior,))
    assert ops.LIMIT==2*1024**3 and ops.own_bytes()==10
