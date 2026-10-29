"""Portable metadata controls; no models, tensors, scheduler or external writes."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location("embedding_capacity_test",Path(__file__).resolve().parents[1]/"scripts/embedding_shared_capacity.py")
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

def pilot():
    return {"cases":[{"case":"case-00"+str(i),"completed_updates":20,"gates_passed":True,
                      "profile":{"bytes_per_update":i,"fixed_bytes":0}} for i in (1,2,3)]}

def profile():
    return {"version":e.VERSION,"proofs":[],"source_manifests":[{"sha256":s} for s in e.MANIFESTS],
            "baseline":{r:{"bytes":0,"files":0} for r in e.ROOTS},"growth_bytes":2*1024**3,"growth_files":150000,
            "job_templates":{k:"python worker "+" ".join(e.MANIFESTS)+" "+flag+" case-001-segment-0001"
                             for k,flag in [("gpu","--node"),("callback","--after")]}}

def test_all_three_full_horizons_plus_control_growth_are_reserved():
    assert e.forecast(pilot()) == 120000+1024**3

@pytest.mark.parametrize("value",[-1,float("nan"),float("inf"),True])
def test_invalid_pilot_bytes_rejected(value):
    p=pilot();p["cases"][0]["profile"]["bytes_per_update"]=value
    with pytest.raises(ValueError):e.forecast(p)

@pytest.mark.parametrize("change",["order","count","failed"])
def test_no_case_or_gate_can_be_omitted(change):
    p=pilot()
    if change=="order":p["cases"].reverse()
    elif change=="count":p["cases"][0]["completed_updates"]=19
    else:p["cases"][0]["gates_passed"]=False
    with pytest.raises(ValueError):e.forecast(p)

def test_only_exact_serial_command_family_is_accounted():
    template=profile()["job_templates"]["gpu"]
    assert e.matches(template.replace("case-001-segment-0001","case-003-segment-0004"),template)
    assert not e.matches(template.replace("worker","different-worker"),template)
    assert not e.matches(template.replace("case-001-segment-0001","case-004-segment-0001"),template)

@pytest.mark.parametrize("change",["source","root","allowance"])
def test_source_root_and_file_allowances_fail_closed(change):
    p=profile()
    if change=="source":p["source_manifests"][0]["sha256"]="f"*64
    elif change=="root":p["baseline"]["/other"]=p["baseline"].pop(e.ROOTS[0])
    else:p["growth_files"]=0
    with pytest.raises(ValueError):e.validate_profile(p)

def test_current_stale_charge_and_remaining_forecast_each_count_once():
    initial={"available_bytes":1000,"available_files":1000,"required_bytes":100,"required_files":100}
    observation={"actual_bytes":200,"actual_files":20,"remaining_bytes":500,"remaining_files":50}
    result=e.augment(initial,observation)
    assert result["available_bytes"]==800 and result["required_bytes"]==600 and result["byte_margin"]==200
    assert result["file_margin"]==830 and initial["available_bytes"]==1000
