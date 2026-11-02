"""Portable metadata-only policy regressions; no model/array imports."""
import importlib.util
from pathlib import Path
import pytest

candidates=[p/"scripts"/"storage_accounting.py" for p in Path(__file__).resolve().parents]
candidates += [p/"scripts/hpc/storage_accounting.py" for p in Path(__file__).resolve().parents]
path=next(p for p in candidates if p.is_file())
spec=importlib.util.spec_from_file_location("_common_storage",path);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
HEADER="name,id,size,hard,files,hard\n"

def qualified_assess(*args):
    return s.assess(*args,research_profile="qualified")

def test_live_CSV_is_exact_bytes_and_preserves_decimals():
    assert s.parse_pool(HEADER+"s234818,306121,49343807488,107374182400,425353,20000000\n",306121,"s234818")["used_bytes"]==49343807488

@pytest.mark.parametrize("bad",["",HEADER+"other,306121,0,1,0,1\n",HEADER+"s234818,2,0,1,0,1\n",HEADER+"s234818,306121,-1,1,0,1\n",HEADER+"s234818,306121,1,0,0,1\n",HEADER+"s234818,306121,1,1,0,1",HEADER+"s234818,306121,1,1,0,1\ns234818,306121,1,1,0,1\n"])
def test_bad_or_partial_quota_rejects(bad):
    with pytest.raises(ValueError):s.parse_pool(bad,306121,"s234818")

def pools():return {i:{"used_bytes":0,"hard_bytes":None,"used_files":0,"hard_files":None} for i in range(1,7)}
def test_all_finite_pools_take_the_minimum():
    p=pools();p[1].update(hard_bytes=900,hard_files=90);p[6].update(hard_bytes=800,hard_files=80)
    assert s.combine_pools(p)=={"available_bytes":800,"available_files":80}
def test_missing_pool_cannot_silently_pass():
    p=pools();p.pop(2)
    with pytest.raises(ValueError):s.combine_pools(p)
def test_existing_files_are_charged_only_by_live_quota():
    a=qualified_assess({"available_bytes":20*s.GIB,"available_files":1000000},30*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[{"id":"run","remaining_bytes":s.GIB,"remaining_files":1}],[])
    assert a["existing_bytes_recharged"]==0 and a["available_bytes"]==19*s.GIB and a["required_bytes"]==5*s.GIB
    assert a["reserve_bytes"]==4*s.GIB and a["file_reserve"]==100000

def test_unknown_writers_have_explicit_budget_without_fake_status():
    a=qualified_assess({"available_bytes":20*s.GIB,"available_files":1000000},30*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[],[],["123"])
    assert a["unprofiled_writer_ids"]==["123"] and a["required_bytes"]==6*s.GIB

def test_duplicate_allowance_rejected():
    row={"id":"run","remaining_bytes":1,"remaining_files":1}
    with pytest.raises(ValueError):qualified_assess({"available_bytes":20*s.GIB,"available_files":1000000},30*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[row],[row])

@pytest.mark.parametrize("value",[-1,float("nan"),float("inf"),True])
def test_invalid_growth_rejects(value):
    with pytest.raises(ValueError):s.remaining_writes(value,0)
def test_remaining_is_cap_minus_written_not_full_lifetime_cap():assert s.remaining_writes(1000,700)==300

def cases():return [{"case":"case-%03d"%i,"gates_passed":True,"completed_updates":20,"profile":{"bytes_per_update":100}} for i in (1,2,3)]
def test_embedding_uses_remaining_updates_and_one_margin():
    ids={"case-%03d"%i:5000 for i in (1,2,3)}
    a=s.embedding_forecast(cases(),ids,{n:100 for n in ids})
    assert a["remaining_bytes"]==s.GIB+3*5000*125

def test_embedding_responds_to_higher_observed_growth():
    ids={"case-%03d"%i:5000 for i in (1,2,3)}
    a=s.embedding_forecast(cases(),ids,{n:200 for n in ids})
    assert a["remaining_bytes"]==s.GIB+3*5000*250

def test_completed_embedding_only_holds_fixed_control_allowance():
    ids={"case-%03d"%i:10000 for i in (1,2,3)}
    assert s.embedding_forecast(cases(),ids,{n:200 for n in ids})["remaining_bytes"]==s.GIB

@pytest.mark.parametrize("state",[-1,10001])
def test_invalid_completed_counts_reject(state):
    ids={"case-%03d"%i:state for i in (1,2,3)}
    with pytest.raises(ValueError):s.embedding_forecast(cases(),ids,{n:200 for n in ids})

def test_filesystem_and_finite_files_still_gate():
    a=qualified_assess({"available_bytes":20*s.GIB,"available_files":99999},30*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[],[])
    assert a["status"]=="failed"
def test_mixed_zero_inode_observation_rejects():
    with pytest.raises(ValueError):qualified_assess({"available_bytes":20*s.GIB,"available_files":1000000},30*s.GIB,{"f_files":1,"f_ffree":0,"f_favail":1},[],[])


@pytest.mark.parametrize("bad",[r"\work3\account\states",r"C:\work3\states"])
def test_remote_plan_rejects_host_native_path_strings(bad):
    with pytest.raises(ValueError):s.validate_remote_paths({"nested":[{"case":bad}]})
def test_remote_plan_accepts_canonical_POSIX_metadata():
    assert s.validate_remote_paths({"case":"/work3/account/case-0"})=={"case":"/work3/account/case-0"}


def test_exploratory_default_fits_small_bounded_probe_without_global_reserves():
    a=s.assess(None,64*1024**2,{"f_files":0,"f_ffree":0,"f_favail":0},[{"id":"probe","remaining_bytes":8*1024**2,"remaining_files":12}])
    assert a["status"]=="passed" and a["research_profile"]=="exploratory"
    assert a["reserve_bytes"]==16*1024**2 and a["required_files"]==28
    assert a["quota_observation"]=="unavailable" and a["warnings"]


def test_explicit_legacy_version_preserves_historical_result():
    args=({"available_bytes":20*s.GIB,"available_files":1000000},30*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[],[],["123"])
    value=s.assess(*args,policy_version=s.VERSION)
    assert value["required_bytes"]==6*s.GIB and value["research_profile"]=="qualified"


def test_other_studies_and_unknown_writers_are_advisory():
    a=s.assess(None,s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[],[{"id":"other","remaining_bytes":100*s.GIB,"remaining_files":999999}], ["123"])
    assert a["status"]=="passed" and a["future_bytes"]==0 and a["unprofiled_writer_allowance_bytes"]==0
    assert a["unprofiled_writer_ids"]==["123"] and len(a["warnings"])>=3


@pytest.mark.parametrize("disk,quota,inodes",[(0,None,{"f_files":0,"f_ffree":0,"f_favail":0}),
    (s.GIB,{"available_bytes":0,"available_files":None},{"f_files":0,"f_ffree":0,"f_favail":0}),
    (s.GIB,None,{"f_files":10,"f_ffree":1,"f_favail":1})])
def test_exploratory_real_byte_or_inode_exhaustion_still_fails(disk,quota,inodes):
    assert s.assess(quota,disk,inodes,[])["status"]=="failed"


def test_exploratory_rejects_duplicate_own_scopes_and_over_budget():
    row={"id":"same","remaining_bytes":1,"remaining_files":1}
    with pytest.raises(ValueError):s.assess(None,s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[row,row])
    with pytest.raises(ValueError):s.remaining_writes(10,11)


def test_exploratory_reserve_scales_and_is_bounded():
    value=s.assess(None,100*s.GIB,{"f_files":0,"f_ffree":0,"f_favail":0},[{"id":"large","remaining_bytes":10*s.GIB,"remaining_files":100000}])
    assert value["reserve_bytes"]==256*1024**2 and value["file_reserve"]==1000


def test_shared_output_admission_needs_no_experiment_controller(tmp_path):
    (tmp_path / "sample.txt").write_text("result")
    result = s.assess_output(tmp_path, 1024**2, 100)
    assert result["status"] == "passed"
    assert result["own_output"]["actual_files"] == 1
    assert result["own_output"]["actual_bytes"] == 6


def test_shared_output_cap_still_rejects_oversized_tree(tmp_path):
    (tmp_path / "sample.txt").write_text("result")
    with pytest.raises(ValueError, match="exceeded"):
        s.assess_output(tmp_path, 1, 100)
