"""Allocated adoption/tamper/launch contracts; fake records are not native proof."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol import chemistry_valid_reference_adoption as adoption


def _sources(tmp_path, source, successor, name="src/latentmol/chemistry_continuation.py"):
    old, new = tmp_path / "old", tmp_path / "new"
    for root, text in ((old, source), (new, successor)):
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    return old, new, {name:"a"*64}, {name:"b"*64}


def test_source_audit_allows_only_explicit_metadata_boundary(tmp_path):
    before="SETTING=1\ndef train_phase(plan):\n    return 1\ndef chemistry_update():\n    return 3\n"
    after=before.replace("return 1", "return 2")
    assert adoption.compare_source(*_sources(tmp_path,before,after)) == ["src/latentmol/chemistry_continuation.py"]


@pytest.mark.parametrize("change", [lambda s:s.replace("SETTING=1","SETTING=2"),
    lambda s:s.replace("return 3","return 4"),lambda s:s.replace("def chemistry_update", "def removed_update")])
def test_source_audit_rejects_numerical_body_constant_or_removed_function(tmp_path,change):
    before="SETTING=1\ndef train_phase(plan):\n    return 1\ndef chemistry_update():\n    return 3\n"
    with pytest.raises(ValueError): adoption.compare_source(*_sources(tmp_path,before,change(before)))


def test_source_audit_rejects_changed_loss_configuration_or_unknown_new_executable(tmp_path):
    with pytest.raises(ValueError): adoption.compare_source(tmp_path,tmp_path,{"configs/loss.json":"a"},{"configs/loss.json":"b"})
    with pytest.raises(ValueError): adoption.compare_source(tmp_path,tmp_path,{}, {"src/latentmol/reward.py":"a"})
    with pytest.raises(ValueError): adoption.compare_source(tmp_path,tmp_path,{"src/latentmol/chemistry.py":"a"},{})


def _derived(tmp_path,monkeypatch):
    plan={"campaign":str(tmp_path),"_plan_sha256":"a"*64,"_benchmark_gate_sha256":None}
    evidence={"adoption.json":{"status":"passed_evidence_adoption_not_new_native_test"},
        "failure_readiness.json":{"status":"passed","source_observations_reused":True},
        "benchmark_gate.json":{"status":"passed","new_native_observation":False}}
    original={"status":"passed","completed_updates":2500,"original_source":True}
    monkeypatch.setattr(adoption,"_build",lambda p:(deepcopy(evidence),deepcopy(original)))
    monkeypatch.setenv("LSB_JOBID","123456")
    return plan,evidence,original


def test_materialization_is_deterministic_additive_and_every_byte_is_reauthenticated(tmp_path,monkeypatch):
    plan,evidence,original=_derived(tmp_path,monkeypatch)
    adoption.materialize_adoption(plan)
    assert adoption.materialize_adoption(plan) == evidence["adoption.json"]
    expected=adoption._sha(tmp_path/"failure_readiness.json")
    assert adoption.authenticate_adoption(plan,expected) == evidence["failure_readiness.json"]
    assert adoption.authenticate_adoption(plan,stability=True) == original
    (tmp_path/"benchmark_gate.json").write_text('{"status":"failed"}')
    with pytest.raises(ValueError):adoption.authenticate_adoption(plan,expected)
    with pytest.raises(ValueError):adoption.materialize_adoption(plan)


def test_materialization_requires_allocation_and_exact_readiness_sha(tmp_path,monkeypatch):
    plan,_,_=_derived(tmp_path,monkeypatch)
    monkeypatch.delenv("LSB_JOBID")
    with pytest.raises(ValueError):adoption.materialize_adoption(plan)
    monkeypatch.setenv("LSB_JOBID","123456");adoption.materialize_adoption(plan)
    with pytest.raises(ValueError):adoption.authenticate_adoption(plan,"b"*64)
    plan["_benchmark_gate_sha256"]="b"*64
    with pytest.raises(ValueError):adoption.authenticate_adoption(plan)


def test_required_cuda_junit_never_accepts_skips_errors_or_partial_coverage(tmp_path):
    path=tmp_path/"fixtures.xml"
    path.write_text('<testsuites><testsuite tests="518" skipped="0" failures="0" errors="0"/></testsuites>')
    adoption._junit(path,518,0)
    for changes in ('tests="517" skipped="0" failures="0" errors="0"',
                    'tests="518" skipped="1" failures="0" errors="0"',
                    'tests="518" skipped="0" failures="1" errors="0"'):
        path.write_text('<testsuites><testsuite '+changes+'/></testsuites>')
        with pytest.raises(ValueError):adoption._junit(path,518,0)


def test_live_or_failed_source_job_never_adopts(tmp_path,monkeypatch):
    root=tmp_path/"allocations"/"one";root.mkdir(parents=True)
    (root/"run.json").write_text(json.dumps(dict(environment={"job_id":"123"},status="completed",returncode=0,
        source_archive_sha256="a",source={"sha256":"b"})))
    for state in ("RUN","PEND","EXIT"):
        monkeypatch.setattr(adoption.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=0,stdout=state,stderr=""))
        with pytest.raises(ValueError):adoption._terminal(tmp_path,"123","a","b")
    monkeypatch.setattr(adoption.subprocess,"run",lambda *a,**k:SimpleNamespace(returncode=255,stdout="",stderr="Job <123> is not found"))
    adoption._terminal(tmp_path,"123","a","b")


def test_first_training_worker_keeps_cuda_gate_before_train_and_later_segments_are_serial(monkeypatch):
    import sys
    scripts=Path(__file__).resolve().parents[1]/"scripts/hpc"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("reference_training_campaign_fixture",scripts/"chemistry_valid_reference_campaign.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    c=object.__new__(module.ValidReferenceCampaign)
    c.work=Path("/work3/s234818/latentmol");c.release=Path("/release");c.path=Path("/plan.json");c.sha="a"*64
    checked=[]
    monkeypatch.setattr(c,"check_launch",lambda command:checked.append(list(command)))
    monkeypatch.setattr(module.GuardedRestart,"worker",lambda self,command,label:command)
    train=["python","-m","latentmol.chemistry_continuation","train","--segment","1"]
    first=c.worker(train,"train")
    assert first[:2] == ["bash","-c"] and "--mode cuda-check" in first[2] and " && " in first[2]
    later=train[:-1]+["2"]
    assert c.worker(later,"train") == later
    assert checked == [train, later]
    assert "tests/test_chemistry_resume_cuda.py" in module.CUDA_FIXTURES
    assert "tests/test_chemistry_valid_reference_training_policy.py" in module.CUDA_FIXTURES
