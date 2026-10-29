"""Portable metadata-only concurrent-producer controls; no models or arrays."""
import importlib.util
from pathlib import Path
import pytest

S=importlib.util.spec_from_file_location('probe_resource_fixture',Path(__file__).resolve().parents[1]/'scripts/probe_shared_capacity.py')
p=importlib.util.module_from_spec(S);S.loader.exec_module(p)


def profile():
    proofs=[{'path':path,'bytes':1,'sha256':pin} for path,pin in (
        (p.SOURCE+'/embedding-release-manifest.json',p.MANIFEST),
        (p.SOURCE+'/configs/embedding_runtime_probe.json',p.PLAN),
        (p.ROOTS[1]+'/cpu/result.json',p.CPU))]
    return {'version':p.VERSION,'proofs':proofs,'baseline':{r:{'bytes':100,'files':2} for r in p.ROOTS},
            'growth_bytes':512*1024**2,'growth_files':3000,'job_id':'29846982',
            'command':'#!/bin/bash;set -euo pipefail;cd '+p.SOURCE+';exec python --output '+p.ROOTS[1]+'/gpu --plan-sha256 '+p.PLAN+' --cpu-sha256 '+p.CPU}


def test_exact_probe_closed_resource_profile():
    v=profile();assert p.validate_profile(v)==v


@pytest.mark.parametrize('key,value',[('job_id','99'),('growth_bytes',1),('growth_files',1),('version','old'),('baseline',{}),('command','other')])
def test_probe_unknown_changed_or_reduced_budget_rejected(key,value):
    v=profile();v[key]=value
    with pytest.raises(ValueError):p.validate_profile(v)


def test_probe_missing_or_changed_immutable_cpu_proof_rejected():
    v=profile();v['proofs'][-1]['sha256']='a'*64
    with pytest.raises(ValueError):p.validate_profile(v)


def test_actual_bytes_plus_unspent_forecast_are_charged_once_and_reserves_retained():
    value={'status':'passed','available_bytes':1000,'available_files':1000,'required_bytes':100,'required_files':100,'reserve_bytes':400,'reserve_files':100}
    result=p.augment(value,{'actual_bytes':20,'actual_files':2,'remaining_bytes':380,'remaining_files':98})
    assert result['byte_margin']==500 and result['file_margin']==800 and result['reserve_bytes']==400
    assert value['available_bytes']==1000
    assert p.augment(value,{'actual_bytes':20,'actual_files':2,'remaining_bytes':1000,'remaining_files':98})['status']=='failed'


def test_nofollow_inventory_counts_actual_paths_and_rejects_links(tmp_path):
    (tmp_path/'file').write_bytes(b'abcdef')
    value=p.inventory(tmp_path);assert value['actual_bytes']>=6 and value['actual_files']==1
    assert value['complete'] and value['nofollow']


def test_missing_inventory_cannot_be_treated_as_zero(tmp_path):
    with pytest.raises(ValueError,match='missing/linked'):p.inventory(tmp_path/'missing')
