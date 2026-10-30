"""Portable closed-account identity/complete additional-forecast controls."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('_readonly_writer_metadata',Path(__file__).resolve().parents[1]/'scripts/qualification_writer_capacity.py')
capacity=importlib.util.module_from_spec(spec);spec.loader.exec_module(capacity)

def profile():
    c=capacity
    return {'job_id':'29852732','source_files_authenticated':638,
        'baseline':{p:{'bytes':1000,'files':1} for p in c.ROOTS},'growth_bytes':512*1024**2,'growth_files':3000,
        'proofs':[{'path':p,'sha256':h} for p,h in {c.SOURCE+'/embedding-release-manifest.json':c.MANIFEST,
            c.SOURCE+'/configs/embedding_v100_qualification.json':c.PLAN,c.ROOTS[1]+'/cpu/result.json':c.CPU,
            c.WORK+'/releases/source-146-embedding-v100-qualification-attempt001.tar.gz':c.ARCHIVE}.items()],
        'command':'#!/bin/bash;set -euo pipefail;cd '+c.SOURCE+';exec --output '+c.ROOTS[1]+'/gpu --plan-sha256 '+c.PLAN+' --cpu-sha256 '+c.CPU}

def test_exact_profile_is_metadata_only():
    capacity.validate(profile())

@pytest.mark.parametrize('field,value',[('job_id','29852733'),('growth_bytes',1),('growth_files',1),
    ('source_files_authenticated',637),('command','foreign')])
def test_changed_or_reduced_external_forecast_rejects(field,value):
    p=profile();p[field]=value
    with pytest.raises(ValueError):capacity.validate(p)

def test_additional_full_forecast_never_releases_old_capacity():
    r={'available_bytes':1000,'available_files':1000,'required_bytes':400,'required_files':300,'reserve_bytes':4294967296}
    observed={'actual_bytes':100,'actual_files':50,'remaining_bytes':200,'remaining_files':150}
    a=capacity.augment(r,observed)
    assert a['available_bytes']==900 and a['required_bytes']==600 and a['available_files']==950 and a['required_files']==450
    assert a['reserve_bytes']==r['reserve_bytes'] and r['available_bytes']==1000 and a['status']=='passed'

def test_forecast_shortage_stays_failed():
    a=capacity.augment({'available_bytes':600,'required_bytes':400,'available_files':1000,'required_files':300},
        {'actual_bytes':100,'actual_files':50,'remaining_bytes':200,'remaining_files':150})
    assert a['status']=='failed' and a['byte_margin']==-100
