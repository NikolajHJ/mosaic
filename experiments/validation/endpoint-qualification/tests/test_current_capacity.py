"""Allocated metadata-only controls for complete current future budgets."""
import importlib.util
import os
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/current_capacity.py'
spec=importlib.util.spec_from_file_location('current_capacity_tests_module',path)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

@pytest.fixture(autouse=True)
def allocated():
    assert os.environ.get('LSB_JOBID'),'Only allocated fixtures'

def prior():
    return {'status':'passed','original_gate_changed':False,'adopted_full_production':True,
            'required_bytes':10**10,'available_bytes':2*10**10,'reserve_bytes':4*1024**3,
            'file_reserve':100000,'chemistry':{'protenix':{'full':6*10**9},'boltz':{'full':4*10**9}}}

def test_full_new_group_and_empirical_excess_are_added_without_reserve_removal():
    p=prior();v=c.validate(p,123)
    assert v['required_bytes']==p['required_bytes']+c.GROUP_ALLOWANCE+123
    assert v['reserve_bytes']==4*1024**3 and v['file_reserve']==100000

def test_group_allowance_shortage_blocks_even_when_parent_passed():
    p=prior();p['available_bytes']=p['required_bytes']+c.GROUP_ALLOWANCE-1
    with pytest.raises(ValueError):c.validate(p,0)

def test_missing_chemistry_future_cannot_admit_a_nonchemistry_job():
    p=prior();del p['chemistry']['boltz']
    with pytest.raises(ValueError):c.validate(p,0)

def test_failed_or_weakened_parent_never_admits():
    for key,value in [('status','failed'),('original_gate_changed',True),('file_reserve',99999),('reserve_bytes',0),('adopted_full_production',False)]:
        p=prior();p[key]=value
        with pytest.raises(ValueError):c.validate(p,0)

def test_negative_boolean_or_fractional_excess_is_rejected():
    for value in (-1,True,.5):
        with pytest.raises(ValueError):c.validate(prior(),value)
