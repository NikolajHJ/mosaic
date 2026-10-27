"""Allocated positive/negative logit measurement controls."""
import importlib.util,os,sys
from pathlib import Path
import numpy as np
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('logit_fixture',SCRIPTS/'logit_admission.py')
l=importlib.util.module_from_spec(spec);spec.loader.exec_module(l)
def run(*,wrong=1.,offset=0.,mutate=False,change_frozen=False,seal=None,ad_offset=0.):
    assert os.environ.get('LSB_JOBID')
    values={'atoms':np.zeros((4,3),np.float32),'bonds':np.zeros((6,5),np.float32)}
    weights={k:np.tile(np.arange(v.shape[1],dtype=np.float32),(v.shape[0],1)) for k,v in values.items()}
    def primal(point):
        if mutate:point['atoms'][0,0]=123.
        return np.float32(offset+sum(np.sum(point[k]*w) for k,w in weights.items()))
    def ad(point):return np.float32(primal(point)+ad_offset),{k:w*wrong for k,w in weights.items()}
    count=0
    def frozen():
        nonlocal count
        count+=1;return {'model':'changed' if change_frozen and count>1 else 'frozen','reference':'fixed','coefficient':0}
    return l.evaluate(values,primal,ad,frozen,seal_callback=seal)
def test_linear_logits_qualify_new_child_only():
    r=run();assert r['status']=='passed' and r['can_admit_new_child'] and not r['can_certify_original_study']
def test_both_logit_fields_required():assert {r['field'] for r in run()['directions']}=={'atoms','bonds'}
def test_original_scales_are_all_measured():assert [p['epsilon'] for p in run()['directions'][0]['probes']]==list(l.EPSILONS)
def test_wrong_gradient_rejected():assert run(wrong=1.8)['status']=='failed'
def test_reversed_gradient_rejected():assert run(wrong=-1.)['status']=='failed'
def test_zero_gradient_rejected():assert run(wrong=0.)['status']=='failed'
def test_unresolved_constant_roundoff_is_not_a_pass():assert run(offset=1e8)['status']=='failed'
def test_callback_input_mutation_rejected():assert run(mutate=True)['status']=='failed'
def test_frozen_model_change_rejected():assert run(change_frozen=True)['status']=='failed'
def test_independent_scalar_parity_required():assert run(ad_offset=1.)['status']=='failed'
def test_pair_is_committed_before_perturbed_calls():
    seen=[];r=run(seal=lambda s:seen.append(s));assert len(seen)==1 and seen[0]==r['sealed_plan'] and seen[0]['uses_observed_FD_for_selection'] is False
def test_mutating_seal_copy_does_not_change_selected_pair():
    def tamper(s):s['directions'][0]['selected_local_pair']=[.01,.003]
    r=run(seal=tamper);assert r['status']=='passed' and r['directions'][0]['selected_local_pair']==[.0001,.00003]
def test_float64_logits_rejected():
    with pytest.raises(AssertionError):l.evaluate({'atoms':np.zeros((4,3),np.float64),'bonds':np.zeros((6,5),np.float32)},None,None,lambda:{})
def test_nonfinite_logits_rejected():
    with pytest.raises(AssertionError):l.evaluate({'atoms':np.full((4,3),np.nan,np.float32),'bonds':np.zeros((6,5),np.float32)},None,None,lambda:{})
