"""Allocated CPU numerical controls, never a native readiness certificate."""
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('forensic',Path(__file__).resolve().parents[1]/'scripts/saved_gradient_forensics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_softmax_chain_sum_zero():
    np=m.numpy();p=np.asarray([[.2,.3,.5]]);assert abs(m.softmax_vjp(p,[[1.,-2.,3.]]).sum())<1e-14
def test_softmax_chain_fd():
    np=m.numpy();z=np.asarray([[.2,-.3,.5]]);g=np.asarray([[1.,-2.,3.]])
    def value(x):
        e=np.exp(x-x.max(-1,keepdims=True));return float((g*e/e.sum(-1,keepdims=True)).sum())
    e=np.asarray([[-1.,0.,1.]]);p=np.exp(z)/np.exp(z).sum(-1,keepdims=True)
    assert abs(float((m.softmax_vjp(p,g)*e).sum())-(value(z+1e-5*e)-value(z-1e-5*e))/2e-5)<1e-9
def test_softmax_zero_gradient():
    np=m.numpy();assert np.array_equal(m.softmax_vjp([[.2,.8]],[[0.,0.]]),[[0.,0.]])
def test_composition_exact():assert m.composition([7.],[ [2.],[1.],[1.] ],[1,2.5,2.5])['maximum_absolute_difference']==0
def test_composition_wrong_gradient_detected():assert m.composition([7.1],[[2.],[1.],[1.]],[1,2.5,2.5])['maximum_absolute_difference']>.09
def test_composition_shape_rejected():
    with pytest.raises(AssertionError):m.composition([1.,2.],[[1.]],[1.])
def test_component_probe_missing_rejected():
    with pytest.raises(AssertionError):m.match_probe({'directions':[],'plan':{'directions':[]}},dict(field='X',row=1,source=0,target=1),{'epsilon':.01})
def test_blob_corruption_rejected():
    with pytest.raises(AssertionError):m.decode({'base64':'AAAAAA==','sha256':'0'*64,'dtype':'float32','shape':[1]})
