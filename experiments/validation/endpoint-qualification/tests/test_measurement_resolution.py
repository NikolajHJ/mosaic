"""Allocated synthetic resolution controls; no native certification."""
import importlib.util,os
from pathlib import Path
import numpy as np
import pytest
spec=importlib.util.spec_from_file_location('resolution_policy',Path(__file__).resolve().parents[1]/'scripts/admission.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def package(value=.3):
    return {'value':np.float64(value),'native_components':{'affinity':np.float64(-3.),'contact':np.float64(1.2),'confidence':np.float64(.12)}}
def test_allocation_required():assert os.environ.get('LSB_JOBID')
def test_weighted_component_floor_exceeds_cancelled_total_floor():
    assert a._scalar(package())['resolution_ulp']>a._scalar(np.float64(.3))['resolution_ulp']
def test_component_weights_are_fixed():assert a._scalar(package())['component_weights']=={'affinity':1.,'contact':2.5,'confidence':2.5}
def test_wrong_total_cannot_be_hidden_by_resolution_metadata():
    with pytest.raises(ValueError):a._scalar(package(2.))
def test_missing_native_component_rejected():
    p=package();del p['native_components']['contact']
    with pytest.raises(ValueError):a._scalar(p)
def test_nonfinite_native_component_rejected():
    p=package();p['native_components']['affinity']=np.float64('nan')
    with pytest.raises(ValueError):a._scalar(p)
def test_arbitrary_component_weight_metadata_rejected():
    p=package();p['weights']=[1,1,1]
    with pytest.raises(ValueError):a._scalar(p)
def test_original_scales_retained_and_resolution_pair_predeclared():
    assert a.ORIGINAL_EPSILONS==(.01,.003,.001) and a.LOCAL_EPSILONS==(.003,.001,.0003,.0001,.00003)
def test_uncertainty_never_buys_an_agreement():assert not a.agreement(1.,1.,.2)['agrees']
def test_wrong_gradient_still_resolved_when_signal_is_informative():assert a.agreement(1.,1.8,.01)['outcome']=='resolved_disagreement'
