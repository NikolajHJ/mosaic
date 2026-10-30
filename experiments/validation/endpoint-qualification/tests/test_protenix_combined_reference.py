"""Portable contracts and allocated synthetic cancellation/negative controls."""
import importlib.util
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/protenix_combined_reference.py'
spec=importlib.util.spec_from_file_location('_combined_reference_test',P)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


@pytest.mark.parametrize('change',[None,'composition','zero','mode','margin','environment'])
def test_metadata_closed_unchanged_objective(change):
    c={'repair_weight':.001,'element_js_weight':0,'objective_mode':'native_plus_chemistry'}
    if change=='composition':c['element_js_weight']=.1
    if change=='zero':c['repair_weight']=0
    if change=='mode':c['objective_mode']='native_only'
    if change=='margin':c['confidence_margin']={}
    if change=='environment':c['neighbor_environment']={}
    if change is None:q.validated_config(c)
    else:
        with pytest.raises(ValueError):q.validated_config(c)


@pytest.mark.parametrize('value',[.1,1.])
def test_metadata_tolerance_cannot_be_inflated(value):
    with pytest.raises(ValueError):q.compare_direction(1,1,value)


def test_metadata_private_configuration_preserves_fields_not_class_identity():
    from dataclasses import dataclass
    from types import SimpleNamespace
    @dataclass(frozen=True)
    class Original:weight:float=.1
    @dataclass(frozen=True)
    class Private:weight:float=.1
    value=q.reference_loss_config(SimpleNamespace(FailureGuidanceConfig=Private),Original(.7))
    assert type(value) is Private and vars(value)==vars(Original(.7))


@pytest.mark.parametrize('phase',['cpu','gpu-chemistry'])
def test_metadata_actual_command_declares_shared_owner_for_running_job_accounting(phase):
    import ast
    from types import SimpleNamespace
    source=P.with_name('protenix_chemistry_qualification_ops.py')
    if not source.exists():source=P.with_name('failure_diagnostic_ops.py')
    tree=ast.parse(source.read_bytes())
    fn=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='argv')
    def require(ok,message):assert ok,message
    def original(role,pin,dep):return ['bsub','-H','-J','old','bash','-l','/own/stage','--capacity-source','/original']
    ns={'q':SimpleNamespace(require=require),'original_argv':original,'SHARED':'/shared/current'}
    exec(compile(ast.Module([fn],[]),'<actual-command-metadata>','exec'),ns)
    vector=ns['argv'](phase,'a'*64)
    assert vector[-2:]==['--shared-capacity-source','/shared/current']
    assert vector[vector.index('-J')+1]=='mosaic-Prochem-reference03-'+phase


@pytest.mark.parametrize('factor',[1.,0.,-1.,1.5])
def test_allocated_independent_precision_resolves_small_combined_loss_and_rejects_wrong_AD(factor):
    import os
    assert os.environ.get('LSB_QUEUE')=='hpc' and os.environ.get('LSB_JOBID','').isdigit()
    import numpy as np
    import logit_admission
    v={'atoms':np.zeros((25,6),dtype=np.float32),'bonds':np.zeros((300,5),dtype=np.float32)}
    gradient={f:np.zeros_like(x) for f,x in v.items()}
    for field in v:gradient[field][1,0]=np.float32(.001)
    def fp64(point):
        return np.float64(-.545026)+sum(np.sum(point[f].astype(np.float64)*gradient[f].astype(np.float64)) for f in v)
    def ad(point):return fp64(point),{f:factor*g for f,g in gradient.items()}
    result=q.reference_logit_evaluate(v,fp64,ad,lambda:{'teacher':'fixed','source':'exact'})
    assert (result['status']=='passed')==(factor==1.)
    assert result['frozen_unchanged']
    if factor==1.:
        assert all(r['selected_local_pair'] for r in result['directions'])
        assert q.compare_direction(.001,.001)
    else:assert not q.compare_direction(factor*.001,.001)


@pytest.mark.parametrize('dtype',['float32','float64'])
def test_allocated_reference_scalar_precision_requires_actual_full_double(dtype):
    import os
    assert os.environ.get('LSB_QUEUE')=='hpc' and os.environ.get('LSB_JOBID','').isdigit()
    import numpy as np
    value=np.asarray(.5,dtype=dtype)
    if dtype=='float32':
        with pytest.raises(ValueError):q.independent_scalar(value)
    else:
        row=q.independent_scalar(value)
        assert row['resolution_ulp']==float(abs(np.spacing(value)))
        assert row['resolution_ulp']<float(abs(np.spacing(np.float32(value))))
