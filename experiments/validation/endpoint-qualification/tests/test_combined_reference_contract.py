"""Allocated synthetic producer/reader regressions; no imports of models.

These are NOT native qualifications and must run on allocated HPC CPU only.
The module itself imports standard-library metadata/pytest only.
"""
from copy import deepcopy
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/combined_reference_contract.py'
spec=importlib.util.spec_from_file_location('_new_combined_reference_contract_tests',P)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    return path


def fixture(tmp_path,*,flat_chemistry=False):
    np=c.allocated_numpy()
    from protenix_combined_reference import reference_logit_evaluate
    root=tmp_path/'cohort';session=root/'reference-session';worker=session/'worker'
    worker.mkdir(parents=True)
    source=tmp_path/'source';source.mkdir();(source/'immutable.py').write_text('original immutable synthetic fixture\n')
    source_manifest=write(source/'source_manifest.json',{'files':{'immutable.py':c.binding(source/'immutable.py')['sha256']}})
    sources=[{'path':str(source),'files':1,'manifest_sha256':c.binding(source_manifest)['sha256']}]
    teacher={'arrays':{'target_atoms':[[6,6]],'route':[1]},'receipt':{'schema_sha256':'s'*64,'attempted_rows':1}}
    teacher['identity']=hashlib.sha256(json.dumps(teacher,sort_keys=True,allow_nan=False).encode()).hexdigest()
    teacher_file=write(tmp_path/'prepared-targets.json',teacher)
    values={f:np.zeros(shape,dtype=np.float32) for f,shape in c.FIELDS.items()}
    true={f:np.zeros(shape,dtype=np.float32) for f,shape in c.FIELDS.items()}
    for f in true:true[f][1,0]=np.float32(.001);true[f][1,1]=np.float32(-.001)
    production_gradient={f:v.copy() for f,v in true.items()}
    key=[0,17];frozen={'production':{'model_digest':'a'*64,'context_digest':'b'*64},
        'reference':{'model_digest':'d'*64,'context_digest':'e'*64}}
    def graph(point,dtype):
        result={}
        for f,field in (('atoms','X'),('bonds','E_upper')):
            z=point[f].astype(dtype);z=z-z.max(-1,keepdims=True);p=np.exp(z);p/=p.sum(-1,keepdims=True)
            result[field]=p
        return result
    baseline=tmp_path/'baseline.npz'
    np.savez_compressed(baseline,atom_logits=values['atoms'],bond_logits=values['bonds'],**graph(values,np.float32))
    production=[];references=[];transport=[];oracle=[]
    def actual_production():
        path=root/('production-'+str(len(production)).zfill(4)+'.npz')
        np.savez_compressed(path,**values,**graph(values,np.float32),primal=np.float64(-.545026),
            AD_value=np.float64(-.545026),**{'gradient_'+f:g for f,g in production_gradient.items()})
        production.append({'raw':c.binding(path),'primal':-.545026,'AD_value':-.545026,
            'gradients':{f:c.sha(g.tobytes()) for f,g in production_gradient.items()},'frozen':frozen['production']})
    actual_production()
    def measure(point,ad):
        i=len(references);g=graph(point,np.float64)
        change=sum(np.sum(point[f].astype(np.float64)*true[f].astype(np.float64)) for f in values)
        native=np.float64(-.545026)+(change if flat_chemistry else 0.)
        chemical=np.float64(0.) if flat_chemistry else np.float64(change)
        total=native+chemical
        path=root/('combined-reference-'+str(i).zfill(4)+'.npz')
        np.savez_compressed(path,**point,**g,total=total,native=native,weighted_chemistry=chemical,
            **({} if not ad else {'gradient_'+f:a.astype(np.float64) for f,a in true.items()}))
        references.append({'raw':c.binding(path),'gradient':ad})
        raw_gradient={field:(true[f].astype(np.float64)/g[field] if flat_chemistry else np.zeros_like(g[field]))
                      for f,field in (('atoms','X'),('bonds','E_upper'))}
        child=worker/('point-'+str(i).zfill(4)+'.npz')
        np.savez_compressed(child,**g,primal=native,key_data=np.asarray(key,dtype=np.uint32),
            **({} if not ad else {'AD_value':native,**{'gradient_'+f:a for f,a in raw_gradient.items()}}))
        digest=c.sha(g['X'].tobytes()+g['E_upper'].tobytes())
        worker_row={'id':i,'gradient':ad,'graph_sha256':digest,'key_data':key,'raw':c.binding(child)}
        oracle.append(worker_row)
        response={'id':i,'graph_sha256':digest,'key_data':key,'value':float(native),'raw':c.binding(child)}
        if ad:response['gradients']={f:a.tolist() for f,a in raw_gradient.items()}
        response_file=write(session/('response-'+str(i).zfill(4)+'.json'),response)
        client=session/('point-client-'+str(i).zfill(4)+'.npz')
        np.savez_compressed(client,**g,value=native,key_data=np.asarray(key,dtype=np.uint32),gradient=np.asarray(ad),
            **({} if not ad else {'gradient_'+f:a for f,a in raw_gradient.items()}))
        transport.append({**worker_row,'raw':c.binding(client),'response':c.binding(response_file)})
        return np.float64(total)
    def ad(point):actual_production();return measure(point,True),production_gradient
    report=reference_logit_evaluate(values,lambda p:measure(p,False),ad,
        lambda:{**frozen,'preparation':teacher['identity']},seal_callback=lambda value:write(root/'pair-seal.json',value))
    actual_production()
    report.update(preparation_identity=teacher['identity'],repair_weight=.001,reference_policy=c.REFERENCE_POLICY,
        reference_measurement_precision_policy=c.PRECISION_POLICY,original_optimizer_or_reference_changed=False,
        softmax_probabilities_bitidentical_between_precisions=False,production_records=production,reference_records=references,
        original_failure_preserved=True)
    path=write(root/'report.json',report)
    identity={'status':'ready_full_double_reference','precision':c.PRECISION,'frozen':frozen['reference'],
              'source_manifest_sha256':sources[0]['manifest_sha256']}
    write(session/'identity.json',identity)
    command=['original-python','original-reference-worker','--output',str(worker)]
    environment={'XLA_FLAGS':'--xla_gpu_autotune_level=0'}
    write(session/'invocation.json',{'job_id':'12345','queue':'gpul40s','argv':command,'environment':environment,
        'key_data':key,'optimizer_updates':0})
    forward=write(worker/'forward-dtype-audit.json',{'status':'passed','aval_counts':{'float64':42,'int32':7},
        'floating_operations_float64':True,'narrow_float_avals':[]})
    adjoint=write(worker/'AD-dtype-audit.json',c.read(forward))
    oracle_file=write(worker/'oracle-points.json',{'requests':oracle,'frozen_before':frozen['reference'],
        'frozen_after':frozen['reference'],'cached_evaluations':0})
    write(session/'client-raw-point-index.json',{'requests':len(transport),'points':transport,
        'cached_or_interpolated_points':0,'reference_frozen':frozen['reference']})
    write(session/'close.json',{'status':'closed','frozen':frozen['reference'],'audit':{
        'status':'passed_full_FP64_reference_audit','floating_operations_float64':True,'narrow_float_avals':[],
        'frozen_unchanged':True,'original_results_changed':False,'optimizer_updates':0,'requests':len(oracle),
        'forward_dtype_audit':c.binding(forward),'AD_dtype_audit':c.binding(adjoint),'raw_points':c.binding(oracle_file)}})
    # Explicit synthetic LSF text, not evidence of a real native allocation.
    lsf=tmp_path/'synthetic-LSF.log';lsf.write_text('Subject: Job 12345: fixture\nqueue <gpul40s>\nSuccessfully completed.\n')
    expected={'version':c.EXPECTATION_VERSION,'coefficient':.001,'preparation_identity':teacher['identity'],
        'preparation_snapshot':c.binding(teacher_file),'baseline':c.binding(baseline),
        'baseline_fields':{'atoms':'atom_logits','bonds':'bond_logits'},'key_data':key,'frozen':frozen,
        'gpu_job_id':'12345','gpu_lsf':c.binding(lsf),'reference_command':command,'reference_environment':environment,
        'reference_identity_subset':{'source_manifest_sha256':sources[0]['manifest_sha256']},
        'sources':sources,'inputs':[c.binding(baseline),c.binding(teacher_file)]}
    return path,expected


@pytest.mark.parametrize('flat_chemistry',[False,True])
def test_allocated_positive_full_witness_and_flat_chemistry_are_non_dispatch_certificates(tmp_path,flat_chemistry):
    path,expected=fixture(tmp_path,flat_chemistry=flat_chemistry)
    result=c.verify(path,expected)
    assert result['status']=='passed_independent_combined_reference_witnesses'
    assert result['can_submit_optimizer'] is False and result['can_certify_or_restart_original_study'] is False
    assert result['reference_points']==78 and result['flat_chemistry_component_allowed']


@pytest.mark.parametrize('field,value',[('reference_policy','wrong'),('reference_measurement_precision_policy','wrapped_FP32'),
    ('repair_weight',.01),('preparation_identity','mutated'),('relative_tolerance',.5),('frozen_unchanged',False)])
def test_allocated_forged_flags_or_changed_policy_are_rejected(tmp_path,field,value):
    path,expected=fixture(tmp_path);report=c.read(path);report[field]=value;write(path,report)
    with pytest.raises(ValueError):c.verify(path,expected)


@pytest.mark.parametrize('factor',[0.,-1.,1.5])
def test_allocated_forced_pass_cannot_hide_actual_wrong_gradient(tmp_path,factor):
    np=c.allocated_numpy();path,expected=fixture(tmp_path);report=c.read(path)
    # Start with all 78 actual reference points and a complete valid seal. Change
    # the recorded production AD and its witnesses coherently, so rejection must
    # come from independent AD evidence rather than missing producer artifacts.
    assert len(report['reference_records'])==78 and report['status']=='passed'
    changed_gradients={}
    for row in report['production_records']:
        values=c.array_file(np,row['raw'])
        for field in c.FIELDS:
            g=(values['gradient_'+field]*np.float32(factor)).astype(np.float32)
            values['gradient_'+field]=g;row['gradients'][field]=c.sha(g.tobytes())
            changed_gradients[field]=g.astype(np.float64)
        np.savez_compressed(row['raw']['path'],**values);row['raw']=c.binding(row['raw']['path'])
    for row in report['baseline_measurement']['raw_gradient_repeats']:
        for field,shape in c.FIELDS.items():
            g=changed_gradients[field];raw=g.tobytes()
            row[field]={'dtype':'float64','shape':list(shape),'sha256':c.sha(raw),
                        'base64':base64.b64encode(raw).decode()}
    write(path,report)
    with pytest.raises(ValueError,match='Declared legacy combined field is not informative|Independent baseline reference AD'):
        c.verify(path,expected)


def test_allocated_constant_production_scalar_shift_cannot_hide_behind_exact_repeats(tmp_path):
    np=c.allocated_numpy();path,expected=fixture(tmp_path);report=c.read(path)
    for row in report['production_records']:
        values=c.array_file(np,row['raw'])
        for field,label in (('primal','primal'),('AD_value','AD_value')):
            values[field]=np.float64(values[field]+.001);row[label]=float(values[field])
        np.savez_compressed(row['raw']['path'],**values);row['raw']=c.binding(row['raw']['path'])
    write(path,report)
    with pytest.raises(ValueError,match='Independent production/reference combined scalar parity'):
        c.verify(path,expected)


@pytest.mark.parametrize('change',['missing_raw','changed_raw','nonfinite_raw','wrong_key','changed_source','changed_target'])
def test_allocated_mutated_missing_or_nonfinite_evidence_is_rejected(tmp_path,change):
    np=c.allocated_numpy();path,expected=fixture(tmp_path);report=c.read(path)
    if change in ('missing_raw','changed_raw','nonfinite_raw'):
        row=report['reference_records'][7]['raw'];raw=Path(row['path'])
        if change=='missing_raw':raw.unlink()
        elif change=='changed_raw':raw.write_bytes(raw.read_bytes()+b'changed')
        else:
            values=c.array_file(np,row);values['total']=np.float64(np.inf);np.savez_compressed(raw,**values)
            report['reference_records'][7]['raw']=c.binding(raw);write(path,report)
    elif change=='wrong_key':
        index_path=path.parent/'reference-session/client-raw-point-index.json';index=c.read(index_path)
        row=index['points'][0]['raw'];values=c.array_file(np,row);values['key_data']=np.asarray([0,99],dtype=np.uint32)
        np.savez_compressed(row['path'],**values);index['points'][0]['raw']=c.binding(row['path']);write(index_path,index)
    elif change=='changed_source':Path(expected['sources'][0]['path'],'immutable.py').write_text('changed')
    else:
        target=c.read(expected['preparation_snapshot']['path']);target['arrays']['route']=[4]
        write(expected['preparation_snapshot']['path'],target)
        expected['preparation_snapshot']=c.binding(expected['preparation_snapshot']['path'])
        expected['inputs'][-1]=deepcopy(expected['preparation_snapshot'])
    with pytest.raises(ValueError):c.verify(path,expected)
