"""Allocated read-only DBMol secant attribution and Protenix co-residency replay."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

CAP=128*1024**2


def require(v,message):
    if not v:raise ValueError(message)


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v
def numpy():
    require(os.environ.get('LSB_JOBID','').isdigit(),'Real allocated numerical execution required')
    import numpy as np
    return np
def bounded(path):
    total=0
    for p in Path(path).rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Linked/special diagnostic output')
        if p.is_file():total+=p.stat().st_size
    require(total<CAP,'Diagnostic output budget exhausted');return total


def secant_attribution(report):
    """Analyze recorded values only; never select a replacement acceptance pair."""
    np=numpy();require(report['status']=='failed' and report['frozen_unchanged'] is True,'Exact failed frozen DBMol report required')
    names=('affinity','contact','confidence');weights=(1.,2.5,2.5);rows=[]
    for direction in report['directions']:
        for p in direction['probes']:
            if not p['evaluated']:continue
            ends=p['endpoint_primal_repeats'];require(len(ends)==2 and all(len(e)==3 for e in ends),'Three recorded repeats per endpoint required')
            means=[float(np.mean([r['value'] for r in e])) for e in ends]
            parts={name:[float(np.mean([r['native_components'][name]['value'] for r in e])) for e in ends] for name in names}
            deltas={name:parts[name][1]-parts[name][0] for name in names}
            recomposed=[sum(w*parts[n][i] for n,w in zip(names,weights)) for i in (0,1)]
            require(np.isfinite([*means,*recomposed,*deltas.values()]).all(),'Nonfinite recorded attribution')
            rows.append({k:direction[k] for k in ('field','row','source','target')}|{
                'epsilon':p['epsilon'],'original_pair_passed':direction['pair_passed'],'original_outcome':p['outcome'],
                'observed_scalar_difference':means[1]-means[0],'predicted_scalar_difference':p['predicted_scalar_difference'],
                'original_resolution_floor':p['resolution_floor'],'weighted_component_differences':{n:w*deltas[n] for n,w in zip(names,weights)},
                'endpoint_scalar_recomposition_max_error':max(abs(a-b) for a,b in zip(means,recomposed)),
                'new_admission_authorized':False})
    require(rows,'No evaluated saved DBMol probes');return rows


def replay_difference(first,other):
    np=numpy();require(set(first)==set(other),'Replay measurement fields changed')
    result={}
    for k,a in first.items():
        a=np.asarray(a);b=np.asarray(other[k]);require(a.shape==b.shape and a.dtype==b.dtype and np.isfinite(a).all() and np.isfinite(b).all(),'Replay shape/dtype/finite mismatch')
        result[k]={'exact_equal':bool(np.array_equal(a,b)),'maximum_absolute_difference':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
    return result


def protenix(plan,output):
    """Original evaluator; diagnostic repeats only, no policy/optimizer call."""
    current=Path(plan['capacity_source']['path'])
    require(sha(plan['native_loader']['path'])==plan['native_loader']['sha256'],'Original native loader changed')
    native=module(plan['native_loader']['path'],'_exact_original_native_loader')
    hooks=module(current/'scripts/continuation_driver.py','_original_point_placement')
    ready=read(native.CONTEXT_READY);require(sha(native.CONTEXT_READY)==native.CONTEXT_READY_SHA,'Original context readiness changed')
    original=read(current/'plan.json');arm=next(r for r in original['arms'] if r['id']=='protenix')
    require(arm['source']['manifest_sha256']==native.SCIENCE_SHA and arm['parent_snapshot']['sha256']==native.ENDPOINT_SHA,'Original native point/source changed')
    life={};session=None;output.mkdir();producer=output/'production';producer.mkdir()
    record={'status':'failed','optimizer_updates':0,'can_certify_or_restart_original_study':False,'can_admit_new_child':False}
    try:
        with native.deadline(5400):
            evaluate,frozen=native.native_evaluator({},arm,hooks,ready,producer,life)
            np=numpy()
            with np.load(native.ENDPOINT,allow_pickle=False) as z:values={k:z[k].copy() for k in ('X','E_upper')}
            initial=frozen();observations=[];baseline=None
            def capture(label):
                nonlocal baseline
                for repeat in range(3):
                    bounded(output);r=evaluate(values)
                    flat={'primal':np.asarray(r['primal']),'autodiff_scalar':np.asarray(r['ad_value']),
                          **{'gradient_'+k:np.asarray(v) for k,v in r['gradients'].items()}}
                    if baseline is None:baseline={k:v.copy() for k,v in flat.items()}
                    path=output/(label+'-'+str(repeat)+'.npz');np.savez_compressed(path,**values,**flat)
                    observations.append({'phase':label,'repeat':repeat,'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),
                                         'versus_first':replay_difference(baseline,flat),'frozen':frozen()})
            capture('before')
            precision=module(current/'scripts/precision_continuation_hooks.py','_unchanged_reference_session')
            cfg=deepcopy(original['reference_configuration']['arms']['protenix'])
            cfg['command_prefix']=precision.expanded_command(cfg,sha(current/'plan.json'))
            session=precision.Session(cfg,output/'reference-session',[0,17]);capture('loaded_reference')
            reference={k:v.astype(np.float64) for k,v in values.items()}
            session.evaluate(reference,True);session.evaluate(reference,False);capture('evaluated_reference')
            session.finish();capture('after_reference_exit');require(frozen()==initial,'Frozen original producer changed')
            record.update(status='diagnostics_completed_original_failures_preserved',observations=observations,
                all_measurements_exactly_equal=all(v['exact_equal'] for r in observations for v in r['versus_first'].values()),
                frozen_unchanged=True,reference_closed=True,original_settings_changed=False)
    finally:
        if session is not None and not session.closed:session.abort()
        if life.get('frozen'):record['frozen_after']=life['frozen']();require(record['frozen_after']==life['frozen_before'],'Original producer model/context mutated')
        record['output_bytes']=bounded(output);write(output/'assessment.json',record)
    require(record['status']=='diagnostics_completed_original_failures_preserved','Diagnostic failed; original failures unchanged')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--pin',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=('cpu','gpu'),required=True);a=p.parse_args()
    require(sha(a.plan)==a.pin,'Diagnostic plan changed');plan=read(a.plan)
    require(os.environ.get('LSB_JOBID','').isdigit(),'Real allocated execution required')
    if a.mode=='gpu':protenix(plan,a.output)
    else:
        require(os.environ.get('LSB_QUEUE')=='hpc','Allocated CPU analysis required');a.output.mkdir()
        report=read(plan['dbmol_failure']['path']);write(a.output/'assessment.json',{
            'status':'diagnostics_completed_original_failures_preserved','original_failures':report['failures'],
            'observations':secant_attribution(report),'optimizer_updates':0,'can_certify_or_restart_original_study':False,'can_admit_new_child':False})


if __name__=='__main__':main()
