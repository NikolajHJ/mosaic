"""Allocated-only analysis of exact saved gradient probes; never admission."""
import argparse,base64,hashlib,json,os
from pathlib import Path

def numpy():
    if not os.environ.get('LSB_JOBID'):raise RuntimeError('Allocated DTU job required for numerical analysis')
    import numpy as np
    return np

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def decode(row):
    np=numpy();raw=base64.b64decode(row['base64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==row['sha256']
    a=np.frombuffer(raw,dtype=row['dtype']).reshape(row['shape']).copy();assert np.isfinite(a).all();return a

def softmax_vjp(probability,gradient):
    np=numpy();p=np.asarray(probability,dtype=np.float64);g=np.asarray(gradient,dtype=np.float64)
    assert p.shape==g.shape and np.isfinite(p).all() and np.isfinite(g).all()
    return p*(g-(p*g).sum(axis=-1,keepdims=True))

def composition(total,parts,weights):
    np=numpy();total=np.asarray(total,dtype=np.float64);assert len(parts)==len(weights)
    expected=sum(float(w)*np.asarray(v,dtype=np.float64) for v,w in zip(parts,weights))
    assert expected.shape==total.shape and np.isfinite(expected).all() and np.isfinite(total).all()
    return {'maximum_absolute_difference':float(np.max(np.abs(total-expected))),
            'total_max_abs':float(np.max(np.abs(total))),'recomposed_max_abs':float(np.max(np.abs(expected)))}

def match_probe(report,direction,probe):
    keys=('field','row','source','target')
    rows=[(d,p,q) for d,q in zip(report['directions'],report['plan']['directions'])
          if all(d[k]==direction[k] for k in keys) for p in d['probes'] if p['epsilon']==probe['epsilon']]
    assert len(rows)==1,'Exact component direction/scale must be unique'
    d,p,q=rows[0];points=[v for v in q['probes'] if v['epsilon']==probe['epsilon']];assert len(points)==1
    return p,points[0]

def dbmol(reports):
    np=numpy();names=('dbmol_affinity_loss','dbmol_contact_loss','dbmol_confidence_loss');weights=(1.,2.5,2.5)
    total=reports['dbmol_total_loss'];parts=[reports[n] for n in names]
    for value in reports.values():
        assert value['frozen_unchanged'] is True and value['original_run_failed_preserved'] is True
        assert value['can_certify_original_study'] is False
        assert value['plan']['baseline']==total['plan']['baseline'],'Components do not share exact graph bytes'
    assert total['status']=='failed' and all(v['status']=='passed' for v in parts)
    fields={};totalg=total['baseline_checks']['gradient_repeats'][0];pg=[v['baseline_checks']['gradient_repeats'][0] for v in parts]
    for field in ('X','E_upper'):fields[field]=composition(decode(totalg[field]),[decode(v[field]) for v in pg],weights)
    means=lambda row:sum(v['value'] for v in row)/len(row)
    baseline=composition(means(total['baseline_checks']['primal_repeats']),[means(v['baseline_checks']['primal_repeats']) for v in parts],weights)
    observations=[]
    for direction,planned in zip(total['directions'],total['plan']['directions']):
        if direction['pair_passed']:continue
        field=direction['field'];row=direction['row'];source=direction['source'];target=direction['target']
        for probe,point in zip(direction['probes'],planned['probes']):
            if not probe['evaluated']:continue
            matched=[match_probe(r,direction,probe) for r in parts]
            for p,q in matched:
                assert p['evaluated'] and p['scheme']==probe['scheme']
                assert all(q[k]==point[k] for k in ('left','right','actual_difference','effective_step')),'Component probe graph bytes differ'
            values=[means(v) for v in probe['endpoint_primal_repeats']]
            pv=[[means(v) for v in p['endpoint_primal_repeats']] for p,q in matched]
            differences=[v[1]-v[0] for v in pv];observed=values[1]-values[0]
            predicted=probe['predicted_scalar_difference'];displacement=decode(point['actual_difference'])
            component_jvp=[float(np.sum(decode(v[field])*displacement,dtype=np.float64)) for v in pg]
            floors=[max(e['resolution_ulp'] for endpoint in p['endpoint_primal_repeats'] for e in endpoint)*8 for p,q in matched]
            composite_floor=sum(w*f for w,f in zip(weights,floors))
            magnitude=sum(abs(w*v) for w,v in zip(weights,differences))
            observations.append({'field':field,'row':row,'source':source,'target':target,'epsilon':probe['epsilon'],
                'original_outcome':probe['outcome'],'original_agrees':probe['agrees'],'selected_local_pair':direction['selected_local_pair'],
                'exact_component_points_matched':True,'endpoint_scalar_recomposition':[composition(values[i],[v[i] for v in pv],weights) for i in (0,1)],
                'component_observed_differences':dict(zip(names,differences)),'component_baseline_JVP':dict(zip(names,component_jvp)),
                'total_observed_difference':observed,'total_baseline_JVP':predicted,
                'recomposed_observed_difference':sum(w*v for w,v in zip(weights,differences)),
                'recomposed_baseline_JVP':sum(w*v for w,v in zip(weights,component_jvp)),
                'difference_cancellation_ratio':magnitude/max(abs(observed),1e-300),
                'original_total_resolution_floor':probe['resolution_floor'],'weighted_component_resolution_floor':composite_floor,
                'original_tolerance_budget':.15*max(abs(observed),abs(predicted)),
                'diagnostic_only_not_new_acceptance':True})
    return {'baseline_scalar_recomposition':baseline,'baseline_gradient_recomposition':fields,'matched_failed_directions':observations,
            'original_status':'failed','new_admission_authorized':False}

def boltz(report,probe_paths):
    np=numpy();assert report['status']=='failed' and report['repair_weight']==0
    output=[]
    for d in report['directions']:
        field=d['field'];key='atom_logits' if field=='atoms' else 'bond_logits';gkey='gradient_'+key
        graph='X' if field=='atoms' else 'E_upper';rawkey='affinity_gradient_'+graph
        row=d['row'];src=d['source'];dst=d['destination'];observations=[]
        for q in d['probes']:
            eps=q['epsilon'];points=[]
            for sign in (-1,1):
                name=f'{field}-{eps:g}-{sign:+d}.npz';assert name in probe_paths
                with np.load(probe_paths[name],allow_pickle=False) as z:
                    assert float(z['repair_weight'])==0
                    for k in (key,gkey,graph,rawkey,'combined_primal','affinity_primal'):assert np.isfinite(z[k]).all()
                    logits=np.asarray(z[key],dtype=np.float64);prob=np.asarray(z[graph],dtype=np.float64)
                    grad=np.asarray(z[gkey],dtype=np.float64);rawgrad=np.asarray(z[rawkey],dtype=np.float64)
                    chain=softmax_vjp(prob,rawgrad)
                    points.append({'sign':sign,'loss':float(z['combined_primal']),'affinity_primal':float(z['affinity_primal']),
                        'directional_AD':float(grad[row,dst]-grad[row,src]),
                        'raw_to_logit_chain_max_error':float(np.max(np.abs(chain-grad))),
                        'raw_to_logit_chain_direction':float(chain[row,dst]-chain[row,src]),
                        'contrast_coordinate':float((logits[row,dst]-logits[row,src])/2)})
            a,b=points;actualstep=b['contrast_coordinate']-a['contrast_coordinate'];assert actualstep>0
            observations.append({'epsilon':eps,'original_agrees':q['agrees'],'baseline_AD':d['analytic'],
                'recorded_secant':q['finite_difference'],'reconstructed_nominal_secant':(b['loss']-a['loss'])/(2*eps),
                'actual_represented_logit_step':actualstep,'represented_secant':(b['loss']-a['loss'])/actualstep,
                'endpoint_AD_average':(a['directional_AD']+b['directional_AD'])/2,'points':points})
        output.append({'field':field,'row':row,'source':src,'destination':dst,'observations':observations})
    return {'directions':output,'original_status':'failed','new_admission_authorized':False,
            'limitation':'Endpoint AD/softmax chain comparisons are diagnostic; no unsaved native midpoint, Jacobian certification or optimizer update is performed'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--startup-only',action='store_true');a=p.parse_args()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan);assert plan['version']=='saved-gradient-forensics-v1'
    for r in plan['inputs']:
        path=Path(r['path']);assert path.is_file() and all(not q.is_symlink() for q in (path,*path.parents)) and path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
    output=Path(a.output);assert str(output)==plan['output'] if not a.startup_only else str(output)==plan['startup_output']
    output.mkdir(parents=True,exist_ok=False)
    if a.startup_only:
        (output/'startup.json').write_text(json.dumps({'status':'passed_actual_metadata_startup','argv_plan_sha256':a.plan_sha256,'numerical_setup_entered':False,'updates':0})+'\n');return
    numpy();reports={name:read(path) for name,path in plan['dbmol_reports'].items()}
    result={'status':'diagnostics_completed_original_failures_preserved','can_certify_or_restart_original_study':False,
       'optimizer_updates':0,'settings_changed':False,'inputs':plan['inputs'],
       'boltz':boltz(read(plan['boltz_report']),plan['boltz_probes']),'dbmol':dbmol(reports)}
    (output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
