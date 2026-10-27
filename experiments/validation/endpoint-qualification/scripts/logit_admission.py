"""Prospective local logit derivative measurement; no optimizer or native setup."""
from copy import deepcopy
import math
import admission as a

VERSION='prospective-fixed-reference-logit-admission-v1'
EPSILONS=(.01,.003,.001,.0003,.0001,.00003)

def evaluate(values,primal,autodiff,frozen,*,seal_callback=None):
    np=a._numpy()
    assert a.VERSION=='prospective-endpoint-admission-v3'
    assert set(values)=={'atoms','bonds'}
    values={k:np.asarray(v).copy() for k,v in values.items()}
    assert all(v.dtype==np.float32 and v.ndim==2 and v.shape[1]>1 and np.isfinite(v).all() for v in values.values())
    before={k:a._record(v) for k,v in values.items()};identity=deepcopy(frozen())
    result={'version':VERSION,'status':'failed','can_admit_new_child':False,'original_failure_preserved':True,
            'can_certify_original_study':False,'baseline':before,'directions':[],'calls':[],
            'relative_tolerance':.15,'epsilons':list(EPSILONS),'selection_before_perturbed_forwards':True}
    def call(callback,point,kind):
        payload={k:v.copy() for k,v in point.items()};expected={k:a._record(v) for k,v in payload.items()}
        r=callback(payload);assert {k:a._record(v) for k,v in payload.items()}==expected
        result['calls'].append({'kind':kind,'point':{k:v['sha256'] for k,v in expected.items()}});return r
    try:
        ps=[a._scalar(call(primal,values,'baseline_primal')) for _ in range(a.REPEATS)]
        ads=[call(autodiff,values,'baseline_AD') for _ in range(a.REPEATS)]
        ss=[a._scalar(v[0]) for v in ads];gs=[]
        for v,g in ads:
            assert set(g)==set(values)
            g={k:np.asarray(x) for k,x in g.items()}
            assert all(g[k].shape==values[k].shape and g[k].dtype.kind=='f' and np.isfinite(g[k]).all() for k in values)
            gs.append({k:v.astype(np.float64) for k,v in g.items()})
        samples=ps+ss;mean=sum(v['value'] for v in ps)/len(ps)
        assert all(abs(v['value']-mean)<=a.PARITY_ATOL for v in ss)
        assert all(float(np.max(np.abs(g[k]-gs[0][k])))<=a.GRADIENT_REPEAT_ATOL for g in gs for k in values)
        spread=max(v['value'] for v in samples)-min(v['value'] for v in samples)
        floor=a.ULP_MULTIPLIER*max(v['resolution_ulp'] for v in samples)+a.SPREAD_MULTIPLIER*spread
        grad={k:np.mean([g[k] for g in gs],axis=0) for k in values}
        result['baseline_measurement']={'primal_repeats':ps,'AD_scalar_repeats':ss,'resolution_floor':floor,
            'raw_gradient_repeats':[{k:a._record(v) for k,v in g.items()} for g in gs]}
        plans=[]
        for field in ('atoms','bonds'):
            gradient=grad[field];row=int((gradient.max(-1)-gradient.min(-1)).argmax())
            source=int(gradient[row].argmin());target=int(gradient[row].argmax());tangent=float(gradient[row,target]-gradient[row,source])
            assert tangent>a.MINIMUM_TANGENT
            probes=[]
            for epsilon in EPSILONS:
                left=values[field].copy();right=values[field].copy()
                for sign,p in ((-1,left),(1,right)):
                    p[row,source]=np.float32(p[row,source]-np.float32(sign*epsilon))
                    p[row,target]=np.float32(p[row,target]+np.float32(sign*epsilon))
                assert np.isfinite(left).all() and np.isfinite(right).all()
                difference=right.astype(np.float64)-left.astype(np.float64)
                effective_step=float((difference[row,target]-difference[row,source])/2)
                assert effective_step>0
                expected=float(np.sum(gradient*difference,dtype=np.float64))
                signals=[abs(float(np.sum(gradient*(v.astype(np.float64)-values[field]),dtype=np.float64))) for v in (left,right)]
                eligible=all(v>a.SIGNAL_TO_FLOOR*floor for v in signals)
                probes.append({'epsilon':epsilon,'left':a._record(left),'right':a._record(right),
                    'actual_difference':a._record(difference),'effective_step':effective_step,'expected_difference':expected,'baseline_endpoint_signals':signals,'eligible':eligible})
            pairs=[[x['epsilon'],y['epsilon']] for x,y in zip(probes,probes[1:]) if x['eligible'] and y['eligible']]
            plans.append({'field':field,'row':row,'source':source,'destination':target,'analytic':tangent,
                          'selected_local_pair':pairs[-1] if pairs else None,'probes':probes})
        seal={'version':VERSION,'baseline':before,'resolution_floor':floor,'directions':plans,
              'uses_observed_FD_for_selection':False,'no_pair_fallback':True}
        result['sealed_plan']=deepcopy(seal);result['plan_sha256']=a._seal(seal)
        if seal_callback:seal_callback(deepcopy(seal))
        for planned in plans:
            row={k:deepcopy(v) for k,v in planned.items() if k!='probes'};row['probes']=[]
            for probe in planned['probes']:
                endpoint=[]
                for name in ('left','right'):
                    point={k:v.copy() for k,v in values.items()};point[planned['field']]=a._decode(probe[name])
                    endpoint.append([a._scalar(call(primal,point,'perturbed_primal')) for _ in range(a.REPEATS)])
                means=[sum(v['value'] for v in es)/len(es) for es in endpoint]
                endspread=max(spread,*[max(v['value'] for v in es)-min(v['value'] for v in es) for es in endpoint])
                resolution=a.ULP_MULTIPLIER*max(v['resolution_ulp'] for es in [samples,*endpoint] for v in es)+a.SPREAD_MULTIPLIER*endspread
                comparison=a.agreement(means[1]-means[0],probe['expected_difference'],resolution)
                row['probes'].append({'epsilon':probe['epsilon'],'expected_difference':probe['expected_difference'],
                    'observed_difference':means[1]-means[0],'endpoint_primal_repeats':endpoint,'resolution_floor':resolution,
                    'finite_difference':(means[1]-means[0])/probe['effective_step'],'analytic':planned['analytic'],
                    'represented_derivative':probe['expected_difference']/probe['effective_step'],
                    'eligible_before_FD':probe['eligible'],**comparison})
            pair=row['selected_local_pair'];by={p['epsilon']:p for p in row['probes']}
            passed=bool(pair and all(by[e]['agrees'] for e in pair))
            finer=[p['epsilon'] for p in row['probes'] if pair and p['epsilon']<min(pair) and p['outcome']=='resolved_disagreement']
            row.update(pair_passed=passed,finer_resolved_disagreements=finer);result['directions'].append(row)
        assert result['sealed_plan']==seal and result['plan_sha256']==a._seal(seal)
        assert all(v['pair_passed'] and not v['finer_resolved_disagreements'] for v in result['directions']), 'Prospective logit derivative pair failed'
        result['status']='passed';result['can_admit_new_child']=True
    except BaseException as error:result['error']=repr(error)
    finally:
        result['frozen_before']=identity;result['frozen_after']=deepcopy(frozen());result['frozen_unchanged']=result['frozen_after']==identity
        if not result['frozen_unchanged']:result.update(status='failed',can_admit_new_child=False,error='Frozen identity changed')
        assert {k:a._record(v) for k,v in values.items()}==before
    return result
