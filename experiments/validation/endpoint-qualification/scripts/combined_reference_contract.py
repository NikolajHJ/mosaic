"""Independent FUTURE-adoption reader for combined reference evidence.

No numerical dependency is imported at module import. ``verify`` requires a
real allocated HPC CPU job before loading arrays. This reader does not submit,
restart, certify an original failed study, or grant optimizer permission.

Expectation schema ``combined-reference-expectation-v1`` is an authenticated
input of the future integration: coefficient, preparation_identity,
preparation_snapshot binding (arrays/receipt/identity), baseline binding and
baseline_fields, key_data, frozen production/reference dictionaries,
gpu_job_id/gpu_lsf binding, exact reference_command/reference_environment,
reference_identity_subset, source manifests and immutable input bindings.
Do not derive these expectations from the report being checked.

This is a compatibility reader for the explicitly named source03 logit-v1
contract, not a prospective default checker. It independently reconstructs the
numerical witnesses, native transport and fixed-target identity. It relies on
authenticated execution of the pinned chemistry source for chemistry values
and VJPs; it does not independently rerun that implementation. Adoption also
requires explicit original chemistry loss/state/trajectory source bindings,
allocated coefficient/VJP regressions, complete native/pilot/runtime/storage
checks and a fresh integration. This module alone cannot establish those.

Current source03 does not save a prepared-target snapshot. Its producer-side
qualification remains separate; future adoption needs that additional bound
artifact and allocated regressions before using this reader.

A separate allocated CPU supplement may reconstruct that snapshot from the
exact cohort FP32 logits, original config/search/progress/global10000/spec/source.
Its identity MUST equal the reported producer identity exactly; no alternate
target or selection is permitted. Timings do not enter preparation identity.
This does not require altering/restarting a queued GPU job for metadata alone.
Future producers should serialize prepared.record() at every cohort directly.
"""
from copy import deepcopy
import base64
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import re

VERSION='combined-reference-contract-v1'
EXPECTATION_VERSION='combined-reference-expectation-v1'
REFERENCE_POLICY='protenix-fixed-chemistry-full-reference-v1'
PRECISION_POLICY='authenticated-full-FP64-scalar-v1'
EPSILONS=(.01,.003,.001,.0003,.0001,.00003)
FIELDS={'atoms':(25,6),'bonds':(300,5)}
RELATIVE_TOLERANCE=.15
REPEATS=3
ULPS=8.
SPREAD=4.
SIGNAL=2./RELATIVE_TOLERANCE
MINIMUM_TANGENT=1e-7
PRECISION={'full_model_float64':True,'parameters_float64':True,
    'native_features_float64':True,'native_operations_float64':True,'final_reduction_only':False}


def require(value,message):
    if not value:raise ValueError(message)


def allocated_numpy():
    require(os.environ.get('LSB_QUEUE')=='hpc' and re.fullmatch(r'[1-9]\d*',os.environ.get('LSB_JOBID','')),
            'Combined reference reader requires a real allocated HPC CPU job')
    import numpy as np
    return np


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()


def read(path):
    def unique(rows):
        value={}
        for k,v in rows:
            require(k not in value,'Duplicate evidence JSON key');value[k]=v
        return value
    path=Path(path);binding(path)
    return json.loads(path.read_bytes(),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite JSON token '+value)))


def binding(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and p.is_file()
            and all(not v.is_symlink() for v in (p,*p.parents)),'Ordinary absolute evidence file required')
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}


def bound(row):
    require(type(row) is dict and set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int
        and row['bytes']>=0 and re.fullmatch('[a-f0-9]{64}',row['sha256']),'Closed raw-file binding required')
    require(binding(row['path'])==row,'Changed or missing evidence bytes: '+row['path'])
    return Path(row['path'])


def subset(actual,expected):
    return isinstance(actual,dict) and all(k in actual and
        (subset(actual[k],v) if isinstance(v,dict) else actual[k]==v) for k,v in expected.items())


def source(row):
    require(set(row)=={'path','manifest_sha256','files'},'Closed expected source declaration required')
    root=Path(row['path']);b=binding(root/'source_manifest.json')
    require(b['sha256']==row['manifest_sha256'],'Source manifest changed')
    m=read(b['path']);require(len(m['files'])==row['files'],'Source member count changed')
    actual=set()
    for p in root.rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Special source member')
        if p.is_file():actual.add(p.relative_to(root).as_posix())
    require(actual==set(m['files'])|{'source_manifest.json'},'Source closure changed')
    for name,digest in m['files'].items():
        part=PurePosixPath(name)
        require(not part.is_absolute() and '..' not in part.parts and '\\' not in name,'Unsafe source member')
        require(binding(root/name)['sha256']==digest,'Source member changed')
    return b


def array_file(np,row):
    path=bound(row)
    with np.load(path,allow_pickle=False) as f:
        arrays={k:f[k].copy() for k in f.files}
    require(arrays and all(a.dtype.kind in 'fiub' and bool(np.isfinite(a).all()) for a in arrays.values()),
            'Nonfinite, object or unsupported raw array')
    return arrays


def recorded_array(np,row,dtype,shape):
    require(set(row)=={'dtype','shape','sha256','base64'} and row['dtype']==dtype and row['shape']==list(shape),
            'Recorded array schema differs')
    b=base64.b64decode(row['base64'],validate=True)
    require(sha(b)==row['sha256'],'Recorded array hash differs')
    array=np.frombuffer(b,dtype=dtype).reshape(shape).copy()
    require(bool(np.isfinite(array).all()),'Nonfinite recorded array')
    return array


def same_array(np,a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def number(value):
    require(type(value) in (int,float) and math.isfinite(value),'Finite recorded number required')
    return float(value)


def close_number(actual,expected):
    # Witness reconstruction tolerance, not an enlarged derivative tolerance.
    require(math.isclose(number(actual),number(expected),rel_tol=2e-12,abs_tol=1e-25),
            'Recorded numerical witness does not match independent reconstruction')


def scalar_record(np,row,value):
    require(row['dtype']=='float64' and row.get('resolution_scope')==
            'independent_full_FP64_binding_and_chemistry_not_FP32_wrapper','Actual FP64 scalar policy missing')
    close_number(row['value'],float(value));close_number(row['resolution_ulp'],float(abs(np.spacing(np.float64(value)))))


def agreement(observed,predicted,uncertainty):
    residual=abs(observed-predicted)
    low=max(max(0.,abs(observed)-uncertainty),abs(predicted))
    high=max(abs(observed)+uncertainty,abs(predicted))
    passed=abs(observed)>uncertainty and observed*predicted>0 and residual+uncertainty<=RELATIVE_TOLERANCE*low
    wrong=max(0.,residual-uncertainty)>RELATIVE_TOLERANCE*high
    return bool(passed),'agreement' if passed else 'resolved_disagreement' if wrong else 'unresolved_precision'


def _verify(report_path,expected):
    """Recompute witnesses from bound arrays; return a non-dispatch certificate.

    Known flat chemistry components are permitted: their weighted value/gradient
    need not be nonzero. The explicit legacy combined v1 contract still requires
    informative combined atom/bond contrasts; this reader does not relabel a flat
    entire combined field as passing that older contract.
    """
    np=allocated_numpy();expected=deepcopy(expected);root=Path(report_path).parent
    require(expected['version']==EXPECTATION_VERSION,'Closed expected contract version required')
    require(type(expected['key_data']) is list and len(expected['key_data'])==2
        and all(type(v) is int and 0<=v<2**32 for v in expected['key_data']),'Exact native uint32[2] key required')
    require(type(expected['coefficient']) in (int,float) and expected['coefficient']>0
            and math.isfinite(expected['coefficient']),'Positive fixed coefficient required')
    source_proofs=[source(row) for row in expected['sources']]
    require(source_proofs and expected['inputs'],'External source/input pins required')
    for row in expected['inputs']:bound(row)
    teacher=read(bound(expected['preparation_snapshot']))
    if set(teacher)=={'arrays','receipt','identity'}:
        receipt=teacher['receipt']
    else:
        require({'arrays','identity','preparation_seconds','search_seconds'}<=set(teacher),
                'Complete detached target snapshot required')
        receipt={k:v for k,v in teacher.items() if k not in ('arrays','identity','preparation_seconds','search_seconds')}
    # Match original preparation.digest exactly (default JSON separators).
    target=sha(json.dumps({'arrays':teacher['arrays'],'receipt':receipt},sort_keys=True,allow_nan=False).encode())
    require(target==teacher['identity']==expected['preparation_identity'],'Prepared targets or receipt mutated')
    baseline=array_file(np,expected['baseline']);values={}
    require(set(expected['baseline_fields'])==set(FIELDS),'Explicit baseline field mapping required')
    for field,shape in FIELDS.items():
        a=baseline[expected['baseline_fields'][field]]
        require(a.dtype==np.float32 and a.shape==shape,'Original baseline logits differ');values[field]=a
    report=read(report_path)
    require(report['version']=='prospective-fixed-reference-logit-admission-v1'
        and report['reference_policy']==REFERENCE_POLICY and report['reference_measurement_precision_policy']==PRECISION_POLICY,
        'Producer reference policy differs')
    require(report['status']=='passed' and report['can_admit_new_child'] is True
        and report['can_certify_original_study'] is False and report['original_failure_preserved'] is True
        and report['original_optimizer_or_reference_changed'] is False,'Producer status/ancestry scope differs')
    require(report['epsilons']==list(EPSILONS) and report['relative_tolerance']==RELATIVE_TOLERANCE
        and report['selection_before_perturbed_forwards'] is True,'Grid/tolerance/selection policy changed')
    require(report['repair_weight']==expected['coefficient'] and report['preparation_identity']==target,
            'Coefficient/fixed preparation changed')
    require(report['softmax_probabilities_bitidentical_between_precisions'] is False,'Cross-precision graph identity mislabelled')
    frozen={'production':expected['frozen']['production'],'reference':expected['frozen']['reference'],'preparation':target}
    require(report['frozen_before']==report['frozen_after']==frozen and report['frozen_unchanged'] is True,
            'Frozen input/model/context/teacher identity changed')
    require(report['sealed_plan']==read(root/'pair-seal.json')
        and report['plan_sha256']==sha(canonical(report['sealed_plan'])),'Changed or unsealed pair plan')
    seal=report['sealed_plan']
    require(seal['uses_observed_FD_for_selection'] is False and seal['no_pair_fallback'] is True,'Post-result selection forbidden')
    for field,shape in FIELDS.items():
        require(same_array(np,recorded_array(np,report['baseline'][field],'float32',shape),values[field]),'Report baseline changed')
        require(seal['baseline'][field]==report['baseline'][field],'Sealed baseline differs')
    job=expected['gpu_job_id'];require(re.fullmatch(r'[1-9]\d*',job),'Actual producer GPU job ID required')
    text=' '.join(bound(expected['gpu_lsf']).read_text().split())
    require(text.count('Subject: Job '+job+':')==1 and 'queue <gpul40s>' in text
        and text.count('Successfully completed.')==1 and 'Exited with exit code' not in text,'Same-job successful GPU LSF missing')
    session=root/'reference-session';invocation=read(session/'invocation.json');identity=read(session/'identity.json')
    require(invocation['job_id']==job and invocation['queue']=='gpul40s'
        and invocation['argv']==expected['reference_command'] and invocation['environment']==expected['reference_environment']
        and invocation['key_data']==expected['key_data'] and invocation['optimizer_updates']==0,'Reference invocation/key scope differs')
    require(invocation['environment'].get('XLA_FLAGS')=='--xla_gpu_autotune_level=0','Original independent reference environment differs')
    require(identity['precision']==PRECISION and identity['frozen']==expected['frozen']['reference']
        and subset(identity,expected['reference_identity_subset']),'Reference native/source/context identity differs')
    closing=read(session/'close.json');audit=closing['audit']
    require(closing['status']=='closed' and closing['frozen']==identity['frozen']
        and audit['status']=='passed_full_FP64_reference_audit' and audit['floating_operations_float64'] is True
        and audit['narrow_float_avals']==[] and audit['frozen_unchanged'] is True
        and audit['original_results_changed'] is False and audit['optimizer_updates']==0,'Incomplete reference close/operation audit')
    for name in ('forward_dtype_audit','AD_dtype_audit'):
        typed=read(bound(audit[name]));counts=typed['aval_counts']
        require(typed['status']=='passed' and typed['floating_operations_float64'] is True
            and typed['narrow_float_avals']==[] and counts.get('float64',0)>0
            and not any(v and k.startswith(('float','bfloat','complex')) and k!='float64' for k,v in counts.items()),
            'Narrow or absent native reference floating-operation proof')
    index=read(session/'client-raw-point-index.json');oracle=read(bound(audit['raw_points']))
    records=report['reference_records'];calls=report['calls']
    require(index['requests']==audit['requests']==len(records)==len(calls)==78
        and index['cached_or_interpolated_points']==0 and oracle['cached_evaluations']==0,
        'Complete actual callback/reference inventory required')
    require(index['reference_frozen']==oracle['frozen_before']==oracle['frozen_after']==identity['frozen'],
            'Oracle frozen-state changed')
    require(len(index['points'])==len(oracle['requests'])==78,'Missing native raw point inventory')
    require([r['kind'] for r in calls]==['baseline_primal']*3+['baseline_AD']*3+['perturbed_primal']*72,
            'Original independent callback order/repeats changed')
    raw=[]
    for i,(record,call,transport,worker) in enumerate(zip(records,calls,index['points'],oracle['requests'])):
        c=array_file(np,record['raw']);client=array_file(np,transport['raw']);native=array_file(np,worker['raw'])
        response=read(bound(transport['response']));is_ad=call['kind']=='baseline_AD'
        require(all(type(v) is bool for v in (record['gradient'],transport['gradient'],worker['gradient']))
                and transport['id']==worker['id']==response['id']==i and record['gradient']==transport['gradient']==worker['gradient']==is_ad,
                'Reference call ordering/kind changed')
        for field,shape in FIELDS.items():
            require(c[field].dtype==np.float32 and c[field].shape==shape
                and sha(c[field].tobytes())==call['point'][field],'Actual represented callback point changed')
            graph_field='X' if field=='atoms' else 'E_upper'
            p=c[graph_field];require(p.dtype==np.float64 and p.shape==shape,'Reference graph dtype/shape changed')
            require(same_array(np,p,client[graph_field]) and same_array(np,p,native[graph_field]),'Raw transport graph changed')
            logits=c[field].astype(np.float64);z=logits-logits.max(-1,keepdims=True);prob=np.exp(z);prob/=prob.sum(-1,keepdims=True)
            tolerance=32*np.finfo(np.float64).eps*np.maximum(abs(prob),abs(p))+32*np.finfo(np.float64).smallest_subnormal
            require(bool((np.abs(prob-p)<=tolerance).all()),'Independent logit-to-graph mapping failed')
            if is_ad:require(c['gradient_'+field].dtype==np.float64 and c['gradient_'+field].shape==shape,'Reference AD array schema differs')
            if i<6:require(same_array(np,c[field],values[field]),'Baseline independent callback changed inputs')
        digest=sha(c['X'].tobytes()+c['E_upper'].tobytes())
        require(transport['graph_sha256']==worker['graph_sha256']==response['graph_sha256']==digest,
                'Native reference graph hash differs')
        for point in (client,native):
            require(point['key_data'].dtype==np.uint32 and point['key_data'].tolist()==expected['key_data'],'Actual native key changed')
        require(transport['key_data']==worker['key_data']==response['key_data']==expected['key_data'],'Recorded native key changed')
        require(all(c[k].shape==() and c[k].dtype==np.float64 for k in ('total','native','weighted_chemistry')),'Combined precision narrowed')
        close_number(c['total'].item(),(c['native']+c['weighted_chemistry']).item())
        close_number(c['native'].item(),client['value'].item());require(abs(c['native'].item()-native['primal'].item())<=1e-10,'Native primal/AD parity failed')
        close_number(response['value'],client['value'].item())
        require(response['raw']==worker['raw'],'Worker response/raw inventory binding differs')
        if is_ad:
            for field in ('X','E_upper'):
                require(same_array(np,client['gradient_'+field],native['gradient_'+field]),'Native raw AD binding changed')
                require(same_array(np,client['gradient_'+field],np.asarray(response['gradients'][field],dtype=np.float64)),
                        'RPC response/raw AD arrays differ')
        raw.append(c)
    production=report['production_records'];require(len(production)==5,'Before/three AD/after production replay missing')
    prod=[]
    for record in production:
        p=array_file(np,record['raw']);require(record['frozen']==expected['frozen']['production'],'Production model/context changed')
        require(all(p[k].shape==() and p[k].dtype==np.float64 for k in ('primal','AD_value')),'Production recorded scalar schema differs')
        for field,shape in FIELDS.items():
            require(same_array(np,p[field],values[field]),'Production baseline changed')
            g=p['gradient_'+field];require(g.dtype==np.float32 and g.shape==shape
                and sha(g.tobytes())==record['gradients'][field],'Production AD digest/schema differs')
            graph='X' if field=='atoms' else 'E_upper'
            require(p[graph].dtype==np.float32 and p[graph].shape==shape and bool((p[graph]>=0).all())
                and bool(np.allclose(p[graph].sum(-1),1,rtol=0,atol=2e-6)),'Invalid production graph')
            if graph in baseline:require(same_array(np,p[graph],baseline[graph]),'Original saved raw graph replay changed')
        close_number(record['primal'],p['primal'].item());close_number(record['AD_value'],p['AD_value'].item())
        prod.append(p)
    require(all(all(same_array(np,p[k],prod[0][k]) for k in prod[0]) for p in prod),'Explicit bitwise production replay failed')
    require(all(abs(prod[i+1]['primal'].item()-raw[i+3]['total'].item())<=2e-5 for i in range(REPEATS)),
            'Independent production/reference combined scalar parity failed')
    measurement=report['baseline_measurement'];scalar_samples=[r['total'].item() for r in raw[:6]]
    for r,value in zip(measurement['primal_repeats']+measurement['AD_scalar_repeats'],scalar_samples):scalar_record(np,r,value)
    require(len(measurement['primal_repeats'])==len(measurement['AD_scalar_repeats'])==REPEATS,'Independent repeat count differs')
    primal_mean=sum(scalar_samples[:3])/3
    require(all(abs(v-primal_mean)<=2e-5 for v in scalar_samples[3:]),'Reference independent-primal/AD parity failed')
    spread=max(scalar_samples)-min(scalar_samples);floor=ULPS*max(float(abs(np.spacing(np.float64(v)))) for v in scalar_samples)+SPREAD*spread
    close_number(measurement['resolution_floor'],floor);close_number(seal['resolution_floor'],floor)
    gradients={}
    for field,shape in FIELDS.items():
        repeats=[recorded_array(np,r[field],'float64',shape) for r in measurement['raw_gradient_repeats']]
        require(len(repeats)==3 and all(same_array(np,g,prod[i+1]['gradient_'+field].astype(np.float64)) for i,g in enumerate(repeats)),
                'Recorded AD differs from actual unchanged production arrays')
        gradients[field]=np.mean(repeats,axis=0)
    cursor=6;verified=[]
    require(len(report['directions'])==len(seal['directions'])==2,'Both declared combined fields required')
    for field,planned,result in zip(FIELDS,seal['directions'],report['directions']):
        gradient=gradients[field];row=int(np.ptp(gradient,-1).argmax());src=int(gradient[row].argmin());dst=int(gradient[row].argmax())
        tangent=float(gradient[row,dst]-gradient[row,src]);require(tangent>MINIMUM_TANGENT,'Declared legacy combined field is not informative')
        for independent in raw[3:6]:
            rg=independent['gradient_'+field];observed=float(rg[row,dst]-rg[row,src])
            require(observed*tangent>0 and abs(observed-tangent)<=RELATIVE_TOLERANCE*max(abs(observed),abs(tangent)),
                    'Independent baseline reference AD contradicts production')
        for d in (planned,result):
            require((d['field'],d['row'],d['source'],d['destination'])==(field,row,src,dst),'Direction selected from results or changed')
            close_number(d['analytic'],tangent)
        eligible=[];outcomes={}
        require(len(planned['probes'])==len(result['probes'])==len(EPSILONS),'Missing predeclared scale')
        for epsilon,probe,observed in zip(EPSILONS,planned['probes'],result['probes']):
            left=values[field].copy();right=values[field].copy()
            for sign,point in ((-1,left),(1,right)):
                point[row,src]=np.float32(point[row,src]-np.float32(sign*epsilon));point[row,dst]=np.float32(point[row,dst]+np.float32(sign*epsilon))
            difference=right.astype(np.float64)-left.astype(np.float64)
            prediction=float(np.sum(gradient*difference,dtype=np.float64));step=float((difference[row,dst]-difference[row,src])/2)
            require(step>0 and math.isfinite(prediction),'Unrepresentable declared stencil')
            signals=[abs(float(np.sum(gradient*(p.astype(np.float64)-values[field]),dtype=np.float64))) for p in (left,right)]
            is_eligible=all(v>SIGNAL*floor for v in signals);eligible.append(is_eligible)
            require(probe['epsilon']==observed['epsilon']==epsilon and probe['eligible']==observed['eligible_before_FD']==is_eligible,'Pre-FD scale eligibility changed')
            require(same_array(np,recorded_array(np,probe['left'],'float32',FIELDS[field]),left)
                and same_array(np,recorded_array(np,probe['right'],'float32',FIELDS[field]),right),'Sealed represented stencil changed')
            close_number(probe['expected_difference'],prediction);close_number(probe['effective_step'],step)
            endpoints=[]
            for point in (left,right):
                rows=raw[cursor:cursor+REPEATS];cursor+=REPEATS
                for r in rows:
                    require(same_array(np,r[field],point) and all(same_array(np,r[f],values[f]) for f in FIELDS if f!=field),
                            'FD callback point moved beyond its predeclared stencil')
                endpoints.append([r['total'].item() for r in rows])
            for stored,actual in zip(observed['endpoint_primal_repeats'],endpoints):
                require(len(stored)==REPEATS,'Missing endpoint repeats')
                for r,value in zip(stored,actual):scalar_record(np,r,value)
            means=[sum(v)/REPEATS for v in endpoints];endspread=max(spread,*[max(v)-min(v) for v in endpoints])
            resolution=ULPS*max(float(abs(np.spacing(np.float64(v)))) for group in (scalar_samples,*endpoints) for v in group)+SPREAD*endspread
            agrees,outcome=agreement(means[1]-means[0],prediction,resolution);outcomes[epsilon]=outcome
            require(type(observed['agrees']) is bool and observed['agrees']==agrees and observed['outcome']==outcome,'Forged derivative agreement flag')
            close_number(observed['observed_difference'],means[1]-means[0]);close_number(observed['expected_difference'],prediction)
            close_number(observed['resolution_floor'],resolution);close_number(observed['finite_difference'],(means[1]-means[0])/step)
        pairs=[[x,y] for x,y,a,b in zip(EPSILONS,EPSILONS[1:],eligible,eligible[1:]) if a and b]
        pair=pairs[-1] if pairs else None
        require(planned['selected_local_pair']==result['selected_local_pair']==pair and pair is not None,'Finest eligible pair selection changed or inconclusive')
        require(all(outcomes[e]=='agreement' for e in pair) and not any(e<min(pair) and o=='resolved_disagreement' for e,o in outcomes.items()),
                'Material derivative disagreement or unresolved required pair')
        require(result['pair_passed'] is True and result['finer_resolved_disagreements']==[],'Producer pair witness differs')
        verified.append({'field':field,'selected_pair':pair,'status':'independently_verified'})
    require(cursor==len(raw),'Unconsumed or missing declared raw point')
    return {'version':VERSION,'status':'passed_independent_combined_reference_witnesses',
        'report':binding(report_path),'coefficient':expected['coefficient'],'preparation_identity':target,
        'validation_gpu_job_id':job,'reference_points':len(raw),'production_replays':len(prod),'directions':verified,
        'source_manifests':source_proofs,'flat_chemistry_component_allowed':True,
        'can_submit_optimizer':False,'can_certify_or_restart_original_study':False,
        'scope':'One combined derivative cohort; complete native/pilots/runtime/storage and new production integration remain separate.'}


def verify(report_path,expected):
    """Fail closed on malformed/missing evidence as well as numerical errors."""
    try:return _verify(report_path,expected)
    except (KeyError,TypeError,ValueError,OSError,IndexError) as error:
        raise ValueError('Independent combined reference contract rejected: '+str(error)) from error
