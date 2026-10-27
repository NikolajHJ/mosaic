"""Separate native component qualification at unchanged chemistry boundaries.

The existing prospective policies and full FP64 reference are used unchanged.
This capsule cannot certify the combined chemistry pilot or submit optimization.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'chemistry-native-qualification-03'
VERSION='chemistry-native-component-qualification-v1'
CAP=512*1024**2


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def binding(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink(),'Ordinary proof required')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def inherited(plan):
    source=Path(plan['diagnostic_source']['path'])
    require(sha(source/'source_manifest.json')==plan['diagnostic_source']['manifest_sha256'],'Parent source differs')
    d=module(source/'scripts/chemistry_endpoint_diagnostic.py','_native_qualification_parent')
    d.verify_inputs(plan)
    return d


def raw_directions(arm):
    if arm['id']=='protenix':
        # The original failure is in logit coordinates. Never invent an
        # authentic raw-coordinate original contrast from that logit report.
        return {'X':[],'E_upper':[]},'not_preexisting_at_new_endpoint'
    rows=read(arm['failed_report']['path'])['rows']
    return {field:[{k:r[k] for k in ('row','source','target')} for r in rows if r['field']==field]
            for field in ('X','E_upper')},'saved_failed_endpoint'


def size(output):
    total=0
    for p in output.rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Special qualification output')
        if p.is_file():total+=p.stat().st_size
    require(total<CAP-2*1024**2,'Qualification output cap exhausted')
    return total


def setup(plan,arm,output,life):
    d=inherited(plan);source=Path(arm['science_source']['path'])
    sys.path[:0]=[str(source/'src'),str(source/'scripts'),str(Path(arm['adapter_source']['path'])/'scripts')]
    os.environ['PYTHONPATH']=os.pathsep.join((str(source/'src'),str(source/'scripts'),os.environ.get('PYTHONPATH','')))
    import jax
    require(not jax.config.jax_enable_x64,'Original FP32 process required')
    jax.config.update('jax_default_matmul_precision','highest')
    require((jax.default_backend()=='gpu')==(arm['id']=='protenix'),'Original backend differs')
    loss,key,measure=d.native_loss(arm,output,life)
    # This is the authentic segment boundary, rather than a derived early pilot.
    local={**arm,'baseline':arm['boundary']};logits=d.exact_logits(local,key,output)
    return d,loss,key,measure,logits


def fp32(plan,arm,output,life):
    d,loss,key,measure,logits=setup(plan,arm,output,life)
    import numpy as np
    import jax.numpy as jnp
    from mosaic.small_molecule.state import RelaxedGraph
    graph=logits.graph();values={f:np.asarray(getattr(graph,f)) for f in ('X','E_upper')}
    if arm['id']=='protenix':
        rows=[measure(graph) for _ in range(3)]
        np.savez_compressed(output/'production.npz',**values,
            primal=np.asarray([r['primal'] for r in rows],np.float32),
            ad_value=np.asarray([r['AD_value'] for r in rows],np.float32),
            **{'gradient_'+f:np.stack([np.asarray(getattr(r['gradients'],f)) for r in rows]) for f in values})
        return {'status':'passed_exact_FP32_boundary_measurements','data':binding(output/'production.npz')}
    import admission
    require(admission.VERSION=='prospective-endpoint-admission-v3','Existing unchanged raw policy required')
    original,scope=raw_directions(arm);planned=admission.plan(values,original,original_direction_scope=scope)
    write(output/'geometry-plan.json',planned);calls=[];cache={};cursor=Counter()
    def point(v):return RelaxedGraph(jnp.asarray(v['X']),jnp.asarray(v['E_upper']))
    def compute(v):
        size(output);r=measure(point(v));path=output/f'point-{len(calls):04d}.npz'
        np.savez_compressed(path,**v,primal=r['primal'],ad_value=r['AD_value'],
            **{'gradient_'+f:np.asarray(getattr(r['gradients'],f)) for f in v})
        calls.append(binding(path));return r
    def call(v,kind):
        digest=hashlib.sha256(b''.join(v[f].tobytes() for f in ('X','E_upper'))).hexdigest()
        if digest not in cache:
            cache[digest]=[compute(v) for _ in range(3)]
        n=cursor[(digest,kind)];cursor[(digest,kind)]+=1;r=cache[digest][n%3]
        return np.float32(r['primal']) if kind=='primal' else (np.float32(r['AD_value']),
            {f:np.asarray(getattr(r['gradients'],f)) for f in ('X','E_upper')})
    result=admission.evaluate(planned,lambda v:call(v,'primal'),lambda v:call(v,'ad'),life['frozen'],
        seal_callback=lambda v:write(output/'selected-pair-before-FD.json',v))
    write(output/'native-admission.json',result)
    # No new evaluations in wrong-gradient controls: all points were presealed.
    for label,factor in [('zero',0.),('reversed',-1.),('scaled',1.5)]:
        cursor.clear();count=len(calls)
        def bad(v):
            scalar,g=call(v,'ad');return scalar,{k:x*np.float32(factor) for k,x in g.items()}
        negative=admission.evaluate(planned,lambda v:call(v,'primal'),bad,life['frozen'])
        write(output/(label+'-control.json'),negative)
        require(not negative['can_admit_new_child'] and len(calls)==count,'Wrong gradient admitted or unsealed point added')
    write(output/'native-points.json',{'calls':calls,'cached_negative_controls':True})
    require(result['can_admit_new_child'],'Prospective native bond/atom measurement failed')
    return {'status':'passed_native_component_only','native_admission':binding(output/'native-admission.json'),
            'combined_chemistry_admission':False}


def reference(plan,arm,output,life):
    d=inherited(plan);source=Path(arm['science_source']['path'])
    sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.features import bootstrap_native_assets
    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.optimizers import evaluate_graph,_nonfinite_paths
    from mosaic.small_molecule.state import RelaxedGraph
    import protenix_precision_reference as precision
    import protenix_precision_admission as policy
    require(jax.config.jax_enable_x64 and len(jax.devices('gpu'))==1
            and 'L40S' in jax.devices('gpu')[0].device_kind,'Separate full FP64 single L40S required')
    jax.config.update('jax_default_matmul_precision','highest')
    prod=read(output.parent/'production/assessment.json')
    require(prod['result']['status']=='passed_exact_FP32_boundary_measurements'
            and prod['result']['data']==binding(output.parent/'production/production.npz'),'Production evidence differs')
    with np.load(output.parent/'production/production.npz',allow_pickle=False) as z:fp={k:z[k].copy() for k in z.files}
    with np.load(arm['boundary']['path'],allow_pickle=False) as z:
        for f in ('X','E_upper'):np.testing.assert_array_equal(fp[f],z[f])
    cache_dir=Path('/work3/s234818/mosaic/cache/mosaic/protenix');bootstrap_native_assets(cache_dir,include_model=True)
    with precision.native_precision_context() as hooks:
        original_model=FrozenProtenixTiny.load(cache_dir=cache_dir)
        ready=read(arm['context_ready']['path'])
        original_context=load_context(Path(arm['context_ready']['path']).parent/'context',ready['context_files'])
        frozen=lambda:{'model_digest':array_digest(original_model),'context_digest':array_digest(original_context)}
        require(frozen()==arm['frozen_identity'],'Original reference inputs differ')
        model,ma=precision.promote_tree(original_model,label='model');context,ca=precision.promote_tree(original_context,label='context')
        require(frozen()==arm['frozen_identity'],'Original frozen inputs mutated by promotion')
        write(output/'promotion.json',{'model':ma,'context':ca})
        identity={'model':array_digest(model),'context':array_digest(context)}
        life['frozen']=lambda:{'model':array_digest(model),'context':array_digest(context)};life['frozen_before']=identity
        precision.floating_dtype_audit(model);precision.floating_dtype_audit(context)
        loss=model.build_loss(context=context,loss=NativeContactLoss(8.0,map_protenix_pocket(context,read(arm['pocket']['path']))))
        key=jax.random.key(17);devices={'X':jax.devices('cpu')[0],'E_upper':jax.devices('gpu')[0]}
        def graph(v):return RelaxedGraph(**{f:jax.device_put(jnp.asarray(v[f],dtype=jnp.float64),devices[f]) for f in devices})
        audit_module=module(Path(plan['precision_source']['path'])/'scripts/protenix_precision_native.py','_chem_native_dtype_audit')
        dtype_audit=audit_module.audit_jaxpr(eqx.filter_make_jaxpr(lambda g:loss(g,key=key))(graph(fp))[0])
        write(output/'native-dtype-audit.json',dtype_audit)
        points=[];cache={};cursor=Counter()
        def compute(v):
            size(output);g=graph(v);primal,aux=loss(g,key=key);ad=evaluate_graph(loss,g,key)
            require(not _nonfinite_paths((primal,aux,ad)),'Nonfinite FP64 reference')
            row={'primal':np.asarray(primal),'ad_value':np.asarray(ad.value),
                 'gradients':{f:np.asarray(getattr(ad.gradients,f)) for f in devices}}
            require(row['primal'].dtype==np.float64 and all(x.dtype==np.float64 for x in row['gradients'].values()),'Reference dtype differs')
            path=output/f'point-{len(points):04d}.npz'
            np.savez_compressed(path,**{f:np.asarray(v[f],np.float64) for f in devices},
                primal=row['primal'],ad_value=row['ad_value'],**{'gradient_'+f:x for f,x in row['gradients'].items()})
            points.append(binding(path));return row
        baseline={f:fp[f].astype(np.float64) for f in devices}
        def ref(v,kind):
            digest=hashlib.sha256(b''.join(v[f].tobytes() for f in devices)).hexdigest()
            if digest not in cache:cache[digest]=[compute(v) for _ in range(3 if all(np.array_equal(v[f],baseline[f]) for f in devices) else 1)]
            i=cursor[(digest,kind)];cursor[(digest,kind)]+=1;r=cache[digest][i%len(cache[digest])]
            return r['primal'] if kind=='primal' else (r['ad_value'],r['gradients'])
        def prodcall(v,kind):
            for f in devices:np.testing.assert_array_equal(v[f],fp[f])
            i=cursor[('production',kind)]%3;cursor[('production',kind)]+=1
            return fp['primal'][i] if kind=='primal' else (fp['ad_value'][i],{f:fp['gradient_'+f][i] for f in devices})
        metadata={'full_model_float64':True,'parameters_float64':True,'native_features_float64':True,
            'native_operations_float64':True,'final_reduction_only':False,'validation_only':True,
            'dtype_audit':binding(output/'native-dtype-audit.json'),'promotion':binding(output/'promotion.json')}
        def run(label,factor=1.):
            cursor.clear()
            def ad(v):
                scalar,g=prodcall(v,'ad');return scalar,{f:(x*np.float32(factor)).astype(np.float32) for f,x in g.items()}
            directions,scope=raw_directions(arm)
            return policy.evaluate({f:fp[f] for f in devices},lambda v:prodcall(v,'primal'),ad,
                lambda v:ref(v,'primal'),lambda v:ref(v,'ad'),life['frozen'],original_directions=directions,
                scope=scope,reference_precision=metadata,original_observations=arm['failed_report'],
                seal_callback=lambda v:write(output/(label+'-selected-before-FD.json'),v))
        result=run('native');write(output/'native-admission.json',result)
        for label,factor in [('zero',0.),('reversed',-1.),('scaled',1.5)]:
            count=len(points);negative=run(label,factor);write(output/(label+'-control.json'),negative)
            require(not negative['can_admit_new_child'] and len(points)==count,'Wrong production gradient admitted')
        require(frozen()==arm['frozen_identity'] and life['frozen']()==identity,'Frozen original/reference state changed')
        write(output/'native-points.json',{'calls':points,'reference_hooks':hooks,'original_frozen_unchanged':True})
        require(result['can_admit_new_child'],'Separate FP64 native component qualification failed')
        return {'status':'passed_native_component_only','native_admission':binding(output/'native-admission.json'),
                'combined_chemistry_admission':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--arm',choices=('saved','protenix','boltz'),required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',choices=('production','reference'))
    p.add_argument('--startup-only',action='store_true');a=p.parse_args(argv)
    require(sha(a.plan)==a.plan_sha256,'Plan changed');plan=read(a.plan);d=inherited(plan)
    if a.arm=='saved':
        require(a.worker is None,'No saved-analysis worker');d.ROOT=ROOT;return d.main(argv)
    expected=ROOT/('cpu/payload/startup-'+a.arm if a.startup_only else 'gpu-'+a.arm+'/payload/native')
    if a.worker:expected=expected/a.worker
    require(a.output==expected and not a.output.exists(),'Fresh declared qualification output required')
    a.output.mkdir(parents=True,exist_ok=False)
    if a.startup_only:
        write(a.output/'startup.json',{'status':'passed_actual_startup','optimizer_updates':0,'numerical_imports':False});return
    require(os.environ['LSB_QUEUE']=='gpul40s','Allocated GPU required')
    if a.arm=='protenix' and a.worker is None:
        started=time.monotonic()
        for worker in ('production','reference'):
            env=dict(os.environ);env['JAX_ENABLE_X64']='1' if worker=='reference' else '0'
            command=[sys.executable,str(Path(__file__).resolve()),'--plan',str(a.plan),'--plan-sha256',a.plan_sha256,
                '--arm',a.arm,'--output',str(a.output/worker),'--worker',worker]
            with (a.output/(worker+'.log')).open('x') as stream:
                remaining=5300-(time.monotonic()-started)
                require(remaining>0,'Whole Protenix qualification deadline expired')
                subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        r=read(a.output/'reference/assessment.json')
        require(r['status']=='native_component_qualified_not_continuation_admission','Reference did not qualify')
        write(a.output/'assessment.json',{'status':r['status'],'reference':binding(a.output/'reference/assessment.json'),
            'optimizer_updates':0,'can_submit_continuation':False});return
    life={};record={'version':VERSION,'status':'failed','arm':a.arm,'optimizer_updates':0,'can_submit_continuation':False}
    try:
        with d.deadline(5300):
            record['result']=reference(plan,plan['arms'][a.arm],a.output,life) if a.worker=='reference' else fp32(plan,plan['arms'][a.arm],a.output,life)
        record['status']='native_component_qualified_not_continuation_admission' if a.worker!='production' else 'production_measurements_complete'
    finally:
        try:
            if life.get('frozen'):
                require(life['frozen']()==life['frozen_before'],'Frozen native model/context changed');record['frozen_unchanged']=True
            d.verify_inputs(plan);record['protected_inputs_unchanged']=True
        except BaseException as e:record.update(status='failed',preservation_error=repr(e))
        if life.get('worker'):life['worker'].close()
        record['output_bytes']=size(a.output);write(a.output/'assessment.json',record)
    require(record['status']!='failed','Native component qualification failed')


if __name__=='__main__':main()
