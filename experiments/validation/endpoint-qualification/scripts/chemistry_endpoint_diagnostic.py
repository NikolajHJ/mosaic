"""Allocated fixed-input chemistry diagnostics; no admission or optimization."""
from contextlib import contextmanager
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

VERSION = 'chemistry-endpoint-diagnostic-v1'
EPSILONS = (.01, .003, .001, .0003, .0001)
BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE/'chemistry-endpoint-recovery-02'
MAX_BYTES = 256*1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x:
                      (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and all(not p.is_symlink() for p in path.parents), 'Ordinary absolute input required')
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def verify_inputs(plan):
    require(plan['version']==VERSION and plan['optimizer_updates']==0
            and plan['can_certify_or_restart_original_study'] is False
            and plan['epsilons']==list(EPSILONS), 'Diagnostic policy/scope differs')
    for row in plan['protected_inputs']:
        require(binding(row['path'])==row, 'Protected original input changed: '+row['path'])
    return True


def load(path, name):
    spec=importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Authenticated module unavailable')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def np_module():
    require(str(os.environ.get('LSB_JOBID','')).isdigit(), 'Allocated numerical execution required')
    import numpy as np
    return np


def pullback(probability, gradient):
    np=np_module();p=np.asarray(probability,dtype=np.float64);g=np.asarray(gradient,dtype=np.float64)
    require(p.shape==g.shape and np.isfinite(p).all() and np.isfinite(g).all(), 'Invalid saved gradient')
    return p*(g-(p*g).sum(-1,keepdims=True))


def saved_protenix(plan):
    """Recompose only actual saved scalars/gradients, without native inference."""
    np=np_module();arm=plan['arms']['protenix'];report=read(arm['failed_report']['path'])
    require(report['status']=='failed' and len(report['directions'])==1, 'Exact failed atom cohort required')
    d=report['directions'][0]
    require(d['field']=='atoms' and (d['row'],d['source'],d['destination'])==(2,1,3), 'Failed direction changed')
    row,src,dst=d['row'],d['source'],d['destination']
    output=[]
    for probe in d['probes']:
        eps=probe['epsilon'];points=[]
        for sign in (-1,1):
            path=arm['saved_probes'][f'atoms-{eps:g}-{sign:+d}.npz']['path']
            with np.load(path,allow_pickle=False) as z:
                require(float(z['repair_weight'])==.001, 'Saved chemistry coefficient differs')
                native=pullback(z['X'],z['contact_gradient_X'])
                combined=np.asarray(z['gradient_atom_logits'],dtype=np.float64)
                points.append({'sign':sign,'native_primal':float(z['contact_primal']),
                    'combined_primal':float(z['combined_primal']),
                    'weighted_chemistry_primal':float(z['combined_primal'])-float(z['contact_primal']),
                    'native_directional_AD':float(native[row,dst]-native[row,src]),
                    'combined_directional_AD':float(combined[row,dst]-combined[row,src]),
                    'weighted_chemistry_directional_AD':float((combined-native)[row,dst]-(combined-native)[row,src])})
        left,right=points
        output.append({'epsilon':eps,'original_agrees':probe['agrees'],
            'original_baseline_AD':d['analytic'], 'recorded_total_FD':probe['finite_difference'],
            'native_FD':(right['native_primal']-left['native_primal'])/(2*eps),
            'weighted_chemistry_FD':(right['weighted_chemistry_primal']-left['weighted_chemistry_primal'])/(2*eps),
            'points':points})
    return {'original_status':'failed','observations':output,
            'interpretation_scope':'Saved endpoint decompositions only; no cause or admission inferred automatically'}


def changed_logits(logits, field, row, src, dst, delta):
    import equinox as eqx
    value=getattr(logits,field)
    value=value.at[row,src].add(-delta).at[row,dst].add(delta)
    return eqx.tree_at(lambda x:getattr(x,field),logits,value)


@contextmanager
def deadline(seconds):
    require(type(seconds) is int and 0<seconds<=5400, 'Bounded native deadline required')
    def timeout(_signum,_frame):
        raise TimeoutError('Separate diagnostic internal deadline expired')
    old=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)


def native_loss(arm, output, life):
    """Use the exact original loaders, never a new pose or model checkpoint."""
    import jax
    import numpy as np
    from mosaic.small_molecule.state import graph_from_hard
    from mosaic.small_molecule.optimizers import evaluate_graph
    key=jax.random.key(17)
    hard=graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)])
    if arm['id']=='protenix':
        from mosaic.small_molecule.protenix import FrozenProtenixTiny
        from mosaic.small_molecule.io import load_context
        from mosaic.small_molecule.pocket import map_protenix_pocket
        from mosaic.small_molecule.losses import NativeContactLoss
        from mosaic.small_molecule.gradient_checks import array_digest
        model=FrozenProtenixTiny.load(cache_dir=Path('/work3/s234818/mosaic/cache/mosaic/protenix'))
        ready=read(arm['context_ready']['path'])
        context=load_context(Path(arm['context_ready']['path']).parent/'context',ready['context_files'])
        pocket=read(arm['pocket']['path']);tokens=map_protenix_pocket(context,pocket)
        loss=model.build_loss(context=context,loss=NativeContactLoss(8.0,tokens))
        def frozen():
            return {'model_digest':array_digest(model),'context_digest':array_digest(context)}
        life['model']=model;life['context']=context
    else:
        from mosaic.small_molecule.boltz2.bridge import AffinityWorker,affinity_loss
        worker=AffinityWorker([arm['worker_python'],'-u','-m','mosaic.small_molecule.boltz2.worker',
            '--context',arm['context']['path'],'--context-sha256',arm['context']['sha256'],
            '--checkpoint','/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt'],stderr=output/'worker.log')
        life['worker']=worker;loss=affinity_loss(worker)
        def frozen():
            value=worker.request('verify_identity')
            return {k:value[k] for k in ('model_digest','context_digest')}
        original=worker.request('native_reference')['values']
        actual,_,_=worker.evaluate(np.asarray(hard.X),np.asarray(hard.E_upper),gradient=False)
        np.testing.assert_allclose(actual,original,atol=2e-5,rtol=0)
        write(output/'native-hard-parity.json',{'status':'passed','atol':2e-5})
    life['frozen']=frozen;before=frozen()
    require(before==arm['frozen_identity'], 'Original frozen model/context changed')
    life['frozen_before']=before;write(output/'frozen-before.json',before)
    # Loss evaluation is independent from graph perturbation generation.
    def measure(graph):
        primal,aux=loss(graph,key=key);value=evaluate_graph(loss,graph,key)
        p,a=float(primal),float(value.value)
        require(np.isfinite(p) and np.isfinite(a) and abs(p-a)<=2e-5, 'Native primal/AD scalar parity failed')
        return {'primal':p,'AD_value':a,'gradients':value.gradients}
    return loss,key,measure


def exact_logits(arm,key,output):
    import jax
    import jax.numpy as jnp
    import numpy as np
    if arm['id']=='protenix':
        from protenix_initialization import ProtenixReplayLogits as Logits
    else:
        from mosaic.small_molecule.chemistry.state import GraphLogits as Logits
    with np.load(arm['baseline']['path'],allow_pickle=False) as z:
        logits=Logits(jnp.asarray(z['atom_logits']),jnp.asarray(z['bond_logits']))
        graph=logits.graph();fields={'atom_logits':logits.atoms,'bond_logits':logits.bonds,
            'X':graph.X,'E_upper':graph.E_upper,'key_data':jax.random.key_data(key)}
        for name,value in fields.items():
            np.testing.assert_array_equal(z[name],np.asarray(value))
            require(z[name].tobytes()==np.asarray(value).tobytes(), 'Exact five-field replay differs')
    write(output/'exact-replay.json',{'status':'passed_exact_five_field_replay','baseline':arm['baseline']})
    return logits


def native_protenix(plan,arm,output,life):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry import trajectory
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    loss,key,measure=native_loss(arm,output,life)
    logits=exact_logits(arm,key,output);spec=mosaic_spec(25)
    helper=load(Path(arm['adapter_source']['path'])/'scripts/full_model_pilot.py','_chem_endpoint_full_model')
    prepared=helper.restore_preparation(read(arm['preparation']['path']),spec)
    reference=read(arm['reference']['path'])['element_counts'];config=read(arm['config']['path'])
    options={'loss':FailureGuidanceConfig(**config['loss']),'repair_weight':.001,
        'element_js_weight':0.,'parity_atol':2e-5,'native_name':'contact','objective_mode':'native_plus_chemistry'}
    failed=read(arm['failed_report']['path'])['directions'][0]
    require(prepared.identity==read(arm['failed_report']['path'])['preparation_identity'], 'Fixed teacher identity changed')
    # The complete cohort is fixed before the first native or chemistry scalar.
    cohort=[('baseline',0.)]+[(f'{eps:g}-{sign:+d}',sign*eps) for eps in EPSILONS for sign in (-1,1)]
    write(output/'cohort.json',{'coordinate':{k:failed[k] for k in ('field','row','source','destination')},
        'points':cohort,'epsilons':list(EPSILONS),'supplementary_scales_certify_original':False})
    original_math=trajectory.failure_aware_repair_loss
    rows=[]
    for variant in ('original','accelerated'):
        # Original path has untouched bindings. Accelerated variant installs only
        # the authenticated existing matrix adapter; no search/teacher refresh.
        from contextlib import nullcontext
        manager=nullcontext()
        if variant=='accelerated':
            from gpu_matrix import install_matrix_loss
            manager=install_matrix_loss('matrix')
        with manager:
            for label,delta in cohort:
                state=logits if delta==0 else changed_logits(logits,failed['field'],failed['row'],failed['source'],failed['destination'],delta)
                value=trajectory.evaluate_combined(loss,state,prepared,spec,reference,key=key,config=options)
                graph=state.graph();raw=value.native.evaluation.gradients
                native=pullback(graph.X,raw.X);g=np.asarray(value.gradients.atoms,dtype=np.float64)
                row,src,dst=failed['row'],failed['source'],failed['destination']
                item={'variant':variant,'point':label,'delta':delta,'combined_primal':value.primal,
                    'combined_AD_value':value.value,'native_primal':value.native.primal,
                    'weighted_chemistry_primal':value.primal-value.native.primal,
                    'combined_directional_AD':float(g[row,dst]-g[row,src]),
                    'native_directional_AD':float(native[row,dst]-native[row,src]),
                    'weighted_chemistry_directional_AD':float((g-native)[row,dst]-(g-native)[row,src]),
                    'preparation_identity':prepared.identity,'repair_weight':value.repair_weight}
                rows.append(item)
                np.savez_compressed(output/(variant+'-'+label+'.npz'),atom_logits=np.asarray(state.atoms),
                    bond_logits=np.asarray(state.bonds),X=np.asarray(graph.X),E_upper=np.asarray(graph.E_upper),
                    native_gradient_X=np.asarray(raw.X),native_gradient_E_upper=np.asarray(raw.E_upper),
                    combined_gradient_atoms=np.asarray(value.gradients.atoms),combined_gradient_bonds=np.asarray(value.gradients.bonds),
                    key_data=np.asarray(jax.random.key_data(key)),**{k:v for k,v in item.items() if isinstance(v,(int,float))})
    require(trajectory.failure_aware_repair_loss is original_math, 'Original chemistry binding was not restored')
    return {'status':'diagnostics_completed_original_failure_preserved','original_check_status':'failed',
        'variant_points':rows,'preparation_identity':prepared.identity,'repair_weight':.001,
        'can_certify_or_restart_original_study':False}


def native_boltz(plan,arm,output,life):
    import jax
    import numpy as np
    from mosaic.small_molecule.state import RelaxedGraph
    loss,key,measure=native_loss(arm,output,life)
    logits=exact_logits(arm,key,output);template=logits.graph()
    failed=read(arm['failed_report']['path'])
    directions=failed['rows'];cohort=[]
    for d in directions:
        for eps in EPSILONS:
            for sign in (-1,1):
                cohort.append({'field':d['field'],'row':d['row'],'source':d['source'],'target':d['target'],
                    'epsilon':eps,'sign':sign})
    write(output/'cohort.json',{'directions':directions,'points':cohort,'epsilons':list(EPSILONS),
        'supplementary_scales_certify_original':False})
    rows=[]
    points=[None]+cohort
    for number,p in enumerate(points):
        graph=template
        if p:
            value=getattr(graph,p['field']);value=value.at[p['row'],p['source']].add(-p['sign']*p['epsilon'])
            value=value.at[p['row'],p['target']].add(p['sign']*p['epsilon'])
            require(np.all(np.asarray(value)>0), 'Original declared diagnostic perturbation leaves simplex interior')
            graph=RelaxedGraph(value,graph.E_upper) if p['field']=='X' else RelaxedGraph(graph.X,value)
        measured=measure(graph);g=measured['gradients']
        item={'point':p,'primal':measured['primal'],'AD_value':measured['AD_value']}
        if p:
            a=np.asarray(getattr(g,p['field']));item['directional_AD']=float(a[p['row'],p['target']]-a[p['row'],p['source']])
        else:
            item['baseline_directional_AD']={d['field']:float(np.asarray(getattr(g,d['field']))[d['row'],d['target']]-np.asarray(getattr(g,d['field']))[d['row'],d['source']]) for d in directions}
        rows.append(item)
        np.savez_compressed(output/f'point-{number:03d}.npz',X=np.asarray(graph.X),E_upper=np.asarray(graph.E_upper),
            gradient_X=np.asarray(g.X),gradient_E_upper=np.asarray(g.E_upper),primal=measured['primal'],AD_value=measured['AD_value'],key_data=np.asarray(jax.random.key_data(key)))
    return {'status':'diagnostics_completed_original_failure_preserved','original_check_status':'failed',
        'raw_points':rows,'can_certify_or_restart_original_study':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--arm',choices=('saved','protenix','boltz'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--startup-only',action='store_true')
    args=parser.parse_args(argv)
    require(sha(args.plan)==args.plan_sha256, 'Exact diagnostic plan required')
    plan=read(args.plan);verify_inputs(plan)
    expected=ROOT/('cpu/payload/startup-'+args.arm if args.startup_only else 'cpu/payload/saved' if args.arm=='saved' else 'gpu-'+args.arm+'/payload/native')
    require(args.output==expected and not args.output.exists(), 'Fresh declared output required')
    args.output.mkdir(parents=True,exist_ok=False)
    if args.startup_only:
        write(args.output/'startup.json',{'status':'passed_actual_startup','optimizer_updates':0,'numerical_imports':False})
        return
    np_module();life={};result={'version':VERSION,'status':'failed','arm':args.arm,'job_id':os.environ['LSB_JOBID'],
        'optimizer_updates':0,'can_certify_or_restart_original_study':False,'original_results_changed':False,
        'optimization_settings_changed':False}
    try:
        if args.arm=='saved':
            require(os.environ['LSB_QUEUE']=='hpc', 'Allocated CPU required')
            result['saved_protenix']=saved_protenix(plan)
        else:
            require(os.environ['LSB_QUEUE']=='gpul40s', 'Allocated L40S required')
            arm=plan['arms'][args.arm];source=Path(arm['science_source']['path'])
            sys.path[:0]=[str(source/'src'),str(source/'scripts'),str(Path(arm['adapter_source']['path'])/'scripts')]
            os.environ['PYTHONPATH']=os.pathsep.join((str(source/'src'),str(source/'scripts'),os.environ.get('PYTHONPATH','')))
            import jax
            jax.config.update('jax_default_matmul_precision','highest')
            require(not jax.config.jax_enable_x64, 'Original FP32 backend required')
            require((jax.default_backend()=='gpu')==(args.arm=='protenix'), 'Original per-arm JAX backend differs')
            with deadline(5400):
                result['native']=native_protenix(plan,arm,args.output,life) if args.arm=='protenix' else native_boltz(plan,arm,args.output,life)
        result['status']='diagnostics_completed_original_failure_preserved'
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        try:
            if life.get('frozen'):
                after=life['frozen']();write(args.output/'frozen-after.json',after)
                require(after==life['frozen_before'], 'Frozen native model/context changed')
                result['frozen_unchanged']=True
            verify_inputs(plan);result['protected_inputs_unchanged']=True
        except BaseException as error:
            result.update(status='failed',preservation_error=repr(error))
        finally:
            if life.get('worker'):life['worker'].close()
            size=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
            require(size<MAX_BYTES-65536, 'Diagnostic output cap exceeded')
            result['output_bytes_before_assessment']=size
            write(args.output/'assessment.json',result)
    require(result['status']=='diagnostics_completed_original_failure_preserved','Diagnostic did not finish')


if __name__=='__main__':
    main()
