"""Predeclared whole-objective precision comparison; no continuation admission."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

EPSILONS=(.01,.003,.001,.0003,.0001,.00003,.00001,.000003,.000001)

def require(value,message):
    if not value:raise ValueError(message)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pin',required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--startup-only',action='store_true');args=parser.parse_args()
    import boltz_precision_ops as ops
    op,plan=ops.context(args.pin);case=plan['arms'][args.arm];root=ops.ROOT/('cpu' if args.startup_only else 'gpu-'+args.arm)/'payload'
    command=[plan['worker_python'],'-u',str(ops.SOURCE/'scripts/boltz_precision_server.py'),'--pin',args.pin,
             '--arm',args.arm,'--output',str(root/('startup-'+args.arm if args.startup_only else 'native'))]
    if args.startup_only:
        command+=['--startup-only'];subprocess.run(command,check=True,timeout=600);return
    require(os.environ.get('LSB_QUEUE')=='gpul40s','Allocated model/numerical execution required')
    sys.path[:0]=[str(Path(plan['science']['path'])/'src'),str(Path(case['source']['path'])/'scripts')]
    import numpy as np
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry.state import GraphLogits,mosaic_spec
    from mosaic.small_molecule.chemistry.loss import failure_aware_repair_loss,FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    require(jax.default_backend()=='cpu' and not jax.config.jax_enable_x64,'Original FP32 CPU logit boundary required')
    with np.load(case['point']['path'],allow_pickle=False) as archive:
        saved={f:archive[f].copy() for f in ('atom_logits','bond_logits','X','E_upper')}
        require(archive['key_data'].dtype==np.uint32 and archive['key_data'].tobytes()==np.asarray(jax.random.key_data(jax.random.key(17))).tobytes(),
                'Exact original key17 identity differs')
    logits=GraphLogits(jnp.asarray(saved['atom_logits']),jnp.asarray(saved['bond_logits']))
    graph=logits.graph()
    for f in ('X','E_upper'):require(np.asarray(getattr(graph,f)).tobytes()==saved[f].tobytes(),'Exact original graph/logit replay differs')
    prepared=None;config=ops.read(case['config']['path']);spec=mosaic_spec(25)
    chemical_function=failure_aware_repair_loss;original_chemistry=None;bridge=None
    if args.arm=='boltz_chemistry':
        prepared=prepare_failure_guidance(logits.decoded(),spec,update=10000,
            search=ProbabilityRepairConfig(**config['search']),progress_search=ProgressSearchConfig(**config['progress_search']))
        require(prepared.identity==case['preparation_identity'],'Reconstructed original fixed target differs; never substitute it')
        ops.write(root/'fixed-preparation.json',prepared.record())
        # Independent original CPU64 math avoids the production FP32 GPU transport.
        # The qualified FP32 producer keeps that actual transport for every call.
        import types
        from mosaic.small_molecule.chemistry import loss as original_chemistry
        private=types.ModuleType('_independent_frozen_chemistry_reference');private.__package__=original_chemistry.__package__
        private.__file__=original_chemistry.__file__
        sys.modules[private.__name__]=private
        exec(compile(Path(original_chemistry.__file__).read_bytes(),original_chemistry.__file__,'exec'),private.__dict__)
        chemical_function=private.failure_aware_repair_loss
        adapter=Path(case['adapter_source']['path']);sys.path.insert(0,str(adapter/'scripts'))
        from gpu_local_bridge import install_offloaded_matrix_loss
        bridge=install_offloaded_matrix_loss(root/'production-chemistry-GPU')
    cohorts=[]
    # Every point and direction is fixed before model setup or any perturbed evaluation.
    for direction in case['directions']:
        for epsilon in EPSILONS:
            points=[]
            for sign in (-1,1):
                values={f:saved[f].copy() for f in ('atom_logits','bond_logits')}
                key='atom_logits' if direction['field']=='X' else 'bond_logits'
                values[key][direction['row'],direction['source']]-=np.float32(sign*epsilon)
                values[key][direction['row'],direction['target']]+=np.float32(sign*epsilon)
                points.append(values)
            cohorts.append({'direction':direction,'epsilon':epsilon,'points':points})
    ops.write(root/'predeclared-cohort.json',{'epsilons':EPSILONS,'directions':case['directions'],
        'selection_before_evaluations':True,'after_result_pair_selection':False,'can_admit_new_child':False})
    log=(root/'native.log').open('x');process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    def receive():
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout,selectors.EVENT_READ);require(bool(selector.select(1200)),'Native response deadline expired')
        line=process.stdout.readline();require(bool(line),'Native process exited; original log preserved');return json.loads(line)
    sequence=0;cache={};responses=[];bridge_entered=False
    try:
        identity=receive();require(identity['status']=='ready_complete_objective_reference','Reference startup failed')
        ops.write(root/'native-identity.json',identity)
        if bridge is not None:bridge.__enter__();bridge_entered=True
        def native(values,reference,gradient,use_cache=True):
            nonlocal sequence
            dtype=np.float64 if reference else np.float32;values={f:np.asarray(v,dtype=dtype) for f,v in values.items()}
            digest=hashlib.sha256(b''.join(values[f].tobytes() for f in ('X','E_upper'))).hexdigest();key=(digest,reference,gradient)
            if use_cache and key in cache:return cache[key]
            sequence+=1;request={'id':sequence,'reference':reference,'gradient':gradient,'graph_sha256':digest,
                'objective':'dbmol_total_loss' if args.arm=='dbmol' else 'dbmol_affinity_loss',**{f:v.tolist() for f,v in values.items()}}
            process.stdin.write(json.dumps(request,allow_nan=False)+'\n');process.stdin.flush();response=receive()
            require(response['id']==sequence and response['graph_sha256']==digest,'Native response identity differs')
            cache[key]=response;responses.append({k:v for k,v in response.items() if k!='gradients'});return response
        def chemical(state,reference):
            if prepared is None:return 0.,{'atoms':np.zeros((25,6)), 'bonds':np.zeros((300,5))}
            function=chemical_function if reference else failure_aware_repair_loss
            def value(point):return function(point.decoded(),prepared,spec,FailureGuidanceConfig(**config['loss']))[0]
            scalar,gradient=jax.value_and_grad(value)(state)
            prepared.verify(spec)
            return float(scalar),{f:np.asarray(getattr(gradient,f)) for f in ('atoms','bonds')}
        def measure(values,reference,gradient=True,use_cache=True):
            manager=jax.experimental.enable_x64() if reference else nullcontext()
            with manager:
                dtype=jnp.float64 if reference else jnp.float32
                state=GraphLogits(jnp.asarray(values['atom_logits'],dtype=dtype),jnp.asarray(values['bond_logits'],dtype=dtype))
                graph=state.graph();raw=native({f:np.asarray(getattr(graph,f)) for f in ('X','E_upper')},reference,gradient,use_cache)
                chem,cg=chemical(state,reference);grads={}
                if gradient:
                    for key,field in (('atoms','X'),('bonds','E_upper')):
                        p=getattr(graph,field);g=jnp.asarray(raw['gradients'][field],dtype=dtype)
                        grads[key]=np.asarray(p*(g-jnp.sum(p*g,-1,keepdims=True)))
                return raw,chem,cg,grads
        baseline_values={f:saved[f] for f in ('atom_logits','bond_logits')}
        baselines={};baseline_repeats={}
        for reference in (False,True):
            rows=[]
            for _ in range(3):
                p=measure(baseline_values,reference,False,False);a=measure(baseline_values,reference,True,False)
                rows.append({'primal':p[0],'autodiff':{k:v for k,v in a[0].items() if k!='gradients'}})
                if str(reference) not in baselines:baselines[str(reference)]=a
            baseline_repeats[str(reference)]=rows
        if prepared:
            raw,chem,cg,grad=baselines['False']
            scalar=np.float32(raw['value']+float(np.float32(.001)*np.float32(chem)))
            with np.load(case['point']['path'],allow_pickle=False) as original:
                require(scalar.tobytes()==original['primal'].tobytes(),'Exact original combined scalar replay differs')
                for field,key in (('atoms','gradient_atom_logits'),('bonds','gradient_bond_logits')):
                    combined=(grad[field]+np.float32(.001)*cg[field]).astype(np.float32)
                    require(combined.tobytes()==original[key].tobytes(),'Exact original combined gradient replay differs: '+field)
            ops.write(root/'original-combined-replay.json',{'status':'passed_exact_original_fixed_teacher_scalar_gradient_replay',
                'preparation_identity':prepared.identity,'repair_weight':.001,'optimizer_updates':0})
        weights=[0.]
        if prepared:
            from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
            schedule=RepairWeightSchedule(**config['repair_schedule'])
            weights=[.001,schedule.for_state(10000),config['repair_schedule']['final']]
        observations=[]
        # Supplementary comparisons retain every coarse/fine scale. No pair chosen to pass.
        for cohort in cohorts:
            endpoint={str(reference):[measure(point,reference,False) for point in cohort['points']] for reference in (False,True)}
            d=cohort['direction'];field='atoms' if d['field']=='X' else 'bonds';key='atom_logits' if field=='atoms' else 'bond_logits'
            displacement=cohort['points'][1][key].astype(np.float64)-cohort['points'][0][key].astype(np.float64)
            rows=[]
            for weight in weights:
                row={'coefficient':weight}
                for reference in (False,True):
                    raw,chem,cg,grad=baselines[str(reference)];ends=endpoint[str(reference)]
                    dtype=np.float64 if reference else np.float32
                    combined=(grad[field].astype(dtype)+dtype(weight)*cg[field].astype(dtype)).astype(dtype)
                    predicted=float(np.sum(combined.astype(np.float64)*displacement))
                    def scalar(item):
                        # Preserve the existing production checker's final FP32 cast;
                        # DBMol's production total is already assembled in FP64.
                        value=item[0]['value']+float(dtype(weight)*dtype(item[1]))
                        return float(np.float64(value) if reference or args.arm=='dbmol' else np.float32(value))
                    observed=scalar(ends[1])-scalar(ends[0])
                    row['reference' if reference else 'production']={'predicted_difference':predicted,'observed_difference':observed,
                        'native_endpoint_components':[r[0]['components'] for r in ends],
                        'relative_error':abs(observed-predicted)/max(abs(observed),abs(predicted)) if max(abs(observed),abs(predicted)) else None}
                rows.append(row)
            observations.append({'direction':d,'epsilon':cohort['epsilon'],'cohorts':rows});ops.budget(plan)
        process.stdin.write('{"action":"close"}\n');process.stdin.flush();closed=receive();process.wait(timeout=30)
        require(closed['status']=='closed' and process.returncode==0,'Full frozen/close handshake failed')
        if prepared:prepared.verify(spec)
        ops.write(root/'assessment.json',{'status':'complete_objective_precision_observations_not_admission','arm':args.arm,
            'identity':identity,'close':closed,'responses':responses,'observations':observations,
            'independent_primal_and_AD_baseline_repeats':baseline_repeats,
            'fixed_preparation_identity':None if prepared is None else prepared.identity,
            'optimizer_updates':0,'can_admit_new_child':False,'can_certify_or_restart_original_study':False,
            'saved_results_changed':False,'acceptance_tolerances_changed':False})
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=30)
        log.close()
        if bridge_entered:bridge.__exit__(None,None,None)

if __name__=='__main__':main()
