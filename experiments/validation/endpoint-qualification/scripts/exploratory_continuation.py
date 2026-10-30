"""New exploratory children of saved studies; never rewrites a predecessor.

Uses the original scorer, fixed-step loop, teacher preparation and acceleration.
FD/native qualification is explicitly not requested. Frozen/input identities,
finite backward values, actual graph/key mapping and own resource caps remain.
"""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def binding(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'Ordinary input file required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def bound(row):
    require(binding(row['path'])==row,'Input changed: '+row['path']);return Path(row['path'])


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


def worker_environment(source,current):
    result=dict(current)
    result['PYTHONPATH']=os.pathsep.join([str(Path(source)/'src'),str(Path(source)/'scripts'),current.get('PYTHONPATH','')])
    return result


def validate(row):
    require(row['research_profile']=='exploratory','Explicit exploratory profile required')
    require(row['name'] in ('protenix','protenix_chemistry','boltz','boltz_chemistry','dbmol'),'Unknown arm')
    pro=row['name'].startswith('protenix');chem=row['name'].endswith('_chemistry')
    require(row['rate']==(.3 if pro else .0025) and row['target']==(50000 if pro else 30000),'Changed rate/target')
    fresh_noise = row.get('logit_noise') is not None
    if fresh_noise:
        from gradient_scaled_noise import validate as validate_noise
        validate_noise(row['logit_noise'])
        require(row['name']=='protenix_chemistry' and row['start']==0,'Noise arm is a fresh chemistry comparison')
    require(type(row['start']) is int and (0<=row['start'] if fresh_noise else 0<row['start'])
        and row['start']<row['target'] and row['steps']==row['target']-row['start'],'Whole remaining budget required')
    require(row['chemistry']==chem and row['matrix_stride']==20 and row['loss_stride']==1,'Changed chemistry/retention')
    require(row['acceptance_policy']=='fixed_step' and row['max_backtracks']==1,'Exactly one fixed candidate required')
    return row


def backward_only_tree(source):
    """Remove two redundant primal calls; preserve all backward/update arithmetic."""
    tree=ast.parse(source);wanted={
        'primal, aux = contact_loss(graph, key=key)':'primal, aux = raw.value, raw.aux',
        'chemical_primal, _ = chemical(logits)':'chemical_primal, _ = chemical_value, terms'}
    counts={key:0 for key in wanted}
    class Change(ast.NodeTransformer):
        def visit_Assign(self,node):
            for before,after in wanted.items():
                if ast.dump(node)==ast.dump(ast.parse(before).body[0]):
                    counts[before]+=1;return ast.copy_location(ast.parse(after).body[0],node)
            return self.generic_visit(node)
    tree=Change().visit(tree);require(all(v==1 for v in counts.values()),'Original independent-primal shape differs')
    return ast.fix_missing_locations(tree)


def inventory(root):
    for attempt in range(3):
        size=count=0
        try:
            for folder,dirs,files in os.walk(root,followlinks=False):
                for name in dirs:require(not Path(folder,name).is_symlink(),'Output directory link')
                for name in files:
                    p=Path(folder,name);require(not p.is_symlink(),'Output file link')
                    size+=p.stat().st_size;count+=1
            return size,count
        except FileNotFoundError:
            if attempt==2:raise


def capacity(plan,phase):
    # The shared helper supplies own-output admission; foreign studies are advisory.
    import storage_accounting as policy
    root=Path(plan['root']);quota=None;warnings=[]
    if shutil.which('beegfs-ctl'):
        import pwd,subprocess
        pools={};uid=os.getuid();user=pwd.getpwuid(uid).pw_name
        for i in range(1,7):
            try:
                r=subprocess.run(['beegfs-ctl','--getquota','--uid',str(uid),'--storagepoolid='+str(i),'--csv',
                    '--cfgFile=/etc/beegfs/beegfs-client4.conf'],capture_output=True,text=True,timeout=3)
                require(r.returncode==0,'Quota command failed');pools[i]=policy.parse_pool(r.stdout,uid,user)
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:warnings.append(str(error));break
        if pools:
            quota={k:min((r[hard]-r[used] for r in pools.values() if r[hard] is not None),default=None)
                for k,hard,used in (('available_bytes','hard_bytes','used_bytes'),('available_files','hard_files','used_files'))}
    scopes=[];actual={}
    for name,row in plan['arms'].items():
        path=root/name;size,count=inventory(path) if path.exists() else (0,0)
        actual[name]={'bytes':size,'files':count}
        scopes.append({'id':name,'remaining_bytes':policy.remaining_writes(row['output_cap_bytes'],size),
                       'remaining_files':policy.remaining_writes(row['file_cap'],count)})
    fs=os.statvfs(root)
    result=policy.assess(quota,shutil.disk_usage(root).free,{k:getattr(fs,k) for k in ('f_files','f_ffree','f_favail')},scopes,research_profile='exploratory')
    result.update(phase=phase,actual=actual);result['warnings']+=warnings
    require(result['status']=='passed','Own remaining output exceeds measured capacity: '+json.dumps(result))
    return result


def run(plan,row,output,smoke_updates=None):
    from gpu_policy import selected,validate_hardware
    gpu=selected(row)
    require(os.environ.get('LSB_QUEUE')==gpu['queue'] and os.environ.get('LSB_JOBID','').isdigit(),'Declared allocated GPU required')
    require(smoke_updates is None or (plan.get('purpose')=='v100_device_comparison' and type(smoke_updates) is int and 1<=smoke_updates<=3),'Explicit bounded device smoke only')
    requested=row['steps'] if smoke_updates is None else smoke_updates
    import fcntl
    lock=(Path(plan['root'])/(row['name']+'.writer.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not output.exists(),'New output only');output.mkdir(parents=True)
    report={'version':'mosaic-exploratory-continuation-v2','status':'running','research_profile':'exploratory',
        'derivative_qualified':False,'original_failures_preserved':True,'saved_results_changed':False,
        'job_id':os.environ['LSB_JOBID'],'name':row['name'],'global_start':row['start'],'global_target':row['target'],
        'primal_scope':'primal_from_backward_call','warnings':row['warnings'],'configuration':row,
        'gpu_profile':gpu,'device_smoke_updates':smoke_updates,'device_smoke_states_adopted':False}
    started=time.monotonic();worker=None;frozen=None;model=context=None
    source=Path(row['science_source']['path']);sys.path[:0]=[str(source/'src'),str(source/'scripts')]
    os.environ.update(worker_environment(source,os.environ))
    os.chdir(source)
    try:
        routing=module(Path(plan['routing_source'])/'scripts/cuda_ownership.py','_exploratory_scheduler_routing')
        visibility=routing.resolve_visibility(os.environ)
        os.environ['CUDA_VISIBLE_DEVICES']=visibility['selected_cuda_visible_devices'];report['CUDA_routing']=visibility
        import subprocess,csv,io
        observation=subprocess.run(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
            '--query-gpu=name,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
        devices=list(csv.reader(io.StringIO(observation.stdout)))
        require(len(devices)==1 and len(devices[0])==2,'Exactly one GPU observation required')
        report['gpu_hardware']=validate_hardware(devices[0][0].strip(),int(devices[0][1].strip()),gpu)
        for key in ('config','snapshot','reference','pocket'):bound(row[key])
        require(binding(source/'source_manifest.json')['sha256']==row['science_source']['manifest_sha256'],'Scientific source identity changed')
        config=read(row['config']['path'])
        require(config['initial_step_size']==row['rate'] and config['chemistry_enabled']==row['chemistry']
            and config['max_backtracks']==1,'Scientific settings differ')
        import jax
        import jax.numpy as jnp
        import numpy as np
        from mosaic.small_molecule.chemistry import trajectory as trajectory
        from mosaic.small_molecule.chemistry.state import GraphLogits,mosaic_spec
        from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
        from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig
        from mosaic.small_molecule.chemistry.preparation import prepare_failure_guidance
        from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
        from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
        from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
        from mosaic.small_molecule.gradient_checks import array_digest
        jax.config.update('jax_default_matmul_precision','highest')
        pro=row['name'].startswith('protenix');db=row['name']=='dbmol'
        require(jax.config.jax_enable_x64==db,'Original scalar precision differs')
        logit_type=GraphLogits
        if pro:
            from protenix_initialization import ProtenixReplayLogits
            logit_type=ProtenixReplayLogits
        key=jax.random.key(17)
        with np.load(row['snapshot']['path'],allow_pickle=False) as saved:
            for k,shape in (('atom_logits',(25,6)),('bond_logits',(300,5)),('X',(25,6)),('E_upper',(300,5))):
                require(saved[k].dtype==np.float32 and saved[k].shape==shape and np.isfinite(saved[k]).all(),'Invalid saved FP32 point')
            require(np.array_equal(saved['key_data'],np.asarray(jax.random.key_data(key))),'Saved key differs')
            initial=logit_type(jnp.asarray(saved['atom_logits']),jnp.asarray(saved['bond_logits']))
            replay={}
            for f in ('X','E_upper'):
                actual=np.asarray(getattr(initial.graph(),f));expected=saved[f]
                require(actual.shape==expected.shape and actual.dtype==expected.dtype and np.isfinite(actual).all(),'Incompatible replay')
                tolerance=32*np.finfo(np.float32).eps*np.abs(expected)+32*np.finfo(np.float32).smallest_subnormal
                require(np.all(np.abs(actual-expected)<=tolerance),'Material saved graph mapping mismatch')
                replay[f]={'byte_equal':bool(np.array_equal(actual,expected)),'max_absolute_difference':float(np.max(np.abs(actual-expected)))}
        report['initialization_replay']={'scope':'exact_saved_logits_key_dtype_relative_graph_replay','graph':replay,
            'relative_tolerance':float(32*np.finfo(np.float32).eps),'absolute_floor':32*float(np.finfo(np.float32).smallest_subnormal)}
        if pro:
            from mosaic.small_molecule.protenix import FrozenProtenixTiny
            from mosaic.small_molecule.losses import NativeContactLoss
            from mosaic.small_molecule.io import load_context
            from mosaic.small_molecule.pocket import map_protenix_pocket
            ready=read(bound(row['context_ready']));context=load_context(Path(row['context_ready']['path']).parent/'context',ready['context_files'])
            model=FrozenProtenixTiny.load(cache_dir=Path('/work3/s234818/mosaic/cache/mosaic/protenix'))
            frozen={'model_digest':array_digest(model),'context_digest':array_digest(context)}
            require(frozen==row['frozen_identity'],'Frozen Protenix model/context differs')
            pocket=read(row['pocket']['path']);tokens=map_protenix_pocket(context,pocket)
            loss=model.build_loss(context=context,loss=NativeContactLoss(8.0,tokens));native_name='contact'
        else:
            bound(row['context'])
            for asset in row['checkpoints']:bound(asset)
            command=[row['worker_python'],'-u']
            if db:
                from dbmol_bridge import DBMolWorker,dbmol_loss
                command+=[str(source/'scripts/dbmol_worker.py')]
            else:
                from mosaic.small_molecule.boltz2.bridge import AffinityWorker,affinity_loss
                command+=['-m','mosaic.small_molecule.boltz2.worker']
            command+=['--context',row['context']['path'],'--context-sha256',row['context']['sha256'],
                      '--checkpoint','/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt']
            if db:command+=['--structure-checkpoint','/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt']
            worker=(DBMolWorker if db else AffinityWorker)(command,stderr=output/'worker.log')
            frozen={k:worker.identity[k] for k in ('model_digest','context_digest')}
            if row.get('frozen_identity'):require(frozen==row['frozen_identity'],'Frozen Boltz model/context differs')
            loss=dbmol_loss(worker) if db else affinity_loss(worker);native_name='dbmol_total' if db else 'affinity'
        report['frozen_before']=frozen
        trajectory.GraphLogits=logit_type
        # The exact backward expressions and candidate update remain unchanged.
        exec(compile(backward_only_tree(inspect.getsource(trajectory.evaluate_combined)),trajectory.__file__,'exec'),trajectory.__dict__)
        def smoke(loss,evaluation,prepared,spec,reference,*,key,config,directory=None):
            arrays=[np.asarray(getattr(evaluation.gradients,f)) for f in ('atoms','bonds')]
            require(all(np.isfinite(a).all() for a in arrays) and math.isfinite(evaluation.value),'Nonfinite backward smoke')
            value={'status':'finite_backward_smoke_only','derivative_qualified':False,'FD':'not_requested',
                'gradient_norms':[float(np.linalg.norm(a)) for a in arrays],
                'flat_component_is_not_failure':True,'primal_scope':'primal_from_backward_call'}
            require(evaluation.preparation_id==prepared.identity,'Teacher changed within step')
            if directory:Path(directory).mkdir(parents=True,exist_ok=False);write(Path(directory)/'report.json',value)
            return value
        trajectory.check_combined_directions=smoke
        if row.get('logit_noise') is not None:
            noise = module(Path(plan['source_root'])/'gradient_scaled_noise.py','_exploratory_binding_noise')
            noise.install(trajectory,row['logit_noise'])
            report['logit_noise']=row['logit_noise']
        retention=module(Path(plan['retention_source'])/'retention.py','_exploratory_retention')
        retention.install_trajectory(trajectory,output)
        spec=mosaic_spec(25);chem=row['chemistry']
        search=ProbabilityRepairConfig(**config['search']);progress=ProgressSearchConfig(**config['progress_search'])
        def prepare(state,update):
            return prepare_failure_guidance(state.decoded(),spec,update=update,search=search,progress_search=progress) if chem else NativeOnlyPreparation(spec.fingerprint(),update)
        options={'loss':FailureGuidanceConfig(**config['loss']),'repair_weight':.001 if chem else 0.,
            'element_js_weight':0.,'parity_atol':2e-5,'native_name':native_name,
            'objective_mode':'native_plus_chemistry' if chem else 'native_only',
            'objective_identity':{'model_digest':frozen['model_digest'],'context_digest':frozen['context_digest'],
                'native_name':native_name,'pocket_definition_sha256':row['pocket']['sha256'],'diffusion_gradient':False}}
        schedule=RepairWeightSchedule(**config['repair_schedule']) if chem else None
        if chem:require(config['repair_schedule']==row['repair_schedule'],'Original global ramp differs')
        old_append=trajectory._append_record;next_gate=time.monotonic()+300
        def append(path,value):
            nonlocal next_gate
            if time.monotonic()-started>row['internal_seconds']:raise TimeoutError('Own allocation budget reached; saved prefix retained')
            if time.monotonic()>=next_gate:
                write(output/'capacity-latest.json',capacity(plan,'periodic'));next_gate=time.monotonic()+300
            old_append(path,value)
        trajectory._append_record=append
        with ExitStack() as stack:
            if chem:
                adapter=Path(row.get('gpu_adapter_source',row['adapter_source']['path']));sys.path.insert(0,str(adapter/'scripts'))
                report['acceleration_source']={'original':row['adapter_source'],'device_guard_overlay':str(adapter)}
                # Child workers must inherit the original science and adapter paths.
                os.environ['PYTHONPATH']=os.pathsep.join([str(source/'src'),str(adapter/'scripts'),os.environ.get('PYTHONPATH','')])
                pilot=module(adapter/'scripts/full_model_pilot.py','_exploratory_existing_acceleration')
                acceleration=output/'acceleration';acceleration.mkdir()
                stack.enter_context(pilot.combined_adapters(row['acceleration_profile'],acceleration))
            result=trajectory.run_chemical_updates(loss,initial,prepare=prepare,spec=spec,
                reference=read(row['reference']['path'])['element_counts'],key=key,directory=output/'case-0',
                steps=requested,step_size=row['rate'],max_backtracks=1,acceptance_policy='fixed_step',
                config=options,schedule=schedule,update_offset=row['start'])
            require(len(result['updates'])==requested,'Incomplete requested update count')
            report.update(status='completed_device_smoke' if smoke_updates is not None else 'completed_exploratory_updates',applied_updates=len(result['updates']),
                update_seconds=[u['elapsed_seconds'] for u in result['updates']],
                global_end=row['start']+len(result['updates']),case_report=str(output/'case-0/report.json'),
                artifact_retention=result['artifact_retention'])
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        try:
            response=worker.request('verify_identity') if worker else None
            after=({k:response[k] for k in ('model_digest','context_digest')} if worker else
                   {'model_digest':array_digest(model),'context_digest':array_digest(context)} if model and context else None)
            if frozen is not None:require(after==frozen,'Frozen model/context changed')
            report['frozen_after']=after
            for k in ('config','snapshot','reference','pocket'):bound(row[k])
        except BaseException as error:report.update(status='failed',preservation_error=repr(error));raise
        finally:
            if worker is not None:worker.close()
            report['elapsed_seconds']=time.monotonic()-started;write(output/'summary.json',report)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--arm',required=True)
    parser.add_argument('--startup-only',action='store_true');parser.add_argument('--smoke-updates',type=int);args=parser.parse_args()
    plan=read(args.plan);row=validate(plan['arms'][args.arm]);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    for k in ('config','snapshot','reference','pocket'):bound(row[k])
    value=capacity(plan,'entry')
    if args.startup_only:
        print(json.dumps({'status':'metadata_startup_passed_no_models','name':args.arm,'capacity':value}));return
    run(plan,row,root/row['name']/'payload/study',args.smoke_updates)


if __name__=='__main__':main()
