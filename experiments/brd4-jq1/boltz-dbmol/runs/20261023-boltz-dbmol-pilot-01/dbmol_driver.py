"""Allocated fresh/exact-replay DBMol pilot; no existing run is writable."""
import argparse, json, os, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ['config','output','cpu-certificate','reference','pocket','prepared']:p.add_argument('--'+k,type=Path,required=True)
    for k in ['manifest','cpu-certificate-sha256','pocket-sha256','context-sha256']:p.add_argument('--'+k,required=True)
    p.add_argument('--mode',choices=['validate','run'],required=True)
    p.add_argument('--ready',type=Path);p.add_argument('--ready-sha256')
    p.add_argument('--offset',type=int,default=0);p.add_argument('--segment-steps',type=int,default=2500)
    p.add_argument('--resume-snapshot',type=Path);p.add_argument('--resume-sha256')
    p.add_argument('--worker-python',default='/work3/s234818/latentmol/env/boltz311/bin/python')
    a=p.parse_args()
    if not os.environ.get('LSB_JOBID'):raise RuntimeError('Allocated DTU HPC required')
    from dbmol_protocol import config25,sha,read,verify_cpu_certificate,verify_readiness,verify_identity,frozen,scope,restore,update_parameters,verify_attempt_rates,POCKET,CONTEXT_FILE
    c=config25(read(a.config));execution=scope(a.offset,a.segment_steps,a.resume_snapshot,a.resume_sha256)
    verify_cpu_certificate(a.cpu_certificate,a.cpu_certificate_sha256,a.manifest)
    if sha(a.reference)!='37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb' or sha(a.pocket)!=a.pocket_sha256 or a.pocket_sha256!=POCKET or a.context_sha256!=CONTEXT_FILE or sha(a.prepared/'context.pt')!=CONTEXT_FILE:raise ValueError('Reference/pocket/context bytes changed')
    if a.mode=='run':
        if a.ready is None or a.ready_sha256 is None:raise ValueError('Actual passed readiness required')
        ready=verify_readiness(a.ready,a.ready_sha256,a.config,a.manifest,execution)
    else:ready=None
    import jax
    import jax.numpy as jnp
    import numpy as np
    from dbmol_bridge import DBMolWorker,dbmol_loss,METRICS
    from dbmol_native_validation import validate_native_components
    from mosaic.small_molecule.chemistry import GraphLogits,mosaic_spec
    from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
    from mosaic.small_molecule.state import graph_from_hard,RelaxedGraph,mix_with_uniform
    from mosaic.small_molecule.io import write_json
    from mosaic.small_molecule.provenance import environment_identity,source_identity
    if not jax.config.jax_enable_x64 or any(d.platform!='cpu' for d in jax.devices()):raise RuntimeError('Require CPU JAX FP64 scalar with FP32 graph')
    jax.config.update('jax_default_matmul_precision','highest')
    a.output.mkdir(parents=True,exist_ok=False)
    r={'status':'running','mode':a.mode,'job_id':os.environ['LSB_JOBID'],'config':c,
       'config_sha256':sha(a.config),'source_manifest_sha256':a.manifest,
       'source':source_identity(Path(__file__).resolve().parents[1]),'environment':environment_identity(),
       'execution':execution,'arm':'boltz2_dbmol','n_atoms':25,'chemistry_enabled':False,
       'pocket_definition_sha256':POCKET,'diffusion_gradient':False,
       'scalar_arithmetic':'contact_head_and_assembly_float64_v1','graph_dtype':'float32'}
    import shutil
    shutil.copyfile(a.pocket,a.output/'pocket_definition.json')
    key=jax.random.key(17);hard=graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)])
    hard=RelaxedGraph(hard.X.astype(jnp.float32),hard.E_upper.astype(jnp.float32))
    initial=restore(a.resume_snapshot,a.resume_sha256,key) if a.resume_snapshot else GraphLogits.from_graph(mix_with_uniform(hard,.5))
    if any(v.dtype!=jnp.float32 for v in [initial.atoms,initial.bonds,initial.graph().X,initial.graph().E_upper]):raise ValueError('Graph initialization changed FP32 precision')
    np.savez_compressed(a.output/'initialization.npz',X=np.asarray(initial.graph().X),E_upper=np.asarray(initial.graph().E_upper),atom_logits=np.asarray(initial.atoms),bond_logits=np.asarray(initial.bonds),key_data=np.asarray(jax.random.key_data(key)))
    r['initialization_sha256']=sha(a.output/'initialization.npz')
    if ready:
        with np.load(a.ready.parent/'initialization.npz') as x,np.load(a.output/'initialization.npz') as y:
            if set(x.files)!=set(y.files):raise ValueError('Native/production initialization fields differ')
            for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    worker=None;started=time.perf_counter()
    try:
        worker=DBMolWorker([a.worker_python,'-u',str(Path(__file__).with_name('dbmol_worker.py')),'--context',str(a.prepared/'context.pt'),'--context-sha256',CONTEXT_FILE,'--checkpoint','/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt','--structure-checkpoint','/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt'],stderr=a.output/'worker.log')
        r['native_identity']=verify_identity(worker.identity)
        if ready and r['native_identity']!=ready['native_identity']:raise ValueError('Production native identities differ from readiness')
        r['model_digest']=worker.identity['model_digest'];r['context_digest']=worker.identity['context_digest']
        loss=dbmol_loss(worker)
        def checks(graph,directory):return validate_native_components(worker,graph,key,directory,hard_graph=hard,make_loss=lambda objective:dbmol_loss(worker,objective=objective))
        check_started=time.perf_counter()
        r['native_gradient_checks']=checks(initial.graph(),a.output/'native_gradient_checks')
        native_check_seconds=time.perf_counter()-check_started
        r['native_hard_parity']=r['native_gradient_checks']['hard_native_parity']
        spec=mosaic_spec(25)
        options={'objective_mode':'native_only','repair_weight':0.,'element_js_weight':0.,'parity_atol':2e-5,'native_name':'dbmol_total',
                 'objective_identity':{'native_identity':worker.identity,'native_name':'dbmol_total','coefficient':1.,'scope':'two frozen trunks/fixed detached original JQ1 pose; no chemistry'}}
        common=dict(prepare=lambda z,u:NativeOnlyPreparation(spec.fingerprint(),u),spec=spec,reference=read(a.reference)['element_counts'],key=key,**update_parameters(c))
        if a.mode=='validate':
            pilot=run_chemical_updates(loss,initial,directory=a.output/'early_pilot',steps=10,config=options,**common)
            if pilot['status']!='passed' or len(pilot['updates'])!=10:raise ValueError('Disposable ten-update pilot incomplete')
            verify_attempt_rates(pilot,c)
            mean=float(np.mean([u['elapsed_seconds'] for u in pilot['updates']]))
            # Native entry checks run once in production and again at terminal.
            # Charge measured startup and both checks in addition to the old
            # update safety factor and fixed 1800s headroom.
            prepilot_seconds=time.perf_counter()-started-sum(u['elapsed_seconds'] for u in pilot['updates'])
            projected=1.5*mean*a.segment_steps+prepilot_seconds+1.5*native_check_seconds+1800
            sizes=[]
            for i in range(1,11):
                root=a.output/'early_pilot'
                sizes.append(sum((root/f/n).stat().st_size for f,n in [('snapshots',f'step-{i:04d}.npz'),('references',f'step-{i:04d}.json'),('trials',f'step-{i-1:04d}.npz')]))
            r['runtime_gate']={'status':'passed' if projected<82800 else 'failed','mean_update_seconds':mean,'conservative_projected_seconds':projected,'segment_updates':a.segment_steps,'internal_allocation_seconds':82800,'observed_startup_entry_check_seconds':prepilot_seconds,'observed_native_check_seconds':native_check_seconds,'terminal_check_safety_factor':1.5,'update_safety_factor':1.5,'reserve_seconds':1800}
            r['storage_projection']={'maximum_pilot_update_bytes':max(sizes),'full_projected_bytes':2*max(sizes)*10000+4*1024**3,'segment_projected_bytes':2*max(sizes)*a.segment_steps+1024**3}
            if projected>=82800:raise RuntimeError('Declared segment runtime admission failed')
        else:
            case=run_chemical_updates(loss,initial,directory=a.output/'case-0',steps=a.segment_steps,config=options,update_offset=a.offset,**common)
            verify_attempt_rates(case,c)
            if case['status']!='passed' or len(case['updates'])!=a.segment_steps or len(case['states'])!=a.segment_steps+1:raise ValueError('Incomplete declared segment')
            if [x['global_step'] for x in case['states']]!=list(range(a.offset,a.offset+a.segment_steps+1)):raise ValueError('Noncontiguous global states')
            r['cases']=[case]
            with np.load(a.output/'case-0'/case['states'][-1]['snapshot']) as z:terminal=RelaxedGraph(jnp.asarray(z['X']),jnp.asarray(z['E_upper']))
            r['terminal_gradient_checks']=checks(terminal,a.output/'terminal_gradient_checks')
        r['status']='passed'
    except Exception as error:
        r.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        try:
            if worker is not None:r['frozen_identity_after']=worker.request('verify_identity');frozen(r['frozen_identity_after'],worker.identity)
        except Exception as error:r.update(status='failed',identity_error=str(error))
        finally:
            if worker is not None:worker.close()
            r['elapsed_seconds']=time.perf_counter()-started;write_json(a.output/'summary.json',r)
    if r['status']!='passed':raise RuntimeError('Final frozen identity failed')
    if a.mode=='validate':
        proofs=[{'path':str(p.resolve()),'sha256':sha(p)} for p in sorted(a.output.rglob('*')) if p.is_file() and p.suffix in ('.json','.npz')]
        assessment={k:r[k] for k in ['status','config_sha256','source_manifest_sha256','native_identity','model_digest','context_digest','initialization_sha256','runtime_gate','storage_projection','execution','arm','n_atoms','pocket_definition_sha256','diffusion_gradient']}
        assessment['proofs']=proofs;write_json(a.output/'assessment.json',assessment)

if __name__=='__main__':main()
