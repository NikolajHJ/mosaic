"""Allocated exact trajectory parity, retained frames and complete scalar streams."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import selected_contract as c
from retention import retained,full_projection,noise_append_hook,policy,verify_case


@pytest.mark.parametrize('name',list(c.PROFILES))
def test_real_generated_retention_command_reaches_actual_cli_without_model(tmp_path,name):
    from retention import ProcessProxy
    profile=c.profile(name);arm=c.SELECTED[name]
    inv=profile['root']/c.legacy.native_name(name,arm)/'payload/driver-invocation.json'
    args=list(c.read(inv)['args']);args[args.index('--mode')+1]='run'
    output=tmp_path/'parent/payload/study';output.parent.mkdir(parents=True)
    args[args.index('--output')+1]=str(output)
    ready=profile['root']/c.legacy.native_name(name,arm)/'payload/study/assessment.json'
    args+=['--ready',str(ready),'--ready-sha256',c.sha(ready)]
    c.exclusive(output.parent/'driver-invocation.json',{'args':args})
    captured=[]
    class Process:
        def run(self,vector,*a,**kw):captured.append(vector)
    manifest=c.sha(c.LAUNCHER/'source_manifest.json')
    ProcessProxy(Process(),name,manifest).run(args)
    vector=captured[0]
    vector.insert(2,'--startup-only')
    result=subprocess.run(vector,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['status']=='actual_retention_cli_accepted_without_model' and receipt['optimizer_updates']==0 and not receipt['scientific_readiness_passed']


@pytest.mark.parametrize('step,keep',[(0,True),(1,False),(19,False),(20,True),(21,False),(40,True),(43,True)])
def test_stride_includes_true_endpoints(step,keep):
    assert retained(step,43)==keep


@pytest.mark.parametrize('unit,total,segments',[(32537,30000,3),(35150,10000,4),(23182,40000,4)])
def test_full_budget_retains_overheads_safety_and_rolling_slots(unit,total,segments):
    value=full_projection(unit,total,segments)
    assert value==2*unit*(total//20+3*segments)+segments*1024**3
    assert segments*1024**3<value<2*unit*total+segments*1024**3


@pytest.mark.parametrize('args',[(True,100,1),(0,100,1),(1,101,3)])
def test_invalid_projection_fails(args):
    with pytest.raises(ValueError):full_projection(*args)


@pytest.mark.parametrize('name',list(c.PROFILES))
def test_actual_sealed_trajectory_dense_sparse_exact_parity(tmp_path,name):
    source=c.profile(name)['source']
    code=r'''
import json,sys
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from mosaic.small_molecule.chemistry import GraphLogits,mosaic_spec
from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform,RelaxedGraph
import mosaic.small_molecule.chemistry.trajectory as t
from retention import trajectory_writes,verify_case
root=Path(sys.argv[1]);hard=graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)])
hard=RelaxedGraph(hard.X.astype(jnp.float32),hard.E_upper.astype(jnp.float32))
initial=GraphLogits.from_graph(mix_with_uniform(hard,.5));spec=mosaic_spec(25);calls=[]
def loss(graph,*,key):
 calls.append(1)
 return (graph.X*jnp.arange(6)).sum()+.2*(graph.E_upper*jnp.arange(5)).sum(),{}
def run(path):
 return t.run_chemical_updates(loss,initial,prepare=lambda z,u:NativeOnlyPreparation(spec.fingerprint(),u),spec=spec,
  reference=jnp.ones(6),key=jax.random.key(17),directory=path,steps=43,step_size=.0025,max_backtracks=1,
  config={'objective_mode':'native_only','repair_weight':0.,'element_js_weight':0.,'native_name':'contact'},acceptance_policy='fixed_step',update_offset=10000)
dense=run(root/'dense');count=len(calls);calls.clear()
with trajectory_writes(t,root/'sparse',43):sparse=run(root/'sparse')
assert len(calls)==count,'I/O policy changed evaluations/tracing'
for a,b in zip(dense['updates'],sparse['updates']):
 assert {k:v for k,v in a.items() if k!='elapsed_seconds'}=={k:v for k,v in b.items() if k!='elapsed_seconds'}
for a,b in zip(dense['states'],sparse['states']):
 for key in ['step','global_step','contact_loss','refreshed_surrogate','autodiff_surrogate','components','repair_weight','preparation_identity','previous_fixed_surrogate_after']:
  assert a[key]==b[key],key
 assert {k:v for k,v in a['readout'].items() if k not in ['element_indices','bond_upper_indices','pairs','elements']}=={k:v for k,v in b['readout'].items() if k not in ['element_indices','bond_upper_indices','pairs','elements']}
 if b['snapshot_retained']:
  with np.load(root/'dense'/a['snapshot']) as x,np.load(root/'sparse'/b['snapshot']) as y:
   assert set(x.files)==set(y.files)
   for key in x.files:assert x[key].dtype==y[key].dtype and np.array_equal(x[key],y[key]),key
assert dense['counts']==sparse['counts']
index=verify_case(sparse,root/'sparse')
assert [x['step'] for x in index['retained_frames']]==[0,20,40,43]
assert len(list((root/'sparse/snapshots').glob('*.npz')))==4
assert len(list((root/'sparse/trials').glob('step-*.npz')))==3
assert len(list((root/'sparse/references').glob('step-*.json')))==4
assert (root/'sparse/latest-state.npz').is_file() and (root/'sparse/best-observed.npz').is_file()
for field,n in [('states',44),('updates',43)]:
 lines=(root/'sparse'/(field+'.jsonl')).read_text().splitlines();assert len(lines)==n
 saved=json.loads((root/'sparse/report.json').read_text())
 assert [json.loads(x) for x in lines]==saved[field]==json.loads(json.dumps(sparse[field]))
assert sum(p.stat().st_size for p in (root/'sparse').rglob('*.npz'))<sum(p.stat().st_size for p in (root/'dense').rglob('*.npz'))/2
'''
    env=os.environ.copy();env['PYTHONPATH']=':'.join([str(c.LAUNCHER),str(source/'scripts'),str(source/'src'),str(source/'tests')])
    env['JAX_PLATFORMS']='cpu';env['CUDA_VISIBLE_DEVICES']=''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],env=env,capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('step',[20,21])
def test_noise_writer_keeps_all_losses_and_only_selected_arrays(tmp_path,step):
    directory=tmp_path/'trajectory';(directory/'snapshots').mkdir(parents=True);(directory/'noise').mkdir()
    snapshot=directory/'snapshots'/f'step-{step:04d}.npz';snapshot.write_bytes(b'exact-frozen-matrix-bytes')
    noise=directory/'noise'/f'update-{step:04d}.npz';noise.write_bytes(b'exact-random-noise-bytes')
    rows=[];hook=noise_append_hook(lambda path,row:rows.append((path,dict(row))),tmp_path)
    state={'step':step,'contact_loss':1.25,'snapshot':str(snapshot.relative_to(tmp_path)),'snapshot_sha256':'matrixhash','readout':{'element_indices':[0],'bond_upper_indices':[1],'sanitized':True}}
    hook(directory/'states.jsonl',state)
    update={'update':step,'before':2.0,'after':1.25,'noise_sha256':'noisehash'}
    hook(directory/'updates.jsonl',update)
    assert rows[0][1]['contact_loss']==1.25 and rows[1][1]['before']==2.0 and rows[1][1]['after']==1.25
    assert snapshot.exists()==noise.exists()==(step==20)
    assert rows[0][1]['snapshot_retained']==(step==20)
    if step==21:
        assert state['snapshot'] is state['snapshot_sha256'] is update['noise_sha256'] is None
        assert (directory/'latest-state.npz').read_bytes()==b'exact-frozen-matrix-bytes'
        assert (directory/'noise/latest-noise.npz').read_bytes()==b'exact-random-noise-bytes'


@pytest.mark.parametrize('fault',['missing_scalar','invented_snapshot'])
def test_index_never_fills_missing_or_invented_states(tmp_path,fault):
    case={'requested_updates':1,'artifact_retention':policy(),'states':[],'updates':[]}
    if fault=='invented_snapshot':
        case['states']=[{'step':0,'snapshot_retained':True,'snapshot':'missing.npz','snapshot_sha256':'x'},
                        {'step':1,'snapshot_retained':True,'snapshot':'missing.npz','snapshot_sha256':'x'}];case['updates']=[{}]
    with pytest.raises(ValueError):verify_case(case,tmp_path)
