"""Allocated Boltz serial scope, exact replay and prospective-capacity contracts."""
import json
from pathlib import Path
import pytest

import boltz_serial_ops as ops
from boltz_serial_protocol import VERSION, EXECUTION, segment_scope, projection, restore_logits, require_boundary
from campaign25_contract import config25


def cfg(arm):
    return json.loads((Path(__file__).resolve().parents[1] / 'examples/configs' / ('boltz_serial_' + arm + '.json')).read_bytes())


@pytest.mark.parametrize('arm', ops.ARMS)
def test_exact_declared_new_pair(arm):
    config = cfg(arm)
    assert config25(config) == config
    assert config['steps'] == 30000 and config['initial_step_size'] == .0025
    assert config['initialization'] == {'kind':'chain_uniform','mixing':.5}
    assert config['seed'] == 17 and config['n_atoms'] == 25


@pytest.mark.parametrize('arm', ops.ARMS)
@pytest.mark.parametrize('field,value', [('steps',10000),('seed',18),('initial_step_size',.005),('max_backtracks',2),('n_atoms',24),('allocation_seconds',86400),('element_js_weight',.1),('acceptance_policy','best_primal'),('chemistry_enabled',None),('arm','protenix')])
def test_scientific_scope_mutations_fail(arm,field,value):
    config = cfg(arm)
    config[field] = value
    with pytest.raises(ValueError):
        config25(config)


@pytest.mark.parametrize('field,value', [('hold_updates',10000),('total_updates',10000),('final',50),('initial',.0001)])
def test_user_stretched_ramp_is_exact(field,value):
    config = cfg('boltz2_chemistry')
    config['repair_schedule'][field] = value
    with pytest.raises(ValueError):
        config25(config)


@pytest.mark.parametrize('arm,index', [(arm,index) for arm in ops.ARMS for index in range(3)])
def test_all_six_declared_global_segment_ranges(arm,index):
    start = ops.offset(arm,index)
    snapshot,digest = ('/saved.npz','a'*64) if start else (None,None)
    execution = segment_scope(cfg(arm),start,10000,snapshot,digest)
    assert execution['start'] == start and execution['end'] == start+10000 and execution['target'] == 30000
    args = ops.command(ops.name(arm,'segment',index),'segment','b'*64)
    assert '-w' not in args and args[args.index('-W')+1] == '23:00'
    assert args[args.index('-gpu')+1] == 'num=1:mode=exclusive_process'
    assert ops.decode(ops.name(arm,'segment',index)) == (arm,'segment',index)


@pytest.mark.parametrize('offset,steps,snapshot,digest', [(0,30000,None,None),(5000,10000,'p','a'),(30000,10000,'p','a'),(10000,10000,None,None),(0,10000,'p','a'),(True,10000,None,None),(0,9999,None,None),(10000,10000,'p',None)])
def test_bad_boundary_scope_fails(offset,steps,snapshot,digest):
    with pytest.raises(ValueError):
        segment_scope(cfg('boltz2'),offset,steps,snapshot,digest)


@pytest.mark.parametrize('arm,index', [('boltz2',3),('boltz2_chemistry',3),('protenix',0),('boltz2',-1),('boltz2',True)])
def test_only_two_finite_declared_chains(arm,index):
    with pytest.raises(ValueError):
        ops.offset(arm,index)


@pytest.mark.parametrize('unit,updates,segments', [(0,30000,3),(1.,30000,3),(10,10000,1),(10,30000,1),(10,40000,4)])
def test_projection_never_uses_single_segment_budget(unit,updates,segments):
    with pytest.raises(ValueError):
        projection(unit,updates,segments)


def test_full_three_segment_projection_and_both_endpoints():
    assert projection(14000) == 2*14000*30000+3*1024**3
    assert all(ops.offset(arm,2)+10000 == 30000 for arm in ops.ARMS)


def test_stage_has_cpu_jax_but_does_not_hide_worker_gpu():
    path = Path(__file__).resolve().parents[1] / 'scripts/hpc/boltz_serial_stage.lsf'
    text = path.read_text()
    assert 'export JAX_PLATFORMS=cpu' in text
    assert "if [[ \"$phase\" == native || \"$phase\" == segment ]]" in text
    assert "else\n  [[ \"$LSB_QUEUE\" == hpc ]]\n  export CUDA_VISIBLE_DEVICES=''" in text


def test_existing_protenix_full_projection_cannot_use_native_allowance(tmp_path,monkeypatch):
    source,root = tmp_path/'source',tmp_path/'pro'
    source.mkdir();root.mkdir()
    (source/'source_manifest.json').write_text('{}')
    (root/'cpu-certificate.json').write_text(json.dumps({'status':'passed','proofs':[]}))
    monkeypatch.setattr(ops,'PRO_SOURCE',source);monkeypatch.setattr(ops,'PRO_ROOT',root)
    monkeypatch.setattr(ops,'sha',lambda path:ops.PRO_MANIFEST if Path(path).name=='source_manifest.json' else ops.PRO_CPU)
    monkeypatch.setattr(ops,'tree',lambda path:0)
    assert ops.protenix_budgets(False)['remaining'] == 2*1024**3
    with pytest.raises(ValueError,match='Await authenticated full'):
        ops.protenix_budgets(True)
    (root/'random-segment-0000').mkdir()
    with pytest.raises(ValueError,match='no full projection'):
        ops.protenix_budgets(False)


def test_dbmol_unknown_long_cannot_use_validation_allowance(tmp_path,monkeypatch):
    root = tmp_path/'pilot'
    root.mkdir()
    monkeypatch.setattr(ops,'DBMOL_ROOT',root)
    monkeypatch.setattr(ops,'tree',lambda path:0)
    assert ops.dbmol_budget()['full'] == 1024**3
    (root/'long-submissions.json').write_text('{}')
    with pytest.raises(ValueError,match='lacks authenticated'):
        ops.dbmol_budget()


def test_native_certificates_without_existing_projection_wait_without_submit(tmp_path,monkeypatch):
    own,other,parent = tmp_path/'own',tmp_path/'other',tmp_path/'parent'
    own.mkdir();other.mkdir();parent.mkdir()
    (own/'projection.json').write_text(json.dumps({arm:{'unit_bytes':42} for arm in ops.ARMS}))
    monkeypatch.setattr(ops,'ROOT',own);monkeypatch.setattr(ops,'PRO_ROOT',other)
    def forbidden(*args,**kwargs):
        raise AssertionError('Waiting path must not submit')
    monkeypatch.setattr(ops,'submit',forbidden)
    monkeypatch.setattr(ops,'queue_dispatch_watch',lambda manifest:{'job_id':'123'})
    ops.dispatch('a'*64,parent)
    record = json.loads((parent/'waiting.json').read_bytes())
    assert record['status']=='waiting_existing_protenix_full_projections' and record['long_submitted'] is False
    assert record['watcher_job']=='123'


def test_unknown_submission_outcome_is_durable_and_not_retried(tmp_path,monkeypatch):
    from subprocess import CompletedProcess
    monkeypatch.setattr(ops,'ROOT',tmp_path)
    monkeypatch.setattr(ops.subprocess,'run',lambda *args,**kwargs:CompletedProcess(args,1,'','rejected'))
    with pytest.raises(ValueError,match='Unknown submission'):
        ops.submit('cpu','cpu','a'*64)
    record = json.loads((tmp_path/'cpu/submission.json').read_bytes())
    assert record['outcome']=='unknown' and (tmp_path/'cpu/submission-started.json').exists()
    with pytest.raises(FileExistsError):
        ops.submit('cpu','cpu','a'*64)


def test_guarded_manual_dispatch_has_a_new_unique_own_job():
    args = ops.command('dispatch-0001','dispatch','a'*64)
    assert 'mosaic-bol30k-dispatch-0001' in args and '-gpu' not in args
    with pytest.raises(ValueError):
        ops.command('after-cpu','dispatch','a'*64)


@pytest.mark.parametrize('change', [None,'status','model_digest','context_digest'])
def test_transport_metadata_retained_but_frozen_digests_and_status_required(change):
    from campaign25_contract import verify_frozen_response
    response={'status':'passed','model_digest':ops.MODEL,'context_digest':ops.CONTEXT,
              'id':736,'elapsed_seconds':.003}
    if change is None:
        verify_frozen_response(response,ops.MODEL,ops.CONTEXT)
        assert response['id']==736 and response['elapsed_seconds']==.003
    else:
        response[change]='bad'
        with pytest.raises(ValueError):
            verify_frozen_response(response,ops.MODEL,ops.CONTEXT)


def test_full_projection_authenticates_maximum_of_all_native_certificates(tmp_path,monkeypatch):
    root,source=tmp_path/'root',tmp_path/'source'
    root.mkdir();(source/'examples/configs').mkdir(parents=True)
    (source/'source_manifest.json').write_text('{}')
    monkeypatch.setattr(ops,'ROOT',root);monkeypatch.setattr(ops,'SOURCE',source)
    manifest=ops.sha(source/'source_manifest.json')
    values={}
    for arm in ops.ARMS:
        config_path=ops.config(arm);config_path.write_text('{}')
        for index,unit in [(0,20),(1,30)]:
            label=ops.name(arm,'native',index)
            ready_path=root/label/'payload/study/assessment.json';ready_path.parent.mkdir(parents=True)
            ready_path.write_text(json.dumps({'status':'passed','arm':arm,'source_manifest_sha256':manifest,'config_sha256':ops.sha(config_path),'storage_projection':{'maximum_pilot_update_bytes':unit,'projected_bytes':2*unit*10000+1024**3}}))
            (root/(label+'-certificate.json')).write_text(json.dumps({'status':'passed','source_manifest_sha256':manifest,'proofs':[{'path':str(ready_path),'sha256':ops.sha(ready_path)}]}))
        values[arm]={'unit_bytes':30}
    (root/'projection.json').write_text(json.dumps(values))
    assert ops.all_budgets()=={arm:projection(30) for arm in ops.ARMS}
    values['boltz2']['unit_bytes']=20
    (root/'projection.json').write_text(json.dumps(values))
    with pytest.raises(ValueError,match='all authenticated'):
        ops.all_budgets()


def test_protenix_initialization_marker_cannot_certify_boltz_boundary():
    execution = segment_scope(cfg('boltz2'),0,10000,None,None)
    summary = {'serial_execution':execution,'exact_boundary_replay':'passed',
               'initialization_execution':{'version':'protenix-fp32-cpu-atom-softmax-v1'}}
    with pytest.raises(ValueError):
        require_boundary(summary)
    summary['initialization_execution'] = {'version':EXECUTION,'jax_platform':'cpu',
        'dtype':'float32','parameterization':'categorical_logits_v1','model_key':17}
    assert require_boundary(summary)==execution


def test_real_chemistry_global_preparation_and_stretched_schedule(tmp_path):
    import jax
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry import GraphLogits,mosaic_spec,prepare_failure_guidance,FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    initial=GraphLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    spec=mosaic_spec(25);seen=[];schedule=RepairWeightSchedule(**cfg('boltz2_chemistry')['repair_schedule'])
    assert schedule.for_update(1)==schedule.for_update(5000)==.001
    assert .001<schedule.for_update(10000)<schedule.for_update(20000)<100
    assert schedule.for_update(30000)==100
    def prepare(logits,update):
        seen.append(update)
        return prepare_failure_guidance(logits.decoded(),spec,update=update)
    def loss(graph,*,key):
        return .1*jnp.sum(graph.X*jnp.arange(6))+.03*jnp.sum(graph.E_upper*jnp.arange(5)),{}
    result=run_chemical_updates(loss,initial,prepare=prepare,spec=spec,reference=jnp.ones(6),key=jax.random.key(17),directory=tmp_path/'chem',steps=2,step_size=.0025,max_backtracks=1,config={'loss':FailureGuidanceConfig(),'repair_weight':.001,'element_js_weight':0.,'native_name':'affinity'},acceptance_policy='fixed_step',schedule=schedule,update_offset=5000)
    assert seen==[5000,5001,5002]
    assert [row['repair_weight'] for row in result['states']]==[schedule.for_state(index) for index in seen]
    for row in result['updates']:
        assert row['repair_weight']==row['attempts'][0]['repair_weight']==schedule.for_update(row['global_step']+1)


def test_exact_cpu_boltz_logit_replay_split_equals_uninterrupted(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits,mosaic_spec
    from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
    from mosaic.small_molecule.io import file_sha256
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    assert all(device.platform=='cpu' for device in jax.devices())
    initial=GraphLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    spec=mosaic_spec(25);key=jax.random.key(17)
    def loss(graph,*,key):
        return .1*jnp.sum(graph.X*jnp.arange(6))+.03*jnp.sum(graph.E_upper*jnp.arange(5)),{}
    options={'objective_mode':'native_only','repair_weight':0.,'element_js_weight':0.,'native_name':'affinity'}
    def run(logits,path,steps,offset):
        return run_chemical_updates(loss,logits,prepare=lambda z,u:NativeOnlyPreparation(spec.fingerprint(),u),spec=spec,reference=jnp.ones(6),key=key,directory=path,steps=steps,step_size=.0025,max_backtracks=1,config=options,acceptance_policy='fixed_step',update_offset=offset)
    full=run(initial,tmp_path/'full',4,10000)
    first=run(initial,tmp_path/'first',2,10000)
    snapshot=tmp_path/'first'/first['states'][-1]['snapshot']
    restored=restore_logits(snapshot,file_sha256(snapshot),GraphLogits,key,file_sha256)
    second=run(restored,tmp_path/'second',2,10002)
    with np.load(tmp_path/'full'/full['states'][-1]['snapshot']) as left,np.load(tmp_path/'second'/second['states'][-1]['snapshot']) as right:
        for field in ['X','E_upper','atom_logits','bond_logits','key_data','affinity_gradient_X','affinity_gradient_E_upper']:
            np.testing.assert_array_equal(left[field],right[field])
    assert [row['global_step'] for row in first['updates']+second['updates']]==list(range(10000,10004))
    assert [row['global_step'] for row in first['states']+second['states'][1:]]==list(range(10000,10005))


@pytest.mark.parametrize('field',['X','E_upper','atom_logits','bond_logits','key_data'])
def test_checkpoint_tampering_is_not_accepted(tmp_path,field):
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    from mosaic.small_molecule.io import file_sha256
    logits=GraphLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    values={'X':np.asarray(logits.graph().X),'E_upper':np.asarray(logits.graph().E_upper),'atom_logits':np.asarray(logits.atoms),'bond_logits':np.asarray(logits.bonds),'key_data':np.asarray(jax.random.key_data(jax.random.key(17)))}
    values[field]=values[field].copy();values[field].flat[0]+=1
    path=tmp_path/'bad.npz';np.savez_compressed(path,**values)
    with pytest.raises((ValueError,AssertionError)):
        restore_logits(path,file_sha256(path),GraphLogits,jax.random.key(17),file_sha256)


def test_lazy_checker_keeps_real_evaluator_after_temporary_monkeypatch(monkeypatch):
    import mosaic.small_molecule.chemistry.trajectory as trajectory
    import mosaic.small_molecule.chemistry.checks as checks
    original=trajectory.evaluate_combined
    assert checks.evaluate_combined is original
    def temporary(*args,**kwargs):
        raise AssertionError('Temporary unit fixture must not leak to real checker')
    with monkeypatch.context() as patch:
        patch.setattr(trajectory,'evaluate_combined',temporary)
        assert trajectory.evaluate_combined is temporary
        assert checks.evaluate_combined is original
    assert trajectory.evaluate_combined is original
    assert checks.evaluate_combined is original
