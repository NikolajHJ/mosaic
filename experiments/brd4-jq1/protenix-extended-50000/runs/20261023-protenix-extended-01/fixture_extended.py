"""Allocated controller scope, exact checkpoint and split-trajectory contracts."""
import copy, json
from pathlib import Path
import pytest
import extended_ops as ops
from extended_protocol import RANDOM, CONTINUATION, segment_scope, projection, restore_logits
from campaign25_contract import config25

def cfg(arm):return json.loads((Path(__file__).resolve().parents[1]/'examples/configs'/('extended_'+arm+'.json')).read_bytes())

@pytest.mark.parametrize('arm',['random','continuation'])
def test_exact_declared_extended_config(arm):
    c=cfg(arm);assert config25(c)==c and c['steps']==50000 and c['initial_step_size']==.3

@pytest.mark.parametrize('arm',['random','continuation'])
@pytest.mark.parametrize('field,value',[('steps',40000),('seed',18),('initial_step_size',3.),('max_backtracks',2),('n_atoms',24),('allocation_seconds',86400),('element_js_weight',.1),('acceptance_policy','best_primal')])
def test_scientific_policy_not_silently_changed(arm,field,value):
    c=cfg(arm);c[field]=value
    with pytest.raises(ValueError):config25(c)

@pytest.mark.parametrize('field,value',[('hold_updates',10000),('total_updates',10000),('final',50),('initial',.0001)])
def test_user_stretched_schedule_is_exact(field,value):
    c=cfg('random');c['repair_schedule'][field]=value
    with pytest.raises(ValueError):config25(c)

@pytest.mark.parametrize('arm,index',[('random',i) for i in range(5)]+[('continuation',i) for i in range(4)])
def test_exact_global_segment_ranges(arm,index):
    start=ops.offset(arm,index);snap='/saved.npz' if start else None;digest='a'*64 if start else None
    x=segment_scope(cfg(arm),start,10000,snap,digest)
    assert x['start']==start and x['end']==start+10000 and x['target']==50000
    args=ops.command(ops.name(arm,'segment',index),'segment','b'*64)
    assert '-w' not in args and args[args.index('-W')+1]=='23:00'
    assert args[args.index('-gpu')+1]=='num=1:mode=exclusive_process'

@pytest.mark.parametrize('offset,steps,snapshot,digest',[(0,50000,None,None),(5000,10000,'p','a'),(50000,10000,'p','a'),(10000,10000,None,None),(0,10000,'p','a'),(True,10000,None,None),(0,9999,None,None)])
def test_bad_boundary_scope_fails(offset,steps,snapshot,digest):
    with pytest.raises(ValueError):segment_scope(cfg('random'),offset,steps,snapshot,digest)

@pytest.mark.parametrize('arm,index',[('random',5),('continuation',4),('boltz2',0),('random',-1)])
def test_only_two_declared_finite_chains(arm,index):
    with pytest.raises(ValueError):ops.offset(arm,index)

@pytest.mark.parametrize('unit,updates,segments',[(0,50000,5),(1.,50000,5),(10,10000,1),(10,50000,1),(10,40000,5)])
def test_full_projection_never_uses_one_segment_budget(unit,updates,segments):
    with pytest.raises(ValueError):projection(unit,updates,segments)

def test_full_projection_and_independent_chain_ends():
    assert projection(14000,50000,5)==2*14000*50000+5*1024**3
    assert ops.offset('random',4)+10000==ops.offset('continuation',3)+10000==50000

def test_numeric_global_schedule_boundaries():
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    s=RepairWeightSchedule(**cfg('random')['repair_schedule'])
    assert s.for_update(1)==s.for_update(5000)==.001
    assert .001<s.for_update(10000)<100 and s.for_update(50000)==100
    assert s.for_state(10000)==s.for_update(10001)

def test_real_chemistry_schedule_and_preparations_keep_global_offset(tmp_path):
    import jax
    import jax.numpy as jnp
    from protenix_initialization import ProtenixReplayLogits
    from mosaic.small_molecule.chemistry import mosaic_spec,prepare_failure_guidance,FailureGuidanceConfig
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    initial=ProtenixReplayLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    spec=mosaic_spec(25);seen=[];schedule=RepairWeightSchedule(**cfg('random')['repair_schedule'])
    def prepare(z,u):
        seen.append(u)
        return prepare_failure_guidance(z.decoded(),spec,update=u)
    def loss(graph,*,key):return .1*jnp.sum(graph.X*jnp.arange(6))+.03*jnp.sum(graph.E_upper*jnp.arange(5)),{}
    r=run_chemical_updates(loss,initial,prepare=prepare,spec=spec,reference=jnp.ones(6),key=jax.random.key(17),directory=tmp_path/'offset-chemistry',steps=2,step_size=.3,max_backtracks=1,config={'loss':FailureGuidanceConfig(),'repair_weight':.001,'element_js_weight':0.,'native_name':'contact'},acceptance_policy='fixed_step',schedule=schedule,update_offset=5000)
    assert seen==[5000,5001,5002]
    assert [s['repair_weight'] for s in r['states']]==[schedule.for_state(i) for i in seen]
    for u in r['updates']:
        assert u['repair_weight']==u['attempts'][0]['repair_weight']==schedule.for_update(u['global_step']+1)

def test_exact_checkpoint_replay_and_split_matches_uninterrupted(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from protenix_initialization import ProtenixReplayLogits
    from mosaic.small_molecule.chemistry import mosaic_spec
    from mosaic.small_molecule.chemistry.native_only import NativeOnlyPreparation
    from mosaic.small_molecule.chemistry.trajectory import run_chemical_updates
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    from mosaic.small_molecule.io import file_sha256
    initial=ProtenixReplayLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    spec=mosaic_spec(25);key=jax.random.key(17)
    def loss(graph,*,key):return .1*jnp.sum(graph.X*jnp.arange(6))+.03*jnp.sum(graph.E_upper*jnp.arange(5)),{}
    options={'objective_mode':'native_only','repair_weight':0.,'element_js_weight':0.,'native_name':'contact'}
    def run(z,path,steps,offset):
        return run_chemical_updates(loss,z,prepare=lambda z,u:NativeOnlyPreparation(spec.fingerprint(),u),spec=spec,reference=jnp.ones(6),key=key,directory=path,steps=steps,step_size=.3,max_backtracks=1,config=options,acceptance_policy='fixed_step',update_offset=offset)
    full=run(initial,tmp_path/'full',4,10000);left=run(initial,tmp_path/'left',2,10000)
    source=tmp_path/'left'/left['states'][-1]['snapshot']
    restored=restore_logits(source,file_sha256(source),ProtenixReplayLogits,key,file_sha256)
    right=run(restored,tmp_path/'right',2,10002)
    with np.load(tmp_path/'full'/full['states'][-1]['snapshot']) as a,np.load(tmp_path/'right'/right['states'][-1]['snapshot']) as b:
        for field in ['X','E_upper','atom_logits','bond_logits','key_data','contact_gradient_X','contact_gradient_E_upper']:
            np.testing.assert_array_equal(a[field],b[field])
    assert [x['global_step'] for x in left['updates']+right['updates']]==list(range(10000,10004))
    assert [x['global_step'] for x in left['states']+right['states'][1:]]==list(range(10000,10005))
    with pytest.raises(ValueError):restore_logits(source,'0'*64,ProtenixReplayLogits,key,file_sha256)

@pytest.mark.parametrize('field',['X','E_upper','atom_logits','bond_logits','key_data'])
def test_checkpoint_tampering_is_not_accepted(tmp_path,field):
    import jax
    import numpy as np
    from protenix_initialization import ProtenixReplayLogits
    from mosaic.small_molecule.state import graph_from_hard,mix_with_uniform
    from mosaic.small_molecule.io import file_sha256
    z=ProtenixReplayLogits.from_graph(mix_with_uniform(graph_from_hard([0]*25,[int(j==i+1) for i in range(25) for j in range(i+1,25)]),.5))
    values={'X':np.asarray(z.graph().X),'E_upper':np.asarray(z.graph().E_upper),'atom_logits':np.asarray(z.atoms),'bond_logits':np.asarray(z.bonds),'key_data':np.asarray(jax.random.key_data(jax.random.key(17)))}
    values[field]=values[field].copy();values[field].flat[0]+=1
    p=tmp_path/'bad.npz';np.savez_compressed(p,**values)
    with pytest.raises((AssertionError,ValueError)):restore_logits(p,file_sha256(p),ProtenixReplayLogits,jax.random.key(17),file_sha256)
