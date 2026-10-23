"""Allocated new random/chemistry configuration, initialization and admission contracts."""
import copy,json
from pathlib import Path
import pytest
import campaign25_contract as contract
import random_chem_ops as ops

def config():
    return json.loads((Path(__file__).resolve().parents[1]/'examples/configs/protenix_random_delayed_chem_10000.json').read_bytes())

def test_declared_random_protenix_chemistry_protocol():
    c=config();assert contract.config25(c)==c
    assert contract.update_parameters(c)=={'step_size':.3,'max_backtracks':1,'acceptance_policy':'fixed_step'}
    assert contract.chemistry_validation_weights(c)==(('low',.001),('high',100.0))

@pytest.mark.parametrize('key,value',[
 ('arm','boltz2'),('arm','boltz2_chemistry'),('chemistry_enabled',False),('initial_step_size',3.),
 ('initial_step_size',.03),('initial_step_size',float('nan')),('max_backtracks',2),('steps',9999),
 ('seed',18),('pilot_steps',9),('allocation_seconds',82801),('element_js_weight',.1),
 ('initialization',{'kind':'chain_uniform','mixing':.5}),
 ('initialization',{'kind':'dirichlet','concentration':1.,'seed':18}),
 ('initialization',{'kind':'dirichlet','concentration':.1,'seed':17}),
 ('initialization',{'kind':'dirichlet','concentration':1.,'seed':17,'redraw':True}),
 ('acceptance_policy','best_primal'),('repair_schedule',None),
 ('repair_schedule',{'version':'delayed-geometric-v1','initial':.001,'final':50.,'hold_updates':5000,'total_updates':10000}),
 ('loss',{'version':'repair-failure-guidance-v1'}),('search',{}),('progress_search',{}),('fixture','other.json'),
 ('pocket',None),('n_atoms',24),('n_atoms',True),('steps',True),('max_backtracks',True),
])
def test_new_version_rejects_undeclared_changes(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):contract.config25(c)

def test_new_version_rejects_undeclared_fields():
    c=config();c['clip']=1.
    with pytest.raises(ValueError):contract.config25(c)

def test_random_initializer_reproducible_positive_and_separate_native_key():
    import jax,jax.numpy as jnp,numpy as np
    from mosaic.small_molecule.state import RelaxedGraph
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    c=config();shape=RelaxedGraph(jnp.full((25,6),1/6),jnp.full((300,5),1/5))
    key=jax.random.key(17);before=np.asarray(jax.random.key_data(key)).copy()
    a=initialize_graph(shape,c['initialization']);b=initialize_graph(shape,c['initialization'])
    np.testing.assert_array_equal(a.X,b.X);np.testing.assert_array_equal(a.E_upper,b.E_upper)
    np.testing.assert_array_equal(before,np.asarray(jax.random.key_data(key)))
    assert a.X.shape==(25,6) and a.E_upper.shape==(300,5)
    assert np.all(np.asarray(a.X)>0) and np.all(np.asarray(a.E_upper)>0)
    assert not np.array_equal(a.X,shape.X) and not np.array_equal(a.E_upper,shape.E_upper)
    np.testing.assert_allclose(np.asarray(a.X).sum(-1),1,atol=1e-6,rtol=0)
    np.testing.assert_allclose(np.asarray(a.E_upper).sum(-1),1,atol=1e-6,rtol=0)

@pytest.mark.parametrize('phase,hours,queue,slots',[('cpu','01:00','hpc','4'),('after-cpu','00:30','hpc','1'),('validate','02:00','gpul40s','4'),('after-ready','00:30','hpc','1'),('run','23:00','gpul40s','4')])
def test_exact_allocated_resources(phase,hours,queue,slots):
    a=ops.resources(phase);assert a[a.index('-q')+1]==queue and a[a.index('-n')+1]==slots and a[a.index('-W')+1]==hours
    if queue=='gpul40s':assert a[a.index('-gpu')+1]=='num=1:mode=exclusive_process'

def test_combined_budget_accounts_all_existing_and_new_remaining():
    old={'protenix':100,'boltz2':200,'boltz2_chemistry':300}
    assert ops.capacity(5000,1000,old)==sum(ops.OLD_FULL.values())-600+4000+2*1024**3+512*1024**2

@pytest.mark.parametrize('full,current,old',[(True,0,ops.OLD_FULL),(100,101,ops.OLD_FULL),(100,-1,ops.OLD_FULL),(100,0,{}),(100,0,{'protenix':-1,'boltz2':0,'boltz2_chemistry':0})])
def test_malformed_or_exceeded_storage_fails(full,current,old):
    with pytest.raises(ValueError):ops.capacity(full,current,old)

@pytest.mark.parametrize('phase',['prepare','boltz2','stop','resume','diagnostic'])
def test_no_other_phases(phase):
    with pytest.raises(ValueError):ops.resources(phase)

def test_driver_command_only_single_new_source_scope():
    a=ops.command('run','run','a'*64)
    assert str(ops.SOURCE/'scripts/hpc/random_chem_stage.lsf') in a
    assert str(ops.ROOT/'run')==a[-1] and str(ops.OLD) not in a
    assert not any(x in a for x in ['bkill','resume','boltz2'])
