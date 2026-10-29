"""Portable metadata/AST controls; full allocation repeats these with all944 inherited fixtures."""
import ast
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest

OWNER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OWNER/'scripts'))
import chemistry_continuation_contract as c
import chemistry_continuation_hooks as h
import chemistry_continuation_driver as d


def plan():
    result={'version':c.VERSION,'root':str(c.ROOT),'source_root':str(c.SOURCE),
            'original_results_changed':False,'optimization_settings_changed':False,'original_failures_preserved':True,
            'raw_policy':'prospective-endpoint-admission-v3','logit_policy':'prospective-fixed-reference-logit-admission-v1',
            'component_only_can_submit':False,'both_complete_pilots_required':True,'actual_coefficient_check_required':True,'arms':{}}
    for name,a in c.ARMS.items():
        result['arms'][name]={**a,'id':name,'chemistry_enabled':True,'element_js_weight':0,'confidence_margin_enabled':False,
                'matrix_stride':20,'loss_stride':1,'max_backtracks':1,'acceptance_policy':'fixed_step',
                'repair_schedule':{'version':'delayed-geometric-v1','initial':.001,'final':100.,'hold_updates':5000,'total_updates':a['target']}}
    return result


def test_closed_original_scope():assert c.validate_plan(plan())['arms']['protenix']['remaining']==40000


def test_exact_empty_metadata_proof_is_not_a_missing_or_changed_file(tmp_path):
    path=tmp_path/'preserved-native.log';path.write_bytes(b'')
    proof=c.binding(path);assert proof['bytes']==0 and c.bound(proof)==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):c.bound(proof)


@pytest.mark.parametrize('arm',['protenix','boltz'])
def test_original_frozen_response_contract_keeps_worker_transport_fields(arm):
    expected={'model_digest':'model','context_digest':'context'}
    response={**expected,**({'id':7,'status':'passed','elapsed_seconds':.2} if arm=='boltz' else {})}
    summary={**expected,'frozen_identity_after':response};calls=[]
    def original(after,model,context):
        calls.append((after,model,context));assert after['status']=='passed'
        assert after['model_digest']==model and after['context_digest']==context
    assert c.frozen_identity(arm,summary,expected,original)
    assert calls==([(response,'model','context')] if arm=='boltz' else [])
    assert summary['frozen_identity_after']==response


@pytest.mark.parametrize('arm',['protenix','boltz'])
def test_mutated_frozen_identity_still_rejected(arm):
    expected={'model_digest':'model','context_digest':'context'}
    summary={**expected,'frozen_identity_after':{**expected,'model_digest':'changed'}}
    def original(after,model,context):
        if after['model_digest']!=model or after['context_digest']!=context:raise ValueError('Original frozen gate failed')
    with pytest.raises(ValueError):c.frozen_identity(arm,summary,expected,original)


@pytest.mark.parametrize('arm,field,value',[('protenix','rate',.03),('boltz','target',50000),('protenix','start',0),
    ('boltz','element_js_weight',1.),('protenix','confidence_margin_enabled',True),('boltz','max_backtracks',2),
    ('protenix','matrix_stride',100),('boltz','remaining',10000)])
def test_settings_changes_rejected(arm,field,value):
    p=plan();p['arms'][arm][field]=value
    with pytest.raises(ValueError):c.validate_plan(p)


@pytest.mark.parametrize('field',['component_only_can_submit','both_complete_pilots_required','actual_coefficient_check_required'])
def test_component_never_substitutes_for_full_gates(field):
    p=plan();p[field]=not p[field]
    with pytest.raises(ValueError):c.validate_plan(p)


@pytest.mark.parametrize('arm',tuple(c.ARMS))
def test_full_measured_horizon(arm):
    result=c.runtime({'runtime_gate':{'status':'passed','mean_update_seconds':[.2,.3]},'elapsed_seconds':120},arm)
    assert result['requested_updates']==c.ARMS[arm]['remaining'] and result['walltime_seconds']%900==0
    assert result['walltime_seconds']>=result['projected_seconds']+600


def test_no_silent_shortening_on_runtime_failure():
    with pytest.raises(ValueError,match='does not fit'):
        c.runtime({'runtime_gate':{'status':'passed','mean_update_seconds':[100,100]},'elapsed_seconds':120},'protenix')


@pytest.mark.parametrize('phase,queue,slots,memory',[('cpu','hpc',4,'4GB'),('after-cpu','hpc',1,'2GB'),
    ('validate','gpul40s',8,'4GB'),('after-validate','hpc',1,'2GB'),('segment','gpul40s',8,'4GB'),
    ('after-segment','hpc',4,'4GB')])
def test_full_horizon_certificate_has_host_memory_without_reserving_gpu(phase,queue,slots,memory):
    assert c.allocation(phase)=={'queue':queue,'slots':slots,'memory_per_slot':memory}


@pytest.mark.parametrize('updates',[20000,40000])
def test_complete_stride20_projection_preserves_loss_and_remaining_byte_budget(updates):
    forecast=c.output_projection(123456,updates,1000)
    assert forecast['full_bytes']==2*123456*((updates+19)//20+3)+1024**3
    assert forecast['remaining_growth_bytes']+1000==forecast['full_bytes']
    assert forecast['loss_stride']==1 and forecast['matrix_stride']==20 and forecast['safety_factor']==2


@pytest.mark.parametrize('unit,updates,actual',[(0,40000,0),(100,10000,0),(100,20000,10**15)])
def test_projection_rejects_partial_horizon_or_exceeded_cap(unit,updates,actual):
    with pytest.raises(ValueError):c.output_projection(unit,updates,actual)


@pytest.mark.parametrize('means',([1],[],[1,float('nan')],[-1,2]))
def test_complete_two_pilot_timings_required(means):
    with pytest.raises(ValueError):c.runtime({'runtime_gate':{'status':'passed','mean_update_seconds':means},'elapsed_seconds':120},'boltz')


def test_added_coefficient_does_not_replace_original_cohorts():
    text='''def main():
    original_low_high_checks()
    means, sizes = [], []
    both_original_ten_update_pilots()
'''
    result=d.actual_coefficient_tree(ast.parse(text));shown=ast.unparse(result)
    assert 'original_low_high_checks()' in shown and 'both_original_ten_update_pilots()' in shown
    assert 'schedule.for_state(args.offset)' in shown and "'actual_coefficient_coupled_checks'" in shown
    assert shown.index('actual_options')<shown.index('means, sizes')


def test_horizon_ast_preserves_optimizer_call():
    raw='''def main():
    result=run_chemical_updates(steps=args.segment_steps)
    if result["status"] != "passed" or len(result["updates"]) != 10000 or [r["step"] for r in result["states"]] != list(range(10001)):
        raise ValueError("Incomplete")
'''
    tree=h.driver_tree(raw,'authenticated_original.py','protenix');result=ast.unparse(tree)
    assert 'run_chemical_updates(steps=args.segment_steps)' in result
    assert 'range(args.segment_steps + 1)' in result
    with pytest.raises(ValueError):h.driver_tree(raw.replace('10001','10002'),'changed.py','protenix')


def test_generated_original_command_keeps_inputs_and_original_vector():
    p=plan();row=p['arms']['boltz'];old=['python','original.py','--mode','run','--config','exact.json','--output','old',
       '--manifest','original-pin','--ready','old-ready','--ready-sha256','old-hash','--segment-steps','10000']
    row.update(original_argv=old[:],boundary={'path':'saved.npz','sha256':'immutable'})
    result=d.argv(p,'boltz','segment',Path('/new/study'),{'path':'new-ready','sha256':'new-hash'})
    assert row['original_argv']==old and result[result.index('--config')+1]=='exact.json'
    assert result[result.index('--manifest')+1]=='original-pin'
    assert result[result.index('--segment-steps')+1]=='20000'
    assert result[result.index('--resume-snapshot')+1]=='saved.npz'
    assert result[result.index('--ready')+1]=='new-ready'


def test_raw_and_combined_algorithms_are_not_reimplemented():
    tree=ast.parse((OWNER/'scripts/chemistry_continuation_hooks.py').read_bytes())
    imports=[n for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any('numpy' in ast.unparse(n) or 'jax' in ast.unparse(n) for n in imports)
    text=(OWNER/'scripts/chemistry_continuation_ops.py').read_text(encoding='utf8')
    assert "component(plan,arm)" in text and "all((c.ROOT/(name+'-ready.json')).exists()" in text
    assert "original_failed_parent_recertified" in text
