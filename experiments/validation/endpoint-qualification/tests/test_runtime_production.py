"""Portable namespace/AST/resource contracts; no arrays, models or fake allocation."""
import ast
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parents[1]
def owner(name):
    path=HERE/'scripts'/(name+'.py')
    spec=importlib.util.spec_from_file_location('_portable_'+name,path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

capacity=owner('runtime_production_capacity');driver=owner('runtime_production_driver')

PLAIN='''def plain(plan, output, *, startup=False):
    argv=['worker','--phase','validate','--offset','20000','--output',str(output)]
    try:
        compiled=stopped_tree(original_tree)
        base.main()
    except BaseException as error:
        if startup: raise
        return records, str(error)
    raise ValueError('old sentinel')
'''

def test_segment_preserves_original_command_except_declared_phase_ready():
    tree=driver.segment_tree(PLAIN);lists=[n for n in ast.walk(tree) if isinstance(n,ast.List)]
    command=next(n for n in lists if any(isinstance(v,ast.Constant) and v.value=='--phase' for v in n.elts))
    values=[n.value if isinstance(n,ast.Constant) else ast.unparse(n) for n in command.elts]
    assert values[:7]==['worker','--phase','segment','--offset','20000','--output','str(output)']
    assert values[7:]==['--ready',"plan['qualified']['ready']['path']",'--ready-sha256',"plan['qualified']['ready']['sha256']"]
    assert any(isinstance(n,ast.Raise) and n.exc is None for n in ast.walk(tree))
    assert not any(isinstance(n,ast.Name) and n.id=='stopped_tree' for n in ast.walk(tree))
    assert isinstance(tree.body[0].body[-1],ast.Return)

@pytest.mark.parametrize('before,after',[('def plain','def chemistry'),("'validate'","'segment'"),("'--phase'","'--phase-missing'"),('stopped_tree(original_tree)','original_tree'),('except BaseException','except Exception')])
def test_undeclared_startup_shape_fails_closed(before,after):
    with pytest.raises((ValueError,AttributeError)):driver.segment_tree(PLAIN.replace(before,after))

@pytest.mark.parametrize('count',[0,2])
def test_exact_capacity_namespace_patch_count_is_mandatory(count):
    with pytest.raises(ValueError):capacity.replace_expression(ast.parse('x=a+1'),'a','b',count=count)

@pytest.mark.parametrize('name', ['build_scopes','_storage','classify_jobs','segment_certificate'])
def test_capacity_patch_inventory_contains_only_namespace_or_registration(name):
    source=(HERE/'scripts/runtime_production_capacity.py').read_text()
    assert name in source
    assert 'c.sparse_segment_cap' not in source and 'c.assess(' not in source
    assert 'inventory_selections' not in source and 'parse_quota' not in source

def test_source_scope_is_distinct_from_shared_control():
    tree=ast.parse((HERE/'scripts/runtime_production_ops.py').read_text())
    defs={n.targets[0].id:ast.unparse(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
    assert defs['SOURCE']=="BASE / 'runtime-production-release-05/source'"
    assert defs['ROOT']=="SCOPE / 'runtime-production-05'"

@pytest.mark.parametrize('needle', ["'05:30'","'num=1:mode=exclusive_process:mps=yes'","'8GB'","'19800'","'19200'"])
def test_full_measured_remaining_horizon_resources(needle):
    assert needle.strip("'") in (HERE/'scripts/runtime_production_ops.py').read_text()

@pytest.mark.parametrize('needle',["'offset':20000","'updates':30000","'global_target':50000","'learning_rate':0.3","'matrix_stride':20","'replicas_per_native_checkpoint':16"])
def test_unchanged_scientific_policy_is_closed(needle):
    assert needle in (HERE/'scripts/runtime_production_driver.py').read_text()

def test_metadata_budget_preserves_original_inventory_and_configuration():
    source=inspect.getsource(driver.reference_budget)
    assert 'helper.diagnostic_budget_tree(inspect.getsource(original))' in source
    assert "study/'native_gradient_checks-prospective'" in source and "study/'terminal_gradient_checks-prospective'" in source
    assert 'precision.SESSION_CAP==64*1024**2' in source and '128*1024**2' in source
    assert 'finally:precision._budget=original' in source

def test_native_check_removal_is_absent():
    source=(HERE/'scripts/runtime_production_driver.py').read_text()
    assert "base.main()" not in source # inherited call is compiled verbatim
    assert 'validation_lookup' in source and 'node.value.orelse=' in source
    assert 'node.value.body=' not in source

def test_periodic_shared_gate_and_owned_cleanup_remain():
    source=(HERE/'scripts/runtime_production_ops.py').read_text()
    assert "storage(op,parent,'periodic')" in source and 'pilot.run_isolated_child' in source
    assert 'pilot._parent_sigterm_cleanup()' in source and 'limit_seconds' in source

def test_callback_preserves_full_original_result_verification():
    source=inspect.getsource(driver.validation_lookup)
    assert "node.targets[0].id=='expected_validation'" in source
    ops=(HERE/'scripts/runtime_production_ops.py').read_text()
    assert ".verify_precision_result(result,original,CAPACITY_PIN,'segment'" in ops
    assert "result['terminal_snapshot']" in ops and "result['frozen_unchanged'] is True" in ops

def test_existing_held_submission_protocol_reused_without_retry():
    source=(HERE/'scripts/runtime_production_ops.py').read_text()
    assert 'submit=q.submit' in source and 'q.argv=argv' in source and "'bsub','-H'" in source
    assert "'ended('+gpu['job_id']+')'" in source

def test_authenticated_dependency_path_precedes_fixture_imports():
    tree=ast.parse((HERE/'scripts/runtime_production_ops.py').read_text())
    context=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='context')
    paths=[n for n in ast.walk(context) if isinstance(n,ast.Call) and ast.unparse(n.func)=='sys.path.insert']
    assert len(paths)==1 and "plan['chemistry_source']['path']" in ast.unparse(paths[0])
    execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    text=ast.unparse(execute)
    assert text.index("namespace['premodel'](plan") < text.index('q.fixture_suite(plan, payload)')

def test_original_startup_receipt_uses_actual_cli_not_invented_readiness_field():
    ready={'path':'/qualified/study/assessment.json','sha256':'abc'}
    value={'status':'passed_metadata_only','job_id':'12','numerical_setup_entered':False,'updates':0,
        'readiness_deferred':False,'original_argv':['driver','--ready',ready['path'],'--ready-sha256','abc',
        '--offset','20000','--segment-steps','30000','--mode','run','--resume-sha256',
        '6be2b3868d58162d4aa3cc9d960cfb85a84d4078748f0b24d50b138ae3e47dbd']}
    assert 'readiness' not in value
    assert driver.check_segment_startup(value,ready,'12') is value
    with pytest.raises(ValueError):driver.check_segment_startup(value,{**ready,'sha256':'wrong'},'12')

def test_startup_requires_each_original_ready_flag_exactly_once():
    with pytest.raises(ValueError):driver.check_segment_startup({'status':'passed_metadata_only','job_id':'12',
        'numerical_setup_entered':False,'updates':0,'readiness_deferred':False,
        'original_argv':['--ready','/r','--ready','/r']},{'path':'/r','sha256':'x'},'12')
