"""Portable validation routing contracts; no scientific arrays or runtime imports."""
import ast
import importlib.util
import json
from pathlib import Path
import pytest

P=Path(__file__).resolve().parents[1]/'scripts/protenix_runtime_qualification.py'
if not P.exists():P=P.with_name('continuation_failure_diagnostics.py')
spec=importlib.util.spec_from_file_location('_runtime_qualification_metadata_tests',P)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def helper_source():
    plan=P.parents[1]/'plan.json'
    if plan.exists():return Path(json.loads(plan.read_bytes())['helper_source']['path'])/'scripts/continuation_failure_diagnostics.py'
    repo=P.parents[4]
    return repo/'results/hpc/20261030-continuation-failure-diagnostics-10/source01/scripts/continuation_failure_diagnostics.py'


def inherited_function(name):
    tree=ast.parse(helper_source().read_bytes())
    return ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))


@pytest.mark.parametrize('kind',['plain','chemistry'])
def test_metadata_actual_full_validation_removes_one_diagnostic_stop_only(kind):
    tree=q.validation_tree(inherited_function(kind),kind)
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='stopped_tree' for n in ast.walk(tree))
    assert isinstance(tree.body[0].body[-1],ast.Return)
    assert tree.body[0].body[-1].value.elts[1].value==q.SUCCESS
    for n in ast.walk(tree):
        if isinstance(n,ast.ExceptHandler):assert len(n.body)==1 and isinstance(n.body[0],ast.Raise)
    # Unchanged capture/16-replica contexts retain original replica0 selection.
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'captured_measurements' in calls and 'repeated_measurements' in calls


@pytest.mark.parametrize('kind',['plain','chemistry'])
def test_metadata_an_optimizer_command_cannot_replace_validation(kind):
    source=inherited_function(kind)
    if kind=='plain':source=source.replace("'--phase', 'validate'","'--phase', 'segment'")
    else:source=source.replace("'protenix', 'validate', study","'protenix', 'segment', study")
    with pytest.raises(ValueError,match='forbidden'):q.validation_tree(source,kind)


@pytest.mark.parametrize('kind',['dbmol','boltz','segment','unknown'])
def test_metadata_no_other_arm_or_segment_is_qualified(kind):
    with pytest.raises(ValueError,match='two Protenix'):q.validation_tree(inherited_function('plain'),kind)


def test_metadata_reference_CLI_loop_keeps_both_reference_metadata_checks():
    tree=q.reference_startup_tree(inherited_function('premodel'))
    loops=[n for n in tree.body[0].body if isinstance(n,ast.For)]
    assert len(loops)==1 and ast.unparse(loops[0].iter)=="('plain', 'chemistry')"
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'worker.main' in calls and 'worker.validate_output' in calls


def policy():
    return {'qualification_arms':['plain','chemistry'],'qualification_policy':{'phase':'validate',
        'optimizer_submission_allowed':False,'both_complete_chemistry_pilots_required':True,
        'replicas_per_native_checkpoint':16},'diagnostic_storage':{'group_cap_bytes':512*1024**2,
        'reference_metadata_cap_bytes':128*1024**2,'inherited_reference_cap_bytes':64*1024**2,
        'all_output_retained':True,'optimizer_admission_allowed':False}}


def test_metadata_explicit_finite_budget_and_validation_only_policy():
    q.validate_policy(policy())


@pytest.mark.parametrize('change',['segment','optimizer','drop_pilot','budget','drop_output'])
def test_metadata_full_checks_and_storage_cannot_be_weakened(change):
    p=policy()
    if change=='segment':p['qualification_policy']['phase']='segment'
    elif change=='optimizer':p['qualification_policy']['optimizer_submission_allowed']=True
    elif change=='drop_pilot':p['qualification_policy']['both_complete_chemistry_pilots_required']=False
    elif change=='budget':p['diagnostic_storage']['group_cap_bytes']=1024**4
    else:p['diagnostic_storage']['all_output_retained']=False
    with pytest.raises(ValueError):q.validate_policy(p)


@pytest.mark.parametrize('phase,slots,queue,wall',[
    ('cpu','4','hpc','01:00'),('after-cpu','1','hpc','00:30'),
    ('gpu-plain','4','gpul40s','02:00'),('gpu-chemistry','8','gpul40s','02:00'),
    ('after-plain','4','hpc','01:00'),('after-chemistry','4','hpc','01:00')])
def test_metadata_actual_generated_allocations_match_original_roles(phase,slots,queue,wall):
    source=P.with_name('protenix_runtime_qualification_ops.py')
    if not source.exists():source=source.with_name('failure_diagnostic_ops.py')
    nodes=[n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in ('require','argv')]
    ns={'Path':Path,'ROOT':Path('/owned/new-qualification'),'SOURCE':Path('/owned/new-source'),'CAPACITY':Path('/owned/original-source')}
    exec(compile(ast.Module(nodes,[]),'<metadata-only-command>','exec'),ns)
    a=ns['argv'](phase,'a'*64)
    assert a[:2]==['bsub','-H'] and a[a.index('-n')+1]==slots and a[a.index('-q')+1]==queue and a[a.index('-W')+1]==wall
    assert ('-gpu' in a)==phase.startswith('gpu-')


@pytest.mark.parametrize('elapsed,expected',[(60,6540),(6599,1),(6600,None)])
def test_metadata_complete_entry_and_child_fit_the_finite_GPU_reservation(elapsed,expected):
    source=P.with_name('protenix_runtime_qualification_ops.py')
    if not source.exists():source=source.with_name('failure_diagnostic_ops.py')
    nodes=[n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('require','native_budget')]
    ns={};exec(compile(ast.Module(nodes,[]),'<metadata-only-runtime>', 'exec'),ns)
    if expected is None:
        with pytest.raises(ValueError,match='deadline'):ns['native_budget'](elapsed)
    else:assert ns['native_budget'](elapsed)==expected

