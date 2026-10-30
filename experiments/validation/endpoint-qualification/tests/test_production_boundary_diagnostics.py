"""Portable control-flow contracts plus allocated numerical capture controls."""
import ast
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest

P = Path(__file__).resolve().parents[1] / 'scripts/production_boundary_diagnostics.py'
if not P.exists():
    P = P.with_name('continuation_failure_diagnostics.py')
S = importlib.util.spec_from_file_location('_production_boundary_diagnostic_tests', P)
d = importlib.util.module_from_spec(S)
S.loader.exec_module(d)


def test_metadata_one_native_boundary_stops_before_following_work():
    tree = d.stopped_tree(ast.parse("report['native_gradient_checks']=check()\nfollowed()"))
    calls = []
    with pytest.raises(RuntimeError, match=d.STOP):
        exec(compile(tree, '<toy>', 'exec'), {'report': {}, 'check': lambda: calls.append('check'),
                                            'followed': lambda: calls.append('unexpected')})
    assert calls == ['check']


@pytest.mark.parametrize('count', [0, 2])
def test_metadata_missing_or_duplicate_boundary_fails_closed(count):
    with pytest.raises(ValueError, match='Exactly one'):
        d.stopped_tree(ast.parse("report['native_gradient_checks']=check()\n" * count))


@pytest.mark.parametrize('name', ['run_chemical_updates', 'run_updates', 'update_parameters'])
def test_metadata_optimizer_calls_are_replaced_without_evaluating_arguments(name):
    tree = d.stopped_tree(ast.parse(name + "(side_effect())\nreport['native_gradient_checks']=check()"))
    with pytest.raises(RuntimeError, match='Optimizer execution is forbidden'):
        exec(compile(tree, '<toy>', 'exec'), {'_diagnostic_forbidden_update': d.forbidden_update})


def test_metadata_unexpected_new_errors_do_not_count_as_observed_original_failures():
    assert d.expected_observation(RuntimeError(d.STOP))
    assert d.expected_observation(ValueError(next(iter(d.EXPECTED))))
    assert not d.expected_observation(RuntimeError(next(iter(d.EXPECTED))))
    assert not d.expected_observation(ValueError('CUDA failed'))


def test_metadata_capture_is_not_a_continuation_dispatch():
    calls = {ast.unparse(n.func) for n in ast.walk(ast.parse(P.read_bytes())) if isinstance(n, ast.Call)}
    assert not calls & {'dispatch', 'submit', 'run_updates', 'run_chemical_updates', 'update_parameters'}
    assert d.CAP == 128 * 1024**2


def test_metadata_exact_owned_payload_study_outputs_for_both_actual_startup_commands():
    root = Path(__file__).resolve().parent / 'metadata-only-root'
    plan = {'root': str(root)}
    for phase, section in (('cpu', 'startup'), ('gpu-plain', 'replay'), ('gpu-chemistry', 'replay')):
        for arm in ('plain', 'chemistry'):
            if phase.startswith('gpu-') and phase != 'gpu-' + arm:
                continue
            output = root / phase / 'payload' / section / arm
            study = d.study_path(plan, output)
            assert study == output / 'payload' / 'study'
            assert study.parent.name == 'payload' and study.name == 'study'


def test_metadata_foreign_sibling_and_traversing_output_scopes_are_rejected():
    root = Path(__file__).resolve().parent / 'metadata-only-root'
    for path in (root / 'elsewhere', root.parent / 'foreign',
                 root / 'gpu/payload/replay/plain/../chemistry'):
        with pytest.raises(ValueError, match='Exact owned'):
            d.study_path({'root': str(root)}, path)


@pytest.mark.parametrize('kind,predicate,expected', [
    ('plain', 'declared == "4"', 4),
    ('chemistry', 'declared == ("4" if startup else "8")', 8)])
def test_metadata_actual_reference_predicate_requires_its_own_native_slot_count(kind, predicate, expected):
    text = 'def allocated_environment(startup=False):\n    require(' + predicate + ', "slots")'
    assert d.native_reference_slots(text, kind) == expected


@pytest.mark.parametrize('kind,predicate', [
    ('plain', 'declared == "8"'), ('chemistry', 'declared == "4"'),
    ('chemistry', 'declared == ("8" if startup else "4")')])
def test_metadata_mismatched_or_reversed_reference_allocations_are_rejected(kind, predicate):
    text = 'def allocated_environment(startup=False):\n    require(' + predicate + ', "slots")'
    with pytest.raises(ValueError, match='slot predicate'):
        d.native_reference_slots(text, kind)


def _submission_controls():
    path=P.with_name('failure_diagnostic_ops.py')
    names={'require','release_argv','captured_command','held_observation'}
    tree=ast.Module(body=[n for n in ast.parse(path.read_bytes()).body
                         if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    namespace={'subprocess':subprocess}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace


def test_metadata_only_exact_persisted_registered_held_job_can_be_released():
    namespace=_submission_controls()
    row={'status':'submitted','dispatch_state':'held','returncode':0,'job_id':'123','argv':['bsub','-H','actual-source']}
    assert namespace['release_argv'](row,row['argv'],dict(row),'123')==['bresume','123']


@pytest.mark.parametrize('change',['persisted','status','returncode','job_id','held','registered','dispatch_state'])
def test_metadata_unknown_foreign_unheld_or_unpersisted_jobs_cannot_be_released(change):
    namespace=_submission_controls()
    row={'status':'submitted','dispatch_state':'held','returncode':0,'job_id':'123','argv':['bsub','-H','actual-source']}
    persisted=dict(row);expected=list(row['argv']);registered='123'
    if change=='persisted':persisted['other']='different'
    elif change=='status':row['status']=persisted['status']='unknown_after_timeout'
    elif change=='returncode':row['returncode']=persisted['returncode']=None
    elif change=='job_id':row['job_id']=persisted['job_id']='123;other'
    elif change=='held':expected=row['argv']=persisted['argv']=['bsub','actual-source']
    elif change=='registered':registered='999'
    elif change=='dispatch_state':row['dispatch_state']=persisted['dispatch_state']='released'
    with pytest.raises(ValueError):
        namespace['release_argv'](row,expected,persisted,registered)


def test_metadata_submission_timeout_records_unknown_and_never_retries_or_releases():
    namespace=_submission_controls();calls=[];writes=[]
    def timeout(argv,**kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv,300,output=b'partial response',stderr=b'error')
    namespace['subprocess']=SimpleNamespace(run=timeout,TimeoutExpired=subprocess.TimeoutExpired)
    namespace['write']=lambda output,value:writes.append((output,value))
    with pytest.raises(subprocess.TimeoutExpired):
        namespace['captured_command'](['bsub','-H','actual-source'],'receipt')
    assert calls==[['bsub','-H','actual-source']]
    assert len(writes)==1 and writes[0][0]=='receipt'
    assert writes[0][1]['status']=='unknown_after_timeout_do_not_retry'
    assert writes[0][1]['returncode'] is None and writes[0][1]['stdout']=='partial response'


@pytest.mark.parametrize('state,identity,expected',[
    ('PSUSP','123',True),('PEND','123',False),('RUN','123','reject'),('PSUSP','999','reject')])
def test_metadata_hold_must_settle_on_exact_owned_job_before_release(state,identity,expected):
    namespace=_submission_controls();line=f'{identity} {state} s234818 exact-own-name\n'
    if expected=='reject':
        with pytest.raises(ValueError):namespace['held_observation'](0,line,'123','exact-own-name')
    else:assert namespace['held_observation'](0,line,'123','exact-own-name') is expected


def test_metadata_diagnostic_replicas_have_exact_predeclared_count():
    assert d.REPEATS == 16


def test_metadata_replicas_return_first_even_when_all_later_results_differ():
    results=[]
    def evaluate():
        value={'replica':len(results)};results.append(value);return value
    result=d.repeat_first(evaluate)
    assert len(results)==16 and result is results[0] and result['replica']==0


def test_metadata_replicas_do_not_hide_later_execution_error_with_first_result():
    calls=[]
    def evaluate():
        calls.append(len(calls))
        if len(calls)==2:raise ValueError('actual later error')
        return {'first':'success'}
    with pytest.raises(ValueError,match='actual later error'):d.repeat_first(evaluate)
    assert calls==[0,1]


def test_metadata_replication_does_not_add_optimizer_or_compiler_flag_changes():
    nodes=[n for n in ast.parse(P.read_bytes()).body if isinstance(n,ast.FunctionDef)
           and n.name in ('repeat_first','repeated_measurements')]
    text='\n'.join(ast.unparse(n) for n in nodes)
    assert len(nodes)==2
    assert not any(word in text for word in ('XLA_FLAGS','CUBLAS_WORKSPACE_CONFIG','run_updates','run_chemical_updates'))


@pytest.mark.parametrize('change', [False, True])
def test_measurement_capture_retains_actual_callback_counts_and_exact_original_result(tmp_path, change):
    np = d.numpy()
    calls = []
    def measurement(callbacks, values):
        p, ad, f = callbacks
        scalar = p(values)
        value, gradient = ad(values)
        return {'scalar': float(scalar), 'ad_scalar': float(value), 'frozen': f(),
                'gradient': gradient['X'].tolist()}
    precision = SimpleNamespace(_measurement=measurement)
    values = {'X': np.asarray([[1., 0.]], np.float32), 'E_upper': np.asarray([[0., 1.]], np.float32)}
    current = [0]
    def primal(v):
        calls.append('p')
        return np.float32(1 + current[0])
    def autodiff(v):
        calls.append('ad')
        return np.float32(1), {k: np.ones_like(a) for k, a in v.items()}
    callbacks = (primal, autodiff, lambda: {'model': 'unchanged'})
    with d.captured_measurements(precision, tmp_path / 'measurements') as records:
        first = precision._measurement(callbacks, values)
        current[0] = int(change)
        second = precision._measurement(callbacks, values)
        assert records[1]['versus_first']['primal']['exact_equal'] is (not change)
        assert second['scalar'] == 1 + int(change)
        assert first['ad_scalar'] == second['ad_scalar'] == 1
    assert calls == ['p', 'ad', 'p', 'ad']
    assert precision._measurement is measurement
