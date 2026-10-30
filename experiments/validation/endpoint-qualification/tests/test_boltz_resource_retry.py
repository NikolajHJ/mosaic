"""Portable resource-only retry rejection; no numerical imports or execution."""
import ast
from pathlib import Path
import pytest

def check(value):
    path=Path(__file__).resolve().parents[1]/'scripts/boltz_precision_ops.py'
    tree=ast.parse(path.read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='predecessor_check')
    def require(v,message):
        if not v:raise ValueError(message)
    ns={'require':require};exec(compile(ast.Module([fn],[]),'actual_predecessor_guard','exec'),ns)
    return ns['predecessor_check'](value)

def example():
    names=['test_reference_precision_context_restores_production_configuration','test_probability_to_logit_chain_matches_independent_AD',
        'test_fixed_target_gradient_flows_without_target_refresh']
    return {'job_ids':{'cpu':'29852949','after-cpu':'29852950'},'status':'fixture_API_ERROR_before_native_startup',
        'actual_numerical_cases':51,'passed_cases':48,'scientific_outputs':0,'GPU_jobs':[],'pretrained_models_constructed':0,
        'failures':[{'name':n,'module':'tests.test_boltz_reference_jax','message':"module 'jax.experimental' has no attribute 'enable_x64'"} for n in names]}

def test_exact_API_failure_is_eligible_for_fresh_full_qualification():check(example())

@pytest.mark.parametrize('field,value',[('failures',[]),('scientific_outputs',1),('GPU_jobs',['unknown'])])
def test_numerical_or_unknown_execution_cannot_be_blindly_retried(field,value):
    p=example();p[field]=value
    with pytest.raises(ValueError):check(p)
