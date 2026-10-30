"""Portable metadata/AST controls; no arrays, models or fabricated allocation."""
import ast
import importlib.util
from pathlib import Path,PurePosixPath
import pytest

OWNER=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_precision_reference_metadata',OWNER/'scripts/boltz_objective_reference.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
SAMPLE='''def validate_graph(x, edges, context):
    assert x.dtype == torch.float32 and edges.dtype == torch.float32
def contact_distogram_float64(module, pair):
    assert pair.dtype == torch.float32 and module.weight.dtype == torch.float32 and module.bias.dtype == torch.float32
    return F.linear(pair.double(), module.weight.double(), module.bias.double())
'''

def test_dtype_only_reference_adapter_retains_every_math_expression():
    tree=reference.reference_tree(SAMPLE,'original')
    assert ast.unparse(tree).count('torch.float64')==5
    assert ast.unparse(tree).endswith('return F.linear(pair.double(), module.weight.double(), module.bias.double())')

@pytest.mark.parametrize('change',[lambda s:s.replace('edges.dtype == torch.float32','True'),
    lambda s:s.replace('module.bias.dtype == torch.float32','True'),lambda s:s.replace('def validate_graph','def other')])
def test_unknown_reference_source_shape_rejects(change):
    with pytest.raises(ValueError):reference.reference_tree(change(SAMPLE),'unrecognized')

def vectors():
    tree=ast.parse((OWNER/'scripts/boltz_precision_ops.py').read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='argv')
    ns={'ROOT':PurePosixPath('/scope/new'),'SOURCE':PurePosixPath('/release/new'),'SHARED':PurePosixPath('/admitted/source'),
        'require':reference.require};exec(compile(ast.Module([fn],[]),'actual-argv','exec'),ns);return ns['argv']

@pytest.mark.parametrize('arm',['boltz','boltz_chemistry','dbmol'])
def test_declared_GPU_resource_and_real_capacity_delegation(arm):
    args=vectors()('gpu-'+arm,'pin')
    assert args[:2]==['bsub','-H'] and args[args.index('-n')+1]=='8'
    assert args[args.index('-gpu')+1]=='num=1:mode=shared:j_exclusive=yes:mps=no'
    assert args[-2:]==['--shared-capacity-source','/admitted/source'] and '-w' not in args
    assert args[args.index('-W')+1]=='02:00'

def test_unknown_phase_cannot_launch_optimization():
    with pytest.raises(ValueError):vectors()('segment','pin')

def test_predeclared_diagnostic_has_no_passing_pair_selection():
    tree=ast.parse((OWNER/'scripts/boltz_precision_driver.py').read_bytes());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and n.targets[0].id=='EPSILONS')
    assert ast.literal_eval(assignment.value)==(.01,.003,.001,.0003,.0001,.00003,.00001,.000003,.000001)
    source=ast.unparse(tree);assert "'can_admit_new_child': False" in source and 'run_chemical_updates' not in source

def test_actual_shared_gate_is_unconditionally_executed():
    source=(OWNER/'scripts/boltz_precision_ops.py').read_text()
    assert "rolling.storage(owner,parent,phase)" in source and 'owner.metadata_bytes()+CAP-budget(plan)<=2*1024**3' in source
    assert "storage_policy_adapter.install(rolling,plan,parent,plan['shared_source146'])" in source
    assert 'SHARED_PIN' in source and "plan['shared_capacity']['manifest_sha256']==SHARED_PIN" in source

def test_plain_quantitative_head_is_not_replaced_by_DBMol_objective():
    tree=ast.parse((OWNER/'scripts/boltz_objective_reference.py').read_bytes())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Evaluator')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='affinity')
    returned=next(n for n in ast.walk(fn) if isinstance(n,ast.Return))
    assert ast.unparse(returned.value)=='values[0]'
    assert 'private.DBMolScorer' in ast.unparse(cls)


def test_private_chemistry_config_uses_own_class_with_identical_fields():
    from dataclasses import dataclass
    from types import SimpleNamespace
    tree=ast.parse((OWNER/'scripts/boltz_precision_driver.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chemistry_config')
    namespace={};exec(compile(ast.Module([fn],[]),'<private-config-only>','exec'),namespace)
    @dataclass(frozen=True)
    class Private:weight:float=.1
    value=namespace['chemistry_config'](SimpleNamespace(FailureGuidanceConfig=Private),{'weight':.7})
    assert type(value) is Private and vars(value)=={'weight':.7}


def test_matrix_worker_environment_retains_exact_science_and_adapter_dependencies():
    import os
    from pathlib import Path
    tree=ast.parse((OWNER/'scripts/boltz_precision_driver.py').read_bytes())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='chemistry_worker_environment')
    namespace={'os':os,'Path':Path};exec(compile(ast.Module([fn],[]),'<actual-child-environment>','exec'),namespace)
    original={'PYTHONPATH':'existing','JAX_ENABLE_X64':'0'}
    result=namespace['chemistry_worker_environment']('/science','/adapter',original)
    assert result['PYTHONPATH'].split(os.pathsep)==[str(Path('/science')/'src'),str(Path('/adapter')/'scripts'),'existing']
    assert original=={'PYTHONPATH':'existing','JAX_ENABLE_X64':'0'} and result['JAX_ENABLE_X64']=='0'
