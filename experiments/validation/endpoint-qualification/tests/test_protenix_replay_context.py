"""Allocated metadata controls for the inherited compiler-context admission."""
import copy
import importlib.util
import os
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('endpoint_context_fixture',Path(__file__).resolve().parents[1]/'scripts/protenix_replay_context.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

@pytest.fixture(autouse=True)
def allocated():
    assert os.environ.get('LSB_JOBID'),'Fixtures require allocated DTU HPC'

def value():
    return {'qualification':{'version':'isolated-replay-context-qualification-v1',
        'selected_policy':'autotune0','selected_environment':{'XLA_FLAGS':'--xla_gpu_autotune_level=0'},
        'native_gradient_passed':False,'original_scientific_gate_reclassified':False,
        'saved_probability_substitution':False,'can_certify_or_restart_original_study':False,
        'fresh_full_model_validation_required':True},'original_failure_reclassified':False,
        'native_gradient_passed':False,'proofs':[{'path':'/original/proof','bytes':1,'sha256':'a'*64}]}

@pytest.mark.parametrize('key',['version','selected_policy','selected_environment','native_gradient_passed',
    'original_scientific_gate_reclassified','saved_probability_substitution',
    'can_certify_or_restart_original_study','fresh_full_model_validation_required'])
def test_wrong_context_or_overclaimed_acceptance_rejected(key):
    v=value();v['qualification'][key]='wrong'
    with pytest.raises(ValueError):c.validate(v)

@pytest.mark.parametrize('key',['original_failure_reclassified','native_gradient_passed','proofs'])
def test_missing_proofs_or_top_level_recertification_rejected(key):
    v=value();v[key]=True if key!='proofs' else []
    with pytest.raises(ValueError):c.validate(v)

@pytest.mark.parametrize('flags',['--xla_gpu_autotune_level=4','--xla_gpu_autotune_level=0 --xla_gpu_autotune_level=0',
    '--xla_gpu_autotune_level'])
def test_ambiguous_inherited_flag_is_not_overridden(flags):
    with pytest.raises(ValueError):c.environment({'XLA_FLAGS':flags},value())

def test_only_compiler_flag_added_without_mutating_inputs_or_scientific_settings():
    before={'JAX_ENABLE_X64':'0','UNCHANGED':'setting'};v=value();saved=copy.deepcopy(v)
    actual=c.environment(before,v)
    assert actual=={**before,'XLA_FLAGS':'--xla_gpu_autotune_level=0'}
    assert before=={'JAX_ENABLE_X64':'0','UNCHANGED':'setting'} and v==saved
    assert c.environment(actual,v)==actual
