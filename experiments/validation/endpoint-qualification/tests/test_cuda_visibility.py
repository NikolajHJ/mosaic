"""Portable scheduler metadata only; native CUDA co-residency remains mandatory."""
import importlib.util
from pathlib import Path
import pytest

_spec=importlib.util.spec_from_file_location('visibility_metadata',Path(__file__).resolve().parents[1]/'scripts/cuda_ownership.py')
c=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(c)


def visibility_environment(slots=4,direct=None):
    result={'LSB_DJOB_NUMPROC':str(slots),**{'CUDA_VISIBLE_DEVICES'+str(i):'GPU-observed-123' for i in range(1,slots+1)}}
    if direct is not None:result['CUDA_VISIBLE_DEVICES']=direct
    return result


@pytest.mark.parametrize('slots,direct',[(4,None),(4,''),(4,'GPU-observed-123'),(8,None),(8,''),(8,'GPU-observed-123')])
def test_complete_one_based_real_allocation_form(slots,direct):
    env=visibility_environment(slots,direct);before=dict(env);value=c.resolve_visibility(env)
    assert value['selected_cuda_visible_devices']=='GPU-observed-123'
    assert value['origin']=='LSF_MPS_complete_one_based_unanimous_mapping' and value['guessed_device'] is False
    assert env==before and value['task_mappings']=={k:v for k,v in env.items() if k.startswith('CUDA_VISIBLE_DEVICES') and k!='CUDA_VISIBLE_DEVICES'}


@pytest.mark.parametrize('change',['missing','extra','conflict','multiple','negative','slots-missing','slots-zero','direct-conflict'])
def test_one_based_partial_or_conflicting_maps_still_block(change):
    env=visibility_environment()
    if change=='missing':env.pop('CUDA_VISIBLE_DEVICES4')
    elif change=='extra':env['CUDA_VISIBLE_DEVICES5']='GPU-observed-123'
    elif change=='conflict':env['CUDA_VISIBLE_DEVICES3']='GPU-other'
    elif change=='multiple':env['CUDA_VISIBLE_DEVICES2']='0,1'
    elif change=='negative':env['CUDA_VISIBLE_DEVICES2']='-1'
    elif change=='slots-missing':env.pop('LSB_DJOB_NUMPROC')
    elif change=='slots-zero':env['LSB_DJOB_NUMPROC']='0'
    else:env['CUDA_VISIBLE_DEVICES']='GPU-other'
    with pytest.raises(ValueError):c.resolve_visibility(env)
