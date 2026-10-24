"""AllocatedGPU-only gate; never skip or label CPU fallback as GPU success."""
import json
import os
from pathlib import Path
import sys

import jax
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gpu_matrix import require_backend
from round2_data import cases
from round2_gpu import check_case,check_features


@pytest.fixture(scope='module')
def saved_cases():
    require_backend('gpu')
    return cases(json.loads(Path(os.environ['MOSAIC_SPEED_PLAN']).read_text()))


def test_real_gpu_allocation():
    assert os.environ.get('LSB_QUEUE')=='gpul40s'
    require_backend('gpu')
    assert len(jax.devices('gpu'))==1
    assert 'L40S' in jax.devices('gpu')[0].device_kind


@pytest.mark.parametrize('index',range(6))
def test_exact_gpu_integer_features(saved_cases,index):
    assert check_features(saved_cases[index])['status']=='passed_exact_integer_cpu_gpu_features'


@pytest.mark.parametrize('method',['scan','matrix'])
@pytest.mark.parametrize('index',range(6))
def test_real_gpu_loss_gradients_and_original_fd_ladder(saved_cases,index,method):
    assert check_case(saved_cases[index],method)['status']=='passed'
