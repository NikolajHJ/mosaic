"""Small metadata controls for shared exploratory report storage accounting."""
from pathlib import Path
import sys

import pytest

OWNER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OWNER / 'scripts'))
# Local checkout and the flat allocated engine both use the same shared helper.
shared = OWNER.parents[2] / 'scripts/hpc'
if shared.exists(): sys.path.insert(0, str(shared))
import quick_runtime as runtime

GIB = 1024**3
INODES = {'f_files': 1000000, 'f_ffree': 900000, 'f_favail': 800000}


def test_live_quota_does_not_recharge_old_report_tree():
    result = runtime.home_exploratory_capacity(
        'You are using 25.00G of 30.00G quota', 100*GIB, INODES, GIB, 100, 4096, 10)
    assert result['status'] == 'passed' and result['future_bytes'] == GIB-100
    assert result['existing_bytes_recharged'] == 0 and result['reserve_bytes'] < GIB
    assert result['available_bytes'] > 4*GIB
    assert result['work3_output_bytes_requested'] == 0
    assert result['scientific_campaign_admission_granted'] is False


@pytest.mark.parametrize('quota,disk,inodes', [
    ('You are using 29.50G of 30.00G quota', 100*GIB, INODES),
    ('You are using 10.00G of 30.00G quota', GIB//2, INODES),
    ('You are using 10.00G of 30.00G quota', 100*GIB,
     {'f_files': 10, 'f_ffree': 5, 'f_favail': 5})])
def test_real_byte_and_file_shortfalls_still_fail(quota, disk, inodes):
    assert runtime.home_exploratory_capacity(quota, disk, inodes, GIB, 0, 4096, 0)['status'] == 'failed'


def test_unknown_quota_is_reported_and_uses_measured_space():
    result = runtime.home_exploratory_capacity('', 2*GIB,
        {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}, GIB, 0, 4096, 0)
    assert result['status'] == 'passed' and result['quota_observation'] == 'unavailable'
    assert result['warnings'] and result['available_files'] is None


def test_exceeded_own_cap_is_not_hidden_as_zero_remaining():
    with pytest.raises(ValueError):
        runtime.home_exploratory_capacity('', 100*GIB, INODES, GIB, GIB+1, 4096, 0)
