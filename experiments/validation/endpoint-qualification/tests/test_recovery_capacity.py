"""Allocated metadata contracts for retained recovery forecasts/reserves."""
import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location('recovery_capacity_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/recovery_capacity.py')
r = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(r)


def prior():
    return {'status': 'passed', 'original_gate_changed': False, 'reserve_bytes': 4 * 1024**3,
            'file_reserve': 100000, 'required_bytes': 10 * 1024**3, 'available_bytes': 20 * 1024**3}


def test_recovery_adds_both_entire_caps_and_stale_bytes():
    result = r.validate(prior(), 123, 456)
    assert result['required_bytes'] == prior()['required_bytes'] + (272 + 512 + 272 + 512) * 1024**2 + 456
    assert result['available_bytes'] == prior()['available_bytes'] - 123
    assert result['forecasts_removed'] is False


@pytest.mark.parametrize('key,value', [('status', 'failed'), ('original_gate_changed', True),
    ('reserve_bytes', 0), ('file_reserve', 99999), ('required_bytes', float('inf'))])
def test_recovery_rejects_missing_full_guard_or_changed_reserve(key, value):
    p = prior(); p[key] = value
    with pytest.raises(ValueError): r.validate(p, 0)


def test_recovery_rejects_capacity_shortfall():
    p = prior(); p['available_bytes'] = p['required_bytes'] + r.EXTRA - 1
    with pytest.raises(ValueError, match='do not fit'): r.validate(p, 0)


@pytest.mark.parametrize('charged,excess', [(-1, 0), (0, -1), (False, 0)])
def test_recovery_rejects_partial_or_invalid_totals(charged, excess):
    with pytest.raises(ValueError): r.validate(prior(), charged, excess)


def test_all_old_and_new_recovery_roots_and_future_caps_remain_charged():
    assert r.EXTRA == (272 + 512 + 272 + 512) * 1024**2
    assert len(r.CHARGED) == len(set(r.CHARGED))
    assert all(name in r.CHARGED for name in (
        'protenix-endpoint-fix-07', 'protenix-endpoint-fix-release-07',
        'protenix-endpoint-fix-08', 'protenix-endpoint-fix-release-08',
        'protenix-endpoint-fix-09', 'protenix-endpoint-fix-release-09',
        'dbmol-endpoint-recovery-10', 'dbmol-endpoint-recovery-release-10',
        'dbmol-endpoint-recovery-11', 'dbmol-endpoint-recovery-release-11',
        'dbmol-endpoint-recovery-12', 'dbmol-endpoint-recovery-release-12',
        'dbmol-endpoint-recovery-13', 'dbmol-endpoint-recovery-release-13'))
