"""Allocated finite work3 quota contracts; no home-budget substitution."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

from types import SimpleNamespace

import pytest

import protenix_lr03_storage as storage


RAW = '''
s234818|306121||0 Byte|unlimited||0|unlimited
s234818|306121||90.81 GiB|100.00 GiB||312210|20000000
'''


def test_finite_quota_rounds_conservatively_and_retains_inode_limit():
    p = storage.parse_work3(RAW)
    assert p['available_bytes_lower'] == 9 * 1024**3
    assert p['available_files_lower'] == 20000000 - 312210


@pytest.mark.parametrize('raw', ['', 's234818|306121||0 Byte|unlimited||0|unlimited',
                              RAW.replace('306121', '999'), RAW.replace('90.81 GiB', '100.01 GiB'),
                              RAW.replace('312210', '20000001'), RAW.replace('90.81 GiB', '90.81 GB')])
def test_missing_unknown_or_exhausted_quota_never_passes(raw):
    with pytest.raises(ValueError): storage.parse_work3(raw)


def test_take_smallest_finite_pool_headroom():
    p = storage.parse_work3(RAW + 's234818|306121||8.1 GiB|10.9 GiB||19|1000000\n')
    assert p['available_bytes_lower'] == 1024**3
    assert p['available_files_lower'] == 999981


@pytest.mark.parametrize('case', ['pass', 'bytes', 'files'])
def test_projection_reserve_and_existing_run_bytes_all_remain_required(tmp_path, monkeypatch, case):
    root = tmp_path / 'fresh'; root.mkdir(); (root / 'existing').write_bytes(b'x' * 100)
    monkeypatch.setattr(storage, 'RUN_ROOT', root)
    raw = RAW if case != 'files' else RAW.replace('312210', '19950000')
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=raw))
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20 * 1024**3))
    projected = (6 if case == 'bytes' else 1) * 1024**3
    if case == 'pass':
        g = storage.check_storage(root / 'payload', projected)
        assert g['required_bytes'] == 5 * 1024**3
        assert g['available_bytes_conservative'] == 9 * 1024**3 - 100
        assert g['reserve_bytes'] == 4 * 1024**3
    else:
        with pytest.raises(ValueError, match='budget'): storage.check_storage(root / 'payload', projected)


def test_output_scope_rejects_existing_home_or_other_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'RUN_ROOT', tmp_path / 'declared-experiment')
    with pytest.raises(ValueError, match='scope'): storage.check_storage(tmp_path, 0)
