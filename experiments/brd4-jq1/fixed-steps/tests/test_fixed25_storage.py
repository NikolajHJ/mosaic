"""Allocated storage scope checks for the three-arm restart."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

from types import SimpleNamespace
import pytest
import fixed25_storage as storage

RAW = 's234818|306121||90.89 GiB|100.00 GiB||312765|20000000\n'


def test_three_arm_projection_and_four_gib_reserve_are_both_required(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'RUN_ROOT', tmp_path / 'declared')
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=RAW))
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20*1024**3))
    gate = storage.check_storage(storage.RUN_ROOT / 'payload', 3*1024**3)
    assert gate['required_bytes'] == 7*1024**3 and gate['status'] == 'passed'
    with pytest.raises(ValueError, match='budget'): storage.check_storage(storage.RUN_ROOT / 'other', 7*1024**3)


def test_scope_fixture_explicitly_uses_an_outside_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'RUN_ROOT', tmp_path / 'declared')
    with pytest.raises(ValueError, match='scope'): storage.check_storage(tmp_path / 'outside', 0)
