"""Allocated filesystem budget fixtures; no real quota/capacity is certified."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(_layout_path(_repository_root(__file__) / 'scripts')))
import fixed25_lrboost_storage as storage


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('High-rate storage fixtures require allocated DTU HPC')


def projections():
    return {'protenix': 2 * 1024**3, 'boltz2': 3 * 1024**3, 'boltz2_chemistry': 5 * 1024**3}


def test_all_three_new_roots_do_not_reuse_any_existing_protenix_or_boltz_output():
    assert storage.output_parent('protenix') == storage.ROOT / 'run-protenix'
    assert storage.output_parent('boltz2') == storage.ROOT / 'run-boltz2'
    assert storage.output_parent('boltz2_chemistry') == storage.HOME_ROOT / 'run-boltz2_chemistry'
    assert all(storage.TAG in str(storage.output_parent(a)) for a in storage.ARMS)


def test_filesystem_groups_include_all_fresh_remaining_native_output():
    current = {'protenix': 1024**3, 'boltz2': 0, 'boltz2_chemistry': 2 * 1024**3}
    budgets = storage.budget_inputs(projections(), current)
    assert budgets['work3']['arms'] == ['protenix', 'boltz2']
    assert budgets['work3']['outstanding_projected_bytes'] == 4 * 1024**3
    assert budgets['zhome']['arms'] == ['boltz2_chemistry']
    assert budgets['zhome']['outstanding_projected_bytes'] == 3 * 1024**3
    assert storage.RESERVE == 4 * 1024**3 and storage.FILE_RESERVE == 100000


@pytest.mark.parametrize('damage', ['missing_arm', 'old_zero_protenix', 'float_projection', 'negative_current', 'exceeded'])
def test_missing_reduced_or_malformed_new_projections_cannot_pass(damage):
    full, current = projections(), {a: 0 for a in storage.ARMS}
    if damage == 'missing_arm': full.pop('protenix')
    elif damage == 'old_zero_protenix': full['protenix'] = 0
    elif damage == 'float_projection': full['protenix'] = float(full['protenix'])
    elif damage == 'negative_current': current['boltz2'] = -1
    elif damage == 'exceeded': current['boltz2'] = full['boltz2'] + 1
    with pytest.raises(ValueError):
        storage.budget_inputs(full, current)


def test_home_physical_inode_evidence_cannot_borrow_work3_allzero_fallback():
    from fixed25_recover_storage import home_inode_headroom
    with pytest.raises(ValueError):
        home_inode_headroom({'f_files': 0, 'f_ffree': 0, 'f_favail': 0})
    value = home_inode_headroom({'f_files': 200000, 'f_ffree': 150000, 'f_favail': 120000})
    assert value['available_files_lower'] == 120000
    assert value['filesystem_inode_reporting'] == 'reported_physical_filesystem_only'


def test_stale_accounting_keeps_recent_old_report_roots_separate_from_new_outputs():
    tags = {p.name for p in storage.HISTORICAL_WORK_ROOTS}
    assert {'20261021-brd4-jq1-25-fixed-report-03', '20261021-brd4-jq1-25-fixed-report-04',
            '20261021-brd4-jq1-25-fixed-report-05'} <= tags
    assert storage.ROOT not in storage.HISTORICAL_WORK_ROOTS


def test_receipts_cannot_escape_new_campaign_root(tmp_path, monkeypatch):
    root = tmp_path / 'new'
    root.mkdir()
    monkeypatch.setattr(storage, 'ROOT', root)
    with pytest.raises(ValueError, match='within'):
        storage.receipt_dir(tmp_path / 'outside')
    assert storage.receipt_dir(root / 'gate') == root / 'gate'
    with pytest.raises(FileExistsError):
        storage.receipt_dir(root / 'gate')


def test_failed_or_partial_home_inventory_stops_before_any_quota_pass(tmp_path, monkeypatch):
    root, home = tmp_path / 'work', tmp_path / 'home'
    root.mkdir(); home.mkdir()
    monkeypatch.setattr(storage, 'ROOT', root)
    monkeypatch.setattr(storage, 'HOME_SCOPE', home)
    monkeypatch.setattr(storage, 'HOME_ROOT', home / 'new')
    monkeypatch.setattr(storage, 'recent_home_bytes', lambda: {'complete': False, 'returncode': 0, 'bytes': 0})
    with pytest.raises(ValueError, match='completed recent'):
        storage.check_campaign_storage(projections(), root / 'failed', phase='dispatch')
    assert (root / 'failed/lrboost-storage-gate.json').is_file()
