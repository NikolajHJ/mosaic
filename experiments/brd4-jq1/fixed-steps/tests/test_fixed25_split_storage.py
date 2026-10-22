"""Allocated all-work3 adoption contracts; no model or scientific arithmetic."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

from types import SimpleNamespace

import pytest

import fixed25_split_storage as storage

GIB = 1024**3
WORK = 's234818|306121||90.89 GiB|100.00 GiB||312765|20000000\n'
PROJECTIONS = {'protenix': 1493301824, 'boltz2': 1770121824, 'boltz2_chemistry': 5164341824}
ZERO_INODES = {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}


def inodes(available):
    return {'f_files': 1000000, 'f_ffree': available, 'f_favail': available}


def test_declared_placement_keeps_all_three_fresh_outputs_on_work3():
    assert storage.TAG == '20261021-brd4-jq1-25-fixed-storage-03'
    assert storage.PLACEMENT == dict.fromkeys(('protenix','boltz2','boltz2_chemistry'),'work3')
    for arm in storage.PLACEMENT:
        assert storage.output_parent(arm).parent == storage.ROOT
    with pytest.raises(ValueError): storage.output_parent('other')


def test_all_assigned_outputs_share_one_budget_and_only_outstanding_bytes_remain():
    budgets = storage.budget_inputs(PROJECTIONS, {'protenix': GIB, 'boltz2': 0, 'boltz2_chemistry': 0})
    assert set(budgets)=={'work3'}
    assert budgets['work3']['full_projected_bytes'] == 8427765472
    assert budgets['work3']['outstanding_projected_bytes'] == 8427765472-GIB


@pytest.mark.parametrize('bad', [{}, {'protenix': 1}, {'protenix': True, 'boltz2': 1, 'boltz2_chemistry': 1},
                                 {'protenix': -1, 'boltz2': 1, 'boltz2_chemistry': 1}])
def test_missing_arms_or_non_integer_projection_never_passes(bad):
    with pytest.raises(ValueError): storage.validate_projection(bad)


def test_declared_home_or_other_placement_is_not_a_runtime_configuration(monkeypatch):
    monkeypatch.setattr(storage,'PLACEMENT',dict(protenix='work3',boltz2='work3',boltz2_chemistry='zhome'))
    with pytest.raises(ValueError,match='work3 only'):storage.validate_projection(PROJECTIONS)
    with pytest.raises(ValueError):storage.output_parent('boltz2_chemistry')


def test_existing_output_cannot_hide_an_exceeded_full_projection():
    with pytest.raises(ValueError, match='exceeded'):
        storage.budget_inputs(dict.fromkeys(storage.PLACEMENT, 10), dict.fromkeys(storage.PLACEMENT, 11))


def test_original_full_projection_and_four_gib_reserve_never_reduce_after_cleanup():
    old = 's234818|306121||91.98 GiB|100.00 GiB||316920|20000000\n'
    recovered = 's234818|306121||86.20 GiB|100.00 GiB||310000|20000000\n'
    failed=storage.gate_record('work3',old,sum(PROJECTIONS.values()),170474164,50*GIB,inodes(200000))
    passed=storage.gate_record('work3',recovered,sum(PROJECTIONS.values()),170474164,50*GIB,inodes(200000))
    assert failed['status']=='failed' and passed['status']=='passed'
    assert failed['required_bytes']==passed['required_bytes']==12722732768
    assert passed['reserve_bytes']==4*GIB and passed['file_reserve']==100000


def test_work3_all_finite_pools_actual_disk_and_inode_headroom_are_hard_limits():
    raw = WORK + 's234818|306121||97.01 GiB|100.00 GiB||0|90000\n'
    assert storage.gate_record('work3', raw, 0, 0, 50*GIB, inodes(999999))['status'] == 'failed'
    assert storage.gate_record('work3', WORK, 0, 0, 3*GIB, inodes(999999))['status'] == 'failed'
    assert storage.gate_record('work3', WORK, 0, 0, 50*GIB, inodes(99999))['status'] == 'failed'
    with pytest.raises(ValueError): storage.gate_record('work3', 'unlimited', 0, 0, 50*GIB, inodes(999999))
    with pytest.raises(ValueError): storage.gate_record('zhome', WORK, 0, 0, 50*GIB, inodes(999999))


def test_all_zero_filesystem_inode_tuple_is_unreported_while_quota_remains_mandatory():
    gate=storage.gate_record('work3',WORK,0,0,50*GIB,ZERO_INODES)
    assert gate['status']=='passed'
    assert gate['filesystem_inode_reporting']=='unreported_all_zero'
    assert gate['filesystem_inode_observation']==ZERO_INODES
    assert gate['available_files_lower']==20000000-312765
    assert gate['file_reserve']==100000
    assert 'no independent filesystem inode proof' in gate['inode_limitation']


def test_unreported_filesystem_never_waives_finite_user_quota_or_headroom():
    small=WORK.replace('312765|20000000','19900001|20000000')
    gate=storage.gate_record('work3',small,0,0,50*GIB,ZERO_INODES)
    assert gate['status']=='failed' and gate['available_files_lower']==99999
    exhausted=WORK.replace('312765|20000000','20000000|20000000')
    with pytest.raises(ValueError):storage.gate_record('work3',exhausted,0,0,50*GIB,ZERO_INODES)
    with pytest.raises(ValueError):storage.gate_record('work3','unlimited',0,0,50*GIB,ZERO_INODES)


@pytest.mark.parametrize('bad', [
    {'f_files':0,'f_ffree':1,'f_favail':0},
    {'f_files':10,'f_ffree':11,'f_favail':1},
    {'f_files':10,'f_ffree':5,'f_favail':6},
    {'f_files':10,'f_ffree':-1,'f_favail':0},
    {'f_files':True,'f_ffree':0,'f_favail':0},
    {'f_files':10.0,'f_ffree':5,'f_favail':1},
    {'f_files':0,'f_ffree':0},
])
def test_inconsistent_unknown_negative_bool_or_noninteger_inode_tuple_rejects(bad):
    with pytest.raises(ValueError):storage.gate_record('work3',WORK,0,0,50*GIB,bad)


def test_positive_reported_filesystem_exhaustion_is_not_unreported():
    gate=storage.gate_record('work3',WORK,0,0,50*GIB,inodes(0))
    assert gate['status']=='failed' and gate['available_files_lower']==0
    assert gate['filesystem_inode_reporting']=='reported_coherent'


def test_complete_current_roots_and_new_release_are_subtracted_again(tmp_path,monkeypatch):
    roots=[tmp_path/'old',tmp_path/'new',tmp_path/'release']
    for index,path in enumerate(roots):
        path.mkdir();(path/'bytes').write_bytes(b'x'*(index+1))
    monkeypatch.setattr(storage,'SCIENCE_ROOT',roots[0]);monkeypatch.setattr(storage,'ROOT',roots[1])
    monkeypatch.setattr(storage,'WORK_RELEASE',roots[2])
    upload=tmp_path/'new-upload.tar.gz';upload.write_bytes(b'xxxx')
    monkeypatch.setattr(storage,'WORK_UPLOAD',upload)
    assert storage.stale_work_bytes()==10


def test_internal_pytest_current_links_charge_metadata_once_without_following(tmp_path):
    root=tmp_path/'tree';real=root/'fixture0';real.mkdir(parents=True)
    payload=real/'bytes';payload.write_bytes(b'abcdef')
    directory_link=root/'fixturecurrent';directory_link.symlink_to(real,target_is_directory=True)
    file_link=root/'filecurrent';file_link.symlink_to(payload)
    assert storage.tree_bytes(root)==6+directory_link.lstat().st_size+file_link.lstat().st_size


@pytest.mark.parametrize('kind',['external','dangling','root_alias','parent_alias','special'])
def test_external_unresolved_root_parent_aliases_and_special_types_reject(tmp_path,kind):
    import os
    root=tmp_path/'tree';root.mkdir();(root/'bytes').write_bytes(b'x')
    outside=tmp_path/'outside';outside.mkdir()
    if kind=='external':
        (root/'current').symlink_to(outside,target_is_directory=True)
    elif kind=='dangling':
        (root/'current').symlink_to(root/'missing')
    elif kind=='root_alias':
        alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True);root=alias
    elif kind=='parent_alias':
        alias=tmp_path/'parent-alias';alias.symlink_to(tmp_path,target_is_directory=True);root=alias/'tree'
    else:
        os.mkfifo(root/'fifo')
    with pytest.raises(ValueError):storage.tree_bytes(root)


def test_cpu_requires_one_gib_plus_reserve_and_preserves_failed_receipt(tmp_path, monkeypatch):
    root = tmp_path / 'new'; root.mkdir()
    monkeypatch.setattr(storage, 'ROOT', root)
    monkeypatch.setattr(storage, 'stale_work_bytes', lambda: 0)
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=WORK))
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=20*GIB))
    monkeypatch.setattr(storage.os, 'statvfs', lambda p: SimpleNamespace(f_files=300000,f_ffree=200000,f_favail=200000), raising=False)
    good = storage.check_cpu_storage(root, root/'cpu-gate')
    assert good['required_bytes'] == 5*GIB
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=4*GIB))
    with pytest.raises(ValueError, match='CPU allowance'): storage.check_cpu_storage(root, root/'bad')
    assert (root/'bad/split-storage-gate.json').is_file()


def test_cpu_preserves_inconsistent_raw_inode_metadata_in_a_failed_receipt(tmp_path,monkeypatch):
    root=tmp_path/'new';root.mkdir();monkeypatch.setattr(storage,'ROOT',root)
    monkeypatch.setattr(storage,'stale_work_bytes',lambda:0)
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=WORK))
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=20*GIB))
    monkeypatch.setattr(storage.os,'statvfs',lambda p:SimpleNamespace(f_files=0,f_ffree=1,f_favail=0),raising=False)
    with pytest.raises(ValueError,match='Inconsistent'):storage.check_cpu_storage(root,root/'bad-inodes')
    import json
    receipt=json.loads((root/'bad-inodes/split-storage-gate.json').read_text())
    assert receipt['status']=='failed' and receipt['filesystem_inode_reporting']=='not_validated'
    assert receipt['filesystem_inode_observation']=={'f_files':0,'f_ffree':1,'f_favail':0}
    assert receipt['inode_policy']==storage.INODE_POLICY


def test_entry_checks_whole_campaign_once_and_never_queries_or_creates_home(tmp_path, monkeypatch):
    root = tmp_path/'work'; root.mkdir()
    monkeypatch.setattr(storage, 'ROOT', root)
    monkeypatch.setattr(storage, 'stale_work_bytes', lambda: 0)
    calls = []
    def run(command, **kwargs):
        assert command==['getquota_work3.sh'];calls.append(command[0]);return SimpleNamespace(stdout=WORK)
    monkeypatch.setattr(storage.subprocess, 'run', run)
    monkeypatch.setattr(storage.shutil, 'disk_usage', lambda p: SimpleNamespace(free=3*GIB))
    monkeypatch.setattr(storage.os, 'statvfs', lambda p: SimpleNamespace(f_files=300000,f_ffree=200000,f_favail=200000), raising=False)
    with pytest.raises(ValueError, match='whole-campaign'):
        storage.check_campaign_storage(PROJECTIONS, root/'gate', phase='entry')
    assert calls == ['getquota_work3.sh']
    import json
    receipt=json.loads((root/'gate/split-storage-gate.json').read_text())
    assert receipt['status']=='failed' and set(receipt['filesystems'])=={'work3'}
    assert receipt['filesystems']['work3']['full_projected_bytes']==sum(PROJECTIONS.values())
    assert list(tmp_path.iterdir())==[root]


def test_receipt_scope_and_projection_scope_rejected_before_subprocess(tmp_path, monkeypatch):
    # Allocated pytest basetemp itself lives inside the production campaign
    # root, so isolate the declared test root before choosing an outside path.
    monkeypatch.setattr(storage, 'ROOT', tmp_path/'declared')
    monkeypatch.setattr(storage.subprocess, 'run', lambda *a, **k: pytest.fail('No external call'))
    with pytest.raises(ValueError, match='scope'):
        storage.check_campaign_storage(PROJECTIONS, tmp_path/'outside', phase='dispatch')

