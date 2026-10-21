"""Allocated recovery storage fixtures; all scientific runs and old sources stay unchanged."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

import fixed25_recover_storage as storage

GIB=1024**3
WORK='s234818|306121||86.20 GiB|100.00 GiB||300000|20000000\n'
HOME='You are using 18.62G of 30.00G quota\n(Information updated ca. every 240 minutes. Last Update: 2026-10-21 19:03:58)\n'
NOW=datetime(2026,10,21,20,0,tzinfo=ZoneInfo('Europe/Copenhagen')).timestamp()
WORK_INODES={'f_files':0,'f_ffree':0,'f_favail':0}
HOME_INODES={'f_files':114162648124,'f_ffree':13520341897,'f_favail':13520341897}


def test_existing_protenix_has_one_existing_parent_and_no_new_submission_root():
    assert storage.TAG=='20261021-brd4-jq1-25-fixed-storage-04'
    assert storage.output_parent('protenix')==storage.ACTIVE_ROOT/'run-protenix'
    assert storage.new_output_parent('boltz2')==storage.ROOT/'run-boltz2'
    assert storage.new_output_parent('boltz2_chemistry')==storage.HOME_ROOT/'run-boltz2_chemistry'
    assert storage.NEW_ARMS==('boltz2','boltz2_chemistry')
    with pytest.raises(ValueError,match='never'):storage.new_output_parent('protenix')
    with pytest.raises(ValueError):storage.output_parent('other')


def test_per_filesystem_budget_includes_existing_protenix_remaining_once():
    current={'protenix':100,'boltz2':20,'boltz2_chemistry':30}
    budget=storage.budget_inputs(storage.FULL_PROJECTIONS,current)
    assert budget['work3']['arms']==['protenix','boltz2']
    assert budget['work3']['full_projected_bytes']==1493301824+1770121824
    assert budget['work3']['outstanding_projected_bytes']==1493301824+1770121824-120
    assert budget['zhome']['arms']==['boltz2_chemistry']
    assert budget['zhome']['outstanding_projected_bytes']==5164341824-30


@pytest.mark.parametrize('mutation',['missing','lower','bool','exceeded','wrong_placement'])
def test_all_original_projections_and_placement_are_immutable(mutation,monkeypatch):
    projection=dict(storage.FULL_PROJECTIONS);current=dict.fromkeys(projection,0)
    if mutation=='missing':del projection['protenix']
    elif mutation=='lower':projection['boltz2']-=1
    elif mutation=='bool':projection['protenix']=True
    elif mutation=='exceeded':current['protenix']=projection['protenix']+1
    else:monkeypatch.setitem(storage.PLACEMENT,'boltz2_chemistry','work3')
    with pytest.raises(ValueError):storage.budget_inputs(projection,current)


def test_home_quota_decimal_rounding_and_explicit_cache_identity():
    quota=storage.parse_home(HOME,now=NOW)
    assert quota['used_bytes_upper']==19*GIB and quota['hard_bytes_lower']==30*GIB
    assert quota['available_bytes_lower']==11*GIB
    assert quota['reported_user_inode_quota'] is False
    assert quota['reported_snapshot'].endswith('+02:00')
    assert quota['maximum_cache_age_seconds']==14400


@pytest.mark.parametrize('raw,now',[(HOME,NOW+5*3600),(HOME,NOW-2*3600),
    ('You are using 18.62G of 30.00G quota',NOW),(HOME+HOME,NOW),
    (HOME.replace('18.62','NaN'),NOW),(HOME.replace('18.62','30.01'),NOW)])
def test_missing_ambiguous_future_stale_or_invalid_home_quota_rejects(raw,now):
    with pytest.raises(ValueError):storage.parse_home(raw,now=now)


@pytest.mark.parametrize('observed',[
    {'f_files':0,'f_ffree':0,'f_favail':0},
    {'f_files':0,'f_ffree':1,'f_favail':0},
    {'f_files':10,'f_ffree':11,'f_favail':1},
    {'f_files':10,'f_ffree':5,'f_favail':6},
    {'f_files':True,'f_ffree':0,'f_favail':0},
    {'f_files':10.0,'f_ffree':5,'f_favail':1},
    {'f_files':10,'f_ffree':0},
])
def test_home_cannot_borrow_work3_quota_or_accept_unknown_physical_inode_totals(observed):
    with pytest.raises(ValueError):storage.gate_record('zhome',HOME,0,0,50*GIB,observed,now=NOW)


def test_physical_home_inode_scope_is_explicit_and_still_needs_100000():
    gate=storage.gate_record('zhome',HOME,0,0,50*GIB,HOME_INODES,now=NOW)
    assert gate['status']=='passed'
    assert gate['filesystem_inode_reporting']=='reported_physical_filesystem_only'
    assert 'no per-user inode quota' in gate['file_headroom_scope']
    small=dict(HOME_INODES,f_favail=99999)
    assert storage.gate_record('zhome',HOME,0,0,50*GIB,small,now=NOW)['status']=='failed'
    zero=dict(HOME_INODES,f_favail=0)
    assert storage.gate_record('zhome',HOME,0,0,50*GIB,zero,now=NOW)['status']=='failed'


def test_four_gib_is_reserved_separately_on_each_filesystem():
    w=storage.gate_record('work3',WORK,3263423648,0,50*GIB,WORK_INODES)
    h=storage.gate_record('zhome',HOME,5164341824,0,50*GIB,HOME_INODES,now=NOW)
    assert w['required_bytes']==3263423648+4*GIB
    assert h['required_bytes']==5164341824+4*GIB
    assert w['status']==h['status']=='passed'
    assert w['file_reserve']==h['file_reserve']==100000
    assert storage.gate_record('zhome',HOME,5164341824,3*GIB,50*GIB,HOME_INODES,now=NOW)['status']=='failed'
    scarce=WORK.replace('300000|20000000','19900001|20000000')
    assert storage.gate_record('work3',scarce,0,0,50*GIB,WORK_INODES)['status']=='failed'


@pytest.mark.parametrize('reply',[SimpleNamespace(returncode=1,stdout='100\n',stderr='partial error'),
    SimpleNamespace(returncode=0,stdout='100',stderr=''),SimpleNamespace(returncode=0,stdout='-1\n',stderr=''),
    SimpleNamespace(returncode=0,stdout='unknown\n',stderr='')])
def test_recent_home_find_requires_complete_successful_integer_byte_inventory(tmp_path,monkeypatch,reply):
    monkeypatch.setattr(storage,'HOME_SCOPE',tmp_path)
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:reply)
    with pytest.raises(ValueError):storage.recent_home_bytes()


def test_recent_home_find_is_read_only_no_follow_bounded_and_counts_file_link_metadata(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'HOME_SCOPE',tmp_path);observed=[]
    def run(command,**kwargs):
        observed.append((command,kwargs));return SimpleNamespace(returncode=0,stdout='100\n7\n',stderr='')
    monkeypatch.setattr(storage.subprocess,'run',run)
    result=storage.recent_home_bytes();command,kwargs=observed[0]
    assert command[:3]==['find','-P',str(tmp_path)]
    assert '-mmin' in command and '-240' in command and '-type' in command and '-delete' not in command
    assert kwargs['timeout']==600 and result['bytes']==107 and result['files_or_links']==2
    assert result['returncode']==0 and result['complete'] is True and len(result['stdout_sha256'])==64


def test_recent_home_find_timeout_never_returns_partial_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(storage,'HOME_SCOPE',tmp_path)
    def run(*a,**k):raise TimeoutError('incomplete inventory')
    monkeypatch.setattr(storage.subprocess,'run',run)
    with pytest.raises(TimeoutError):storage.recent_home_bytes()


def test_all_original_active_new_release_and_upload_work3_bytes_remain_charged(tmp_path,monkeypatch):
    names=('SCIENCE_ROOT','ACTIVE_ROOT','ROOT','WORK_RELEASE','WORK_UPLOAD')
    for index,name in enumerate(names):
        path=tmp_path/name
        if name=='WORK_UPLOAD':path.write_bytes(b'x'*(index+1))
        else:path.mkdir();(path/'bytes').write_bytes(b'x'*(index+1))
        monkeypatch.setattr(storage,name,path)
    assert storage.stale_work_bytes()==15


@pytest.fixture
def prepared_paths(tmp_path,monkeypatch):
    root=tmp_path/'work';root.mkdir()
    home=tmp_path/'mosaic-home';home.mkdir()
    home_root=home/storage.TAG
    active=tmp_path/'storage03';(active/'run-protenix').mkdir(parents=True)
    (active/'run-protenix/progress').write_bytes(b'old-prefix')
    monkeypatch.setattr(storage,'ROOT',root);monkeypatch.setattr(storage,'HOME_SCOPE',home)
    monkeypatch.setattr(storage,'HOME_ROOT',home_root);monkeypatch.setattr(storage,'ACTIVE_ROOT',active)
    monkeypatch.setattr(storage,'SCIENCE_ROOT',tmp_path/'science02')
    monkeypatch.setattr(storage,'stale_work_bytes',lambda:GIB)
    monkeypatch.setattr(storage.time,'time',lambda:NOW)
    monkeypatch.setattr(storage,'recent_home_bytes',lambda:{'bytes':GIB,'complete':True,'returncode':0})
    monkeypatch.setattr(storage.os,'statvfs',lambda path:SimpleNamespace(**(WORK_INODES if path==root else HOME_INODES)),raising=False)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=50*GIB))
    calls=[]
    def run(command,**kwargs):
        calls.append(command[0]);return SimpleNamespace(stdout=WORK if command[0]=='getquota_work3.sh' else HOME)
    monkeypatch.setattr(storage.subprocess,'run',run)
    return root,home_root,active,calls


def test_cpu_preflight_queries_work3_only_and_never_scans_or_creates_home(prepared_paths,monkeypatch):
    root,home,active,calls=prepared_paths
    monkeypatch.setattr(storage,'recent_home_bytes',lambda:pytest.fail('No home scan for CPU'))
    gate=storage.check_cpu_storage(root,root/'cpu-gate')
    assert gate['status']=='passed' and gate['filesystems']['work3']['required_bytes']==5*GIB
    assert calls==['getquota_work3.sh'] and not home.exists()
    assert (active/'run-protenix/progress').read_bytes()==b'old-prefix'


def test_grouped_gate_preserves_existing_protenix_and_home_uncreated_until_dispatch(prepared_paths):
    root,home,active,calls=prepared_paths
    historical=active/'entry-quota-boltz2';historical.mkdir()
    failure=historical/'split-storage-gate.json';failure.write_bytes(b'{"status":"failed"}')
    gate=storage.check_campaign_storage(storage.FULL_PROJECTIONS,root/'dispatch-gate',phase='dispatch')
    assert gate['status']=='passed' and set(gate['filesystems'])=={'work3','zhome'}
    assert gate['filesystems']['work3']['arms']==['protenix','boltz2']
    assert gate['existing_protenix_parent']==str(active/'run-protenix')
    assert gate['existing_protenix_job']=='29777709' and not home.exists()
    assert not (root/'run-protenix').exists()
    assert (active/'run-protenix/progress').read_bytes()==b'old-prefix'
    assert failure.read_bytes()==b'{"status":"failed"}'
    assert calls==['getquota_work3.sh','getquota_zhome.sh']


def test_complete_new_home_root_is_charged_separately_from_recent_find(prepared_paths):
    root,home,active,calls=prepared_paths
    (home/'run-boltz2_chemistry').mkdir(parents=True)
    (home/'run-boltz2_chemistry/bytes').write_bytes(b'new-home')
    gate=storage.check_campaign_storage(storage.FULL_PROJECTIONS,root/'entry-gate',phase='entry')
    assert gate['complete_home_root_bytes_subtracted_again']==8
    assert gate['filesystems']['zhome']['stale_bytes_subtracted']==GIB+8
    assert gate['filesystems']['zhome']['outstanding_projected_bytes']==5164341824-8


def test_either_filesystem_failure_preserves_failed_gate_after_both_are_checked(prepared_paths,monkeypatch):
    root,home,active,calls=prepared_paths
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda path:SimpleNamespace(free=GIB if path==root else 50*GIB))
    with pytest.raises(ValueError,match='grouped'):storage.check_campaign_storage(storage.FULL_PROJECTIONS,root/'failed',phase='entry')
    import json
    record=json.loads((root/'failed/recovery-storage-gate.json').read_text())
    assert record['status']=='failed' and set(record['filesystems'])=={'work3','zhome'}
    assert record['filesystems']['work3']['status']=='failed' and record['filesystems']['zhome']['status']=='passed'
    assert calls==['getquota_work3.sh','getquota_zhome.sh']
    assert not home.exists()


def test_incomplete_recent_inventory_blocks_without_quotas_or_home_creation(prepared_paths,monkeypatch):
    root,home,active,calls=prepared_paths
    monkeypatch.setattr(storage,'recent_home_bytes',lambda:{'bytes':1,'complete':False,'returncode':0})
    with pytest.raises(ValueError,match='Incomplete'):storage.check_campaign_storage(storage.FULL_PROJECTIONS,root/'bad-find',phase='entry')
    assert not calls and not home.exists()


def test_missing_home_parent_is_not_created_by_a_storage_gate(prepared_paths,monkeypatch):
    root,home,active,calls=prepared_paths
    missing=root.parent/'missing-home-parent'
    monkeypatch.setattr(storage,'HOME_SCOPE',missing)
    monkeypatch.setattr(storage,'HOME_ROOT',missing/storage.TAG)
    with pytest.raises(ValueError,match='Existing Mosaic home parent'):
        storage.check_campaign_storage(storage.FULL_PROJECTIONS,root/'missing-home',phase='dispatch')
    assert not missing.exists() and not calls


def test_receipt_alias_and_outside_scope_reject_before_subprocess(tmp_path,monkeypatch):
    root=tmp_path/'declared';root.mkdir()
    monkeypatch.setattr(storage,'ROOT',root)
    monkeypatch.setattr(storage.subprocess,'run',lambda *a,**k:pytest.fail('No subprocess'))
    with pytest.raises(ValueError,match='escapes'):storage.check_cpu_storage(root,tmp_path/'outside')
    (root/'real').mkdir();(root/'alias').symlink_to(root/'real',target_is_directory=True)
    with pytest.raises(ValueError,match='alias'):storage.check_cpu_storage(root,root/'alias/new')
