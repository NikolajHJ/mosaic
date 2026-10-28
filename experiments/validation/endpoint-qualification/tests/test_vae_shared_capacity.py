"""Portable VAE resource/provenance controls; no VAE/scientific imports."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('vae_shared_capacity_fixture',Path(__file__).resolve().parents[1]/'scripts/vae_shared_capacity.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


def profile():
    def p(path,digest):return {'path':str(path),'bytes':100,'sha256':digest}
    return {'version':v.VERSION,'plan':p(v.PLAN,v.PLAN_SHA),'archive':p(str(v.SOURCE)+'.tar.gz',v.ARCHIVE_SHA),
            'manifest':p(v.SOURCE/'release_manifest.json','a'*64),'quota_before':p(v.ROOT/'quota-before-001.json','b'*64),
            'baseline_bytes':100,'baseline_files':2,'growth_bytes':10*v.GIB,'growth_files':30000,
            'job_templates':{'gpu':'python worker --archive-sha256 '+v.ARCHIVE_SHA+' --manifest-sha256 '+v.MANIFEST_SHA+
                ' --plan-sha256 '+v.PLAN_SHA+' --name source104-chemistry-segment-001 --segment 1 --segment 1 --resume /original/epoch40.pt --resume-sha256 '+'a'*64,
                'callback':'python controller --plan-sha256 '+v.PLAN_SHA+' --after-stage segment-001'}}


def test_fixed_source_plan_headroom_and_unchanged_vae_identity():
    p=profile();assert v.validate_profile(p)==p


@pytest.mark.parametrize('field',['plan','archive','manifest','quota_before'])
def test_different_foreign_paths_cannot_be_substituted(field):
    p=profile();p[field]['path']='/unrelated/file'
    with pytest.raises(ValueError):v.validate_profile(p)


@pytest.mark.parametrize('field',['plan','archive'])
def test_different_scientific_release_or_plan_cannot_be_adopted(field):
    p=profile();p[field]['sha256']='c'*64
    with pytest.raises(ValueError):v.validate_profile(p)


@pytest.mark.parametrize('bad',[-1,True,1.0,float('inf'),float('nan')])
def test_finite_integer_only_forecast_inputs(bad):
    p=profile();p['baseline_bytes']=bad
    with pytest.raises(ValueError):v.validate_profile(p)


@pytest.mark.parametrize('field',['growth_bytes','growth_files'])
def test_declared_headroom_cannot_be_reduced(field):
    p=profile();p[field]-=1
    with pytest.raises(ValueError):v.validate_profile(p)


def test_original_exact_gpu_and_callback_commands_are_recognized():
    for kind,template in profile()['job_templates'].items():assert v.matches(template,template,kind)


def test_serial_successor_only_changes_owned_index_and_resume():
    p=profile();old=p['job_templates']['gpu']
    new=old.replace('segment-001','segment-002').replace('--segment 1','--segment 2').replace('/original/epoch40.pt',str(v.ROOT/'segments/002/epoch41.pt'))
    assert v.matches(new,old,'gpu')
    callback=p['job_templates']['callback']
    assert v.matches(callback.replace('segment-001','segment-002'),callback,'callback')


@pytest.mark.parametrize('change',['extra_flag','different_plan','different_lr','outside_resume','malformed_resume','mixed_index','too_many_segments'])
def test_changed_scientific_command_or_unowned_resume_is_rejected(change):
    old=profile()['job_templates']['gpu'];new=old
    if change=='extra_flag':new+=' --force'
    if change=='different_plan':new=new.replace(v.PLAN_SHA,'b'*64)
    if change=='different_lr':new+=' --lr 1'
    if change=='outside_resume':new=new.replace('/original/epoch40.pt','/other/epoch.pt')
    if change=='malformed_resume':new=new.replace('segment-001','segment-002').replace('--segment 1','--segment 2').replace('/original/epoch40.pt','/evil'+str(v.ROOT/'segments/x.pt'))
    if change=='mixed_index':new=new.replace('--segment 1 --segment 1','--segment 2 --segment 1')
    if change=='too_many_segments':new=new.replace('--segment 1','--segment 81')
    assert not v.matches(new,old,'gpu')


def test_whole_external_forecast_charged_once_and_original_reserves_preserved():
    base={'available_bytes':5000,'available_files':500,'required_bytes':1000,'required_files':100,'reserve_bytes':1000,'file_reserve':100,'status':'passed'}
    shared={'inventory':{'actual_bytes':400,'actual_files':20},'remaining_bytes':600,'remaining_files':30}
    result=v.augment(base,shared)
    assert result['byte_margin']==3000 and result['file_margin']==350
    assert result['reserve_bytes']==1000 and result['file_reserve']==100
    assert base['available_bytes']==5000


@pytest.mark.parametrize('field',['available_bytes','available_files'])
def test_external_growth_can_correctly_fail_an_otherwise_passed_gate(field):
    base={'available_bytes':5000,'available_files':500,'required_bytes':1000,'required_files':100,'status':'passed'}
    base[field]=1001 if field=='available_bytes' else 101
    result=v.augment(base,{'inventory':{'actual_bytes':400,'actual_files':20},'remaining_bytes':600,'remaining_files':30})
    assert result['status']=='failed'


def test_complete_owned_tree_inventory_without_model_work(tmp_path):
    root=tmp_path/'vae';root.mkdir();(root/'one').write_bytes(b'abc');(root/'sub').mkdir();(root/'sub/two').write_bytes(b'de')
    value=v.inventory(root);assert value['complete'] and value['actual_files']==2 and value['actual_bytes']==5


def test_other_account_writer_command_gets_no_vae_exemption():
    assert not v.matches('python unrelated_training.py',profile()['job_templates']['callback'],'callback')
