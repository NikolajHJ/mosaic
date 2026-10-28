"""Portable exact-command/profile checks; no rendering or molecular work."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('home_reader_fixture',Path(__file__).resolve().parents[1]/'scripts/home_report_reader.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)


def command(phase='render'):
    request=h.HOME/'requests/request-test/request.json' if phase=='render' else h.SOURCE.parent/'validation/request.json'
    return ' '.join([h.PYTHON,'-B','-u',str(h.SOURCE/'scripts/quick_runtime.py'),phase,'--source',str(h.SOURCE),'--pin',h.PIN,'--request',str(request)])


@pytest.mark.parametrize('phase',['render','validate'])
def test_exact_home_command_is_recognized_without_work3_write_admission(phase):
    assert h.command_request(command(phase)) is not None


@pytest.mark.parametrize('bad',['extra','pin','interpreter','phase','request','source'])
def test_modified_or_different_output_commands_get_no_exception(bad):
    c=command()
    if bad=='extra':c+=' --force'
    if bad=='pin':c=c.replace(h.PIN,'a'*64)
    if bad=='interpreter':c=c.replace(h.PYTHON,'/other/python')
    if bad=='phase':c=c.replace(' render ',' run ')
    if bad=='request':c=c.replace(str(h.HOME/'requests/request-test/request.json'),'/work3/s234818/mosaic/out/request.json')
    if bad=='source':c=c.replace(str(h.SOURCE),' /work3/other/source ')
    assert h.command_request(c) is None


def profile():return {'output_filesystem':'home','remote_root':str(h.HOME),'home_quota_root':str(h.HOME.parent),
                      'python':h.PYTHON,'additional_owned_paths':[],'request_cap_bytes':1024**3,'validation_cap_bytes':64*1024**2}


def test_exact_home_output_profile_accepts_only_qualified_scopes():h.profile_home_only(profile())


@pytest.mark.parametrize('key,value',[('output_filesystem','work3'),('remote_root','/work3/s234818/mosaic/output'),
    ('home_quota_root','/work3/s234818'),('python','/other/python'),('request_cap_bytes',True),
    ('request_cap_bytes',2*1024**3),('validation_cap_bytes',-1),('additional_owned_paths',['/work3/s234818/mosaic/cache'])])
def test_invalid_profile_cannot_turn_a_writer_into_a_reader(key,value):
    p=profile();p[key]=value
    with pytest.raises(ValueError):h.profile_home_only(p)


def test_unrelated_jobs_get_no_readonly_exemption():
    assert h.observe([{'COMMAND':'python train.py','JOBID':'1','QUEUE':'hpc'}])['work3_read_only_job_ids']==[]
