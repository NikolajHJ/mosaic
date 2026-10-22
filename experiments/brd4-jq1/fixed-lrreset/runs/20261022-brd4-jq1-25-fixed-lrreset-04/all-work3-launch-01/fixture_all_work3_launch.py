"""Allocated CPU metadata contracts; no models, molecular arrays or optimizer work."""
import subprocess
from pathlib import Path
import pytest
import all_work3_launch as a


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    root = tmp_path / 'run'; root.mkdir()
    control = root / 'control'; control.mkdir()
    monkeypatch.setattr(a, 'ROOT', root); monkeypatch.setattr(a, 'CONTROL', control)
    monkeypatch.setattr(a.science.storage, 'HOME_ROOT', tmp_path / 'absent-home')
    return root, control


@pytest.mark.parametrize('arm', a.ARMS)
def test_exact_long_resources(arm):
    v = a.long_command(arm, 'm', 'c')
    assert v[v.index('-q')+1] == 'gpul40s' and v[v.index('-n')+1] == '4'
    assert v[v.index('-W')+1] == '23:00' and v[v.index('-gpu')+1] == 'num=1:mode=exclusive_process'
    assert 'rusage[mem=8GB]' in v and 'span[hosts=1]' in v and '-w' not in v
    assert v[-5:] == [str(a.SOURCE), 'm', 'run', arm, 'c']


def test_cpu_own_ended_dependency():
    c = a.cpu_command('cpu', 'm'); d = a.cpu_command('after-cpu', 'm', '123')
    assert '-w' not in c and d[d.index('-w')+1] == 'ended(123)'
    assert c[c.index('-q')+1] == d[d.index('-q')+1] == 'hpc'
    assert c[c.index('-n')+1] == '2' and d[d.index('-n')+1] == '1'


def test_full_remaining_sum():
    p = {x: (i+2)*1024**3 for i,x in enumerate(a.ARMS)}
    used = {x: 100 for x in a.ARMS}
    assert a.remaining(p, used) == sum(p.values())-300


@pytest.mark.parametrize('bad', ['missing','small','bool','negative','exceed','extra'])
def test_invalid_budget_rejected(bad):
    p = dict.fromkeys(a.ARMS, 2*1024**3); c = dict.fromkeys(a.ARMS, 0)
    if bad == 'missing': p.pop('protenix')
    if bad == 'small': p['protenix'] = 1024**3-1
    if bad == 'bool': c['protenix'] = False
    if bad == 'negative': c['boltz2'] = -1
    if bad == 'exceed': c['boltz2'] = 3*1024**3
    if bad == 'extra': c['other'] = 0
    with pytest.raises(ValueError): a.remaining(p,c)


@pytest.mark.parametrize('arm', a.ARMS)
def test_driver_changes_output_only(arm, monkeypatch):
    original = a.science.driver_command(arm, 'run', a.SCI_MANIFEST, a.SCI_CPU)
    command = a.driver_command(arm)
    i = original.index('--output')+1
    assert command[:i] == original[:i] and command[i+1:] == original[i+1:]
    assert command[i] == str(a.ROOT / ('run-'+arm) / 'payload/study')
    assert command[1] == str(a.SCIENCE/'examples/brd4_25_atoms.py')


@pytest.mark.parametrize('mode', ['timeout','ambiguous'])
def test_unknown_submission_cannot_repeat(mode, sandbox, monkeypatch):
    monkeypatch.setattr(a.science, 'sha', lambda _: 'config')
    def fake(*args, **kwargs):
        if mode == 'timeout': raise subprocess.TimeoutExpired('bsub',60)
        return subprocess.CompletedProcess(args[0],0,'Job <1> Job <2>','')
    monkeypatch.setattr(a.subprocess,'run',fake)
    with pytest.raises((ValueError,subprocess.TimeoutExpired)): a.submit_long('protenix','m','c')
    p = a.output_parent('protenix')
    assert a.read(p/'submission.json')['outcome'] == 'unknown'
    assert (p/'submission-started.json').exists()
    with pytest.raises(ValueError): a.submit_long('protenix','m','c')


def test_failed_readiness_blocks_dispatch(sandbox, monkeypatch):
    monkeypatch.setattr(a,'inherited',lambda: (_ for _ in ()).throw(ValueError('failed native')))
    monkeypatch.setattr(a,'submit_long',lambda *x: pytest.fail('must not submit'))
    with pytest.raises(ValueError): a.dispatch('m')
    assert not (a.ROOT/'long-dispatch-started.json').exists()


def test_full_gate_before_exact_three_submissions(sandbox, monkeypatch):
    native = {'projections':dict.fromkeys(a.ARMS,2*1024**3)}; events=[]
    monkeypatch.setattr(a,'inherited',lambda:native)
    monkeypatch.setattr(a,'verify_cpu',lambda m:{'sha256':'c'})
    def gate(p,path,**kw):
        assert p==native['projections'] and kw['phase']=='dispatch'
        events.append('gate'); return {'status':'passed'}
    monkeypatch.setattr(a,'storage_gate',gate)
    def submit(arm,*args):
        assert a.read(a.ROOT/'long-dispatch-started.json')['adoption']['sha256']==a.sha(a.CONTROL/'adoption.json')
        events.append(arm); return {'arm':arm,'job_id':str(len(events)),'outcome':'submitted'}
    monkeypatch.setattr(a,'submit_long',submit)
    a.dispatch('m')
    assert events==['gate',*a.ARMS] and a.read(a.ROOT/'long-submissions.json')['all_three_submitted']
    with pytest.raises(ValueError): a.dispatch('m')


def test_entry_repeats_full_gate_before_driver(sandbox, monkeypatch):
    arm='boltz2_chemistry'; p=a.output_parent(arm);p.mkdir()
    a.exclusive(p/'submission.json',{'outcome':'submitted','job_id':'123','args':a.long_command(arm,'m','c')})
    ad=a.CONTROL/'adoption.json';a.exclusive(ad,{})
    native={'projections':dict.fromkeys(a.ARMS,2*1024**3)}
    a.exclusive(a.ROOT/'long-dispatch-started.json',{'launch_manifest_sha256':'m','source_manifest_sha256':a.SCI_MANIFEST,'adoption':a.proof(ad),**native})
    monkeypatch.setenv('LSB_JOBID','123');monkeypatch.setenv('LSB_QUEUE','gpul40s')
    monkeypatch.setattr(a,'verify_cpu',lambda m:{'sha256':'c'});monkeypatch.setattr(a,'inherited',lambda:native)
    events=[]
    def gate(projections,path,**kw):
        assert projections==native['projections'] and kw['phase']=='entry'
        assert not (p/'payload/driver-invocation.json').exists();events.append('gate')
    monkeypatch.setattr(a,'storage_gate',gate)
    monkeypatch.setattr(a,'driver_command',lambda arm:['unchanged-driver'])
    def run(command,**kwargs):
        assert (p/'payload/driver-invocation.json').exists() and kwargs['cwd']==a.SCIENCE
        events.append('driver')
    monkeypatch.setattr(a.subprocess,'run',run)
    a.entry(arm,'m','c');assert events==['gate','driver']


def test_all_new_outputs_use_work3():
    for arm in a.ARMS:
        assert a.output_parent(arm)==a.ROOT/('run-'+arm)
        assert str(a.output_parent(arm)).startswith('/work3/')
