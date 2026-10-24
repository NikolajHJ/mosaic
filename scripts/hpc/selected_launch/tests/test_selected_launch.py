"""Allocated checks for independent routing and unchanged original proof adoption."""
import pytest
import json
import subprocess
import sys
import selected_contract as c
import selected_ops as op
from selected_storage import budget_remaining


def test_selected_science_scope():
    assert c.SELECTED == {'boltz30k': 'boltz2', 'dbmol10k': 'boltz2_dbmol', 'protenix50k40k': 'continuation'}


def test_actual_deferred_absence_and_failed_observers():
    rows = c.deferred_absence()
    assert [r['arm'] for r in rows] == ['boltz2_chemistry', 'random']
    assert all(r['production_absent'] and not r['future_admission'] for r in rows)
    assert c.previous_admission()['sha256']=='c75c65397c2b36378c8127be7da3d2dde803fa8d38ab51a594e2b1d3839896ab'


@pytest.mark.parametrize('phase', ['cpu', 'after-cpu', 'dispatch', 'native', 'segment', 'after-native', 'after-segment', 'noise-cpu', 'noise-gpu'])
def test_resources_and_exclusive_gpu(phase):
    args = op.resources(phase)
    if phase in ('native', 'segment', 'noise-gpu'):
        assert args[1] == 'gpul40s' and args[3] == '4' and 'num=1:mode=exclusive_process' in args
        assert args[-1] == ('23:00' if phase == 'segment' else '02:00')
    else:
        assert args[1] == 'hpc' and '-gpu' not in args


@pytest.mark.parametrize('name,phase,index', [
    ('boltz30k', 'segment', 0), ('boltz30k', 'segment', 2),
    ('dbmol10k', 'segment', 0), ('dbmol10k', 'segment', 3),
    ('protenix50k40k', 'segment', 0), ('protenix50k40k', 'segment', 3),
    ('boltz30k', 'native', 1), ('dbmol10k', 'native', 1),
    ('protenix50k40k', 'native', 1), ('protenix50k40k', 'after-segment', 2)])
def test_only_selected_immutable_boundaries(name, phase, index):
    label = c.arm_label(name, phase, index)
    assert c.decode(name, label) == (phase, index)
    args = op.command(name, label, phase, 'a' * 64)
    assert args[-3:] == [name, phase, str(c.profile(name)['root'] / label)]


@pytest.mark.parametrize('name,label', [('boltz30k', 'boltz2_chemistry-segment-0000'),
    ('protenix50k40k', 'random-segment-0000'), ('boltz30k', 'boltz2-native-0000'),
    ('dbmol10k', 'pilot-native-0000'), ('boltz30k', 'boltz2-segment-0003'),
    ('protenix50k40k', 'continuation-segment-0004')])
def test_deferred_unknown_or_replayed_native_rejected(name, label):
    with pytest.raises(ValueError):
        c.decode(name, label)


@pytest.mark.parametrize('full,current,expected', [(100, 0, 100), (100, 99, 1), (100, 100, 0), (2**40, 20, 2**40 - 20)])
def test_full_remaining_never_drops_existing_bytes(full, current, expected):
    assert budget_remaining(full, current) == expected


@pytest.mark.parametrize('full,current', [(1, 2), (1, -1), (1, .5), (True, 0), (10, True)])
def test_invalid_or_exceeded_budget_fails(full, current):
    with pytest.raises(ValueError):
        budget_remaining(full, current)


@pytest.mark.parametrize('name', list(c.PROFILES))
def test_actual_original_source_cpu_native_and_projection(name):
    # A production allocation imports one scientific release. Exercise each
    # adapter in its own interpreter so cached imports cannot cross releases.
    code = '''
import selected_contract as c, selected_ops as op
name = NAME
proof = c.native_units(name)
original = op.load_original(name, 'a' * 64)
_, ready = op.selected_readiness(name, 0, original)
assert proof['status'] == ready['status'] == 'passed'
assert proof['unit_bytes'] == ready['storage_projection']['maximum_pilot_update_bytes']
root = c.profile(name)['source' if name != 'dbmol10k' else 'original_launcher']
assert original.produce.__code__.co_filename.startswith(str(root))
'''.replace('NAME', repr(name))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


def test_original_noise_source_config_unchanged():
    proof = c.verify_noise_source()
    cfg = c.read(proof['path'])
    assert cfg['steps'] == 1000 and cfg['noise']['seed'] == 230017 and cfg['objective'] == 'native_only_pocket_contact'


@pytest.mark.parametrize('dependency', ['done(123)', 'ended(123)&&ended(456)'])
def test_unapproved_cross_arm_dependencies_rejected(dependency):
    with pytest.raises(ValueError):
        op.command('control', 'cpu', 'cpu', 'a' * 64, dependency)


def test_owner_role_roundtrip_accepts_only_real_passed_proofs(tmp_path,monkeypatch):
    proof=tmp_path/'unchanged-proof';proof.write_text('exact original proof')
    value={'status':'passed','kind':'selected_owner','profile':'boltz30k','selected_arm':'boltz2',
           'launcher_manifest_sha256':'a'*64,'proofs':[{'path':str(proof),'sha256':c.sha(proof)}]}
    c.exclusive(tmp_path/'nonchem-stride20-v3-owner.json',value)
    monkeypatch.setattr(op,'profile',lambda name:{'root':tmp_path})
    assert op.ownership('boltz30k','a'*64)==value


def test_all_initial_owners_are_adopted_before_any_independent_submission(tmp_path,monkeypatch):
    import noisy_entry
    order=[]
    monkeypatch.setattr(op,'CONTROL',tmp_path)
    monkeypatch.setattr(op,'final_job',lambda *a:[])
    monkeypatch.setattr(c,'verify_xml',lambda *a:op.CASES)
    monkeypatch.setattr(op,'sha',lambda *a:'a'*64)
    monkeypatch.setattr(op,'native_units',lambda name:{})
    monkeypatch.setattr(c,'deferred_absence',lambda:None)
    monkeypatch.setattr(c,'previous_admission',lambda:None)
    monkeypatch.setattr(op,'exclusive',lambda *a:None)
    monkeypatch.setattr(op,'adopt',lambda name,parent,m,original:order.append(('adopt',name,original)))
    monkeypatch.setattr(op,'submit',lambda name,*a:order.append(('submit',name)))
    monkeypatch.setattr(noisy_entry,'adopt_cpu',lambda *a:order.append(('noise',)))
    op.certify_cpu('a'*64,tmp_path)
    assert order[:3]==[('adopt',name,None) for name in c.PROFILES]
    assert order[3:6]==[('submit',name) for name in c.PROFILES]
    assert len(order)==6


@pytest.mark.parametrize('name',list(c.PROFILES))
def test_persistence_proxy_is_scoped_after_startup_and_restored(name):
    from retention import ProcessProxy
    from types import SimpleNamespace
    events=[];original=SimpleNamespace(run=lambda *a,**k:None)
    fake=SimpleNamespace(subprocess=original)
    def verified(*args):
        assert fake.subprocess is original
        events.append('startup')
    fake.verify_command_capture=fake.ensure_gpu_startup=fake.verify_startup=fake.verify_startup_entry=verified
    def produce(*args):
        assert isinstance(fake.subprocess,ProcessProxy) and fake.subprocess.original is original
        events.append('production')
        return 'complete'
    fake.produce=produce
    assert op.produce_with_retention(name,fake,c.profile(name)['root']/'unused','segment',0,'a'*64)=='complete'
    assert fake.subprocess is original and events[-1]=='production'
    assert events[:-1]==(['startup','startup'] if name=='protenix50k40k' else ['startup'])


@pytest.mark.parametrize('name,production',[
    ('boltz2-segment-0000',True),('pilot-after-segment-0000',True),
    ('continuation-segment-0000-certificate.json',True),('random-segment-0004',True),
    ('startup-command-continuation-segment-0000-started.json',False),
    ('startup-boltz2-segment-0000-certificate.json',False),
    ('pilot-stride20-v3-segment-0000-startup-started.json',False),
    ('continuation-stride20-v3-command-segment-0000.json',False)])
def test_actual_production_names_never_confuse_startup_proofs(tmp_path,name,production):
    (tmp_path/name).write_text('{}')
    assert bool(c.production_entries(tmp_path))==production
