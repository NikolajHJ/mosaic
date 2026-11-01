"""Portable device selection and fake-scheduler migration controls; no models."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

OWNER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(OWNER/'scripts'), str(OWNER)]
for parent in OWNER.parents:
    if (parent/'scripts/hpc/gpu_policy.py').exists():
        sys.path.insert(0, str(parent/'scripts/hpc')); break
import gpu_policy as gpu
import chemistry_screen_control as control
import chemistry_screen_ops as ops
import protenix_chemistry_screen as screen


@pytest.mark.parametrize('name,queue,hardware,memory', [
    ('v100-32gb-v1','gpuv100','Tesla V100-SXM2-32GB',32768),
    ('l40s-48gb-v1','gpul40s','NVIDIA L40S',46068),
    ('a10-24gb-v1','gpua10','NVIDIA A10',23028),
    ('a100-40gb-v1','gpua100','NVIDIA A100-PCIE-40GB',40960),
    ('a40-48gb-v1','gpua40','NVIDIA A40',46068),
])
def test_explicit_device_profile_keeps_default_and_ownership(name, queue, hardware, memory):
    row = {'gpu_profile': name, 'queue': queue}
    assert gpu.selected(row)['queue'] == queue
    assert gpu.validate_hardware(hardware, memory, gpu.selected(row))['profile'] == name
    assert all(x in gpu.resource(row) for x in ['num=1', 'mode=shared', 'j_exclusive=yes', 'mps=no'])
    assert gpu.profile()['id'] == 'v100-32gb-v1'
    with pytest.raises(ValueError): gpu.selected({**row,'queue':'foreign'})
    with pytest.raises(ValueError): gpu.validate_hardware(hardware, 1000, gpu.selected(row))


def test_a10_does_not_accept_a100_by_substring():
    with pytest.raises(ValueError): gpu.validate_hardware('NVIDIA A100-PCIE-40GB', 40960, gpu.profile('a10-24gb-v1'))


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), -1])
def test_invalid_device_timing_is_rejected(seconds):
    with pytest.raises(ValueError): control.measured_wall({'warm_seconds_max':seconds}, 20000)


def fixture_plan(tmp_path):
    root=tmp_path/'new'; (root/'jobs/probe').mkdir(parents=True)
    old=tmp_path/'old'; old.mkdir()
    sub={'job_id':'123','job_name':'old-arm','argv':['bsub','-q','gpuv100','bash','-l','/old/stage.sh']}
    screen.write(old/'submission.json',sub)
    newsub={'job_id':'456','job_name':'new-probe'}
    screen.write(root/'jobs/probe/submission.json',newsub)
    (root/'jobs/probe/stdout.456.log').write_text('Subject: Job 456: <new-probe> in cluster <dcc> Done\nSuccessfully completed.\n')
    (root/'probe').mkdir()
    screen.write(root/'probe/assessment.json',{'status':'passed','job_id':'456','results':[{'arm':'arm','warm_seconds_max':1.}]})
    plan={'root':str(root),'source_root':str(tmp_path),'arms':{'arm':{'queue':'gpua100','screen':{'updates':20000},'timing_probe':'arm'}},
          'probe_arms':['arm'],'production_arms':['arm'],
          'migration':{'arm':{'submission':screen.binding(old/'submission.json'),'payload':str(old/'payload')}}}
    path=tmp_path/'plan.json';screen.write(path,plan)
    return plan,path,root,old


@pytest.mark.parametrize('case', ['success','already_running','hold_race','unknown_replacement','old_payload','probe_failed'])
def test_conditional_migration_preserves_live_or_unknown_writers(tmp_path, monkeypatch, case):
    plan,path,root,old=fixture_plan(tmp_path)
    state={'old':'RUN' if case=='already_running' else 'PEND'};events=[]
    if case=='old_payload':(old/'payload').mkdir()
    if case=='unknown_replacement':
        (root/'production/arm').mkdir(parents=True);(root/'production/arm/submission-intent.json').write_text('{}')
    def observed(job):
        if job=='456':return {'JOBID':'456','USER':'s234818','JOB_NAME':'new-probe','QUEUE':'gpua100','STAT':'EXIT' if case=='probe_failed' else 'DONE'}
        return {'JOBID':'123','USER':'s234818','JOB_NAME':'old-arm','QUEUE':'gpuv100','STAT':state['old'],'COMMAND':'bash -l /old/stage.sh'}
    def command(argv,timeout=60):
        events.append(argv[0])
        if argv[0]=='bstop':state['old']='USUSP' if case=='hold_race' else 'PSUSP'
        if argv[0]=='bkill':state['old']='EXIT'
        if argv[0]=='bresume':state['old']='RUN'
        return {'returncode':0,'stdout':'submitted, no execution' if argv[0]=='bhist' else '', 'stderr':''}
    def submit(*args):
        assert state['old']=='EXIT';events.append('submit');return '789'
    monkeypatch.setenv('LSB_JOBID','777')
    monkeypatch.setattr(control,'observed_job',observed)
    monkeypatch.setattr(ops,'command',command)
    monkeypatch.setattr(control,'capacity',lambda *a:{'status':'passed'})
    monkeypatch.setattr(control,'submit',submit)
    control.migrate(plan,path)
    result=json.loads((root/'migration-summary.json').read_text())
    if case=='success':
        assert events.index('bstop')<events.index('bkill')<events.index('submit')
        assert result['decisions']['arm']['job_id']=='789'
    else:
        assert 'submit' not in events and 'bkill' not in events
    if case=='hold_race':assert events[-1]=='bresume'


@pytest.mark.parametrize('profile,queue', [('a10-24gb-v1','gpua10'),('a100-40gb-v1','gpua100'),('a40-48gb-v1','gpua40')])
def test_actual_submit_command_uses_selected_queue_and_ended_callback(tmp_path, monkeypatch, profile, queue):
    plan={'root':str(tmp_path),'source_root':'/source','job_name_prefix':'test','arms':{'arm':{'gpu_profile':profile,'queue':queue}},'probe_arms':['arm']}
    p=tmp_path/'plan.json';screen.write(p,plan);calls=[]
    def command(argv,timeout=60):
        calls.append(argv)
        if argv[0]=='bsub':return {'returncode':0,'stdout':f'Job <123> is submitted to queue <{queue}>.','stderr':''}
        if argv[0]=='bresume':return {'returncode':0,'stdout':'resumed','stderr':''}
        rows=[] if '-u' in argv else [{'JOBID':'123','USER':'s234818','STAT':'PSUSP','JOB_NAME':'test-probe'}]
        return {'returncode':0,'stdout':json.dumps({'RECORDS':rows}),'stderr':''}
    monkeypatch.setattr(ops,'capacity',lambda *a:{'status':'passed'});monkeypatch.setattr(ops,'command',command)
    assert ops.submit(p,'probe',wall_minutes=30,dependency='99',dependency_type='ended')=='123'
    argv=next(a for a in calls if a[0]=='bsub')
    assert argv[argv.index('-q')+1]==queue
    assert argv[argv.index('-gpu')+1]==gpu.resource(plan['arms']['arm'])
    assert argv[argv.index('-w')+1]=='ended(99)'
