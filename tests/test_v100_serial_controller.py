"""Allocated CPU simulation of the V100 serial callback without scheduler writes."""
import hashlib,importlib.util,json,shutil,sys
from pathlib import Path
from types import SimpleNamespace
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/hpc/embedding_v100_serial_controller.py'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def asset(p):return dict(path=str(p),sha256=sha(p))

@pytest.fixture
def simulation(tmp_path,monkeypatch):
 work=tmp_path/'latentmol';release=work/'releases/source-152-v100-serial-continuation-attempt001';(release/'scripts/hpc').mkdir(parents=True);(release/'configs').mkdir();script=release/'scripts/hpc/embedding_v100_serial_controller.py';shutil.copy2(SOURCE,script)
 spec=importlib.util.spec_from_file_location('fake_serial_controller',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);monkeypatch.setattr(module,'WORK',work)
 states={}
 for n,u in [(1,1899),(2,20),(3,20)]:
  p=work/'runs'/'saved'/('case-%03d'%n)/('state-%06d.pt'%u);p.parent.mkdir(parents=True);p.write_bytes(('state%d'%n).encode());states['case-%03d'%n]=dict(evidence=dict(case='case-%03d'%n,update=u),**asset(p))
 clones=work/'runs/source149-v100-real-rebase-attempt001/cpu/result.json';clones.parent.mkdir(parents=True);clones.write_text(json.dumps(dict(cases=list(states.values()))))
 state=work/'runs/source151-v100-continuation-attempt001/gpu/trajectory/states/state-000021.pt';state.parent.mkdir(parents=True);state.write_bytes(b'next-state')
 source='a'*64;parent_source='b'*64;parent_plan='c'*64
 result=dict(status='completed_first_v100_segment',job_id='1',source_sha256=parent_source,plan_sha256=parent_plan,result=dict(case_id='case-002',starting_update=20,completed_update=21,scientific_stall=False,frozen_after=dict(frozen_state_unchanged=True,context_unchanged=True),current_state=asset(state)))
 (state.parent.parent.parent/'result.json').write_text(json.dumps(result))
 cfg=dict(first_job='1',first_source_sha256=parent_source,first_plan_sha256=parent_plan,old_a100_job='2',old_callback='3',real_clones=asset(clones),max_updates_per_segment=2000)
 config=release/'configs/embedding_v100_serial_continuation.json';config.write_text(json.dumps(cfg));plan=sha(config)
 monkeypatch.setenv('LSB_JOBID','9');monkeypatch.setenv('LATENTMOL_SOURCE_SHA256',source)
 monkeypatch.setattr(sys,'argv',['controller','--source-sha256',source,'--plan-sha256',plan,'--after','first','--job','1'])
 monkeypatch.setattr(module,'load',lambda *a:SimpleNamespace(authenticate=lambda *a:None,quota=lambda:dict(minimum_free_gib=50)))
 monkeypatch.setattr(module,'successful_job',lambda *a:None)
 monkeypatch.setattr(module,'scheduler',lambda job:dict(JOBID=job,STAT={'2':'PSUSP','3':'PEND'}[job]))
 events=[]
 def fake_submit(args,command,release,control,name):
  events.append(('submit',name));return dict(status='submitted',job_id='4' if name.endswith('-worker') else '5')
 def fake_run(args,**kw):events.append(('command',args[0],args[1]));return SimpleNamespace(returncode=0,stdout='ok',stderr='')
 monkeypatch.setattr(module,'submit',fake_submit);monkeypatch.setattr(module.subprocess,'run',fake_run)
 return module,work,events


def test_successful_first_callback_advances_next_molecule_and_retires_old_jobs(simulation):
 module,work,events=simulation;module.main();control=work/'submissions/source152-v100-serial-continuation-attempt001'
 result=json.loads((control/'after-first.json').read_text());request=json.loads((control/'segment-000001-request.json').read_text())
 assert result['status']=='next_segment_submitted' and result['completed_case']=='case-002' and result['completed_update']==21
 assert request['case']=='case-003' and request['start_update']==20 and request['max_updates']==2000
 assert request['vector']['case-001']['update']==1899 and request['vector']['case-002']['update']==21
 assert events==[('submit','segment-000001-worker'),('submit','segment-000001-callback'),('command','bkill','3'),('command','bkill','2')]


def test_failed_first_segment_restores_original_pending_job_and_dispatches_nothing(simulation,monkeypatch):
 module,work,events=simulation
 def fail(*a):raise ValueError('failed predecessor')
 monkeypatch.setattr(module,'successful_job',fail)
 with pytest.raises(ValueError):module.main()
 result=json.loads((work/'submissions/source152-v100-serial-continuation-attempt001/after-first.json').read_text())
 assert result['status']=='stopped_no_successor' and events==[('command','bresume','2')]
