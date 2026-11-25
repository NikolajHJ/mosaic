"""Serial callback for the V100 continuation; original scientific code stays frozen."""
import argparse,hashlib,importlib.util,json,os,re,shlex,subprocess,sys,time,traceback
from pathlib import Path
WORK=Path('/work3/s234818/latentmol');NAME='source152-v100-serial-continuation-attempt001';CASES=('case-001','case-002','case-003')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def verified(asset):
 p=Path(asset['path']).resolve()
 if not p.is_relative_to(WORK/'runs') or sha(p)!=asset['sha256']:raise ValueError('Bound state changed: '+str(p))
 return p
def scheduler(job):
 r=subprocess.run(['bjobs','-a','-json','-o','jobid stat job_name',job],text=True,capture_output=True,timeout=30)
 if r.returncode or r.stderr.strip():raise RuntimeError('Scheduler unavailable')
 v=json.loads(r.stdout);rows=v.get('RECORDS',[])
 if len(rows)!=1 or rows[0].get('JOBID')!=job:raise ValueError('Scheduler returned wrong job')
 return rows[0]
def submit(args,command,release,control,name):
 intent=control/(name+'-intent.json');receipt=control/(name+'-receipt.json')
 if intent.exists() or receipt.exists():raise FileExistsError('Submission already attempted')
 script='#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(release))+'\nexec '+shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','LATENTMOL_SOURCE_ARCHIVE='+str(release)+'.tar.gz','bash','scripts/hpc/boltz_run.sh',*command])+'\n'
 write(intent,dict(arguments=args,command=command))
 try:r=subprocess.run(args,input=script,text=True,capture_output=True,timeout=60)
 except Exception as e:write(receipt,dict(status='ambiguous_no_retry',error=str(e),intent_sha256=sha(intent)));raise
 m=re.search(r'Job <([0-9]+)>',r.stdout);result=dict(status='submitted' if r.returncode==0 and m else 'failed_no_retry',job_id=m.group(1) if m else None,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent_sha256=sha(intent));write(receipt,result)
 if result['status']!='submitted':raise RuntimeError('Submission failed; no automatic retry')
 return result

def successful_job(job,label,release):
 row=scheduler(job)
 if row.get('STAT')=='DONE' and row.get('JOB_NAME')==label:return
 if row.get('ERROR')!='Job <'+job+'> is not found':raise ValueError('Predecessor is not exact-name DONE')
 path=WORK/'logs'/(label+'.'+job+'.out');text=path.read_text(errors='replace')
 if ('Subject: Job '+job+': <'+label+'> in cluster <dcc> Done' not in text or 'Successfully completed.' not in text or ' SOURCE='+str(release) not in text):raise ValueError('Missing authenticated terminal proof')

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-sha256',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--after',required=True);p.add_argument('--job',required=True);a=p.parse_args()
 release=Path(__file__).resolve().parents[2];root=WORK/'runs'/NAME;control=WORK/'submissions'/NAME;config_path=release/'configs/embedding_v100_serial_continuation.json'
 if not os.environ.get('LSB_JOBID') or release!=WORK/'releases/source-152-v100-serial-continuation-attempt001' or sha(config_path)!=a.plan_sha256 or os.environ.get('LATENTMOL_SOURCE_SHA256')!=a.source_sha256:raise ValueError('Require exact allocated controller release')
 op=load(release/'scripts/hpc/embedding_operator.py','_op');op.authenticate(release,a.source_sha256,a.plan_sha256,'configs/embedding_v100_serial_continuation.json');config=read(config_path);root.mkdir(parents=True,exist_ok=True);control.mkdir(parents=True,exist_ok=True)
 evidence=control/('after-'+a.after+'.json')
 if evidence.exists():raise FileExistsError('Callback already attempted')
 first=a.after=='first'
 if first:
  if a.job!=config['first_job']:raise ValueError('Wrong first V100 job')
  node=WORK/'runs/source151-v100-continuation-attempt001/gpu';label='source151-v100-continuation-attempt001-gpu';worker_release=WORK/'releases/source-151-v100-continuation-attempt001';expected_source=config['first_source_sha256'];expected_plan=config['first_plan_sha256']
  clones=read(verified(config['real_clones']));vector={r['evidence']['case']:dict(update=r['evidence']['update'],state=dict(path=r['path'],sha256=r['sha256'])) for r in clones['cases']};sequence=0;case='case-002';start=20
 else:
  request_path=control/(a.after+'-request.json');request=read(request_path);vector=request['vector'];sequence=request['sequence'];case=request['case'];start=request['start_update'];node=root/'jobs'/a.after;label=NAME+'-'+a.after;worker_release=release;expected_source=a.source_sha256;expected_plan=a.plan_sha256
  receipt=read(control/(a.after+'-worker-receipt.json'))
  if receipt['job_id']!=a.job or receipt['status']!='submitted':raise ValueError('Wrong predecessor receipt')
 try:
  successful_job(a.job,label,worker_release);result=read(node/'result.json');core=result['result']
  expected_status='completed_first_v100_segment' if first else 'completed_v100_segment'
  if (result.get('status')!=expected_status or result.get('job_id')!=a.job or result.get('source_sha256')!=expected_source or result.get('plan_sha256')!=expected_plan or core.get('case_id')!=case or core.get('starting_update')!=start or core.get('completed_update',0)<=start or core.get('scientific_stall') is not False or core['frozen_after'].get('frozen_state_unchanged') is not True or core['frozen_after'].get('context_unchanged') is not True):raise ValueError('V100 predecessor did not complete its guarded segment')
  state=core['current_state'];verified(state);end=core['completed_update'];assert end<=10000
 except Exception as e:
  rollback=None
  if first:
   old=scheduler(config['old_a100_job'])
   if old.get('STAT')=='PSUSP':
    r=subprocess.run(['bresume',config['old_a100_job']],text=True,capture_output=True,timeout=30);rollback=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  write(evidence,dict(status='stopped_no_successor',error=str(e),traceback=traceback.format_exc(),rollback=rollback));raise
 vector[case]=dict(update=end,state=state)
 if all(vector[x]['update']==10000 for x in CASES):
  write(evidence,dict(status='all_trajectories_complete_native_followup_pending',vector=vector));return
 nextcase=next(CASES[(CASES.index(case)+n)%3] for n in range(1,4) if vector[CASES[(CASES.index(case)+n)%3]]['update']<10000)
 sequence+=1;nextnode='segment-%06d'%sequence;selected=vector[nextcase];request=dict(sequence=sequence,case=nextcase,start_update=selected['update'],max_updates=min(config['max_updates_per_segment'],10000-selected['update']),state=selected['state'],vector=vector)
 verified(request['state']);fresh=op.quota();
 if fresh['minimum_free_gib'] < 7:raise RuntimeError('Insufficient projected storage headroom')
 request_path=control/(nextnode+'-request.json');write(request_path,request)
 name=NAME+'-'+nextnode;output=root/'jobs'/nextnode;output.parent.mkdir(parents=True,exist_ok=True)
 command=[str(WORK/'env/boltz311/bin/python'),str(release/'scripts/hpc/embedding_v100_serial_worker.py'),'--config',str(config_path),'--request',str(request_path),'--request-sha256',sha(request_path),'--source-sha256',a.source_sha256,'--plan-sha256',a.plan_sha256,'--output',str(output)]
 bsub=['bsub','-J',name,'-q','gpuv100','-n','4','-R','span[hosts=1]','-R','rusage[mem=16GB]','-R','select[gpu32gb]','-W','24:00','-oo',str(WORK/'logs'/(name+'.%J.out')),'-eo',str(WORK/'logs'/(name+'.%J.err')),'-gpu','num=1:mode=exclusive_process']
 worker=submit(bsub,command,release,control,nextnode+'-worker')
 cbname=name+'-after';cbcommand=[str(WORK/'env/boltz311/bin/python'),str(release/'scripts/hpc/embedding_v100_serial_controller.py'),'--source-sha256',a.source_sha256,'--plan-sha256',a.plan_sha256,'--after',nextnode,'--job',worker['job_id']]
 callback=['bsub','-J',cbname,'-q','hpc','-n','1','-W','0:30','-R','rusage[mem=2GB]','-R','span[hosts=1]','-w','ended('+worker['job_id']+')','-oo',str(WORK/'logs'/(cbname+'.%J.out')),'-eo',str(WORK/'logs'/(cbname+'.%J.err'))]
 after=submit(callback,cbcommand,release,control,nextnode+'-callback')
 cutover=None
 if first:
  old=scheduler(config['old_a100_job']);oldcb=scheduler(config['old_callback'])
  if old.get('STAT')!='PSUSP' or oldcb.get('STAT')!='PEND':raise RuntimeError('Original job/callback changed; preserve hold and inspect')
  stops=[]
  for job in (config['old_callback'],config['old_a100_job']):
   r=subprocess.run(['bkill',job],text=True,capture_output=True,timeout=30);stops.append(dict(job=job,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
   if r.returncode:raise RuntimeError('Original pending job retirement failed; inspect')
  cutover=stops
 write(evidence,dict(status='next_segment_submitted',completed_case=case,completed_update=end,next_case=nextcase,worker=worker,callback=after,original_pending_jobs_retired=cutover,quota=fresh,vector=vector))
if __name__=='__main__':main()
