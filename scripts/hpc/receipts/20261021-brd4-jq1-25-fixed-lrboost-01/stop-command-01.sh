set -euo pipefail
python3 - <<'PY'
from pathlib import Path
import subprocess,json,re,datetime,hashlib,base64,time
root=Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-lrboost-stop-01');root.mkdir(exist_ok=False)
report=Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-report-04');jobs=json.loads((report/'jobs.json').read_text());assert jobs['native_jobs']=={'protenix':'29777709','boltz2':'29778203','boltz2_chemistry':'29778204'}
assert jobs['collectors']['boltz2']=='29778246' and jobs['collectors']['boltz2_chemistry']=='29778247' and jobs['final']=='29778249'
science=Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04');home=Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-storage-04')
for arm,parent,job in [('boltz2',science/'run-boltz2','29778203'),('boltz2_chemistry',home/'run-boltz2_chemistry','29778204')]:
 record=json.loads((parent/'submission.json').read_text());assert record['job_id']==job and record['outcome']=='submitted'
(root/'report-jobs-original.json').write_bytes((report/'jobs.json').read_bytes())
ids=['29778246','29778247','29778249','29778203','29778204'];a=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=45);raw=a.stdout+a.stderr;(root/'scheduler-before.txt').write_text(raw)
parts=re.split(r'(?=\nJob <\d+>)',raw);records={}
for part in parts:
 compact=re.sub(r'\s+','',part);match=re.search(r'Job<(\d+)>',compact)
 if match:
  job=match.group(1);statuses=re.findall(r'Status<([^>]+)>',compact);assert len(statuses)==1;records[job]={'status':statuses[0],'raw':part,'compact':compact}
result={'authorization':'Human21October2026 stopcurrent/restartfreshall3 withPro3.0/Bol.05/chem.05','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'actions':[],'preserved_completed_protenix':'29777709','scientific_inputs_mutated':False}
retire=[]
for j,parent in [('29778246','complete-boltz2'),('29778247','complete-boltz2_chemistry'),('29778249','combined-report')]:
 v=records.get(j)
 if v is None:result['actions'].append({'job':j,'action':'no_cancel_missing_history'});continue
 assert 'Queue<hpc>' in v['compact'] and str(report/parent).replace(' ','') in v['compact'],'Wrongreportidentity'
 if v['status']=='PEND':
  assert 'Started' not in v['raw'];retire.append(j)
 else:result['actions'].append({'job':j,'action':'no_pending_cancel','actual_before':v['status']})
if retire:
 a=subprocess.run(['bkill',*retire],capture_output=True,text=True,timeout=35);(root/'retire-report-bkill.txt').write_text(a.stdout+a.stderr);result['actions'].append({'jobs':retire,'action':'retire_verified_pending_unstarted_reports','returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr});assert a.returncode==0
stop=[]
for j in ['29778203','29778204']:
 v=records.get(j)
 if v is None:result['actions'].append({'job':j,'action':'no_cancel_missing_history'});continue
 assert 'Queue<gpul40s>' in v['compact'] and str(science).replace(' ','') in v['compact'],'Wrongnativeidentity'
 if v['status'] in ['RUN','PEND','SSUSP','USUSP','PSUSP']:stop.append(j)
 else:result['actions'].append({'job':j,'action':'no_active_cancel','actual_before':v['status']})
if stop:
 a=subprocess.run(['bkill',*stop],capture_output=True,text=True,timeout=35);(root/'native-bkill.txt').write_text(a.stdout+a.stderr);result['actions'].append({'jobs':stop,'action':'user_requested_native_stop','returncode':a.returncode,'stdout':a.stdout,'stderr':a.stderr});assert a.returncode==0
for attempt in range(10):
 a=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=35);raw=a.stdout+a.stderr;(root/('scheduler-after-'+str(attempt)+'.txt')).write_text(raw);final={}
 for part in re.split(r'(?=\nJob <\d+>)',raw):
  compact=re.sub(r'\s+','',part);m=re.search(r'Job<(\d+)>.*?Status<([^>]+)>',compact)
  if m:final[m.group(1)]=m.group(2)
 if all(final.get(j) in ['EXIT','DONE'] for j in stop+retire):break
 time.sleep(2)
result['actual_final']=final;result['stopped_confirmed']=all(final.get(j) in ['EXIT','DONE'] for j in stop);result['retirement_confirmed']=all(final.get(j) in ['EXIT','DONE'] for j in retire)
for arm,parent,job in [('boltz2',science/'run-boltz2','29778203'),('boltz2_chemistry',home/'run-boltz2_chemistry','29778204')]:
 for name in ['stdout.'+job+'.log','stderr.'+job+'.log','submission.json','payload/driver-invocation.json']:
  q=parent/name
  if q.exists():
   dest=root/arm/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(q.read_bytes())
(root/'stop-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
files={}
for q in root.rglob('*'):
 if q.is_file():files[q.relative_to(root).as_posix()]={'sha256':hashlib.sha256(q.read_bytes()).hexdigest(),'bytes':q.stat().st_size,'base64':base64.b64encode(q.read_bytes()).decode()}
print('STOP_EVIDENCE_JSON='+json.dumps({'root':str(root),'receipt':result,'files':files}))
PY