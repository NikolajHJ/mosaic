"""One compact readonly scheduler/artifact query; no local numerical execution."""
from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-random-chem-01')
out={'utc':datetime.now(timezone.utc).isoformat(),'exists':r.exists(),'records':{},'parents':{}};ids=[]
if r.exists():
 for n in ['source-staging.json','jobs.json','cpu-certificate.json','long-dispatch-started.json']:
  if (r/n).exists():out['records'][n]=json.loads((r/n).read_bytes())
 for p in r.iterdir():
  if not p.is_dir() or not (p/'submission.json').exists():continue
  sub=json.loads((p/'submission.json').read_bytes());job=sub.get('job_id');a={'submission':sub};out['parents'][p.name]=a
  if job:ids.append(job)
  for stream in ['stdout','stderr']:
   f=p/(stream+'.'+str(job)+'.log')
   if f.exists():
    text=f.read_text(errors='replace');a[stream]={'same_job_lsf':('Subject: Job '+str(job)+':' in text),'success':'Successfully completed.' in text,'tail':text[-5000:]}
  if p.name=='cpu':
   a['xml']={}
   for f in p.glob('payload/*.xml'):
    try:
     tree=ET.parse(f);a['xml'][f.name]={'cases':len(list(tree.iter('testcase'))),'failures':len(list(tree.iter('failure'))),'errors':len(list(tree.iter('error'))),'skipped':len(list(tree.iter('skipped')))}
    except Exception as e:a['xml'][f.name]={'incomplete':repr(e)}
   a['test_logs']={f.name:f.read_text(errors='replace')[-6000:] for f in p.glob('payload/*.log')}
  for n in ['payload/study/assessment.json','payload/study/summary.json']:
   f=p/n
   if f.exists():
    try:
     j=json.loads(f.read_bytes());a[n]={k:v for k,v in j.items() if k not in ['cases','proofs','executed_source','context_files','environment']}
    except Exception as e:a[n]={'incomplete':repr(e)}
if ids:
 q=subprocess.run(['bjobs','-o','jobid stat queue job_name',*ids],capture_output=True,text=True,timeout=30);out['scheduler']={'stdout':q.stdout,'stderr':q.stderr,'returncode':q.returncode}
print('OBS_BEGIN');print(json.dumps(out))
'''
def main():
    folder=REPO/'results/hpc'/HERE.name/'observations';folder.mkdir(parents=True,exist_ok=True);stem=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (folder/(stem+'-stdout.bin')).write_bytes(q.stdout);(folder/(stem+'-stderr.bin')).write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-1000:]
    j=json.loads(q.stdout.split(b'OBS_BEGIN\n')[1]);(folder/(stem+'.json')).write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j))
if __name__=='__main__':main()
