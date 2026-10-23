"""Single readonly SSH query of actual analysis receipts, jobs and small saved artifacts."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/lrreset-analysis-02')
out={'utc':datetime.now(timezone.utc).isoformat(),'records':{},'parents':{},'archives':{}};ids=['29792672']
for name in ['jobs.json','cpu-certificate.json','analysis-dispatch-started.json']:
 if (r/name).exists():out['records'][name]=json.loads((r/name).read_bytes())
for p in r.iterdir():
 if not p.is_dir() or not (p/'submission.json').exists():continue
 sub=json.loads((p/'submission.json').read_bytes());job=sub.get('job_id');a={'submission':sub};out['parents'][p.name]=a
 if job:ids.append(job)
 for stream in ['stdout','stderr']:
  f=p/(stream+'.'+str(job)+'.log')
  if f.exists():
   value=f.read_text(errors='replace');a[stream]={'same_job_lsf':('Subject: Job '+str(job)+':' in value),'success':'Successfully completed.' in value,'exit1':'Exited with exit code 1.' in value,'tail':value[-3200:]}
 f=p/'payload/report-tests.xml'
 if f.exists():
  try:
   t=ET.parse(f);a['xml']={'cases':len(list(t.iter('testcase'))),'failures':len(list(t.iter('failure'))),'errors':len(list(t.iter('error'))),'skipped':len(list(t.iter('skipped')))}
  except Exception as e:a['xml']={'incomplete':repr(e)}
 for name in ['assessment.json','report/report_receipt.json']:
  f=p/'payload'/name
  if f.exists():
   try:
    j=json.loads(f.read_bytes());a[name]={k:v for k,v in j.items() if k not in ['proofs','points','provenance','source','input_snapshots','environment']}
    if 'points' in j:a[name]['point_steps']=[x['step'] for x in j['points']]
   except Exception as e:a[name]={'incomplete':repr(e)}
 for f in p.glob('payload/report-tests.log'):
  value=f.read_text(errors='replace');a['test_log_tail']=value[-6000:]
for p in r.glob('*.tar.gz'):out['archives'][p.name]={'path':str(p),'bytes':p.stat().st_size}
for p in r.glob('*.json'):
 if p.name.startswith(('interim-','combined-report')):out['records'][p.name]=json.loads(p.read_bytes())
q=subprocess.run(['bjobs','-o','jobid stat queue job_name']+list(dict.fromkeys(ids)),capture_output=True,text=True,timeout=30)
out['scheduler']={'stdout':q.stdout,'stderr':q.stderr,'returncode':q.returncode}
print('STATUS_BEGIN');print(json.dumps(out))
'''


def main():
    folder=REPO/'results/hpc/20261023-brd4-jq1-lrreset-analysis-02/observations';folder.mkdir(parents=True,exist_ok=True)
    stem=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (folder/(stem+'-stdout.bin')).write_bytes(q.stdout);(folder/(stem+'-stderr.bin')).write_bytes(q.stderr)
    assert q.returncode==0,q.stderr.decode()[-1000:]
    value=json.loads(q.stdout.split(b'STATUS_BEGIN\n')[1]);(folder/(stem+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
