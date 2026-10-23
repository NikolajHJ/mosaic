"""One compact readonly query of actual extended jobs and small proof metadata."""
from datetime import datetime,timezone
import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-extended-01')
out={'utc':datetime.now(timezone.utc).isoformat(),'records':{},'parents':{}}
for n in ['jobs.json','cpu-certificate.json','parent-adoption.json','projection.json','random-completion.json','continuation-completion.json']:
 if (r/n).exists():out['records'][n]=json.loads((r/n).read_bytes())
for p in r.iterdir():
 if not p.is_dir() or not (p/'submission.json').exists():continue
 sub=json.loads((p/'submission.json').read_bytes());a={'submission':sub};out['parents'][p.name]=a
 for label in ['stdout','stderr']:
  f=p/(label+'.'+sub['job_id']+'.log')
  if f.exists():a[label+'_tail']=f.read_text(errors='replace')[-3500:]
 if p.name=='cpu':
  a['xml']={}
  for f in p.glob('payload/*.xml'):
   try:
    t=ET.parse(f);a['xml'][f.name]={k:len(list(t.iter(k))) for k in ['testcase','failure','error','skipped']}
   except Exception as e:a['xml'][f.name]={'incomplete':repr(e)}
  a['logs']={f.name:f.read_text(errors='replace')[-4500:] for f in p.glob('payload/*.log')}
 for f in p.glob('storage-*/gate.json'):a[f.parent.name]=json.loads(f.read_bytes())['status']
 f=p/'payload/study/assessment.json'
 if f.exists():a['assessment']=json.loads(f.read_bytes())
jobs=out['records'].get('jobs.json',{});ids=list(jobs.values())+['29796551','29792672']
q=subprocess.run(['bjobs','-o','jobid stat queue job_name',*ids],capture_output=True,text=True,timeout=30)
out['scheduler']={'stdout':q.stdout,'stderr':q.stderr,'returncode':q.returncode}
print('STATUS_BEGIN');print(json.dumps(out))
'''
def main():
    out=REPO/'results/hpc'/HERE.name/'observations';out.mkdir(parents=True,exist_ok=True);stem=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=60)
    (out/(stem+'-stdout.bin')).write_bytes(q.stdout);(out/(stem+'-stderr.bin')).write_bytes(q.stderr);assert q.returncode==0,q.stderr.decode()[-1000:]
    j=json.loads(q.stdout.split(b'STATUS_BEGIN\n')[1]);(out/(stem+'.json')).write_bytes((json.dumps(j,indent=2)+'\n').encode())
    print(json.dumps({'utc':j['utc'],'scheduler':j['scheduler'],'cpu_xml':j['parents'].get('cpu',{}).get('xml'),'cpu_log_tails':j['parents'].get('cpu',{}).get('logs'),'callback':j['parents'].get('after-cpu',{}).get('stderr_tail'),'certificate':j['records'].get('cpu-certificate.json')}))
if __name__=='__main__':main()
