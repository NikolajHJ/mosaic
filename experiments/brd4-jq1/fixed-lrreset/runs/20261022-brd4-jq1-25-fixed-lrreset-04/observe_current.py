"""One compact read-only scheduler/artifact observation with a fresh local receipt."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[5]
REMOTE = r'''
import hashlib,json,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime,timezone
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04')
home=Path('/zhome/95/e/206121/mosaic-runs/20261022-brd4-jq1-25-fixed-lrreset-04')
out={'captured_utc':datetime.now(timezone.utc).isoformat(),'records':{},'arms':{},'cpu_xml':{}};ids=['29792389','29792390']
for name in ['cpu-certificate.json','native-submissions.json','long-submissions.json','long-dispatch-started.json']:
 p=root/name
 if not p.exists():out['records'][name]={'exists':False};continue
 b=p.read_bytes();j=json.loads(b);item={'exists':True,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 if name=='cpu-certificate.json':item.update({k:v for k,v in j.items() if k!='proofs'})
 elif name=='long-dispatch-started.json':item['projections']=j['projections'];item['storage_gate']=j['storage_gate']
 else:
  def walk(x):
   if isinstance(x,dict):
    if 'job_id' in x:ids.append(x['job_id'])
    for v in x.values():walk(v)
   elif isinstance(x,list):
    for v in x:walk(v)
  walk(j)
  item['jobs']=[{'arm':x['arm'],'native':x['native']['job_id'],'callback':x['callback']['job_id']} for x in j['jobs']] if name.startswith('native') else [{k:v for k,v in x.items() if k in ['job_id','arm','outcome','output_parent','learning_rate']} for x in j['jobs']]
 out['records'][name]=item
for p in (root/'cpu/payload').glob('*.xml'):
 try:
  t=ET.parse(p);out['cpu_xml'][p.name]={'cases':len(list(t.iter('testcase'))),'failures':len(list(t.iter('failure'))),'errors':len(list(t.iter('error'))),'skipped':len(list(t.iter('skipped')))}
 except Exception as e:out['cpu_xml'][p.name]={'incomplete':str(e)}
for arm in ['protenix','boltz2','boltz2_chemistry']:
 a={}
 for name in ['decision-'+arm+'.json','validate-'+arm+'/payload/study/assessment.json','validate-'+arm+'/payload/study/summary.json']:
  p=root/name
  if p.exists():
   b=p.read_bytes();j=json.loads(b);a[name]={'sha256':hashlib.sha256(b).hexdigest(),**{k:v for k,v in j.items() if k in ['status','error','native_job','job_id','baseline_initialization_match','initialization_execution','runtime_gate','storage_projection']}}
 parent=(home if arm=='boltz2_chemistry' else root)/('run-'+arm)
 a['driver_invocation_exists']=(parent/'payload/driver-invocation.json').exists()
 for p in parent.glob('**/*storage-gate.json'):
  j=json.loads(p.read_bytes());a['entry_gate']={'path':str(p),'status':j.get('status'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 for name in ['states','updates']:
  p=parent/('payload/study/case-0/'+name+'.jsonl')
  if p.exists():
   count=0;last=None;contiguous=True
   for line in p.open('rb'):
    if not line.endswith(b'\n'):break
    j=json.loads(line);contiguous=contiguous and j['step']==count;last=j['step'];count+=1
   a[name]={'committed_rows':count,'last_index':last,'contiguous':contiguous}
 out['arms'][arm]=a
for parent in ['cpu','after-cpu','validate-protenix','validate-boltz2','validate-boltz2_chemistry','after-ready-protenix','after-ready-boltz2','after-ready-boltz2_chemistry']:
 for p in (root/parent).glob('stderr.*.log'):
  s=p.read_text(errors='replace')
  if 'Traceback' in s:out['records'][str(p.relative_to(root))]=s[-3600:]
s=subprocess.run(['bjobs','-o','jobid stat queue job_name']+list(dict.fromkeys(ids)),text=True,capture_output=True,timeout=30)
out['scheduler']={'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}
print('OBSERVATION_JSON_BEGIN');print(json.dumps(out))
'''


def main():
    folder = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/observations'
    folder.mkdir(parents=True, exist_ok=True)
    stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c 'python3 -'"], input=REMOTE.encode(), capture_output=True, timeout=60)
    (folder / (stem + '-stdout.bin')).write_bytes(result.stdout); (folder / (stem + '-stderr.bin')).write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-1000:]
    report = json.loads(result.stdout.split(b'OBSERVATION_JSON_BEGIN\n', 1)[1])
    (folder / (stem + '.json')).write_bytes(json.dumps(report, indent=2).encode() + b'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
