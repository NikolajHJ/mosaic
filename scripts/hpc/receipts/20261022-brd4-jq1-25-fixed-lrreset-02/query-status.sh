set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02')
analysis=root.parent
record={'utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'artifacts':{},'cpu_xml':{}}
ids=[]
def ref(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
for name in ('source-staging.json','initial-submissions-complete.json','cpu/submission.json','after-cpu/submission.json','cpu-certificate.json','native-submissions.json','long-submissions.json','native-validation-dispatch-quota/lrreset-storage-gate.json','cpu-entry-quota/lrreset-storage-gate.json','preflight-quota/lrreset-storage-gate.json',*('decision-'+a+'.json' for a in ('protenix','boltz2','boltz2_chemistry'))):
 p=root/name
 if p.is_file():
  v=json.loads(p.read_text());record['artifacts'][name]={'file':ref(p),'content':v}
  if name.endswith('/submission.json'):ids.append(v['job_id'])
  if name in ('native-submissions.json','long-submissions.json'):
   for item in v['jobs']:
    if 'job_id' in item:ids.append(item['job_id'])
    else:ids.extend([item['native']['job_id'],item['callback']['job_id']])
for group in ('hpc-contract','legacy','native-mapping','chemistry'):
 p=root/'cpu/payload'/(group+'.xml')
 if p.is_file():
  t=ET.parse(p);cases=t.findall('.//testcase');record['cpu_xml'][group]={'file':ref(p),'cases':len(cases),'failures':len(list(t.iter('failure'))),'errors':len(list(t.iter('error'))),'skips':len(list(t.iter('skipped'))),'problem_texts':[(x.text or '')[-2000:] for tag in ('failure','error') for x in t.iter(tag)]}
for parent in ('cpu','after-cpu'):
 for p in (root/parent).glob('*.*.log'):
  record.setdefault('log_tails',{})[str(p)]=p.read_text(errors='replace')[-6000:]
p=root/'cpu/payload/hpc-contract.log'
if p.is_file():record.setdefault('log_tails',{})[str(p)]=p.read_text(errors='replace')[-6000:]
jobs=json.loads((analysis/'jobs.json').read_text());ids.extend([*jobs['diagnostics'].values(),*jobs['diagnostic_captures'].values(),jobs['final']])
r=subprocess.run(['bjobs','-l',*dict.fromkeys(ids)],capture_output=True,text=True,timeout=30)
record['scheduler']={'returncode':r.returncode,'raw':r.stdout+r.stderr}
record['analysis_jobs']=jobs
print(json.dumps(record,indent=2))
PY
