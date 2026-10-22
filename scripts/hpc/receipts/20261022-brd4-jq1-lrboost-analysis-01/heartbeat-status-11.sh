set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,datetime,hashlib,xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
jobs=json.loads((root/'jobs.json').read_text()); ids=[jobs['cpu'],'29780815','29779179']
for field in ('diagnostics','diagnostic_captures','collectors'):
 ids.extend(jobs.get(field,{}).values())
for field in ('active_interim','final'):
 if jobs.get(field):ids.append(jobs[field])
r=subprocess.run(['bjobs','-l',*dict.fromkeys(ids)],capture_output=True,text=True,timeout=30)
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'scheduler':{'returncode':r.returncode,'raw':r.stdout+r.stderr},'artifacts':{}}
for name in ('source-staging.json','initial-submissions.json','cpu-certificate.json','analysis-dispatch-started.json','after-cpu/storage-dispatch/gate.json','interim-0002-dispatch-guard.json','interim-0002/storage-interim/gate.json'):
 p=root/name
 if p.exists():record['artifacts'][name]={'content':json.loads(p.read_text()),'file':ref(p)}
p=root/'cpu/payload/report-tests.xml'
if p.exists():
 t=ET.parse(p);cases=list(t.iter('testcase'));record['cpu_xml']={'cases':len(cases),'failures':len(list(t.iter('failure'))),'errors':len(list(t.iter('error'))),'skips':len(list(t.iter('skipped'))),'file':ref(p)}
 for tag in ('failure','error'):
  if list(t.iter(tag)):record['cpu_xml'][tag+'_texts']=[q.text for q in t.iter(tag)]
for role in ('cpu','after-cpu'):
 p=root/role/'payload/report-tests.log' if role=='cpu' else root/role/'stderr.29780815.log'
 if p.exists():record[role+'_log_tail']=p.read_text(errors='replace')[-12000:]
for name in [*('diagnostic-'+a for a in ('protenix','boltz2')),*('diagnostic-capture-'+a for a in ('protenix','boltz2')),*('complete-'+a for a in ('protenix','boltz2','boltz2_chemistry')),'interim-0001','interim-0002','combined-report']:
 p=root/name/'payload/assessment.json'
 if p.exists():record['artifacts'][name]={'file':ref(p),'content':json.loads(p.read_text()) if p.stat().st_size<200000 else {'scope':'large assessment referenced only'}}
 for child in ('submission.json',):
  p=root/name/child
  if p.exists():record['artifacts'][name+'/'+child]={'content':json.loads(p.read_text()),'file':ref(p)}
 p=root/(name+'.tar.receipt.json')
 if p.exists():record['artifacts'][name+'/archive']={'content':json.loads(p.read_text()),'file':ref(p)}
chem=Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2_chemistry/payload/study')
record['chemistry_saved_prefix']={}
for kind in ('states','updates'):
 p=chem/'case-0'/(kind+'.jsonl')
 if p.exists():
  count=0; last=None; remainder=False
  with p.open('rb') as handle:
   for row in handle:
    if not row.endswith(b'\n'):remainder=True;break
    item=json.loads(row);count+=1;last={k:item[k] for k in ('step','state_index','update_index','index','applied_update') if k in item}
  record['chemistry_saved_prefix'][kind]={'newline_committed_rows':count,'last_index_fields':last,'uncommitted_tail_ignored':remainder,'bytes_observed':p.stat().st_size}
p=chem/'summary.json'
if p.exists():
 v=json.loads(p.read_text());record['chemistry_saved_prefix']['summary']={'file':ref(p),'status':v.get('status'),'keys':list(v),'errors':v.get('errors')}
print(json.dumps(record,indent=2))
PY
