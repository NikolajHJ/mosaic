"""Single SSH metadata/scheduler query; no scientific calculations or imports."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

RECEIPTS = Path(__file__).resolve().parent
TAG = RECEIPTS.name
CODE = r'''
from datetime import datetime,timezone
from pathlib import Path
import subprocess,json,hashlib,re,xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/TAG_VALUE')
home=Path('/zhome/95/e/206121/mosaic-runs/TAG_VALUE')
def hashed(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
records={};parents=[root/'cpu',root/'after-cpu']
for phase in ('validate','after-ready'):
 for arm in ('protenix','boltz2','boltz2_chemistry'):parents.append(root/(phase+'-'+arm))
for arm in ('protenix','boltz2','boltz2_chemistry'):parents.append((home if arm=='boltz2_chemistry' else root)/('run-'+arm))
for parent in parents:
 path=parent/'submission.json'
 if path.exists():records[str(parent)]={'submission':json.loads(path.read_text()),'submission_proof':hashed(path)}
ids=[v['submission']['job_id'] for v in records.values() if v['submission'].get('outcome')=='submitted']
scheduler=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=40)
out={'utc':datetime.now(timezone.utc).isoformat(),'records':records,'scheduler':{'job_ids':ids,'returncode':scheduler.returncode,'stdout':scheduler.stdout,'stderr':scheduler.stderr},'artifacts':{},'xml':{}}
for name in ('source-staging.json','initial-submissions-complete.json','cpu-certificate.json','native-submissions.json','long-submissions.json','long-dispatch-started.json','decision-protenix.json','decision-boltz2.json','decision-boltz2_chemistry.json'):
 path=root/name
 if path.exists():out['artifacts'][name]={'record':json.loads(path.read_text()),'proof':hashed(path)}
for parent,record in records.items():
 path=Path(parent)
 record['logs']={}
 for stem in ('stdout','stderr'):
  log=path/(stem+'.'+record['submission'].get('job_id','')+'.log')
  if log.exists():record['logs'][stem]={'bytes':log.stat().st_size,'tail':log.read_text(errors='replace')[-3500:]}
 study=path/'payload/study'
 if study.exists():
  record['study']={}
  for name in ('states.jsonl','updates.jsonl'):
   file=study/name
   if file.exists():
    rows=0;last=None
    with file.open('rb') as stream:
     for line in stream:
      if not line.endswith(b'\n'):break
      json.loads(line);rows+=1;last=line.decode()[-800:]
    record['study'][name]={'committed_rows':rows,'last_suffix':last,'bytes_observed':file.stat().st_size}
  for name in ('assessment.json','summary.json'):
   file=study/name
   if file.exists():record['study'][name]=hashed(file)
for file in (root/'cpu/payload').glob('*.xml'):
 cases=ET.parse(file).findall('.//testcase')
 out['xml'][file.name]={'proof':hashed(file),'cases':len(cases),'failed':sum(c.find('failure') is not None for c in cases),'errors':sum(c.find('error') is not None for c in cases),'skipped':sum(c.find('skipped') is not None for c in cases),'modules':sorted({c.get('classname') for c in cases})}
for file in (root/'cpu/payload').glob('*.log'):
 out.setdefault('cpu_logs',{})[file.name]=file.read_text(errors='replace')[-1800:]
print(json.dumps(out))
'''.replace('TAG_VALUE', TAG)
script = "set -euo pipefail\n/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'\n" + CODE + "\nPY\n"
index=1
while (RECEIPTS/f'status-{index:02d}.json').exists():index+=1
result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -s'],input=script.encode(),capture_output=True,timeout=75)
(RECEIPTS/f'status-{index:02d}-stdout.txt').write_bytes(result.stdout)
(RECEIPTS/f'status-{index:02d}-stderr.txt').write_bytes(result.stderr)
record={'connection_returncode':result.returncode,'local_utc':datetime.now(timezone.utc).isoformat()}
if result.returncode==0:
 lines=[line for line in result.stdout.decode().splitlines() if line.startswith('{')]
 assert len(lines)==1
 record['remote']=json.loads(lines[0])
output=RECEIPTS/f'status-{index:02d}.json'
output.write_bytes((json.dumps(record,indent=2)+'\n').encode())
print('Saved',output.name,'connection_rc',result.returncode)
if 'remote' in record:
 remote=record['remote'];print('UTC',remote['utc'])
 import re
 compact=re.sub(r'\s+','',remote['scheduler']['stdout'])
 print('Jobs/status:',re.findall(r'Job<([^>]+)>.*?Status<([^>]+)>',compact))
 print('XML:',json.dumps({n:{k:v for k,v in d.items() if k not in ('proof','modules')} for n,d in remote['xml'].items()}))
 print('Artifacts:',list(remote['artifacts']))
 for n,d in remote.get('cpu_logs',{}).items():print(n,d[-600:])
else:print(result.stderr.decode(errors='replace')[-800:])
