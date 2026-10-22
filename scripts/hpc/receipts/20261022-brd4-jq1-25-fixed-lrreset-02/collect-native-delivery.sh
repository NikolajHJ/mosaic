set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,tarfile
analysis=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=analysis/'restart-lrreset-02';source=analysis/'restart-lrreset-release-02/source'
dest=root/'native-delivery-01';archive=root/'native-delivery-01.tar.gz'
if dest.exists() or archive.exists():raise ValueError('Delivery exists or outcome unknown; do not duplicate')
dest.mkdir();files={};origins={}
def add(name,b,origin=None):
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 if origin:origins[name]=str(origin)
def include(name,p):
 if not p.is_file() or p.is_symlink() or p.resolve()!=p:raise ValueError('Missing/unsafe required proof '+str(p))
 add(name,p.read_bytes(),p)
for name in ('source-staging.json','cpu-certificate.json','native-submissions.json','native-dispatch-started.json','native-recovery-01/adoption.json','native-recovery-01/started.json','native-validation-recovery-01-quota/lrreset-storage-gate.json','native-validation-dispatch-quota/lrreset-storage-gate.json','readonly-worker-stop.json'):
 include('campaign/'+name,root/name)
include('source/source_manifest.json',source/'source_manifest.json')
jobs=json.loads((root/'native-submissions.json').read_text());ids=[]
for row in jobs['jobs']:
 for role in ('native','callback'):
  item=row[role];parent=Path(item['output_parent']);ids.append(item['job_id'])
  for name in ('submission.json','submission-started.json','scheduler-at-submission.txt'):
   include('campaign/'+parent.relative_to(root).as_posix()+'/'+name,parent/name)
 for name in ('decision-'+row['arm']+'.json',):
  p=root/name
  if p.is_file():include('campaign/'+name,p)
for name in ('backup-receipt.json','cleanup-plan.json','cleanup_verified_work3.py','attempt-02-operator-started.json','attempt-02/receipt.json','attempt-02/deletion-journal.jsonl','attempt-02/verified-candidates.json','attempt-02/jobs-before-verification.json','attempt-02/jobs-before-removal.json','attempt-02/quota-after.txt'):
 include('cleanup/'+name,root/'cleanup-recovery-02'/name)
old=json.loads((analysis/'jobs.json').read_text())
ids.extend([*old['diagnostics'].values(),*old['diagnostic_captures'].values(),old['final']])
r=subprocess.run(['bjobs','-l',*dict.fromkeys(ids)],capture_output=True,text=True,timeout=30)
add('scheduler-current.txt',(r.stdout+r.stderr).encode())
state={'utc':datetime.now(timezone.utc).isoformat(),'native_jobs':ids[:6],
 'new_scientific_long_submission_file_exists':(root/'long-submissions.json').exists(),
 'new_long_dispatch_marker_exists':(root/'long-dispatch-started.json').exists(),
 'production_output_parents_exist':{a:(Path('/zhome/95/e/206121/mosaic-runs/20261022-brd4-jq1-25-fixed-lrreset-02') if a=='boltz2_chemistry' else root).joinpath('run-'+a).exists() for a in ('protenix','boltz2','boltz2_chemistry')},'scheduler_returncode':r.returncode,'old_analysis_jobs':old,
 'scope':'Metadata/byte collection only; scientific GPU pass is not inferred from scheduler status'}
add('capture-state.json',(json.dumps(state,indent=2)+'\n').encode())
add('origin-map.json',(json.dumps(origins,indent=2)+'\n').encode())
(dest/'evidence_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
with archive.open('xb') as output:
 with tarfile.open(fileobj=output,mode='w:gz',compresslevel=3) as t:
  for p in sorted(dest.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(dest).as_posix(),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'utc':datetime.now(timezone.utc).isoformat()}))
PY
