set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,tarfile
analysis=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=analysis/'restart-lrreset-02';source=analysis/'restart-lrreset-release-02/source'
dest=root/'verification-evidence-01';archive=root/'verification-evidence-01.tar.gz'
if dest.exists() or archive.exists():raise ValueError('Evidence outcome exists/unknown; inspect before recovery')
if not (root/'cpu-certificate.json').exists():raise ValueError('CPU certification has not completed')
if not (root/'native-validation-dispatch-quota/lrreset-storage-gate.json').exists():raise ValueError('Native dispatch quota has not completed')
dest.mkdir();files={};origins={}
def add(name,b,origin=None):
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(b)
 files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
 if origin:origins[name]=str(origin)
def include(name,p):
 if p.is_file():
  if p.is_symlink() or p.resolve()!=p:raise ValueError('Unsafe proof path')
  add(name,p.read_bytes(),p)
def parent_files(prefix,p,job):
 for name in ('submission.json','submission-started.json','scheduler-at-submission.txt',f'stdout.{job}.log',f'stderr.{job}.log'):
  include(prefix+'/'+name,p/name)
 for q in p.glob('scheduler-final-*.txt'):include(prefix+'/'+q.name,q)
for attempt,cpu,callback in [('01','29781350','29781351'),('02','29781363','29781364')]:
 r=analysis/('restart-lrreset-'+attempt)
 parent_files(attempt+'/cpu',r/'cpu',cpu);parent_files(attempt+'/after-cpu',r/'after-cpu',callback)
 for name in ('source-staging.json','initial-submissions-complete.json','cpu-certificate.json','native-submissions.json','native-dispatch-started.json','long-submissions.json'):
  include(attempt+'/'+name,r/name)
 for q in (r/'cpu/payload').iterdir():
  if q.is_file() and q.suffix in ('.xml','.log','.json'):include(attempt+'/cpu/payload/'+q.name,q)
 for phase in ('preflight-quota','cpu-entry-quota','native-validation-dispatch-quota'):
  for q in (r/phase).glob('*.json'):include(attempt+'/'+phase+'/'+q.name,q)
 s=analysis/('restart-lrreset-release-'+attempt)/'source'
 include(attempt+'/source/source_manifest.json',s/'source_manifest.json')
 for arm in ('protenix','boltz2','boltz2_chemistry'):
  n=f'examples/configs/brd4_jq1_25_fixed_lrreset_{arm}_10000.json';include(attempt+'/source/'+n,s/n)
 for role,job in [('cpu',cpu),('after-cpu',callback)]:
  rj=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
  add(attempt+'/scheduler-current-'+role+'.txt',(rj.stdout+rj.stderr).encode())
add('origin-map.json',(json.dumps(origins,indent=2)+'\n').encode())
(dest/'evidence_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
with archive.open('xb') as output:
 with tarfile.open(fileobj=output,mode='w:gz') as t:
  for q in sorted(dest.rglob('*')):
   if q.is_file():t.add(q,arcname=q.relative_to(dest).as_posix(),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'utc':datetime.now(timezone.utc).isoformat()}))
PY
