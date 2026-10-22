set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,hashlib,tarfile,io,re
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01/source')
manifest='214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
def digest(b):return hashlib.sha256(b).hexdigest()
dest=root/'cpu-evidence-01';dest.mkdir(exist_ok=False);files={};origin={}
def add_bytes(name,data,original=None):
 assert name not in files
 p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 files[name]={'bytes':len(data),'sha256':digest(data)}
 if original:origin[name]=str(original)
def add(p):
 p=Path(p);assert p.is_file() and not p.is_symlink()
 if p.is_relative_to(root):name='analysis/'+p.relative_to(root).as_posix()
 elif p.is_relative_to(source):name='source/'+p.relative_to(source).as_posix()
 else:raise ValueError('Unexpected new proof path '+str(p))
 if name not in files:add_bytes(name,p.read_bytes(),p)
for role,job in [('cpu','29780814'),('after-cpu','29780815')]:
 r=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=r.stdout+r.stderr;c=re.sub(r'\s+','',raw)
 assert r.returncode==0 and re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c)==['DONE'] and re.findall(r'Queue<([^>]+)>',c)==['hpc']
 add_bytes('scheduler-'+role+'.txt',raw.encode())
 for name in ('submission.json','submission-started.json','scheduler-at-submission.txt','stdout.'+job+'.log','stderr.'+job+'.log'):
  if (root/role/name).exists():add(root/role/name)
cert=json.loads((root/'cpu-certificate.json').read_text())
assert cert['status']=='passed' and cert['producer_job_id']=='29780814' and cert['tests']==278 and cert['source_manifest_sha256']==manifest
for proof in cert['proofs']:
 p=Path(proof['path']);assert digest(p.read_bytes())==proof['sha256'];add(p)
for name in ('source-staging.json','initial-submissions.json','cpu-certificate.json','jobs.json','analysis-dispatch-started.json','storage-preflight/gate.json','cpu/storage-cpu/gate.json','after-cpu/storage-dispatch/gate.json'):
 add(root/name)
jobs=json.loads((root/'jobs.json').read_text())
for parent in ['diagnostic-protenix','diagnostic-boltz2','diagnostic-capture-protenix','diagnostic-capture-boltz2',*('complete-'+a for a in jobs['collectors']),'interim-0001','combined-report']:
 for name in ('submission.json','submission-started.json','scheduler-at-submission.txt'):
  p=root/parent/name
  if p.exists():add(p)
 p=root/parent
 for gate in p.glob('storage-*/gate.json'):add(gate)
add(source/'source_manifest.json');assert digest((source/'source_manifest.json').read_bytes())==manifest
add_bytes('origin-map.json',(json.dumps(origin,indent=2)+'\n').encode())
manifest_bytes=(json.dumps({'files':files},indent=2)+'\n').encode()
(dest/'evidence_manifest.json').write_bytes(manifest_bytes)
archive=root/'cpu-evidence-01.tar.gz'
with archive.open('xb') as out:
 with tarfile.open(fileobj=out,mode='w:gz') as t:
  for p in sorted(dest.rglob('*')):
   if p.is_file():t.add(p,arcname=p.relative_to(dest).as_posix(),recursive=False)
receipt={'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive.read_bytes()),'files':len(files)}
(root/'cpu-evidence-01.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
PY
