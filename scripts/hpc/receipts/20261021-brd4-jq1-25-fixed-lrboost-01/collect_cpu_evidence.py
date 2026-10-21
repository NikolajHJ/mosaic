"""Archive/download actual CPU and launch metadata; no scientific evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

P = Path(__file__).resolve().parent
TAG = P.name
REPO = P.parents[3]
OUT = REPO / 'results/hpc' / TAG
SSH = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc']
CODE = r'''
from pathlib import Path
import subprocess,json,hashlib,re,tarfile
root=Path('/work3/s234818/mosaic/runs/TAG_VALUE')
source=Path('/work3/s234818/mosaic/releases/TAG_VALUE/source')
manifest='9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
def digest(b):return hashlib.sha256(b).hexdigest()
records={}
for role,job in [('cpu','29778388'),('after-cpu','29778391')]:
 r=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
 raw=r.stdout+r.stderr;c=re.sub(r'\s+','',raw)
 assert r.returncode==0 and re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c) in (['DONE'],['EXIT'])
 records[role]=(job,raw)
dest=root/'cpu-evidence-01';assert not dest.exists();dest.mkdir()
files={};origin={}
def add_bytes(name,b,remote=None):
 path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(b)
 files[name]={'bytes':len(b),'sha256':digest(b)}
 if remote:origin[name]=str(remote)
def add(path):
 path=Path(path);assert path.is_file() and not path.is_symlink()
 if path.is_relative_to(root):name='campaign/'+path.relative_to(root).as_posix()
 elif path.is_relative_to(source):name='source/'+path.relative_to(source).as_posix()
 else:raise ValueError('Unexpected CPU proof path')
 if name not in files:add_bytes(name,path.read_bytes(),path)
for role,(job,raw) in records.items():
 add_bytes('scheduler-'+role+'.txt',raw.encode())
 for name in ('submission.json','submission-started.json','scheduler-at-submission.txt','stdout.'+job+'.log','stderr.'+job+'.log'):
  path=root/role/name
  if path.exists():add(path)
for name in ('source-staging.json','initial-submissions-complete.json','cpu-certificate.json','native-submissions.json','native-dispatch-started.json'):
 path=root/name
 if path.exists():add(path)
cert=json.loads((root/'cpu-certificate.json').read_text())
assert cert['status']=='passed' and cert['producer_job_id']=='29778388' and cert['source_manifest_sha256']==manifest
for proof in cert['proofs']:
 path=Path(proof['path']);assert digest(path.read_bytes())==proof['sha256'];add(path)
for path in (root/'cpu/payload').glob('*.log'):add(path)
for name in ('preflight-quota','cpu-entry-quota','native-validation-dispatch-quota'):
 for path in (root/name).glob('*.json'):add(path)
if (root/'native-submissions.json').exists():
 for row in json.loads((root/'native-submissions.json').read_text())['jobs']:
  for role in ('native','callback'):
   record=row[role];parent=Path(record['output_parent'])
   for name in ('submission.json','submission-started.json','scheduler-at-submission.txt'):
    if (parent/name).exists():add(parent/name)
add(source/'source_manifest.json')
assert digest((source/'source_manifest.json').read_bytes())==manifest
for arm in ('protenix','boltz2','boltz2_chemistry'):add(source/('examples/configs/brd4_jq1_25_fixed_lrboost_'+arm+'_10000.json'))
add_bytes('origin-map.json',(json.dumps(origin,indent=2)+'\n').encode())
(dest/'evidence_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
archive=root/'cpu-evidence-01.tar.gz'
with archive.open('xb') as output:
 with tarfile.open(fileobj=output,mode='w:gz') as tar:
  for path in sorted(dest.rglob('*')):
   if path.is_file():tar.add(path,arcname=path.relative_to(dest).as_posix(),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':digest(archive.read_bytes()),'files':len(files)}))
'''.replace('TAG_VALUE',TAG)
script="set -euo pipefail\n/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'\n"+CODE+"\nPY\n"
(P/'cpu-evidence-command-01.sh').write_bytes(script.encode())
with (P/'cpu-evidence-started-01.json').open('x') as f:json.dump({'scope':'one metadata proof archive, no retries/duplicate if unknown'},f)
r=subprocess.run(SSH+['bash -l -s'],input=script.encode(),capture_output=True,timeout=90)
(P/'cpu-evidence-stdout-01.txt').write_bytes(r.stdout);(P/'cpu-evidence-stderr-01.txt').write_bytes(r.stderr)
if r.returncode:
 print(r.stderr.decode(errors='replace')[-2200:]);raise SystemExit(r.returncode)
rows=[json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith('{')];assert len(rows)==1
receipt=rows[0];(P/'cpu-evidence-archive.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
archive=OUT/'cpu-evidence-01.tar.gz';assert not archive.exists()
with archive.open('xb') as f:
 download=subprocess.run(SSH+['cat '+receipt['path']],stdout=f,stderr=subprocess.PIPE,timeout=90)
assert download.returncode==0 and archive.stat().st_size==receipt['bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
subprocess.run([sys.executable,'scripts/verify_hpc_evidence.py',str(archive),receipt['sha256']],cwd=REPO,check=True)
print(json.dumps(receipt))
