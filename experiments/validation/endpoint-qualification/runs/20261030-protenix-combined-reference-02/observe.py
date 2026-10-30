"""Read-only compact metadata/byte collection, never imports model libraries."""
import argparse,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
OWNER=Path(__file__).resolve().parents[2]
REPO=OWNER.parents[2]
TAG='20261030-protenix-combined-reference-02'
PROGRAM=r"""
from pathlib import Path
import base64,hashlib,json,subprocess,datetime,stat
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
scope=base/'nonchem-precision-continuation-26';root=scope/'chemistry-continuation-05/combined-reference-qualification-02';source=scope/'combined-reference-qualification-release-02/source'
def binding(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and all(not q.is_symlink() for q in (p,*p.parents))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'bytes':s.st_size,'sha256':h.hexdigest()}
files={};binary=[]
for folder in (source,root):
 for p in folder.rglob('*'):
  assert not p.is_symlink()
  if not p.is_file():continue
  b=binding(p)
  if p.suffix in ('.npz','.pt','.ckpt') or b['bytes']>4*1024**2:binary.append(b)
  else:b['base64']=base64.b64encode(p.read_bytes()).decode();files[str(p)]=b
m=json.loads((source/'source_manifest.json').read_bytes());plan=json.loads((source/'plan.json').read_bytes());sources=[]
for row in m['parent_sources']:
 path=Path(row['path']);manifest=binding(path/'source_manifest.json');assert manifest['sha256']==row['manifest_sha256']
 members=json.loads((path/'source_manifest.json').read_bytes())['files'];rows=[]
 for name,h in members.items():
  b=binding(path/name);assert b['sha256']==h;rows.append(b)
 sources.append({'manifest':manifest,'members':rows})
inputs=[]
for row in plan['inputs']:
 b=binding(row['path']);assert b==row;inputs.append(b)
submissions={p.parent.name:json.loads(p.read_bytes()) for p in root.glob('*/submission.json')}
commands=[]
for a in [['bjobs','-u','s234818','-json','-o','jobid stat queue job_name command'],*[['bjobs','-a','-l',v['job_id']] for v in submissions.values()]]:
 r=subprocess.run(a,capture_output=True,text=True);commands.append({'argv':a,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'binary_bindings_remote_only':binary,'parent_sources':sources,'inputs_rehashed':inputs,'submissions':submissions,'commands':commands,'registry':json.loads((scope/'jobs.json').read_bytes()),'mutations':False}))
"""
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--label',required=True);args=parser.parse_args()
 out=REPO/'results/hpc'/TAG/args.label;assert not out.exists();out.mkdir(parents=True)
 (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
 command='python3 -B -c '+shlex.quote(PROGRAM)
 r=subprocess.run([r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe','exec','--timeout','180','bash -lc '+shlex.quote(command)],capture_output=True,timeout=200)
 (out/'stdout.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr)
 (out/'transport.json').write_text(json.dumps({'returncode':r.returncode})+'\n')
 assert r.returncode==0,r.stderr.decode();value=json.loads(r.stdout)
 print(json.dumps({'output':str(out),'observed_utc':value['observed_utc'],'jobs':{k:v['job_id'] for k,v in value['submissions'].items()},'files':len(value['files'])}))
