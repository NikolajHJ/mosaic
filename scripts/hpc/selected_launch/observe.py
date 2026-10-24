"""One bounded read-only selected-launch scheduler/proof delivery; stdlib only."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'results/hpc/20261025-nonchem-stride20-startup-03'
REMOTE = r'''
import base64,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=base/'nonchem-stride20-startup-release-03/source';control=base/'nonchem-stride20-02/startup-fix-03'
sys.path.insert(0,str(source))
import selected_contract as c
c.verify_own(MANIFEST)
roots=[control,*[c.profile(k)['root'] for k in c.PROFILES]]
files={};total=0
def add(p):
 global total
 if not p.is_file() or p.is_symlink():return
 if str(p) in files:return
 raw=p.read_bytes();assert len(raw)<=12*1024**2
 total+=len(raw);assert total<=64*1024**2
 files[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
parents=[];jobs=[]
for root in roots:
 for p in sorted(root.iterdir()):
  if not p.is_dir() or p.is_symlink() or not (p/'submission.json').is_file():continue
  sub=c.read(p/'submission.json')
  if sub.get('launcher_manifest_sha256')!=MANIFEST:continue
  parents.append(p);jobs.append(sub['job_id'])
 for rel in ['jobs.json','cpu-certificate.json','initial-submissions-complete.json','source-staging.json','projection.json','nonchem-stride20-v3-owner.json','cpu-ready.json']:
  add(root/rel)
for p in [control/'preflight',control/'post-stage',*parents]:
 if not p.exists():continue
 for q in sorted(p.rglob('*')):
  if q.is_file() and not q.is_symlink() and not any(k in q.parts for k in ['cache','.pytest_cache','snapshots']) and q.suffix in ['.json','.xml','.log','.txt']:
   # Scientific full summaries/frames stay remote, not in this compact delivery.
   if 'payload/study' in str(q):continue
   add(q)
for name in c.PROFILES:
 root=c.profile(name)['root'];prefix='pilot' if name=='dbmol10k' else c.SELECTED[name]
 for q in root.glob('startup-'+prefix+'*'):add(q)
 for q in root.glob('*command*'+prefix+'*'):add(q)
 for q in root.glob(prefix+'-stride20-v3-*.json'):add(q)
for root in [Path('/work3/s234818/mosaic/runs/20261024-protenix-noisy-04'),c.NOISE_ROOT]:
 ready=root/'cpu-ready.json';add(ready)
 if ready.is_file():
  for row in c.read(ready)['proofs']:
   q=Path(row['path'])
   if q.suffix in ['.xml','.json','.log','.txt']:add(q)
 for rel in ['gpu/summary.json','gpu/runtime-gate.json','gpu/trajectory/result.json','gpu/trajectory/retained-frames.json']:
  add(root/rel)
old_noise=base/'nonchem-stride20-02/noise-gpu'
if (old_noise/'submission.json').is_file():
 add(old_noise/'submission.json');jobs.append(c.read(old_noise/'submission.json')['job_id'])
 for q in old_noise.glob('*.log'):add(q)
 parents.append(old_noise)
native={name:c.native_units(name) for name in c.PROFILES}
prefixes={}
for p in parents:
 noisy=p.name=='noise-gpu'
 case=c.NOISE_ROOT/'gpu/trajectory' if noisy else p/'payload/study/case-0'
 row={};saved=[]
 for field in ['states','updates']:
  q=case/(field+'.jsonl');count=0;last=None
  if q.is_file():
   with q.open() as f:
    for line in f:
     if not line.endswith('\n'):break
     value=json.loads(line);assert (value['update']-1 if noisy and field=='updates' else value['step'])==count
     count+=1;last={key:value.get(key) for key in ['step','update','global_step','snapshot','snapshot_sha256']}
     if field=='states' and value.get('snapshot_retained'):
      anchor=c.NOISE_ROOT/'gpu' if noisy else case
      frame=anchor/value['snapshot'];assert frame.is_file() and not frame.is_symlink() and c.sha(frame)==value['snapshot_sha256']
      saved.append({'step':value['step'],'path':str(frame),'bytes':frame.stat().st_size,'sha256':value['snapshot_sha256']})
  row[field]={'committed_rows':count,'last':last}
 prefixes[str(p)]=row
 row['actual_retained_frames_byte_verified']=saved
 for q in p.glob('startup-*.json'):add(q)
q=subprocess.run(['bjobs','-l',*sorted(set(jobs))],capture_output=True,text=True,timeout=45)
result={'utc':datetime.now(timezone.utc).isoformat(),'manifest_sha256':MANIFEST,'files':files,'native_units':native,
 'scheduler':{'args':q.args,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'prefixes':prefixes,
 'source_seal_verified':True,'local_scientific_arrays_downloaded':False,'mutation':False}
print(json.dumps(result,allow_nan=False))
'''


def main():
    label = sys.argv[1]
    assert label.isdigit()
    out = LOCAL / ('evidence-' + label)
    out.mkdir(exist_ok=False)
    prep = json.loads((LOCAL / 'source02/preparation.json').read_text())
    code = 'MANIFEST=' + repr(prep['manifest_sha256']) + '\n' + REMOTE
    (out / 'query.py').write_text(code, encoding='utf-8', newline='\n')
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc',
            'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'stdout.bin').open('xb') as stdout, (out/'stderr.bin').open('xb') as stderr:
        result = subprocess.run(args,input=code.encode(),stdout=stdout,stderr=stderr,timeout=240)
    assert result.returncode == 0, (out/'stderr.bin').read_text(errors='replace')[-5000:]
    raw=(out/'stdout.bin').read_bytes();data=json.loads(raw)
    for index,(path,rec) in enumerate(data['files'].items()):
        content=base64.b64decode(rec.pop('base64'))
        assert len(content)==rec['bytes'] and hashlib.sha256(content).hexdigest()==rec['sha256']
        target=out/'files'/format(index,'04d')/Path(path).name
        target.parent.mkdir(parents=True)
        target.write_bytes(content)
        rec['local_path']=target.relative_to(out).as_posix()
    (out/'metadata.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    receipt={'status':'compact_bytes_verified_only','utc':data['utc'],'files':len(data['files']),
             'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),
             'source_manifest_sha256':data['manifest_sha256'],'scientific_array_calculation':False}
    (out/'delivery.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt,indent=2))
    import re
    for block in data['scheduler']['stdout'].split('Job <')[1:]:
        compact=re.sub(r'\s+','',block)
        print(compact.split('>',1)[0],re.findall(r'Status<([^>]+)>',compact))


if __name__=='__main__':main()
