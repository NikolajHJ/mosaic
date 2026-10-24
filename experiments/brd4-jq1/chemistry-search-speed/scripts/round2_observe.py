"""One read-only scheduler/artifact query and verified byte delivery, no retries."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-chemistry-speed-02'
REMOTE=r'''
import base64,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-search-speed-02')
jobs=json.loads((root/'jobs.json').read_text())
out={'captured_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'scheduler':{},'assessments':{},'tails':{},'files':{}}
for phase,job in jobs.items():
 q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
 out['scheduler'][phase]={'job_id':job,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
 for name in ['tests.log','benchmark.log']:
  p=root/phase/'payload'/name
  if p.is_file():out['tails'][phase+'/'+name]=p.read_text(errors='replace')[-12000:]
 p=root/phase/'payload/assessment.json'
 if p.is_file():out['assessments'][phase]=json.loads(p.read_text())
 p=root/phase/'gpu-submission-complete.json'
 if p.is_file():out['assessments'][phase]=json.loads(p.read_text())
if any(k in out['assessments'] for k in ['cpu','gpu']) or any('Status <EXIT>' in q['stdout'] for q in out['scheduler'].values()):
 total=0
 for p in root.rglob('*'):
  if not p.is_file():continue
  assert not p.is_symlink()
  if any(part in ['cuda-cache','cache','pytest-cache'] for part in p.relative_to(root).parts):continue
  assert p.stat().st_size<32*1024**2
  raw=p.read_bytes();total+=len(raw);assert total<128*1024**2
  out['files'][p.relative_to(root).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
print(json.dumps(out))
'''


def main():
    label=sys.argv[1];assert label.isdigit()
    out=LOCAL/('observation-'+label);out.mkdir(exist_ok=False)
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    r=subprocess.run(args,input=REMOTE.encode(),capture_output=True,timeout=150)
    (out/'raw.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr)
    if r.returncode:
        print('Connection/query failed',r.returncode);print(r.stderr.decode(errors='replace')[-1500:]);return
    d=json.loads(r.stdout)
    for name,row in d['files'].items():
        rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
        raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        p=out/'files'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    statuses={}
    for phase,row in d['scheduler'].items():
        found=re.search(r'Status<([^>]+)>',''.join(row['stdout'].split()));statuses[phase]={'job_id':row['job_id'],'status':found.group(1) if found else 'missing_live_record'}
    tails={k:v[-1800:] for k,v in d['tails'].items()}
    print(json.dumps({'captured_utc':d['captured_utc'],'jobs':statuses,
                      'assessment_status':{k:v.get('status') for k,v in d['assessments'].items()},
                      'tails':tails,'verified_delivery_files':len(d['files'])},indent=2)[:11000])


if __name__=='__main__':main()
