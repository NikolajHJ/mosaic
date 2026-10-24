"""One compact read-only query and byte delivery of adoption03 plus parent02."""
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-gpu-adoption-03'
REMOTE = r'''
import base64,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
output={}
for label,name in [('adoption','chemistry-search-speed-gpu-adoption-03'),('parent','chemistry-search-speed-02')]:
 root=base/name;jobs=json.loads((root/'jobs.json').read_text())
 out={'captured_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'scheduler':{},'assessments':{},'tails':{},'files':{}}
 for phase,job in jobs.items():
  q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
  out['scheduler'][phase]={'job_id':job,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
  p=root/phase/'payload/assessment.json'
  if p.is_file():out['assessments'][phase]=json.loads(p.read_text())
  for log in ['tests.log','benchmark.log']:
   p=root/phase/'payload'/log
   if p.is_file():out['tails'][phase+'/'+log]=p.read_text(errors='replace')[-1600:]
 total=0
 for p in root.rglob('*'):
  if not p.is_file():continue
  assert not p.is_symlink()
  if any(part in ['cuda-cache','cache','pytest-cache'] for part in p.relative_to(root).parts):continue
  raw=p.read_bytes();total+=len(raw);assert len(raw)<32*1024**2 and total<128*1024**2
  out['files'][p.relative_to(root).as_posix()]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 output[label]=out
print(json.dumps(output))
'''


def main():
    label = sys.argv[1]; assert label.isdigit()
    out = LOCAL / ('observation-' + label); out.mkdir(exist_ok=False)
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    r = subprocess.run(args, input=REMOTE.encode(), capture_output=True, timeout=150)
    (out / 'raw.json').write_bytes(r.stdout); (out / 'stderr.txt').write_bytes(r.stderr)
    if r.returncode:
        print('Query failed', r.returncode, r.stderr.decode(errors='replace')[-2000:]); return
    payload = json.loads(r.stdout); summaries = {}
    for name, data in payload.items():
        folder = out / name; folder.mkdir()
        (folder / 'raw.json').write_text(json.dumps(data) + '\n', encoding='utf-8', newline='\n')
        for rel, row in data['files'].items():
            path = Path(rel); assert not path.is_absolute() and '..' not in path.parts
            raw = base64.b64decode(row['base64'])
            assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
            p = folder / 'files' / path; p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('xb') as f:
                f.write(raw)
        states = {}
        for phase, row in data['scheduler'].items():
            match = re.search(r'Status<([^>]+)>', ''.join(row['stdout'].split()))
            states[phase] = {'job_id': row['job_id'], 'status': match.group(1) if match else 'missing_live_record'}
        summaries[name] = {'captured_utc': data['captured_utc'], 'jobs': states,
                           'assessments': {k: v.get('status') for k, v in data['assessments'].items()},
                           'tails': data['tails'], 'verified_files': len(data['files'])}
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
