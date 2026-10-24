"""One compact read-only query and byte-authenticated final delivery, no retries."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-01'
CODE = r'''
import base64,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-search-speed-01')
p=root/'cpu/submission.json'
assert p.is_file(),'No known submission; inspect staging evidence'
submission=json.loads(p.read_text());job=submission['job_id']
q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'job_id':job,'scheduler':q.stdout,'scheduler_stderr':q.stderr,
        'scheduler_rc':q.returncode,'files':{},'tail':{}}
for rel in ['cpu/payload/tests.log','cpu/payload/benchmark.log']:
 p=root/rel
 if p.is_file():result['tail'][rel]=p.read_text(errors='replace')[-10000:]
p=root/'cpu/payload/assessment.json'
result['assessment']=json.loads(p.read_text()) if p.is_file() else None
if result['assessment'] is not None:
 # Original inputs/source remain untouched. Return all small original evidence,
 # including failed fixture logs; profiles are analysis outputs, not native data.
 paths=[root/'plan.json',root/'source.tar.gz']
 paths+=list((root/'cpu').rglob('*'))+list((root/'preflight').rglob('*'))
 total=0
 for p in paths:
  if not p.is_file():continue
  assert not p.is_symlink() and p.stat().st_size<32*1024**2
  raw=p.read_bytes();total+=len(raw);assert total<128*1024**2
  result['files'][str(p.relative_to(root))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 plan=json.loads((root/'plan.json').read_text())
 for row in plan['inputs']:
  p=Path(row['path']);assert p.is_file() and not p.is_symlink()
  raw=p.read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  result['files']['input-bytes/'+row['label']+'.npz']={'bytes':len(raw),'sha256':row['sha256'],'base64':base64.b64encode(raw).decode()}
print(json.dumps(result))
'''


def main():
    label = sys.argv[1]
    assert label.isdigit()
    out = LOCAL / ('observation-' + label)
    out.mkdir(exist_ok=False)
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    r=subprocess.run(args,input=CODE.encode(),capture_output=True,timeout=120)
    (out/'raw.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr)
    if r.returncode:
        print('SSH/query failed',r.returncode);print(r.stderr.decode(errors='replace')[-1500:]);return
    data=json.loads(r.stdout)
    for rel, row in data['files'].items():
        p=Path(rel)
        assert not p.is_absolute() and '..' not in p.parts
        raw=base64.b64decode(row['base64'])
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        target=out/'files'/p;target.parent.mkdir(exist_ok=True,parents=True)
        with target.open('xb') as f:f.write(raw)
    status=re.search(r'Status <([^>]+)>',data['scheduler'])
    recent=[]
    for line in data['tail'].get('cpu/payload/benchmark.log','').splitlines():
        if line.startswith('{'):
            recent.append(json.loads(line))
    print(json.dumps({'captured_utc':data['captured_utc'],'job_id':data['job_id'],
                     'status':status.group(1) if status else 'missing_live_record',
                     'test_log_tail':data['tail'].get('cpu/payload/tests.log','')[-1400:],
                     'recent_benchmark_rows':recent[-4:],
                     'assessment':data['assessment']},indent=2)[:10000])
    print('Verified transferred files:',len(data['files']))


if __name__ == '__main__':
    main()
