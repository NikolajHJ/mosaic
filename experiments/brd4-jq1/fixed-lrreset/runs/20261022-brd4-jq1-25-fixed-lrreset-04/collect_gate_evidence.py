"""Byte-only current CPU/native/entry proof delivery; no numerical evaluation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

REPO = Path(__file__).resolve().parents[5]
REMOTE = r'''
import base64,hashlib,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
mode=sys.argv[1]
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04')
source=r.parent/'restart-lrreset-release-04/source'
home=Path('/zhome/95/e/206121/mosaic-runs/20261022-brd4-jq1-25-fixed-lrreset-04')
paths=set();refs={};ids=['29792389','29792390']
prefixes={}
placements={}
p=r/'long-submissions.json'
if p.exists():placements={v['arm']:Path(v['output_parent']) for v in json.loads(p.read_bytes())['jobs']}
def add(p):
 p=Path(p)
 if p.is_file() and not p.is_symlink():paths.add(p)
def scan(x):
 if isinstance(x,dict):
  if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):refs[x['path']]=x['sha256']
  if isinstance(x.get('job_id'),str):ids.append(x['job_id'])
  for v in x.values():scan(v)
 elif isinstance(x,list):
  for v in x:scan(v)
for name in ['source-staging.json','initial-submissions-complete.json','cpu-certificate.json','native-submissions.json','long-submissions.json','long-dispatch-started.json']:
 p=r/name
 if p.exists():add(p);scan(json.loads(p.read_bytes()))
for parent in [r/'cpu',r/'after-cpu']:
 for p in parent.glob('*'):
  if p.suffix in ['.json','.txt','.log']:add(p)
for p in [source/'source_manifest.json',source/'scripts/protenix_initialization.py',source/'examples/brd4_25_atoms.py',source/'src/mosaic/small_molecule/chemistry/trajectory.py']:
 add(p)
for p in source.glob('examples/configs/brd4_jq1_25_fixed_lrreset_*.json'):add(p)
for p in r.glob('*quota*/*storage-gate.json'):add(p)
for control in sorted(r.glob('all-work3-launch-*')):
 if not control.is_dir():continue
 for p in control.rglob('*'):
  if p.is_file() and p.suffix in ['.json','.xml','.txt','.log']:
   add(p)
   if p.suffix=='.json':scan(json.loads(p.read_bytes()))
for launcher in r.parent.glob('restart-lrreset-allwork3-launch-*/source'):
 for p in launcher.glob('*'):add(p)
if mode!='cpu':
 for arm in ['protenix','boltz2','boltz2_chemistry']:
  for parent in [r/('validate-'+arm),r/('after-ready-'+arm), placements.get(arm,(home if arm=='boltz2_chemistry' else r)/('run-'+arm))]:
   for p in parent.glob('*'):
    if p.suffix in ['.json','.txt','.log']:add(p)
   for p in parent.glob('**/*storage-gate.json'):add(p)
   for p in parent.glob('payload/driver-invocation.json'):add(p)
  parent=placements.get(arm,(home if arm=='boltz2_chemistry' else r)/('run-'+arm))
  prefixes[arm]={}
  for label in ['states','updates']:
   p=parent/('payload/study/case-0/'+label+'.jsonl')
   if p.exists():
    count=0;h=hashlib.sha256();last=None
    for line in p.open('rb'):
     if not line.endswith(b'\n'):break
     row=json.loads(line);assert row['step']==count;h.update(line);last=row;count+=1
    prefixes[arm][label]={'path':str(p),'committed_rows':count,'committed_bytes_sha256':h.hexdigest(),'last':last,'contiguous':True}
  for p in [r/('decision-'+arm+'.json'),r/('validate-'+arm)/'payload/study/assessment.json',r/('validate-'+arm)/'payload/study/summary.json']:
   if p.exists():add(p);scan(json.loads(p.read_bytes()))
raw_rehash=[]
for path,expected in refs.items():
 p=Path(path);assert str(p).startswith('/work3/s234818/mosaic/') or str(p).startswith('/zhome/95/e/206121/mosaic-runs/')
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected
 raw_rehash.append({'path':path,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 if p.stat().st_size<=64*1024**2:add(p)
rows=[];total=0
for p in sorted(paths):
 b=p.read_bytes();total+=len(b);assert total<256*1024**2
 name=('source/'+p.relative_to(source).as_posix()) if p.is_relative_to(source) else ('work/'+p.relative_to(r).as_posix()) if p.is_relative_to(r) else ('home/'+p.relative_to(home).as_posix()) if p.is_relative_to(home) else ('external/'+hashlib.sha256(str(p).encode()).hexdigest()+'/'+p.name)
 rows.append({'path':name,'remote_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
s=subprocess.run(['bjobs','-l']+list(dict.fromkeys(ids)),text=True,capture_output=True,timeout=30)
print('EVIDENCE_JSON_BEGIN')
print(json.dumps({'mode':mode,'captured_utc':datetime.now(timezone.utc).isoformat(),'files':rows,'remote_rehash':raw_rehash,'committed_prefixes':prefixes,'scheduler':{'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['cpu', 'native', 'start'], required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert args.name.isascii() and all(c.isalnum() or c == '-' for c in args.name)
    folder = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04' / args.name
    folder.mkdir(parents=True, exist_ok=False)
    # The phase is a fixed single token from argparse; the script arrives on stdin.
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c 'python3 - " + args.mode + "'"]
    result = subprocess.run(ssh, input=REMOTE.encode(), capture_output=True, timeout=180)
    (folder / 'ssh-stdout.bin').write_bytes(result.stdout); (folder / 'ssh-stderr.bin').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-1800:]
    report = json.loads(result.stdout.split(b'EVIDENCE_JSON_BEGIN\n', 1)[1])
    seen = set()
    for row in report['files']:
        rel = PurePosixPath(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen
        seen.add(str(rel))
        raw = base64.b64decode(row.pop('base64'), validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        p = folder / 'payload' / str(rel); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    (folder / 'evidence-index.json').write_bytes(json.dumps(report, indent=2).encode() + b'\n')
    print(json.dumps({'status': 'downloaded_all_listed_bytes_verified', 'files': len(report['files']), 'remote_rehashed_proofs': len(report['remote_rehash']), 'captured_utc': report['captured_utc']}))


if __name__ == '__main__':
    main()
