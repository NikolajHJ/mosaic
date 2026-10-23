"""Fresh byte-only delivery of completed replay probe and Pro endpoint evidence."""
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

REPO = Path(__file__).resolve().parents[5]
REMOTE = r'''
import base64,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04')
diag=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-02')
files={}; references={}
for parent,label in [(root/'initialization-probe-cpu','probe/cpu'),(root/'initialization-probe-gpu','probe/gpu'),(diag/'diagnostic-protenix','diagnostic/native'),(diag/'diagnostic-capture-protenix','diagnostic/capture')]:
 for p in parent.rglob('*'):
  if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.log','.txt','.npz','.xml']:
   assert p.stat().st_size<4000000,'Unexpected large compact proof'
   files[label+'/'+p.relative_to(parent).as_posix()]=p
assessment=json.loads((diag/'diagnostic-capture-protenix/payload/assessment.json').read_bytes())
assert assessment['status']=='passed' and assessment['scientific_passed'] is False
for pf in assessment['proofs']:
 p=Path(pf['path']);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==pf['sha256']
 references[str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest(),'downloaded':str(p) in {str(x) for x in files.values()}}
rows=[];total=0
for name,p in sorted(files.items()):
 b=p.read_bytes();total+=len(b);assert total<64*1024**2
 rows.append({'path':name,'remote_path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
s=subprocess.run(['bjobs','-l','29792346','29792347','29792158','29792159'],capture_output=True,text=True,timeout=30)
print('EVIDENCE_JSON_BEGIN')
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'files':rows,'remote_reference_rehash':references,'scheduler':{'returncode':s.returncode,'stdout':s.stdout,'stderr':s.stderr}}))
'''


def main():
    folder = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/probe-diagnostic-evidence01'
    folder.mkdir(parents=True, exist_ok=False)
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c 'python3 -'"]
    result = subprocess.run(args, input=REMOTE.encode(), capture_output=True, timeout=120)
    (folder / 'ssh-stdout.bin').write_bytes(result.stdout); (folder / 'ssh-stderr.bin').write_bytes(result.stderr)
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-2000:]
    report = json.loads(result.stdout.split(b'EVIDENCE_JSON_BEGIN\n', 1)[1])
    seen = set()
    for row in report['files']:
        rel = PurePosixPath(row['path'])
        assert not rel.is_absolute() and '..' not in rel.parts and str(rel) not in seen
        seen.add(str(rel))
        raw = base64.b64decode(row.pop('base64'), validate=True)
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        path = folder / 'payload' / str(rel)
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    (folder / 'evidence-index.json').write_bytes(json.dumps(report, indent=2).encode() + b'\n')
    print(json.dumps({'status': 'all_delivered_bytes_verified', 'files': len(seen), 'remote_rehashed_references': len(report['remote_reference_rehash']), 'captured_utc': report['captured_utc']}))


if __name__ == '__main__':
    main()
