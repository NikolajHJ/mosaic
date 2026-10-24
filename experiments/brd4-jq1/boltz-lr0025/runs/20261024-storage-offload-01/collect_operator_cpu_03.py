"""One compact read-only operator-CPU receipt/source/LSF collector."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import base64,hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
paths=set((root/'operator-source-03').glob('*'))|set((root/'operator-preflight-03').glob('*'))|set((root/'cpu-03').glob('*.json'))|set((root/'cpu-03').glob('*.log'))|set((root/'cpu-03/payload').glob('*'))|set((root/'operator-source-02').glob('*'))|set((root/'operator-preflight-02').glob('*'))|set((root/'cpu-02').glob('*.json'))|set((root/'cpu-02').glob('*.log'))|set((root/'cpu-02/payload').glob('*'))|set(root.glob('*.json'))|set((root/'operator-source-01').glob('*'))|set((root/'operator-preflight-01').glob('*'))|set((root/'cpu-01').glob('*.json'))|set((root/'cpu-01').glob('*.log'))|set((root/'cpu-01/payload').glob('*'))
files={}
for p in sorted(paths):
 if p.is_dir():continue
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<4*1024**2
 raw=p.read_bytes();files[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
queries=[]
for args in [['bjobs','-l','29806228'],['bjobs','-noheader','-o','jobid stat queue job_name']]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=30);queries.append({'args':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'files':files,'scheduler':queries}))
'''


def main():
    import sys
    folder = REPO / 'results/hpc/20261024-storage-offload-01' / sys.argv[1]
    folder.mkdir(exist_ok=False)
    command = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -'
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (folder / 'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': args}) + '\n')
    with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
        result = subprocess.run(args, input=REMOTE.encode(), stdout=out, stderr=err, timeout=90)
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    assert result.returncode == 0
    data = json.loads((folder / 'stdout.bin').read_bytes())
    for name, record in data['files'].items():
        raw = base64.b64decode(record['base64'], validate=True)
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    print(data['utc'], 'verified-byte-records', len(data['files']))
    print(data['scheduler'][0]['stdout'])


if __name__ == '__main__':
    main()
