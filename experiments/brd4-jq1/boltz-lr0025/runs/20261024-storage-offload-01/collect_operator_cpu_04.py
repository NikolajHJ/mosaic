"""One compact read-only CPU04/source/history receipt collector; no submission."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
MANIFEST = '8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
REMOTE = r'''
import base64,hashlib,json,os,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
job=__ACTUAL_JOB__
source=root/'operator-source-04'
manifest_path=source/'source-manifest.json'
assert root.resolve()==root and source.resolve()==source and not source.stat().st_mode&0o222
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
manifest=json.loads(manifest_path.read_bytes())
assert len(manifest['files'])==11 and {p.name for p in source.iterdir()}==set(manifest['files'])|{'source-manifest.json'}
paths=set(root.glob('*.json'))
for suffix in ('01','02','03','04'):
 for folder,pattern in [('operator-source-'+suffix,'*'),('operator-preflight-'+suffix,'*'),('cpu-'+suffix,'*.json'),('cpu-'+suffix,'*.log'),('cpu-'+suffix+'/payload','*')]:
  paths.update((root/folder).glob(pattern))
paths.update(Path(name) for name in manifest['parent_adoption']['proofs'])
files={}
for path in sorted(paths):
 assert path.resolve()==path and path.is_relative_to(root) and not path.is_symlink()
 if path.is_dir():continue
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 with os.fdopen(fd,'rb') as stream:
  before=os.fstat(stream.fileno());assert stat.S_ISREG(before.st_mode) and before.st_size<=4*1024**2
  raw=stream.read(4*1024**2+1);after=os.fstat(stream.fileno())
 literal=path.stat()
 def sig(x):return(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 assert len(raw)==before.st_size and sig(before)==sig(after)==sig(literal)
 files[str(path)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
for name,digest in manifest['files'].items():
 path=source/name;assert files[str(path)]['sha256']==digest and not path.stat().st_mode&0o222
for name,pin in manifest['parent_adoption']['proofs'].items():
 assert {key:files[name][key] for key in ('bytes','sha256')}==pin
submission=json.loads(base64.b64decode(files[str(root/'cpu-04/submission.json')]['base64']))
assert submission['job_id']==job and submission['source_manifest_sha256']=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
queries=[]
for args in [['bjobs','-l',job],['bjobs','-noheader','-o','jobid stat queue job_name']]:
 result=subprocess.run(args,capture_output=True,text=True,timeout=30)
 queries.append({'args':args,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'files':files,'scheduler':queries}))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    assert args.job == '29806439' and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', args.evidence)
    remote = REMOTE.replace('__ACTUAL_JOB__', repr(args.job))
    ast.parse(remote)
    folder = REPO / 'results/hpc/20261024-storage-offload-01' / args.evidence
    folder.mkdir(exist_ok=False)
    # The full read-only code travels on stdin; Windows receives only a short command.
    command = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -'
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
           'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (folder / 'started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'args':ssh,'job_id':args.job,'read_only':True,'remote_code_sha256':hashlib.sha256(remote.encode()).hexdigest()})+'\n')
    (folder / 'remote-collector.py').write_text(remote,encoding='utf8',newline='\n')
    try:
        with (folder/'stdout.bin').open('xb') as out,(folder/'stderr.bin').open('xb') as err:
            result = subprocess.run(ssh,input=remote.encode(),stdout=out,stderr=err,timeout=90)
    except BaseException as error:
        (folder/'failed-unknown.json').write_text(json.dumps({'error':repr(error),'read_only':True,'partial_preserved':True})+'\n')
        raise
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    assert result.returncode == 0
    bundle = json.loads((folder/'stdout.bin').read_bytes())
    assert bundle['read_only'] is True
    for name, record in bundle['files'].items():
        raw = base64.b64decode(record['base64'],validate=True)
        assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
    print(bundle['utc'],'verified-byte-records',len(bundle['files']),'no scientific/deletion pass claimed')
    print(bundle['scheduler'][0]['stdout'])


if __name__ == '__main__':
    main()
