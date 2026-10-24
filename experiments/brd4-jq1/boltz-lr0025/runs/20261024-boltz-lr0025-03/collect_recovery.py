"""One compact read-only SSH snapshot of exact retry jobs and compact proofs."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import base64,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
roots=[b/'boltz-lr0025-03',b/'boltz-dbmol-pilot-02']
r={'captured_utc':datetime.now(timezone.utc).isoformat(),'roots':{},'files':{},'external_files':{},'scheduler':{},'releases':{}}
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add(path):
 path=Path(path);key=str(path)
 if key in r['files'] or key in r['external_files']:return
 assert not path.is_symlink() and path.is_file()
 size=path.stat().st_size
 if size>4*1024*1024:
  r['external_files'][key]={'bytes':size,'sha256':sha(path),'local_bytes_downloaded':False};return
 data=path.read_bytes();assert len(data)==size
 r['files'][key]={'bytes':size,'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
def proofwalk(value):
 if isinstance(value,dict):
  if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
   p=Path(value['path'])
   if p.is_file() and not p.is_symlink():
    add(p)
    found=r['files'].get(str(p),r['external_files'].get(str(p)))
    assert found['sha256']==value['sha256'],'Referenced proof changed: '+str(p)
  for v in value.values():proofwalk(v)
 elif isinstance(value,list):
  for v in value:proofwalk(v)
for root in roots:
 assert root.is_dir() and not root.is_symlink()
 jobs=json.loads((root/'jobs.json').read_bytes());r['roots'][str(root)]={'jobs':jobs}
 for path in root.glob('*.json'):add(path)
 for parent in root.iterdir():
  if not parent.is_dir() or parent.is_symlink():continue
  if (parent/'submission.json').is_file():
   for pat in ['*.json','*.txt','stdout.*.log','stderr.*.log']:
    for path in parent.glob(pat):add(path)
   if parent.name in ['cpu','cpu-05']:
    for path in (parent/'payload').glob('*'):
     if path.is_file() and path.suffix in ['.xml','.json','.log']:add(path)
   for path in parent.glob('storage-*/*.json'):add(path)
 for job in jobs.values():
  if job in r['scheduler']:continue
  result=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
  r['scheduler'][job]={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
for name,expected in [('boltz-lr0025-release-03','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'),('boltz-dbmol-pilot-release-02','600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'),('boltz-dbmol-storage-release-05','865303de06fc0a0bdf9297857c5bbc78c6ccd9ee05177e372a51869adc599bc3')]:
 source=b/name/'source';m=source/'source_manifest.json';assert sha(m)==expected
 manifest=json.loads(m.read_bytes());files=manifest['files']
 assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}==set(files)|{'source_manifest.json'}
 for p in [source,*source.rglob('*')]:assert not p.is_symlink() and(p.is_file() or p.is_dir()) and not(p.stat().st_mode&0o222)
 for rel,digest in files.items():assert sha(source/rel)==digest
 add(m);r['releases'][str(source)]={'manifest_sha256':expected,'source_files':len(files),'all_members_rehashed':True,'readonly':True}
queue=list(r['files']);seen=set()
while queue:
 key=queue.pop()
 if key in seen:continue
 seen.add(key)
 if key.endswith('.json'):
  try:value=json.loads(base64.b64decode(r['files'][key]['base64']))
  except (ValueError,UnicodeError):continue
  proofwalk(value)
  queue.extend(k for k in r['files'] if k not in seen)
assert sum(v['bytes'] for v in r['files'].values())<=16*1024*1024,'Not a compact proof snapshot'
print(json.dumps(r,sort_keys=True))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(REPO / 'results/hpc')
    out.mkdir(parents=True, exist_ok=False)
    command = ' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE])
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (out / 'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': argv}, indent=2) + '\n')
    try:
        result = subprocess.run(argv, capture_output=True, timeout=150)
    except subprocess.TimeoutExpired as error:
        (out / 'timeout.json').write_text(json.dumps({'error': repr(error), 'remote_mutations': False}) + '\n')
        raise
    (out / 'stdout.bin').write_bytes(result.stdout)
    (out / 'stderr.bin').write_bytes(result.stderr)
    (out / 'result.json').write_text(json.dumps({'returncode': result.returncode, 'remote_mutations': False,
        'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(), 'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()}, indent=2) + '\n')
    assert result.returncode == 0, result.stderr.decode(errors='replace')[-3000:]
    metadata = json.loads(result.stdout)
    for index, (remote, value) in enumerate(metadata['files'].items()):
        local = 'files/' + str(index).zfill(4) + '/' + PurePosixPath(remote).name
        data = base64.b64decode(value.pop('base64'), validate=True)
        assert len(data) == value['bytes'] and hashlib.sha256(data).hexdigest() == value['sha256']
        path = out / local
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as handle:
            handle.write(data)
        value['local_path'] = local
    with (out / 'metadata.json').open('x', encoding='utf8') as handle:
        json.dump(metadata, handle, indent=2); handle.write('\n')
    statuses = {}
    for job, value in metadata['scheduler'].items():
        import re
        statuses[job] = re.findall(r'Status<([^>]+)>', re.sub(r'\s+', '', value['stdout'])) or 'unavailable'
    print(json.dumps({'captured_utc': metadata['captured_utc'], 'jobs': statuses,
        'downloaded_files': len(metadata['files']), 'external_metadata_only': len(metadata['external_files']),
        'new_release_byte_seals': metadata['releases']}, indent=2))


if __name__ == '__main__':
    main()
