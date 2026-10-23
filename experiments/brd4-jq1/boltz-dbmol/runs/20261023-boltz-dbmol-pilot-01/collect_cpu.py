"""Fresh read-only compact staging/CPU delivery; post-seal stdlib tooling only.

Supports a failed preflight with no jobs.json. It never launches, retries or
changes a job. Each two-digit edition is exclusive, including unknown partials.
Run verify_cpu.py separately; delivery itself is not scientific acceptance.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import base64,hashlib,json,re,stat,subprocess
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'boltz-dbmol-pilot-01';source=base/'boltz-dbmol-pilot-release-01/source'
files={};references={};scheduler={}
def digest(path):
 value=hashlib.sha256()
 with path.open('rb') as stream:
  for data in iter(lambda:stream.read(1048576),b''):value.update(data)
 return value.hexdigest()
def add(path):
 assert path.is_file() and not path.is_symlink(),'Nonregular proof'
 assert path.is_relative_to(root) or path.is_relative_to(source),'Foreign proof'
 assert path.resolve().is_relative_to(root.resolve()) or path.resolve().is_relative_to(source.resolve()),'Resolved proof escapes campaign/source'
 name=('run/'+path.relative_to(root).as_posix()) if path.is_relative_to(root) else ('source/'+path.relative_to(source).as_posix())
 if name in files:return
 data=path.read_bytes();assert len(data)<20000000,'Large original/native files must remain authenticated references'
 files[name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
def optional(path):
 if path.exists():add(path)
def proof_files(c):
 for proof in c.get('proofs',[]):
  path=Path(proof['path']);assert path.is_relative_to(root) and path.is_file() and not path.is_symlink()
  assert digest(path)==proof['sha256'],'Original proof changed'
  if path.stat().st_size<1000000 and path.suffix in ('.json','.npz','.xml','.log','.txt'):add(path)
  else:references[str(path)]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}
for name in ('jobs.json','source-staging.json','cpu-certificate.json','projection.json','completion.json'):
 optional(root/name)
certificate=root/'cpu-certificate.json'
if certificate.exists():proof_files(json.loads(certificate.read_bytes()))
jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').exists() else {}
for name,job in jobs.items():
 assert name in ('cpu','after-cpu') or re.fullmatch(r'(?:pilot-(?:native|after-native|segment|after-segment)|dispatch)-\d{4}',name),'Undeclared campaign parent'
 assert isinstance(job,str) and re.fullmatch(r'\d+',job),'Malformed actual job ID'
 parent=root/name;add(parent/'submission.json');optional(parent/'submission-started.json')
 for path in parent.glob('scheduler-*.txt'):add(path)
 for path in parent.glob('storage-*/gate.json'):add(path)
 if name=='cpu':
  for path in (parent/'payload').glob('*'):
   if path.is_file() and path.suffix in ('.xml','.log','.json'):add(path)
 for label in ('stdout','stderr'):optional(parent/(label+'.'+job+'.log'))
 try:
  query=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=15)
  scheduler[name]={'job_id':job,'returncode':query.returncode,'raw':query.stdout+query.stderr}
 except subprocess.TimeoutExpired as e:scheduler[name]={'job_id':job,'returncode':None,'error':repr(e),'raw':((e.stdout or b'')+(e.stderr or b'')).decode(errors='replace')}
 ready=parent/'payload/study/assessment.json'
 if ready.exists():
  add(ready);proof_files(json.loads(ready.read_bytes()))
  summary=ready.with_name('summary.json')
  if summary.is_file() and summary.stat().st_size<1000000:add(summary)
if root.exists():
 for path in root.glob('storage-*/gate.json'):add(path)
 for path in root.glob('*-certificate.json'):add(path)
 for admission in root.glob('admission-*'):
  if admission.is_dir() and not admission.is_symlink():
   for path in admission.rglob('*.json'):add(path)
 for pattern in ('*retire*.json','*cleanup*.json'):
  for path in root.glob(pattern):add(path)
 # Preserve an unknown outcome even when jobs.json was not atomically updated.
 for path in root.glob('*/submission-started.json'):add(path)
 for path in root.glob('*/submission.json'):add(path)
for name in ('source_manifest.json','examples/configs/dbmol_pilot.json','scripts/hpc/dbmol_cpu.lsf','scripts/hpc/dbmol_stage.lsf'):
 optional(source/name)
integrity={'status':'missing','files':0,'readonly':False}
if (source/'source_manifest.json').is_file():
 try:
  raw=(source/'source_manifest.json').read_bytes();manifest=json.loads(raw)
  assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in [source,*source.rglob('*')]),'Unexpected source link/special entry'
  expected=set(manifest['files'])|{'source_manifest.json'}
  actual={p.relative_to(source).as_posix() for p in source.rglob('*') if not p.is_dir()}
  assert actual==expected,'Source member scope differs'
  for name,expected_digest in manifest['files'].items():
   path=source/name;assert path.is_file() and not path.is_symlink() and digest(path)==expected_digest,'Source member changed: '+name
  readonly=all(not (p.stat().st_mode & 0o222) for p in [source,*source.rglob('*')])
  assert readonly,'Sealed source is writable'
  integrity={'status':'passed','manifest_sha256':hashlib.sha256(raw).hexdigest(),'files':len(manifest['files']),'readonly':True}
 except Exception as e:integrity={'status':'failed','error':repr(e),'files':0,'readonly':False}
assets={}
for key,path in [('affinity',Path('/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt')),('structural',Path('/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt')),('prepared_context',Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt'))]:
 if path.is_file() and not path.is_symlink():assets[key]={'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'downloaded':False}
 else:assets[key]={'path':str(path),'status':'missing_or_symlink','downloaded':False}
print('EVIDENCE_BEGIN')
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'files':files,'scheduler':scheduler,'jobs_file_present':(root/'jobs.json').is_file(),'source_integrity':integrity,'asset_references':assets,'remote_only_native_references':references,'scope':'Read-only staging/CPU metadata and byte hashes. No model, numerical test, launch, retry or saved-result change.'},allow_nan=False))
'''


def safe_name(name):
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or '..' in path.parts or '\\' in name
            or ':' in name or path.parts[0] not in ('run', 'source')):
        raise ValueError('Unsafe evidence member name')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', default='01')
    parser.add_argument('--expect-failure', action='store_true', help='Collect a failed preflight/CPU without requiring jobs.json')
    args = parser.parse_args()
    if not args.edition.isdigit() or len(args.edition) != 2:
        raise ValueError('Use a new two-digit edition')
    base = REPO / 'results/hpc' / HERE.name
    output = base / ('cpu-evidence-' + args.edition)
    output.mkdir(exist_ok=False)
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
               '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c 'python3 -'"]
    marker = {'utc': datetime.now(timezone.utc).isoformat(), 'args': command,
              'outcome': 'unknown', 'expect_failure': args.expect_failure,
              'scope': 'One read-only SSH byte/metadata request'}
    (output / 'transfer-started.json').write_bytes((json.dumps(marker, indent=2) + '\n').encode())
    try:
        result = subprocess.run(command, input=REMOTE.encode(), capture_output=True, timeout=240)
    except subprocess.TimeoutExpired as error:
        if error.stdout: (output / 'stdout.bin').write_bytes(error.stdout)
        if error.stderr: (output / 'stderr.bin').write_bytes(error.stderr)
        (output / 'transfer-timeout.json').write_bytes((json.dumps({'outcome': 'unknown', 'error': repr(error)}, indent=2) + '\n').encode())
        raise
    (output / 'stdout.bin').write_bytes(result.stdout)
    (output / 'stderr.bin').write_bytes(result.stderr)
    (output / 'transfer-result.json').write_bytes((json.dumps({'returncode': result.returncode}, indent=2) + '\n').encode())
    if result.returncode != 0:
        raise RuntimeError('Read-only collection failed; preserve output before recovery')
    sentinel = b'EVIDENCE_BEGIN\n'
    if result.stdout.count(sentinel) != 1:
        raise ValueError('Missing/ambiguous evidence sentinel')
    payload = json.loads(result.stdout.split(sentinel, 1)[1])
    files = payload.pop('files')
    if not args.expect_failure and not payload['jobs_file_present']:
        raise ValueError('No actual jobs.json; use explicit failed-attempt scope after inspecting preflight')
    for name, record in files.items():
        safe_name(name)
        data = base64.b64decode(record.pop('base64'), validate=True)
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Delivered proof bytes differ: ' + name)
        path = output / name
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('Evidence escaped output')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
    raw = (json.dumps({**payload, 'files': files}, indent=2) + '\n').encode()
    (output / 'metadata.json').write_bytes(raw)
    archive = base / ('cpu-evidence-' + args.edition + '.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for name in sorted([*files, 'metadata.json']):
                data = (output / name).read_bytes()
                member = tarfile.TarInfo('cpu-evidence-' + args.edition + '/' + name)
                member.mode = 0o444
                member.size = len(data)
                tar.addfile(member, io.BytesIO(data))
    receipt = {'status': 'bytes_delivered_not_yet_independently_verified',
               'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'listed_files': len(files), 'safe_members': len(files) + 1,
               'metadata_sha256': hashlib.sha256(raw).hexdigest(),
               'original_native_large_files': 'Remote-only authenticated references, not downloaded'}
    (output / 'delivery-receipt.json').write_bytes((json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
