"""Fresh read-only dual-source CPU evidence, delivered as a direct binary tar.

Post-seal delivery tooling only. No launch, retry, test or result mutation.
Each edition is exclusive, including failed and interrupted transfers.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import hashlib,io,json,re,subprocess,sys,tarfile
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'boltz-dbmol-pilot-01';science=base/'boltz-dbmol-pilot-release-01/source';launcher=base/'boltz-dbmol-storage-release-03/source'
files={};content={};references={};scheduler={}
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def namespace(path):
 for label,parent in [('run',root),('science',science),('launcher',launcher)]:
  if path.is_relative_to(parent):return label+'/'+path.relative_to(parent).as_posix()
 assert path.is_relative_to(base),'Foreign proof'
 return 'external/'+path.relative_to(base).as_posix()
def add(path):
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),'Nonregular/escaping proof'
 name=namespace(path)
 if name in files:return
 assert path.stat().st_size<16000000,'Large scientific files are references only'
 data=path.read_bytes();assert sum(len(v) for v in content.values())+len(data)<100000000,'Compact evidence exceeds100MB'
 files[name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};content[name]=data
def optional(path):
 if path.exists():add(path)
def reference(path):
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),'Foreign large proof'
 references[str(path)]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'downloaded':False}
def proofs(record):
 for proof in record.get('proofs',[]):
  path=Path(proof['path']);assert sha(path)==proof['sha256'],'Authenticated proof changed'
  if path.stat().st_size<16000000:add(path)
  else:reference(path)
for name in ['jobs.json','source-staging.json','cpu-certificate.json','projection.json','completion.json']:
 optional(root/name)
jobs=json.loads((root/'jobs.json').read_bytes()) if (root/'jobs.json').is_file() else {}
for label,job in jobs.items():
 assert label in ['cpu','after-cpu','cpu-02','after-cpu-02','cpu-03','after-cpu-03'] or re.fullmatch(r'(?:pilot-(?:native|after-native|segment|after-segment)|dispatch)-\d{4}',label),'Undeclared job parent'
 assert isinstance(job,str) and re.fullmatch(r'\d+',job),'Malformed actual job'
 parent=root/label
 add(parent/'submission.json');optional(parent/'submission-started.json')
 for p in parent.glob('scheduler-*.txt'):add(p)
 for kind in ['stdout','stderr']:optional(parent/(kind+'.'+job+'.log'))
 for p in parent.glob('storage-*/*'):
  if p.is_file() and p.suffix in ['.json','.txt','.log']:add(p)
 if label in ['cpu','cpu-02','cpu-03']:
  for p in (parent/'payload').glob('*'):
   if p.is_file() and p.suffix in ['.xml','.log','.json']:add(p)
 ready=parent/'payload/study/assessment.json'
 if ready.is_file():add(ready);proofs(json.loads(ready.read_bytes()))
 try:
  q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=15)
  scheduler[label]={'job_id':job,'returncode':q.returncode,'raw':q.stdout+q.stderr}
 except subprocess.TimeoutExpired as e:
  scheduler[label]={'job_id':job,'returncode':None,'error':repr(e),'raw':(e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace')}
for p in root.glob('*/submission-started.json'):add(p)
for p in root.glob('*/submission.json'):add(p)
for p in root.glob('*-certificate.json'):add(p);proofs(json.loads(p.read_bytes()))
for p in root.glob('storage-*/*'):
 if p.is_file() and p.suffix in ['.json','.txt','.log']:add(p)
for control in ['admission-02','storage-adoption-02','storage-adoption-03']:
 p=root/control
 if p.is_dir():
  for f in p.rglob('*'):
   if f.is_file() and f.suffix in ['.json','.txt','.log']:add(f)
integrity={}
for label,source in [('science',science),('launcher',launcher),('previous_launcher',base/'boltz-dbmol-storage-release-02/source')]:
 raw=(source/'source_manifest.json').read_bytes();m=json.loads(raw);expected=set(m['files'])|{'source_manifest.json'}
 assert all(not p.is_symlink() and(p.is_dir() or p.is_file()) and not(p.stat().st_mode&0o222) for p in [source,*source.rglob('*')]),'Seal not regular/read-only'
 assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==expected,'Source inventory differs'
 for name,digest in m['files'].items():assert sha(source/name)==digest,'Source changed:'+name
 integrity[label]={'status':'passed','manifest_sha256':hashlib.sha256(raw).hexdigest(),'files':len(m['files']),'readonly':True}
 add(source/'source_manifest.json')
 selected=['examples/configs/dbmol_pilot.json','scripts/hpc/dbmol_cpu.lsf','scripts/hpc/dbmol_stage.lsf'] if label=='science' else m['files']
 for name in selected:add(source/name)
assets={}
for key,path in [('affinity',Path('/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt')),('structural',Path('/work3/s234818/latentmol/cache/boltz2/boltz2_conf.ckpt')),('prepared_context',Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared/context.pt'))]:
 assert path.is_file() and not path.is_symlink()
 assets[key]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'downloaded':False}
original={}
add(base/'restart-lrreset-release-04/source/source_manifest.json')
for arm,job in [('protenix','29792670'),('boltz2','29792671')]:
 parent=base/'restart-lrreset-04'/('run-'+arm);summary=parent/'payload/study/summary.json'
 reference(summary);add(parent/'submission.json');add(parent/('stdout.'+job+'.log'))
 data=json.loads(summary.read_bytes());cases=data.get('cases',[])
 original[arm]={'status':data.get('status'),'cases':len(cases),'case_status':cases[0].get('status') if len(cases)==1 else None,'updates':len(cases[0].get('updates',[])) if len(cases)==1 else None,'states':len(cases[0].get('states',[])) if len(cases)==1 else None,'large_original_bytes_downloaded':False}
metadata={'observed_utc':datetime.now(timezone.utc).isoformat(),'files':files,'scheduler':scheduler,'jobs_file_present':(root/'jobs.json').is_file(),'source_integrity':integrity,'asset_references':assets,'remote_only_references':references,'original_terminal_metadata':original,'scope':'Read-only stdlib bytes, metadata and scheduler. No model, numerical evaluation, launch, retry or saved-result change.'}
raw=(json.dumps(metadata,indent=2,allow_nan=False)+'\n').encode();content['metadata.json']=raw
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
 for name,data in sorted(content.items()):
  member=tarfile.TarInfo(name);member.mode=0o444;member.size=len(data);archive.addfile(member,io.BytesIO(data))
data=stream.getvalue();sys.stdout.buffer.write(('MOSAIC_EVIDENCE '+str(len(data))+' '+hashlib.sha256(data).hexdigest()+'\n').encode());sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''


def safe(name):
    p = PurePosixPath(name)
    if (not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name
            or ':' in name or (name != 'metadata.json' and p.parts[0] not in ('run', 'science', 'launcher', 'external'))):
        raise ValueError('Unsafe evidence member')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', default='01')
    parser.add_argument('--expect-failure', action='store_true')
    parser.add_argument('--admission-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{2}', args.edition):
        raise ValueError('Use a fresh two-digit edition')
    base = REPO / 'results/hpc' / HERE.name
    output = base / ('cpu-evidence-' + args.edition)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = base / ('cpu-evidence-' + args.edition + '.tar.gz')
    if archive_path.exists():
        raise ValueError('Existing archive: inspect the earlier outcome')
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -'"]
    marker = {'utc': datetime.now(timezone.utc).isoformat(), 'args': command, 'outcome': 'unknown', 'expect_failure': args.expect_failure, 'admission_only': args.admission_only}
    (output / 'transfer-started.json').write_text(json.dumps(marker, indent=2) + '\n')
    try:
        result = subprocess.run(command, input=REMOTE.encode(), capture_output=True, timeout=360)
    except subprocess.TimeoutExpired as e:
        (output / 'stdout.partial.bin').write_bytes(e.stdout or b'')
        (output / 'stderr.bin').write_bytes(e.stderr or b'')
        (output / 'transfer-timeout.json').write_text(json.dumps({'outcome': 'unknown', 'error': repr(e)}) + '\n')
        raise
    (output / 'stdout.bin').write_bytes(result.stdout)
    (output / 'stderr.bin').write_bytes(result.stderr)
    (output / 'transfer-result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    if result.returncode:
        raise RuntimeError('Read-only transfer failed; preserve partial evidence before recovery')
    matches = list(re.finditer(rb'(?m)^MOSAIC_EVIDENCE (\d+) ([0-9a-f]{64})\r?\n', result.stdout))
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous binary evidence header')
    match = matches[0]
    data = result.stdout[match.end():]
    if len(data) != int(match[1]) or hashlib.sha256(data).hexdigest().encode() != match[2]:
        raise ValueError('Binary archive length/hash mismatch')
    with archive_path.open('xb') as stream:
        stream.write(data)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            safe(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError('Nonregular or duplicate evidence member')
            seen.add(member.name)
            path = output / member.name
            if not path.resolve().is_relative_to(output.resolve()):
                raise ValueError('Evidence path escapes delivery')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.extractfile(member).read())
    raw = (output / 'metadata.json').read_bytes()
    metadata = json.loads(raw)
    if seen != set(metadata['files']) | {'metadata.json'}:
        raise ValueError('Compact archive inventory differs')
    for name, record in metadata['files'].items():
        value = (output / name).read_bytes()
        if len(value) != record['bytes'] or hashlib.sha256(value).hexdigest() != record['sha256']:
            raise ValueError('Delivered file mismatch: ' + name)
    receipt = {'status': 'bytes_delivered_not_yet_independently_verified', 'archive': str(archive_path), 'bytes': len(data), 'sha256': match[2].decode(), 'listed_files': len(metadata['files']), 'safe_members': len(seen), 'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'large_original_native_assets': 'Authenticated remote references only; not downloaded'}
    (output / 'delivery-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
