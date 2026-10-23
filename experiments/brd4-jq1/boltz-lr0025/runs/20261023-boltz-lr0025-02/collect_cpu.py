"""Fresh compact metadata/CPU-proof delivery; no numerical execution or job action.

This post-seal helper is not part of the scientific release. A failed or unknown
transfer keeps its directory/markers and is never replayed into the same output.
Run verify_cpu.py separately for independent stdlib acceptance.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import base64,hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'boltz-lr0025-02';source=base/'boltz-lr0025-release-02/source'
files={};references={};scheduler={}
def digest(path):
 with path.open('rb') as stream:
  value=hashlib.sha256()
  for data in iter(lambda:stream.read(1048576),b''):value.update(data)
 return value.hexdigest()
def record(path):
 assert path.is_file() and not path.is_symlink()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)}
def add(path):
 assert path.is_relative_to(root) or path.is_relative_to(source),'Foreign proof path'
 name=('run/'+path.relative_to(root).as_posix()) if path.is_relative_to(root) else ('source/'+path.relative_to(source).as_posix())
 if name in files:return
 data=path.read_bytes();assert len(data)<20000000,'Use authenticated reference for large native input'
 files[name]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
for name in ['jobs.json','source-staging.json']:add(root/name)
certificate_path=root/'cpu-certificate.json'
if certificate_path.exists():
 add(certificate_path);certificate=json.loads(certificate_path.read_bytes())
 for proof in certificate['proofs']:
  path=Path(proof['path']);assert digest(path)==proof['sha256'];add(path)
jobs=json.loads((root/'jobs.json').read_bytes())
assert jobs['cpu']=='29798538' and jobs['after-cpu']=='29798539','Wrong actual campaign'
for name,job in jobs.items():
 parent=root/name;add(parent/'submission.json')
 for path in parent.glob('scheduler-*.txt'):add(path)
 for path in parent.glob('storage-*/gate.json'):add(path)
 if name=='cpu':
  for path in (parent/'payload').glob('*'):
   if path.is_file() and path.suffix in ['.xml','.log','.json']:add(path)
 if name in ['cpu','after-cpu']:
  for label in ['stdout','stderr']:
   path=parent/(label+'.'+job+'.log')
   if path.exists():add(path)
 query=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
 scheduler[name]={'job_id':job,'returncode':query.returncode,'raw':query.stdout+query.stderr}
 # Readiness is optional metadata here, not a scientific acceptance shortcut.
 ready=parent/'payload/study/assessment.json'
 if ready.is_file():
  add(ready)
  summary=ready.with_name('summary.json')
  if summary.is_file() and summary.stat().st_size<20000000:add(summary)
  for proof in json.loads(ready.read_bytes()).get('proofs',[]):
   path=Path(proof['path']);value=record(path)
   assert value['sha256']==proof['sha256'],'Native proof bytes changed'
   if path.stat().st_size<1000000 and path.suffix in ['.json','.npz','.xml','.log']:
    add(path)
   else:references[str(path)]=value
for path in root.glob('storage-*/gate.json'):add(path)
for path in root.glob('*-certificate.json'):add(path)
for name in ['projection.json','boltz2-completion.json','boltz2_chemistry-completion.json']:
 path=root/name
 if path.exists():add(path)
for name in ['source_manifest.json','examples/configs/boltz_serial_boltz2.json','examples/configs/boltz_serial_boltz2_chemistry.json','scripts/hpc/boltz_serial_cpu.lsf','scripts/hpc/boltz_serial_stage.lsf']:add(source/name)
print('EVIDENCE_BEGIN')
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'files':files,'scheduler':scheduler,'remote_only_native_references':references,'scope':'Read-only compact CPU/metadata evidence, no model calls or original saved-result changes'}))
'''


def safe_name(name):
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or path.parts[0] not in ('run','source'):
        raise ValueError('Unsafe evidence member name')
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--edition',default='01')
    args=parser.parse_args()
    if not args.edition.isdigit() or len(args.edition)!=2:
        raise ValueError('Use a new two-digit evidence edition')
    base=REPO/'results/hpc'/HERE.name
    output=base/('cpu-evidence-'+args.edition)
    output.mkdir(exist_ok=False)
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"]
    marker={'utc':datetime.now(timezone.utc).isoformat(),'args':command,'outcome':'unknown','scope':'One read-only SSH metadata/CPU-proof request'}
    (output/'transfer-started.json').write_bytes((json.dumps(marker,indent=2)+'\n').encode())
    try:
        result=subprocess.run(command,input=REMOTE.encode(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired as error:
        if error.stdout:(output/'stdout.bin').write_bytes(error.stdout)
        if error.stderr:(output/'stderr.bin').write_bytes(error.stderr)
        (output/'transfer-timeout.json').write_bytes((json.dumps({'outcome':'unknown','error':repr(error)},indent=2)+'\n').encode())
        raise
    (output/'stdout.bin').write_bytes(result.stdout)
    (output/'stderr.bin').write_bytes(result.stderr)
    (output/'transfer-result.json').write_bytes((json.dumps({'returncode':result.returncode},indent=2)+'\n').encode())
    if result.returncode!=0:
        raise RuntimeError('Read-only evidence request failed; output preserved: '+result.stderr.decode(errors='replace')[-1500:])
    prefix=b'EVIDENCE_BEGIN\n'
    if result.stdout.count(prefix)!=1:
        raise ValueError('Missing/unexpected evidence sentinel')
    payload=json.loads(result.stdout.split(prefix,1)[1])
    files=payload.pop('files')
    for name,record in files.items():
        safe_name(name)
        data=base64.b64decode(record.pop('base64'),validate=True)
        if len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']:
            raise ValueError('Delivered file byte identity differs: '+name)
        path=output/name
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('Evidence path escaped')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    metadata={**payload,'files':files}
    raw=(json.dumps(metadata,indent=2)+'\n').encode()
    (output/'metadata.json').write_bytes(raw)
    archive=base/('cpu-evidence-'+args.edition+'.tar.gz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            for name in sorted([*files,'metadata.json']):
                data=(output/name).read_bytes()
                member=tarfile.TarInfo('cpu-evidence-'+args.edition+'/'+name)
                member.mode=0o444;member.size=len(data)
                tar.addfile(member,io.BytesIO(data))
    receipt={'status':'bytes_delivered_not_yet_independently_verified',
        'archive':str(archive),'bytes':archive.stat().st_size,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'listed_files':len(files),'safe_members':len(files)+1,
        'metadata_sha256':hashlib.sha256(raw).hexdigest(),
        'original_native_large_files':'Remote-only authenticated references; not claimed downloaded'}
    (output/'delivery-receipt.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode())
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
