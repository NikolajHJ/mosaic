"""Fresh read-only native/callback delivery, not a native acceptance shortcut.

Direct SSH binary stream; no launcher/controller/model imports or job mutations.
Every edition is exclusive. Failed/unknown partial outcomes are preserved.
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
root=base/'boltz-lr0025-02';source=base/'boltz-lr0025-release-02/source'
manifest_sha='3689071a4e7dd135071ff5b420c026543c4d10887c951077dff63d498d9f812c'
expected={'boltz2-native-0000':'29798605','boltz2-after-native-0000':'29798606','boltz2_chemistry-native-0000':'29798607','boltz2_chemistry-after-native-0000':'29798608'}
files={};content={};references={};scheduler={};prefixes={}
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def name(path):
 for label,parent in [('run',root),('source',source)]:
  if path.is_relative_to(parent) and path.resolve().is_relative_to(parent.resolve()):return label+'/'+path.relative_to(parent).as_posix()
 raise ValueError('Proof is outside exact campaign/source')
def record(path):
 assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(base.resolve()),'Foreign/nonregular proof'
 name(path)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def reference(path):
 value=record(path);value['downloaded']=False;references[str(path)]=value;return value
def add(path):
 n=name(path)
 if n in files:return
 assert path.is_file() and not path.is_symlink(),'Nonregular compact proof'
 if path.stat().st_size>=16000000:return reference(path)
 data=path.read_bytes();assert sum(map(len,content.values()))+len(data)<100000000,'Compact evidence exceeds100MB'
 content[n]=data;files[n]={'path':str(path),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def optional(path):
 if path.exists():add(path)
def proofs(value):
 for proof in value.get('proofs',[]):
  p=Path(proof['path']);r=record(p);assert r['sha256']==proof['sha256'],'Certificate/assessment proof changed'
  if p.stat().st_size>=1000000 and (p.suffix in ['.npz','.pt','.eqx'] or p.name=='summary.json'):reference(p)
  else:add(p)
mp=source/'source_manifest.json';raw=mp.read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest_sha
m=json.loads(raw);assert len(m['files'])==633
assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) and not(p.stat().st_mode&0o222) for p in [source,*source.rglob('*')]),'Source is not sealed read-only'
assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==set(m['files'])|{'source_manifest.json'}
for n,d in m['files'].items():
 p=source/n;assert p.resolve().is_relative_to(source.resolve()) and sha(p)==d,'Source member changed:'+n
integrity={'status':'passed','manifest_sha256':manifest_sha,'files':633,'readonly':True}
for n in ['source_manifest.json','examples/configs/boltz_serial_boltz2.json','examples/configs/boltz_serial_boltz2_chemistry.json','scripts/boltz_serial_ops.py','scripts/boltz_serial_protocol.py','scripts/campaign25_contract.py','scripts/hpc/boltz_serial_cpu.lsf','scripts/hpc/boltz_serial_stage.lsf']:add(source/n)
for n in ['jobs.json','source-staging.json','cpu-certificate.json','projection.json','boltz2-completion.json','boltz2_chemistry-completion.json']:optional(root/n)
cpu=json.loads((root/'cpu-certificate.json').read_bytes());assert cpu['status']=='passed' and cpu['source_manifest_sha256']==manifest_sha;proofs(cpu)
jobs=json.loads((root/'jobs.json').read_bytes());assert all(jobs.get(k)==v for k,v in expected.items()),'Original actual validators/callbacks changed'
for label,job in jobs.items():
 assert re.fullmatch(r'\d+',job)
 if label in ['cpu','after-cpu']:continue
 assert label in expected or re.fullmatch(r'(?:boltz2|boltz2_chemistry)-(?:native|after-native|segment|after-segment)-\d{4}|dispatch-\d{4}',label),'Undeclared actual parent'
 parent=root/label;sub=json.loads((parent/'submission.json').read_bytes());assert sub['outcome']=='submitted' and sub['job_id']==job
 add(parent/'submission.json');optional(parent/'submission-started.json')
 for pattern in ['scheduler-*.txt','waiting*.json']:
  for p in parent.glob(pattern):add(p)
 for kind in ['stdout','stderr']:optional(parent/(kind+'.'+job+'.log'))
 for p in parent.glob('storage-*/*'):
  if p.is_file() and p.suffix in ['.json','.txt','.log']:add(p)
 optional(parent/'payload/driver-invocation.json')
 study=parent/'payload/study'
 assessment=study/'assessment.json'
 if assessment.is_file():
  add(assessment);proofs(json.loads(assessment.read_bytes()))
 for suffix in ['summary.json','native_gradient_checks/report.json','terminal_gradient_checks/report.json','early_pilot/report.json','high_pilot/report.json','low_coupled_checks/report.json','high_coupled_checks/report.json']:
  p=study/suffix
  if p.is_file():
   if p.name=='summary.json' and p.stat().st_size>=1000000:reference(p)
   else:add(p)
 if sub['phase']=='segment':
  for field in ['states','updates']:
   p=study/'case-0'/(field+'.jsonl')
   if p.is_file():
    count=0;size=0;last=None;h=hashlib.sha256();contiguous=True
    with p.open('rb') as stream:
     for line in stream:
      if not line.endswith(b'\n'):break
      value=json.loads(line);contiguous=contiguous and value.get('step')==count
      if last is not None:contiguous=contiguous and value.get('global_step')==last.get('global_step')+1
      last=value;count+=1;size+=len(line);h.update(line)
    prefixes[str(p)]={'path':str(p),'newline_committed_rows':count,'committed_prefix_bytes':size,'committed_prefix_sha256':h.hexdigest(),'contiguous_saved_local_and_global_indices':contiguous,'last_local_step':None if last is None else last.get('step'),'last_global_step':None if last is None else last.get('global_step'),'last_saved_record':last,'downloaded':False,'scope':'Saved newline-committed metadata only; no arrays or inferred intermediate states'}
 try:
  q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=15)
  scheduler[label]={'job_id':job,'returncode':q.returncode,'raw':q.stdout+q.stderr}
 except subprocess.TimeoutExpired as e:
  scheduler[label]={'job_id':job,'returncode':None,'error':repr(e),'raw':(e.stdout or b'').decode(errors='replace')+(e.stderr or b'').decode(errors='replace')}
for p in root.glob('*-certificate.json'):
 if p.name=='cpu-certificate.json':continue
 add(p);proofs(json.loads(p.read_bytes()))
for p in root.glob('*/submission-started.json'):optional(p)
for p in root.glob('*/submission.json'):optional(p)
metadata={'observed_utc':datetime.now(timezone.utc).isoformat(),'root':str(root),'source':str(source),'source_integrity':integrity,'files':files,'scheduler':scheduler,'remote_only_references':references,'production_prefixes':prefixes,'scope':'Read-only native/callback outcome evidence and saved progress metadata. No numerical evaluation, models, controller imports, job actions, or saved-result changes. Missing live history is not fabricated DONE/EXIT.'}
content['metadata.json']=(json.dumps(metadata,indent=2,allow_nan=False)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
 for n,data in sorted(content.items()):
  member=tarfile.TarInfo(n);member.mode=0o444;member.size=len(data);archive.addfile(member,io.BytesIO(data))
data=stream.getvalue();sys.stdout.buffer.write(('MOSAIC_EVIDENCE '+str(len(data))+' '+hashlib.sha256(data).hexdigest()+'\n').encode());sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
'''


def safe(name):
    p = PurePosixPath(name)
    if (not p.parts or p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name
            or (name != 'metadata.json' and p.parts[0] not in ('run', 'source'))):
        raise ValueError('Unsafe evidence member')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', default='01')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{2}', args.edition):
        raise ValueError('Use a fresh two-digit edition')
    base = REPO / 'results/hpc' / HERE.name
    output = base / ('native-evidence-' + args.edition)
    output.mkdir(parents=True, exist_ok=False)
    archive_path = base / ('native-evidence-' + args.edition + '.tar.gz')
    if archive_path.exists():
        raise ValueError('Existing archive: inspect original outcome')
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', "bash -l -c '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -'"]
    (output / 'transfer-started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': command, 'outcome': 'unknown', 'scope': 'One read-only native outcome collection'}, indent=2) + '\n')
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
        raise RuntimeError('Read-only collection failed; inspect preserved output before any recovery')
    matches = list(re.finditer(rb'(?m)^MOSAIC_EVIDENCE (\d+) ([0-9a-f]{64})\r?\n', result.stdout))
    if len(matches) != 1:
        raise ValueError('Missing/ambiguous binary evidence header')
    match = matches[0];data = result.stdout[match.end():]
    if len(data) != int(match[1]) or hashlib.sha256(data).hexdigest().encode() != match[2]:
        raise ValueError('Binary archive length/hash differs')
    with archive_path.open('xb') as stream:stream.write(data)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            safe(member.name)
            if not member.isfile() or member.name in seen:
                raise ValueError('Nonregular or duplicate compact member')
            seen.add(member.name);path = output / member.name
            if not path.resolve().is_relative_to(output.resolve()):raise ValueError('Evidence escaped output')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:stream.write(archive.extractfile(member).read())
    raw = (output / 'metadata.json').read_bytes();metadata = json.loads(raw)
    if seen != set(metadata['files']) | {'metadata.json'}:
        raise ValueError('Compact archive inventory differs')
    for name, record in metadata['files'].items():
        value = (output / name).read_bytes()
        if len(value) != record['bytes'] or hashlib.sha256(value).hexdigest() != record['sha256']:
            raise ValueError('Delivered proof bytes changed: ' + name)
    receipt = {'status': 'bytes_delivered_not_yet_independently_verified', 'archive': str(archive_path), 'bytes': len(data), 'sha256': match[2].decode(), 'listed_files': len(metadata['files']), 'safe_members': len(seen), 'metadata_sha256': hashlib.sha256(raw).hexdigest(), 'large_native_files': 'Remote-only path/length/SHA references; not downloaded', 'native_pass_established_by_delivery': False, 'optimization_start_established_by_delivery': False}
    (output / 'delivery-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
