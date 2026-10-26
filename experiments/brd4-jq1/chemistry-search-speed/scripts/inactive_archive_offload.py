"""Explicit verified laptop backup, then exact inactive-file retirement.

Byte/metadata operations only. No action on import; no model/array imports.
Each phase refuses an existing outcome and records interruptions for inspection.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO/'results/hpc/20261026-chemistry-storage-recovery-01'
REMOTE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/chemistry-storage-recovery-01'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
PLAN_RELATIVE = 'plan-06/plan.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def plan():
    raw=(LOCAL/PLAN_RELATIVE).read_bytes(); value=json.loads(raw)
    assert value['version']=='chemistry-inactive-byte-offload-v1'
    names=set(); protected=set(value['protected_paths'])
    for row in value['files']:
        p=PurePosixPath(row['path'])
        assert p.is_absolute() and '..' not in p.parts and '\\' not in str(p)
        assert str(p).startswith('/work3/s234818/mosaic/') and str(p) not in protected
        assert str(p) not in names; names.add(str(p))
        assert row['bytes']>0 and len(row['sha256'])==64 and row['identity'][-1]==1
    return raw,value


def stream_receipt(path):
    # SSH may print a login notice on stderr. Preserve it verbatim and require
    # exactly one structured completion line, never reinterpret a banner/error.
    rows=[]
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.startswith('{') and line.endswith('}'):
            value=json.loads(line)
            if value.get('status')=='complete_remote_stream_originals_unchanged':rows.append(value)
    assert len(rows)==1, 'Unique actual remote backup completion receipt required'
    return rows[0]


def verify_existing_backup():
    """Recover only local receipt parsing; never reconnect or repeat a stream."""
    raw,value=plan();out=LOCAL/'backup-01';target=out/'inactive-results.tar.gz'
    receipt=json.loads((out/'result.json').read_bytes());assert receipt['returncode']==0
    remote=stream_receipt(out/'remote-receipt.txt')
    assert remote['plan_sha256']==hashlib.sha256(raw).hexdigest()
    assert remote['archive_bytes']==target.stat().st_size and remote['archive_sha256']==sha(target)
    expected={r['path'].lstrip('/'):r for r in value['files']};seen=set()
    with tarfile.open(target,'r:gz') as archive:
        for member in archive:
            part=PurePosixPath(member.name)
            assert member.isfile() and not part.is_absolute() and '..' not in part.parts and member.name not in seen
            seen.add(member.name)
            with archive.extractfile(member) as stream:
                if member.name=='offload-plan.json':assert stream.read()==raw
                else:
                    row=expected[member.name]
                    assert member.size==row['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
        assert seen==set(expected)|{'offload-plan.json'}
    write(out/'laptop-verification.json',{'status':'complete_laptop_archive_every_member_verified','archive_sha256':sha(target),
          'archive_bytes':target.stat().st_size,'files':len(expected),'plan_sha256':hashlib.sha256(raw).hexdigest(),
          'scientific_values_changed':False,'scope':'Complete original stream; local SSH-banner receipt parsing adapted, no repeated transfer'})
    print(json.dumps({'status':'complete_laptop_archive_every_member_verified','files':len(expected),'archive_bytes':target.stat().st_size}))


COMMON = r'''
import base64,gzip,hashlib,io,json,os,re,stat,subprocess,sys,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-storage-recovery-01'
raw=base64.b64decode(PLAN,validate=True)
assert hashlib.sha256(raw).hexdigest()==PIN
plan=json.loads(raw);protected=set(plan['protected_paths'])
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def activity():
 q=subprocess.run(['bjobs','-u','s234818','-l'],capture_output=True,text=True,timeout=30)
 assert q.returncode==0,'Actual current scheduler query failed'
 compact=''.join(q.stdout.split())
 for block in re.split(r'(?=Job<[1-9][0-9]*>,JobName<)',compact):
  if '/mosaic/' not in block:continue
  assert 'Status<RUN>' not in block,'Concurrent Mosaic job prohibits retirement'
  if 'Status<PEND>' in block:
   assert 'Dependencyconditioninvalidorneversatisfied' in block,'Eligible/pending Mosaic work prohibits retirement'
 return {'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
def current():
 source=base/'chemistry-production-release-02/source';sys.path.insert(0,str(source/'scripts'))
 import production_ops as op
 m,p=op.own_plan('71fa3a69ca783632243dc1f9279db3b30902eb7aaacbf1be2efd7a429a9579e6')
 result={a:op.pilot_pass(a,p) for a in ('protenix','boltz')}
 assert all(v['status']=='passed' for v in result.values())
 guard=op.module(op.GUARD_SOURCE/'scripts/protenix_precision_ops.py','_offload_protected_guard')
 gm=guard.verify(op.GUARD_SOURCE,op.GUARD_PIN);parent=guard.parent_ops();forensic=guard.forensic_ops()
 objects=[gm,guard.read(guard.SOURCE/'plan.json'),parent.read(parent.SOURCE/'source_manifest.json'),parent.read(parent.SOURCE/'plan.json'),forensic.read(forensic.SOURCE/'source_manifest.json'),forensic.read(forensic.SOURCE/'plan.json')]
 def bindings(value):
  if isinstance(value,dict):
   if isinstance(value.get('path'),str) and value['path'].startswith(('/work3/s234818/mosaic/','/zhome/95/e/206121/mosaic-runs/')) and 'sha256' in value:op.check_binding(value)
   for item in value.values():bindings(item)
  elif isinstance(value,list):
   for item in value:bindings(item)
 for value in objects:bindings(value)
 return op,result
def check(row):
 p=Path(row['path']);assert str(p) not in protected
 assert p.is_relative_to(Path('/work3/s234818/mosaic')) and '..' not in p.parts
 cur=p
 while cur!=Path('/work3/s234818/mosaic'):
  assert not cur.is_symlink(),'Linked path/parent rejected';cur=cur.parent
 s=p.lstat();identity=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink]
 assert stat.S_ISREG(s.st_mode) and identity==row['identity'],'File identity changed'
 assert sha(p)==row['sha256'],'File bytes changed'
 return p
'''

BACKUP = r'''
assert not os.path.lexists(root),'Existing/unknown offload prohibits backup replay'
before=activity();op,adoptions=current()
root.mkdir();(root/'plan.json').write_bytes(raw)
write(root/'backup-started.json',{'plan_sha256':PIN,'files':len(plan['files']),'logical_bytes':sum(r['bytes'] for r in plan['files']),'activity':before})
class DigestWriter:
 def __init__(self):self.h=hashlib.sha256();self.bytes=0
 def write(self,data):
  sys.stdout.buffer.write(data);self.h.update(data);self.bytes+=len(data);return len(data)
 def flush(self):sys.stdout.buffer.flush()
writer=DigestWriter();started=time.monotonic()
with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as compressed:
 with tarfile.open(fileobj=compressed,mode='w|') as archive:
  info=tarfile.TarInfo('offload-plan.json');info.size=len(raw);info.mode=0o444
  archive.addfile(info,io.BytesIO(raw))
  for row in plan['files']:
   assert time.monotonic()-started<1200,'Bounded backup deadline'
   p=check(row);info=tarfile.TarInfo(str(p).lstrip('/'));info.size=row['bytes'];info.mode=0o444
   with p.open('rb') as f:archive.addfile(info,f)
writer.flush();op,after=current()
receipt={'status':'complete_remote_stream_originals_unchanged','archive_bytes':writer.bytes,'archive_sha256':writer.h.hexdigest(),'plan_sha256':PIN,'files':len(plan['files']),'completed_utc':datetime.now(timezone.utc).isoformat(),'scientific_settings_changed':False}
write(root/'backup-complete.json',receipt)
sys.stderr.write(json.dumps(receipt)+'\n');sys.stderr.flush()
'''

EXECUTE = r'''
assert root.is_dir() and not root.is_symlink() and sha(root/'plan.json')==PIN
assert not os.path.lexists(root/'execution-started.json'),'Existing/unknown deletion outcome; never replay'
backup=json.loads((root/'backup-complete.json').read_bytes())
assert backup['status'] in ('complete_remote_stream_originals_unchanged','authenticated_exact_subset_original_failure_preserved')
assert backup['archive_sha256']==ATTEST['archive_sha256'] and backup['archive_bytes']==ATTEST['archive_bytes']
assert ATTEST['status']=='complete_laptop_archive_every_member_verified' and ATTEST['plan_sha256']==PIN and ATTEST['files']==len(plan['files'])
before=activity();op,adoptions=current()
write(root/'execution-started.json',{'plan_sha256':PIN,'laptop_verification':ATTEST,'activity':before})
removed=[];retained=[];probes=[];total=0;last_probe=0;started=time.monotonic()
def capacity():
 parent=root/('capacity-'+str(len(probes)).zfill(4));parent.mkdir()
 try:
  guard=op.module(op.GUARD_SOURCE/'scripts/protenix_precision_ops.py','_cleanup_original_guard')
  original=guard.storage(parent,'entry')
  full={a:op.c.projection(adoptions[a]['unit_bytes'],a) for a in ('protenix','boltz')}
  required=original['required_bytes']+op.c.CONTROL_CAP+sum(full.values())
  result={'status':'passed' if original['available_bytes']>=required else 'failed','available_bytes':original['available_bytes'],'required_bytes':required,'full_chemistry_projections':full,'reserve_bytes':original['reserve_bytes'],'file_reserve':original['file_reserve'],'original_gate_changed':False,'cleanup_extra_headroom_bytes':1024**3}
  result['cleanup_stop_with_extra_headroom']=result['status']=='passed' and original['available_bytes']>=required+1024**3
 except Exception as e:result={'status':'failed','error':repr(e),'cleanup_stop_with_extra_headroom':False}
 write(parent/'whole-chemistry-capacity.json',result);probes.append(result);return result
with (root/'journal.jsonl').open('x',encoding='utf8') as journal:
 for index,row in enumerate(plan['files']):
  assert time.monotonic()-started<1200,'Bounded retirement deadline; inspect journal before recovery'
  if total-last_probe>=1024**3:
   result=capacity();last_probe=total
   if result['cleanup_stop_with_extra_headroom']:
    retained.extend({'path':r['path'],'reason':'Enough verified capacity plus1GiB extra; not removed'} for r in plan['files'][index:]);break
  try:p=check(row)
  except (AssertionError,FileNotFoundError):
   retained.append({'path':row['path'],'reason':'Absent/changed/linked; retained'});continue
  old_mode=None;source_manifest=None;source_pin=None
  if row['kind']=='redundant_old_source_archive_extracted_sealed_tree_retained':
   assert p.name=='source.tar.gz' and p.parent.parent==Path('/work3/s234818/mosaic/releases')
   source_manifest=p.parent/'source/source_manifest.json'
   if not source_manifest.is_file() or source_manifest.is_symlink():
    retained.append({'path':row['path'],'reason':'Extracted source manifest not independently available; retained'});continue
   source_pin=sha(source_manifest);old_mode=stat.S_IMODE(p.parent.stat().st_mode)
  intent={'event':'intent','path':row['path'],'bytes':row['bytes'],'sha256':row['sha256'],
          'parent_mode_restored':old_mode,'retained_source_manifest_sha256':source_pin}
  journal.write(json.dumps(intent)+'\n');journal.flush();os.fsync(journal.fileno())
  try:
   if old_mode is not None:p.parent.chmod(old_mode|0o200)
   p.unlink()
  finally:
   if old_mode is not None:p.parent.chmod(old_mode)
  if source_manifest is not None:assert sha(source_manifest)==source_pin
  total+=row['bytes'];removed.append(row)
  journal.write(json.dumps({'event':'removed','path':row['path']})+'\n');journal.flush();os.fsync(journal.fileno())
final=capacity();op,after=current();assert all(v['status']=='passed' for v in after.values())
for row in removed:assert not os.path.lexists(row['path'])
receipt={'status':'completed_exact_verified_retirement','removed_files':len(removed),'removed_logical_bytes':total,'retained_files':len(retained),'retained':retained,'plan_sha256':PIN,'journal_sha256':sha(root/'journal.jsonl'),'all_current_native_source_proofs_passed_after':True,'capacity':final,'scientific_settings_changed':False,'completed_utc':datetime.now(timezone.utc).isoformat()}
write(root/'completion.json',receipt);print(json.dumps(receipt))
'''


def run(phase):
    raw,value=plan();out=LOCAL/(phase+'-01');out.mkdir(exist_ok=False)
    prefix='PLAN='+repr(base64.b64encode(raw).decode())+'\nPIN='+repr(hashlib.sha256(raw).hexdigest())+'\n'
    if phase=='backup':
        assert shutil.disk_usage(REPO).free>sum(r['bytes'] for r in value['files'])+2*1024**3
        suffix=BACKUP;target=out/'inactive-results.tar.gz';error=out/'remote-receipt.txt'
    else:
        attestation=json.loads((LOCAL/'backup-01/laptop-verification.json').read_bytes())
        assert sha(LOCAL/'backup-01/inactive-results.tar.gz')==attestation['archive_sha256']
        prefix+='ATTEST='+repr(attestation)+'\n';suffix=EXECUTE;target=out/'stdout.json';error=out/'stderr.txt'
    program=prefix+COMMON+suffix
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    write(out/'started.json',{'status':'unknown_until_receipt_verified','phase':phase,'program_sha256':hashlib.sha256(program.encode()).hexdigest()})
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(PYTHON+' -')]
    receipt={'status':'unknown_inspect_before_recovery'}
    try:
        with target.open('xb') as output,error.open('xb') as err:
            result=subprocess.run(command,input=program.encode(),stdout=output,stderr=err,timeout=1300)
        receipt.update(returncode=result.returncode,completed_utc=datetime.now(timezone.utc).isoformat())
        assert result.returncode==0,error.read_text(errors='replace')[-3000:]
        if phase=='backup':
            remote=stream_receipt(error)
            assert remote['archive_bytes']==target.stat().st_size and remote['archive_sha256']==sha(target)
            expected={r['path'].lstrip('/'):r for r in value['files']};verified=set()
            with tarfile.open(target,'r:gz') as archive:
                for member in archive:
                    path=PurePosixPath(member.name)
                    assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name not in verified
                    verified.add(member.name)
                    with archive.extractfile(member) as stream:
                        if member.name=='offload-plan.json':assert stream.read()==raw
                        else:
                            row=expected[member.name]
                            assert member.size==row['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256']
                assert verified==set(expected)|{'offload-plan.json'}
            write(out/'laptop-verification.json',{'status':'complete_laptop_archive_every_member_verified','archive_sha256':sha(target),'archive_bytes':target.stat().st_size,'files':len(expected),'plan_sha256':hashlib.sha256(raw).hexdigest(),'scientific_values_changed':False})
            receipt['status']='complete_laptop_backup_all_members_verified_no_deletion'
        else:
            value=json.loads(target.read_bytes());assert value['all_current_native_source_proofs_passed_after'] is True
            receipt['status']=value['status']
        print(json.dumps(receipt));print(json.dumps(value if phase=='execute' else remote))
    finally:write(out/'result.json',receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['backup','execute','verify-existing'])
    phase=parser.parse_args().phase
    if phase=='verify-existing':verify_existing_backup()
    else:run(phase)
