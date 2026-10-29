"""Exact laptop backup before retiring inactive source/report copies, metadata only."""
import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile

REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
LOCAL=REPO/'results/hpc/20261029-shared-writer-offload-24'
RUN=OWNER/'runs/20261029-shared-writer-offload-24'
BACKUP=Path('C:/Users/nikol/HPC_Archives/Mosaic/2026-10-29/shared-writer-24')
COMMON=r'''
import os,sys,json,hashlib,stat,re,importlib.util,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
audit=base/'shared-writer-offload-24';source=base/'nonchem-precision-release-24/source'
pin='4c2a86ec4a8f65aa37d35df94416fe6d8995e74d666d26f186629287f7ee725b'
sys.dont_write_bytecode=True;sys.path.insert(0,str(source/'scripts'))
spec=importlib.util.spec_from_file_location('_offload24_current',source/'scripts/precision_continuation_ops.py')
op=importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def current():
 m=op.verify(source,pin);plan=op.read(source/'plan.json');result=op.external(m,plan)
 protected=set()
 def collect(v):
  if isinstance(v,str) and v.startswith('/'):protected.add(v)
  elif isinstance(v,dict):
   for x in v.values():collect(x)
  elif isinstance(v,list):
   for x in v:collect(x)
 collect(m);collect(plan)
 for env in (Path('/work3/s234818/mosaic/envs/protenix-tiny'),Path('/work3/s234818/latentmol/env/boltz311')):
  for site in (env/'lib').glob('python*/site-packages'):
   for pattern in ('*.pth','*.egg-link','__editable__*.py'):
    for metadata in site.glob(pattern):
     assert metadata.stat().st_size<100000
     for path in re.findall(r'/work3/[^\s\x27\x22,;)}]+',metadata.read_text()):protected.add(path)
 scheduler=subprocess.run(['bjobs','-json','-o','jobid stat queue command'],capture_output=True,text=True,timeout=45)
 assert scheduler.returncode==0,scheduler.stderr
 return protected,json.loads(scheduler.stdout)['RECORDS'],result
def check(row,removed_links=None):
 p=Path(row['path']);assert p.is_relative_to(Path('/work3/s234818/mosaic'))
 assert all(not q.is_symlink() for q in (p,*p.parents))
 i=p.lstat();assert stat.S_ISREG(i.st_mode) and i.st_uid==os.getuid()
 expected=list(row['identity'])
 if removed_links is not None:expected[4]-=removed_links.get((expected[0],expected[1]),0)
 assert [i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_nlink,i.st_mode]==expected
 assert digest(p)==row['sha256'];return p
'''


def raw(v):return (json.dumps(v,indent=2,sort_keys=True)+'\n').encode()
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def transport():
    s=importlib.util.spec_from_file_location('_offload_transport',OWNER/'scripts/manage_protenix_precision.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.LOCAL=LOCAL;return m


def plan():
    assert (RUN/'README.md').is_file()
    LOCAL.mkdir(exist_ok=False)
    code=COMMON+r'''
assert not audit.exists(),'Existing/unknown cleanup: inspect, never replay'
protected,jobs,adoption=current();mosaic=Path('/work3/s234818/mosaic');roots=[];rows=[]
for d in sorted((mosaic/'releases').iterdir()):
 if d.is_dir() and re.fullmatch(r'2026101[0-9]-[a-z0-9-]+',d.name):
  assert not d.is_symlink()
  if any(p==str(d) or p.startswith(str(d)+'/') for p in protected):continue
  if any(str(d) in j['COMMAND'] for j in jobs):continue
  roots.append(str(d))
  for p in sorted(d.rglob('*')):
   assert not p.is_symlink() and (p.is_file() or p.is_dir())
   if p.is_file() and p.name not in ('source_manifest.json','README.md'):rows.append(p)
reports=[base/'nonchem-live-report-08/render/viewer',base/'reusable-saved-report-v1/requests/request-20261028T121744Z-5115340e/viewer']
for d in reports:
 assert not any(str(d) in j['COMMAND'] for j in jobs)
 roots.append(str(d))
 for p in sorted(d.glob('*.json')):
  if re.fullmatch(r'[a-z0-9_-]+-(?:frames-[0-9]{4}|scalars)\.json',p.name):rows.append(p)
old=base/'nonchem-live-report-07/edition-0004.tar.gz'
if old.is_file():rows.append(old);roots.append(str(old))
files=[]
for p in rows:
 assert str(p) not in protected and all(not q.is_symlink() for q in (p,*p.parents))
 i=p.lstat();assert stat.S_ISREG(i.st_mode) and i.st_uid==os.getuid()
 files.append({'path':str(p),'bytes':i.st_size,'sha256':digest(p),'allocated_bytes':i.st_blocks*512,
    'identity':[i.st_dev,i.st_ino,i.st_size,i.st_mtime_ns,i.st_nlink,i.st_mode]})
assert files and len({r['path'] for r in files})==len(files)
assert sum(r['allocated_bytes'] for r in files)>2500000000,'No sufficient safe inactive selection; no deletion'
value={'version':'exact-inactive-sources-report-copies-v1','created_utc':datetime.now(timezone.utc).isoformat(),
 'files':files,'allowed_roots':roots,'protected_paths':sorted(protected),'adoption':adoption,
 'saved_native_results_changed':False,'external_jobs_changed':False}
audit.mkdir();write(audit/'plan.json',value)
print(json.dumps(value))
'''
    v=transport().command(code,'plan01',420)
    (LOCAL/'plan.json').write_bytes(raw(v));(RUN/'plan-summary.json').write_bytes(raw({
        'files':len(v['files']),'logical_bytes':sum(r['bytes'] for r in v['files']),
        'allocated_bytes':sum(r['allocated_bytes'] for r in v['files']),'roots':v['allowed_roots'],
        'status':'planned_no_deletion','plan_sha256':sha(LOCAL/'plan.json')}))
    print((RUN/'plan-summary.json').read_text())


def backup():
    plan=json.loads((LOCAL/'plan.json').read_bytes())
    amount=sum(r['bytes'] for r in plan['files']);assert shutil.disk_usage(REPO).free>amount+2*1024**3
    BACKUP.mkdir(parents=True,exist_ok=False)
    prefix='PLAN='+repr(plan)+'\n'
    code=prefix+COMMON+r'''
import gzip,io,tarfile
assert audit.exists() and not (audit/'backup-started.json').exists()
protected,jobs,before=current();assert not any(r['path'] in protected for r in PLAN['files'])
assert not any(any(str(root) in j['COMMAND'] for root in PLAN['allowed_roots']) for j in jobs)
write(audit/'backup-started.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':len(PLAN['files'])})
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.bytes=0
 def write(self,b):sys.stdout.buffer.write(b);self.h.update(b);self.bytes+=len(b);return len(b)
 def flush(self):sys.stdout.buffer.flush()
writer=Writer();start=time.monotonic()
with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as zipped:
 with tarfile.open(fileobj=zipped,mode='w|') as tar:
  b=(json.dumps(PLAN,sort_keys=True)+'\n').encode();i=tarfile.TarInfo('restore-map.json');i.size=len(b);tar.addfile(i,io.BytesIO(b))
  for row in PLAN['files']:
   assert time.monotonic()-start<1200,'Bounded archive transfer'
   p=check(row);i=tarfile.TarInfo(str(p).lstrip('/'));i.size=row['bytes'];i.mode=stat.S_IMODE(row['identity'][-1])
   with p.open('rb') as f:tar.addfile(i,f)
writer.flush();current()
receipt={'status':'backup_complete_originals_unchanged','files':len(PLAN['files']),'archive_bytes':writer.bytes,'archive_sha256':writer.h.hexdigest()}
write(audit/'backup-complete.json',receipt);sys.stderr.write(json.dumps(receipt)+'\n')
'''
    out=LOCAL/'backup01';out.mkdir(exist_ok=False);(out/'program.py').write_text(code,encoding='utf8',newline='\n')
    m=transport();argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
                      'bash -l -c '+shlex.quote(m.PYTHON+' -')]
    with (BACKUP/'data.tar.gz').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
        result=subprocess.run(argv,input=code.encode(),stdout=stdout,stderr=stderr,timeout=1250)
    (out/'receipt.json').write_bytes(raw({'returncode':result.returncode,'archive':str(BACKUP/'data.tar.gz')}))
    assert result.returncode==0,(out/'stderr.txt').read_text()[-2500:]
    receipt=json.loads((out/'stderr.txt').read_text().splitlines()[-1]);assert sha(BACKUP/'data.tar.gz')==receipt['archive_sha256']
    assert (BACKUP/'data.tar.gz').stat().st_size==receipt['archive_bytes']
    expected={r['path'].lstrip('/'):r for r in plan['files']};seen=set()
    with tarfile.open(BACKUP/'data.tar.gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name);assert member.isfile() and not name.is_absolute() and '..' not in name.parts and member.name not in seen
            seen.add(member.name)
            with tar.extractfile(member) as stream:
                if member.name=='restore-map.json':assert json.load(stream)==plan
                else:
                    r=expected[member.name];assert member.size==r['bytes'] and hashlib.file_digest(stream,'sha256').hexdigest()==r['sha256']
    assert seen==set(expected)|{'restore-map.json'}
    receipt.update(status='all_laptop_archive_members_verified',archive=str(BACKUP/'data.tar.gz'),logical_bytes=amount)
    (BACKUP/'restore-map.json').write_bytes(raw(plan));(BACKUP/'verification.json').write_bytes(raw(receipt))
    (LOCAL/'laptop-verification.json').write_bytes(raw(receipt));print(json.dumps(receipt))


def retire():
    verification=json.loads((LOCAL/'laptop-verification.json').read_bytes());assert verification['status']=='all_laptop_archive_members_verified'
    assert sha(verification['archive'])==verification['archive_sha256']
    plan=json.loads((LOCAL/'plan.json').read_bytes())
    code='PLAN='+repr(plan)+'\nLAPTOP='+repr(verification)+'\n'+COMMON+r'''
assert (audit/'backup-complete.json').exists() and not (audit/'retire-started.json').exists(),'Unknown/existing retirement; inspect journal'
backup=json.loads((audit/'backup-complete.json').read_bytes());assert backup['archive_sha256']==LAPTOP['archive_sha256'] and backup['archive_bytes']==LAPTOP['archive_bytes']
protected,jobs,before=current();assert not any(r['path'] in protected for r in PLAN['files'])
assert not any(any(root in j['COMMAND'] for root in PLAN['allowed_roots']) for j in jobs)
for row in PLAN['files']:check(row)
write(audit/'retire-started.json',{'laptop':LAPTOP,'utc':datetime.now(timezone.utc).isoformat()})
modes={};removed=0;allocated=0;removed_links={}
try:
 with (audit/'journal.jsonl').open('x') as journal:
  for row in PLAN['files']:
   p=check(row,removed_links);parent=p.parent;before_stat=p.lstat()
   if parent not in modes:modes[parent]=stat.S_IMODE(parent.stat().st_mode)
   parent.chmod(modes[parent]|0o200)
   journal.write(json.dumps({'event':'intent',**row})+'\n');journal.flush();os.fsync(journal.fileno())
   p.unlink();assert not p.exists();removed+=1
   key=(before_stat.st_dev,before_stat.st_ino);removed_links[key]=removed_links.get(key,0)+1
   if before_stat.st_nlink==1:allocated+=before_stat.st_blocks*512
   journal.write(json.dumps({'event':'removed','path':str(p)})+'\n');journal.flush();os.fsync(journal.fileno())
finally:
 for parent,mode in modes.items():parent.chmod(mode)
after=current();assert all(not Path(r['path']).exists() for r in PLAN['files'])
quota=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=45)
value={'status':'exact_inactive_copies_offloaded_after_verified_laptop_backup','files':removed,'allocated_bytes_nlink1':allocated,
 'logical_bytes':sum(r['bytes'] for r in PLAN['files']),'journal_sha256':digest(audit/'journal.jsonl'),'laptop':LAPTOP,
 'current_inputs_sources_results_reauthenticated':True,'saved_native_results_changed':False,'external_jobs_changed':False,
 'quota':{'returncode':quota.returncode,'stdout':quota.stdout,'stderr':quota.stderr},'utc':datetime.now(timezone.utc).isoformat()}
write(audit/'completion.json',value);print(json.dumps(value))
'''
    v=transport().command(code,'retire01',420)
    (RUN/'completion.json').write_bytes(raw(v));(LOCAL/'completion.json').write_bytes(raw(v));print(json.dumps({k:v[k] for k in ('status','files','allocated_bytes_nlink1','logical_bytes','utc')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('plan','backup','retire'));a=p.parse_args()
    {'plan':plan,'backup':backup,'retire':retire}[a.phase]()
