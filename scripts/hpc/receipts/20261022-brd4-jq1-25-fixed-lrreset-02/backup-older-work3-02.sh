set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,os,re,stat,subprocess,sys,tarfile
base=Path('/work3/s234818/mosaic')
control=base/'runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/cleanup-recovery-02'
targets=[base/'runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study',
 base/'runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study',
 base/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001/payload',
 base/'runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001.tar.gz',
 base/'runs/20261021-brd4-jq1-25-fixed-02/native-evidence-01.tar.gz',
 base/'20261021-brd4-jq1-25-fixed-02.tar.gz',
 base/'20261021-brd4-jq1-25-fixed-lrboost-01.tar.gz']
if control.exists() or control.resolve()!=control:raise ValueError('Backup/control exists or unsafe; inspect, never duplicate')
control.mkdir()
def exclusive(name,v):
 with (control/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
exclusive('backup-started.json',{'utc':datetime.now(timezone.utc).isoformat(),'targets':[str(p) for p in targets],'scope':'Byte backup only; no deletion'})
r=subprocess.run(['bjobs','-noheader','-o','jobid stat job_name:100 command:2000'],capture_output=True,text=True,timeout=30)
if r.returncode:raise ValueError('Unfinished-job inventory unavailable')
groups=[str(p) for p in targets]
for line in r.stdout.splitlines():
 if any(s in line for s in groups):raise ValueError('An unfinished job directly references a backup target')
exclusive('jobs-before-backup.json',{'returncode':r.returncode,'raw':r.stdout,'stderr':r.stderr})
all_files=[]
for p in targets:
 if not p.exists() or p.is_symlink() or p.resolve()!=p:raise ValueError('Missing/linked backup target: '+str(p))
 if p.is_file():all_files.append(p);continue
 for root,dirs,files in os.walk(p,followlinks=False):
  for name in dirs+files:
   q=Path(root)/name;s=q.lstat()
   if stat.S_ISLNK(s.st_mode) or not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):raise ValueError('Unsafe backup entry')
   if stat.S_ISREG(s.st_mode):all_files.append(q)
all_files=sorted(all_files)
if len(all_files)!=len(set(all_files)):raise ValueError('Duplicate backup member')
total=sum(p.stat().st_size for p in all_files)
if not 0<total<=2*1024**3:raise ValueError('Declared twoGiB backup/removal cap exceeded')
class Writer:
 def __init__(self):self.digest=hashlib.sha256();self.size=0;self.buffer=bytearray()
 def write(self,b):
  self.digest.update(b);self.size+=len(b);self.buffer.extend(b)
  if len(self.buffer)>=1048576:sys.stdout.buffer.write(self.buffer);self.buffer.clear()
  return len(b)
 def flush(self):
  if self.buffer:sys.stdout.buffer.write(self.buffer);self.buffer.clear()
  sys.stdout.buffer.flush()
writer=Writer();manifest={};identities={}
with tarfile.open(fileobj=writer,mode='w|gz',compresslevel=0) as archive:
 for p in all_files:
  before=p.lstat()
  if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1:raise ValueError('Nonregular/hardlinked source')
  with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
  name='files/'+p.relative_to(base).as_posix()
  info=tarfile.TarInfo(name);info.size=before.st_size;info.mode=0o444;info.mtime=0
  with p.open('rb') as f:archive.addfile(info,f)
  after=p.lstat()
  identity=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
  if identity!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Source changed during backup')
  manifest[name]={'bytes':before.st_size,'sha256':digest}
  identities[name]={'path':str(p),'device':before.st_dev,'inode':before.st_ino,'mtime_ns':before.st_mtime_ns}
 def add(name,b):
  info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o444;info.mtime=0;archive.addfile(info,io.BytesIO(b))
 add('origin-map.json',(json.dumps(identities,indent=2)+'\n').encode())
 # The same manifest format is independently verified on the user's machine.
 origin_bytes=(json.dumps(identities,indent=2)+'\n').encode();manifest['origin-map.json']={'bytes':len(origin_bytes),'sha256':hashlib.sha256(origin_bytes).hexdigest()}
 add('evidence_manifest.json',(json.dumps({'files':manifest},indent=2)+'\n').encode())
writer.flush()
receipt={'status':'backup_stream_completed','utc':datetime.now(timezone.utc).isoformat(),'bytes':writer.size,'sha256':writer.digest.hexdigest(),'original_files':len(all_files),'original_logical_bytes':total,'control':str(control),'deletion_performed':False}
exclusive('backup-receipt.json',receipt)
print(json.dumps(receipt),file=sys.stderr,flush=True)
PY
