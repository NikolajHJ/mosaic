"""Retire only exact verified older-work3 byte copies under the existing cleanup scope."""
from datetime import datetime,timezone
import hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
h=Path(__file__).resolve().parent;repo=h.parents[3]
meta=json.loads((h/'cleanup-plan-02.receipt.json').read_text())
plan=Path(meta['plan_local_path']);helper=h/'cleanup_verified_work3.py'
assert hashlib.sha256(plan.read_bytes()).hexdigest()==meta['plan_sha256']
assert hashlib.sha256(helper.read_bytes()).hexdigest()==meta['helper_sha256']
control='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/cleanup-recovery-02'
upload=control+'/cleanup-inputs.tar.gz'
out=repo/'results/hpc'/h.name/'cleanup-inputs-02.tar.gz'
with out.open('xb') as stream:
 with tarfile.open(fileobj=stream,mode='w:gz') as archive:
  for name,path in [('cleanup-plan.json',plan),('cleanup_verified_work3.py',helper)]:
   b=path.read_bytes();t=tarfile.TarInfo(name);t.size=len(b);t.mode=0o444;t.mtime=0;archive.addfile(t,io.BytesIO(b))
size=out.stat().st_size;digest=hashlib.sha256(out.read_bytes()).hexdigest()
ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc']
code='''from pathlib import Path
import hashlib,sys,json,tarfile,subprocess,os
control=Path(sys.argv[1]);upload=control/'cleanup-inputs.tar.gz'
assert control.resolve()==control and not upload.exists() and not (control/'attempt-01').exists()
assert (control/'backup-receipt.json').is_file()
with upload.open('xb') as stream:
 for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):stream.write(block)
assert upload.stat().st_size==int(sys.argv[2]) and hashlib.sha256(upload.read_bytes()).hexdigest()==sys.argv[3]
with tarfile.open(upload) as archive:
 members=archive.getmembers();assert len(members)==2 and {m.name for m in members}=={'cleanup-plan.json','cleanup_verified_work3.py'} and all(m.isfile() for m in members)
 for m in members:
  target=control/m.name
  with target.open('xb') as stream:stream.write(archive.extractfile(m).read())
  target.chmod(0o444)
assert hashlib.sha256((control/'cleanup-plan.json').read_bytes()).hexdigest()==sys.argv[4]
assert hashlib.sha256((control/'cleanup_verified_work3.py').read_bytes()).hexdigest()==sys.argv[5]
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(control/'cleanup_verified_work3.py'),'--plan',str(control/'cleanup-plan.json'),'--plan-sha256',sys.argv[4],'--attempt','attempt-01'],check=True)
'''
command=' '.join(shlex.quote(v) for v in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',code,control,str(size),digest,meta['plan_sha256'],meta['helper_sha256']])
record={'started_utc':datetime.now(timezone.utc).isoformat(),'outcome':'unknown','archive_sha256':digest,'archive_bytes':size,'plan_sha256':meta['plan_sha256'],'helper_sha256':meta['helper_sha256'],'control':control,'scope':'Exact matching locally backed-up inactive older-work3 files only; current three studies/allhome/releases/cache/model/env untouched'}
with (h/'cleanup-execution-02-started.json').open('x') as f:json.dump(record,f,indent=2)
try:
 with out.open('rb') as stream:r=subprocess.run(ssh+[command],stdin=stream,capture_output=True,timeout=1200)
 (h/'cleanup-execution-02-stdout.txt').write_bytes(r.stdout);(h/'cleanup-execution-02-stderr.txt').write_bytes(r.stderr)
 record.update(returncode=r.returncode,outcome='completed' if r.returncode==0 else 'failed_or_partial_inspect_journal')
 rows=[json.loads(line) for line in r.stdout.decode().splitlines() if line.startswith('{')]
 if r.returncode==0:
  assert rows and rows[-1]['status']=='completed'
  record['receipt']=rows[-1]
  with (h/'cleanup-completion-02.json').open('x') as f:json.dump(rows[-1],f,indent=2);f.write('\n')
 else:raise RuntimeError(r.stderr.decode(errors='replace')[-2000:])
except Exception as error:
 record['error']=repr(error)
 raise
finally:
 record['finished_utc']=datetime.now(timezone.utc).isoformat()
 with (h/'cleanup-execution-02-result.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record))
