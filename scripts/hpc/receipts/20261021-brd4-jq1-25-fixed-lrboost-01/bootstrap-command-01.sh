set -euo pipefail
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" - <<'PY'
from pathlib import Path
import hashlib,json,tarfile,subprocess,os,re
from datetime import datetime,timezone
source=Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source');root=Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01');upload=Path('/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-lrboost-01.tar.gz')
expected='9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323';archive_sha='e1a100aebf44698515ba161c1a001e338a902b250bf1a737f888df33f55098d1';archive_size=39582615
assert not source.parent.exists() and not root.exists()
assert upload.stat().st_size==archive_size and hashlib.sha256(upload.read_bytes()).hexdigest()==archive_sha
with tarfile.open(upload) as archive:
 members=archive.getmembers()
 assert len(members)==611 and all(m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts for m in members)
 assert len({m.name for m in members})==611
 manifest_bytes=archive.extractfile('source/source_manifest.json').read()
 assert hashlib.sha256(manifest_bytes).hexdigest()==expected
 manifest=json.loads(manifest_bytes)
 assert len(manifest['files'])==610
 for member in members:
  name=member.name.removeprefix('source/')
  if name!='source_manifest.json':assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==manifest['files'][name]
 source.parent.mkdir(exist_ok=False)
 source.mkdir(exist_ok=False)
 for member in members:
  target=source/member.name.removeprefix('source/');target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(archive.extractfile(member).read())
  target.chmod(0o444)
for path in sorted((p for p in source.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.chmod(0o555)
source.chmod(0o555)
root.mkdir(exist_ok=False)
(root/'cpu').mkdir();(root/'after-cpu').mkdir()
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(source/'scripts/verify_stage0_release.py'),str(source),expected],check=True)
for name in ('fixed25_lrboost_cpu.lsf','fixed25_lrboost_stage.lsf'):subprocess.run(['bash','-n',str(source/'scripts/hpc'/name)],check=True)
env={**os.environ,'PYTHONPATH':str(source/'scripts'),'PYTHONDONTWRITEBYTECODE':'1'}
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(source/'scripts/fixed25_lrboost_ops.py'),'--source',str(source),'--root',str(root),'--manifest',expected,'--phase','preflight'],env=env,check=True)
def exclusive(path,obj):
 with path.open('x') as stream:json.dump(obj,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
exclusive(root/'source-staging.json',{'status':'passed','source_manifest_sha256':expected,'archive_sha256':archive_sha,'archive_bytes':archive_size,'files':610,'utc':datetime.now(timezone.utc).isoformat()})
def submit(parent,args):
 exclusive(parent/'submission-started.json',{'args':args,'outcome':'unknown'})
 result=subprocess.run(args,capture_output=True,text=True,timeout=60)
 ids=re.findall(r'Job <(\d+)>',result.stdout)
 record={'args':args,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode,'outcome':'unknown','source_manifest_sha256':expected,'utc':datetime.now(timezone.utc).isoformat()}
 if result.returncode==0 and len(ids)==1:record.update(job_id=ids[0],outcome='submitted')
 exclusive(parent/'submission.json',record)
 print(json.dumps(record),flush=True)
 assert record['outcome']=='submitted','Unknown submission; do not duplicate'
 scheduler=subprocess.run(['bjobs','-l',ids[0]],capture_output=True,text=True,timeout=30)
 (parent/'scheduler-at-submission.txt').write_text(scheduler.stdout+scheduler.stderr)
 return record
cpu_args=['bsub','-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-jq1-lrboost-cpu','-oo',str(root/'cpu/stdout.%J.log'),'-eo',str(root/'cpu/stderr.%J.log'),'bash','-l',str(source/'scripts/hpc/fixed25_lrboost_cpu.lsf'),str(source),str(root),expected]
cpu=submit(root/'cpu',cpu_args)
exclusive(root/'initial-submissions.json',{'cpu':cpu,'callback':None,'all_initial_submitted':False})
callback_args=['bsub','-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:30','-w','ended('+cpu['job_id']+')','-J','mosaic-jq1-lrboost-after-cpu','-oo',str(root/'after-cpu/stdout.%J.log'),'-eo',str(root/'after-cpu/stderr.%J.log'),'bash','-l',str(source/'scripts/hpc/fixed25_lrboost_stage.lsf'),str(source),str(root),expected,'after-cpu','none',str(root/'after-cpu'),cpu['job_id'],'0']
callback=submit(root/'after-cpu',callback_args)
exclusive(root/'initial-submissions-complete.json',{'cpu':cpu,'callback':callback,'all_initial_submitted':True})
print(json.dumps({'status':'staged_and_cpu_submitted','cpu_job':cpu['job_id'],'callback_job':callback['job_id'],'source_manifest_sha256':expected}),flush=True)
PY
