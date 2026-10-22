set -euo pipefail
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
"$python" - <<'PY'
from pathlib import Path
import hashlib,json,tarfile,subprocess,os,re
from datetime import datetime,timezone
source=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-01/source');root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-01');upload=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/20261022-brd4-jq1-25-fixed-lrreset-01.tar.gz')
expected='47263fedea42e593455b43d4b75c1c874670bb53d934dd93a531aec439abf403';archive_sha='838389193a60d2828ceebd8a4c3cc0d38abcc10a76accfaa91b9137eb9ecad08';archive_size=39600744
assert not source.parent.exists() and not root.exists()
assert upload.stat().st_size==archive_size and hashlib.sha256(upload.read_bytes()).hexdigest()==archive_sha
with tarfile.open(upload) as archive:
 members=archive.getmembers()
 assert len(members)==623 and all(m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts for m in members)
 assert len({m.name for m in members})==623
 manifest_bytes=archive.extractfile('source/source_manifest.json').read()
 assert hashlib.sha256(manifest_bytes).hexdigest()==expected
 manifest=json.loads(manifest_bytes)
 assert len(manifest['files'])==622
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
for name in ('fixed25_lrreset_cpu.lsf','fixed25_lrreset_stage.lsf'):subprocess.run(['bash','-n',str(source/'scripts/hpc'/name)],check=True)
env={**os.environ,'PYTHONPATH':str(source/'scripts'),'PYTHONDONTWRITEBYTECODE':'1'}
subprocess.run(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python',str(source/'scripts/fixed25_lrreset_ops.py'),'--source',str(source),'--root',str(root),'--manifest',expected,'--phase','preflight'],env=env,check=True)
def exclusive(path,obj):
 with path.open('x') as stream:json.dump(obj,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
exclusive(root/'source-staging.json',{'status':'passed','source_manifest_sha256':expected,'archive_sha256':archive_sha,'archive_bytes':archive_size,'files':622,'utc':datetime.now(timezone.utc).isoformat()})
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
cpu_args=['bsub','-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-jq1-lrreset-cpu','-oo',str(root/'cpu/stdout.%J.log'),'-eo',str(root/'cpu/stderr.%J.log'),'bash','-l',str(source/'scripts/hpc/fixed25_lrreset_cpu.lsf'),str(source),str(root),expected]
cpu=submit(root/'cpu',cpu_args)
exclusive(root/'initial-submissions.json',{'cpu':cpu,'callback':None,'all_initial_submitted':False})
callback_args=['bsub','-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:30','-w','ended('+cpu['job_id']+')','-J','mosaic-jq1-lrreset-after-cpu','-oo',str(root/'after-cpu/stdout.%J.log'),'-eo',str(root/'after-cpu/stderr.%J.log'),'bash','-l',str(source/'scripts/hpc/fixed25_lrreset_stage.lsf'),str(source),str(root),expected,'after-cpu','none',str(root/'after-cpu'),cpu['job_id'],'0']
callback=submit(root/'after-cpu',callback_args)
exclusive(root/'initial-submissions-complete.json',{'cpu':cpu,'callback':callback,'all_initial_submitted':True})
print(json.dumps({'status':'staged_and_cpu_submitted','cpu_job':cpu['job_id'],'callback_job':callback['job_id'],'source_manifest_sha256':expected}),flush=True)
PY
