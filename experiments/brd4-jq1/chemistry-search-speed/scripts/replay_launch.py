"""Once-only new diagnostic staging/CPU submission; inspect unknown outcomes."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-chemistry-replay-01'
REMOTE=r'''
import base64,hashlib,importlib.util,io,json,os,stat,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
release=base/'chemistry-replay-release-01';source=release/'source';root=base/'chemistry-replay-01'
def require(value,message):
 if not value:raise ValueError(message)
require(not root.exists() and not release.exists(),'Existing/unknown replay root/release prohibits staging replay')
data=base64.b64decode(ARCHIVE,validate=True)
require(len(data)==LENGTH and hashlib.sha256(data).hexdigest()==SHA,'Archive identity differs')
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 rows=archive.getmembers()
 require(len(rows)==MEMBERS and all(p.isfile() for p in rows) and len({p.name for p in rows})==len(rows),'Archive scope/type differs')
 for p in rows:
  rel=PurePosixPath(p.name)
  require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in p.name,'Unsafe archive member')
  members[p.name]=archive.extractfile(p).read()
require(hashlib.sha256(members['source_manifest.json']).hexdigest()==MANIFEST,'Manifest differs')
declared=json.loads(members['source_manifest.json'])
require(set(members)==set(declared['files'])|{'source_manifest.json'},'Manifest membership differs')
for name,pin in declared['files'].items():require(hashlib.sha256(members[name]).hexdigest()==pin,'Source member hash differs')
release.mkdir();source.mkdir()
(release/'source.tar.gz').write_bytes(data)
for name,data in members.items():
 path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as output:output.write(data)
 path.chmod(0o444)
for path in sorted((p for p in source.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.chmod(0o555)
source.chmod(0o555)
sys.path.insert(0,str(source/'scripts'))
import replay_ops
replay_ops.own_source(MANIFEST)
cpu,after=replay_ops.bootstrap(MANIFEST)
receipt={'status':'replay_CPU_and_own_callback_submitted_not_passed','cpu':cpu,'after_cpu':after,
 'source_manifest_sha256':MANIFEST,'source_files':MEMBERS-1,'archive_bytes':LENGTH,'archive_sha256':SHA,
 'production_authorized':False,'GPU_submitted':False,'optimizer_updates':0,'model_evaluations':0}
replay_ops.write(root/'staging-complete.json',receipt);replay_ops.caps();print(json.dumps(receipt))
'''


def main():
 if REPO.name!='mosaic':raise ValueError('Wrong authorized repository')
 package=LOCAL/'source01';preparation=json.loads((package/'preparation.json').read_bytes());data=(package/'source.tar.gz').read_bytes()
 if len(data)!=preparation['archive_bytes'] or hashlib.sha256(data).hexdigest()!=preparation['archive_sha256']:
  raise ValueError('Fresh local source capsule differs')
 output=LOCAL/'launch-01';output.mkdir(exist_ok=False)
 code='\n'.join(key+'='+repr(value) for key,value in {'ARCHIVE':base64.b64encode(data).decode(),
  'LENGTH':len(data),'SHA':preparation['archive_sha256'],'MANIFEST':preparation['manifest_sha256'],'MEMBERS':preparation['files']+1}.items())+'\n'+REMOTE
 (output/'remote-program.py').write_text(code,encoding='utf8',newline='\n')
 (output/'started.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),
  'program_sha256':hashlib.sha256(code.encode()).hexdigest(),'unknown_outcome_policy':'Never replay; inspect exact remote receipts first'},indent=2)+'\n')
 command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
  'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
 receipt={'status':'unknown_inspect_remote_before_recovery'}
 try:
  with (output/'stdout.json').open('xb') as out,(output/'stderr.txt').open('xb') as err:
   result=subprocess.run(command,input=code.encode(),stdout=out,stderr=err,timeout=300)
  receipt.update(returncode=result.returncode,stdout_sha256=hashlib.sha256((output/'stdout.json').read_bytes()).hexdigest(),
   completed_utc=datetime.now(timezone.utc).isoformat())
  if result.returncode==0:
   delivery=json.loads((output/'stdout.json').read_bytes())
   if not (delivery['status']=='replay_CPU_and_own_callback_submitted_not_passed'
    and delivery['source_manifest_sha256']==preparation['manifest_sha256']
    and delivery['archive_bytes']==preparation['archive_bytes'] and delivery['archive_sha256']==preparation['archive_sha256']
    and delivery['source_files']==preparation['files'] and delivery['production_authorized'] is False
    and delivery['GPU_submitted'] is False and delivery['optimizer_updates']==0 and delivery['model_evaluations']==0):
    raise ValueError('Returned submission receipt is detached from the prepared source/scope')
   for phase,key in (('cpu','cpu'),('after-cpu','after_cpu')):
    sub=delivery[key]
    if not (sub['status']=='submitted' and sub['phase']==phase and sub['source_manifest_sha256']==preparation['manifest_sha256']
     and sub['returncode']==0 and re.fullmatch('[1-9][0-9]*',sub['job_id']) is not None
     and re.findall(r'Job <([1-9][0-9]*)> is submitted',sub['stdout'])==[sub['job_id']]):
     raise ValueError('Returned actual phase submission is not authenticated')
   if delivery['cpu']['dependency'] is not None or delivery['after_cpu']['dependency']!='ended('+delivery['cpu']['job_id']+')':
    raise ValueError('Returned own-ended CPU dependency differs')
   receipt.update(status='actual_CPU_and_callback_submission_receipt_authenticated_not_passed',
    cpu_job_id=delivery['cpu']['job_id'],callback_job_id=delivery['after_cpu']['job_id'])
 except BaseException as error:receipt['error']=repr(error);raise
 finally:(output/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
 if result.returncode:raise RuntimeError('Staging/admission failed; preserve exact remote/local outcomes and never replay')


if __name__=='__main__':main()
