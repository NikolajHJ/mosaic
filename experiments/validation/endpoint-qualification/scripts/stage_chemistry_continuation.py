"""Single-attempt authenticated metadata staging and conditional CPU submission.

No numerical work runs on the login node. Existing or unknown outcomes are never
replayed. The complete allocated CPU/startup gates own any later GPU dispatch.
"""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]


PROGRAM=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,os,stat,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation']
source=Path(p['source_root']);root=Path(p['root']);scope=root.parent
assert scope==Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-25')
assert source==scope/'chemistry-continuation-release-03/source' and root==scope/'chemistry-continuation-03'
assert scope.is_dir() and all(not x.is_symlink() for x in (scope,*scope.parents))
assert not source.parent.exists() and not root.exists(),'Existing/unknown stage outcome; inspect receipts'
previous=scope/'chemistry-continuation-02'
previous_pin='d87701b3eb422d3e1c1f9255eb61607011b96eb621a67d286ceefe961e36c250'
# Read actual previous outcomes BEFORE staging or deciding whether a native
# validator must be reused. Never race an old still-running dispatcher.
for phase in ['cpu','after-cpu']:
 parent=previous/phase;sub=json.loads((parent/'submission.json').read_bytes())
 assert sub['status']=='submitted' and sub['source_manifest_sha256']==previous_pin
 text=' '.join((parent/('stdout.'+sub['job_id']+'.log')).read_text().split())
 assert text.count('Subject: Job '+sub['job_id']+':')==1 and 'queue <hpc>' in text
 assert ('Successfully completed.' in text)!=('Exited with exit code' in text),'Previous CPU/controller terminal outcome not established'
for arm in ['protenix','boltz']:
 assert not (previous/(arm+'-segment')).exists(),'Previous optimizer/unknown long outcome prevents replacement'
 for phase in ['validate','after-validate']:
  parent=previous/(arm+'-'+phase)
  if parent.exists():
   sub=json.loads((parent/'submission.json').read_bytes())
   assert sub['status']=='submitted' and sub['job_id'].isdigit() and sub['source_manifest_sha256']==previous_pin
data=base64.b64decode(request['archive'],validate=True)
assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 members=archive.getmembers();assert len(members)==p['files']+1
 names=set()
 for member in members:
  rel=PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name and member.name not in names
  names.add(member.name)
 manifest=archive.extractfile('source_manifest.json').read()
 assert hashlib.sha256(manifest).hexdigest()==p['manifest_sha256']
 files=json.loads(manifest)['files'];assert names==set(files)|{'source_manifest.json'}
 for member in members:
  raw=archive.extractfile(member).read()
  if member.name!='source_manifest.json':assert hashlib.sha256(raw).hexdigest()==files[member.name]
 source.mkdir(parents=True)
 for member in members:
  path=source/member.name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(archive.extractfile(member).read())
for path in source.rglob('*'):
 if path.is_file():path.chmod(0o444)
for path in sorted(source.rglob('*'),reverse=True):
 if path.is_dir():path.chmod(0o555)
source.chmod(0o555)
root.mkdir();(root/'stage-receipt.json').write_text(json.dumps(p,indent=2)+'\n')
run=subprocess.run(['bash','-n',str(source/'scripts/chemistry_continuation_stage.lsf')],capture_output=True,text=True)
assert run.returncode==0,run.stderr
sys.path.insert(0,str(source/'scripts'))
import chemistry_continuation_ops as op
manifest,plan=op.own_source(p['manifest_sha256'])
op.storage(root,'preflight',plan)
cpu=op.submit('cpu',p['manifest_sha256'])
callback=op.submit('after-cpu',p['manifest_sha256'],dependency='ended('+cpu['job_id']+')')
result={'status':'allocated_CPU_submitted_not_passed','cpu':cpu,'after_cpu':callback,'source':p,'optimizer_updates':0}
op.c.write(root/'initial-submissions.json',result)
print(json.dumps(result))
'''


def main():
    package=REPO/'results/hpc/20261029-chemistry-continuation-03/source02'
    receipt=json.loads((package/'preparation.json').read_bytes());archive=(package/'source.tar.gz').read_bytes()
    if len(archive)!=receipt['archive_bytes'] or hashlib.sha256(archive).hexdigest()!=receipt['archive_sha256']:
        raise ValueError('Local sealed archive differs')
    output=REPO/'results/hpc/20261029-chemistry-continuation-03'/datetime.now(timezone.utc).strftime('stage-%Y%m%dT%H%M%SZ')
    output.mkdir();(output/'remote-program.py').write_text(PROGRAM,encoding='utf8')
    request=json.dumps({'preparation':receipt,'archive':base64.b64encode(archive).decode()}).encode()
    # -c leaves stdin exclusively for the closed JSON transfer, not shell text.
    command='/work3/s234818/mosaic/envs/protenix-tiny/bin/python -B -c '+shlex.quote(PROGRAM)
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    (output/'started.json').write_text(json.dumps({'status':'outcome_unknown_until_receipt','source':receipt},indent=2)+'\n',encoding='utf8')
    try:
        run=subprocess.run(argv,input=request,capture_output=True,timeout=300)
        (output/'stdout.json').write_bytes(run.stdout);(output/'stderr.txt').write_bytes(run.stderr)
        (output/'transport.json').write_text(json.dumps({'returncode':run.returncode},indent=2)+'\n',encoding='utf8')
        print(output);print(run.stdout.decode());print(run.stderr.decode())
        if run.returncode:raise SystemExit(run.returncode)
    except subprocess.TimeoutExpired as error:
        (output/'timeout.json').write_text(json.dumps({'status':'unknown_remote_outcome_no_blind_replay','error':repr(error)})+'\n',encoding='utf8')
        raise


if __name__=='__main__':main()
