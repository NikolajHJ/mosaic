"""Stage once, authenticate, run metadata storage admission and submit CPU only.

All scientific evaluations and fixtures execute in allocated jobs. Existing
remote paths or unknown outcomes prohibit replay of this one-shot bootstrap.
"""
from datetime import datetime,timezone
import base64
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'
RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-continuation-admission-01'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',choices=('01','02'),default='01');args=parser.parse_args()
    global LOCAL,RUN
    if args.attempt=='02':
        LOCAL=REPO/'results/hpc/20261025-continuation-worker-fix-02'
        RUN=REPO/'experiments/validation/endpoint-qualification/runs/20261025-continuation-worker-fix-02'
    source=LOCAL/'source01';preparation=json.loads((source/'preparation.json').read_bytes())
    archive=(source/'source.tar.gz').read_bytes()
    assert len(archive)==preparation['archive_bytes'] and hashlib.sha256(archive).hexdigest()==preparation['archive_sha256']
    program='ATTEMPT='+repr(args.attempt)+'\nPREPARATION='+repr(preparation)+'\nARCHIVE='+repr(base64.b64encode(archive).decode())+'\n'+r'''
from datetime import datetime,timezone
import base64,hashlib,importlib.util,io,json,os,stat,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=base/('nonchem-endpoint-release-'+ATTEMPT)/'source';root=base/('nonchem-endpoint-continuation-'+ATTEMPT)
assert not source.parent.exists() and not root.exists(),'Existing/unknown attempt: inspect, never replay'
assert base.is_dir() and not base.is_symlink()
raw=base64.b64decode(ARCHIVE,validate=True)
assert len(raw)==PREPARATION['archive_bytes'] and hashlib.sha256(raw).hexdigest()==PREPARATION['archive_sha256']
contents={}
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for member in archive.getmembers():
  rel=PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name
  assert member.name not in contents
  value=archive.extractfile(member).read();assert len(value)==member.size
  contents[member.name]=value
manifest=json.loads(contents['source_manifest.json'])
assert hashlib.sha256(contents['source_manifest.json']).hexdigest()==PREPARATION['manifest_sha256']
assert set(contents)==set(manifest['files'])|{'source_manifest.json'}
assert len(manifest['files'])==PREPARATION['files']
for name,digest in manifest['files'].items():assert hashlib.sha256(contents[name]).hexdigest()==digest
for parent in manifest['parent_sources']:
 p=Path(parent['path']);assert p.is_dir() and not p.is_symlink()
 data=(p/'source_manifest.json').read_bytes();assert hashlib.sha256(data).hexdigest()==parent['manifest_sha256']
 files=json.loads(data)['files'];assert len(files)==parent['files']
 for name,digest in files.items():
  value=p/name;assert value.is_file() and not value.is_symlink()
  with value.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==digest
root.mkdir();source.mkdir(parents=True)
for name,value in contents.items():
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as out:out.write(value)
 target.chmod(0o444)
with (source.parent/'source.tar.gz').open('xb') as out:out.write(raw)
(source.parent/'source.tar.gz').chmod(0o444)
for directory in sorted((p for p in source.rglob('*') if p.is_dir()),reverse=True):directory.chmod(0o555)
source.chmod(0o555)
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(source/'scripts'))
import continuation_ops as ops
assert ops.SOURCE==source and ops.ROOT==root
pin=PREPARATION['manifest_sha256'];ops.verify(source,pin)
plan=ops.validate_plan(ops.read(source/'plan.json'))
adoption=ops.external(manifest,plan)
ops.write(root/'failed-original-adoption.json',adoption)
preflight=root/'preflight';preflight.mkdir()
gate=ops.storage(preflight,'preflight')
cpu=ops.submit('cpu',pin)
callback=ops.submit('after-cpu',pin,dependency='ended('+cpu['job_id']+')')
receipt={'status':'cpu_and_callback_submitted_not_passed','manifest_sha256':pin,
 'plan_sha256':ops.sha(source/'plan.json'),'files':len(manifest['files']),
 'archive_bytes':len(raw),'archive_sha256':hashlib.sha256(raw).hexdigest(),
 'cpu':cpu,'after_cpu':callback,'storage':gate,'original_results_changed':False,
 'captured_utc':datetime.now(timezone.utc).isoformat()}
ops.write(root/'bootstrap-complete.json',receipt)
print(json.dumps(receipt))
'''
    out=LOCAL/'stage-01';out.mkdir(exist_ok=False)
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    record={'status':'unknown','argv':argv,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
            'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        with (out/'stdout.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            result=subprocess.run(argv,input=program.encode(),stdout=stdout,stderr=stderr,timeout=360)
        record.update(status='returned',returncode=result.returncode)
    finally:
        record['ended_utc']=datetime.now(timezone.utc).isoformat()
        record['stdout_sha256']=hashlib.sha256((out/'stdout.json').read_bytes()).hexdigest()
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8',newline='\n')
    assert result.returncode==0,(out/'stderr.txt').read_text()
    actual=json.loads((out/'stdout.json').read_bytes())
    (RUN/'submission.json').write_text(json.dumps(actual,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps({key:actual[key] for key in ('status','manifest_sha256','plan_sha256','captured_utc')},indent=2))
    print('Actual CPU',actual['cpu']['job_id'],'callback',actual['after_cpu']['job_id'])


if __name__=='__main__':main()
