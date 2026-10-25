"""One-shot immutable report staging/CPU launch, with unknown outcomes preserved."""
from datetime import datetime,timezone
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'
PREFIX=r'''
import base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
sys.dont_write_bytecode=True
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=base/'nonchem-live-report-release-03/source';root=base/'nonchem-live-report-03'
assert not root.exists() and not source.parent.exists(), 'Existing/unknown source or report root: never replay'
archive=base/'nonchem-live-report-release-03.tar.gz'
assert not archive.exists(), 'Existing upload outcome: inspect instead of retry'
raw=base64.b64decode(ARCHIVE)
assert len(raw)==LENGTH and hashlib.sha256(raw).hexdigest()==DIGEST
with archive.open('xb') as stream:stream.write(raw)
source.mkdir(parents=True);root.mkdir()
with tarfile.open(fileobj=io.BytesIO(raw)) as bundle:
 items=bundle.getmembers();assert len({x.name for x in items})==len(items)
 for item in items:
  name=PurePosixPath(item.name)
  assert item.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in item.name
  path=source.joinpath(*name.parts);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(bundle.extractfile(item).read())
  path.chmod(0o444)
assert hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest()==PIN
sys.path.insert(0,str(source/'scripts'))
import report_ops as ops
manifest=ops.verify(source,PIN);ops.external(manifest)
subprocess.run(['bash','-n',str(source/'scripts/report_stage.lsf')],check=True)
for folder in sorted([source,*[p for p in source.rglob('*') if p.is_dir()]],key=lambda p:len(p.parts),reverse=True):folder.chmod(0o555)
preflight=root/'preflight';preflight.mkdir();ops.storage(preflight,'preflight')
cpu=ops.submit('cpu',PIN)
callback=ops.submit('after-cpu',PIN,'ended('+cpu['job_id']+')')
result={'status':'submitted_not_passed','cpu':cpu,'after_cpu':callback,'manifest_sha256':PIN}
ops.write(root/'initial-submissions.json',result)
print(json.dumps(result))
'''


def main():
    package=LOCAL/'source03';preparation=json.loads((package/'preparation.json').read_bytes())
    raw=(package/'source.tar.gz').read_bytes()
    assert len(raw)==preparation['archive_bytes'] and hashlib.sha256(raw).hexdigest()==preparation['archive_sha256']
    program='\n'.join(['ARCHIVE='+repr(base64.b64encode(raw).decode()),'LENGTH='+str(len(raw)),
                       'DIGEST='+repr(preparation['archive_sha256']),'PIN='+repr(preparation['manifest_sha256']),PREFIX])
    out=LOCAL/'launch-03';out.mkdir(exist_ok=False)
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown','program_sha256':hashlib.sha256(program.encode()).hexdigest(),'argv':command}
    try:
        with (out/'stdout.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            run=subprocess.run(command,input=program.encode(),stdout=stdout,stderr=stderr,timeout=240)
        receipt.update(status='returned',returncode=run.returncode)
        assert run.returncode==0,(out/'stderr.txt').read_text()
    finally:
        receipt.update(captured_utc=datetime.now(timezone.utc).isoformat())
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=json.loads((out/'stdout.json').read_bytes())
    print(json.dumps({'status':result['status'],'cpu':result['cpu']['job_id'],'after_cpu':result['after_cpu']['job_id']},indent=2))


if __name__=='__main__':main()
