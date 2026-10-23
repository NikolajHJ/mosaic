"""One-shot authenticated source staging/CPU submission; unknown outcomes block."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import ast,hashlib,json,subprocess,sys,tarfile
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'boltz-lr0025-02';release=base/'boltz-lr0025-release-02';source=release/'source';upload=base/'boltz-lr0025-02.tar.gz'
expected,size,manifest,count=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
assert not any(path.exists() for path in [root,release,upload]),'Inspect prior staging/submission; never duplicate'
with upload.open('xb') as stream:
 for data in iter(lambda:sys.stdin.buffer.read(1048576),b''):stream.write(data)
assert upload.stat().st_size==size
with upload.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==expected
release.mkdir();source.mkdir();seen=set()
with tarfile.open(upload) as archive:
 for member in archive:
  assert member.isfile() and member.name.startswith('source/')
  name=member.name.removeprefix('source/');path=source/name
  assert path.resolve().is_relative_to(source) and name not in seen
  seen.add(name);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(archive.extractfile(member).read())
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
meta=json.loads(raw);assert len(meta['files'])==count and seen==set(meta['files'])|{'source_manifest.json'}
for name,digest in meta['files'].items():
 data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 if name.endswith('.py'):ast.parse(data,filename=name)
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
subprocess.run([python,str(source/'scripts/verify_stage0_release.py'),str(source),manifest],check=True)
subprocess.run([python,str(source/'scripts/verify_stage0_release.py'),str(base/'restart-lrreset-release-04/source'),'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'],check=True)
for wrapper in ['boltz_serial_cpu.lsf','boltz_serial_stage.lsf']:subprocess.run(['bash','-n',str(source/'scripts/hpc'/wrapper)],check=True)
for path in source.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
source.chmod(0o555);root.mkdir();sys.dont_write_bytecode=True;sys.path[:0]=[str(source/'scripts'),str(source/'src')]
from campaign25_contract import config25
import boltz_serial_ops as ops
for arm in ops.ARMS:config25(json.loads(ops.config(arm).read_bytes()))
ops.exclusive(root/'source-staging.json',{'status':'passed','files':count,'manifest':manifest,'archive_sha256':expected,'archive_bytes':size,'readonly':True,'original_saved_results_unchanged':True})
gate=ops.storage_gate(root,'preflight')
ops.exclusive(root/'jobs.json',{})
cpu=ops.submit('cpu','cpu',manifest);callback=ops.submit('after-cpu','after-cpu',manifest,'ended('+cpu['job_id']+')')
print(json.dumps({'status':'submitted_not_passed','source':str(source),'root':str(root),'manifest':manifest,'cpu':cpu,'callback':callback,'preflight':gate}))
'''


def main():
    folder=REPO/'results/hpc'/HERE.name
    package=json.loads((HERE/'package.json').read_bytes())
    archive=folder/'source01/source.tar.gz'
    assert archive.stat().st_size==package['archive_bytes'] and hashlib.sha256(archive.read_bytes()).hexdigest()==package['archive_sha256']
    marker=folder/'submission-started.json'
    assert not marker.exists(),'Inspect exact current outcome; never replay'
    command=' '.join(shlex.quote(value) for value in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE,package['archive_sha256'],str(package['archive_bytes']),package['manifest_sha256'],str(package['source_files'])])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    marker.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args},indent=2)+'\n')
    try:
        result=subprocess.run(args,input=archive.read_bytes(),capture_output=True,timeout=240)
    except subprocess.TimeoutExpired as error:
        (folder/'submission-timeout.json').write_text(json.dumps({'outcome':'unknown','error':repr(error)})+'\n')
        raise
    (folder/'submission-stdout.bin').write_bytes(result.stdout)
    (folder/'submission-stderr.bin').write_bytes(result.stderr)
    (folder/'submission-result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print(result.stdout.decode()[-12000:]);print(result.stderr.decode()[-2000:])
    assert result.returncode==0,'Preserved failed/unknown outcome; no blind retry'


if __name__=='__main__':
    main()
