"""One-shot exclusive staging/submission of the small allocated operator checks."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
source=root/'operator-source-01';parent=root/'cpu-01'
expected,archive_sha,archive_bytes=sys.argv[1],sys.argv[2],int(sys.argv[3])
assert not os.path.lexists(root),'Existing control root/unknown staging; inspect, do not replay'
assert root.parent.is_dir() and root.parent.resolve()==root.parent
root.mkdir()
def save(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
save(root/'operator-staging-started.json',{'outcome':'unknown','source_manifest_sha256':expected,'archive_sha256':archive_sha})
upload=root/'operator-source-01.tar.gz'
with upload.open('xb') as f:
 for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(block)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert upload.stat().st_size==archive_bytes and sha(upload)==archive_sha
source.mkdir();seen=set()
with tarfile.open(upload) as tar:
 for member in tar:
  p=PurePosixPath(member.name)
  assert member.isfile() and len(p.parts)==1 and p.as_posix()==member.name and member.name not in seen and ':' not in member.name and '\\' not in member.name
  seen.add(member.name)
  with (source/member.name).open('xb') as f:f.write(tar.extractfile(member).read())
assert sha(source/'source-manifest.json')==expected
m=json.loads((source/'source-manifest.json').read_bytes())
assert len(m['files'])==11 and seen==set(m['files'])|{'source-manifest.json'}
for name,digest in m['files'].items():
 p=source/name;assert sha(p)==digest
 if name.endswith('.py'):ast.parse(p.read_bytes(),filename=name)
for p in source.iterdir():p.chmod(0o444)
source.chmod(0o555)
subprocess.run(['bash','-n',str(source/'cleanup_cpu.lsf')],check=True)
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;sys.path.insert(0,str(source))
import cleanup_cpu
cleanup_cpu.verify_source(expected)
cleanup_cpu.admission(root/'operator-preflight-01','cleanup-only-preflight')
parent.mkdir()
args=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30','-J','mosaic-storage-offload-cpu-01','-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log'),'bash','-l',str(source/'cleanup_cpu.lsf'),expected]
save(parent/'submission-started.json',{'outcome':'unknown','args':args,'source_manifest_sha256':expected})
r=subprocess.run(args,capture_output=True,text=True,timeout=60)
ids=re.findall(r'Job <(\d+)> is submitted',r.stdout)
result={'args':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'source_manifest_sha256':expected,'archive_sha256':archive_sha,'outcome':'submitted' if r.returncode==0 and len(ids)==1 else 'unknown'}
if result['outcome']=='submitted':result['job_id']=ids[0]
save(parent/'submission.json',result)
assert result['outcome']=='submitted','Unknown submission outcome; inspect exact marker'
save(root/'operator-staging-complete.json',{'source_manifest_sha256':expected,'archive_sha256':archive_sha,'archive_bytes':archive_bytes,'source_files':len(m['files']),'cpu_job_id':ids[0]})
print(json.dumps(result))
'''


def main():
    package = json.loads((HERE / 'operator-package-01.json').read_bytes())
    archive = Path(package['archive'])
    assert hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest() == package['archive_sha256']
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-submission-01'
    folder.mkdir(exist_ok=False)
    remote_args = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE,
                   package['manifest_sha256'], package['archive_sha256'], str(package['archive_bytes'])]
    command = ' '.join(shlex.quote(x) for x in remote_args)
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (folder / 'started.json').write_text(json.dumps({'outcome': 'unknown', 'args': args}) + '\n')
    try:
        with archive.open('rb') as data, (folder / 'stdout.bin').open('xb') as stdout, (folder / 'stderr.bin').open('xb') as stderr:
            result = subprocess.run(args, stdin=data, stdout=stdout, stderr=stderr, timeout=300)
    except BaseException as error:
        (folder / 'failed-unknown.json').write_text(json.dumps({'error': repr(error), 'outcome': 'unknown_inspect_remote'}) + '\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    print('returncode', result.returncode)
    print((folder / 'stdout.bin').read_text())
    print((folder / 'stderr.bin').read_text()[-4000:])
    assert result.returncode == 0


if __name__ == '__main__':
    main()
