"""One-shot metadata-only staging and allocated CPU submission, never replay."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import ast,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'boltz-dbmol-pilot-02'
history=base/'boltz-dbmol-pilot-01'
science=base/'boltz-dbmol-pilot-release-02/source'
release=base/'boltz-dbmol-storage-release-05';source=release/'source'
upload=base/'boltz-dbmol-storage-05.tar.gz'
control=root/'storage-adoption-05'
archive_sha,size,launcher_sha=sys.argv[1],int(sys.argv[2]),sys.argv[3]
science_sha='600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'
def sha(path):
 with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def verify(path,expected):
 assert sha(path/'source_manifest.json')==expected
 files=json.loads((path/'source_manifest.json').read_bytes())['files']
 assert {p.relative_to(path).as_posix() for p in path.rglob('*') if p.is_file() or p.is_symlink()}==set(files)|{'source_manifest.json'}
 for p in [path,*path.rglob('*')]:assert not p.is_symlink() and(p.is_file() or p.is_dir()) and not(p.stat().st_mode&0o222)
 for name,digest in files.items():
  p=path/name;assert p.resolve().is_relative_to(path.resolve()) and sha(p)==digest
 return files
assert not any(os.path.lexists(p) for p in [release,upload,root]),'Inspect prior target/exclusive outcome; never duplicate'
assert not list(root.glob('pilot-*')) and not list(root.glob('dispatch-*')),'Inspect prior native/unknown outcome, never duplicate'
assert len(verify(science,science_sha))==657
jobs=json.loads((history/'jobs.json').read_bytes())
assert jobs=={'cpu':'29798549','after-cpu':'29798550','cpu-02':'29798699','after-cpu-02':'29798700','cpu-03':'29798769','after-cpu-03':'29798770','cpu-04':'29802621','after-cpu-04':'29802622','pilot-native-0000':'29802648','pilot-after-native-0000':'29802649'},'Unexpected historical production or unknown outcome'
assert not list(history.glob('pilot-segment-*')) and not (history/'projection.json').exists()
science_meta=json.loads((science/'source_manifest.json').read_bytes());assert science_meta['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743' and science_meta['parent_manifest_sha256']=='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
with upload.open('xb') as stream:
 for data in iter(lambda:sys.stdin.buffer.read(1048576),b''):stream.write(data)
assert upload.stat().st_size==size and sha(upload)==archive_sha
release.mkdir();source.mkdir();seen=set()
with tarfile.open(upload) as archive:
 for member in archive:
  assert member.isfile() and member.name.startswith('source/')
  name=member.name.removeprefix('source/');path=source/name
  assert name not in seen and '..' not in Path(name).parts and path.resolve().is_relative_to(source.resolve())
  seen.add(name);path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as stream:stream.write(archive.extractfile(member).read())
assert sha(source/'source_manifest.json')==launcher_sha
manifest=json.loads((source/'source_manifest.json').read_bytes())
assert manifest['scientific_source_manifest_sha256']==science_sha and seen==set(manifest['files'])|{'source_manifest.json'}
for name,digest in manifest['files'].items():
 assert sha(source/name)==digest
 if name.endswith('.py'):ast.parse((source/name).read_bytes(),filename=name)
for wrapper in ['dbmol_storage_cpu.lsf','dbmol_storage_stage.lsf']:
 subprocess.run(['bash','-n',str(source/'scripts/hpc'/wrapper)],check=True)
for path in source.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
source.chmod(0o555)
verify(source,launcher_sha)
root.mkdir()
sys.dont_write_bytecode=True;sys.path[:0]=[str(source/'scripts'),str(science/'scripts')]
import dbmol_storage_ops as ops
ops.configure_launcher(launcher_sha);ops.verify_launcher(science_sha,launcher_sha)
old_proofs=ops.verify_failed_old_cpu()
control.mkdir();ops.exclusive(control/'started.json',{'status':'staged_not_passed','launcher_manifest_sha256':launcher_sha,'science_manifest_sha256':science_sha,'archive_bytes':size,'archive_sha256':archive_sha,'preserved_old_failed_cpu_proofs':old_proofs})
gate=ops.storage_gate(control,'preflight')
ops.exclusive(root/'jobs.json',{})
ops.exclusive(root/'source-staging.json',{'status':'passed','science_manifest_sha256':science_sha,'launcher_manifest_sha256':launcher_sha,'old_saved_results_unchanged':True,'readonly':True})
cpu=ops.submit('cpu-05','cpu',science_sha)
callback=ops.submit('after-cpu-05','after-cpu',science_sha,'ended('+cpu['job_id']+')')
record={'status':'submitted_not_passed','launcher_manifest_sha256':launcher_sha,'science_manifest_sha256':science_sha,'launcher_source':str(source),'root':str(root),'cpu':cpu,'callback':callback,'preflight':gate,'old_failed_cpu_proofs':old_proofs,'scientific_source_changed':False}
ops.exclusive(control/'submission-complete.json',record)
print(json.dumps(record))
'''


def main():
    package = json.loads((HERE / 'package.json').read_bytes())
    folder = REPO / 'results/hpc' / HERE.name
    archive = folder / 'source01/source.tar.gz'
    assert archive.stat().st_size == package['archive_bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256']
    marker = folder / 'submission-started.json'
    assert not marker.exists(), 'Inspect original outcome; never repeat staging or submission'
    command = ' '.join(shlex.quote(value) for value in [
        '/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE,
        package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256'],
    ])
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ConnectionAttempts=1', 'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    marker.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': argv}, indent=2) + '\n')
    try:
        result = subprocess.run(argv, input=archive.read_bytes(), capture_output=True, timeout=240)
    except subprocess.TimeoutExpired as error:
        (folder / 'submission-timeout.json').write_text(json.dumps({'outcome': 'unknown', 'error': repr(error)}) + '\n')
        raise
    (folder / 'submission-stdout.bin').write_bytes(result.stdout)
    (folder / 'submission-stderr.bin').write_bytes(result.stderr)
    (folder / 'submission-result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')

    if result.returncode==0:
        value=json.loads(result.stdout.decode()[result.stdout.decode().index('{'):]);print(json.dumps({'status':value['status'],'cpu':value['cpu']['job_id'],'callback':value['callback']['job_id'],'preflight':{k:value['preflight']['work3'][k] for k in ['required_bytes','available_bytes_conservative']}}))
    else:print(result.stdout.decode(errors='replace')[-3000:])
    print(result.stderr.decode()[-3000:])
    assert result.returncode == 0, 'Preserved failed/unknown outcome; no blind retry'


if __name__ == '__main__':
    main()
