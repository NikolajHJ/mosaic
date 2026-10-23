"""One-shot manifest-only source staging. No test/model/job submission."""
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
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
release=b/'boltz-dbmol-pilot-release-02';source=release/'source';upload=b/'boltz-dbmol-pilot-02.tar.gz'
old=b/'boltz-dbmol-pilot-release-01/source';ancestor=b/'restart-lrreset-release-04/source'
expected,size,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
assert not any(os.path.lexists(p) for p in [release,upload,b/'boltz-dbmol-pilot-02']),'Inspect unknown staging/launch outcome; never repeat'
assert sha(old/'source_manifest.json')=='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
assert sha(ancestor/'source_manifest.json')=='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
previous=json.loads((old/'source_manifest.json').read_bytes());trusted=json.loads((ancestor/'source_manifest.json').read_bytes())
assert trusted['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743'
for name,digest in previous['files'].items():assert sha(old/name)==digest
with upload.open('xb') as stream:
 for data in iter(lambda:sys.stdin.buffer.read(1048576),b''):stream.write(data)
assert upload.stat().st_size==size and sha(upload)==expected
release.mkdir();source.mkdir();seen=set()
with tarfile.open(upload) as archive:
 for item in archive:
  assert item.isfile() and item.name.startswith('source/')
  name=item.name.removeprefix('source/');target=source/name
  assert name not in seen and '..' not in Path(name).parts and target.resolve().is_relative_to(source.resolve())
  seen.add(name);target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as stream:stream.write(archive.extractfile(item).read())
assert sha(source/'source_manifest.json')==manifest
value=json.loads((source/'source_manifest.json').read_bytes())
assert value['files']==previous['files'] and len(value['files'])==657 and seen==set(value['files'])|{'source_manifest.json'}
assert value['git_revision_reference']==trusted['git_revision_reference'] and value['parent_manifest_sha256']==sha(old/'source_manifest.json')
for name,digest in value['files'].items():
 assert sha(source/name)==digest
 if name.endswith('.py'):ast.parse((source/name).read_bytes(),filename=name)
for path in source.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
source.chmod(0o555)
assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) and not p.stat().st_mode&0o222 for p in [source,*source.rglob('*')])
print(json.dumps({'status':'staged_readonly_not_tested','source':str(source),'manifest_sha256':manifest,'archive_bytes':size,'archive_sha256':expected,'files':657,'all657_member_bytes_unchanged':True,'git_revision_reference':value['git_revision_reference'],'jobs_submitted':False,'saved_results_changed':False}))
'''


def main():
    package = json.loads((HERE.parent / '20261024-boltz-dbmol-storage-05/science-package.json').read_bytes())
    folder = REPO / 'results/hpc' / HERE.name
    archive = folder / 'source01/source.tar.gz'
    assert archive.stat().st_size == package['archive_bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256']
    marker = folder / 'staging-started.json'
    assert not marker.exists(), 'Inspect exact remote/transfer outcome; never blindly replay'
    command = ' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE,
                                              package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256']])
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    marker.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': args}, indent=2) + '\n', encoding='utf8')
    try:
        result = subprocess.run(args, input=archive.read_bytes(), capture_output=True, timeout=240)
    except subprocess.TimeoutExpired as error:
        (folder / 'staging-timeout.json').write_text(json.dumps({'outcome': 'unknown', 'error': repr(error)}) + '\n')
        raise
    (folder / 'staging-stdout.bin').write_bytes(result.stdout)
    (folder / 'staging-stderr.bin').write_bytes(result.stderr)
    (folder / 'staging-result.json').write_text(json.dumps({'returncode': result.returncode}) + '\n')
    print(result.stdout.decode(errors='replace')[-2000:])
    print(result.stderr.decode(errors='replace')[-1000:])
    assert result.returncode == 0, 'Preserved failed/unknown staging outcome; inspect before recovery'


if __name__ == '__main__':
    main()
