"""Submit only corrected diagnostic CPU gate and its Protenix-only callback."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[5]
TAG = '20261022-brd4-jq1-lrboost-analysis-02'
REMOTE = r'''
import ast,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
tag='20261022-brd4-jq1-lrboost-analysis-02'
root=Path('/work3/s234818/mosaic/runs')/tag;release=Path('/work3/s234818/mosaic/releases')/tag
source=release/'source';upload=Path('/work3/s234818/mosaic')/(tag+'.tar.gz')
expected,expected_bytes,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert not any(p.exists() for p in (root,release,upload)), 'Existing/unknown outcome prohibits bootstrap replay'
with upload.open('xb') as out:
 for block in iter(lambda:sys.stdin.buffer.read(1048576),b''):out.write(block)
assert upload.stat().st_size==expected_bytes and hashlib.sha256(upload.read_bytes()).hexdigest()==expected
release.mkdir();source.mkdir()
with tarfile.open(upload) as tar:
 seen=set()
 for m in tar:
  name=m.name.removeprefix('source/');rel=Path(name)
  assert m.isfile() and m.name.startswith('source/') and not rel.is_absolute() and '..' not in rel.parts and name not in seen
  seen.add(name);p=source/rel;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as out:out.write(tar.extractfile(m).read())
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
metadata=json.loads(raw);assert len(metadata['files'])==16 and set(metadata['files'])|{'source_manifest.json'}==seen
for name,expected_sha in metadata['files'].items():
 data=(source/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected_sha
 if name.endswith('.py'):ast.parse(data,filename=name)
science=Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source')
assert hashlib.sha256((science/'source_manifest.json').read_bytes()).hexdigest()=='9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
old=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/diagnostic-protenix/payload/assessment.json')
assert hashlib.sha256(old.read_bytes()).hexdigest()=='25c3cfd7c03776b1de86ba98e0c911ff6c3028376fc06e3adc64ee1d2f74dbbd'
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
subprocess.run([python,str(source/'scripts/verify_stage0_release.py'),str(source),manifest],check=True)
subprocess.run(['bash','-n',str(source/'scripts/hpc/jq1_lrboost_analysis_stage.lsf')],check=True)
for p in source.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
source.chmod(0o555);root.mkdir()
def write(path,obj):
 with path.open('x') as out:json.dump(obj,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
write(root/'source-staging.json',{'status':'passed','files':16,'manifest_sha256':manifest,'archive_sha256':expected,'archive_bytes':expected_bytes,'scope':'Separate corrected Protenix-only diagnostic; all saved original studies unchanged'})
sys.path[:0]=[str(source/'scripts'),str(science/'scripts'),str(science/'src')]
import jq1_lrboost_report_ops as ops
parent=root/'preflight';parent.mkdir();gate=ops.check_storage(parent,'preflight')
cpu=ops.submit('cpu','cpu',manifest)
write(root/'jobs.json',{'cpu':cpu['job_id']})
callback=ops.submit('after-cpu','certify-protenix-only',manifest,dependency='ended('+cpu['job_id']+')')
ops.atomic(root/'jobs.json',{'cpu':cpu['job_id'],'cpu_callback':callback['job_id']})
write(root/'initial-submissions.json',{'cpu':cpu,'after_cpu':callback})
print(json.dumps({'status':'submitted_not_passed','root':str(root),'manifest':manifest,'cpu':cpu['job_id'],'after_cpu':callback['job_id'],'preflight_required':gate['required_bytes'],'preflight_available':gate['available_bytes_conservative']}))
'''


def main():
    folder = REPO / 'results/hpc' / TAG
    package = json.loads((folder / 'source02/package.json').read_text())
    archive = folder / 'source02/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256']
    marker = folder / 'submission-started.json'
    assert not marker.exists(), 'Inspect exact prior outcome before recovery'
    command = ' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE,
        package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256']])
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    marker.write_bytes(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': args}, indent=2).encode())
    result = subprocess.run(args, input=archive.read_bytes(), capture_output=True, timeout=180)
    (folder / 'submission-stdout.txt').write_bytes(result.stdout)
    (folder / 'submission-stderr.txt').write_bytes(result.stderr)
    (folder / 'submission-result.json').write_bytes(json.dumps({'returncode': result.returncode,
        'stdout_sha256': hashlib.sha256(result.stdout).hexdigest()}, indent=2).encode())
    print(result.stdout.decode()[-1600:]); print(result.stderr.decode()[-700:])
    assert result.returncode == 0, 'Failed/unknown staging preserved; do not replay'


if __name__ == '__main__':
    main()
