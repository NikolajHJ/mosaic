"""One-shot separate analysis source staging and allocated CPU/callback submission."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
TAG='20261023-brd4-jq1-lrreset-analysis-02'
REMOTE=r'''
import ast,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
r=b/'lrreset-analysis-02';release=b/'lrreset-analysis-release-02';s=release/'source';upload=b/'lrreset-analysis-02.tar.gz'
expected,size,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert not any(p.exists() for p in (r,release,upload)), 'Existing or unknown analysis outcome; no replay'
with upload.open('xb') as f:
 for data in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(data)
assert upload.stat().st_size==size and hashlib.sha256(upload.read_bytes()).hexdigest()==expected
release.mkdir();s.mkdir();seen=set()
with tarfile.open(upload) as tar:
 for m in tar:
  name=m.name.removeprefix('source/');p=Path(name)
  assert m.isfile() and m.name.startswith('source/') and not p.is_absolute() and '..' not in p.parts and name not in seen
  seen.add(name);dest=s/name;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb') as f:f.write(tar.extractfile(m).read())
raw=(s/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
metadata=json.loads(raw);assert len(metadata['files'])==17 and seen==set(metadata['files'])|{'source_manifest.json'}
for name,expected in metadata['files'].items():
 data=(s/name).read_bytes();assert hashlib.sha256(data).hexdigest()==expected
 if name.endswith('.py'):ast.parse(data,filename=name)
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
science=b/'restart-lrreset-release-04/source'
subprocess.run([python,str(s/'scripts/verify_stage0_release.py'),str(s),manifest],check=True)
subprocess.run([python,str(s/'scripts/verify_stage0_release.py'),str(science),'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'],check=True)
subprocess.run(['bash','-n',str(s/'scripts/hpc/jq1_lrboost_analysis_stage.lsf')],check=True)
for p in s.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
s.chmod(0o555);r.mkdir()
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
write(r/'source-staging.json',{'status':'passed','files':17,'manifest':manifest,'archive_sha256':sys.argv[1],'archive_bytes':size,'readonly':True,'original_scientific_results_unchanged':True})
sys.dont_write_bytecode=True;sys.path[:0]=[str(s/'scripts'),str(science/'scripts'),str(science/'src')]
import jq1_lrboost_report_contract as contract
for arm in contract.ARMS:contract.authenticate_inputs(arm)
import jq1_lrboost_report_ops as ops
gate=ops.check_storage(r,'preflight')
cpu=ops.submit('cpu','cpu',manifest)
ops.exclusive(r/'jobs.json',{'cpu':cpu['job_id']})
callback=ops.submit('after-cpu','certify-cpu',manifest,dependency='ended('+cpu['job_id']+')')
ops.atomic(r/'jobs.json',{'cpu':cpu['job_id'],'after_cpu':callback['job_id']})
print(json.dumps({'status':'submitted_not_passed','manifest':manifest,'source':str(s),'root':str(r),'cpu':cpu,'callback':callback,'preflight_gate':gate}))
'''


def main():
    folder=REPO/'results/hpc'/TAG
    package=json.loads((folder/'source01/package.json').read_bytes());archive=folder/'source01/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==package['archive_sha256']
    marker=folder/'submission-started.json';assert not marker.exists(),'Read exact existing outcome; never replay'
    command=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE,package['archive_sha256'],str(package['archive_bytes']),package['manifest_sha256']])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    marker.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args},indent=2)+'\n')
    try:q=subprocess.run(args,input=archive.read_bytes(),capture_output=True,timeout=180)
    except subprocess.TimeoutExpired as e:
        (folder/'submission-timeout.json').write_text(json.dumps({'outcome':'unknown','error':repr(e)})+'\n');raise
    (folder/'submission-stdout.bin').write_bytes(q.stdout);(folder/'submission-stderr.bin').write_bytes(q.stderr)
    (folder/'submission-result.json').write_text(json.dumps({'returncode':q.returncode})+'\n')
    print(q.stdout.decode()[-11000:]);print(q.stderr.decode()[-1500:])
    assert q.returncode==0,'Failed/unknown outcome preserved; do not replay'


if __name__=='__main__':main()
