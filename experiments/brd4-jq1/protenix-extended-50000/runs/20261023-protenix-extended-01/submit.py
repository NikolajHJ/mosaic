"""One-shot immutable staging and allocated CPU submission; never replay markers."""
from datetime import datetime,timezone
import hashlib,json,shlex,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import ast,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
r=b/'protenix-extended-01';release=b/'protenix-extended-release-01';s=release/'source';upload=b/'protenix-extended-01.tar.gz'
expected,size,manifest,count=sys.argv[1],int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
assert r.is_dir() and {p.name for p in r.iterdir()} == {'predecessor-retirement'} and not release.exists() and not upload.exists(), 'Inspect unknown staging/submission outcome'
with upload.open('xb') as f:
 for d in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(d)
assert upload.stat().st_size==size
with upload.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected
release.mkdir();s.mkdir();seen=set()
with tarfile.open(upload) as tar:
 for m in tar:
  assert m.isfile() and m.name.startswith('source/')
  n=m.name.removeprefix('source/');p=s/n
  assert p.resolve().is_relative_to(s) and n not in seen;seen.add(n);p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(tar.extractfile(m).read())
raw=(s/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
meta=json.loads(raw);assert len(meta['files'])==count and seen==set(meta['files'])|{'source_manifest.json'}
for n,h in meta['files'].items():
 d=(s/n).read_bytes();assert hashlib.sha256(d).hexdigest()==h
 if n.endswith('.py'):ast.parse(d,filename=n)
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
old=b/'restart-lrreset-release-04/source'
subprocess.run([python,str(s/'scripts/verify_stage0_release.py'),str(s),manifest],check=True)
subprocess.run([python,str(s/'scripts/verify_stage0_release.py'),str(old),'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'],check=True)
ready=b/'restart-lrreset-04/validate-protenix/payload/study/assessment.json'
assert hashlib.sha256(ready.read_bytes()).hexdigest()=='9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
for wrapper in ['extended_cpu.lsf','extended_stage.lsf']:subprocess.run(['bash','-n',str(s/'scripts/hpc'/wrapper)],check=True)
for p in s.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
s.chmod(0o555);sys.dont_write_bytecode=True;sys.path[:0]=[str(s/'scripts'),str(s/'src')]
from campaign25_contract import config25
import extended_ops as ops

for arm in ops.ARMS:config25(json.loads(ops.config(arm).read_bytes()))
assert json.loads((s/'docs/extended_retirement.json').read_bytes()) == json.loads((r/'predecessor-retirement/assessment.json').read_bytes())
ops.exclusive(r/'source-staging.json',{'status':'passed','files':count,'manifest':manifest,'archive_sha256':expected,'archive_bytes':size,'readonly':True,'original_scientific_results_unchanged':True})
gate=ops.storage_gate(r,'preflight')
ops.exclusive(r/'jobs.json',{})
cpu=ops.submit('cpu','cpu',manifest)
callback=ops.submit('after-cpu','after-cpu',manifest,'ended('+cpu['job_id']+')')
print(json.dumps({'status':'submitted_not_passed','source':str(s),'root':str(r),'manifest':manifest,'cpu':cpu,'callback':callback,'preflight':gate}))
'''
def main():
    folder=REPO/'results/hpc'/HERE.name;package=json.loads((HERE/'package.json').read_bytes());arc=folder/'source02/source.tar.gz'
    assert arc.stat().st_size==package['archive_bytes'] and hashlib.sha256(arc.read_bytes()).hexdigest()==package['archive_sha256']
    marker=folder/'submission-started.json';assert not marker.exists(),'Inspect exact current outcome; never replay'
    command=' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE,package['archive_sha256'],str(package['archive_bytes']),package['manifest_sha256'],str(package['source_files'])])
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(command)]
    marker.write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'args':args},indent=2)+'\n')
    try:q=subprocess.run(args,input=arc.read_bytes(),capture_output=True,timeout=240)
    except subprocess.TimeoutExpired as e:(folder/'submission-timeout.json').write_text(json.dumps({'outcome':'unknown','error':repr(e)})+'\n');raise
    (folder/'submission-stdout.bin').write_bytes(q.stdout);(folder/'submission-stderr.bin').write_bytes(q.stderr);(folder/'submission-result.json').write_text(json.dumps({'returncode':q.returncode})+'\n')
    print(q.stdout.decode()[-12000:]);print(q.stderr.decode()[-1500:]);assert q.returncode==0,'Preserved failed/unknown outcome; no blind retry'
if __name__=='__main__':main()
