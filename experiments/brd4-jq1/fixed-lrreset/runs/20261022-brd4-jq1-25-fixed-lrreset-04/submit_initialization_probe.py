"""One-time immutable probe packaging/transfer and allocated job submission."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
TAG = '20261022-brd4-jq1-25-fixed-lrreset-04'
REMOTE = r'''
import ast,hashlib,json,os,re,subprocess,sys,tarfile
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'restart-lrreset-04';source=base/'restart-lrreset-initialization-probe-04';upload=base/'restart-lrreset-initialization-probe-04.tar.gz'
expected,n,manifest=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert not any(p.exists() for p in (root,source,upload)),'Existing or unknown outcome; inspect, no replay'
with upload.open('xb') as f:
    for b in iter(lambda:sys.stdin.buffer.read(1048576),b''):f.write(b)
assert upload.stat().st_size==n and hashlib.sha256(upload.read_bytes()).hexdigest()==expected
source.mkdir();seen=set()
with tarfile.open(upload) as t:
    for m in t:
        p=Path(m.name)
        assert m.isfile() and not p.is_absolute() and '..' not in p.parts and len(p.parts)==1 and m.name not in seen
        seen.add(m.name);(source/m.name).write_bytes(t.extractfile(m).read())
raw=(source/'source_manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest
meta=json.loads(raw);assert set(meta['files'])|{'source_manifest.json'}==seen
for name,h in meta['files'].items():
    b=(source/name).read_bytes();assert hashlib.sha256(b).hexdigest()==h
    if name.endswith('.py'):ast.parse(b)
subprocess.run(['bash','-n',str(source/'initialization_probe.lsf')],check=True)
for p in source.iterdir():p.chmod(0o444)
source.chmod(0o555);root.mkdir()
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
write(root/'initialization-probe-source.json',{'files':len(meta['files']),'manifest_sha256':manifest,'archive_sha256':expected,'archive_bytes':n,'source':str(source)})
jobs={}
for phase in ['cpu','gpu']:
    parent=root/('initialization-probe-'+phase);parent.mkdir()
    command=['bsub','-q','hpc' if phase=='cpu' else 'gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]' if phase=='cpu' else 'rusage[mem=8GB]','-W','00:10' if phase=='cpu' else '00:20','-J','mosaic-init04-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if phase=='gpu':command+=['-gpu','num=1:mode=exclusive_process','-w','done('+jobs['cpu']['job_id']+')']
    command+=['bash','-l',str(source/'initialization_probe.lsf'),str(source),manifest,phase,str(root)]
    write(parent/'submission-started.json',{'outcome':'unknown','args':command})
    rec={'outcome':'unknown','args':command,'phase':phase,'manifest_sha256':manifest}
    try:
        r=subprocess.run(command,text=True,capture_output=True,timeout=60);rec.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        ids=re.findall(r'Job <(\d+)>',r.stdout)
        if r.returncode==0 and len(ids)==1:rec.update(outcome='submitted',job_id=ids[0])
    finally:write(parent/'submission.json',rec)
    assert rec['outcome']=='submitted','Unknown outcome; no replay'
    jobs[phase]=rec
write(root/'initialization-probe-jobs.json',jobs)
print(json.dumps({'status':'submitted_not_passed','jobs':jobs,'root':str(root)}))
'''


def main():
    output = REPO / 'results/hpc' / TAG / 'initialization-probe'
    output.mkdir(parents=True, exist_ok=False)
    files = {n: (HERE / n).read_bytes() for n in ['initialization_probe.py', 'initialization_probe.lsf', 'README.md']}
    ast.parse(files['initialization_probe.py'])
    raw = json.dumps({'version': 'exact-initialization-replay-probe-v1', 'files': {n: hashlib.sha256(b).hexdigest() for n, b in files.items()}}, indent=2).encode() + b'\n'
    files['source_manifest.json'] = raw
    archive = output / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as t:
        for name, b in sorted(files.items()):
            m = tarfile.TarInfo(name); m.size = len(b); m.mode = 0o444; t.addfile(m, io.BytesIO(b))
    package = {'utc': datetime.now(timezone.utc).isoformat(), 'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'archive_bytes': archive.stat().st_size}
    (HERE / 'probe-package.json').write_bytes(json.dumps(package, indent=2).encode() + b'\n')
    command = shlex.join(['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE, package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256']])
    ssh = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (output / 'submission-started.json').write_bytes(json.dumps({'args': ssh, 'outcome': 'unknown'}).encode())
    r = subprocess.run(ssh, input=archive.read_bytes(), capture_output=True, timeout=180)
    (output / 'stdout.txt').write_bytes(r.stdout); (output / 'stderr.txt').write_bytes(r.stderr)
    (output / 'result.json').write_bytes(json.dumps({'returncode': r.returncode, 'stdout_sha256': hashlib.sha256(r.stdout).hexdigest()}).encode())
    print(r.stdout.decode(errors='replace')[-5000:]); print(r.stderr.decode(errors='replace')[-500:])
    assert r.returncode == 0, 'Preserved failed/unknown outcome; inspect before recovery'


if __name__ == '__main__':
    main()
