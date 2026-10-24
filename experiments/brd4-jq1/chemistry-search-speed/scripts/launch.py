"""One-shot stdlib staging and bounded allocatedCPU submission; no job replay."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-01'
REMOTE = r'''
import ast,base64,hashlib,io,json,os,re,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
release=base/'chemistry-search-speed-release-01';source=release/'source';root=base/'chemistry-search-speed-01'
assert not any(os.path.lexists(p) for p in (release,root)), 'Existing/unknown outcome: inspect, never replay'
archive=base64.b64decode(ARCHIVE)
assert len(archive)==LENGTH and hashlib.sha256(archive).hexdigest()==SHA
with tarfile.open(fileobj=io.BytesIO(archive)) as t:
 rows=t.getmembers()
 assert len(rows)==10 and len({r.name for r in rows})==len(rows)
 assert all(r.isfile() and not PurePosixPath(r.name).is_absolute() and '..' not in PurePosixPath(r.name).parts for r in rows)
 data={r.name:t.extractfile(r).read() for r in rows}
assert hashlib.sha256(data['source_manifest.json']).hexdigest()==MANIFEST
m=json.loads(data['source_manifest.json']);assert set(data)==set(m['files'])|{'source_manifest.json'}
for name,pin in m['files'].items():
 assert hashlib.sha256(data[name]).hexdigest()==pin
 if name.endswith('.py'):ast.parse(data[name],filename=name)
root.mkdir()
(root/'source.tar.gz').write_bytes(archive)
source.mkdir(parents=True)
for name,raw in data.items():
 p=source/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('xb') as f:f.write(raw)
 p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);release.chmod(0o555)
sys.path.insert(0,str(source/'scripts'))
import worker
worker.verify(source,MANIFEST)
worker.verify(worker.ORIGINAL,'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')
paths=[('random-early0','protenix-extended-03/random-native-0000/payload/study/early_pilot/snapshots/step-0000.npz'),
       ('random-early1','protenix-extended-03/random-native-0000/payload/study/early_pilot/snapshots/step-0001.npz'),
       ('random-early2','protenix-extended-03/random-native-0000/payload/study/early_pilot/snapshots/step-0002.npz'),
       ('random-high10','protenix-extended-03/random-native-0000/payload/study/high_pilot/snapshots/step-0010.npz'),
       ('boltz-valid0','boltz-lr0025-03/boltz2_chemistry-native-0000/payload/study/early_pilot/snapshots/step-0000.npz'),
       ('boltz-invalid0','boltz-lr0025-03/boltz2_chemistry-native-0000/payload/study/high_pilot/snapshots/step-0000.npz')]
inputs=[]
for label,rel in paths:
 p=base/rel;assert p.is_file() and not p.is_symlink(),str(p)
 inputs.append({'label':label,'path':str(p),'bytes':p.stat().st_size,'sha256':worker.sha(p)})
config=worker.ORIGINAL/'examples/configs/extended_random.json';c=json.loads(config.read_text())
plan={'version':'chemistry-search-speed-v1','inputs':inputs,'search':c['search'],'progress_search':c['progress_search'],
      'config':{'path':str(config),'sha256':worker.sha(config)},'repeats':3,
      'production_or_optimizer_changes':False,'cpu_cases':79,'output_cap_bytes':worker.LIMIT,
      'workers':[2,4],'monitor_remains_paused':True}
(root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
pre=root/'preflight';pre.mkdir();gate=worker.guard(pre,'preflight')
syntax=subprocess.run(['bash','-n',str(source/'scripts/stage.lsf')],capture_output=True,check=True)
parent=root/'cpu';parent.mkdir()
argv=['bsub','-q','hpc','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','2:00',
      '-J','mosaic-chem-speed01','-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log'),
      'bash','-l',str(source/'scripts/stage.lsf'),str(source),MANIFEST]
with (parent/'submission-started.json').open('x') as f:json.dump({'argv':argv,'manifest':MANIFEST},f)
submission=subprocess.run(argv,capture_output=True,text=True,timeout=60)
(parent/'bsub.stdout.txt').write_text(submission.stdout);(parent/'bsub.stderr.txt').write_text(submission.stderr)
assert submission.returncode==0,submission.stderr
ids=re.findall(r'Job <(\d+)> is submitted',submission.stdout);assert len(ids)==1
receipt={'job_id':ids[0],'argv':argv,'status':'submitted_not_passed','manifest_sha256':MANIFEST,
         'source':str(source),'root':str(root),'plan_sha256':worker.sha(root/'plan.json'),
         'storage':gate,'inputs':inputs,'saved_results_changed':False}
with (parent/'submission.json').open('x') as f:json.dump(receipt,f,indent=2)
jobs=subprocess.run(['bjobs','-l',ids[0]],capture_output=True,text=True,timeout=30)
(parent/'scheduler-submitted.txt').write_text(jobs.stdout)
print(json.dumps(receipt))
'''


def main():
    out = LOCAL / 'launch-01'
    out.mkdir(exist_ok=False)
    package = LOCAL / 'source01'
    p = json.loads((package / 'preparation.json').read_text())
    archive = (package / 'source.tar.gz').read_bytes()
    assert len(archive)==p['archive_bytes'] and hashlib.sha256(archive).hexdigest()==p['archive_sha256']
    values = dict(ARCHIVE=base64.b64encode(archive).decode(),LENGTH=len(archive),SHA=p['archive_sha256'],MANIFEST=p['manifest_sha256'])
    code = '\n'.join(k+'='+repr(v) for k,v in values.items())+'\n'+REMOTE
    (out/'remote-program.py').write_text(code,encoding='utf-8',newline='\n')
    (out/'started.json').write_text(json.dumps({'started_utc':datetime.now(timezone.utc).isoformat(),'program_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2))
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'stdout.json').open('xb') as f,(out/'stderr.txt').open('xb') as e:
        result=subprocess.run(argv,input=code.encode(),stdout=f,stderr=e,timeout=240)
    receipt={'returncode':result.returncode,'ended_utc':datetime.now(timezone.utc).isoformat(),
             'stdout_sha256':hashlib.sha256((out/'stdout.json').read_bytes()).hexdigest()}
    (out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode:
        print((out/'stderr.txt').read_text(errors='replace')[-5000:])
    else:
        data=json.loads((out/'stdout.json').read_bytes())
        print(json.dumps({'job':data['job_id'],'status':data['status'],'required_bytes':data['storage']['required_bytes'],'available_bytes':data['storage']['available_bytes']},indent=2))


if __name__ == '__main__':
    main()
