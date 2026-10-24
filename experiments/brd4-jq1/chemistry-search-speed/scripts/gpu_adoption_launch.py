"""One-shot stdlib package/stage/submit of separate metadata-only correction."""
import ast
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
LOCAL = REPO / 'results/hpc/20261025-chemistry-speed-gpu-adoption-03'
RUN = HERE / 'runs/20261025-chemistry-speed-gpu-adoption-03'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'

REMOTE = r'''
import ast,base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-search-speed-gpu-adoption-03'
release=base/'chemistry-search-speed-gpu-adoption-release-03';source=release/'source'
assert not any(os.path.lexists(p) for p in [root,release]),'Existing/unknown correction; never replay'
assert not os.path.lexists(base/'chemistry-search-speed-02/gpu'),'Existing/unknown GPU; never duplicate'
archive=base64.b64decode(ARCHIVE);assert len(archive)==LENGTH and hashlib.sha256(archive).hexdigest()==SHA
with tarfile.open(fileobj=io.BytesIO(archive)) as t:
 rows=t.getmembers();assert len(rows)==6 and len({r.name for r in rows})==6
 assert all(r.isfile() and not PurePosixPath(r.name).is_absolute() and '..' not in PurePosixPath(r.name).parts for r in rows)
 data={r.name:t.extractfile(r).read() for r in rows}
assert hashlib.sha256(data['source_manifest.json']).hexdigest()==PIN
m=json.loads(data['source_manifest.json']);assert set(data)==set(m['files'])|{'source_manifest.json'}
for name,digest in m['files'].items():
 assert hashlib.sha256(data[name]).hexdigest()==digest
 if name.endswith('.py'):ast.parse(data[name],filename=name)
root.mkdir();(root/'source.tar.gz').write_bytes(archive)
source.mkdir(parents=True)
for name,raw in data.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);release.chmod(0o555)
sys.path.insert(0,str(source/'scripts'));import gpu_adoption as w
w.manifest(PIN);parent=w.original_proofs()
subprocess.run(['bash','-n',str(source/'scripts/gpu_adoption_stage.lsf')],check=True,capture_output=True)
pre=root/'preflight';pre.mkdir();gate=parent.storage(pre,'gpu-adoption03-preflight')
jobs={};submissions={}
for phase in ['cpu','after-cpu']:
 p=root/phase;p.mkdir();dependency='ended('+jobs['cpu']+')' if phase=='after-cpu' else None
 argv=w.expected_argv(PIN,phase,dependency)
 receipt={'status':'unknown','phase':phase,'argv':argv,'dependency':dependency,'manifest_sha256':PIN}
 w.write(p/'submission-started.json',receipt)
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=60)
  import re
  ids=re.findall(r'Job <(\d+)> is submitted',r.stdout)
  receipt.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  if r.returncode==0 and len(ids)==1:receipt.update(status='submitted',job_id=ids[0])
 except BaseException as error:receipt['error']=repr(error)
 finally:w.write(p/'submission.json',receipt)
 assert receipt['status']=='submitted','Unknown outcome; do not duplicate'
 jobs[phase]=receipt['job_id'];submissions[phase]=receipt
 partial=root/('jobs-'+phase+'.partial');w.write(partial,jobs);partial.replace(root/'jobs.json')
 q=subprocess.run(['bjobs','-l',receipt['job_id']],capture_output=True,text=True,timeout=30)
 w.write(p/'scheduler-at-submission.json',{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
result={'status':'metadata_CPU_and_ownended_callback_submitted_not_passed','jobs':jobs,'submissions':submissions,
        'manifest_sha256':PIN,'archive_sha256':SHA,'archive_bytes':LENGTH,'metadata_cases':19,'storage':gate,
        'parent_source_manifest_sha256':w.PARENT_PIN,'original_CPU_143_checks_preserved':True,'GPU_submitted':False}
w.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''


def main():
    LOCAL.mkdir(exist_ok=False)
    members = {}
    for name in ['scripts/gpu_adoption.py', 'scripts/gpu_adoption_terminal.py',
                 'scripts/gpu_adoption_stage.lsf', 'tests/test_gpu_adoption_terminal.py']:
        raw = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
        members[name] = raw
    members['README.md'] = (RUN / 'README.md').read_bytes().replace(b'\r\n', b'\n')
    old = REPO / 'results/hpc/20261025-chemistry-speed-02/observation-11'
    listing = json.loads((old / 'raw.json').read_bytes())['files']
    failure = []
    for rel in ['after-cpu/stdout.29809478.log', 'after-cpu/stderr.29809478.log',
                'after-cpu/cpu-final-scheduler.json']:
        p = old / 'files' / rel; digest = hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest == listing[rel]['sha256']
        failure.append({'path': BASE + 'chemistry-search-speed-02/' + rel, 'sha256': digest})
    m = {'version': 'chemistry-GPU-admission-scheduler-lag-v1', 'metadata_cases': 19,
         'parent_source_manifest_sha256': '09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222',
         'scientific_source_changed': False, 'preserved_callback_failure': failure,
         'files': {k: hashlib.sha256(v).hexdigest() for k, v in sorted(members.items())}}
    raw = (json.dumps(m, indent=2, sort_keys=True) + '\n').encode(); pin = hashlib.sha256(raw).hexdigest()
    members['source_manifest.json'] = raw
    package = LOCAL / 'source.tar.gz'
    with tarfile.open(package, 'x:gz') as t:
        for name, data in sorted(members.items()):
            item = tarfile.TarInfo(name); item.size = len(data); item.mode = 0o444; item.mtime = 0
            t.addfile(item, io.BytesIO(data))
    (LOCAL / 'source_manifest.json').write_bytes(raw); (RUN / 'source-manifest.json').write_bytes(raw)
    archive = package.read_bytes(); digest = hashlib.sha256(archive).hexdigest()
    prep = {'status': 'AST_byte_packaged_not_allocated_pass', 'manifest_sha256': pin,
            'archive_sha256': digest, 'archive_bytes': len(archive), 'files': 5, 'metadata_cases': 19}
    (RUN / 'campaign.json').write_text(json.dumps(prep, indent=2) + '\n', encoding='utf-8', newline='\n')
    bindings = {'ARCHIVE': base64.b64encode(archive).decode(), 'LENGTH': len(archive), 'SHA': digest, 'PIN': pin}
    code = '\n'.join(k + '=' + repr(v) for k, v in bindings.items()) + '\n' + REMOTE
    ast.parse(code)
    launch = LOCAL / 'launch-01'; launch.mkdir(exist_ok=False)
    (launch / 'remote-program.py').write_text(code, encoding='utf-8', newline='\n')
    (launch / 'started.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
                                                   'remote_program_sha256': hashlib.sha256(code.encode()).hexdigest()}, indent=2))
    argv = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (launch / 'stdout.json').open('xb') as f, (launch / 'stderr.txt').open('xb') as e:
        r = subprocess.run(argv, input=code.encode(), stdout=f, stderr=e, timeout=240)
    status = {'returncode': r.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
              'stdout_sha256': hashlib.sha256((launch / 'stdout.json').read_bytes()).hexdigest()}
    (launch / 'result.json').write_text(json.dumps(status, indent=2) + '\n'); print(json.dumps(status))
    if r.returncode:
        print((launch / 'stderr.txt').read_text(errors='replace')[-6000:])
        raise SystemExit(r.returncode)
    result = json.loads((launch / 'stdout.json').read_bytes())
    (RUN / 'campaign.json').write_text(json.dumps({**prep, **result}, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'jobs': result['jobs'], **prep}, indent=2))


if __name__ == '__main__':
    main()
