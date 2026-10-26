"""One-shot fresh immutable capsule stage and allocated CPU/callback submission."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261026-chemistry-full-model-05'
REMOTE = r'''
import ast,base64,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path,PurePosixPath
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-full-model-05';release=base/'chemistry-full-model-release-05';source=release/'source'
assert not any(os.path.lexists(p) for p in [root,release]),'Existing/unknown study outcome; inspect, never replay'
archive=base64.b64decode(ARCHIVE,validate=True)
assert len(archive)==LENGTH and hashlib.sha256(archive).hexdigest()==SHA
with tarfile.open(fileobj=io.BytesIO(archive)) as target:
 rows=target.getmembers()
 assert len(rows)==MEMBERS and len({r.name for r in rows})==len(rows)
 assert all(r.isfile() and not PurePosixPath(r.name).is_absolute() and '..' not in PurePosixPath(r.name).parts for r in rows)
 data={r.name:target.extractfile(r).read() for r in rows}
assert hashlib.sha256(data['source_manifest.json']).hexdigest()==MANIFEST
m=json.loads(data['source_manifest.json']);assert set(data)==set(m['files'])|{'source_manifest.json'}
for name,pin in m['files'].items():
 assert hashlib.sha256(data[name]).hexdigest()==pin
 if name.endswith('.py'):ast.parse(data[name],filename=name)
root.mkdir()
with (root/'source.tar.gz').open('xb') as f:f.write(archive)
source.mkdir(parents=True)
for name,raw in data.items():
 p=source/name;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('xb') as f:f.write(raw)
 p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);release.chmod(0o555)
sys.path.insert(0,str(source/'scripts'))
import full_model_ops as ops
ops.verify(source,MANIFEST);ops.external(m)
subprocess.run(['bash','-n',str(source/'scripts/full_model_stage.lsf')],check=True,capture_output=True)
pre=root/'preflight';pre.mkdir();gate=ops.storage(pre,'preflight')
qualified=ops.context_tools().qualify_predecessor(ops.context_api(),m)
ops.write(pre/'actual-replay-context.json',qualified)
cpu=ops.submit('cpu',MANIFEST);after=ops.submit('after-cpu',MANIFEST,'ended('+cpu['job_id']+')')
receipt={'status':'CPU_and_ownended_callback_submitted_not_passed','cpu':cpu,'after_cpu':after,
 'source_manifest_sha256':MANIFEST,'archive_sha256':SHA,'archive_bytes':LENGTH,'source_files':MEMBERS-1,
 'storage':gate,'full_model_jobs_submitted':False,
 'plan_sha256':{arm:ops.sha(source/'plans'/(arm+'.json')) for arm in ['protenix','boltz']}}
ops.write(root/'initial-submissions.json',receipt)
print(json.dumps(receipt))
'''


def main():
    out = LOCAL / 'launch-01'; out.mkdir(exist_ok=False)
    package = LOCAL / 'source01'
    preparation = json.loads((package / 'preparation.json').read_bytes())
    raw = (package / 'source.tar.gz').read_bytes()
    assert len(raw) == preparation['archive_bytes'] and hashlib.sha256(raw).hexdigest() == preparation['archive_sha256']
    bindings = {'ARCHIVE': base64.b64encode(raw).decode(), 'LENGTH': len(raw),
                'SHA': preparation['archive_sha256'], 'MANIFEST': preparation['manifest_sha256'],
                'MEMBERS': preparation['files']+1}
    code = '\n'.join(key+'='+repr(value) for key, value in bindings.items()) + '\n' + REMOTE
    (out / 'remote-program.py').write_text(code, encoding='utf-8', newline='\n')
    (out / 'started.json').write_text(json.dumps({'started_utc': datetime.now(timezone.utc).isoformat(),
              'program_sha256': hashlib.sha256(code.encode()).hexdigest()}, indent=2)+'\n')
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
               'dtu-hpc', 'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    try:
        with (out/'stdout.json').open('xb') as output, (out/'stderr.txt').open('xb') as error:
            result = subprocess.run(command, input=code.encode(), stdout=output, stderr=error, timeout=240)
        receipt = {'returncode': result.returncode, 'completed_utc': datetime.now(timezone.utc).isoformat(),
                   'stdout_sha256': hashlib.sha256((out/'stdout.json').read_bytes()).hexdigest()}
    except BaseException as error:
        receipt = {'status': 'unknown_inspect_remote_before_any_recovery', 'error': repr(error)}
        raise
    finally:
        (out/'result.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))
    if result.returncode:
        print((out/'stderr.txt').read_text(errors='replace')[-6000:])
        raise SystemExit(result.returncode)
    data = json.loads((out/'stdout.json').read_bytes())
    print(json.dumps({'cpu': data['cpu']['job_id'], 'after_cpu': data['after_cpu']['job_id'],
                      'status': data['status'], 'required_bytes': data['storage']['required_bytes'],
                      'available_bytes': data['storage']['available_bytes']}, indent=2))


if __name__ == '__main__':
    main()
