"""Authenticate allocated operator03 and exclusively stage the approved protection.

Metadata-only caller. It does not create the backup operation directory,
stream originals, remove a file, submit a job or change a sealed operator.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RESULTS = REPO / 'results/hpc/20261024-storage-offload-01'

REMOTE = r'''
import hashlib,json,os,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
source=root/'operator-source-03'
payload=json.loads(sys.stdin.buffer.read(64*1024**2+1))
def raw(path):
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.resolve()==p
 return p.read_bytes()
def sha(data):return hashlib.sha256(data).hexdigest()
manifest=raw(source/'source-manifest.json')
assert sha(manifest)==payload['source_manifest_sha256']=='9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
assert not (source/'source-manifest.json').stat().st_mode&0o222
m=json.loads(manifest);assert len(m['files'])==11
assert set(p.name for p in source.iterdir())==set(m['files'])|{'source-manifest.json'}
for name,digest in m['files'].items():
 p=source/name;assert sha(raw(p))==digest and not p.stat().st_mode&0o222
assert not source.stat().st_mode&0o222
for name,pin in payload['cpu_proofs'].items():
 data=raw(name);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']
parent=root/'cpu-03'
rows=list(ET.fromstring(raw(parent/'payload/operator-contract.xml')).iter('testcase'))
assert len(rows)==32 and all(not list(row) for row in rows)
assert json.loads(raw(parent/'payload/assessment.json'))['status']=='passed'
log=raw(parent/'stdout.29806228.log').decode()
assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',log,re.M)==[('29806228','Done')]
assert 'Successfully completed.' in log
protection=payload['protection_raw'].encode('utf8')
assert sha(protection)==payload['protection_sha256']
assert json.loads(protection)['version']=='mosaic-storage-offload-protection-v1'
assert root.is_dir() and root.resolve()==root
assert not os.path.lexists(root/'legacy-offload-01'),'Existing backup operation/unknown outcome; inspect'
target=root/'protection-01.json'
assert not os.path.lexists(target),'Existing staged protection/unknown outcome; inspect'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True;sys.path.insert(0,str(source))
import offload_protocol as protocol
observation=protocol.active_jobs()
assert {row['job_id'] for row in observation['jobs']}<=set(payload['audited_active_job_ids'])
fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444)
with os.fdopen(fd,'wb') as stream:
 stream.write(protection);stream.flush();os.fsync(stream.fileno())
assert raw(target)==protection
print(json.dumps({'status':'approved_protection_staged_only','source_manifest_sha256':sha(manifest),
 'cpu_job_id':'29806228','cpu_cases':len(rows),'protection_path':str(target),
 'protection_bytes':len(protection),'protection_sha256':sha(protection),'queue':observation,
 'operation_directory_created':False,'backup_or_deletion_started':False}))
'''


def main():
    approval = json.loads((HERE / 'approved-selection-01.json').read_bytes())
    protection = (HERE / 'flat-protection-approved-01.json').read_bytes()
    assert hashlib.sha256(protection).hexdigest() == approval['protection_sha256']
    plan_raw = (RESULTS / 'backup-01/approved-plan.json').read_bytes()
    assert hashlib.sha256(plan_raw).hexdigest() == approval['plan_sha256']
    plan = json.loads(plan_raw)
    assert plan['operation_id'] == 'legacy-offload-01'
    assert plan['protection_manifest'] == {
        'path': '/work3/s234818/mosaic/runs/20261024-storage-offload-01/protection-01.json',
        'sha256': approval['protection_sha256']}
    audit = json.loads((HERE / 'operator-verification-03.json').read_bytes())
    assert audit['status'] == 'passed' and audit['cases'] == 32 and audit['job_id'] == '29806228'
    prefix = '/work3/s234818/mosaic/runs/20261024-storage-offload-01/cpu-03/'
    proofs = {name: pin for name, pin in audit['proofs'].items() if name.startswith(prefix)}
    assert len(proofs) == 9
    payload = {'source_manifest_sha256': audit['source_manifest_sha256'], 'cpu_proofs': proofs,
        'protection_raw': protection.decode('utf8'), 'protection_sha256': approval['protection_sha256'],
        'audited_active_job_ids': plan['audited_active_job_ids']}
    ast.parse(REMOTE)
    folder = RESULTS / 'protection-staging-01'; folder.mkdir(exist_ok=False)
    (folder / 'remote-caller.py').write_text(REMOTE, encoding='utf8', newline='\n')
    argv = ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', REMOTE]
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        'dtu-hpc', 'bash -l -c ' + shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder / 'started.json').write_text(json.dumps({'outcome': 'unknown', 'args': args,
        'utc': datetime.now(timezone.utc).isoformat(), 'plan_sha256': approval['plan_sha256']}) + '\n')
    try:
        with (folder / 'stdout.bin').open('xb') as out, (folder / 'stderr.bin').open('xb') as err:
            result = subprocess.run(args, input=json.dumps(payload).encode(), stdout=out, stderr=err, timeout=120)
    except BaseException as error:
        (folder / 'failed-unknown.json').write_text(json.dumps({'error': repr(error), 'outcome': 'unknown_inspect_remote'})+'\n')
        raise
    (folder / 'result.json').write_text(json.dumps({'returncode': result.returncode})+'\n')
    print((folder / 'stdout.bin').read_text()); print((folder / 'stderr.bin').read_text())
    assert result.returncode == 0


if __name__ == '__main__':
    main()
