"""Fresh read-only remote source/CPU/protection/absence check before backup02."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
ROOT='/work3/s234818/mosaic/runs/20261024-storage-offload-01'
MANIFEST='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
REMOTE=r'''
import hashlib,json,os,re,sys
from pathlib import Path
import xml.etree.ElementTree as ET
root=Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
source=root/'operator-source-04';payload=json.loads(sys.stdin.buffer.read(1024*1024))
def raw(path):
 p=Path(path);assert p.is_file() and not p.is_symlink() and p.resolve()==p
 return p.read_bytes()
def sha(data):return hashlib.sha256(data).hexdigest()
manifest=raw(source/'source-manifest.json')
assert sha(manifest)==payload['source_manifest_sha256']=='8a6420047f7260eedd1219c4313c835dae4bb2a13cd39d4914f9e87abcd5bafe'
m=json.loads(manifest);assert len(m['files'])==11 and m['allocated_fixture_rows_expected']==44 and m['gzip_compresslevel']==1
assert {p.name for p in source.iterdir()}==set(m['files'])|{'source-manifest.json'} and not source.stat().st_mode&0o222
for name in ['source-manifest.json',*m['files']]:
 p=source/name;data=raw(p);assert not p.stat().st_mode&0o222
 assert sha(data)==(sha(manifest) if name=='source-manifest.json' else m['files'][name])
for name,pin in payload['cpu_proofs'].items():
 data=raw(name);assert len(data)==pin['bytes'] and sha(data)==pin['sha256']
parent=root/'cpu-04';rows=list(ET.fromstring(raw(parent/'payload/operator-contract.xml')).iter('testcase'))
assert len(rows)==44 and all(not list(row) for row in rows)
assert json.loads(raw(parent/'payload/assessment.json'))['status']=='passed'
log=raw(parent/'stdout.29806439.log').decode()
assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',log,re.M)==[('29806439','Done')] and 'Successfully completed.' in log
assert sha(raw(root/'protection-01.json'))==payload['protection_sha256']=='35c78f3571b26304e453eb9da1e04c1fe0bdd05d0a6e3756e81e3f825d121511'
assert sha(raw(root/'stream-retirement-complete-01.json'))=='41d238499d97bdb0e415c684a5ccf95bb2c73b7264395c310816e144e83988ab'
assert root.resolve()==root and not os.path.lexists(root/'legacy-offload-02'),'Existing operation02/unknown outcome; inspect'
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.path.insert(0,str(source))
import offload_protocol as protocol
observation=protocol.active_jobs()
assert {x['job_id'] for x in observation['jobs']}<=set(payload['audited_active_job_ids'])
print(json.dumps({'status':'authenticated_backup02_remote_readiness','read_only':True,
 'source_manifest_sha256':sha(manifest),'cpu_job_id':'29806439','cpu_cases':44,
 'protection_sha256':payload['protection_sha256'],'approved_plan_sha256':payload['plan_sha256'],
 'operation02_absent':True,'queue':observation,'files_deleted':0,'backup_replayed':False}))
'''


def main():
    approval=json.loads((HERE/'approved-selection-02.json').read_bytes())
    raw=(REPO/'results/hpc/20261024-storage-offload-01/backup-02/approved-plan.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==approval['plan_sha256']
    plan=json.loads(raw);assert plan['operation_id']=='legacy-offload-02' and plan['operator_source_manifest_sha256']==MANIFEST
    audit=json.loads((HERE/'operator-verification-04.json').read_bytes())
    assert audit['status']=='passed' and audit['cases']==44 and audit['job_id']=='29806439' and audit['source_manifest_sha256']==MANIFEST
    prefix=ROOT+'/cpu-04/'
    proofs={p:v for p,v in audit['proofs'].items() if p.startswith(prefix)};assert len(proofs)==9
    payload={'source_manifest_sha256':MANIFEST,'cpu_proofs':proofs,
        'protection_sha256':approval['protection_sha256'],'plan_sha256':approval['plan_sha256'],
        'audited_active_job_ids':plan['audited_active_job_ids']}
    folder=REPO/'results/hpc/20261024-storage-offload-01/backup-02/remote-readiness-01';folder.mkdir(exist_ok=False)
    argv=['/work3/s234818/mosaic/envs/protenix-tiny/bin/python','-c',REMOTE]
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(' '.join(shlex.quote(x) for x in argv))]
    (folder/'started.json').write_text(json.dumps({'outcome':'unknown','args':command})+'\n')
    try:
        with (folder/'stdout.bin').open('xb') as out,(folder/'stderr.bin').open('xb') as err:
            result=subprocess.run(command,input=json.dumps(payload).encode(),stdout=out,stderr=err,timeout=90)
    except BaseException as error:
        (folder/'failed.json').write_text(json.dumps({'error':repr(error),'read_only':True})+'\n');raise
    (folder/'result.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    print((folder/'stdout.bin').read_text());assert result.returncode==0


if __name__=='__main__':main()
