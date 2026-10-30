"""One-shot immutable preparation/staging of read-only failed-boundary diagnostics."""
import argparse
import ast
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

OWNER=Path(__file__).resolve().parents[1]
REPO=OWNER.parents[2]
TAG='20261030-continuation-failure-diagnostics-07'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/chemistry-continuation-05/boundary-diagnostics-07'
SOURCE=SCOPE+'/boundary-diagnostics-release-07/source'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def prepare():
    out=LOCAL/'source01';assert not out.exists(),'Existing immutable local package'
    observations=REPO/'results/hpc/20261030-mosaic-heartbeat/20261030T000404Z'
    verification=json.loads((observations/'terminal-verification.json').read_bytes())
    assert verification['status']=='passed_independent_terminal_source_byte_LSF_metadata_and_remote_raw_hash_audit'
    captured=json.loads((observations/'stdout.json').read_bytes())['files']
    def proof(suffix):
        rows=[r for n,r in captured.items() if n.endswith(suffix)];assert len(rows)==1
        r=rows[0];b=base64.b64decode(r['base64'],validate=True);assert len(b)==r['bytes'] and sha(b)==r['sha256']
        return {k:r[k] for k in ('path','bytes','sha256')}
    previous=REPO/'results/hpc/20261030-continuation-failure-diagnostics-01/source01'
    plan=json.loads((previous/'plan.json').read_bytes())
    chemistry={'path':SCOPE+'/chemistry-continuation-release-05/source',
               'manifest_sha256':'2b0356e63e74de8c1e112c47b8aedfdad9eaea70b66bf6a790abee29488718d4','files':10}
    logit=proof('/chemistry-continuation-05/boltz-validate/payload/study/low_coupled_checks/report.json')
    diagnostics=proof('/boundary-diagnostics-01/gpu/payload/replay/assessment.json')
    plan.update(version='read-only-actual-producer-boundary-diagnostics-v7',root=ROOT,source_root=SOURCE,
                chemistry_source=chemistry,boltz_logit_failure=logit,previous_simple_diagnostic=diagnostics)
    plan['inputs'] += [logit,diagnostics]
    recovery=REPO/'results/hpc/20261030-mosaic-heartbeat/20261030T015916Z'
    failure=json.loads((recovery/'terminal-verification.json').read_bytes())
    assert failure['job_id']=='29851676' and failure['status']=='verified_reference_allocation_mismatch_before_FP64_model'
    failed_records=json.loads((recovery/'stdout.json').read_bytes())['files']
    for suffix in ('/gpu/stdout.29851676.log','/gpu/payload/replay/plain/assessment.json'):
        candidates=[r for path,r in failed_records.items() if path.endswith(suffix)];assert len(candidates)==1
        row=candidates[0];content=base64.b64decode(row['base64'],validate=True)
        assert len(content)==row['bytes'] and sha(content)==row['sha256']
        plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    for arm,jid in [('protenix','29851426'),('boltz','29851428')]:
        plan['inputs'] += [proof('/chemistry-continuation-05/'+arm+'-validate/stdout.'+jid+'.log'),
                           proof('/chemistry-continuation-05/'+arm+'-validate/payload/driver.log')]
    reconciled=REPO/'results/hpc/20261030-continuation-failure-diagnostics-04/reconciled-accepted-01/stdout.json'
    accepted=json.loads(reconciled.read_bytes())
    receipt=accepted['value']
    assert receipt['job_id']=='29851770' and receipt['status']=='accepted_submission_reconciled_terminal_EXIT_before_model'
    assert receipt['actual_exit_code']==1 and receipt['model_loaded'] is False and receipt['chemistry_submission_attempted'] is False
    for record in accepted['files'].values():
        content=base64.b64decode(record['base64'],validate=True)
        assert len(content)==record['bytes'] and sha(content)==record['sha256']
    plan['inputs'].append(accepted['receipt'])
    retired=json.loads((REPO/'results/hpc/20261030-continuation-failure-diagnostics-06/retire-unstarted01/stdout.json').read_bytes())
    assert retired['value']['job_id']=='29851792' and retired['value']['model_loaded'] is False
    retirement_bytes=base64.b64decode(retired['base64'],validate=True)
    assert len(retirement_bytes)==retired['receipt']['bytes'] and sha(retirement_bytes)==retired['receipt']['sha256']
    plan['inputs'].append(retired['receipt'])

    canonical=json.loads((REPO/'results/hpc/20261030-continuation-failure-diagnostics-04/reconciled-canonical-01/stdout.json').read_bytes())
    assert canonical['value']['returncode'] is None and canonical['value']['scheduler_accepted'] is True
    plan['inputs'].append(canonical['canonical_submission'])
    assert len(canonical['failed05'])==1
    failed_path,failed_text=canonical['failed05'][0]
    plan['inputs'].append({'path':failed_path,'bytes':len(failed_text.encode()),'sha256':sha(failed_text.encode())})

    plan['held_submission_protocol']={'version':'persist-before-release-v1','bsub_option':'-H',
        'receipt_and_registry_before_bresume':True,'unknown_outcome_forbids_retry':True}
    plan['native_GPU_CPU_slots']={'plain':4,'chemistry':8}
    plan['diagnostic_runtime']={'wall_seconds':3600,'entry_seconds':600,'child_seconds':2500,'coordinator_seconds':2700}
    capacity=plan['capacity_source']
    old_manifest=json.loads((previous/'source_manifest.json').read_bytes())
    parents=old_manifest['parent_sources']+[chemistry,{
        'path':SCOPE+'/boundary-diagnostics-release-02/source',
        'manifest_sha256':'6c2f080b5355d2526d690a48e80aad4c65fc6af7bb60f39c2e7b0dc6b12f8879','files':6},{
        'path':SCOPE+'/boundary-diagnostics-release-03/source',
        'manifest_sha256':'c9ee1b491f6e292032662b4fe0a957a66ff71e121a6805fc40c5a0a134b1baec','files':6},{
        'path':SCOPE+'/boundary-diagnostics-release-04/source',
        'manifest_sha256':'54463071451fdc728fc23d7e9d226b7ffee6bef7f8edb931bb1bd3f163df3d9e','files':6},{
        'path':SCOPE+'/boundary-diagnostics-release-05/source',
        'manifest_sha256':'c30234e4a9a8bef64ec58577d2bf9f8a745121bb46b9978051c7c75e2b016943','files':6},{
        'path':SCOPE+'/boundary-diagnostics-release-06/source',
        'manifest_sha256':'2c52935397059e8684d7ae4b1d084b7d5b4a563266f7cd929077052c9354c594','files':6}]
    names={'scripts/continuation_failure_diagnostics.py':'scripts/production_boundary_diagnostics.py',
           'scripts/failure_diagnostic_ops.py':'scripts/failure_diagnostic_ops.py',
           'scripts/failure_diagnostic_stage.lsf':'scripts/failure_diagnostic_stage.lsf',
           'tests/test_continuation_failure_diagnostics.py':'tests/test_production_boundary_diagnostics.py'}
    members={n:(OWNER/local).read_bytes().replace(b'\r\n',b'\n') for n,local in names.items()}
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for n,b in members.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    manifest={'version':plan['version'],'files':{n:sha(b) for n,b in members.items()},'parent_sources':parents,
              'capacity_source':capacity,'cpu_cases':30,'optimizer_updates':0,'can_admit_new_child':False}
    members['source_manifest.json']=raw(manifest);out.mkdir(parents=True)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        for n,b in members.items():
            p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            i=tarfile.TarInfo(n);i.size=len(b);i.mode=0o444;t.addfile(i,io.BytesIO(b))
    archive=buf.getvalue();(out/'source.tar.gz').write_bytes(archive)
    result={'status':'prepared_not_allocated_pass','tag':TAG,'root':ROOT,'source':SOURCE,'files':6,'manifest_sha256':sha(members['source_manifest.json']),
            'archive_bytes':len(archive),'archive_sha256':sha(archive),'plan_sha256':sha(members['plan.json']),'optimizer_updates':0}
    for name,b in [('preparation.json',raw(result)),('source-manifest.json',members['source_manifest.json']),('campaign.json',raw(result))]:(RUN/name).write_bytes(b)
    (out/'preparation.json').write_bytes(raw(result));print(json.dumps(result))


STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation'];root=Path(p['root']);source=Path(p['source'])
scope=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26')
assert root==scope/'chemistry-continuation-05/boundary-diagnostics-07' and source==scope/'boundary-diagnostics-release-07/source'
assert not root.exists() and not source.parent.exists(),'Existing or unknown diagnostic stage; never replay'
data=base64.b64decode(request['archive'],validate=True);assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
 for row in t:
  n=PurePosixPath(row.name);assert row.isfile() and not n.is_absolute() and '..' not in n.parts and '\\' not in row.name and row.name not in members
  members[row.name]=t.extractfile(row).read()
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==p['manifest_sha256']
assert len(m['files'])==6 and set(members)==set(m['files'])|{'source_manifest.json'}
for n,d in m['files'].items():assert hashlib.sha256(members[n]).hexdigest()==d
root.mkdir();source.mkdir(parents=True)
for n,b in members.items():
 f=source/n;f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('xb') as s:s.write(b)
 f.chmod(0o444)
for d in sorted([d for d in source.rglob('*') if d.is_dir()],reverse=True):d.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/failure_diagnostic_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import failure_diagnostic_ops as ops
capacity,plan=ops.context(p['manifest_sha256']);pre=root/'preflight';pre.mkdir();gate=ops.storage(capacity,pre,'preflight')
cpu=ops.submit('cpu',p['manifest_sha256']);callback=ops.submit('after-cpu',p['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_and_conditional_GPU_diagnostic_controller_submitted','cpu':cpu,'callback':callback,'preparation':p,
        'whole_storage':{k:gate[k] for k in ('required_bytes','available_bytes','reserve_bytes','file_reserve')},'optimizer_jobs':[]}
ops.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''


def stage():
    packet=LOCAL/'source01';p=json.loads((packet/'preparation.json').read_bytes());archive=(packet/'source.tar.gz').read_bytes()
    assert sha(archive)==p['archive_sha256'];out=LOCAL/datetime.now(timezone.utc).strftime('stage-%Y%m%dT%H%M%SZ');out.mkdir()
    (out/'program.py').write_bytes(STAGE.encode());(out/'started.json').write_bytes(raw({'status':'unknown_until_actual_receipt','preparation':p}))
    command='/work3/s234818/mosaic/envs/protenix-tiny/bin/python -B -c '+shlex.quote(STAGE)
    r=subprocess.run([r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe','exec','--timeout','1600','bash -lc '+shlex.quote(command)],
        input=raw({'preparation':p,'archive':base64.b64encode(archive).decode()}),capture_output=True,timeout=1630)
    (out/'stdout.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr);(out/'transport.json').write_bytes(raw({'returncode':r.returncode}))
    print(out)
    if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
    v=json.loads(r.stdout);(RUN/'submission.json').write_bytes(raw(v));campaign={**p,'status':v['status'],'jobs':{'cpu':v['cpu']['job_id'],'after-cpu':v['callback']['job_id']}}
    (RUN/'campaign.json').write_bytes(raw(campaign));print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage'));a=p.parse_args()
    prepare() if a.action=='prepare' else stage()
