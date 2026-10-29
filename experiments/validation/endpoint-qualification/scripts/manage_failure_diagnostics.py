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
TAG='20261030-continuation-failure-diagnostics-01'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/boundary-diagnostics-01'
SOURCE=SCOPE+'/boundary-diagnostics-release-01/source'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def prepare():
    out=LOCAL/'source01';assert not out.exists(),'Existing immutable local package'
    captured=json.loads((REPO/'results/hpc/20261030-mosaic-heartbeat/failure-details01/stdout.json').read_bytes())['files']
    def proof(suffix):
        rows=[r for n,r in captured.items() if n.endswith(suffix)];assert len(rows)==1
        r=rows[0];b=base64.b64decode(r['base64'],validate=True);assert len(b)==r['bytes'] and sha(b)==r['sha256']
        return {k:r[k] for k in ('path','bytes','sha256')}
    capacity={'path':BASE+'/nonchem-precision-release-26/source','manifest_sha256':'8324c1d312adeaa4c67c475c01ef1217df6f3ad4044b997d1b73e168325412d3','files':50}
    forensic={'path':BASE+'/protenix-endpoint-forensics-release-01/source','manifest_sha256':'d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8','files':13}
    loader=(OWNER/'scripts/protenix_forensics_native.py').read_bytes().replace(b'\r\n',b'\n')
    assert sha(loader)=='11523a3e8e3543fa763260398c8897a78e8889e4ac615ca7e88dfdd8d487e387'
    native={'path':forensic['path']+'/scripts/protenix_forensics_native.py','bytes':len(loader),'sha256':sha(loader)}
    context={'path':BASE+'/restart-lrreset-04/validate-protenix/payload/study/assessment.json','bytes':21266,'sha256':'9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'}
    old=json.loads((REPO/'results/hpc/20261029-precision-continuation-26/source01/plan.json').read_bytes())
    arm=next(r for r in old['arms'] if r['id']=='protenix')
    db=proof('/dbmol_total_loss/admission.json');pro=proof('/protenix-segment-20000/payload/assessment.json')
    plan={'version':'read-only-continuation-failure-diagnostics-v1','root':ROOT,'source_root':SOURCE,
          'capacity_source':capacity,'native_loader':native,'context_ready':context,'dbmol_failure':db,
          'inputs':[db,pro,native,context,arm['parent_snapshot'],arm['config'],arm['inputs']['pocket']],
          'optimizer_updates':0,'can_admit_new_child':False,'can_certify_or_restart_original_study':False,
          'production_environment':{'XLA_FLAGS':'--xla_gpu_autotune_level=0'},'max_bytes':128*1024**2}
    parents=[capacity,forensic,arm['source']]
    members={}
    for name in ('scripts/continuation_failure_diagnostics.py','scripts/failure_diagnostic_ops.py','scripts/failure_diagnostic_stage.lsf','tests/test_continuation_failure_diagnostics.py'):
        members[name]=(OWNER/name).read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for n,b in members.items():
        if n.endswith('.py'):ast.parse(b,filename=n)
    manifest={'version':plan['version'],'files':{n:sha(b) for n,b in members.items()},'parent_sources':parents,
              'capacity_source':capacity,'cpu_cases':8,'optimizer_updates':0,'can_admit_new_child':False}
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
assert root==scope/'boundary-diagnostics-01' and source==scope/'boundary-diagnostics-release-01/source'
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
    r=subprocess.run([r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe','exec','--timeout','300','bash -lc '+shlex.quote(command)],
        input=raw({'preparation':p,'archive':base64.b64encode(archive).decode()}),capture_output=True,timeout=330)
    (out/'stdout.json').write_bytes(r.stdout);(out/'stderr.txt').write_bytes(r.stderr);(out/'transport.json').write_bytes(raw({'returncode':r.returncode}))
    print(out)
    if r.returncode:print(r.stderr.decode());raise SystemExit(r.returncode)
    v=json.loads(r.stdout);(RUN/'submission.json').write_bytes(raw(v));campaign={**p,'status':v['status'],'jobs':{'cpu':v['cpu']['job_id'],'after-cpu':v['callback']['job_id']}}
    (RUN/'campaign.json').write_bytes(raw(campaign));print(json.dumps(v))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage'));a=p.parse_args()
    prepare() if a.action=='prepare' else stage()
