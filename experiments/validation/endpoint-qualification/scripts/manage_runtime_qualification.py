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
TAG='20261030-protenix-runtime-qualification-01'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/chemistry-continuation-05/runtime-qualification-01'
SOURCE=SCOPE+'/runtime-qualification-release-01/source'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()


def prepare():
    out=LOCAL/'source02';assert not out.exists(),'Existing immutable local package'
    prior=REPO/'results/hpc/20261030-continuation-failure-diagnostics-10/source01'
    expected='38bdf38decb4c3d96ad270e24d6b9c1a1d6a81980020d474db6adbbf70270d22'
    old_raw=(prior/'source_manifest.json').read_bytes();assert sha(old_raw)==expected
    old=json.loads(old_raw);plan_raw=(prior/'plan.json').read_bytes();assert sha(plan_raw)==old['files']['plan.json']
    plan=json.loads(plan_raw)
    delivery=REPO/'results/hpc/20261030-continuation-failure-diagnostics-10/status06'
    audit=json.loads((delivery/'terminal-verification.json').read_bytes())
    assert audit['status']=='passed_independent_candidate10_complete_native_diagnostic_terminal_source_input_raw_binding_budget_audit'
    assert audit['findings']['chemistry']['complete_diagnostic_pass'] is True
    packet=json.loads((delivery/'stdout.json').read_bytes())
    for suffix in ('/gpu-chemistry/stdout.29852164.log','/gpu-chemistry/payload/preservation.json',
                   '/gpu-chemistry/payload/replay/chemistry/assessment.json',
                   '/gpu-chemistry/payload/replay/chemistry/payload/study/native_gradient_checks-prospective/admission.json',
                   '/gpu-chemistry/payload/replay/chemistry/payload/study/native_gradient_checks-prospective/reference-session/worker/close-audit.json'):
        rows=[r for path,r in packet['files'].items() if path.endswith(suffix)];assert len(rows)==1
        row=rows[0];data=base64.b64decode(row['base64'],validate=True);assert sha(data)==row['sha256'] and len(data)==row['bytes']
        plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    helper={'path':SCOPE+'/boundary-diagnostics-release-10/source','files':6,'manifest_sha256':expected}
    plan.update(version='complete-protenix-runtime-qualification-v1',root=ROOT,source_root=SOURCE,helper_source=helper,
        qualification_arms=['plain','chemistry'],qualification_policy={'phase':'validate','optimizer_submission_allowed':False,
        'both_complete_chemistry_pilots_required':True,'replicas_per_native_checkpoint':16})
    plan.pop('diagnostic_arms',None)
    plan['diagnostic_storage']['group_cap_bytes']=512*1024**2
    main=json.loads((REPO/'results/hpc/20261029-precision-continuation-26/source01/source_manifest.json').read_bytes())['cpu_counts']
    chem=json.loads((REPO/'results/hpc/20261030-chemistry-continuation-05/source01/source/source_manifest.json').read_bytes())['cpu_counts']
    own={'tests.test_runtime_qualification':24};counts={**main,**chem,**own}
    assert sum(main.values())==958 and sum(chem.values())==47 and sum(counts.values())==1029
    names={'scripts/continuation_failure_diagnostics.py':'scripts/protenix_runtime_qualification.py',
        'scripts/failure_diagnostic_ops.py':'scripts/protenix_runtime_qualification_ops.py',
        'scripts/failure_diagnostic_stage.lsf':'scripts/failure_diagnostic_stage.lsf',
        'tests/test_runtime_qualification.py':'tests/test_protenix_runtime_qualification.py'}
    members={name:(OWNER/path).read_bytes().replace(b'\r\n',b'\n') for name,path in names.items()}
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for name,data in members.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    manifest={'version':plan['version'],'files':{name:sha(data) for name,data in members.items()},
        'parent_sources':old['parent_sources']+[helper],'capacity_source':plan['capacity_source'],
        'cpu_counts':own,'all_cpu_counts':counts,'optimizer_continuation_updates':0,'optimizer_submission_allowed':False}
    members['source_manifest.json']=raw(manifest);out.mkdir(parents=True)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,data in members.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;archive.addfile(item,io.BytesIO(data))
    data=buf.getvalue();(out/'source.tar.gz').write_bytes(data)
    result={'status':'prepared_not_allocated_pass','tag':TAG,'root':ROOT,'source':SOURCE,'files':6,
        'manifest_sha256':sha(members['source_manifest.json']),'plan_sha256':sha(members['plan.json']),
        'archive_bytes':len(data),'archive_sha256':sha(data),'CPU_cases':1029,'optimizer_continuation_updates':0}
    for name,data in [('preparation.json',raw(result)),('source-manifest.json',members['source_manifest.json']),('campaign.json',raw(result))]:
        (RUN/name).write_bytes(data)
    (out/'preparation.json').write_bytes(raw(result));print(json.dumps(result))


STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation'];root=Path(p['root']);source=Path(p['source'])
scope=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26')
assert root==scope/'chemistry-continuation-05/runtime-qualification-01' and source==scope/'runtime-qualification-release-01/source'
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
result={'status':'CPU_and_conditional_full_validation_controller_submitted','cpu':cpu,'callback':callback,'preparation':p,
        'whole_storage':{k:gate[k] for k in ('required_bytes','available_bytes','reserve_bytes','file_reserve')},'optimizer_jobs':[]}
ops.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''


def stage():
    packet=LOCAL/'source02';p=json.loads((packet/'preparation.json').read_bytes());archive=(packet/'source.tar.gz').read_bytes()
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
