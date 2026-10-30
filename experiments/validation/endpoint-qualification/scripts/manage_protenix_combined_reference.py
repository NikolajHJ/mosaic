"""Prepare/stage a fresh, complete Protenix chemistry reference qualification."""
import argparse
import ast
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import tarfile

OWNER=Path(__file__).resolve().parents[1];REPO=OWNER.parents[2]
TAG='20261030-protenix-combined-reference-03'
LOCAL=REPO/'results/hpc'/TAG;RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/chemistry-continuation-05/combined-reference-qualification-03'
SOURCE=SCOPE+'/combined-reference-qualification-release-03/source'
Q=SCOPE+'/runtime-qualification-release-01/source';QP='9108a418953a4ee7f4d481667ac4a5035402c96f10179f1e8a44c3358af965ab'
P=BASE+'/runtime-production-release-05/source';PP='36e1bfcd9acb864a557e3cf2c317334626fd596337d39306b352c1b0131da11f'


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()


def prepare():
    out=LOCAL/'source01';assert not out.exists(),'Immutable package already exists'
    prior=REPO/'results/hpc/20261030-protenix-runtime-qualification-01/source02'
    old=prior.joinpath('source_manifest.json').read_bytes();assert sha(old)==QP
    manifest=json.loads(old);plan=json.loads((prior/'plan.json').read_bytes())
    bp=json.loads((REPO/'results/hpc/20261030-boltz-objective-precision-05/source01/plan.json').read_bytes())
    parents=manifest['parent_sources']+[{'path':Q,'manifest_sha256':QP,'files':6},
        {'path':P,'manifest_sha256':PP,'files':11}]
    failed=REPO/'results/hpc/20261030-protenix-combined-reference-01/status01/stdout.json'
    packet=json.loads(failed.read_bytes());old_root=SCOPE+'/chemistry-continuation-05/combined-reference-qualification-01'
    previous={'path':SCOPE+'/combined-reference-qualification-release-01/source','files':12,
        'manifest_sha256':'4dcfed8fcc095095c6c747ab7849ce1f5268eec9d92ce04742123cd179ae8d38'}
    parents.append(previous)
    for phase,jid in [('cpu','29853536'),('after-cpu','29853540')]:
        for name in ('stdout.','stderr.'):
            row=packet['files'][old_root+'/'+phase+'/'+name+jid+'.log'];b=base64.b64decode(row['base64'],validate=True)
            assert sha(b)==row['sha256'] and len(b)==row['bytes']
            assert (b'Exited with exit code 1.' in b) if name=='stdout.' else (b'Unknown nonterminal Mosaic allocation(s)' in b)
            plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    assert not any('/payload/tests.xml' in p for p in packet['files'])
    packet=json.loads((REPO/'results/hpc/20261030-protenix-combined-reference-02/status02/stdout.json').read_bytes())
    previous_root=SCOPE+'/chemistry-continuation-05/combined-reference-qualification-02'
    parents.append({'path':SCOPE+'/combined-reference-qualification-release-02/source','files':12,
        'manifest_sha256':'85a812d69c85860800e854763cee286962e059507e1e2853bc09c0efbc408e7b'})
    for phase,jid in [('cpu','29853562'),('after-cpu','29853563')]:
        row=packet['files'][previous_root+'/'+phase+'/stdout.'+jid+'.log'];b=base64.b64decode(row['base64'],validate=True)
        assert sha(b)==row['sha256'] and b'Exited with exit code 1.' in b
        plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    row=packet['files'][previous_root+'/cpu/payload/combined-reference.xml'];b=base64.b64decode(row['base64'],validate=True)
    assert sha(b)==row['sha256'];plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    plan.update(root=ROOT,source_root=SOURCE,qualified_helper={'path':Q,'manifest_sha256':QP,'files':6},
        combined_reference_policy='protenix-fixed-chemistry-full-reference-v1',
        shared_capacity={'path':P,'manifest_sha256':PP,'files':11},
        shared_source146=bp['shared_source146'],storage_policy=bp['storage_policy'])
    plan['combined_reference_scope']={'production':'unchanged_FP32_original_AD',
        'reference':'original_full_FP64_binding_and_private_CPU64_fixed_chemistry',
        'same_represented_logit_stencil':True,'same_probabilities_between_precisions':False,
        'all_original_low_high_actual_cohorts_and_two_pilots':True,'optimizer_submission_allowed':False}
    names={'scripts/continuation_failure_diagnostics.py':'scripts/protenix_chemistry_qualification.py',
        'scripts/failure_diagnostic_ops.py':'scripts/protenix_chemistry_qualification_ops.py',
        'scripts/failure_diagnostic_stage.lsf':'scripts/failure_diagnostic_stage.lsf',
        'scripts/protenix_combined_reference.py':'scripts/protenix_combined_reference.py',
        'scripts/storage_policy_adapter.py':'scripts/storage_policy_adapter.py',
        'scripts/qualification_writer_capacity.py':'scripts/qualification_writer_capacity.py',
        'tests/test_protenix_combined_reference.py':'tests/test_protenix_combined_reference.py'}
    members={n:(OWNER/p).read_bytes().replace(b'\r\n',b'\n') for n,p in names.items()}
    shell=members['scripts/failure_diagnostic_stage.lsf']
    assert shell.count(b'[[ $# == 5 ')==1
    shell=shell.replace(b'[[ $# == 5 ',b'[[ $# == 7 ')
    shell=shell.replace(b'capacity_source=$5\n',b'capacity_source=$5\n[[ "$6" == --shared-capacity-source && "$7" == /work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/runtime-production-release-05/source ]]\n')
    members['scripts/failure_diagnostic_stage.lsf']=shell
    # Synthetic tests use exact inherited measurement implementation, not a copy
    # of the concurrently edited common-policy working tree.
    main=REPO/'results/hpc/20261029-precision-continuation-26/source01'
    mm=json.loads((main/'source_manifest.json').read_bytes())
    for name in ('admission.py','logit_admission.py'):
        value=(main/'scripts'/name).read_bytes();assert sha(value)==mm['files']['scripts/'+name]
        members['scripts/'+name]=value
    members['scripts/storage_accounting.py']=(REPO/'scripts/hpc/storage_accounting.py').read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for name,b in members.items():
        if name.endswith('.py'):ast.parse(b,filename=name)
    own={'tests.test_protenix_combined_reference':17}
    m={'version':'complete-combined-reference-qualification-v1','files':{n:sha(b) for n,b in members.items()},
        'parent_sources':parents,'capacity_source':plan['capacity_source'],'shared_capacity':plan['shared_capacity'],
        'cpu_counts':own,'all_cpu_counts':{**manifest['all_cpu_counts'],**own},'optimizer_submission_allowed':False}
    assert sum(m['all_cpu_counts'].values())==1046
    members['source_manifest.json']=raw(m);out.mkdir(parents=True);buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,b in members.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
            info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o444;archive.addfile(info,io.BytesIO(b))
    data=buf.getvalue();(out/'source.tar.gz').write_bytes(data)
    value={'status':'prepared_not_numerically_qualified','tag':TAG,'root':ROOT,'source':SOURCE,'files':len(m['files']),
        'manifest_sha256':sha(members['source_manifest.json']),'plan_sha256':sha(members['plan.json']),
        'archive_bytes':len(data),'archive_sha256':sha(data),'CPU_cases':1046,'optimizer_jobs':[]}
    for name,b in [('preparation.json',raw(value)),('source-manifest.json',members['source_manifest.json']),('campaign.json',raw(value))]:
        (RUN/name).write_bytes(b)
    (out/'preparation.json').write_bytes(raw(value));print(json.dumps(value))


STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,subprocess,sys,tarfile
r=json.loads(sys.stdin.buffer.read());p=r['preparation'];root=Path(p['root']);source=Path(p['source'])
scope=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26')
assert root==scope/'chemistry-continuation-05/combined-reference-qualification-03' and source==scope/'combined-reference-qualification-release-03/source'
assert not root.exists() and not source.parent.exists(),'Existing/unknown stage: do not replay'
b=base64.b64decode(r['archive'],validate=True);assert len(b)==p['archive_bytes'] and hashlib.sha256(b).hexdigest()==p['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as t:
 for row in t:
  part=PurePosixPath(row.name);assert row.isfile() and not part.is_absolute() and '..' not in part.parts and '\\' not in row.name and row.name not in members
  members[row.name]=t.extractfile(row).read()
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==p['manifest_sha256']
assert len(m['files'])==p['files'] and set(members)==set(m['files'])|{'source_manifest.json'}
for n,h in m['files'].items():assert hashlib.sha256(members[n]).hexdigest()==h
root.mkdir();source.mkdir(parents=True)
for n,b in members.items():
 path=source/n;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(b)
 path.chmod(0o444)
for d in sorted([d for d in source.rglob('*') if d.is_dir()],reverse=True):d.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/failure_diagnostic_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import failure_diagnostic_ops as ops
op,plan=ops.context(p['manifest_sha256']);pre=root/'preflight';pre.mkdir();gate=ops.storage(op,pre,'preflight')
cpu=ops.submit('cpu',p['manifest_sha256']);callback=ops.submit('after-cpu',p['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_and_conditional_complete_chemistry_GPU_qualification_submitted','cpu':cpu,'callback':callback,'preparation':p,
 'whole_storage':{k:gate[k] for k in ('required_bytes','available_bytes','reserve_bytes','file_reserve')},'optimizer_jobs':[]}
ops.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''


def stage():
    packet=LOCAL/'source01';p=json.loads((packet/'preparation.json').read_bytes());b=(packet/'source.tar.gz').read_bytes()
    assert sha(b)==p['archive_sha256'];out=LOCAL/datetime.now(timezone.utc).strftime('stage-%Y%m%dT%H%M%SZ');out.mkdir()
    (out/'program.py').write_text(STAGE);(out/'started.json').write_bytes(raw({'status':'unknown_until_actual_receipt','preparation':p}))
    command='/work3/s234818/mosaic/envs/protenix-tiny/bin/python -B -c '+shlex.quote(STAGE)
    result=subprocess.run([r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe','exec','--timeout','1600','bash -lc '+shlex.quote(command)],
        input=raw({'preparation':p,'archive':base64.b64encode(b).decode()}),capture_output=True,timeout=1630)
    (out/'stdout.json').write_bytes(result.stdout);(out/'stderr.txt').write_bytes(result.stderr)
    (out/'transport.json').write_bytes(raw({'returncode':result.returncode}));print(out)
    if result.returncode:print(result.stderr.decode());raise SystemExit(result.returncode)
    value=json.loads(result.stdout);(RUN/'submission.json').write_bytes(raw(value))
    (RUN/'campaign.json').write_bytes(raw({**p,'status':value['status'],'jobs':{'cpu':value['cpu']['job_id'],'after-cpu':value['callback']['job_id']}}))
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','stage'));a=p.parse_args()
    prepare() if a.action=='prepare' else stage()
