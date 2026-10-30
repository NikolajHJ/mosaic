"""One-shot authenticated production integration preparation and initial CPU stage."""
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

OWNER=Path(__file__).resolve().parents[1];REPO=OWNER.parents[2]
TAG='20261030-protenix-runtime-production-03';LOCAL=REPO/'results/hpc'/TAG;RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
SCOPE=BASE+'/nonchem-precision-continuation-26';ROOT=SCOPE+'/runtime-production-03';SOURCE=BASE+'/runtime-production-release-03/source'
QROOT=SCOPE+'/chemistry-continuation-05/runtime-qualification-01';QSOURCE=SCOPE+'/runtime-qualification-release-01/source'
QP='9108a418953a4ee7f4d481667ac4a5035402c96f10179f1e8a44c3358af965ab'
def raw(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(value):return hashlib.sha256(value).hexdigest()


def prepare():
    out=LOCAL/'source01';assert not out.exists(),'Existing package must never be overwritten'
    qualified=REPO/'results/hpc/20261030-protenix-runtime-qualification-01'
    audit=json.loads((qualified/'status04/terminal-verification.json').read_bytes())
    assert audit['status']=='passed_independent_complete_plain_runtime_qualification_source_input_NPZ_LSF_native_pilot_runtime_capture_audit'
    packet=json.loads((qualified/'status04/stdout.json').read_bytes())
    def binding(suffix):
        rows=[r for p,r in packet['files'].items() if p==QROOT+suffix];assert len(rows)==1
        row=rows[0];data=base64.b64decode(row['base64'],validate=True);assert sha(data)==row['sha256'] and len(data)==row['bytes']
        return {k:row[k] for k in ('path','bytes','sha256')}
    qm=(qualified/'source02/source_manifest.json').read_bytes();assert sha(qm)==QP
    original=REPO/'results/hpc/20261029-precision-continuation-26/source01'
    plan=json.loads((original/'plan.json').read_bytes());old=json.loads(qm)
    inherit=json.loads((qualified/'source02/plan.json').read_bytes())
    for key in ('capacity_source','chemistry_source','helper_source','context_ready','execution_profile','inputs'):
        plan[key]=inherit[key]
    plain='/gpu-plain/payload/replay/plain/payload'
    pins={'ready':binding(plain+'/study/assessment.json'),'summary':binding(plain+'/study/summary.json'),
        'result':binding(plain+'/assessment.json'),'capture':binding('/after-plain/payload/assessment.json'),
        'native_lsf':binding('/gpu-plain/stdout.29852401.log'),'capture_lsf':binding('/after-plain/stdout.29852403.log'),
        'cpu_certificate':binding('/cpu-certificate.json'),'preservation':binding('/gpu-plain/payload/preservation.json')}
    result=json.loads(base64.b64decode(packet['files'][pins['result']['path']]['base64']))
    plan.update(version='protenix-qualified-runtime-production-v1',root=SCOPE,run_root=ROOT,source_root=SOURCE,
        qualified={**pins,'study':QROOT+plain+'/study','native_parent':QROOT+'/gpu-plain','native_job_id':'29852401',
                   'capture_parent':QROOT+'/after-plain','capture_job_id':'29852403'},
        runtime_budget=result['precision_runtime_budget'],
        production_policy={'arm':'protenix','phase':'segment','offset':20000,'updates':30000,'global_target':50000,
            'learning_rate':0.3,'matrix_stride':20,'replicas_per_native_checkpoint':16})
    plan['inputs']+=list(pins.values())
    for suffix in ('/cpu/payload/tests.xml','/cpu/payload/torch.xml','/cpu/payload/jax.xml',
                   '/cpu/payload/chemistry.xml','/cpu/payload/qualification.xml','/cpu/payload/startup/assessment.json',
                   '/cpu/stdout.29852397.log','/after-cpu/stdout.29852398.log'):
        plan['inputs'].append(binding(suffix))
    previous=REPO/'results/hpc/20261030-protenix-runtime-production-01/status02/stdout.json'
    previous_packet=json.loads(previous.read_bytes())
    assert previous_packet['jobs']=={'cpu':'29852547','after-cpu':'29852549'}
    previous_root=SCOPE+'/runtime-production-01'
    for suffix in ('/cpu/stdout.29852547.log','/cpu/stderr.29852547.log'):
        row=previous_packet['files'][previous_root+suffix];data=base64.b64decode(row['base64'],validate=True)
        assert sha(data)==row['sha256'] and len(data)==row['bytes']
        if 'stdout.' in suffix:assert b'Exited with exit code 1.' in data and b'Successfully completed.' not in data
        else:assert b"No module named 'chemistry_continuation_contract'" in data
        plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    second=REPO/'results/hpc/20261030-protenix-runtime-production-02/status03'
    assert json.loads((second/'verification.json').read_bytes())['CPU_cases']==1035
    previous_packet=json.loads((second/'stdout.json').read_bytes())
    assert previous_packet['jobs']=={'cpu':'29852577','after-cpu':'29852578'}
    for suffix in ('/after-cpu/stdout.29852578.log','/after-cpu/stderr.29852578.log'):
        row=previous_packet['files'][SCOPE+'/runtime-production-02'+suffix];data=base64.b64decode(row['base64'],validate=True)
        assert sha(data)==row['sha256'] and len(data)==row['bytes']
        if 'stdout.' in suffix:assert b'Exited with exit code 1.' in data and b'Successfully completed.' not in data
        else:assert b"KeyError: 'readiness'" in data
        plan['inputs'].append({k:row[k] for k in ('path','bytes','sha256')})
    goals=plan['rolling_storage_policy']['deferred_goals']
    assert not any(r['id']=='dbmol' for r in goals)
    goals.append({'id':'dbmol','saved_updates':7500,'target_updates':10000,'next_updates':0,'next_scope_id':None,
        'later_updates_deferred':2500,'readmission_before_later_execution':True})
    assert sum(json.loads((original/'source_manifest.json').read_bytes())['cpu_counts'].values())==958
    chem=REPO/'results/hpc/20261030-chemistry-continuation-05/source01/source'
    inherited={**json.loads((original/'source_manifest.json').read_bytes())['cpu_counts'],
        **json.loads((chem/'source_manifest.json').read_bytes())['cpu_counts']}
    own={'tests.test_runtime_production':32};counts={**inherited,**own};assert sum(counts.values())==1037
    names={**{'scripts/'+name+'.py':'scripts/'+name+'.py' for name in ('runtime_production_driver','runtime_production_capacity','runtime_production_ops')},
        'scripts/runtime_production_stage.lsf':'scripts/runtime_production_stage.lsf','tests/test_runtime_production.py':'tests/test_runtime_production.py'}
    members={name:(OWNER/local).read_bytes().replace(b'\r\n',b'\n') for name,local in names.items()}
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for name,data in members.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    parents=old['parent_sources']+[{'path':QSOURCE,'manifest_sha256':QP,'files':6},
        {'path':BASE+'/runtime-production-release-01/source','manifest_sha256':'443b4c098bf859163a1fd16ee70bfc7d8ab7b9fbf730c2b32ac19c285155ceda','files':7},
        {'path':BASE+'/runtime-production-release-02/source','manifest_sha256':'69029fea3bdce08737d9f67e11edcbf6e36594266f2b4518f1c34d0da42f6bb1','files':7}]
    manifest={'version':plan['version'],'files':{name:sha(data) for name,data in members.items()},'parent_sources':parents,
        'capacity_source':plan['capacity_source'],'cpu_counts':own,'all_cpu_counts':counts,
        'optimization_targets':{'protenix':50000},'original_failures_preserved':True}
    members['source_manifest.json']=raw(manifest);out.mkdir(parents=True)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as archive:
        for name,data in members.items():
            path=out/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;archive.addfile(item,io.BytesIO(data))
    data=buf.getvalue();(out/'source.tar.gz').write_bytes(data)
    result={'status':'prepared_not_allocated_pass','tag':TAG,'root':ROOT,'source':SOURCE,'files':len(manifest['files']),
        'manifest_sha256':sha(members['source_manifest.json']),'plan_sha256':sha(members['plan.json']),
        'archive_bytes':len(data),'archive_sha256':sha(data),'CPU_cases':sum(counts.values()),'optimizer_jobs':[]}
    (out/'preparation.json').write_bytes(raw(result))
    for name,data in [('preparation.json',raw(result)),('source-manifest.json',members['source_manifest.json']),('campaign.json',raw(result))]:
        (RUN/name).write_bytes(data)
    print(json.dumps(result))


STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation'];root=Path(p['root']);source=Path(p['source'])
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
assert root==base/'nonchem-precision-continuation-26/runtime-production-03' and source==base/'runtime-production-release-03/source'
assert not root.exists() and not source.parent.exists(),'Existing or unknown production stage; never replay'
data=base64.b64decode(request['archive'],validate=True);assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for row in archive:
  name=PurePosixPath(row.name);assert row.isfile() and not name.is_absolute() and '..' not in name.parts and '\\' not in row.name and row.name not in members
  members[row.name]=archive.extractfile(row).read()
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==p['manifest_sha256']
assert len(m['files'])==p['files'] and set(members)==set(m['files'])|{'source_manifest.json'}
for n,h in m['files'].items():assert hashlib.sha256(members[n]).hexdigest()==h
root.mkdir();source.mkdir(parents=True)
for name,data in members.items():
 path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(data)
 path.chmod(0o444)
for path in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):path.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/runtime_production_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import runtime_production_ops as ops
op,plan=ops.context(p['manifest_sha256']);parent=root/'preflight';parent.mkdir();gate=ops.storage(op,parent,'preflight')
cpu=ops.submit('cpu',p['manifest_sha256']);callback=ops.submit('after-cpu',p['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_and_conditional_full_remainder_controller_submitted','cpu':cpu,'callback':callback,'preparation':p,
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
    value=json.loads(r.stdout);(RUN/'submission.json').write_bytes(raw(value))
    campaign={**p,'status':value['status'],'jobs':{'cpu':value['cpu']['job_id'],'after-cpu':value['callback']['job_id']}}
    (RUN/'campaign.json').write_bytes(raw(campaign));print(json.dumps(value))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','stage'));args=parser.parse_args()
    prepare() if args.action=='prepare' else stage()
