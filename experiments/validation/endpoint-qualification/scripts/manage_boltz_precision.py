"""Seal and stage the explicitly funded complete-objective reference candidate."""
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
TAG='20261030-boltz-objective-precision-02';LOCAL=REPO/'results/hpc'/TAG;RUN=OWNER/'runs'/TAG
INPUT_LOCAL=REPO/'results/hpc/20261030-boltz-objective-precision-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01';SCOPE=BASE+'/nonchem-precision-continuation-26'
ROOT=SCOPE+'/runtime-production-03/boltz-precision-02';SOURCE=SCOPE+'/boltz-precision-release-02/source'
SHARED=BASE+'/runtime-production-release-03/source';PIN='6356f3a7068658e17b25b5bdcc5239feb5cfc51e6cdeb1894e47e4752a86e0f8'
def raw(x):return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()

def prepare():
    output=LOCAL/'source01';assert not output.exists(),'Immutable prepared package already exists'
    packets=[json.loads((INPUT_LOCAL/p/'stdout.json').read_bytes()) for p in ('inspection01','inspection02','inspection04')];files={}
    for packet in packets:
        for name,row in packet['files'].items():
            if 'base64' in row:
                data=base64.b64decode(row['base64'],validate=True);assert sha(data)==row['sha256'] and len(data)==row['bytes']
            files[name]=row
    def row(path):return {k:files[path][k] for k in ('path','bytes','sha256')}
    def read(path):return json.loads(base64.b64decode(files[path]['base64']))
    old=read(SHARED+'/source_manifest.json');assert files[SHARED+'/source_manifest.json']['sha256']==PIN
    main=read(BASE+'/nonchem-precision-release-26/source/plan.json');db=next(v for v in main['arms'] if v['id']=='dbmol')
    chemistry=read(SCOPE+'/chemistry-continuation-release-05/source/plan.json')['arms']['boltz']
    plain=BASE+'/nonchem-endpoint-continuation-09/boltz-segment-10000/payload/study'
    plain_report=read(plain+'/terminal_gradient_checks/report.json')
    chem_study=SCOPE+'/chemistry-continuation-05/boltz-validate/payload/study'
    chem_report=read(chem_study+'/low_coupled_checks/report.json')
    total=read(SCOPE+'/dbmol-validate-07500/payload/study/native_gradient_checks-prospective/dbmol_total_loss/admission.json')
    dd=[{k:d[k] for k in ('field','row','source','target')} for d in total['directions']
        if not d['pair_passed'] or d['field']=='X' and d['row']==1]
    assert {d['row'] for d in dd if d['field']=='E_upper'}=={209,195,264} and len(dd)==4
    cd=[{'field':'X' if d['field']=='atoms' else 'E_upper','row':d['row'],'source':d['source'],'target':d['destination']}
        for d in chem_report['directions']]
    # The raw simplex checker aborted before producing rows. These two logit
    # controls are explicitly predeclared diagnostics, not invented original rows.
    assert plain_report['status']=='failed' and not plain_report['rows']
    pd=[{'field':'X','row':15,'source':1,'target':4},
        {'field':'E_upper','row':132,'source':1,'target':0}]
    checkpoints=json.loads((INPUT_LOCAL/'inspection03/stdout.json').read_bytes())
    assert checkpoints['boltz2_aff.ckpt']['sha256']=='dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e'
    assert checkpoints['boltz2_conf.ckpt']['sha256']=='090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1'
    science=db['source'];bolscience=chemistry['science_source']
    plan={'version':'complete-boltz-objective-reference-diagnostic-v1','root':ROOT,'source_root':SOURCE,
        'shared_capacity':{'path':SHARED,'files':7,'manifest_sha256':PIN},
        'capacity_science':{'path':BASE+'/nonchem-precision-release-26/source','files':50,'manifest_sha256':'8324c1d312adeaa4c67c475c01ef1217df6f3ad4044b997d1b73e168325412d3'},
        'science':science,'chemistry_source':{'path':SCOPE+'/chemistry-continuation-release-05/source','files':10,
            'manifest_sha256':'2b0356e63e74de8c1e112c47b8aedfdad9eaea70b66bf6a790abee29488718d4'},
        'context':chemistry['context'],'affinity_checkpoint':checkpoints['boltz2_aff.ckpt'],
        'structure_checkpoint':checkpoints['boltz2_conf.ckpt'],'worker_python':chemistry['worker_python'],
        'arms':{'boltz':{'point':row(plain+'/case-0/snapshots/step-10000.npz'),'source':bolscience,
                       'config':row(bolscience['path']+'/examples/configs/boltz_serial_boltz2.json'),'directions':pd,
                       'direction_scope':'predeclared_logit_controls_original_raw_checker_has_no_rows',
                       'saved_updates':20000,'target_updates':30000},
                'boltz_chemistry':{'point':row(chem_study+'/low_coupled_checks/point-0000.npz'),'source':bolscience,
                    'config':chemistry['config'],'directions':cd,'saved_updates':10000,'target_updates':30000,
                    'adapter_source':chemistry['adapter_source'],'preparation_identity':chem_report['preparation_identity'],
                    'point_scope':'original_failed_invalid_assay_not_a_modified_saved_molecule'},
                'dbmol':{'point':db['parent_snapshot'],'source':science,'config':db['config'],'directions':dd,
                         'saved_updates':7500,'target_updates':10000}},
        'inputs':list(checkpoints.values())+[chemistry['context'],row(plain+'/terminal_gradient_checks/report.json'),
            row(chem_study+'/low_coupled_checks/report.json'),row(SCOPE+'/dbmol-validate-07500/payload/study/native_gradient_checks-prospective/dbmol_total_loss/admission.json')],
        'optimizer_updates':0,'can_admit_new_child':False,'output_cap_bytes':256*1024**2,'file_cap':2048,
        'fixed_original_chemistry_teacher_identity_required':True,'full_original_qualification_still_required':True,
        'production_optimizer_settings_changed':False,'acceptance_tolerances_changed':False}
    for arm in plan['arms'].values():plan['inputs']+=[arm['point'],arm['config']]
    # Only these identical immutable modules are shared by all arm loaders.
    comparison=packets[-1]['common_source_comparison']
    plan['identical_scientific_modules']={n:v['dbmol'] for n,v in comparison.items() if not n.endswith('/trajectory.py')}
    assert len(plan['identical_scientific_modules'])==19 and all(v['equal'] for n,v in comparison.items() if not n.endswith('/trajectory.py'))
    unique={p['path']:p for p in plan['inputs']};plan['inputs']=list(unique.values())
    external=json.loads((INPUT_LOCAL/'foreign-writer02/stdout.json').read_bytes())
    observed=json.loads((INPUT_LOCAL/'reconcile01/stdout.json').read_bytes())
    account=json.loads(observed['commands'][0]['stdout'])['RECORDS']
    command=next(r['COMMAND'] for r in account if r['JOBID']=='29852732')
    plan['shared_source146']={k:external[k] for k in ('baseline','proofs','source_files_authenticated','growth_bytes','growth_files')}
    plan['shared_source146'].update(job_id='29852732',command=command)
    parents=old['parent_sources']+[plan['shared_capacity'],science,bolscience,chemistry['adapter_source']]
    # The exact original complete cleanup helper is used by the existing05 source.
    cp=read(SCOPE+'/chemistry-continuation-release-05/source/plan.json')['arms']['protenix']['adapter_source']
    if cp['path'] not in {p['path'] for p in parents}:parents.append(cp)
    names={**{'scripts/'+n+'.py':'scripts/'+n+'.py' for n in ('boltz_objective_reference','boltz_precision_driver','boltz_precision_server','boltz_precision_ops','qualification_writer_capacity')},
        'scripts/boltz_precision_stage.lsf':'scripts/boltz_precision_stage.lsf',
        **{'tests/'+n+'.py':'tests/'+n+'.py' for n in ('test_boltz_precision_portable','test_boltz_reference_torch','test_boltz_reference_jax','test_qualification_writer_capacity')}}
    members={name:(OWNER/local).read_bytes().replace(b'\r\n',b'\n') for name,local in names.items()}
    members['plan.json']=raw(plan);members['README.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for name,data in members.items():
        if name.endswith('.py'):ast.parse(data,filename=name)
    tc={'tests.test_boltz_reference_torch':8};jc={'tests.test_boltz_precision_portable':11,'tests.test_boltz_reference_jax':3,'tests.test_qualification_writer_capacity':8}
    m={'version':plan['version'],'files':{n:sha(b) for n,b in members.items()},'parent_sources':parents,
        'shared_capacity':plan['shared_capacity'],'torch_cpu_counts':tc,'jax_cpu_counts':jc,
        'all_cpu_counts':{'tests.test_dbmol_reference':17,**tc,**jc},'optimizer_submission_allowed':False}
    members['source_manifest.json']=raw(m);output.mkdir(parents=True);buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name,data in members.items():
            path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;archive.addfile(item,io.BytesIO(data))
    data=buffer.getvalue();(output/'source.tar.gz').write_bytes(data)
    value={'status':'prepared_not_allocated_pass','tag':TAG,'root':ROOT,'source':SOURCE,'files':len(m['files']),
        'manifest_sha256':sha(members['source_manifest.json']),'plan_sha256':sha(members['plan.json']),
        'archive_bytes':len(data),'archive_sha256':sha(data),'CPU_cases':47,'optimizer_jobs':[]}
    for name,data in [('preparation.json',raw(value)),('source-manifest.json',members['source_manifest.json']),('campaign.json',raw(value))]:(RUN/name).write_bytes(data)
    (output/'preparation.json').write_bytes(raw(value));print(json.dumps(value))

STAGE=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,io,json,subprocess,sys,tarfile
request=json.loads(sys.stdin.buffer.read());p=request['preparation'];root=Path(p['root']);source=Path(p['source'])
scope=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-precision-continuation-26')
assert root==scope/'runtime-production-03/boltz-precision-02' and source==scope/'boltz-precision-release-02/source'
assert not root.exists() and not source.parent.exists(),'Existing or unknown stage; do not repeat'
data=base64.b64decode(request['archive'],validate=True);assert len(data)==p['archive_bytes'] and hashlib.sha256(data).hexdigest()==p['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive:
  n=PurePosixPath(member.name);assert member.isfile() and not n.is_absolute() and '..' not in n.parts and '\\' not in member.name and member.name not in members
  members[member.name]=archive.extractfile(member).read()
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==p['manifest_sha256']
assert len(m['files'])==p['files'] and set(members)==set(m['files'])|{'source_manifest.json'}
for n,h in m['files'].items():assert hashlib.sha256(members[n]).hexdigest()==h
root.mkdir();source.mkdir(parents=True)
for n,b in members.items():
 path=source/n;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as stream:stream.write(b)
 path.chmod(0o444)
for path in sorted([x for x in source.rglob('*') if x.is_dir()],reverse=True):path.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/boltz_precision_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import boltz_precision_ops as ops
op,plan=ops.context(p['manifest_sha256']);parent=root/'preflight';parent.mkdir();gate=ops.storage(op,parent,'preflight')
cpu=ops.submit('cpu',p['manifest_sha256']);callback=ops.submit('after-cpu',p['manifest_sha256'],'ended('+cpu['job_id']+')')
result={'status':'CPU_and_conditional_three_GPU_reference_controller_submitted','cpu':cpu,'callback':callback,
 'preparation':p,'whole_storage':{k:gate[k] for k in ('required_bytes','available_bytes','reserve_bytes','file_reserve')},'optimizer_jobs':[]}
ops.write(root/'initial-submissions.json',result);print(json.dumps(result))
'''

def stage():
    package=LOCAL/'source01';p=json.loads((package/'preparation.json').read_bytes());archive=(package/'source.tar.gz').read_bytes()
    assert sha(archive)==p['archive_sha256'];out=LOCAL/datetime.now(timezone.utc).strftime('stage-%Y%m%dT%H%M%SZ');out.mkdir()
    (out/'program.py').write_text(STAGE);(out/'started.json').write_bytes(raw({'status':'unknown_until_actual_receipt','preparation':p}))
    command='/work3/s234818/mosaic/envs/protenix-tiny/bin/python -B -c '+shlex.quote(STAGE)
    result=subprocess.run([r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe','exec','--timeout','1600','bash -lc '+shlex.quote(command)],
        input=raw({'preparation':p,'archive':base64.b64encode(archive).decode()}),capture_output=True,timeout=1630)
    (out/'stdout.json').write_bytes(result.stdout);(out/'stderr.txt').write_bytes(result.stderr);(out/'transport.json').write_bytes(raw({'returncode':result.returncode}))
    print(out)
    if result.returncode:print(result.stderr.decode());raise SystemExit(result.returncode)
    data=json.loads(result.stdout);(RUN/'submission.json').write_bytes(raw(data));(RUN/'campaign.json').write_bytes(raw({**p,
        'status':data['status'],'jobs':{'cpu':data['cpu']['job_id'],'after-cpu':data['callback']['job_id']}}));print(json.dumps(data))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','stage'));args=parser.parse_args()
    prepare() if args.action=='prepare' else stage()
