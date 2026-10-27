"""Stdlib operator for the distinct chemistry native-qualification04 capsule."""
import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
TAG='20261027-chemistry-native-qualification-04'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/chemistry-native-qualification-04'
SOURCE=BASE+'/chemistry-native-qualification-release-04/source'
PARENT=BASE+'/chemistry-endpoint-recovery-release-02/source'
PIN='d3d8678542db3cf6100006f8f9dbd3ce761f169bd904ddcae3d4525a53252f51'
PRECISION=BASE+'/protenix-endpoint-fix-release-07/source'
PRECISION_PIN='63db0a002c8aca2a282d50f9e7227d57b69ac6820b8ae0e06287ec95770dd8a3'


def transport():
    path=OWNER/'scripts/manage_chemistry_endpoint.py'
    spec=importlib.util.spec_from_file_location('_qualification_transport',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.LOCAL=LOCAL
    m.ROOT=ROOT;m.SOURCE=SOURCE
    return m


def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(v):return hashlib.sha256(v).hexdigest()


def prepare():
    m=transport()
    code=m.HEADER+'NEWROOT='+repr(ROOT)+'\nNEWSOURCE='+repr(SOURCE)+'\nPRECISION='+repr(PRECISION)+'\n'+r'''
assert not Path(NEWROOT).exists() and not Path(NEWSOURCE).parent.exists(),'Existing/unknown candidate; inspect before preparation'
parent=Path(SOURCE);pm=verify(parent,'d3d8678542db3cf6100006f8f9dbd3ce761f169bd904ddcae3d4525a53252f51',10)
plan=json.loads((parent/'plan.json').read_bytes());protected={r['path']:r for r in plan['protected_inputs']}
for r in protected.values():assert binding(r['path'])==r
prior=Path(ROOT)
assert json.loads((prior/'jobs.json').read_bytes())=={'cpu':'29830696','after-cpu':'29830697','gpu-protenix':'29830750','gpu-boltz':'29830751'}
for phase,job in json.loads((prior/'jobs.json').read_bytes()).items():
 text=(prior/phase/('stdout.'+job+'.log')).read_text();text=''.join(text.split())
 assert text.count('Subject:Job'+job+':')==1 and 'Successfullycompleted.' in text and 'Exitedwithexit' not in text
for arm in ('protenix','boltz'):
 result=json.loads((prior/('gpu-'+arm)/'payload/native/assessment.json').read_bytes())
 assert result['status']=='diagnostics_completed_original_failure_preserved' and result['protected_inputs_unchanged'] and result['frozen_unchanged']
 assert result['can_certify_or_restart_original_study'] is False
for p in prior.rglob('*'):
 assert not p.is_symlink() and (p.is_dir() or p.is_file())
 if p.is_file():protected[str(p)]=binding(p)
precision=Path(PRECISION);pr=verify(precision,'63db0a002c8aca2a282d50f9e7227d57b69ac6820b8ae0e06287ec95770dd8a3',18)
adoption={}
for name,source in [('admission.py',base/'nonchem-endpoint-release-09/source'),
 ('protenix_precision_reference.py',precision),('protenix_precision_admission.py',precision)]:
 adoption['scripts/'+name]=binding(source/'scripts'/name)
parents=plan['parent_sources']+[{'path':str(parent),'manifest_sha256':sha(parent/'source_manifest.json'),'files':10},
 {'path':str(precision),'manifest_sha256':sha(precision/'source_manifest.json'),'files':18}]
failed=base/'chemistry-native-qualification-03';failed_source=base/'chemistry-native-qualification-release-03/source'
verify(failed_source,'966034fd61600cca47f30fb0e939671bd678490a8fcdf4960d2d570b080d0b9e',12)
assert json.loads((failed/'jobs.json').read_bytes())=={'cpu':'29833834','after-cpu':'29833835'}
for phase,job in [('cpu','29833834'),('after-cpu','29833835')]:
 text=''.join((failed/phase/('stdout.'+job+'.log')).read_text().split())
 assert text.count('Subject:Job'+job+':')==1 and 'Exitedwithexitcode1.' in text
assert not (failed/'cpu/payload').exists() and not (failed/'gpu-protenix').exists() and not (failed/'gpu-boltz').exists()
for p in failed.rglob('*'):
 assert not p.is_symlink() and (p.is_file() or p.is_dir())
 if p.is_file():protected[str(p)]=binding(p)
parents.append({'path':str(failed_source),'manifest_sha256':sha(failed_source/'source_manifest.json'),'files':12})

plan.update(output_root=NEWROOT,protected_inputs=list(protected.values()),parent_sources=parents,
 qualification_version='chemistry-native-component-qualification-v1',can_submit_continuation=False,
 diagnostic_source=parents[-3],precision_source=parents[-2],adopted_files=adoption,
 prior_failed_qualification=parents[-1],inventory_retry_policy='exact-observed-atomic-temp-full-gate-three-attempts-v1',
 native_component_only=True,combined_chemistry_admission=False,original_failures_preserved=True,
 max_native_bytes=512*1024**2,internal_seconds=5300)
print(json.dumps({'status':'originals_and_complete_diagnostic_point_inventory_authenticated','plan':plan}))
'''
    packet=m.connect(code,'inputs01',300);plan=packet['plan'];members={}
    names=('scripts/chemistry_native_qualification.py','scripts/chemistry_qualification_ops.py',
           'tests/test_chemistry_native_qualification.py','scripts/admission.py',
           'scripts/protenix_precision_reference.py','scripts/protenix_precision_admission.py')
    for name in names:
        content=(OWNER/name).read_bytes().replace(b'\r\n',b'\n');ast.parse(content,filename=name)
        if name in plan['adopted_files']:assert sha(content)==plan['adopted_files'][name]['sha256'],name
        members[name]=content
    shell=(OWNER/'scripts/chemistry_endpoint_stage.lsf').read_bytes().replace(b'\r\n',b'\n')
    shell=shell.replace(b'chemistry_endpoint_ops.py',b'chemistry_qualification_ops.py')
    members['scripts/chemistry_endpoint_stage.lsf']=shell
    members['plan.json']=raw(plan)
    members['protocol.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    for path in ('README.md','scripts/README.md','tests/README.md'):
        members[path]=b'# Separate chemistry native component qualification\n\nAllocated DTU HPC only. No optimizer or original recertification. See protocol.md.\n'
    counts={}
    tree=ast.parse(members['tests/test_chemistry_native_qualification.py'])
    for c in tree.body:
        if isinstance(c,ast.ClassDef):
            count=sum(isinstance(n,ast.FunctionDef) and n.name.startswith('test_') for n in c.body)
            if count:counts['tests.test_chemistry_native_qualification.'+c.name]=count
    assert sum(counts.values())==29
    manifest={'version':'chemistry-native-qualification-source-v1','files':{k:sha(v) for k,v in sorted(members.items())},
        'parent_sources':plan['parent_sources'],'cpu_counts':counts,'diagnostic_only':True,
        'optimizer_updates':0,'original_results_changed':False,'can_submit_continuation':False}
    members['source_manifest.json']=raw(manifest)
    out=LOCAL/'source01';out.mkdir(exist_ok=False);buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
        for name,data in sorted(members.items()):
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(data))
    archive=buffer.getvalue();(out/'source.tar.gz').write_bytes(archive)
    prep={'status':'prepared_not_validated','files':len(manifest['files']),'manifest_sha256':sha(members['source_manifest.json']),
        'archive_bytes':len(archive),'archive_sha256':sha(archive),'plan_sha256':sha(members['plan.json']),
        'protected_input_files':len(plan['protected_inputs']),'cpu_counts':counts,'cpu_cases':29}
    (out/'preparation.json').write_bytes(raw(prep));(RUN/'preparation.json').write_bytes(raw(prep))
    (RUN/'source-manifest.json').write_bytes(members['source_manifest.json'])
    (RUN/'campaign.json').write_bytes(raw({**prep,'optimizer_jobs_submitted':False,'settings_changed':False,
        'saved_results_changed':False,'monitor_resumed':False}))
    print(json.dumps(prep,indent=2))


def stage():
    m=transport();out=LOCAL/'source01';prep=json.loads((out/'preparation.json').read_bytes())
    data=(out/'source.tar.gz').read_bytes();assert len(data)==prep['archive_bytes'] and sha(data)==prep['archive_sha256']
    code=m.HEADER.replace('ROOT='+repr(BASE+'/chemistry-endpoint-recovery-02'),'ROOT='+repr(ROOT))
    code=code.replace('SOURCE='+repr(PARENT),'SOURCE='+repr(SOURCE))
    code+='PREP='+repr(prep)+'\nARCHIVE='+repr(base64.b64encode(data).decode())+'\n'+r'''
import io,subprocess,tarfile
assert not root.exists() and not source.parent.exists(),'Existing/unknown stage; never duplicate'
data=base64.b64decode(ARCHIVE,validate=True);assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for item in tar:
  rel=Path(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts and item.name not in members
  members[item.name]=tar.extractfile(item).read();assert len(members[item.name])==item.size
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==PREP['manifest_sha256']
assert len(m['files'])==PREP['files'] and set(members)==set(m['files'])|{'source_manifest.json'}
for name,pin in m['files'].items():assert hashlib.sha256(members[name]).hexdigest()==pin
for parent in m['parent_sources']:verify(Path(parent['path']),parent['manifest_sha256'],parent['files'])
for r in json.loads(members['plan.json'])['protected_inputs']:assert binding(r['path'])==r
root.mkdir();source.mkdir(parents=True)
for name,data in members.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555);subprocess.run(['bash','-n',str(source/'scripts/chemistry_endpoint_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import chemistry_qualification_ops as adapter
op=adapter.inherited();op.external(op.verify(source,PREP['manifest_sha256']))
preflight=root/'preflight';preflight.mkdir();gate=op.storage(preflight,'preflight')
cpu=op.submit('cpu',PREP['manifest_sha256']);callback=op.submit('after-cpu',PREP['manifest_sha256'],'ended('+cpu['job_id']+')')
print(json.dumps({'status':'CPU_and_own_callback_submitted_not_passed','cpu':cpu,'callback':callback,'whole_storage':gate}))
'''
    try:
        value=m.connect(code,'stage01',420)
    except BaseException:
        campaign=json.loads((RUN/'campaign.json').read_bytes())
        receipt=LOCAL/'stage01/receipt.json'
        campaign.update(status='staging_outcome_unknown_inspect_before_recovery',
            stage_receipt=json.loads(receipt.read_bytes()) if receipt.exists() else None,
            remote_source_staged='unknown',cpu_job='unknown',callback_job='unknown',
            native_jobs='unknown_do_not_duplicate',continuation_submission='not_authorized_by_this_capsule')
        (RUN/'campaign.json').write_bytes(raw(campaign))
        raise
    (RUN/'submission.json').write_bytes(raw(value));campaign=json.loads((RUN/'campaign.json').read_bytes())
    campaign.update(status=value['status'],cpu_job=value['cpu']['job_id'],callback_job=value['callback']['job_id'],remote_source_staged=True)
    (RUN/'campaign.json').write_bytes(raw(campaign));print(json.dumps({'cpu':value['cpu']['job_id'],'callback':value['callback']['job_id']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage','observe','inspect'))
    p.add_argument('--label');a=p.parse_args();assert REPO.name=='mosaic'
    if a.action=='prepare':prepare()
    elif a.action=='stage':stage()
    else:
        m=transport();m.HEADER=m.HEADER.replace('ROOT='+repr(BASE+'/chemistry-endpoint-recovery-02'),'ROOT='+repr(ROOT))
        m.HEADER=m.HEADER.replace('SOURCE='+repr(PARENT),'SOURCE='+repr(SOURCE))
        if a.action=='observe':m.observe(a.label)
        else:
            assert a.label and __import__('re').fullmatch('[a-z0-9-]{1,60}',a.label)
            expected=json.loads((RUN/'preparation.json').read_bytes())['manifest_sha256']
            code=m.HEADER+'EXPECTED_PIN='+repr(expected)+'\n'+r'''
import subprocess
value={'status':'read_only_unknown_stage_inspection','root_exists':root.exists(),'source_exists':source.exists(),
 'files':{},'jobs':{},'source_verified':False}
if source.exists():
 manifest=json.loads((source/'source_manifest.json').read_bytes()) if (source/'source_manifest.json').is_file() else None
 if manifest:
  verify(source,EXPECTED_PIN,12);value['source_verified']=True
  value['source_manifest_sha256']=sha(source/'source_manifest.json')
if root.exists():
 for p in root.rglob('*'):
  assert not p.is_symlink() and (p.is_dir() or p.is_file())
  if p.is_file() and p.suffix in ('.json','.xml','.log') and p.stat().st_size<2000000:
   value['files'][str(p)]={'binding':binding(p),'base64':base64.b64encode(p.read_bytes()).decode()}
 if (root/'jobs.json').is_file():value['jobs']=json.loads((root/'jobs.json').read_bytes())
if value['jobs']:
 r=subprocess.run(['bjobs','-l',*value['jobs'].values()],capture_output=True,text=True,timeout=45)
 value['scheduler']={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(value))
'''
            value=m.connect(code,a.label,180)
            for r in value['files'].values():
                b=base64.b64decode(r['base64'],validate=True)
                assert len(b)==r['binding']['bytes'] and sha(b)==r['binding']['sha256']
            print(json.dumps({k:v for k,v in value.items() if k not in ('files','scheduler')},indent=2))


if __name__=='__main__':main()
