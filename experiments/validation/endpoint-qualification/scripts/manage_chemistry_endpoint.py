"""Stdlib preparation, one-shot submission and observation of chemistry diagnostics."""
import argparse
import ast
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
TAG='20261027-chemistry-endpoint-recovery-02'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/chemistry-endpoint-recovery-02'
SOURCE=BASE+'/chemistry-endpoint-recovery-release-02/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def raw(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def connect(code,label,timeout=240):
    ast.parse(code);out=LOCAL/label;out.mkdir(parents=True,exist_ok=False)
    (out/'program.py').write_text(code,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote(PYTHON+' -')]
    receipt={'status':'unknown_inspect_before_recovery','started_utc':datetime.now(timezone.utc).isoformat(),
             'program_sha256':digest(code.encode())}
    try:
        with (out/'stdout.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            run=subprocess.run(argv,input=code.encode(),stdout=stdout,stderr=stderr,timeout=timeout)
        receipt['returncode']=run.returncode
        if run.returncode:raise RuntimeError((out/'stderr.txt').read_text(errors='replace')[-3500:])
        value=json.loads((out/'stdout.json').read_bytes());receipt['status']=value.get('status','received');return value
    finally:
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        receipt['output_sha256']=digest((out/'stdout.json').read_bytes())
        (out/'receipt.json').write_bytes(raw(receipt))


HEADER=f'BASE={BASE!r}\nROOT={ROOT!r}\nSOURCE={SOURCE!r}\n'+r'''
import base64,hashlib,json,os,re,stat,sys
from pathlib import Path
base=Path(BASE);root=Path(ROOT);source=Path(SOURCE)
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def binding(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and all(not v.is_symlink() for v in p.parents)
 return {'path':str(p),'bytes':s.st_size,'sha256':sha(p)}
def verify(p,pin,count=None):
 assert sha(p/'source_manifest.json')==pin
 m=json.loads((p/'source_manifest.json').read_bytes());assert count is None or len(m['files'])==count
 assert {q.relative_to(p).as_posix() for q in p.rglob('*') if q.is_file() or q.is_symlink()}==set(m['files'])|{'source_manifest.json'}
 for rel,digest in m['files'].items():assert binding(p/rel)['sha256']==digest
 return m
'''


def prepare():
    code=HEADER+r'''
assert not root.exists() and not source.parent.exists(),'Existing/unknown attempt: inspect before preparation'
science={
 'protenix':('protenix-extended-release-03/source','da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',643,'examples/configs/extended_random.json'),
 'boltz':('boltz-lr0025-release-03/source','c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6',635,'examples/configs/boltz_serial_boltz2_chemistry.json')}
adapters={
 'protenix':('chemistry-full-model-release-05/source','ab875927a328c34650d2e43baf6dfbe323e8f9bcf5600f39bac78f3ccdfb5d8c'),
 'boltz':('chemistry-full-model-release-02/source','05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9')}
protected={};parents=[];arms={}
def protect(p):
 row=binding(p);protected[row['path']]=row;return row
for name,(rel,pin,count,config) in science.items():
 src=base/rel;verify(src,pin,count)
 parent={'path':str(src),'manifest_sha256':pin,'files':count};parents.append(parent)
 arel,apin=adapters[name];asrc=base/arel;am=verify(asrc,apin)
 adapter={'path':str(asrc),'manifest_sha256':apin,'files':len(am['files'])};parents.append(adapter)
 segment=(base/'chemistry-production-03/protenix/random-segment-0000' if name=='protenix'
          else base/'chemistry-production-04/boltz/boltz2_chemistry-segment-0000')
 failure=(base/'chemistry-production-03/protenix/random-native-0001' if name=='protenix' else segment)
 own=str(segment/'payload/study')
 for p in (segment/'payload/study').rglob('*'):
  assert not p.is_symlink() and (p.is_file() or p.is_dir())
  if p.is_file():protect(p)
 if failure!=segment:
  for p in (failure/'payload/study').rglob('*'):
   assert not p.is_symlink() and (p.is_file() or p.is_dir())
   if p.is_file():protect(p)
 for parent_dir in (segment,failure):
  for p in parent_dir.glob('*.json'):protect(p)
  for p in parent_dir.glob('*.log'):protect(p)
  for p in (parent_dir/'payload').glob('*.json'):protect(p)
 sub=json.loads((failure/'submission.json').read_bytes())
 assert sub['job_id']==('29821558' if name=='protenix' else '29825209')
 text=(failure/('stdout.'+sub['job_id']+'.log')).read_text();compact=''.join(text.split())
 assert compact.count('Subject:Job'+sub['job_id']+':')==1 and 'Exitedwithexitcode1.' in compact
 args=json.loads((failure/'payload/driver-invocation.json').read_bytes())['args']
 def option(key):return args[args.index(key)+1]
 for key in ('--reference','--pocket','--cpu-certificate'):protect(option(key))
 endpoint=segment/'payload/study/case-0/snapshots/step-10000.npz'
 baseline=failure/'payload/study/early_pilot/snapshots/step-0000.npz' if name=='protenix' else endpoint
 report=failure/'payload/study/early_pilot/combined_initial_checks/report.json' if name=='protenix' else failure/'payload/study/terminal_gradient_checks/report.json'
 arm={'id':name,'science_source':parent,'adapter_source':adapter,'config':protect(src/config),
   'boundary':protect(endpoint),'baseline':protect(baseline),'failed_report':protect(report),
   'reference':protect(option('--reference')),'pocket':protect(option('--pocket')),
   'original_failure_job':sub['job_id'],'target_updates':50000 if name=='protenix' else 30000,
   'learning_rate':.3 if name=='protenix' else .0025,'saved_probes':{},
   'frozen_identity':{'model_digest':'8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2' if name=='protenix' else '395876da13ab6d5d175e6603db2118a82954c7c06484842fe2314dbd77a31a16',
      'context_digest':'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4' if name=='protenix' else 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'}}
 if name=='protenix':
  arm['preparation']=protect(failure/'payload/study/early_pilot/references/step-0000.json')
  for p in report.parent.glob('atoms-*.npz'):arm['saved_probes'][p.name]=protect(p)
  assert len(arm['saved_probes'])==6
  ready=base/'restart-lrreset-04/validate-protenix/payload/study/assessment.json';arm['context_ready']=protect(ready)
  old=json.loads(ready.read_bytes())
  assert arm['context_ready']['sha256']=='9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
  for n,pin in old['context_files'].items():assert protect(ready.parent/'context'/n)['sha256']==pin
 else:
  arm['context']=protect(Path(option('--prepared'))/'context.pt');arm['worker_python']=option('--worker-python')
  assert arm['context']['sha256']=='a18a331269170aac3281b0a2ecad313aab81756c9b77e048d7e2cbaa4d8f1093'
 arms[name]=arm
parent=base/'nonchem-endpoint-release-09/source';m=verify(parent,'f15443b51c5197b95205bf59c90e05192f9ceabb955e6006cccfbbdff257fc9b',19)
parents.append({'path':str(parent),'manifest_sha256':sha(parent/'source_manifest.json'),'files':19})
# Transport the already qualified production compiler environment, not a newly
# selected numerical setting. Authentication is repeated by allocated startup.
prod=base/'chemistry-production-release-03/source';pm=verify(prod,'c73796b041d8432ef1634e8c56920006170a094d3bdc74867c08f3b5415126d7',14)
sys.path.insert(0,str(prod/'scripts'))
import production_ops as po,production_context as pc
context=pc.read_context(po,'protenix',json.loads((prod/'plan.json').read_bytes()))
assert context['environment']=={'XLA_FLAGS':'--xla_gpu_autotune_level=0'}
parents.append({'path':str(prod),'manifest_sha256':sha(prod/'source_manifest.json'),'files':14})
def visit(v):
 if isinstance(v,dict):
  if set(v)=={'path','bytes','sha256'}:assert protect(v['path'])==v
  else:
   for x in v.values():visit(x)
 elif isinstance(v,list):
  for x in v:visit(x)
visit(context)
failed_root=base/'chemistry-endpoint-recovery-01';failed_source=base/'chemistry-endpoint-recovery-release-01/source'
old_manifest=verify(failed_source,'c94314f0370e792f26ebd3958bc08b398b39cb525a995dcd65b063a74d120223',10)
assert json.loads((failed_root/'jobs.json').read_bytes())=={'cpu':'29830639','after-cpu':'29830640'},'Existing/unknown diagnostic GPU prevents retry'
import xml.etree.ElementTree as ET
cases=list(ET.parse(failed_root/'cpu/payload/tests.xml').iter('testcase'))
assert len(cases)==32 and all(not list(x) for x in cases)
assert {x.get('classname') for x in cases}=={'tests.test_chemistry_endpoint_diagnostic.ChemistryEndpointDiagnosticTests','tests.test_chemistry_endpoint_ops.ChemistryEndpointOpsTests'}
for phase,job in [('cpu','29830639'),('after-cpu','29830640')]:
 parent=failed_root/phase;text=(parent/('stdout.'+job+'.log')).read_text();compact=''.join(text.split())
 assert compact.count('Subject:Job'+job+':')==1 and 'Exitedwithexitcode1.' in compact,'Original controller not actually failed yet'
 for p in parent.glob('*.json'):protect(p)
 for p in parent.glob('*.log'):protect(p)
for p in (failed_root/'cpu/payload').rglob('*'):
 if p.is_file():protect(p)
assert not (failed_root/'cpu/payload/saved').exists(),'Old saved analysis outcome differs'
parents.append({'path':str(failed_source),'manifest_sha256':sha(failed_source/'source_manifest.json'),'files':10})
plan={'version':'chemistry-endpoint-diagnostic-v1','optimizer_updates':0,'can_certify_or_restart_original_study':False,
 'epsilons':[.01,.003,.001,.0003,.0001],'arms':arms,'protected_inputs':list(protected.values()),
 'parent_sources':parents,'qualified_production_context':context,'output_root':str(root),'max_native_bytes':256*1024**2,
 'internal_seconds':5400,'monitor_remains_paused':True}
print(json.dumps({'status':'read_only_chemistry_inputs_authenticated','plan':plan}))
'''
    packet=connect(code,'input02',300);plan=packet['plan'];members={};counts={}
    for rel in ('scripts/chemistry_endpoint_diagnostic.py','scripts/chemistry_endpoint_ops.py',
                'scripts/chemistry_endpoint_stage.lsf','tests/test_chemistry_endpoint_diagnostic.py','tests/test_chemistry_endpoint_ops.py'):
        content=(OWNER/rel).read_bytes().replace(b'\r\n',b'\n')
        if rel.endswith('.py'):
            tree=ast.parse(content,filename=rel)
            if rel.startswith('tests/'):
                for node in tree.body:
                    if isinstance(node,ast.ClassDef):
                        n=sum(isinstance(x,ast.FunctionDef) and x.name.startswith('test_') for x in node.body)
                        if n:counts['tests.'+Path(rel).stem+'.'+node.name]=n
        members[rel]=content
    members['README.md']=b'# Separate chemistry endpoint diagnostics\n\nNo optimizer dispatch or original recertification. All numerical work allocated DTU HPC.\n'
    members['scripts/README.md']=b'# Diagnostic execution\n\nFixed saved inputs, original model loaders, metadata-only controller and authenticated shell entry.\n'
    members['tests/README.md']=b'# Allocated controls\n\nPositive/negative decomposition, ownership, actual-command and output/isolation controls.\n'
    members['protocol.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n')
    members['plan.json']=raw(plan)
    manifest={'version':'chemistry-endpoint-diagnostic-source-v1','files':{k:digest(v) for k,v in sorted(members.items())},
              'parent_sources':plan['parent_sources'],'cpu_counts':counts,'diagnostic_only':True,
              'optimizer_updates':0,'original_results_changed':False}
    members['source_manifest.json']=raw(manifest)
    out=LOCAL/'source01';out.mkdir(exist_ok=False);buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for rel,content in sorted(members.items()):
            path=out/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
            info=tarfile.TarInfo(rel);info.size=len(content);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(content))
    archive=buf.getvalue();(out/'source.tar.gz').write_bytes(archive)
    prep={'status':'prepared_not_validated','files':len(manifest['files']),'manifest_sha256':digest(members['source_manifest.json']),
          'archive_bytes':len(archive),'archive_sha256':digest(archive),'cpu_counts':counts,'cpu_cases':sum(counts.values()),
          'protected_input_files':len(plan['protected_inputs']),'plan_sha256':digest(members['plan.json'])}
    (out/'preparation.json').write_bytes(raw(prep));(RUN/'preparation.json').write_bytes(raw(prep));(RUN/'source-manifest.json').write_bytes(members['source_manifest.json'])
    (RUN/'campaign.json').write_bytes(raw({**prep,'settings_changed':False,'saved_results_changed':False,'optimizer_jobs_submitted':False,'monitor_resumed':False}))
    print(json.dumps(prep,indent=2))


def stage():
    local=LOCAL/'source01';prep=json.loads((local/'preparation.json').read_bytes());data=(local/'source.tar.gz').read_bytes()
    assert len(data)==prep['archive_bytes'] and digest(data)==prep['archive_sha256']
    code=HEADER+'PREP='+repr(prep)+'\nARCHIVE='+repr(base64.b64encode(data).decode())+'\n'+r'''
import io,subprocess,tarfile
assert not root.exists() and not source.parent.exists(),'Existing/unknown attempt; never bootstrap again'
data=base64.b64decode(ARCHIVE,validate=True);assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for item in tar:
  part=Path(item.name);assert item.isfile() and not part.is_absolute() and '..' not in part.parts and item.name not in members
  members[item.name]=tar.extractfile(item).read();assert len(members[item.name])==item.size
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==PREP['manifest_sha256']
assert len(m['files'])==PREP['files'] and set(members)==set(m['files'])|{'source_manifest.json'}
for name,pin in m['files'].items():assert hashlib.sha256(members[name]).hexdigest()==pin
for row in m['parent_sources']:verify(Path(row['path']),row['manifest_sha256'],row['files'])
for row in json.loads(members['plan.json'])['protected_inputs']:assert binding(row['path'])==row
root.mkdir();source.mkdir(parents=True)
for name,content in members.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555)
subprocess.run(['bash','-n',str(source/'scripts/chemistry_endpoint_stage.lsf')],check=True)
sys.path.insert(0,str(source/'scripts'));import chemistry_endpoint_ops as op
op.external(op.verify(source,PREP['manifest_sha256']))
preflight=root/'preflight';preflight.mkdir();gate=op.storage(preflight,'preflight')
cpu=op.submit('cpu',PREP['manifest_sha256']);callback=op.submit('after-cpu',PREP['manifest_sha256'],'ended('+cpu['job_id']+')')
print(json.dumps({'status':'allocated_CPU_and_own_callback_submitted_not_passed','cpu':cpu,'callback':callback,'whole_storage_gate':gate}))
'''
    value=connect(code,'stage01',420);(RUN/'submission.json').write_bytes(raw(value))
    campaign=json.loads((RUN/'campaign.json').read_bytes());campaign.update(status=value['status'],
        cpu_job=value['cpu']['job_id'],callback_job=value['callback']['job_id'],remote_source_staged=True)
    (RUN/'campaign.json').write_bytes(raw(campaign))
    print(json.dumps({'status':value['status'],'cpu':value['cpu']['job_id'],'callback':value['callback']['job_id']},indent=2))


def observe(label):
    code=HEADER+r'''
import subprocess
record={'status':'read_only_observation','files':{},'jobs':json.loads((root/'jobs.json').read_bytes())}
for phase,job in record['jobs'].items():
 parent=root/phase
 paths=list(parent.glob('*.json'))+list(parent.glob('*.log'))+list((parent/'payload').rglob('assessment.json'))+list((parent/'payload').rglob('startup.json'))+list((parent/'payload').rglob('frozen-*.json'))+list((parent/'payload').glob('*.json'))+list((parent/'payload').glob('*.xml'))+list((parent/'payload').glob('*.log'))
 for p in paths:
  if p.stat().st_size<=2000000:record['files'][str(p)]={'binding':binding(p),'base64':base64.b64encode(p.read_bytes()).decode()}
for p in root.glob('*.json'):record['files'][str(p)]={'binding':binding(p),'base64':base64.b64encode(p.read_bytes()).decode()}
q=subprocess.run(['bjobs','-l',*record['jobs'].values()],capture_output=True,text=True,timeout=45)
record['scheduler']={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
print(json.dumps(record))
'''
    value=connect(code,label,180)
    for row in value['files'].values():
        b=base64.b64decode(row['base64'],validate=True)
        assert len(b)==row['binding']['bytes'] and digest(b)==row['binding']['sha256']
    states=re.findall(r'Job<(\d+)>.*?Status<([^>]+)>',''.join(value['scheduler']['stdout'].split()))
    print(json.dumps({'jobs':value['jobs'],'states':states,'files':len(value['files'])},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','stage','observe'))
    p.add_argument('--label');args=p.parse_args()
    assert REPO.name=='mosaic'
    if args.action=='prepare':prepare()
    elif args.action=='stage':stage()
    else:
        assert args.label and re.fullmatch('[a-z0-9-]{1,60}',args.label)
        observe(args.label)


if __name__=='__main__':main()
