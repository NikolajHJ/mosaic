"""Stdlib one-shot capsule preparation/submission and compact own-job evidence."""
import argparse,ast,base64,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
from datetime import datetime,timezone
REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
TAG='20261027-saved-gradient-forensics-01'
LOCAL=REPO/'results/hpc'/TAG
RUN=OWNER/'runs'/TAG
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/saved-gradient-forensics-01'
SOURCE=BASE+'/saved-gradient-forensics-release-01/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
PARENT=BASE+'/nonchem-endpoint-release-06/source'
PARENT_PIN='1519d5761069f73fb6659bc1740cb04c155a04bc3218f143a3f28a73e65a1f69'
PRECISION=BASE+'/protenix-endpoint-fix-release-05/source'
PRECISION_PIN='65b4bd350386a8550c3b588eb7de2b0f6084b32ad790a907851c632dba0f023e'
def raw(v):return (json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def connection(code,label,timeout=360):
    ast.parse(code);out=LOCAL/label;out.mkdir(parents=True,exist_ok=False);(out/'program.py').write_text(code,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote(PYTHON+' -')]
    receipt={'status':'unknown_inspect_before_any_recovery','started_utc':datetime.now(timezone.utc).isoformat(),'program_sha256':digest(code.encode())}
    try:
        with (out/'stdout.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:r=subprocess.run(argv,input=code.encode(),stdout=output,stderr=error,timeout=timeout)
        receipt['returncode']=r.returncode
        if r.returncode:raise RuntimeError((out/'stderr.txt').read_text(errors='replace')[-3500:])
        value=json.loads((out/'stdout.json').read_bytes());receipt['status']=value.get('status','metadata_received');return value
    finally:
        receipt['output_sha256']=digest((out/'stdout.json').read_bytes());receipt['ended_utc']=datetime.now(timezone.utc).isoformat();(out/'receipt.json').write_bytes(raw(receipt))

HEADER=f"ROOT={ROOT!r}\nSOURCE={SOURCE!r}\nPARENT={PARENT!r}\nPARENT_PIN={PARENT_PIN!r}\nPRECISION={PRECISION!r}\nPRECISION_PIN={PRECISION_PIN!r}\n"+r'''
import base64,hashlib,json,os,stat,sys
from pathlib import Path
root=Path(ROOT);source=Path(SOURCE);parent=Path(PARENT);precision=Path(PRECISION)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def binding(p):
 p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and all(not q.is_symlink() for q in (p,*p.parents))
 return {'path':str(p),'bytes':s.st_size,'sha256':sha(p)}
def verify(p,pin):
 assert sha(p/'source_manifest.json')==pin
 m=json.loads((p/'source_manifest.json').read_bytes())
 for rel,digest in m['files'].items():assert binding(p/rel)['sha256']==digest
 return m
assert len(verify(parent,PARENT_PIN)['files'])==16 and len(verify(precision,PRECISION_PIN)['files'])==16
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1'
'''

def prepare():
    known=json.loads((OWNER/'runs/20261026-endpoint-investigation-07/observed-failures-02.json').read_bytes())['evidence']
    bindings=[known[k]['binding'] for k in ('123','125','126','127','128')]
    code=HEADER+'BINDINGS='+repr(bindings)+'\n'+r'''
assert not root.exists() and not source.parent.exists(),'Existing/unknown outcome; inspect rather than replay'
for b in BINDINGS:assert binding(b['path'])==b
report=json.loads(Path(BINDINGS[0]['path']).read_bytes());assert report['status']=='failed' and report['repair_weight']==0
probes={};inputs=list(BINDINGS)
for d in report['directions']:
 for q in d['probes']:
  for sign in (-1,1):
   name=f"{d['field']}-{q['epsilon']:g}-{sign:+d}.npz";p=Path(BINDINGS[0]['path']).parent/name
   b=binding(p);assert b['bytes']<2*1024**2;inputs.append(b);probes[name]=str(p)
assert len(probes)==12
names=('dbmol_total_loss','dbmol_affinity_loss','dbmol_confidence_loss','dbmol_contact_loss')
plan={'version':'saved-gradient-forensics-v1','inputs':inputs,'boltz_report':BINDINGS[0]['path'],'boltz_probes':probes,
 'dbmol_reports':{n:b['path'] for n,b in zip(names,BINDINGS[1:])},'output':str(root/'cpu/payload/analysis'),
 'startup_output':str(root/'cpu/payload/startup'),'original_results_changed':False,'optimizer_updates':0}
print(json.dumps(plan))
'''
    plan=connection(code,'input-metadata01',180);members={}
    for rel in ('scripts/saved_gradient_forensics.py','tests/test_saved_gradient_forensics.py'):
        b=(OWNER/rel).read_bytes().replace(b'\r\n',b'\n');ast.parse(b);members[rel]=b
    members['protocol.md']=(RUN/'README.md').read_bytes().replace(b'\r\n',b'\n');members['plan.json']=raw(plan)
    stage=r'''#!/bin/bash
set -euo pipefail
[[ $# == 2 && -n "${LSB_JOBID:-}" && "$LSB_QUEUE" == hpc && "$LSB_DJOB_NUMPROC" == 4 ]]
source_root=$1
pin=$2
module load python3/3.12.11
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''
"$python" - "$source_root" "$pin" <<'RUN'
import hashlib,json,os,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
source=Path(sys.argv[1]);pin=sys.argv[2];root=source.parent.parent/'UNUSED'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(source/'source_manifest.json')==pin
m=json.loads((source/'source_manifest.json').read_bytes());assert len(m['files'])==5
for name,digest in m['files'].items():assert not (source/name).is_symlink() and sha(source/name)==digest
assert {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}==set(m['files'])|{'source_manifest.json'}
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');root=base/'saved-gradient-forensics-01';parent=root/'cpu'
sub=json.loads((parent/'submission.json').read_bytes());assert sub['job_id']==os.environ['LSB_JOBID'] and sub['source_manifest_sha256']==pin and sub['status']=='submitted'
assert sub['argv']==['bsub','-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-saved-gradient-forensics01','-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log'),'bash','-l',str(source/'stage.lsf'),str(source),pin]
precision=base/'protenix-endpoint-fix-release-05/source';ppin='65b4bd350386a8550c3b588eb7de2b0f6084b32ad790a907851c632dba0f023e'
assert sha(precision/'source_manifest.json')==ppin
pm=json.loads((precision/'source_manifest.json').read_bytes())
for name,digest in pm['files'].items():assert sha(precision/name)==digest
sys.dont_write_bytecode=True;sys.path.insert(0,str(precision/'scripts'));import protenix_precision_ops as ops
ops.verify(precision,ppin);ops.external(pm,ops.read(precision/'plan.json'));ops.storage(parent,'entry')
old=base/'nonchem-endpoint-release-06/source';oldpin='1519d5761069f73fb6659bc1740cb04c155a04bc3218f143a3f28a73e65a1f69'
om=ops.verify(old,oldpin);control=ops.module(old/'scripts/continuation_ops.py','_original_control_for_accounting')
assert control.metadata_bytes()+64*1024**2<=512*1024**2,'Combined existing/new diagnostic/control cap exceeded'
payload=parent/'payload';payload.mkdir(exist_ok=False);deadline=time.monotonic()+3000
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
command=[python,str(source/'scripts/saved_gradient_forensics.py'),'--plan',str(source/'plan.json'),'--plan-sha256',sha(source/'plan.json'),'--output',str(payload/'startup'),'--startup-only']
with (payload/'startup-invocation.json').open('x') as f:json.dump({'argv':command},f)
subprocess.run(command,check=True,timeout=180)
command=[python,'-m','pytest',str(source/'tests'),'-q','-o','addopts=','-p','no:cacheprovider','--junitxml='+str(payload/'tests.xml')]
with (payload/'tests.log').open('x') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=180)
cases=list(ET.parse(payload/'tests.xml').getroot().iter('testcase'));assert len(cases)==8 and all(not any(c.tag in ('failure','error','skipped') for c in case) for case in cases)
command=[python,str(source/'scripts/saved_gradient_forensics.py'),'--plan',str(source/'plan.json'),'--plan-sha256',sha(source/'plan.json'),'--output',str(payload/'analysis')]
with (payload/'analysis-invocation.json').open('x') as f:json.dump({'argv':command},f)
try:
 with (payload/'analysis.log').open('x') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.monotonic()))
 result=json.loads((payload/'analysis/assessment.json').read_bytes());assert result['status']=='diagnostics_completed_original_failures_preserved' and result['can_certify_or_restart_original_study'] is False
finally:
 for b in json.loads((source/'plan.json').read_bytes())['inputs']:assert Path(b['path']).stat().st_size==b['bytes'] and sha(b['path'])==b['sha256']
 assert sum(p.stat().st_size for tree in (root,source.parent) for p in tree.rglob('*') if p.is_file())<64*1024**2
 with (payload/'preservation.json').open('x') as f:json.dump({'status':'exact_inputs_preserved','source_manifest_sha256':pin,'updates':0,'cases':8},f)
RUN
'''
    members['stage.lsf']=stage.encode();manifest={'version':'saved-gradient-forensics-source-v1','files':{k:digest(v) for k,v in members.items()},'parent_source':{'path':PARENT,'sha256':PARENT_PIN},'precision_guard':{'path':PRECISION,'sha256':PRECISION_PIN},'optimizer_updates':0}
    assert len(members)==5;members['source_manifest.json']=raw(manifest)
    out=LOCAL/'source01';out.mkdir(exist_ok=False)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for name,b in members.items():
            p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);info=tarfile.TarInfo(name);info.size=len(b);info.mode=0o444;info.mtime=0;tar.addfile(info,io.BytesIO(b))
    data=buf.getvalue();(out/'source.tar.gz').write_bytes(data)
    prep={'status':'prepared_not_validated','files':5,'manifest_sha256':digest(members['source_manifest.json']),'archive_bytes':len(data),'archive_sha256':digest(data),'plan_sha256':digest(members['plan.json']),'cpu_cases':8}
    (out/'preparation.json').write_bytes(raw(prep));(RUN/'preparation.json').write_bytes(raw(prep));(RUN/'source-manifest.json').write_bytes(members['source_manifest.json']);print(json.dumps(prep,indent=2))

def stage():
    out=LOCAL/'source01';prep=json.loads((out/'preparation.json').read_bytes());archive=(out/'source.tar.gz').read_bytes();assert len(archive)==prep['archive_bytes'] and digest(archive)==prep['archive_sha256']
    code=HEADER+'PREP='+repr(prep)+'\nARCHIVE='+repr(base64.b64encode(archive).decode())+'\n'+r'''
import io,tarfile,subprocess,re
assert not root.exists() and not source.parent.exists(),'Existing/unknown stage; never replay'
data=base64.b64decode(ARCHIVE,validate=True);assert len(data)==PREP['archive_bytes'] and hashlib.sha256(data).hexdigest()==PREP['archive_sha256']
members={}
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 for item in tar:
  p=Path(item.name);assert item.isfile() and not p.is_absolute() and '..' not in p.parts and item.name not in members
  members[item.name]=tar.extractfile(item).read();assert len(members[item.name])==item.size
m=json.loads(members['source_manifest.json']);assert hashlib.sha256(members['source_manifest.json']).hexdigest()==PREP['manifest_sha256']
assert set(members)==set(m['files'])|{'source_manifest.json'} and len(m['files'])==5
for k,v in m['files'].items():assert hashlib.sha256(members[k]).hexdigest()==v
for b in json.loads(members['plan.json'])['inputs']:assert binding(b['path'])==b
root.mkdir();source.mkdir(parents=True)
for name,b in members.items():
 p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);p.chmod(0o444)
for p in sorted([p for p in source.rglob('*') if p.is_dir()],reverse=True):p.chmod(0o555)
source.chmod(0o555)
subprocess.run(['bash','-n',str(source/'stage.lsf')],check=True)
sys.path.insert(0,str(precision/'scripts'));import protenix_precision_ops as ops
ops.external(verify(precision,PRECISION_PIN),ops.read(precision/'plan.json'))
preflight=root/'preflight';preflight.mkdir();gate=ops.storage(preflight,'preflight')
parent=root/'cpu';parent.mkdir()
argv=['bsub','-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00','-J','mosaic-saved-gradient-forensics01','-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log'),'bash','-l',str(source/'stage.lsf'),str(source),PREP['manifest_sha256']]
record={'status':'unknown','argv':argv,'source_manifest_sha256':PREP['manifest_sha256']};ops.write(parent/'submission-started.json',record)
try:
 r=subprocess.run(argv,capture_output=True,text=True,timeout=60);record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
 ids=re.findall(r'Job <([1-9][0-9]*)> is submitted',r.stdout)
 if r.returncode==0 and len(ids)==1:record.update(status='submitted',job_id=ids[0])
finally:ops.write(parent/'submission.json',record)
assert record['status']=='submitted','Unknown outcome must be inspected, never duplicated'
ops.write(root/'jobs.json',{'cpu':record['job_id']})
print(json.dumps({'status':'saved_only_forensics_cpu_submitted_not_passed','submission':record,'whole_storage':gate}))
'''
    value=connection(code,'stage01',400);(RUN/'submission.json').write_bytes(raw(value));print(json.dumps({'status':value['status'],'job_id':value['submission']['job_id']},indent=2))

def observe(label):
    assert re.fullmatch('[a-z0-9-]{1,50}',label)
    code=HEADER+r'''
import subprocess
files=[]
for p in (root/'cpu').rglob('*'):
 if p.is_file() and p.stat().st_size<4*1024**2 and p.suffix in ('.json','.xml','.log'):
  b=p.read_bytes();files.append({**binding(p),'text':b.decode()})
sub=json.loads((root/'cpu/submission.json').read_bytes());q=subprocess.run(['bjobs','-l',sub['job_id']],capture_output=True,text=True,timeout=30)
print(json.dumps({'status':'read_only_own_forensic_evidence','files':files,'scheduler':{'job_id':sub['job_id'],'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}}))
'''
    value=connection(code,label,150)
    for row in value['files']:assert len(row['text'].encode())==row['bytes'] and digest(row['text'].encode())==row['sha256']
    print(json.dumps({'status':value['status'],'files':len(value['files']),'scheduler':value['scheduler']},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','stage','observe'));p.add_argument('--label');a=p.parse_args()
    if a.phase=='prepare':prepare()
    elif a.phase=='stage':stage()
    else:observe(a.label)
if __name__=='__main__':main()
