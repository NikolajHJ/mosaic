"""Prepare and submit one owned saved-point audit through the DTU connector."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
HELPER = r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
PAIRED_KINDS = (
    'projection-pilot', 'binding-forward-pilot', 'projection-feedback-pilot',
    'projection-feedback-confirmation', 'local-validity-margin',
)

def sha(data): return hashlib.sha256(data).hexdigest()
def dump(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

STAGE = '''#!/bin/bash
set -euo pipefail
source_dir=$1
module load python3/3.12.11 cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
unset PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
export JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true'
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
original=$($python -c 'import json,sys;print(json.load(open(sys.argv[1]))["science_source"])' "$source_dir/plan.json")
export PYTHONPATH="$source_dir:$original/scripts"
source "$original/scripts/hpc/boltz_node_tmpdir.sh"
cd "$source_dir"
cap_mib=$($python -c 'import json,sys;print(json.load(open(sys.argv[1]))["complete_output_cap_mib"])' "$source_dir/plan.json")
file_cap=$($python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("file_cap",5000))' "$source_dir/plan.json")
entrypoint=$($python -c 'import json,sys;print(json.load(open(sys.argv[1]))["entrypoint"])' "$source_dir/plan.json")
gpu_profile=$($python -c 'import json,sys;print(json.load(open(sys.argv[1])).get("gpu_profile", ""))' "$source_dir/plan.json")
if [[ -n "$gpu_profile" ]]; then export MOSAIC_GPU_PROFILE="$gpu_profile"; fi
$python -B storage_accounting.py --output-root "$(dirname "$source_dir")" --cap-mib "$cap_mib" --file-cap "$file_cap" > "$(dirname "$source_dir")/capacity-entry.json"
exec "$python" -B -u "$entrypoint" --plan "$source_dir/plan.json"
'''

REMOTE = r'''
import datetime,hashlib,json,os,re,subprocess,tarfile,time
from pathlib import Path
root=Path(ROOT); source=root/'source'; archive=root/'source.tar.gz'
def write(path,data):
 tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert digest(archive)==ARCHIVE_SHA,'Archive bytes differ'
assert not source.exists(),'Source already staged; reconcile, never replay'
with tarfile.open(archive,'r:gz') as tf:
 members=tf.getmembers()
 assert all(m.isfile() and '/' not in m.name and '\\' not in m.name and m.name not in ('.','..') for m in members),'Unsafe member'
 assert len({m.name for m in members})==len(members),'Duplicate member'
 source.mkdir()
 for m in members:
  (source/m.name).write_bytes(tf.extractfile(m).read())
manifest=json.loads((source/'checksums.json').read_text())
for name,record in manifest['files'].items():
 path=source/name;assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],name
for p in source.iterdir():p.chmod(0o444)
plan=json.loads((source/'plan.json').read_text())
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
subprocess.run(['bash','-n',str(source/'stage.sh')],check=True,timeout=10)
capacity=subprocess.run([python,'-B',str(source/'storage_accounting.py'),'--output-root',str(root),'--cap-mib',str(plan['complete_output_cap_mib']),'--file-cap',str(plan.get('file_cap',5000))],capture_output=True,text=True,timeout=30)
write(root/'capacity-dispatch-raw.json',dict(returncode=capacity.returncode,stdout=capacity.stdout,stderr=capacity.stderr))
assert capacity.returncode==0,'Own bounded storage admission failed'
startup=subprocess.run([python,'-B',str(source/plan['entrypoint']),'--plan',str(source/'plan.json'),'--startup-only'],capture_output=True,text=True,timeout=45)
write(root/'startup.json',dict(returncode=startup.returncode,stdout=startup.stdout,stderr=startup.stderr))
assert startup.returncode==0,'Actual metadata startup failed'
assert not (root/'submission.json').exists() and not (root/'submission-intent.json').exists(),'Existing/unknown submission'
name='mosaic-'+TAG
wall=plan['wall_minutes'];wall_text='%02d:%02d'%(wall//60,wall%60)
argv=['bsub','-H','-q',plan['queue'],'-J',name,'-n',str(plan['cpu_slots']),'-R','span[hosts=1]','-R','rusage[mem=%d]'%plan['memory_mib_per_slot'],'-W',wall_text,'-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE','-o',str(root/'stdout.%J.log'),'-e',str(root/'stderr.%J.log'),'-gpu',plan.get('gpu_request','num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G'),'bash','-l',str(source/'stage.sh'),str(source)]
with (root/'submission-intent.json').open('x') as f:json.dump(dict(argv=argv,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
try:r=subprocess.run(argv,capture_output=True,text=True,timeout=90)
except subprocess.TimeoutExpired as e:
 write(root/'submission-unknown.json',dict(error=str(e),stdout=str(e.stdout),stderr=str(e.stderr)));raise
write(root/'submission-response.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
matches=re.findall(r'Job <(\d+)> is submitted',r.stdout)
assert r.returncode==0 and len(matches)==1,'Submission rejected/unknown; reconcile'
job=matches[0]
write(root/'submission.json',dict(job_id=job,argv=argv,name=name,queue=plan['queue'],status='accepted_held',source_sha256=digest(source/'checksums.json'),plan_sha256=digest(source/'plan.json'),response=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
observations=[]
for attempt in range(12):
 q=subprocess.run(['bjobs','-a','-noheader','-o','jobid stat user job_name:100',job],capture_output=True,text=True,timeout=15)
 observations.append(dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr))
 fields=q.stdout.strip().split()
 assert q.returncode==0 and len(fields)==4 and fields[0]==job and fields[2]=='s234818' and fields[3]==name,'Owned scheduler row differs'
 if fields[1]=='PSUSP':break
 assert fields[1]=='PEND','Unheld/ambiguous job; do not release'
 time.sleep(5)
else:raise RuntimeError('Held state not established; no release')
write(root/'held-proof.json',dict(job_id=job,observations=observations))
r=subprocess.run(['bresume',job],capture_output=True,text=True,timeout=30)
write(root/'release.json',dict(job_id=job,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
assert r.returncode==0,'Release outcome needs reconciliation'
q=subprocess.run(['bjobs','-a','-w',job],capture_output=True,text=True,timeout=20)
write(root/'dispatch-complete.json',dict(job_id=job,scheduler=dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr),version=plan['version'],requested_updates=plan.get('updates',0),points=len(plan.get('points',[]))))
print(json.dumps(dict(root=str(root),job_id=job,version=plan['version'],scheduler=q.stdout,source_sha256=digest(source/'checksums.json'),plan_sha256=digest(source/'plan.json'))))
'''

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--inputs',type=Path,required=True)
 parser.add_argument('--tag',required=True)
 parser.add_argument('--kind', choices=('score-transfer', 'sample-candidate-audit', *PAIRED_KINDS), default='score-transfer')
 parser.add_argument('--candidate-inputs',type=Path)
 parser.add_argument('--confirmation-pilot', type=Path, help='Verified terminal pilot result binding')
 args=parser.parse_args()
 assert all(c.isalnum() or c=='-' for c in args.tag),'Unsafe tag'
 local=REPO/'results/hpc'/args.tag;local.mkdir(parents=True,exist_ok=False)
 source=local/'source';source.mkdir()
 remote='/work3/s234818/mosaic/runs/'+args.tag
 data=json.loads(args.inputs.read_text(encoding='utf-8'));base=data['arms'][0]
 assert all(a['row']['frozen_identity']==base['row']['frozen_identity'] for a in data['arms']),'Contexts differ'
 points=[]
 for a in data['arms']:
  for p in a['files']:
   assert 'sha256' in p['snapshot'] and 'sha256' in p['reference'],'Missing saved inputs'
   points.append(dict(id=a['arm']+'-'+str(p['step']),arm=a['arm'],**p,temperatures=[.4,.7,1.] if p['step'] in (0,50000) else []))
 source_files=[]
 for a in data['arms']:
  source_files.extend([a['plan'],a['row']['config'],a['row']['snapshot'],a['row']['reference']])
 plan=dict(version='protenix-score-transfer-audit-v1',source_root=base['code_source_root'],routing_source=base['routing_source'],science_source=base['row']['science_source']['path'],model_cache='/work3/s234818/mosaic/cache/mosaic/protenix',context_ready=base['row']['context_ready'],pocket=base['row']['pocket'],frozen_identity=base['row']['frozen_identity'],points=points,source_files=source_files,output=remote+'/payload',optimizer_updates=0,queue='gpuv100',internal_seconds=2700,output_cap_bytes=128*1024**2,complete_output_cap_mib=256,entrypoint='score_transfer_audit.py',wall_minutes=60,cpu_slots=4,memory_mib_per_slot=8192)
 if args.kind=='sample-candidate-audit':
  assert args.candidate_inputs is not None,'Declare saved sample bindings'
  candidates=json.loads(args.candidate_inputs.read_text(encoding='utf-8'))
  baseline=next(p for p in points if p['step']==30000 and p['arm']==base['arm'])
  plan.update(points=[baseline],candidate_samples=candidates['groups'],
              maximum_unique_samples=candidates['maximum_unique_samples'],wall_minutes=20,internal_seconds=900)
 if args.kind in PAIRED_KINDS:
  parent_step = 6800 if args.kind == 'local-validity-margin' else 30000
  parent=next(p for p in base['files'] if p['step']==parent_step)
  plan.update(version='protenix-projection-paired-pilot-v1',parent_plan=base['plan'],parent_arm=base['arm'],parent_snapshot=parent['snapshot'],parent_reference=parent['reference'],parent_step=parent_step,updates=2000,temperature=.4,chemistry_coefficient=30.,learning_rate=.3,smoothing=.02,internal_seconds=4800,output_cap_bytes=512*1024**2,complete_output_cap_mib=512,entrypoint='projected_start_pilot.py',wall_minutes=90,cpu_slots=8,memory_mib_per_slot=4096)
  plan.pop('points');plan.pop('optimizer_updates')
  storage_bytes=(REPO/'scripts/hpc/storage_accounting.py').read_bytes()
  plan['storage_module']=dict(path=remote+'/source/storage_accounting.py',bytes=len(storage_bytes),sha256=sha(storage_bytes))
  plan['file_cap']=5000
  if args.kind=='local-validity-margin':
   assert base['arm']=='structural10-chain-s17' and parent['route']==0,'Require the valid chain checkpoint'
   import sys
   sys.path.insert(0,str(REPO/'scripts/hpc'))
   from gpu_policy import profile
   placement=profile('a40-48gb-v1')
   plan.update(comparison='local_validity_margin',updates=1000,temperature=1.,chemistry_coefficient=10.,
               smoothing=0.,wall_minutes=60,internal_seconds=3300,complete_output_cap_mib=256,
               output_cap_bytes=252*1024**2,file_cap=4000,
               queue=placement['queue'],gpu_profile=placement['id'],gpu_request=placement['resource'])
   for key,name in [('screen_source','protenix_chemistry_screen.py'),
                    ('margin_source','local_validity_margin.py'),
                    ('margin_controls','local_validity_margin_controls.py')]:
    data=(HERE.parent/name).read_bytes()
    (source/name).write_bytes(data)
    plan[key]=dict(path=remote+'/source/'+name,bytes=len(data),sha256=sha(data))
  if args.kind=='binding-forward-pilot':plan['comparison']='binding_forward'
  if args.kind=='projection-feedback-pilot':plan.update(comparison='projection_feedback',block_updates=500)
  if args.kind=='projection-feedback-confirmation':
   assert args.confirmation_pilot is not None, 'Bind the completed pilot before confirmation'
   plan.update(comparison='projection_feedback', block_updates=500, updates=20000,
               confirmation_pilot=json.loads(args.confirmation_pilot.read_text(encoding='utf-8')),
               wall_minutes=480, internal_seconds=28200, complete_output_cap_mib=1024,
               output_cap_bytes=1016*1024**2, file_cap=8000)
 dump(source/'plan.json',plan)
 for src,dst in [(HERE.parent/'score_transfer_audit.py','score_transfer_audit.py'),(REPO/'scripts/hpc/storage_accounting.py','storage_accounting.py')]:
  (source/dst).write_bytes(src.read_bytes())
 if args.kind in PAIRED_KINDS:(source/'projected_start_pilot.py').write_bytes((HERE.parent/'projected_start_pilot.py').read_bytes())
 (source/'stage.sh').write_text(STAGE,encoding='utf-8',newline='\n')
 manifest={'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in source.iterdir()}}
 dump(source/'checksums.json',manifest)
 archive=local/'source.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in sorted(source.iterdir()):tf.add(p,arcname=p.name)
 packet=dict(remote=remote,kind=args.kind,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive.read_bytes()),source_manifest_sha256=sha((source/'checksums.json').read_bytes()),plan_sha256=sha((source/'plan.json').read_bytes()),points=len(points) if args.kind=='score-transfer' else None,updates_per_arm=plan.get('updates'))
 dump(local/'prepared.json',packet)
 # Claim a fresh remote owner before upload. A failed later step is never replayed.
 claim=subprocess.run([HELPER,'exec','--timeout','30','mkdir '+remote],capture_output=True,text=True,timeout=40)
 dump(local/'claim.json',dict(returncode=claim.returncode,stdout=claim.stdout,stderr=claim.stderr))
 assert claim.returncode==0,'Remote owner exists or unavailable; reconcile'
 upload=subprocess.run([HELPER,'upload','--timeout','90',str(archive),remote+'/source.tar.gz'],capture_output=True,text=True,timeout=105)
 dump(local/'upload.json',dict(returncode=upload.returncode,stdout=upload.stdout,stderr=upload.stderr))
 assert upload.returncode==0,'Transfer did not complete; inspect fresh owner'
 program='ROOT='+repr(remote)+'\nARCHIVE_SHA='+repr(packet['archive_sha256'])+'\nTAG='+repr(args.tag)+'\n'+REMOTE
 (local/'stage-submit.py').write_text(program,encoding='utf-8',newline='\n')
 result=subprocess.run([HELPER,'exec','--timeout','240',"bash -lc 'python3 -B -'"],input=program,capture_output=True,text=True,timeout=255)
 (local/'stdout.json').write_text(result.stdout,encoding='utf-8');(local/'stderr.txt').write_text(result.stderr,encoding='utf-8')
 dump(local/'transport.json',dict(returncode=result.returncode))
 if result.returncode:raise RuntimeError('Staging/submission outcome requires inspection: '+result.stderr[-1500:])
 actual=json.loads(result.stdout);dump(local/'dispatch.json',actual)
 print(json.dumps(actual))

if __name__=='__main__':main()
