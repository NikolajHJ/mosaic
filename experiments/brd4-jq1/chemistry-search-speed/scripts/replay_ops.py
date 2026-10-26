"""Allocated metadata admission for a bounded, read-only GPU replay diagnostic.

There is deliberately no model-loading, optimizer or production-dispatch branch.
Original failed pilots and saved study bytes remain immutable inputs.
"""
from collections import Counter
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'chemistry-replay-02'
SOURCE=BASE/'chemistry-replay-release-02/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
VERSION='chemistry-softmax-replay-source-v2'
LIMIT=64*1024**2
GROUP_LIMIT=2*1024**3
GUARD=BASE/'protenix-endpoint-fix-release-02/source'
GUARD_PIN='09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b'
PILOTS={
 'protenix':{'root':str(BASE/'chemistry-full-model-03'),'source':str(BASE/'chemistry-full-model-release-03/source'),
  'manifest_sha256':'8480a39c05d612d1a80ea1892d816c9429fcd471e97adb31655f8132d9007f31',
  'cpu_certificate_sha256':'ea6c825017788181a81da3421fc83e16f2d74236914eb588d6b5acb4ce852db7','job_id':'29816284'},
 'boltz':{'root':str(BASE/'chemistry-full-model-02'),'source':str(BASE/'chemistry-full-model-release-02/source'),
  'manifest_sha256':'05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9',
  'cpu_certificate_sha256':'c46f39e679f503720ae31b561b8b3273a1d38c160bf3c7f3d6feba28f2b5c687','job_id':'29816184'},
}


def require(value,message):
 if not value:raise ValueError(message)


def read(path):
 return json.loads(Path(path).read_bytes(),parse_constant=lambda value:
  (_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
 with Path(path).open('x',encoding='utf8',newline='\n') as stream:
  json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def binding(path):
 path=Path(path);info=path.lstat()
 require(path.is_absolute() and '..' not in path.parts and stat.S_ISREG(info.st_mode)
  and not path.is_symlink() and all(not p.is_symlink() for p in path.parents),'Nonregular proof '+str(path))
 return {'path':str(path),'bytes':info.st_size,'sha256':sha(path)}


def check(row):
 require(type(row) is dict and set(row)=={'path','bytes','sha256'} and binding(row['path'])==row,
  'Original byte-bound input changed')
 return row


def verify(source,pin):
 source=Path(source)
 require(type(pin) is str and re.fullmatch('[a-f0-9]{64}',pin) is not None
  and source.is_dir() and not source.is_symlink() and sha(source/'source_manifest.json')==pin,'Source seal differs')
 manifest=read(source/'source_manifest.json');files=manifest['files']
 actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}
 require(type(files) is dict and files and actual==set(files)|{'source_manifest.json'},'Source membership differs')
 for p in (source,*source.rglob('*')):
  require(not p.is_symlink() and (p.is_dir() or p.is_file()) and not p.stat().st_mode&0o222,'Source is not ordinary read-only')
 for name,digest in files.items():
  rel=PurePosixPath(name)
  require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name and sha(source/name)==digest,'Source member changed')
 return manifest


def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);require(spec and spec.loader,'Cannot load verified metadata')
 result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result);return result


def lsf(raw,job,name,queue,success):
 compact=''.join(raw.split())
 subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
 subjects=[tuple(''.join(part.split()) for part in row) for row in subjects]
 require(compact.count('Subject:Job')==1 and subjects==[(job,name,'dcc','Done' if success else 'Exited')]
  and re.findall(r'inqueue<([^>]+)>',compact)==[queue],'Actual same-job LSF identity differs')
 require((compact.count('Successfullycompleted.')==1 and 'Exitedwithexit' not in compact) if success else
  (re.findall(r'Exitedwithexitcode(\d+)\.',compact)==['1'] and 'Successfullycompleted.' not in compact),
  'Actual original LSF outcome differs')


def external(manifest):
 admission=read(SOURCE/'admission.json');plan=read(SOURCE/'plan.json')
 require(admission['version']==VERSION and admission['pilots']==PILOTS,'Original pilot identity differs')
 require(plan['version']=='chemistry-softmax-replay-diagnostic-v2' and len(plan['points'])==8,'Replay scope differs')
 require(plan['scientific_source']=={'path':str(BASE/'protenix-extended-release-03/source'),
  'manifest_sha256':'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562','files':643}
  and plan['failed_pilot_source']=={'path':PILOTS['protenix']['source'],
  'manifest_sha256':PILOTS['protenix']['manifest_sha256'],'files':30},'Diagnostic original code producer changed')
 for row in admission['parent_sources']:
  require(len(verify(row['path'],row['manifest_sha256'])['files'])==row['files'],'Original source count differs')
 for row in admission['preserved_proofs']:check(row)
 prior_root=BASE/'chemistry-replay-01'
 prior=read(prior_root/'gpu/submission.json')
 require(prior['status']=='submitted' and prior['job_id']=='29816561'
  and prior['source_manifest_sha256']=='dbfef057ac054a0c4eb452b1603933db7815d91d38dcb3ffcad007105f3312d5',
  'Failed original replay job/source identity differs')
 lsf((prior_root/'gpu/stdout.29816561.log').read_text(),'29816561','mosaic-chem-replay01-gpu','gpul40s',False)
 prior_result=read(prior_root/'gpu/payload/diagnostic/assessment.json')
 require(prior_result['status']=='failed' and prior_result['all_controls_completed'] is False
  and len(prior_result['controls'])==10,'Original failed diagnostic reclassified')
 for row in prior_result['controls']:
  old=read(row['assessment'])
  require(old['status']=='failed' and len(old['points'])==8
   and old['input_after_error']=="RuntimeError('Replay diagnostic cannot open model/context checkpoint assets')",
   'Original failed diagnostic final-integrity cause changed')
 for arm,row in PILOTS.items():
  source=Path(row['source']);root=Path(row['root']);old=verify(source,row['manifest_sha256'])
  ops=module(source/'scripts/full_model_ops.py','_replay_original_'+arm)
  ops.external(old);ops.verify_cpu_certificate(row['manifest_sha256'],old,row['cpu_certificate_sha256'])
  sub=read(root/arm/'submission.json')
  require(sub['status']=='submitted' and sub['job_id']==row['job_id']
   and sub['argv']==ops.argv(arm,row['manifest_sha256'])
   and sub['cpu_certificate_sha256']==row['cpu_certificate_sha256'],'Original pilot submission differs')
  lsf((root/arm/('stdout.'+row['job_id']+'.log')).read_text(),row['job_id'],
      'mosaic-chem-full'+('03-' if arm=='protenix' else '02-')+arm,'gpul40s',arm=='boltz')
  top=read(root/arm/'payload/assessment.json');comparison=root/arm/'payload/comparison/assessment.json'
  require(top['job_id']==row['job_id'] and top['source_manifest_sha256']==row['manifest_sha256'],
   'Original allocated pilot assessment identity differs')
  if arm=='protenix':
   paired=read(root/arm/'payload/comparison/combined/paired-checks/assessment.json')
   require(top['status']=='failed' and paired['status']=='failed' and paired['points']==[]
    and '399 / 1500' in paired['error'] and '1.1920929e-07' in paired['error'],
    'Original exact replay failure was changed or reclassified')
  else:
   require(top['status']=='passed' and top['comparison_sha256']==sha(comparison)
    and read(comparison)['status']=='passed_full_model_comparison'
    and sha(comparison)=='e6e33bb5848981a60e21be546b8f7e1edd51bcecd08cf53f6cd04706ec539b6d',
    'Independent completed Boltz comparison changed')
 expected={(variant,label) for variant in ('baseline','combined') for label in ('early-0','early-10','high-0','high-10')}
 require({(p['variant'],p['label']) for p in plan['points']}==expected,'Replay point identity differs')
 by_path={row['path']:row for row in admission['preserved_proofs']}
 for point in plan['points']:
  for key in ('snapshot','replay'):
   check(point[key]);require(by_path.get(point[key]['path'])==point[key],'Replay point lacks protected original binding')
  record=read(point['replay']['path'])
  require(record['status']=='passed_exact_five_field_replay' and record['snapshot']==point['snapshot'],
   'Original same-child replay canary changed')
 return {'status':'original_failure_and_independent_Boltz_pass_preserved','points':8,
  'input_files':len(admission['preserved_proofs']),'can_certify_or_restart_original_study':False}


def own_source(pin):
 manifest=verify(SOURCE,pin)
 require(manifest['version']==VERSION and manifest['persistent_cap_bytes']==LIMIT
  and manifest['group_cap_bytes']==GROUP_LIMIT and manifest['production_authorized'] is False,'Wrong diagnostic source scope')
 return manifest


def tree_bytes(roots):
 roots=list(dict.fromkeys(Path(p) for p in roots))
 roots=[p for p in roots if not any(p!=q and p.is_relative_to(q) for q in roots)]
 for attempt in range(3):
  total=0
  try:
   for root in roots:
    if not root.exists():continue
    require(root.is_dir() and not root.is_symlink(),'Owned group root is linked/special')
    for directory,dirs,files in os.walk(root,followlinks=False):
     for name in dirs+files:
      path=Path(directory)/name;info=path.lstat()
      if stat.S_ISLNK(info.st_mode):
       require(any(path.resolve(strict=True).is_relative_to(p.resolve()) for p in roots if p.exists()),'External output link')
       total+=info.st_size
      elif stat.S_ISREG(info.st_mode):total+=info.st_size
      else:require(stat.S_ISDIR(info.st_mode),'Special output file')
   return total
  except FileNotFoundError:
   if attempt==2:raise
   time.sleep(.01)


def caps():
 own=tree_bytes((ROOT,SOURCE.parent));group=tree_bytes((ROOT,SOURCE.parent,
  BASE/'chemistry-full-model-02',BASE/'chemistry-full-model-release-02',
  BASE/'chemistry-full-model-03',BASE/'chemistry-full-model-release-03',
  BASE/'chemistry-replay-01',BASE/'chemistry-replay-release-01'))
 remaining=max(0,LIMIT-own);reserved=group+remaining
 require(own<=LIMIT and group<GROUP_LIMIT,'Bounded diagnostic/group persistent cap exceeded')
 require(reserved<=GROUP_LIMIT,'Retained validations plus the full new diagnostic cap exceed the existing group allowance')
 return {'own_bytes':own,'own_cap':LIMIT,'group_bytes':group,'group_cap':GROUP_LIMIT,
  'reserved_own_remaining_bytes':remaining,'reserved_group_total_bytes':reserved}


def finish(path,assessment):
 """A known cap failure can never leave a passed completion assessment."""
 error=None
 try:
  measured=caps();assessment.update(measured)
  length=len((json.dumps(assessment,indent=2,allow_nan=False)+'\n').encode('utf8'))
  require(measured['own_bytes']+length<=LIMIT and measured['group_bytes']+length<GROUP_LIMIT,
   'Completion metadata would exceed hard persistent caps')
 except BaseException as failure:
  error=failure;assessment.update(status='failed',cap_error=repr(failure))
 write(path,assessment)
 if error is not None:raise error
 caps()


def storage(parent,phase):
 verify(GUARD,GUARD_PIN);guard=module(GUARD/'scripts/protenix_precision_ops.py','_replay_current_guard')
 prior=guard.storage(parent,'preflight' if phase=='preflight' else 'dispatch' if phase=='dispatch' else 'entry')
 record={'status':'failed','whole_current_gate':prior,'additional_allowance_bytes':0,
  'reserved_chemistry_validation_bytes':GROUP_LIMIT,'required_bytes':prior['required_bytes'],
  'available_bytes':prior['available_bytes'],'reserve_bytes':4*1024**3,'file_reserve':100000,
  'original_gate_changed':False,'production_authorized':False}
 try:
  retained=prior['parent_gate']
  require(prior['status']=='passed' and retained['status']=='passed' and prior['reserve_bytes']==4*1024**3
   and prior['file_reserve']==100000 and retained['additional_allowance_bytes']==1024**3+GROUP_LIMIT+768*1024**2+512*1024**2,
   'Current complete storage guard and original reserved validation allowance must pass')
  record.update(caps());require(record['available_bytes']>=record['required_bytes'],'Current full storage does not fit')
  record['status']='passed'
 finally:write(Path(parent)/('replay-storage-'+phase+'.json'),record)
 return record


def resources(phase):
 require(phase in ('cpu','after-cpu','gpu'),'Undeclared diagnostic phase')
 if phase=='gpu':return ['-q','gpul40s','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]',
  '-gpu','num=1:mode=exclusive_process','-W','00:20']
 return ['-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]',
  '-W','00:40' if phase=='cpu' else '00:20']


def argv(phase,pin,dependency=None):
 parent=ROOT/phase;args=['bsub',*resources(phase),'-J','mosaic-chem-replay02-'+phase,
  '-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
 if phase=='after-cpu':
  require(re.fullmatch(r'ended\([1-9][0-9]*\)',dependency or '') is not None,'Own-ended CPU dependency required')
  args+=['-w',dependency]
 else:require(dependency is None,'Diagnostic CPU/GPU must be independent')
 return args+['bash','-l',str(SOURCE/'scripts/replay_stage.lsf'),str(SOURCE),pin,phase]


def submit(phase,pin,dependency=None):
 parent=ROOT/phase;parent.mkdir(exist_ok=False)
 row={'status':'unknown','phase':phase,'argv':argv(phase,pin,dependency),'dependency':dependency,
  'source_manifest_sha256':pin,'submitted_utc':datetime.now(timezone.utc).isoformat()}
 if phase=='gpu':row['cpu_certificate_sha256']=sha(ROOT/'cpu-certificate.json')
 write(parent/'submission-started.json',row)
 try:
  result=subprocess.run(row['argv'],capture_output=True,text=True,timeout=60)
  row.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
  jobs=re.findall(r'Job <([1-9][0-9]*)> is submitted',result.stdout)
  if result.returncode==0 and len(jobs)==1:row.update(status='submitted',job_id=jobs[0])
 except BaseException as error:row['error']=repr(error)
 finally:write(parent/'submission.json',row)
 require(row['status']=='submitted','Unknown submission outcome; never replay')
 return row


def bootstrap(pin):
 manifest=own_source(pin);external(manifest)
 require(not ROOT.exists(),'Existing/unknown diagnostic root forbids bootstrap')
 ROOT.mkdir();write(ROOT/'bootstrap-started.json',{'source_manifest_sha256':pin})
 storage(ROOT,'preflight')
 cpu=submit('cpu',pin);after=submit('after-cpu',pin,'ended('+cpu['job_id']+')')
 write(ROOT/'initial-submissions.json',{'cpu':cpu,'after_cpu':after,'production_authorized':False})
 caps();return cpu,after


def own(phase,pin):
 path=ROOT/phase/'submission.json'
 for _ in range(20):
  if path.exists():break
  time.sleep(1)
 row=read(path)
 require(row['status']=='submitted' and row['job_id']==os.environ.get('LSB_JOBID')
  and row['source_manifest_sha256']==pin and row['argv']==argv(phase,pin,row['dependency']),
  'Allocated stage differs from its actual exact submission')
 require(os.environ.get('LSB_QUEUE')==('gpul40s' if phase=='gpu' else 'hpc')
  and os.environ.get('LSB_DJOB_NUMPROC')==('8' if phase=='gpu' else '4' if phase=='cpu' else '1'),
  'Actual queue/CPU resources differ')
 if phase=='after-cpu':require(row['dependency']=='ended('+read(ROOT/'cpu/submission.json')['job_id']+')','Wrong own CPU dependency')
 if phase=='gpu':require(row['cpu_certificate_sha256']==sha(ROOT/'cpu-certificate.json'),'GPU CPU certificate pin changed')
 return row


def xml(path,counts):
 rows=list(ET.parse(path).iter('testcase'))
 require(rows and all(not list(r) for r in rows) and len({(r.get('classname'),r.get('name')) for r in rows})==len(rows)
  and dict(Counter(r.get('classname') for r in rows))==counts,'Allocated exact CPU scope failed/skipped/error/duplicate')
 return len(rows)


def certify_cpu(pin,manifest):
 sub=read(ROOT/'cpu/submission.json');require(sub['argv']==argv('cpu',pin),'Original CPU command differs')
 parent=ROOT/'after-cpu';proofs=[];job=sub['job_id']
 for index in range(16):
  result=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
  path=parent/('cpu-terminal-'+format(index,'03d')+'.json')
  write(path,{'job_id':job,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr});proofs.append(binding(path))
  raw=''.join(result.stdout.split());state=re.findall(r'Status<([^>]+)>',raw)
  require(result.returncode==0 and re.findall(r'Job<(\d+)>',raw)==[job]
   and re.findall(r'JobName<([^>]+)>',raw)==['mosaic-chem-replay02-cpu']
   and re.findall(r'Queue<([^>]+)>',raw)==['hpc'] and state in (['PEND'],['RUN'],['DONE']),
   'Actual same CPU terminal scheduler unavailable/failed; never infer success')
  if state==['DONE']:
   try:
    log=ROOT/'cpu'/('stdout.'+job+'.log');lsf(log.read_text(),job,'mosaic-chem-replay02-cpu','hpc',True);break
   except (ValueError,FileNotFoundError):
    if index==15:raise
  require(index<15,'Actual CPU did not finish within bounded metadata wait');time.sleep(2)
 assessment=read(ROOT/'cpu/payload/assessment.json');count=xml(ROOT/'cpu/payload/tests.xml',manifest['cpu_counts'])
 require(assessment['status']=='passed' and assessment['job_id']==job and assessment['source_manifest_sha256']==pin
  and assessment['cases']==count,'Allocated CPU assessment identity differs')
 cert={'status':'passed','source_manifest_sha256':pin,'job_id':job,'cpu_counts':manifest['cpu_counts'],'cases':count,
  'proofs':[binding(ROOT/'cpu/submission.json'),*proofs,binding(log),binding(ROOT/'cpu/payload/assessment.json'),
   binding(ROOT/'cpu/payload/tests.xml'),binding(ROOT/'cpu/payload/tests-command.json')]}
 write(ROOT/'cpu-certificate.json',cert);return cert


def verify_cpu(pin,manifest):
 cert=read(ROOT/'cpu-certificate.json')
 require(cert['status']=='passed' and cert['source_manifest_sha256']==pin and cert['cpu_counts']==manifest['cpu_counts'],
  'Own CPU certificate differs')
 for proof in cert['proofs']:check(proof)
 require(xml(ROOT/'cpu/payload/tests.xml',manifest['cpu_counts'])==cert['cases'],'Own exact raw CPU scope differs')
 return cert


def diagnostic(command,parent):
 process=subprocess.Popen(command,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 try:out,error=process.communicate(timeout=660)
 except BaseException:
  if process.poll() is None:
   os.killpg(process.pid,signal.SIGTERM)
   try:out,error=process.communicate(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);out,error=process.communicate(timeout=10)
  else:out,error=process.communicate()
  write(parent/'diagnostic-command.json',{'argv':command,'returncode':process.returncode,
   'stdout':out.decode(errors='replace'),'stderr':error.decode(errors='replace'),'interrupted':True});raise
 write(parent/'diagnostic-command.json',{'argv':command,'returncode':process.returncode,
  'stdout':out.decode(errors='replace'),'stderr':error.decode(errors='replace')})
 require(process.returncode==0,'Bounded separate replay diagnostic failed')


def completed_diagnostic(result):
 expected=[(control,repeat) for control in ('plain','matrix_import','matrix_installed','combined','native_backend_import',
  'plain_hash0','combined_hash0','plain_autotune0','combined_autotune0',
  'plain_hash0_autotune0','combined_hash0_autotune0')
  for repeat in (0,1)]
 rows=result.get('controls')
 require(result.get('status')=='diagnostics_completed_original_failure_preserved'
  and result.get('all_controls_completed') is True
  and result.get('original_gate_reclassified') is False and result.get('can_certify_or_restart_original_study') is False
  and result.get('optimizer_updates')==0 and result.get('model_evaluations')==0,
  'Diagnostic is incomplete or overstates its scope')
 require(type(rows) is list and len(rows)==22
  and [(row.get('control'),row.get('repeat')) for row in rows]==expected
  and all(type(row.get('returncode')) is int and row['returncode']==0
   and row.get('status')=='diagnostic_control_completed_original_failure_preserved'
   and type(row.get('assessment_sha256')) is str and re.fullmatch('[a-f0-9]{64}',row['assessment_sha256']) is not None
   for row in rows),'All22 exact declared successful controls are mandatory')
 return True


def run(pin,phase):
 require(os.environ.get('LSB_JOBID') and not sys.flags.optimize,'Allocated unoptimized HPC only')
 manifest=own_source(pin);preserved=external(manifest);sub=own(phase,pin);parent=ROOT/phase
 if phase=='after-cpu':
  certify_cpu(pin,manifest);storage(parent,'dispatch');caps()
  require(not (ROOT/'gpu').exists(),'Unknown/existing GPU outcome prohibits replay')
  gpu=submit('gpu',pin);write(parent/'gpu-submission-complete.json',gpu);caps();return
 payload=parent/'payload';payload.mkdir(exist_ok=False)
 assessment={'status':'failed','job_id':sub['job_id'],'source_manifest_sha256':pin,'phase':phase,
  'original_results_changed':False,'production_authorized':False,'optimizer_updates':0,'model_evaluations':0}
 try:
  assessment['storage']=storage(payload,'entry');assessment['preserved_inputs']=preserved
  if phase=='cpu':
   command=[PYTHON,'-m','pytest','-q','-o','addopts=','-p','no:cacheprovider',
    *[str(SOURCE/'tests'/(name+'.py')) for name in manifest['cpu_modules']],
    '--junitxml='+str(payload/'tests.xml')]
   result=subprocess.run(command,cwd=SOURCE,capture_output=True,text=True,timeout=1800)
   write(payload/'tests-command.json',{'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
   require(result.returncode==0,'New replay controller/diagnostic metadata fixtures failed')
   assessment['cases']=xml(payload/'tests.xml',manifest['cpu_counts'])
  else:
   verify_cpu(pin,manifest)
   command=[PYTHON,str(SOURCE/'scripts/softmax_replay_diagnostic.py'),'--plan',str(SOURCE/'plan.json'),
    '--plan-sha256',sha(SOURCE/'plan.json'),'--output',str(payload/'diagnostic')]
   diagnostic(command,payload);result=read(payload/'diagnostic/assessment.json')
   completed_diagnostic(result)
   for row in result['controls']:
    path=payload/'diagnostic'/(row['control']+'-'+str(row['repeat']))/'control.json'
    require(row['assessment']==str(path) and sha(path)==row['assessment_sha256'],
     'Completed child control receipt path/hash differs')
    control=read(path)
    require(control['status']=='diagnostic_control_completed_original_failure_preserved'
     and control['control']==row['control'] and control['repeat']==row['repeat']
     and control['optimizer_updates']==0 and control['model_evaluations']==0
     and control['can_certify_or_restart_original_study'] is False and control['original_gate_reclassified'] is False,
     'Completed child control overstates its scientific scope')
   assessment['diagnostic_assessment']=binding(payload/'diagnostic/assessment.json')
  assessment['preserved_inputs_after']=external(manifest);assessment.update(caps());assessment['status']='passed'
 except BaseException as error:assessment['error']=repr(error);raise
 finally:finish(payload/'assessment.json',assessment)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('phase',choices=('cpu','after-cpu','gpu'))
 args=parser.parse_args();run(args.manifest,args.phase)


if __name__=='__main__':main()
