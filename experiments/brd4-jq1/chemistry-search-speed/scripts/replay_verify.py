"""Independent stdlib byte/schema audit of delivered replay-only HPC evidence.

No native NPZ is decoded, model is imported or scientific calculation is run.
An unfinished or failed diagnostic remains unfinished or failed in this audit.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-chemistry-replay-01'
BASE='/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT=BASE+'/chemistry-replay-01';SOURCE=BASE+'/chemistry-replay-release-01/source'
PIN='dbfef057ac054a0c4eb452b1603933db7815d91d38dcb3ffcad007105f3312d5'
ARCHIVE_SHA='85ab8c06c10ca704013dcc3b0b144a4ce62fd7e547026620cbe19ca2995670ac'
PLAN_SHA='19102b7e629bf873c4b6f72453ad924bc91fe52d9defd47b2741f533c924a404'
COUNTS={'tests.test_replay_ops.ReplayControllerTests':30,
 'tests.test_softmax_replay_diagnostic.SoftmaxReplayMetadataTests':30}
CONTROLS=('plain','matrix_import','matrix_installed','combined','native_backend_import')
POINTS=[(variant,label) for variant in ('baseline','combined') for label in ('early-0','early-10','high-0','high-10')]
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def require(value,message):
 if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


def parse(raw):
 return json.loads(raw,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('Nonfinite metadata '+value)))


def read(path):return parse(Path(path).read_bytes())


def lsf_outcome(raw,sub):
 compact=''.join(raw.split());args=sub['argv']
 if 'Subject:Job' not in compact:return 'not_yet_terminal',None
 subjects=re.findall(r'Subject:\s*Job\s*(\d+)\s*:\s*<([^>]+)>\s*in\s*cluster\s*<([^>]+)>\s*([A-Za-z]+)[ \t]*(?=\r?\n|$)',raw)
 subjects=[tuple(''.join(part.split()) for part in row) for row in subjects]
 require(compact.count('Subject:Job')==1 and len(subjects)==1
  and subjects[0][:3]==(sub['job_id'],args[args.index('-J')+1],'dcc')
  and re.findall(r'inqueue<([^>]+)>',compact)==[args[args.index('-q')+1]],'Raw LSF same-job/name/queue identity differs')
 if subjects[0][3]=='Done':
  require(compact.count('Successfullycompleted.')==1 and 'Exitedwithexit' not in compact,'Raw LSF DONE lacks success')
  return 'actual_LSF_success',0
 require(subjects[0][3]=='Exited' and 'Successfullycompleted.' not in compact,'Unknown raw terminal LSF outcome')
 codes=re.findall(r'Exitedwithexitcode(\d+)\.',compact)
 require(len(codes)==1 and int(codes[0])>0,'Raw LSF EXIT lacks unique failure code')
 return 'actual_LSF_failed',int(codes[0])


def resources(phase):
 if phase=='gpu':return ['-q','gpul40s','-n','8','-R','span[hosts=1]','-R','rusage[mem=4GB]',
  '-gpu','num=1:mode=exclusive_process','-W','00:20']
 require(phase in ('cpu','after-cpu'),'Unknown submitted diagnostic phase')
 return ['-q','hpc','-n','4' if phase=='cpu' else '1','-R','span[hosts=1]','-R','rusage[mem=4GB]',
  '-W','00:40' if phase=='cpu' else '00:20']


def scheduler_states(query,phases):
 """Authenticate actual live blocks; absence is never converted into DONE."""
 raw=query['stdout'];matches=list(re.finditer(r'(?m)^\s*Job <([1-9][0-9]*)>,',raw))
 blocks={}
 for index,match in enumerate(matches):
  job=match.group(1);require(job not in blocks,'Duplicated live scheduler job block')
  blocks[job]=raw[match.start():matches[index+1].start() if index+1<len(matches) else len(raw)]
 require(set(blocks)<=set(query['job_ids']),'Live scheduler returned an unrequested job')
 observed={}
 for phase,sub in phases.items():
  job=sub['job_id']
  if job not in blocks:
   missing=re.search(r'Job\s*<'+re.escape(job)+r'>\s*is not found',query['stderr']) is not None
   observed[phase]={'state':None,'live_record':'absent_aged_out' if missing else 'unavailable_no_absence_proof',
    'job_id':job,'terminal_success_inferred':False}
   continue
  compact=''.join(blocks[job].split());args=sub['argv'];state=re.findall(r'Status<([^>]+)>',compact)
  require(re.findall(r'Job<(\d+)>',compact)==[job]
   and re.findall(r'JobName<([^>]+)>',compact)==[args[args.index('-J')+1]]
   and re.findall(r'Queue<([^>]+)>',compact)==[args[args.index('-q')+1]]
   and len(state)==1 and state[0] in ('PEND','RUN','DONE','EXIT','PSUSP','USUSP','SSUSP','WAIT','UNKWN','ZOMBI'),
   'Actual live scheduler same-job/name/queue/state differs')
  command=re.findall(r'Command<([^>]+)>',compact)
  require(command==[''.join(args[args.index('bash'):])],'Actual live command/source/phase differs')
  requested=re.findall(r'RequestedResources<([^>]+)>',compact)
  require(len(requested)==1 and 'span[hosts=1]' in requested[0]
   and set(re.findall(r'rusage\[mem=([0-9.]+)\]',requested[0])) in ({'4096'},{'4096.0'},{'4096.00'}),
   'Actual live requested memory/host resources differ')
  slots=re.findall(r'(\d+)Task\(s\),RequestedResources',compact)
  require(slots==[args[args.index('-n')+1]] or (phase=='after-cpu' and not slots),
   'Actual live requested CPU task count differs')
  limits=re.findall(r'RUNLIMIT([0-9.]+)min',compact)
  require(limits==['40.0' if phase=='cpu' else '20.0'],'Actual live wall-time limit differs')
  dependencies=re.findall(r'DependencyCondition<([^>]+)>',compact)
  require(dependencies==([sub['dependency']] if sub['dependency'] is not None else []),
   'Actual live own-ended dependency differs')
  requested_gpu=re.findall(r'RequestedGPU<([^>]+)>',compact)
  if phase=='gpu':
   require(requested_gpu==['num=1:mode=exclusive_process']
    and 'GPUREQUIREMENTDETAILS:Combined:num=1:mode=exclusive_process:mps=no:j_exclusive=yes' in compact,
    'Actual live exclusive whole-job GPU request differs')
  else:require(not requested_gpu,'CPU/callback unexpectedly requests GPU')
  observed[phase]={'state':state[0],'live_record':'actual_same_job_record_verified','job_id':job,
   'raw_block_sha256':sha(blocks[job].encode()),'terminal_success_inferred':False}
 return observed


def verify(observation,package):
 receipt=read(observation/'receipt.json');raw=(observation/'raw.json').read_bytes()
 require(receipt['returncode']==0 and receipt['stdout_sha256']==sha(raw)
  and receipt['program_sha256']==sha((observation/'program.py').read_bytes()),'Delivered observation program/raw outcome differs')
 packet=parse(raw)
 require(packet['version']=='chemistry-replay-observation-v1' and packet['read_only'] is True
  and packet['root']==ROOT and packet['source']==SOURCE,'Wrong compact observation root/source/scope')
 files={};remote={};total=0
 for row in packet['files']:
  relative=PurePosixPath(row['scope'])/row['relative']
  require(row['scope'] in ('root','source','external') and not relative.is_absolute() and '..' not in relative.parts
   and '\\' not in str(relative) and str(relative) not in files and row['path'] not in remote,'Unsafe/duplicate compact member')
  data=base64.b64decode(row['base64'],validate=True)
  require(type(row['bytes']) is int and len(data)==row['bytes'] and sha(data)==row['sha256']
   and (observation/'files'/Path(*relative.parts)).read_bytes()==data,'Delivered metadata member bytes differ')
  if row['scope']!='external':require(row['path']==(SOURCE if row['scope']=='source' else ROOT)+'/'+row['relative'],
   'Compact member remote path redirected')
  require(PurePosixPath(row['path']).suffix!='.npz','Compact diagnostic packet unexpectedly contains native NPZ bytes')
  files[str(relative)]=data;remote[row['path']]=data;total+=len(data)
 require(total==packet['compact_file_bytes'] and total<=64*1024**2,'Compact packet cap/total differs')
 def data(path):
  require(path in remote,'Required delivered compact proof absent: '+path);return remote[path]
 def obj(path):return parse(data(path))
 def bound(row,required=True):
  require(type(row) is dict and set(row)=={'path','bytes','sha256'} and type(row['bytes']) is int and row['bytes']>0
   and type(row['path']) is str and PurePosixPath(row['path']).is_absolute() and '..' not in PurePosixPath(row['path']).parts
   and re.fullmatch('[a-f0-9]{64}',row['sha256']) is not None,'Malformed original proof binding')
  if row['path'] not in remote:
   require(not required,'Bound compact proof absent: '+row['path']);return False
  value=data(row['path']);require(len(value)==row['bytes'] and sha(value)==row['sha256'],'Bound compact proof bytes changed');return True
 preparation=read(package/'preparation.json');archive=package/'source.tar.gz';archive_raw=archive.read_bytes()
 require(preparation['manifest_sha256']==PIN and preparation['archive_sha256']==ARCHIVE_SHA
  and preparation['archive_bytes']==24127 and preparation['files']==11 and preparation['cpu_counts']==COUNTS
  and len(archive_raw)==24127 and sha(archive_raw)==ARCHIVE_SHA,'Original staged capsule/preparation differs')
 with tarfile.open(archive) as capsule:
  members=capsule.getmembers()
  require(len(members)==12 and len({m.name for m in members})==12 and all(m.isfile() for m in members),'Expected exact12 regular source members')
  archived={}
  for member in members:
   rel=PurePosixPath(member.name)
   require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in member.name,'Unsafe original source archive member')
   archived[member.name]=capsule.extractfile(member).read()
 manifest_raw=data(SOURCE+'/source_manifest.json');manifest=parse(manifest_raw)
 require(sha(manifest_raw)==PIN and manifest_raw==archived['source_manifest.json']
  and manifest['version']=='chemistry-softmax-replay-source-v1' and manifest['cpu_counts']==COUNTS
  and manifest['persistent_cap_bytes']==64*1024**2 and manifest['group_cap_bytes']==2*1024**3
  and manifest['additional_allowance_bytes']==0 and manifest['production_authorized'] is False
  and manifest['monitor_remains_paused'] is True and len(manifest['files'])==11,'New seal scope/caps/CPU count differs')
 require(set(archived)==set(manifest['files'])|{'source_manifest.json'}
  and {name.removeprefix('source/') for name in files if name.startswith('source/')}==set(archived),
  'Exact delivered complete source membership differs')
 for name,digest in manifest['files'].items():
  require(sha(archived[name])==digest and data(SOURCE+'/'+name)==archived[name],'Actual source member changed: '+name)
 plan=obj(SOURCE+'/plan.json');admission=obj(SOURCE+'/admission.json')
 require(sha(data(SOURCE+'/plan.json'))==PLAN_SHA and plan['version']=='chemistry-softmax-replay-diagnostic-v1'
  and [(p['variant'],p['label']) for p in plan['points']]==POINTS,'Exact saved-point plan changed')
 references=admission['preserved_proofs'];reference_map={r['path']:r for r in references}
 require(len(reference_map)==len(references) and admission['production_authorized'] is False
  and admission['original_results_changed'] is False,'Admission original-proof scope differs')
 for point in plan['points']:
  for key in ('snapshot','replay'):
   require(reference_map.get(point[key]['path'])==point[key],'Original point is not byte-bound in admission')
   bound(point[key],required=False)
 expected_science={'path':BASE+'/protenix-extended-release-03/source','manifest_sha256':'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562','files':643}
 expected_failed={'path':BASE+'/chemistry-full-model-release-03/source','manifest_sha256':'8480a39c05d612d1a80ea1892d816c9429fcd471e97adb31655f8132d9007f31','files':30}
 require(plan['scientific_source']==expected_science and plan['failed_pilot_source']==expected_failed,'Original scientific/replay code scope differs')
 phases={};unknown=[]
 for path,value in remote.items():
  if not path.startswith(ROOT+'/') or PurePosixPath(path).name!='submission.json':continue
  row=parse(value)
  if row.get('status')!='submitted':unknown.append({'path':path,'record':row});continue
  phase=row['phase'];require(path==ROOT+'/'+phase+'/submission.json' and phase not in phases,'Actual phase path duplicated/redirected')
  require(re.fullmatch('[1-9][0-9]*',row['job_id']) is not None,'Actual submitted job lacks ID')
  command=['bsub',*resources(phase),'-J','mosaic-chem-replay01-'+phase,
   '-oo',ROOT+'/'+phase+'/stdout.%J.log','-eo',ROOT+'/'+phase+'/stderr.%J.log']
  if row['dependency'] is not None:command+=['-w',row['dependency']]
  command+=['bash','-l',SOURCE+'/scripts/replay_stage.lsf',SOURCE,PIN,phase]
  require(row['argv']==command and row['source_manifest_sha256']==PIN and row['returncode']==0
   and re.findall(r'Job <([1-9][0-9]*)> is submitted',row['stdout'])==[row['job_id']],
   'Actual submitted resources/source/argv/job result differs')
  marker=obj(ROOT+'/'+phase+'/submission-started.json')
  require(marker['status']=='unknown' and all(marker[k]==row[k] for k in
   ('phase','argv','dependency','source_manifest_sha256')),'Exclusive unknown marker was replaced/changed')
  phases[phase]=row
 require({'cpu','after-cpu'}<=set(phases) and phases['cpu']['job_id']=='29816497'
  and phases['after-cpu']['job_id']=='29816498' and phases['cpu']['dependency'] is None
  and phases['after-cpu']['dependency']=='ended(29816497)','Actual initial CPU/callback identity/dependency differs')
 if 'gpu' in phases:require(phases['gpu']['dependency'] is None,'Diagnostic GPU must have no inter-arm dependency')
 require(len({r['job_id'] for r in phases.values()})==len(phases),'Duplicate actual phase job IDs')
 query=packet['scheduler_query'];actual_ids=sorted([r['job_id'] for r in phases.values()],key=int)
 require(query['job_ids']==actual_ids and query['invoked'] is True,'Fresh combined query uses invented/missing job IDs')
 live=scheduler_states(query,phases)
 lsf={}
 for phase,sub in phases.items():
  log=ROOT+'/'+phase+'/stdout.'+sub['job_id']+'.log'
  lsf[phase]=lsf_outcome(data(log).decode(),sub) if log in remote else ('not_yet_terminal',None)
  if lsf[phase][0]=='actual_LSF_success' and live[phase]['state'] is not None:
   require(live[phase]['state']=='DONE','Known live scheduler conflicts with same-job successful terminal LSF')
  if lsf[phase][0]=='actual_LSF_failed' and live[phase]['state'] is not None:
   require(live[phase]['state']=='EXIT','Known live scheduler conflicts with same-job failed terminal LSF')
 cpu_status='not_certified';certificate=ROOT+'/cpu-certificate.json'
 if certificate in remote:
  cert=obj(certificate)
  require(cert['status']=='passed' and cert['source_manifest_sha256']==PIN and cert['job_id']=='29816497'
   and cert['cpu_counts']==COUNTS and cert['cases']==60,'Actual CPU certificate scope differs')
  for proof in cert['proofs']:bound(proof)
  assessment=obj(ROOT+'/cpu/payload/assessment.json');cases=list(ET.fromstring(data(ROOT+'/cpu/payload/tests.xml')).iter('testcase'))
  require(assessment['status']=='passed' and assessment['job_id']=='29816497' and assessment['source_manifest_sha256']==PIN
   and assessment['cases']==60 and assessment['production_authorized'] is False
   and assessment['model_evaluations']==0 and assessment['optimizer_updates']==0,'Actual CPU assessment differs')
  expected_preserved={'status':'original_failure_and_independent_Boltz_pass_preserved','points':8,
   'input_files':len(references),'can_certify_or_restart_original_study':False}
  require(assessment['preserved_inputs']==assessment['preserved_inputs_after']==expected_preserved,
   'CPU did not preserve the exact original failure and independent Boltz pass before/after')
  require(len(cases)==60 and len({(r.get('classname'),r.get('name')) for r in cases})==60 and all(not list(r) for r in cases)
   and dict(Counter(r.get('classname') for r in cases))==COUNTS,'Raw CPU60 class-attributed XML failed/missing/skip/duplicate')
  command=obj(ROOT+'/cpu/payload/tests-command.json')
  expected=[PYTHON,'-m','pytest','-q','-o','addopts=','-p','no:cacheprovider',
   SOURCE+'/tests/test_replay_ops.py',SOURCE+'/tests/test_softmax_replay_diagnostic.py',
   '--junitxml='+ROOT+'/cpu/payload/tests.xml']
  require(command['argv']==expected and command['returncode']==0 and re.search(r'\b60\s+passed\b',command['stdout'])
   and re.search(r'\b[1-9][0-9]*\s+deselected\b',command['stdout']) is None,'Actual raw CPU command/count differs')
  require(lsf['cpu']==('actual_LSF_success',0),'CPU certificate lacks same-job terminal raw LSF success')
  queries=sorted(p for p in remote if p.startswith(ROOT+'/after-cpu/cpu-terminal-'))
  require(1<=len(queries)<=16 and queries==[ROOT+'/after-cpu/cpu-terminal-'+format(i,'03d')+'.json' for i in range(len(queries))],
   'CPU certificate lacks exact bounded contiguous terminal scheduler proof paths')
  expected_proofs=[ROOT+'/cpu/submission.json',*queries,ROOT+'/cpu/stdout.29816497.log',
   ROOT+'/cpu/payload/assessment.json',ROOT+'/cpu/payload/tests.xml',ROOT+'/cpu/payload/tests-command.json']
  require([p['path'] for p in cert['proofs']]==expected_proofs,'CPU certificate proof membership/order differs from sealed controller')
  states=[]
  for path in queries:
   record=obj(path);compact=''.join(record['stdout'].split());state=re.findall(r'Status<([^>]+)>',compact)
   require(record['returncode']==0 and record['job_id']=='29816497'
    and re.findall(r'Job<(\d+)>',compact)==['29816497']
    and re.findall(r'JobName<([^>]+)>',compact)==['mosaic-chem-replay01-cpu']
    and re.findall(r'Queue<([^>]+)>',compact)==['hpc'] and state in (['RUN'],['PEND'],['DONE']),
    'Persisted same-job CPU scheduler proof failed/detached')
   states.append(state[0])
  require(states[-1]=='DONE','Persisted scheduler never established original CPU terminal success')
  cpu_status='actual_CPU_60_pass_raw_XML_LSF_certificate_verified'
 callback_status='pending_or_running_callback_no_terminal_outcome'
 if lsf['after-cpu'][0]=='actual_LSF_failed':callback_status='actual_callback_failed_original_failure_preserved'
 elif lsf['after-cpu'][0]=='actual_LSF_success':
  callback_status='actual_callback_terminal_success'
  require('gpu' in phases,'Completed successful callback lacks its actual GPU submission; take a fresh observation')
  complete=obj(ROOT+'/after-cpu/gpu-submission-complete.json')
  require(complete==phases['gpu'],'Callback GPU completion record differs from actual submitted vector')
  callback_status='actual_callback_successfully_submitted_one_diagnostic_GPU'
 elif live['after-cpu']['state']=='EXIT':callback_status='actual_callback_EXIT_raw_terminal_LSF_not_yet_captured'
 elif live['after-cpu']['state']=='DONE':callback_status='actual_callback_DONE_raw_terminal_LSF_not_yet_captured'
 diagnostic_status='not_submitted_no_findings';diagnostic_result=None;controls=[]
 if 'gpu' in phases:
  require(certificate in remote and phases['gpu']['cpu_certificate_sha256']==sha(data(certificate)),
   'Actual diagnostic GPU is detached from its own CPU certificate')
  diagnostic_status='submitted_pending_or_running_no_findings'
  gpu_assessment=ROOT+'/gpu/payload/assessment.json'
  if gpu_assessment in remote:
   current=obj(gpu_assessment)
   require(current['job_id']==phases['gpu']['job_id'] and current['source_manifest_sha256']==PIN
    and current['production_authorized'] is False and current['original_results_changed'] is False
    and current['optimizer_updates']==0 and current['model_evaluations']==0,'GPU wrapper scope differs')
   if current['status']=='failed':
    diagnostic_status='failed_diagnostic_original_failure_preserved'
    diagnostic_result={'assessment':current,'raw_LSF_outcome':lsf['gpu']}
   else:
    require(current['status']=='passed','Unknown GPU wrapper outcome')
    expected_preserved={'status':'original_failure_and_independent_Boltz_pass_preserved','points':8,
     'input_files':len(references),'can_certify_or_restart_original_study':False}
    require(current['preserved_inputs']==current['preserved_inputs_after']==expected_preserved,
     'GPU wrapper did not preserve original failure/Boltz evidence before/after')
    bound(current['diagnostic_assessment']);result=obj(ROOT+'/gpu/payload/diagnostic/assessment.json')
    execution=obj(ROOT+'/gpu/payload/diagnostic-command.json')
    require(execution['argv']==[PYTHON,SOURCE+'/scripts/softmax_replay_diagnostic.py','--plan',SOURCE+'/plan.json',
     '--plan-sha256',PLAN_SHA,'--output',ROOT+'/gpu/payload/diagnostic'] and execution['returncode']==0
     and not execution.get('interrupted',False),'Actual parent diagnostic execution failed/differs')
    require(result['version']=='chemistry-softmax-replay-diagnostic-v1'
     and result['status']=='diagnostics_completed_original_failure_preserved' and result['all_controls_completed'] is True
     and result['original_gate_reclassified'] is False and result['can_certify_or_restart_original_study'] is False
     and result['matching_control_selected_as_fix'] is False and result['optimizer_updates']==0 and result['model_evaluations']==0
     and result['persistent_cap_bytes']==64*1024**2 and result['internal_seconds']==600
     and result['input_before']==result['input_after'],'Completed diagnostic scope/input restoration differs')
    expected_points=[p[key] for p in plan['points'] for key in ('snapshot','replay')]
    require(result['input_before']=={'sources':{'scientific_source':expected_science,'failed_pilot_source':expected_failed},
     'point_bindings':expected_points},'Diagnostic original source/8-point identity differs')
    rows=result['controls'];expected_controls=[(control,repeat) for control in CONTROLS for repeat in (0,1)]
    require(type(rows) is list and len(rows)==10 and [(r['control'],r['repeat']) for r in rows]==expected_controls,
     'Exactly ten declared controls are mandatory')
    for row in rows:
     prefix=ROOT+'/gpu/payload/diagnostic/'+row['control']+'-'+str(row['repeat']);path=prefix+'/control.json'
     require(type(row['returncode']) is int and row['returncode']==0
      and row['status']=='diagnostic_control_completed_original_failure_preserved'
      and row['assessment']==path and sha(data(path))==row['assessment_sha256'],'Control did not complete with exact delivered receipt')
     expected=[PYTHON,SOURCE+'/scripts/softmax_replay_diagnostic.py','--plan',SOURCE+'/plan.json','--plan-sha256',PLAN_SHA,
      '--output',prefix,'--control',row['control'],'--repeat',str(row['repeat'])]
     require(row['argv']==expected,'Actual private child control argv differs')
     control=obj(path)
     require(control['version']=='chemistry-softmax-replay-diagnostic-v1'
      and control['status']==row['status'] and control['control']==row['control'] and control['repeat']==row['repeat']
      and control['scientific_gate_passed'] is False and control['optimizer_updates']==0 and control['model_evaluations']==0
      and control['probability_substitution'] is False and control['can_certify_or_restart_original_study'] is False
      and control['original_gate_reclassified'] is False and control['input_before']==control['input_after']==result['input_before']
      and [(p['variant'],p['label']) for p in control['points']]==POINTS,'Completed child scope/point/input restoration differs')
     controls.append({'control':row['control'],'repeat':row['repeat'],'assessment_sha256':row['assessment_sha256']})
    diagnostic_result=result
    diagnostic_status='diagnostic_metadata_complete_awaiting_actual_terminal_proof'
    if lsf['gpu']==('actual_LSF_success',0):
     if live['gpu']['state']=='DONE':diagnostic_status='completed_diagnostic_evidence_verified_original_failure_preserved'
     elif live['gpu']['live_record']=='absent_aged_out':
      diagnostic_status='completed_diagnostic_persisted_LSF_verified_live_record_aged_out_original_failure_preserved'
    if lsf['gpu'][0]=='actual_LSF_failed':diagnostic_status='failed_wrapper_after_diagnostic_metadata_original_failure_preserved'
  elif lsf['gpu'][0]=='actual_LSF_failed':diagnostic_status='failed_before_diagnostic_assessment_original_failure_preserved'
 undownloaded=[r for r in references if not bound(r,required=False)]
 return {'version':'chemistry-replay-independent-delivery-v1','status':'verified_compact_source_and_actual_execution_scope',
  'verified_utc':datetime.now(timezone.utc).isoformat(),'raw_sha256':sha(raw),'source_manifest_sha256':PIN,
  'source_files':11,'archive_members':12,'compact_files':len(files),'CPU_status':cpu_status,
  'actual_submissions':phases,'actual_raw_LSF_outcomes':lsf,'diagnostic_status':diagnostic_status,
  'actual_live_scheduler_records':live,'callback_status':callback_status,
  'diagnostic_result':diagnostic_result,'verified_completed_controls':controls,'unknown_submission_records':packet['unknown_submission_records'],
  'original_reference_bindings_not_downloaded':undownloaded,'native_NPZ_downloaded':False,
  'scientific_results_recomputed_or_recertified':False,'original_failed_scientific_gate_reclassified':False,
  'scope':'Metadata/source/CPU/diagnostic byte proof only; pending is not a finding and failed diagnostics remain failed; original native arrays stay HPC references'}


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--observation',type=Path,required=True)
 parser.add_argument('--package',type=Path,default=LOCAL/'source01');parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();require(not args.output.exists(),'Choose a fresh independent verification output')
 result=verify(args.observation,args.package)
 with args.output.open('x',encoding='utf8',newline='\n') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'status':result['status'],'CPU_status':result['CPU_status'],'diagnostic_status':result['diagnostic_status'],
  'jobs':{k:v['job_id'] for k,v in result['actual_submissions'].items()},'audit_sha256':sha(args.output.read_bytes())},indent=2))


if __name__=='__main__':main()
