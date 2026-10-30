"""Read-only diagnostic orchestration; no optimizer/continuation branch."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
CAPACITY=BASE/'nonchem-precision-release-26/source'
CAPACITY_PIN='8324c1d312adeaa4c67c475c01ef1217df6f3ad4044b997d1b73e168325412d3'
SCOPE=BASE/'nonchem-precision-continuation-26'
ROOT=SCOPE/'chemistry-continuation-05/boundary-diagnostics-07'
SOURCE=SCOPE/'boundary-diagnostics-release-07/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'


def require(v,message):
    if not v:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);v=importlib.util.module_from_spec(s);sys.modules[name]=v;s.loader.exec_module(v);return v
def authenticate_capacity():
    require(not any(p.is_symlink() for p in (CAPACITY,*CAPACITY.parents)),'Linked capacity source')
    raw=(CAPACITY/'source_manifest.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest()==CAPACITY_PIN,'Current capacity source manifest changed')
    m=json.loads(raw);require(len(m['files'])==50,'Exact current capacity source count required')
    for name,digest in m['files'].items():
        p=CAPACITY/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not p.is_symlink(),'Invalid capacity member')
        with p.open('rb') as f:require(hashlib.file_digest(f,'sha256').hexdigest()==digest,'Current capacity member changed')
def context(pin):
    authenticate_capacity()
    op=module(CAPACITY/'scripts/precision_continuation_ops.py','_current_diagnostic_capacity')
    current=op.verify(CAPACITY,CAPACITY_PIN);op.certificate(CAPACITY_PIN,current)
    own=op.verify(SOURCE,pin);plan=read(SOURCE/'plan.json')
    for row in own['parent_sources']:op.verify(Path(row['path']),row['manifest_sha256'])
    for b in plan['inputs']:op.check_binding(b)
    return op,plan
def argv(phase,pin,dependency=None):
    require(phase in ('cpu','after-cpu','gpu-plain','gpu-chemistry'),'Unknown diagnostic phase')
    gpu=phase.startswith('gpu-');slots=8 if phase=='gpu-chemistry' else 1 if phase=='after-cpu' else 4;p=ROOT/phase
    a=['bsub','-H','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
       '-q','gpul40s' if gpu else 'hpc','-n',str(slots),'-R','span[hosts=1]','-R',
       'rusage[mem='+('8GB' if phase=='gpu-plain' else '2GB' if slots==1 else '4GB')+']','-W','01:00' if gpu else '00:30' if slots==1 else '01:00',
       '-J','mosaic-boundary-diagnostics07-'+phase,'-o',str(p/'stdout.%J.log'),'-e',str(p/'stderr.%J.log')]
    if gpu:a+=['-gpu','num=1:mode=exclusive_process:mps=yes']
    if dependency:a+=['-w',dependency]
    return a+['bash','-l',str(SOURCE/'scripts/failure_diagnostic_stage.lsf'),str(SOURCE),pin,phase,'--capacity-source',str(CAPACITY)]
def release_argv(row,expected,persisted,registered):
    require(row==persisted,'Held receipt must already be persisted exactly')
    require(row.get('status')=='submitted' and row.get('returncode')==0,
            'Only an authenticated accepted held submission may be released')
    require(row.get('dispatch_state')=='held','Only the initial owned hold may be released')
    require(isinstance(row.get('job_id'),str) and row['job_id'].isdigit(), 'Exact numeric owned job ID required')
    require(row.get('argv')==expected and expected[:2]==['bsub','-H'], 'Held source/resource vector changed')
    require(registered==row['job_id'], 'Held job must already be registered exactly')
    return ['bresume',row['job_id']]

def captured_command(command,output,env=None,timeout=300):
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout,env=env)
    except subprocess.TimeoutExpired as error:
        def decoded(value):return value.decode(errors='replace') if isinstance(value,bytes) else value or ''
        write(output,{'status':'unknown_after_timeout_do_not_retry','argv':command,'returncode':None,
                      'stdout':decoded(error.stdout),'stderr':decoded(error.stderr),'timeout_seconds':timeout})
        raise
    write(output,{'argv':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    return result

def held_observation(returncode,stdout,jid,name):
    lines=[line.split() for line in stdout.splitlines() if line.strip()]
    require(returncode==0 and len(lines)==1 and len(lines[0])==4,
            'Unambiguous actual held scheduler row required')
    actual_id,state,user,actual_name=lines[0]
    require((actual_id,user,actual_name)==(jid,'s234818',name),'Held scheduler identity differs')
    require(state in ('PEND','PSUSP'),'Unexpected held job state; reconcile before release')
    return state=='PSUSP'

def submit(phase,pin,dependency=None):
    parent=ROOT/phase;parent.mkdir();a=argv(phase,pin,dependency)
    write(parent/'submission-started.json',{'status':'unknown_until_actual_receipt','argv':a})
    env={k:v for k,v in os.environ.items() if not re.fullmatch(r'CUDA_VISIBLE_DEVICES\d*',k) and not k.startswith('CUDA_MPS_')}
    r=captured_command(a,parent/'submission-raw.json',env=env)
    ids=re.findall(r'Job <([0-9]+)> is submitted',r.stdout);require(r.returncode==0 and len(ids)==1,'Unknown submission; do not replay')
    row={'status':'submitted','dispatch_state':'held','returncode':0,'job_id':ids[0],'argv':a,'dependency':dependency,'source_manifest_sha256':pin}
    write(parent/'submission.json',row)
    with (SCOPE/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);jobs=read(SCOPE/'jobs.json');name=parent.relative_to(SCOPE).as_posix()
        require(name not in jobs,'Existing/unknown job registry entry');jobs[name]=ids[0]
        tmp=SCOPE/'jobs.diagnostic.partial';write(tmp,jobs);tmp.replace(SCOPE/'jobs.json')
    release=release_argv(row,a,read(parent/'submission.json'),read(SCOPE/'jobs.json')[name])
    for attempt in range(12):
        state=captured_command(['bjobs','-a','-noheader','-o','jobid stat user job_name:160',row['job_id']],
                               parent/('held-scheduler-'+str(attempt).zfill(2)+'.json'),timeout=20)
        if held_observation(state.returncode,state.stdout,row['job_id'],a[a.index('-J')+1]):break
        if attempt<11:time.sleep(5)
    else:raise ValueError('Own accepted job did not reach held state; reconcile without retry')
    write(parent/'release-started.json',{'status':'unknown_until_actual_release_receipt','argv':release,
                                         'submission':row,'optimizer_jobs':[]})
    released=captured_command(release,parent/'release-raw.json',timeout=60)
    require(released.returncode==0,'Held release outcome unknown; reconcile, never resubmit')
    write(parent/'release.json',{'status':'released_exact_owned_held_job','job_id':row['job_id'],
                               'argv':release,'returncode':0,'receipt_persisted_before_release':True})
    return row
def storage(op,parent,phase):
    r=op.storage(parent,phase);cap=read(CAPACITY/'plan.json')['rolling_storage_policy']['control_cap_bytes']
    require(op.metadata_bytes()+128*1024**2<=cap,'Current funded control scope cannot fit complete128MiB diagnostic')
    return r
def execute(phase,pin):
    started=time.monotonic();op,plan=context(pin);parent=ROOT/phase;gpu=phase.startswith('gpu-')
    for _ in range(30):
        if (parent/'submission.json').exists():break
        time.sleep(1)
    own=read(parent/'submission.json');require(own['job_id']==os.environ.get('LSB_JOBID') and own['argv']==argv(phase,pin,own['dependency']),'Exact own allocation/source differs')
    require(os.environ.get('LSB_QUEUE')==('gpul40s' if gpu else 'hpc') and os.environ.get('LSB_DJOB_NUMPROC')==('8' if phase=='gpu-chemistry' else '1' if phase=='after-cpu' else '4'),'Actual diagnostic queue/slots differ')
    storage(op,parent,'entry');payload=parent/'payload';payload.mkdir()
    if phase=='after-cpu':
        sub=read(ROOT/'cpu/submission.json');text=' '.join((ROOT/'cpu'/('stdout.'+sub['job_id']+'.log')).read_text().split())
        require(text.count('Subject: Job '+sub['job_id']+':')==1 and 'queue <hpc>' in text and 'Successfully completed.' in text and 'Exited with exit code' not in text,'Actual same CPU LSF success required')
        rows=list(ET.parse(ROOT/'cpu/payload/tests.xml').getroot().iter('testcase'))
        require(len(rows)==30 and all(not list(r) for r in rows),'All30 allocated fixtures must pass')
        require(read(ROOT/'cpu/payload/analysis/assessment.json')['status']=='diagnostics_completed_original_failures_preserved','Saved-probe CPU analysis incomplete')
        startup=read(ROOT/'cpu/payload/startup/assessment.json')
        require(startup['status']=='passed_actual_original_pre_model_startups' and startup['model_constructed'] is False
                and startup['optimizer_updates']==0 and startup['can_admit_new_child'] is False
                and [r['arm'] for r in startup['startups']]==['plain','chemistry'],
                'Both actual original startup commands must pass before GPU submission')
        for row in startup['startups']:
            value=row['startup']
            require(value['status']=='passed_metadata_only' and value['numerical_setup_entered'] is False
                    and value['updates']==0 and value['job_id']==sub['job_id'], 'Actual CPU startup identity differs')
        references=startup['reference_metadata']
        require([r['arm'] for r in references]==['plain','chemistry'],'Both actual reference metadata paths required')
        for row in references:
            require(row['actual_reference_metadata_passed'] is True and row['model_constructed'] is False,
                    'Original reference metadata validation must pass without models')
            require(row['actual_worker_startup']['job_id']==sub['job_id']
                    and row['actual_worker_startup']['status']=='passed_metadata_only'
                    and row['actual_worker_startup']['numerical_setup_entered'] is False
                    and row['historical_receipt_preserved'] is True,
                    'Both actual original reference CLI startups must pass without historical writes')
            request=argv('gpu-'+row['arm'],pin)
            require(int(request[request.index('-n')+1])==row['native_GPU_CPU_slots'],
                    'Proposed GPU allocation differs from unchanged actual reference worker predicate')
        storage(op,parent,'dispatch')
        gpus={kind:submit('gpu-'+kind,pin) for kind in ('plain','chemistry')}
        write(payload/'assessment.json',{'status':'CPU30_actual_startups_reference_metadata_passed_GPU_diagnostics_submitted',
              'gpus':gpus,'optimizer_jobs':[]});return
    if phase=='cpu':
        command=[PYTHON,'-m','pytest',str(SOURCE/'tests'),'-q','--rootdir',str(SOURCE),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
                 '--basetemp',str(Path(os.environ['TMPDIR'])/'boundary-diagnostics'),'--junitxml='+str(payload/'tests.xml')]
        with (payload/'tests.log').open('x') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        startup=[PYTHON,str(SOURCE/'scripts/continuation_failure_diagnostics.py'),'--plan',str(SOURCE/'plan.json'),
                 '--pin',op.sha(SOURCE/'plan.json'),'--mode','startup','--output',str(payload/'startup')]
        write(payload/'actual-startup-command.json',{'argv':startup,'model_execution_authorized':False})
        with (payload/'startup.log').open('x') as f:subprocess.run(startup,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    else:
        routing_path=(CAPACITY/'scripts/cuda_ownership.py' if phase=='gpu-plain'
                      else Path(plan['chemistry_source']['path'])/'scripts/chemistry_cuda_ownership.py')
        routing=module(routing_path,'_same_qualified_role_specific_cuda_routing')
        routing.run_probe(parent/'cuda-ownership',submission=parent/'submission.json',expected_job_id=own['job_id'],
            input_bindings={'source_manifest':op.binding(SOURCE/'source_manifest.json'),'plan':op.binding(SOURCE/'plan.json'),'context_identity':plan['context_ready']},deadline=started+600)
        require(time.monotonic()-started<600,'Complete diagnostic entry/smoke budget exhausted')
    mode=phase if gpu else 'cpu';target=payload/('replay' if gpu else 'analysis')
    command=[PYTHON,str(SOURCE/'scripts/continuation_failure_diagnostics.py'),'--plan',str(SOURCE/'plan.json'),'--pin',op.sha(SOURCE/'plan.json'),'--mode',mode,'--output',str(target)]
    with (payload/'execution.log').open('x') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=2700 if gpu else 600)
    context(pin);require(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2,'Complete diagnostic tree cap exceeded')
    write(payload/'preservation.json',{'status':'inputs_and_sources_preserved','optimizer_updates':0,'can_certify_or_restart_original_study':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('cpu','after-cpu','gpu-plain','gpu-chemistry'));p.add_argument('--pin',required=True);a=p.parse_args();execute(a.phase,a.pin)
