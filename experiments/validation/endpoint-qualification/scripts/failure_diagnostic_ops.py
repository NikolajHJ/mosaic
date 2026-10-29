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
ROOT=SCOPE/'boundary-diagnostics-01'
SOURCE=SCOPE/'boundary-diagnostics-release-01/source'
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
    require(phase in ('cpu','after-cpu','gpu'),'Unknown diagnostic phase')
    gpu=phase=='gpu';slots=1 if phase=='after-cpu' else 4;p=ROOT/phase
    a=['bsub','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
       '-q','gpul40s' if gpu else 'hpc','-n',str(slots),'-R','span[hosts=1]','-R',
       'rusage[mem='+('8GB' if gpu else '2GB' if slots==1 else '4GB')+']','-W','02:00' if gpu else '00:20' if slots==1 else '01:00',
       '-J','mosaic-boundary-diagnostics01-'+phase,'-o',str(p/'stdout.%J.log'),'-e',str(p/'stderr.%J.log')]
    if gpu:a+=['-gpu','num=1:mode=exclusive_process:mps=yes']
    if dependency:a+=['-w',dependency]
    return a+['bash','-l',str(SOURCE/'scripts/failure_diagnostic_stage.lsf'),str(SOURCE),pin,phase,'--capacity-source',str(CAPACITY)]
def submit(phase,pin,dependency=None):
    parent=ROOT/phase;parent.mkdir();a=argv(phase,pin,dependency)
    write(parent/'submission-started.json',{'status':'unknown_until_actual_receipt','argv':a})
    env={k:v for k,v in os.environ.items() if not re.fullmatch(r'CUDA_VISIBLE_DEVICES\d*',k) and not k.startswith('CUDA_MPS_')}
    r=subprocess.run(a,capture_output=True,text=True,timeout=90,env=env)
    write(parent/'submission-raw.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    ids=re.findall(r'Job <([0-9]+)> is submitted',r.stdout);require(r.returncode==0 and len(ids)==1,'Unknown submission; do not replay')
    row={'status':'submitted','returncode':0,'job_id':ids[0],'argv':a,'dependency':dependency,'source_manifest_sha256':pin}
    write(parent/'submission.json',row)
    with (SCOPE/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);jobs=read(SCOPE/'jobs.json');name=parent.relative_to(SCOPE).as_posix()
        require(name not in jobs,'Existing/unknown job registry entry');jobs[name]=ids[0]
        tmp=SCOPE/'jobs.diagnostic.partial';write(tmp,jobs);tmp.replace(SCOPE/'jobs.json')
    return row
def storage(op,parent,phase):
    r=op.storage(parent,phase);cap=read(CAPACITY/'plan.json')['rolling_storage_policy']['control_cap_bytes']
    require(op.metadata_bytes()+128*1024**2<=cap,'Current funded control scope cannot fit complete128MiB diagnostic')
    return r
def execute(phase,pin):
    op,plan=context(pin);parent=ROOT/phase
    for _ in range(30):
        if (parent/'submission.json').exists():break
        time.sleep(1)
    own=read(parent/'submission.json');require(own['job_id']==os.environ.get('LSB_JOBID') and own['argv']==argv(phase,pin,own['dependency']),'Exact own allocation/source differs')
    require(os.environ.get('LSB_QUEUE')==('gpul40s' if phase=='gpu' else 'hpc') and os.environ.get('LSB_DJOB_NUMPROC')==('1' if phase=='after-cpu' else '4'),'Actual diagnostic queue/slots differ')
    storage(op,parent,'entry');payload=parent/'payload';payload.mkdir()
    if phase=='after-cpu':
        sub=read(ROOT/'cpu/submission.json');text=' '.join((ROOT/'cpu'/('stdout.'+sub['job_id']+'.log')).read_text().split())
        require(text.count('Subject: Job '+sub['job_id']+':')==1 and 'queue <hpc>' in text and 'Successfully completed.' in text and 'Exited with exit code' not in text,'Actual same CPU LSF success required')
        rows=list(ET.parse(ROOT/'cpu/payload/tests.xml').getroot().iter('testcase'))
        require(len(rows)==8 and all(not list(r) for r in rows),'All8 allocated fixtures must pass')
        require(read(ROOT/'cpu/payload/analysis/assessment.json')['status']=='diagnostics_completed_original_failures_preserved','Saved-probe CPU analysis incomplete')
        storage(op,parent,'dispatch');gpu=submit('gpu',pin)
        write(payload/'assessment.json',{'status':'CPU8_and_saved_analysis_passed_GPU_diagnostic_submitted','gpu':gpu,'optimizer_jobs':[]});return
    if phase=='cpu':
        command=[PYTHON,'-m','pytest',str(SOURCE/'tests'),'-q','--rootdir',str(SOURCE),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
                 '--basetemp',str(Path(os.environ['TMPDIR'])/'boundary-diagnostics'),'--junitxml='+str(payload/'tests.xml')]
        with (payload/'tests.log').open('x') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    else:
        routing=module(CAPACITY/'scripts/cuda_ownership.py','_same_qualified_cuda_routing')
        routing.run_probe(parent/'cuda-ownership',submission=parent/'submission.json',expected_job_id=own['job_id'],
            input_bindings={'source_manifest':op.binding(SOURCE/'source_manifest.json'),'plan':op.binding(SOURCE/'plan.json'),'context_identity':plan['context_ready']},deadline=time.monotonic()+600)
    mode='gpu' if phase=='gpu' else 'cpu';target=payload/('replay' if phase=='gpu' else 'analysis')
    command=[PYTHON,str(SOURCE/'scripts/continuation_failure_diagnostics.py'),'--plan',str(SOURCE/'plan.json'),'--pin',op.sha(SOURCE/'plan.json'),'--mode',mode,'--output',str(target)]
    with (payload/'execution.log').open('x') as f:subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,timeout=5600 if phase=='gpu' else 600)
    context(pin);require(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())<128*1024**2,'Complete diagnostic tree cap exceeded')
    write(payload/'preservation.json',{'status':'inputs_and_sources_preserved','optimizer_updates':0,'can_certify_or_restart_original_study':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('cpu','after-cpu','gpu'));p.add_argument('--pin',required=True);a=p.parse_args();execute(a.phase,a.pin)
