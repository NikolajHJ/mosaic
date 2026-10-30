"""Read-only diagnostic orchestration; no optimizer/continuation branch."""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import inspect
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
ROOT=SCOPE/'chemistry-continuation-05/runtime-qualification-01'
SOURCE=SCOPE/'runtime-qualification-release-01/source'
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
    sys.path.insert(0,str(Path(plan['chemistry_source']['path'])/'scripts'))
    for row in own['parent_sources']:op.verify(Path(row['path']),row['manifest_sha256'])
    for b in plan['inputs']:op.check_binding(b)
    return op,plan
def argv(phase,pin,dependency=None):
    require(phase in ('cpu','after-cpu','gpu-plain','gpu-chemistry','after-plain','after-chemistry'),'Unknown diagnostic phase')
    gpu=phase.startswith('gpu-');slots=8 if phase=='gpu-chemistry' else 4 if phase.startswith('after-') and phase!='after-cpu' else 1 if phase=='after-cpu' else 4;p=ROOT/phase
    a=['bsub','-H','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
       '-q','gpul40s' if gpu else 'hpc','-n',str(slots),'-R','span[hosts=1]','-R',
       'rusage[mem='+('8GB' if phase=='gpu-plain' else '2GB' if slots==1 else '4GB')+']','-W','02:00' if gpu else '00:30' if slots==1 else '01:00',
       '-J','mosaic-runtime-qualification01-'+phase,'-o',str(p/'stdout.%J.log'),'-e',str(p/'stderr.%J.log')]
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
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def owned_bytes(path):
    total=0
    for item in Path(path).rglob('*'):
        require(not item.is_symlink() and (item.is_file() or item.is_dir()),'Special qualification output')
        if item.is_file():total+=item.stat().st_size
    return total
def storage(op,parent,phase):
    result=op.storage(parent,phase)
    cap=read(CAPACITY/'plan.json')['rolling_storage_policy']['control_cap_bytes']
    own=owned_bytes(ROOT)
    require(own<512*1024**2 and op.metadata_bytes()+512*1024**2-own<=cap,
            'Complete qualification allowance cannot fit original shared control cap')
    return result

def terminal(parent, job, queue):
    text=' '.join((parent/('stdout.'+job+'.log')).read_text().split())
    require(text.count('Subject: Job '+job+':')==1 and 'queue <'+queue+'>' in text
            and text.count('Successfully completed.')==1 and 'Exited with exit code' not in text,
            'Actual same-job successful LSF required')
    return {'path':str(parent/('stdout.'+job+'.log')),'sha256':sha(parent/('stdout.'+job+'.log'))}

def fixture_suite(plan, payload):
    main=read(CAPACITY/'source_manifest.json')['cpu_counts']
    chem=Path(plan['chemistry_source']['path']);chem_counts=read(chem/'source_manifest.json')['cpu_counts']
    own_counts=read(SOURCE/'source_manifest.json')['cpu_counts']
    groups=[('torch','/work3/s234818/latentmol/env/boltz311/bin/python',CAPACITY,{'tests.test_dbmol_reference':main['tests.test_dbmol_reference']}),
            ('jax',PYTHON,CAPACITY,{k:v for k,v in main.items() if k!='tests.test_dbmol_reference'}),
            ('chemistry',PYTHON,chem,chem_counts),('qualification',PYTHON,SOURCE,own_counts)]
    merged=ET.Element('testsuites')
    inherited=module(chem/'scripts/chemistry_continuation_ops.py','_qualified_inherited_XML_validator')
    for name,python,source,counts in groups:
        command=[python,'-m','pytest',*[str(source/(n.replace('.','/')+'.py')) for n in counts],'-q',
            '--rootdir',str(source),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
            '--basetemp',str(Path(os.environ['TMPDIR'])/('runtime-qualification-'+name)),
            '--junitxml='+str(payload/(name+'.xml'))]
        write(payload/(name+'-command.json'),{'argv':command,'counts':counts})
        environment=dict(os.environ);environment['PYTHONPATH']=str(source/'scripts')
        with (payload/(name+'.log')).open('x') as stream:
            run=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
        require(run.returncode==0,'Full inherited/qualification CPU suite failed: '+name)
        inherited.xml(payload/(name+'.xml'),counts)
        root=ET.parse(payload/(name+'.xml')).getroot();merged.extend([root] if root.tag=='testsuite' else list(root))
    ET.ElementTree(merged).write(payload/'tests.xml',encoding='utf8',xml_declaration=True)
    expected={**main,**chem_counts,**own_counts}
    return inherited.xml(payload/'tests.xml',expected)

def helper(plan):
    path=Path(plan['helper_source']['path'])/'scripts/continuation_failure_diagnostics.py'
    return module(path,'_qualified_runtime_diagnostic_helpers')

def native_budget(elapsed):
    require(type(elapsed) in (int,float) and 0 <= elapsed < 6600,
            'Complete entry/validation deadline exhausted')
    return max(1,int(6600-elapsed))

def capture_original_validation(plan, kind, directory, pin):
    """Allocated CPU metadata/boundary certification; no optimizer submission."""
    sub=read(directory/'submission.json');lsf=terminal(directory,sub['job_id'],'gpul40s')
    output=directory/'payload/replay'/kind/'payload/study'
    observation=read(directory/'payload/replay'/kind/'qualification-assessment.json')
    require(observation['status']=='passed_complete_original_validation_pending_independent_capture'
            and observation['can_submit_optimizer'] is False and observation['optimizer_continuation_updates']==0,
            'Complete original validation failed or attempted production')
    require(observation['execution_profile']==plan['execution_profile'],'Execution profile differs')
    require(read(directory/'payload/preservation.json')['status']=='inputs_and_sources_preserved',
            'Original sources/inputs post-execution preservation missing')
    if kind=='plain':
        original=read(CAPACITY/'plan.json');original['_sha256']=sha(CAPACITY/'plan.json');original['_manifest']=CAPACITY_PIN
        adapter=module(CAPACITY/'scripts/precision_continuation_driver.py','_qualified_full_plain_verifier')
        result=read(output.parent/'assessment.json')
        row=next(v for v in original['arms'] if v['id']=='protenix')
        adapter.verify_precision_result(result,original,CAPACITY_PIN,'validate','protenix',20000,row['parent_snapshot'])
        require(result['job_id']==sub['job_id'],'Original producer belongs to another GPU job')
        ready={'readiness':{'path':str(output.parent/'assessment.json'),'sha256':sha(output.parent/'assessment.json')},
               'runtime_budget':result['precision_runtime_budget']}
    else:
        chem=Path(plan['chemistry_source']['path'])
        inherited=module(chem/'scripts/chemistry_continuation_ops.py','_qualified_full_chemistry_verifier')
        original=inherited.driver.authenticated_plan(chem/'plan.json',sha(chem/'plan.json'))
        # Preserve the entire original accept function. Only its output destination
        # is redirected to this NEW owner; every numerical/boundary/readiness check stays.
        source=inspect.getsource(inherited.accept)
        tree=ast.parse(source);changed=0
        for node in ast.walk(tree):
            if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write'
                    and len(node.args)==2 and ast.unparse(node.args[0])=="c.ROOT / (arm + '-ready.json')"):
                node.func=ast.Name('_new_qualification_ready',ast.Load());node.args=[node.args[1]];changed+=1
        require(changed==1,'Original chemistry readiness write shape differs')
        namespace=dict(inherited.__dict__)
        namespace['parent']=lambda phase,arm:directory
        namespace['study']=lambda phase,arm:output
        namespace['_new_qualification_ready']=lambda value:write(ROOT/'chemistry-original-ready.json',value)
        exec(compile(ast.fix_missing_locations(tree),str(chem/'scripts/chemistry_continuation_ops.py'),'exec'),namespace)
        ready=namespace['accept'](original,'protenix','validate',pin)
    return {'status':'passed_full_validation_pending_separate_production_integration','arm':kind,
        'source_manifest_sha256':pin,'plan_sha256':sha(SOURCE/'plan.json'),'validation_job_id':sub['job_id'],
        'original_validation':ready,'execution_profile':plan['execution_profile'],
        'same_job_LSF':lsf,'optimizer_continuation_updates':0,'can_submit_optimizer':False,
        'original_results_changed':False,'original_failures_preserved':True}

def execute(phase,pin):
    started=time.monotonic();op,plan=context(pin);parent=ROOT/phase;gpu=phase.startswith('gpu-')
    own=read(parent/'submission.json')
    require(own['job_id']==os.environ.get('LSB_JOBID') and own['argv']==argv(phase,pin,own['dependency']),
            'Actual own qualification allocation differs')
    slots=8 if phase=='gpu-chemistry' else 1 if phase=='after-cpu' else 4
    require(os.environ.get('LSB_QUEUE')==('gpul40s' if gpu else 'hpc')
            and os.environ.get('LSB_DJOB_NUMPROC')==str(slots),'Actual queue/slots differ')
    storage(op,parent,'entry');payload=parent/'payload';payload.mkdir()
    if phase=='cpu':
        count=fixture_suite(plan,payload)
        h=helper(plan);h.probe_runtime_flag(payload/'candidate-runtime-probe.json')
        command=[PYTHON,str(SOURCE/'scripts/continuation_failure_diagnostics.py'),'--plan',str(SOURCE/'plan.json'),
                 '--pin',sha(SOURCE/'plan.json'),'--mode','startup','--output',str(payload/'startup')]
        write(payload/'actual-startup-command.json',{'argv':command,'models_authorized':False})
        with (payload/'startup.log').open('x') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        write(payload/'assessment.json',{'status':'passed','cases':count,'job_id':own['job_id'],
            'source_manifest_sha256':pin,'can_submit_optimizer':False});return
    if phase=='after-cpu':
        cpu=read(ROOT/'cpu/submission.json');lsf=terminal(ROOT/'cpu',cpu['job_id'],'hpc')
        counts=read(SOURCE/'source_manifest.json')['all_cpu_counts']
        inherited=module(Path(plan['chemistry_source']['path'])/'scripts/chemistry_continuation_ops.py','_full_XML_capture')
        count=inherited.xml(ROOT/'cpu/payload/tests.xml',counts)
        require(read(ROOT/'cpu/payload/assessment.json')['cases']==count,'Allocated CPU assessment differs')
        startup=read(ROOT/'cpu/payload/startup/assessment.json')
        require(startup['status']=='passed_actual_full_validation_startups' and startup['model_constructed'] is False
                and [r['arm'] for r in startup['startups']]==['plain','chemistry']
                and [r['arm'] for r in startup['reference_metadata']]==['plain','chemistry'], 'Actual full CLI startups incomplete')
        write(ROOT/'cpu-certificate.json',{'status':'passed','source_manifest_sha256':pin,'plan_sha256':sha(SOURCE/'plan.json'),
            'cases':count,'job_id':cpu['job_id'],'same_job_LSF':lsf})
        storage(op,parent,'dispatch')
        jobs={}
        for kind in plan['qualification_arms']:
            native=submit('gpu-'+kind,pin);callback=submit('after-'+kind,pin,'ended('+native['job_id']+')')
            jobs[kind]={'native':native,'callback':callback}
        write(payload/'assessment.json',{'status':'full_CPU_and_startups_passed_validation_GPUs_submitted',
            'jobs':jobs,'optimizer_jobs':[]});return
    if phase.startswith('after-'):
        kind=phase.removeprefix('after-');result=capture_original_validation(plan,kind,ROOT/('gpu-'+kind),pin)
        write(payload/'assessment.json',result);return
    kind=phase.removeprefix('gpu-')
    require(read(ROOT/'cpu-certificate.json')['source_manifest_sha256']==pin,'New CPU certificate required')
    h=helper(plan);h.guard_candidate_environment(os.environ)
    routing=module((CAPACITY/'scripts/cuda_ownership.py' if kind=='plain' else
        Path(plan['chemistry_source']['path'])/'scripts/chemistry_cuda_ownership.py'),'_original_full_qualification_CUDA_ownership')
    routing.run_probe(parent/'cuda-ownership',submission=parent/'submission.json',expected_job_id=own['job_id'],
        input_bindings={'source_manifest':op.binding(SOURCE/'source_manifest.json'),'plan':op.binding(SOURCE/'plan.json'),
                        'context_identity':plan['context_ready']},deadline=started+600)
    chemistry=read(Path(plan['chemistry_source']['path'])/'plan.json')
    adapter=Path(chemistry['arms']['protenix']['adapter_source']['path'])
    pilot=module(adapter/'scripts/full_model_pilot.py','_original_owned_full_validation_cleanup');pilot.OUTPUT_CAP=512*1024**2
    target=payload/'replay';target.mkdir()
    command=[PYTHON,str(SOURCE/'scripts/continuation_failure_diagnostics.py'),'--plan',str(SOURCE/'plan.json'),
             '--pin',sha(SOURCE/'plan.json'),'--mode',kind,'--output',str(target/kind)]
    write(payload/'actual-native-command.json',{'argv':command,'phase':'validate','optimizer_continuation_updates':0})
    with (payload/'execution.log').open('x') as stream:
        with pilot._parent_sigterm_cleanup():
            execution=pilot.run_isolated_child(command,SOURCE,dict(os.environ),stream,
                native_budget(time.monotonic()-started),ROOT)
        write(payload/'execution.json',execution)
    context(pin);require(owned_bytes(ROOT)<512*1024**2,'Complete qualification cap exceeded')
    write(payload/'preservation.json',{'status':'inputs_and_sources_preserved','optimizer_continuation_updates':0,
        'can_submit_optimizer':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',required=True,choices=('cpu','after-cpu','gpu-plain','gpu-chemistry','after-plain','after-chemistry'))
    parser.add_argument('--pin',required=True)
    args=parser.parse_args();execute(args.phase,args.pin)
