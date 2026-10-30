"""New single-arm production owner; inherited science and capacity stay sealed."""
import argparse
import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SCOPE=BASE/'nonchem-precision-continuation-26'
ROOT=SCOPE/'runtime-production-05'
SOURCE=BASE/'runtime-production-release-05/source'
QUALIFIED=SCOPE/'runtime-qualification-release-01/source'
QP='9108a418953a4ee7f4d481667ac4a5035402c96f10179f1e8a44c3358af965ab'


def load(path,name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


# Wrapper authenticates own and every parent file BEFORE this import.
q=load(QUALIFIED/'scripts/failure_diagnostic_ops.py','_qualified_production_protocol')
q.ROOT=ROOT;q.SOURCE=SOURCE
require=q.require;read=q.read;write=q.write;module=q.module;sha=q.sha
PYTHON=q.PYTHON;CAPACITY=q.CAPACITY;CAPACITY_PIN=q.CAPACITY_PIN
submit=q.submit;terminal=q.terminal


def runtime_adapter(plan):
    adapter=module(CAPACITY/'scripts/precision_continuation_driver.py','_production_original_verifier')
    from runtime_production_driver import validation_lookup
    validation_lookup(adapter,plan['qualified']['summary']['path'])
    return adapter


def verify_qualified(op,plan):
    value=plan['qualified']
    for name in ('ready','summary','result','capture','native_lsf','capture_lsf','cpu_certificate','preservation'):
        op.check_binding(value[name])
    q.terminal(Path(value['native_parent']),value['native_job_id'],'gpul40s')
    q.terminal(Path(value['capture_parent']),value['capture_job_id'],'hpc')
    accepted=read(value['capture']['path'])
    require(accepted['status']=='passed_full_validation_pending_separate_production_integration'
        and accepted['validation_job_id']==value['native_job_id'] and accepted['source_manifest_sha256']==QP
        and accepted['execution_profile']==plan['execution_profile'] and accepted['can_submit_optimizer'] is False,
        'Exact complete new runtime capture required')
    require(read(value['cpu_certificate']['path'])['cases']==1029,'Full runtime qualification CPU certificate required')
    require(read(value['preservation']['path'])['status']=='inputs_and_sources_preserved','Qualification preservation missing')
    original=read(CAPACITY/'plan.json');original['_sha256']=sha(CAPACITY/'plan.json');original['_manifest']=CAPACITY_PIN
    row=next(v for v in original['arms'] if v['id']=='protenix');result=read(value['result']['path'])
    runtime_adapter(plan).verify_precision_result(result,original,CAPACITY_PIN,'validate','protenix',20000,row['parent_snapshot'])
    require(result['job_id']==value['native_job_id'] and result['original_summary']==value['summary'],
            'New runtime validation producer/summary differs')
    require(result['precision_runtime_budget']==plan['runtime_budget'] and plan['runtime_budget']['requested_updates']==30000
        and plan['runtime_budget']['walltime_seconds']==19800 and plan['runtime_budget']['limit_seconds']==19200,
        'Measured largest whole remainder runtime differs')
    return result


def context(pin):
    q.authenticate_capacity();op=module(CAPACITY/'scripts/precision_continuation_ops.py','_production_capacity_owner')
    original=op.verify(CAPACITY,CAPACITY_PIN);op.certificate(CAPACITY_PIN,original)
    manifest=op.verify(SOURCE,pin);plan=read(SOURCE/'plan.json')
    sys.path.insert(0,str(Path(plan['chemistry_source']['path'])/'scripts'))
    require(plan['run_root']==str(ROOT) and plan['root']==str(SCOPE) and plan['source_root']==str(SOURCE),
            'Exact production namespace required')
    for row in manifest['parent_sources']:op.verify(Path(row['path']),row['manifest_sha256'])
    for proof in plan['inputs']:op.check_binding(proof)
    verify_qualified(op,plan)
    old=plan['failed_production_predecessor']
    for row in old['proofs']:op.check_binding(row)
    for phase,jid in old['jobs'].items():
        text=''.join((Path(old['root'])/phase/('stdout.'+jid+'.log')).read_text().split())
        require(text.count('Subject:Job'+jid+':')==1 and text.count('Exitedwithexitcode1.')==1,'Exact failed prior production proof required')
    case=Path(old['case']);states=(case/'states.jsonl').read_bytes()
    require(states.endswith(b'\n') and len(states.splitlines())==1 and json.loads(states)['global_step']==20000
        and not (case/'updates.jsonl').exists(),'Prior child unexpectedly made scientific progress')
    return op,plan


def argv(phase,pin,dependency=None):
    require(phase in ('cpu','after-cpu','gpu-plain','after-plain'),'Only one declared production arm')
    gpu=phase=='gpu-plain';slots=1 if phase=='after-cpu' else 4;parent=ROOT/phase
    result=['bsub','-H','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
        '-q','gpul40s' if gpu else 'hpc','-n',str(slots),'-R','span[hosts=1]','-R',
        'rusage[mem='+('8GB' if gpu else '2GB' if slots==1 else '4GB')+']','-W','05:30' if gpu else '00:30' if slots==1 else '01:00',
        '-J','mosaic-runtime-production05-'+phase,'-o',str(parent/'stdout.%J.log'),'-e',str(parent/'stderr.%J.log')]
    if gpu:result+=['-gpu','num=1:mode=exclusive_process:mps=yes']
    if dependency:result+=['-w',dependency]
    return result+['bash','-l',str(SOURCE/'scripts/runtime_production_stage.lsf'),str(SOURCE),pin,phase,'--capacity-source',str(CAPACITY)]


def ready_envelope(op,plan,pin):
    proofs=[plan['qualified'][name] for name in ('ready','summary','result','capture','native_lsf','capture_lsf','cpu_certificate','preservation')]
    return {'status':'passed','arm':'protenix','offset':20000,'source_manifest_sha256':pin,
        'plan_sha256':sha(SOURCE/'plan.json'),'proofs':proofs,'original_failures_preserved':True,
        'execution_profile':plan['execution_profile'],'qualified_result':plan['qualified']['result']}


def storage(op,parent,phase):
    plan=read(SOURCE/'plan.json');pin=sha(SOURCE/'source_manifest.json')
    from runtime_production_capacity import proxy,configure
    result=verify_qualified(op,plan);owner=proxy(op,plan,SOURCE,ROOT,ready_envelope(op,plan,pin),result)
    rolling=module(CAPACITY/'scripts/rolling_capacity.py','_new_production_complete_rolling_capacity')
    configure(rolling,owner,plan)
    import storage_policy_adapter
    storage_policy_adapter.install(rolling,plan,parent)
    return rolling.storage(owner,parent,phase)


def original_result(op,plan,parent,jid):
    study=parent/'payload/replay/plain/payload/study';result=read(study.parent/'assessment.json')
    original=read(CAPACITY/'plan.json');original['_sha256']=sha(CAPACITY/'plan.json');original['_manifest']=CAPACITY_PIN
    snapshot=next(v for v in original['arms'] if v['id']=='protenix')['parent_snapshot']
    runtime_adapter(plan).verify_precision_result(result,original,CAPACITY_PIN,'segment','protenix',20000,snapshot)
    require(result['job_id']==jid and result['global_end']==50000 and result['frozen_unchanged'] is True
        and result['original_results_changed'] is False,'Original full segment identity differs')
    op.check_binding(result['terminal_snapshot'])
    observation=read(study.parent.parent/'runtime-production-assessment.json')
    require(observation['status']=='passed_original_segment_pending_independent_capture'
        and observation['execution_profile']==plan['execution_profile'],'New runtime execution receipt absent')
    return result


def execute(phase,pin):
    started=time.monotonic();op,plan=context(pin);parent=ROOT/phase;own=read(parent/'submission.json')
    require(own['job_id']==os.environ.get('LSB_JOBID') and own['argv']==argv(phase,pin,own['dependency']),
            'Actual own command/allocation differs')
    require(os.environ.get('LSB_QUEUE')==('gpul40s' if phase=='gpu-plain' else 'hpc')
        and os.environ.get('LSB_DJOB_NUMPROC')==('1' if phase=='after-cpu' else '4'),'Actual queue/slots differ')
    storage(op,parent,'entry');payload=parent/'payload';payload.mkdir()
    if phase=='cpu':
        helper=q.helper(plan);helper.probe_runtime_flag(payload/'candidate-runtime-probe.json')
        command=[PYTHON,str(SOURCE/'scripts/runtime_production_driver.py'),'--pin',pin,'--mode','startup',
                 '--output',str(payload/'startup/plain')]
        (payload/'startup').mkdir();write(payload/'actual-startup-command.json',{'argv':command,'models_authorized':False})
        with (payload/'startup.log').open('x') as stream:subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        from runtime_production_driver import check_segment_startup
        check_segment_startup(read(payload/'startup/plain/actual-startup.json'),plan['qualified']['ready'],own['job_id'])
        # Execute the unchanged original reference startup body, restricted to plain Protenix.
        tree=ast.parse(inspect.getsource(helper.premodel));fn=tree.body[0]
        fn.body=[n for n in fn.body if not (isinstance(n,ast.For) and isinstance(n.target,ast.Tuple))]
        loops=[n for n in fn.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='kind']
        require(len(loops)==1,'One original reference startup loop required');loops[0].iter=ast.Tuple([ast.Constant('plain')],ast.Load())
        namespace=dict(vars(helper));exec(compile(ast.fix_missing_locations(tree),helper.__file__,'exec'),namespace)
        namespace['premodel'](plan,payload/'reference-startup')
        count=q.fixture_suite(plan,payload)
        write(payload/'assessment.json',{'status':'passed','cases':count,'job_id':own['job_id'],'source_manifest_sha256':pin});return
    if phase=='after-cpu':
        cpu=read(ROOT/'cpu/submission.json');lsf=terminal(ROOT/'cpu',cpu['job_id'],'hpc')
        inherited=module(Path(plan['chemistry_source']['path'])/'scripts/chemistry_continuation_ops.py','_production_XML')
        count=inherited.xml(ROOT/'cpu/payload/tests.xml',read(SOURCE/'source_manifest.json')['all_cpu_counts'])
        require(read(ROOT/'cpu/payload/assessment.json')['cases']==count,'Actual CPU case count differs')
        startup=read(ROOT/'cpu/payload/startup/plain/actual-startup.json')
        from runtime_production_driver import check_segment_startup
        check_segment_startup(startup,plan['qualified']['ready'],cpu['job_id'])
        write(ROOT/'cpu-certificate.json',{'status':'passed','cases':count,'source_manifest_sha256':pin,
            'job_id':cpu['job_id'],'same_job_LSF':lsf,'actual_segment_startup':op.binding(ROOT/'cpu/payload/startup/plain/actual-startup.json')})
        write(ROOT/'qualified-ready.json',ready_envelope(op,plan,pin))
        import fcntl
        with (SCOPE/'dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);storage(op,parent,'dispatch')
            gpu=submit('gpu-plain',pin);callback=submit('after-plain',pin,'ended('+gpu['job_id']+')')
            write(payload/'assessment.json',{'status':'qualified_full_remainder_submitted','native':gpu,'callback':callback})
        return
    if phase=='after-plain':
        gpu=ROOT/'gpu-plain';sub=read(gpu/'submission.json');lsf=terminal(gpu,sub['job_id'],'gpul40s')
        result=original_result(op,plan,gpu,sub['job_id'])
        require(read(gpu/'payload/preservation.json')['status']=='inputs_and_sources_preserved','New source/input post-check missing')
        certificate={'status':'passed','arm':'protenix','offset':20000,'global_end':50000,'source_manifest_sha256':pin,
            'plan_sha256':sha(SOURCE/'plan.json'),'execution_profile':plan['execution_profile'],
            'original_failures_preserved':True,'original_results_changed':False,
            'proofs':[lsf,op.binding(gpu/'payload/preservation.json'),*result['proofs']]}
        write(ROOT/'completion-certificate.json',certificate);write(payload/'assessment.json',certificate);return
    require(read(ROOT/'cpu-certificate.json')['source_manifest_sha256']==pin,'New allocated CPU pass required')
    helper=q.helper(plan);helper.guard_candidate_environment(os.environ)
    routing=module(CAPACITY/'scripts/cuda_ownership.py','_new_production_CUDA_ownership')
    routing.run_probe(parent/'cuda-ownership',submission=parent/'submission.json',expected_job_id=own['job_id'],
        input_bindings={'source_manifest':op.binding(SOURCE/'source_manifest.json'),'plan':op.binding(SOURCE/'plan.json'),
            'context_identity':plan['context_ready']},deadline=started+600)
    chemical=read(Path(plan['chemistry_source']['path'])/'plan.json')
    pilot=module(Path(chemical['arms']['protenix']['adapter_source']['path'])/'scripts/full_model_pilot.py','_production_owned_cleanup')
    claim=read(ROOT/'storage-claims/protenix-segment-20000.json');pilot.OUTPUT_CAP=claim['cap_bytes']+2*1024**3
    old_inventory=pilot.output_bytes;next_gate=time.monotonic()+300
    def inventory(path):
        nonlocal next_gate
        if time.monotonic()>=next_gate:storage(op,parent,'periodic');next_gate=time.monotonic()+300
        return old_inventory(path)
    pilot.output_bytes=inventory
    target=payload/'replay';target.mkdir()
    command=[PYTHON,str(SOURCE/'scripts/runtime_production_driver.py'),'--pin',pin,'--mode','segment','--output',str(target/'plain')]
    write(parent/'driver-invocation.json',{'argv':command,'runtime_budget':plan['runtime_budget'],'global_start':20000,'global_end':50000})
    remaining=plan['runtime_budget']['limit_seconds']-(time.monotonic()-started)
    require(remaining>0,'Measured full entry/child deadline exhausted')
    with (payload/'execution.log').open('x') as stream:
        with pilot._parent_sigterm_cleanup():execution=pilot.run_isolated_child(command,SOURCE,dict(os.environ),stream,remaining,ROOT)
    write(payload/'execution.json',execution);context(pin);original_result(op,plan,parent,own['job_id'])
    write(payload/'preservation.json',{'status':'inputs_and_sources_preserved','original_results_changed':False})


q.argv=argv;q.context=context;q.storage=storage
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=('cpu','after-cpu','gpu-plain','after-plain'))
    parser.add_argument('--pin',required=True);args=parser.parse_args();execute(args.phase,args.pin)
