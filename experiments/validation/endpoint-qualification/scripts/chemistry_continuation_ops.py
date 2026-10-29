"""Conditional complete chemistry qualification and continuation orchestration.

Metadata only here. Numerical checks/optimization execute in distinct allocated
children. Every submission has an exclusive unknown marker and actual receipt;
callbacks never turn a component-only result into continuation readiness.
"""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import chemistry_continuation_contract as c
import chemistry_continuation_driver as driver

PREVIOUS_ROOT=c.BASE/'nonchem-precision-continuation-25/chemistry-continuation-02'
PREVIOUS_SOURCE=c.BASE/'nonchem-precision-continuation-25/chemistry-continuation-release-02/source'
PREVIOUS_PIN='d87701b3eb422d3e1c1f9255eb61607011b96eb621a67d286ceefe961e36c250'


def module(path,name):return driver.module(path,name)


def scope():
    op=module(c.CURRENT/'scripts/precision_continuation_ops.py','_chemistry_current_capacity')
    manifest=op.verify(c.CURRENT,c.CURRENT_PIN);op.certificate(c.CURRENT_PIN,manifest)
    return op,manifest


def own_source(pin):
    op,prior=scope();manifest=op.verify(c.SOURCE,pin)
    plan=c.validate_plan(c.read(c.SOURCE/'plan.json'))
    for row in manifest['parent_sources']:op.verify(Path(row['path']),row['manifest_sha256'])
    for row in plan['protected_inputs']:c.bound(row)
    return manifest,plan


def parent(phase,arm='all'):return c.ROOT/c.label(phase,arm)


def study(phase,arm):
    root=c.OUTPUTS if phase=='segment' else c.ROOT
    return root/c.label(phase,arm)/'payload/study'


def command(phase,pin,arm='all',dependency=None,budget=None):
    label=c.label(phase,arm);gpu=phase in ('validate','segment');cpu=phase=='cpu'
    allocation=c.allocation(phase)
    if phase=='segment':
        c.require(budget is not None and budget.get('status')=='passed','Measured complete segment allocation required')
        wall=budget['walltime']
    else:wall='02:00' if gpu else ('01:00' if cpu else '00:30')
    argv=['bsub','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
          '-q',allocation['queue'],'-n',str(allocation['slots']),
          '-R','span[hosts=1]','-R','rusage[mem='+allocation['memory_per_slot']+']',
          '-W',wall,'-J','mosaic-chem-complete01-'+label,'-o',str(parent(phase,arm)/'stdout.%J.log'),
          '-e',str(parent(phase,arm)/'stderr.%J.log')]
    if gpu:argv+=['-gpu','num=1:mode=exclusive_process:mps=yes']
    if dependency:argv+=['-w',dependency]
    return argv+['bash','-l',str(c.SOURCE/'scripts/chemistry_continuation_stage.lsf'),str(c.SOURCE),pin,phase,arm,
                 '--capacity-source',str(c.CURRENT)]


def atomic(path,value):
    temporary=path.with_suffix('.partial');c.write(temporary,value);temporary.replace(path)


def submit(phase,pin,arm='all',dependency=None,budget=None):
    directory=parent(phase,arm);directory.mkdir(exist_ok=False)
    argv=command(phase,pin,arm,dependency,budget)
    c.write(directory/'submission-started.json',{'status':'unknown_until_actual_receipt','argv':argv,
            'source_manifest_sha256':pin,'replay_allowed':False})
    env={k:v for k,v in os.environ.items() if not re.fullmatch(r'CUDA_VISIBLE_DEVICES\d*',k)
         and k not in ('CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','CUDA_MPS_ACTIVE_THREAD_PERCENTAGE')}
    run=subprocess.run(argv,capture_output=True,text=True,timeout=90,env=env)
    c.write(directory/'submission-raw.json',{'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    ids=re.findall(r'Job <(\d+)> is submitted',run.stdout)
    c.require(run.returncode==0 and len(ids)==1,'Unknown/failed submission; inspect marker before recovery')
    row={'status':'submitted','job_id':ids[0],'returncode':0,'argv':argv,'dependency':dependency,
         'source_manifest_sha256':pin,'runtime_budget':budget,'phase':phase,'arm':arm}
    c.write(directory/'submission.json',row)
    op,_=scope()
    with (op.ROOT/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=op.ROOT/'jobs.json';jobs=op.read(path)
        key=directory.relative_to(op.ROOT).as_posix();c.require(key not in jobs,'Existing shared job/unknown outcome')
        jobs[key]=row['job_id'];atomic(path,jobs)
    return row


def own(phase,pin,arm):
    directory=parent(phase,arm)
    for _ in range(30):
        if (directory/'submission.json').exists():break
        time.sleep(1)
    row=c.read(directory/'submission.json')
    c.require(row['job_id']==os.environ.get('LSB_JOBID') and row['source_manifest_sha256']==pin
              and row['argv']==command(phase,pin,arm,row['dependency'],row['runtime_budget']),
              'Exact own actual launch/resources differ')
    allocation=c.allocation(phase)
    c.require(os.environ.get('LSB_QUEUE')==allocation['queue'] and
              os.environ.get('LSB_DJOB_NUMPROC')==str(allocation['slots']),
              'Real allocated queue/slots differ')
    return directory,row


def terminal(directory,row,queue):
    text=' '.join((directory/('stdout.'+row['job_id']+'.log')).read_text(encoding='utf8').split())
    c.require(text.count('Subject: Job '+row['job_id']+':')==1 and 'queue <'+queue+'>' in text
              and text.count('Successfully completed.')==1 and 'Exited with exit code' not in text,
              'Actual same-job successful LSF required')
    return c.binding(directory/('stdout.'+row['job_id']+'.log'))


def component(plan,arm):
    row=plan['components'][arm];directory=Path(row['parent']);sub=c.read(directory/'submission.json')
    c.require(sub['status']=='submitted' and sub['job_id']==row['job_id']
              and sub['source_manifest_sha256']==row['manifest_sha256'], 'Exact prerequisite component job differs')
    terminal(directory,sub,'gpul40s');value=c.read(directory/'payload/assessment.json')
    c.require(value['status']=='native_component_qualified_not_continuation_admission'
              and value['job_id']==row['job_id'] and value['can_submit_continuation'] is False,
              'Native component failed/incomplete; never a combined pass')
    c.bound(value['native_assessment'])
    return c.binding(directory/'payload/assessment.json')


def inventory(root):
    size=files=0
    if not root.exists():return size,files
    for folder,dirs,names in os.walk(root,followlinks=False):
        for name in dirs+names:c.require(not (Path(folder)/name).is_symlink(),'New output inventory found link')
        for name in names:
            path=Path(folder)/name;c.require(path.is_file(),'Nonregular output');size+=path.stat().st_size;files+=1
    return size,files


def storage(directory,phase,plan,full=False):
    op,_=scope();attempt=Path(directory)/('whole-capacity-'+phase+'-'+str(time.time_ns()));attempt.mkdir()
    prior=op.storage(attempt,'entry' if phase=='entry' else 'dispatch')
    actual,nfiles=inventory(c.OUTPUTS);projection={};extra=0;extra_files=0
    if full:
        retained=plan['retention_source']
        op.verify(Path(retained['path']),retained['manifest_sha256'])
        retention=module(Path(retained['path'])/'retention.py','_chemistry_qualified_projection')
        c.require(retention.policy()['matrix_stride']==20 and retention.policy()['loss_stride']==1,
                  'Original qualified persistence policy changed')
        for arm,row in plan['arms'].items():
            ready=c.read(c.ROOT/(arm+'-ready.json'));unit=ready['maximum_pilot_update_bytes']
            current,files=inventory(c.OUTPUTS/c.label('segment',arm))
            forecast=c.output_projection(unit,row['remaining'],current)
            c.require(forecast['full_bytes']==retention.full_projection(unit,row['remaining'],1),
                      'Exact original stride20 physical projection differs')
            projection[arm]=forecast
            extra+=forecast['remaining_growth_bytes'];extra_files+=max(0,4*row['remaining']+5000-files)
    result={'status':'passed','version':c.VERSION,'prior_full_current_gate':prior,
            'available_bytes':prior['available_bytes']-actual,'required_bytes':prior['required_bytes']+extra,
            'available_files':prior['available_files']-nfiles,'required_files':prior['required_files']+extra_files,
            'new_chemistry_full_projections':projection,'entire_new_outputs_stale_charge':actual,
            'reserve_bytes':4*c.GIB,'reserve_borrowed':False,'original_gates_changed':False}
    if result['available_bytes']<result['required_bytes'] or result['available_files']<result['required_files']:result['status']='failed'
    c.write(attempt/'chemistry-assessment.json',result)
    c.require(result['status']=='passed','Complete existing plus BOTH chemistry futures do not fit; failed receipt preserved')
    return result


def xml(path,expected):
    root=ET.parse(path).getroot();cases=list(root.iter('testcase'))
    c.require(not any(list(root.iter(k)) for k in ('failure','error','skipped')),'Allocated CPU failures/errors/skips')
    found=Counter()
    for case in cases:
        matches=[m for m in expected if m.rsplit('.',1)[-1] in case.get('classname','').split('.')]
        c.require(len(matches)==1,'Raw CPU module attribution differs');found[matches[0]]+=1
    c.require(dict(found)==expected,'Exact actual CPU module row counts differ');return len(cases)


def driver_command(plan,arm,phase,pin,output=None,startup=False,defer=False):
    output=Path(output or study(phase,arm));args=[c.PYTHON,str(c.SOURCE/'scripts/chemistry_continuation_driver.py'),
        '--plan',str(c.SOURCE/'plan.json'),'--plan-sha256',c.sha(c.SOURCE/'plan.json'),
        '--arm',arm,'--phase',phase,'--output',str(output)]
    if phase=='segment':
        ready=c.read(c.ROOT/(arm+'-ready.json'))['readiness'] if not defer else plan['arms'][arm]['old_ready']
        args+=['--ready',ready['path'],'--ready-sha256',ready['sha256']]
    if startup:args+=['--startup-only']
    if defer:args+=['--startup-defer-ready']
    return args


def startups(plan,pin,payload,defer=True):
    checks=[]
    for arm in c.ARMS:
        for phase in ('validate','segment'):
            directory=payload/('startup-'+arm+'-'+phase);directory.mkdir()
            output=directory/'study';command=driver_command(plan,arm,phase,pin,output,True,defer and phase=='segment')
            run=subprocess.run(command,capture_output=True,text=True,timeout=180)
            c.write(directory/'invocation.json',{'argv':command,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
            c.require(run.returncode==0,'Actual generated original chemistry startup command failed')
            value=c.read(directory/'endpoint-startup.json')
            c.require(value['status']=='passed_metadata_only' and value['updates']==0 and value['numerical_setup_entered'] is False,
                      'Actual startup entered numerical setup')
            checks.append(c.binding(directory/'endpoint-startup.json'))
    c.write(payload/'actual-startups.json',{'status':'passed_metadata_only','checks':checks})


def cpu(manifest,plan,pin,payload):
    inherited=c.read(c.CURRENT/'source_manifest.json')['cpu_counts'];merged=ET.Element('testsuites')
    groups=[('torch','/work3/s234818/latentmol/env/boltz311/bin/python',['tests.test_dbmol_reference']),
            ('jax',c.PYTHON,[m for m in inherited if m!='tests.test_dbmol_reference']),
            ('chemistry',c.PYTHON,list(manifest['cpu_counts']))]
    for name,python,names in groups:
        source=c.SOURCE if name=='chemistry' else c.CURRENT
        args=[python,'-m','pytest',*[str(source/(m.replace('.','/')+'.py')) for m in names],'-q',
              '--rootdir',str(source),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
              '--basetemp',str(Path(os.environ['TMPDIR'])/('chemistry-full-'+name)),
              '--junitxml='+str(payload/(name+'.xml'))]
        c.write(payload/(name+'-command.json'),{'argv':args})
        with (payload/(name+'.log')).open('x') as log:
            run=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        c.require(run.returncode==0,'Complete allocated '+name+' fixture suite failed')
        root=ET.parse(payload/(name+'.xml')).getroot();merged.extend([root] if root.tag=='testsuite' else list(root))
    ET.ElementTree(merged).write(payload/'tests.xml',encoding='utf8',xml_declaration=True)
    count=xml(payload/'tests.xml',{**inherited,**manifest['cpu_counts']})
    startups(plan,pin,payload)
    args=[c.PYTHON,str(c.SOURCE/'scripts/chemistry_reference_worker.py'),'--reference-plan',str(c.SOURCE/'plan.json'),
          '--reference-plan-sha256',c.sha(c.SOURCE/'plan.json'),'--startup-only','--startup-output',str(payload/'protenix-reference-startup.json')]
    run=subprocess.run(args,capture_output=True,text=True,timeout=180)
    c.write(payload/'reference-startup-command.json',{'argv':args,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
    c.require(run.returncode==0,'Actual allocated Protenix reference command failed')
    c.write(payload/'assessment.json',{'status':'passed','cases':count,'job_id':os.environ['LSB_JOBID'],
            'source_manifest_sha256':pin,'optimizer_updates':0})


def cpu_certificate(manifest,pin):
    path=c.ROOT/'cpu-certificate.json'
    if path.exists():
        value=c.read(path);c.require(value['source_manifest_sha256']==pin,'CPU source differs')
        for row in value['proofs']:c.bound(row)
        return value
    directory=parent('cpu');sub=c.read(directory/'submission.json');lsf=terminal(directory,sub,'hpc')
    expected={**c.read(c.CURRENT/'source_manifest.json')['cpu_counts'],**manifest['cpu_counts']}
    count=xml(directory/'payload/tests.xml',expected);assessment=c.read(directory/'payload/assessment.json')
    c.require(assessment['status']=='passed' and assessment['cases']==count and assessment['job_id']==sub['job_id'],
              'CPU assessment/source owner differs')
    proofs=[c.binding(p) for p in sorted((directory/'payload').rglob('*')) if p.is_file()]
    value={'status':'passed','source_manifest_sha256':pin,'cases':count,'proofs':[lsf,*proofs]}
    c.write(path,value);return value


def accept(plan,arm,phase,pin,*,adopt_from=None):
    c.require(adopt_from in (None,PREVIOUS_ROOT) and (adopt_from is None or phase=='validate'),
              'Only exact previous validation evidence may be adopted')
    directory=parent(phase,arm) if adopt_from is None else adopt_from/c.label(phase,arm)
    sub=c.read(directory/'submission.json');lsf=terminal(directory,sub,'gpul40s')
    output=study(phase,arm) if adopt_from is None else directory/'payload/study'
    summary=c.read(output/'summary.json');row=plan['arms'][arm]
    c.require(summary['status']=='passed' and summary['native_hard_parity']['status']=='passed'
              and summary['native_gradient_checks']['status']=='passed' and summary['diffusion_gradient'] is False,
              'Original native/parity/null/finite checks did not pass')
    c.require(summary['source_manifest_sha256']==row['science_source']['manifest_sha256']
              and summary['config_sha256']==row['config']['sha256'] and summary['job_id']==sub['job_id'],
              'Exact scientific config/source/job identity changed')
    source=Path(row['science_source']['path']);sys.path.insert(0,str(source/'scripts'))
    contract=module(source/'scripts/campaign25_contract.py','_chemistry_original_contract_'+arm)
    c.frozen_identity(arm,summary,row['frozen_identity'],contract.verify_frozen_response)
    protocol=module(source/('scripts/extended_protocol.py' if arm=='protenix' else 'scripts/boltz_serial_protocol.py'),
                    '_chemistry_original_boundary_'+arm)
    require_boundary=protocol.require_boundary
    if phase=='segment':require_boundary=driver.hooks.protocol_function(require_boundary,arm,'require_boundary',row['remaining'])
    execution=require_boundary(summary)
    c.require(execution['start']==row['start'] and execution['end']==row['start']+(10000 if phase=='validate' else row['remaining'])
              and execution['input_snapshot_sha256']==row['boundary']['sha256'],'Exact original boundary/backend/key scope changed')
    base,_=driver.modules();base._boundary(row['boundary']['path'],output/'initialization.npz')
    evidence=c.read(output.parent/'prospective-evidence.json')
    c.require(len(evidence['native'])==(1 if phase=='validate' else 2),'Both required raw endpoint cohorts must be present')
    c.require(evidence['native'] and all(r['status']=='passed' and r['can_admit_new_child'] is True for r in evidence['native']),
              'Fresh prospective raw native endpoint incomplete')
    for record in evidence['native']:
        value=c.read(record['path']);c.require(value['original_failed_parent_recertified'] is False
                if 'original_failed_parent_recertified' in value else value.get('can_certify_original_study') is False,
                'Historical failed run may never be recertified')
    if phase=='validate':
        c.require(summary['chemistry_assay']['status']=='passed','Nonzero chemistry assay failed')
        for name in ('low','high','actual_coefficient'):
            gate=c.read(output/(name+'_coupled_checks/report.json'))
            c.require(gate['status']=='passed' and gate['frozen_unchanged'] is True and gate['can_admit_new_child'] is True,
                      'Combined fixed-teacher coefficient gate incomplete: '+name)
        for name in ('early','high'):
            report=c.read(output/(name+'_pilot/report.json'))
            c.require(report['status']=='passed' and len(report['updates'])==10 and len(report['states'])==11,
                      'BOTH complete original pilots required')
        unit=summary['storage_projection']['maximum_pilot_update_bytes']
        c.require(type(unit) is int and unit>0 and summary['storage_projection']['projected_bytes']==2*unit*10000+c.GIB,
                  'Original full pilot projection formula changed')
        ready=c.binding(output/'assessment.json')
        contract.verify_readiness(Path(ready['path']),ready['sha256'],Path(row['config']['path']),row['science_source']['manifest_sha256'])
        value={'status':'passed','arm':arm,'readiness':ready,'runtime_budget':c.runtime(summary,arm),
               'maximum_pilot_update_bytes':unit,'input_snapshot':row['boundary'],'source_manifest_sha256':pin,
               'validation_owner':str(directory),'validation_job_id':sub['job_id'],
               'previous_callback_relabelled':False,'adopted_unchanged_previous_scientific_evidence':adopt_from is not None,
               'proofs':[lsf,c.binding(output/'summary.json'),c.binding(output.parent/'prospective-evidence.json')]}
        c.write(c.ROOT/(arm+'-ready.json'),value);return value
    c.require(summary['terminal_gradient_checks']['status']=='passed','Terminal raw derivative gate failed')
    # The unmodified result/JSONL/rate/retained-frame validation is supplemented
    # by exact original FP32/key replay of the retained input before dispatch.
    case=c.read(output/'case-0/report.json');wrapped=summary['cases'][0]
    c.require({k:v for k,v in wrapped.items() if k not in ('label','initialization')}==case
              and case['status']=='passed' and len(case['updates'])==row['remaining']
              and len(case['states'])==row['remaining']+1,'Complete child case/summary budget differs')
    c.require(case['global_range']=={'start':row['start'],'end':row['target']},'Global child horizon differs')
    contract.verify_attempt_rates(case,c.read(row['config']['path']))
    for field in ('states','updates'):
        count=0
        with (output/('case-0/'+field+'.jsonl')).open() as stream:
            for i,line in enumerate(stream):
                value=json.loads(line);c.require(line.endswith('\n') and value==case[field][i] and value['step']==i
                        and value['global_step']==row['start']+i,'Exact contiguous committed child JSONL differs');count+=1
        c.require(count==len(case[field]),'Missing committed scalar rows')
    for state in case['states']:
        if state['step']%20==0 or state['step']==row['remaining']:
            path=output/'case-0'/state['snapshot'];c.require(path.resolve().is_relative_to((output/'case-0').resolve())
                    and c.sha(path)==state['snapshot_sha256'],'Retained child matrix changed')
    c.require(case['states'][-1]['repair_weight']==100.,'Original terminal chemistry ramp differs')
    value={'status':'passed_child_only_original_failure_preserved','arm':arm,'new_updates':row['remaining'],
            'global_end':row['target'],'original_results_changed':False,'original_failed_parent_recertified':False,
            'proofs':[lsf,c.binding(output/'summary.json'),c.binding(output/'case-0/report.json'),
                      c.binding(output/'case-0/states.jsonl'),c.binding(output/'case-0/updates.jsonl')]}
    c.write(c.ROOT/(arm+'-completion.json'),value);return value


def previous_validation_metadata(plan,arm):
    """Authenticate an already submitted validator without asserting success."""
    directory=PREVIOUS_ROOT/c.label('validate',arm)
    if not directory.exists():return None
    op,_=scope();previous=op.verify(PREVIOUS_SOURCE,PREVIOUS_PIN)
    old_contract=module(PREVIOUS_SOURCE/'scripts/chemistry_continuation_contract.py','_chemistry_previous_contract')
    old_op=module(PREVIOUS_SOURCE/'scripts/chemistry_continuation_ops.py','_chemistry_previous_op')
    old_op.c=old_contract
    c.require((PREVIOUS_ROOT/'cpu-certificate.json').is_file(),'Passed previous full CPU required before GPU adoption')
    old_op.cpu_certificate(previous,PREVIOUS_PIN)
    for name in ('chemistry_continuation_driver.py','chemistry_continuation_hooks.py','chemistry_cuda_ownership.py'):
        c.require(c.sha(PREVIOUS_SOURCE/'scripts'/name)==c.sha(c.SOURCE/'scripts'/name),
                  'Scientific driver/checker/CUDA implementation changed; old GPU work cannot certify it')
    sub=c.read(directory/'submission.json')
    c.require(sub['source_manifest_sha256']==PREVIOUS_PIN and sub['status']=='submitted'
              and sub['argv']==old_op.command('validate',PREVIOUS_PIN,arm,sub['dependency']),
              'Exact previous GPU job/source/startup/resources differ')
    old_plan=c.read(PREVIOUS_SOURCE/'plan.json');new=plan['arms'][arm];old=old_plan['arms'][arm]
    c.require(new==old,'Original settings/input/check scope changed; native proof adoption forbidden')
    # The invocation does not exist while the validator is still queued. Its
    # exact original scientific vector is additionally checked at adoption.
    callback=PREVIOUS_ROOT/c.label('after-validate',arm)
    callback_sub=c.read(callback/'submission.json') if callback.exists() else None
    if callback_sub:
        c.require(callback_sub['source_manifest_sha256']==PREVIOUS_PIN and callback_sub['status']=='submitted'
                  and callback_sub['argv']==old_op.command('after-validate',PREVIOUS_PIN,arm,callback_sub['dependency'])
                  and callback_sub['dependency']=='ended('+sub['job_id']+')','Previous own callback differs')
    return {'submission':sub,'callback':callback_sub,'parent':str(directory)}


def adopt_previous_validation(plan,arm,pin):
    """Reuse genuine passed GPU work; never rerun an unknown/scientific failure.

    Previous controller failures remain failures. The new allocated CPU owns
    the corrected response-schema acceptance and complete prospective checks.
    """
    record=previous_validation_metadata(plan,arm)
    if record is None:return False
    directory=Path(record['parent'])
    invocation=c.read(directory/'payload/driver-invocation.json')
    c.require(invocation['original_argv']==driver.argv(plan,arm,'validate',directory/'payload/study'),
              'Original scientific entrypoint/inputs changed')
    value=accept(plan,arm,'validate',pin,adopt_from=PREVIOUS_ROOT)
    # Preserve the old sealed source/CPU evidence, including any callback EXIT.
    value['proofs'] += [c.binding(PREVIOUS_SOURCE/'source_manifest.json'),c.binding(PREVIOUS_ROOT/'cpu-certificate.json'),
                       c.binding(directory/'submission.json'),c.binding(directory/'payload/driver-invocation.json')]
    path=c.ROOT/(arm+'-ready.json');amend=path.with_suffix('.adopted.partial')
    c.write(amend,value);amend.replace(path)
    return True


def dispatch_ready(plan,pin,directory,payload):
    """Caller owns dispatch.lock; both complete proofs and full futures first."""
    if not all((c.ROOT/(name+'-ready.json')).exists() for name in c.ARMS):return
    c.require(not any((PREVIOUS_ROOT/c.label('segment',name)).exists() for name in c.ARMS),
              'Previous optimizer submission/unknown outcome prohibits replacement')
    c.require(not any(parent('segment',name).exists() for name in c.ARMS),'Existing/unknown long outcome prevents dispatch')
    storage(directory,'dispatch',plan,True)
    for name in c.ARMS:
        ready=c.read(c.ROOT/(name+'-ready.json'))
        for proof in ready['proofs']:c.bound(proof)
        startup=payload/('admitted-startup-'+name);startup.mkdir()
        args=driver_command(plan,name,'segment',pin,startup/'study',True)
        subprocess.run(args,check=True,timeout=180)
        run=submit('segment',pin,name,budget=ready['runtime_budget'])
        submit('after-segment',pin,name,'ended('+run['job_id']+')')


def execute(phase,pin,arm):
    manifest,plan=own_source(pin);directory,sub=own(phase,pin,arm)
    storage(directory,'entry',plan,phase in ('segment','after-segment'));payload=directory/'payload';payload.mkdir()
    if phase=='cpu':cpu(manifest,plan,pin,payload);return
    cpu_certificate(manifest,pin)
    if phase=='after-cpu':
        with (c.ROOT/'dispatch.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);storage(directory,'dispatch',plan)
            for name in c.ARMS:
                previous=previous_validation_metadata(plan,name)
                if previous is not None:
                    dependency='ended('+previous['submission']['job_id']+')'
                    if previous['callback'] is not None:dependency+=' && ended('+previous['callback']['job_id']+')'
                    submit('after-validate',pin,name,dependency);continue
                dependency='ended('+plan['components'][name]['job_id']+')' if name=='protenix' else None
                if name=='boltz':component(plan,name)
                native=submit('validate',pin,name,dependency)
                submit('after-validate',pin,name,'ended('+native['job_id']+')')
            dispatch_ready(plan,pin,directory,payload)
        c.write(payload/'assessment.json',{'status':'combined_qualification_jobs_submitted','optimizer_submitted':False});return
    if phase in ('after-validate','after-segment'):
        before=phase.removeprefix('after-')
        if phase=='after-validate' and not parent('validate',arm).exists():
            c.require(adopt_previous_validation(plan,arm,pin),'No own or authenticated previous validation')
            value=c.read(c.ROOT/(arm+'-ready.json'))
        else:value=accept(plan,arm,before,pin)
        c.write(payload/'assessment.json',value)
        if before=='validate':
            with (c.ROOT/'dispatch.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                dispatch_ready(plan,pin,directory,payload)
        return
    component(plan,arm)
    routing=module(c.SOURCE/'scripts/chemistry_cuda_ownership.py','_chemistry_cuda_ownership')
    identity=plan['arms'][arm].get('context_ready',plan['arms'][arm].get('context'))
    c.bound(identity)
    routing.run_probe(directory/'cuda-ownership',submission=directory/'submission.json',expected_job_id=sub['job_id'],
        input_bindings={'source_manifest':c.binding(c.SOURCE/'source_manifest.json'),'plan':c.binding(c.SOURCE/'plan.json'),
                        'context_identity':identity,'adoption':c.binding(c.ROOT/'cpu-certificate.json')},deadline=time.monotonic()+600)
    output=study(phase,arm);output.parent.mkdir(parents=True,exist_ok=phase=='validate')
    args=driver_command(plan,arm,phase,pin);c.write(payload/'driver-invocation.json',{'argv':args,
            'original_argv':driver.argv(plan,arm,phase,output,c.read(c.ROOT/(arm+'-ready.json'))['readiness'] if phase=='segment' else None)})
    env=dict(os.environ);env.update(plan['execution_context'][arm]['environment'])
    timeout=6900 if phase=='validate' else sub['runtime_budget']['walltime_seconds']-600
    started=time.monotonic();next_gate=started+300
    with (payload/'driver.log').open('x') as log:
        child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                c.require(time.monotonic()-started<timeout,'Measured complete allocation exhausted')
                if time.monotonic()>next_gate:
                    storage(directory,'periodic',plan,phase=='segment');next_gate=time.monotonic()+300
                time.sleep(3)
            c.require(child.returncode==0,'Original chemistry driver failed; saved evidence preserved')
        finally:
            if child.poll() is None:
                import signal
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pin',required=True)
    p.add_argument('--phase',required=True,choices=('cpu','after-cpu','validate','after-validate','segment','after-segment'))
    p.add_argument('--arm',default='all',choices=('all',*c.ARMS));a=p.parse_args()
    execute(a.phase,a.pin,a.arm)


if __name__=='__main__':main()
