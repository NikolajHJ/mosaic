"""Funded three-arm precision diagnostics sharing the admitted Pro control owner."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
SCOPE=BASE/'nonchem-precision-continuation-26'
SHARED=BASE/'runtime-production-release-05/source'
SHARED_PIN='36e1bfcd9acb864a557e3cf2c317334626fd596337d39306b352c1b0131da11f'
ROOT=SCOPE/'runtime-production-05/boltz-precision-05'
SOURCE=SCOPE/'boltz-precision-release-05/source'
CAP=256*1024**2
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'

def require(value,message):
    if not value:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def write(p,value):
    with Path(p).open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def binding(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def predecessor_check(value):
    require(value['job_ids']=={'cpu':'29852949','after-cpu':'29852950'} and value['status']=='fixture_API_ERROR_before_native_startup',
            'Exact failed fixture/API predecessor required')
    require(value['actual_numerical_cases']==51 and value['passed_cases']==48 and value['scientific_outputs']==0
            and value['GPU_jobs']==[] and value['pretrained_models_constructed']==0,'Unexpected numerical/native execution cannot be retried')
    expected={'test_reference_precision_context_restores_production_configuration',
        'test_probability_to_logit_chain_matches_independent_AD','test_fixed_target_gradient_flows_without_target_refresh'}
    require(len(value['failures'])==3 and {r['name'] for r in value['failures']}==expected
        and all(r['module']=='tests.test_boltz_reference_jax'
            and "module 'jax.experimental' has no attribute 'enable_x64'" in r['message'] for r in value['failures']),
        'Only the authenticated obsolete JAX API fixture failure is repaired')

# The shell authenticates the shared owner and every parent before importing it.
sys.path.insert(1,str(SHARED/'scripts'))
shared=module(SHARED/'scripts/runtime_production_ops.py','_admitted_shared_Protenix_capacity_owner')
protocol=module(shared.QUALIFIED/'scripts/failure_diagnostic_ops.py','_qualified_held_diagnostic_protocol')
protocol.ROOT=ROOT;protocol.SOURCE=SOURCE
submit=protocol.submit

def context(pin):
    require(sha(SHARED/'source_manifest.json')==SHARED_PIN,'Actual shared owner changed')
    op,current=shared.context(SHARED_PIN);manifest=op.verify(SOURCE,pin);plan=read(SOURCE/'plan.json')
    require(plan['shared_capacity']['path']==str(SHARED) and plan['shared_capacity']['manifest_sha256']==SHARED_PIN,
            'Exact shared capacity delegation required')
    require(plan['root']==str(ROOT) and plan['source_root']==str(SOURCE) and plan['output_cap_bytes']==CAP
            and plan['optimizer_updates']==0 and plan['can_admit_new_child'] is False,'Closed bounded diagnostic owner required')
    for row in manifest['parent_sources']:op.verify(Path(row['path']),row['manifest_sha256'])
    for row in plan['inputs']:op.check_binding(row)
    previous=plan['failed_predecessor'];predecessor_check(previous)
    for phase,jid in previous['job_ids'].items():
        parent=Path(previous['root'])/phase;sub=read(parent/'submission.json')
        require(sub['job_id']==jid,'Failed predecessor job identity differs')
        text=''.join((parent/('stdout.'+jid+'.log')).read_text().split())
        require(text.count('Subject:Job'+jid+':')==1 and text.count('Exitedwithexitcode1.')==1
            and 'Successfullycompleted.' not in text,'Actual failed same-job LSF required')
    require(not list(Path(previous['root']).rglob('*.npz')) and not (Path(previous['root'])/'cpu/payload/tests.xml').exists(),
            'Failed predecessor unexpectedly contains native/GPU or merged-success output')
    cases=[]
    for group in ('original-Torch','new-Torch','new-JAX'):
        cases+=list(ET.parse(Path(previous['root'])/'cpu/payload'/(group+'.xml')).getroot().iter('testcase'))
    failures=[{'module':r.get('classname'),'name':r.get('name'),'message':r.find('failure').get('message','')}
              for r in cases if r.find('failure') is not None]
    require(len(cases)==51 and failures==previous['failures'] and not any(r.find('error') is not None or r.find('skipped') is not None for r in cases),
            'Original fixture failures/raw XML differ')
    for source in {p['source']['path'] for p in plan['arms'].values()}:
        for name,digest in plan['identical_scientific_modules'].items():
            require(sha(Path(source)/name)==digest,'Shared native/chemistry mathematics differ between original arms')
    return op,plan

def budget(plan):
    total=0;count=0
    for p in ROOT.rglob('*'):
        require(not p.is_symlink() and (p.is_file() or p.is_dir()),'Special diagnostic output')
        if p.is_file():total+=p.stat().st_size;count+=1
    require(total<CAP and count<2048,'Complete diagnostic output/file cap exceeded')
    return total

def storage(op,parent,phase):
    plan=read(SOURCE/'plan.json');current=read(SHARED/'plan.json')
    proxy=module(SHARED/'scripts/runtime_production_capacity.py','_original_shared_metadata_walker')
    owner=proxy.proxy(op,current,SHARED,shared.ROOT,shared.ready_envelope(op,current,SHARED_PIN),shared.verify_qualified(op,current))
    require(owner.metadata_bytes()+CAP-budget(plan)<=2*1024**3,'New diagnostic group cannot fit original shared2GiB control cap')
    rolling=module(shared.CAPACITY/'scripts/rolling_capacity.py','_unchanged_complete_shared_storage')
    proxy.configure(rolling,owner,current)
    import storage_policy_adapter
    storage_policy_adapter.install(rolling,plan,parent,plan['shared_source146'])
    # Explicit newly qualified live-quota/remaining-write resource policy.
    # No scientific or saved-result change; all inventories and caps remain.
    return rolling.storage(owner,parent,phase)

def argv(phase,pin,dependency=None):
    require(phase in ('cpu','after-cpu','gpu-boltz','gpu-boltz_chemistry','gpu-dbmol'),'Unknown phase')
    gpu=phase.startswith('gpu-');slots=8 if gpu else 1 if phase=='after-cpu' else 4;parent=ROOT/phase
    values=['bsub','-H','-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
        '-q','gpul40s' if gpu else 'hpc','-n',str(slots),'-R','span[hosts=1]','-R','rusage[mem='+('2GB' if slots==1 else '4GB')+']',
        '-W','02:00' if gpu else '00:30' if slots==1 else '01:00','-J','mosaic-boltz-precision05-'+phase,
        '-o',str(parent/'stdout.%J.log'),'-e',str(parent/'stderr.%J.log')]
    if gpu:values+=['-gpu','num=1:mode=shared:j_exclusive=yes:mps=no']
    if dependency:values+=['-w',dependency]
    return values+['bash','-l',str(SOURCE/'scripts/boltz_precision_stage.lsf'),str(SOURCE),pin,phase,
        '--shared-capacity-source',str(SHARED)]

def xml(path,counts):
    cases=list(ET.parse(path).getroot().iter('testcase'));observed={name:0 for name in counts}
    for row in cases:
        require(not any(list(row.iter(tag)) for tag in ('failure','error','skipped')),'Raw CPU case failed/skipped')
        names=[name for name in counts if row.get('classname')==name or row.get('classname','').startswith(name+'.')]
        require(len(names)==1,'CPU module attribution differs');observed[names[0]]+=1
    require(observed==counts,'Exact allocated module counts differ');return len(cases)

def cpu(plan,payload):
    manifest=read(SOURCE/'source_manifest.json');merged=ET.Element('testsuites');total=0
    for name,python,source,counts in (
        ('original-Torch',plan['worker_python'],Path(plan['capacity_science']['path']),{'tests.test_dbmol_reference':17}),
        ('new-Torch',plan['worker_python'],SOURCE,manifest['torch_cpu_counts']),
        ('new-JAX',PYTHON,SOURCE,manifest['jax_cpu_counts'])):
        command=[python,'-m','pytest',*[str(source/(module.replace('.','/')+'.py')) for module in counts],'-q',
            '--rootdir',str(source),'--import-mode=importlib','-o','addopts=','-p','no:cacheprovider',
            '--basetemp',str(Path(os.environ['TMPDIR'])/('boltz-reference-'+name)),'--junitxml='+str(payload/(name+'.xml'))]
        write(payload/(name+'-command.json'),{'argv':command,'counts':counts})
        environment={**os.environ,'PYTHONPATH':str(SOURCE/'scripts')+os.pathsep+str(source/'scripts')+os.pathsep+str(Path(plan['science']['path'])/'src')}
        with (payload/(name+'.log')).open('x') as stream:run=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
        require(run.returncode==0,'CPU suite failed: '+name);total+=xml(payload/(name+'.xml'),counts)
        tree=ET.parse(payload/(name+'.xml')).getroot();merged.extend([tree] if tree.tag=='testsuite' else list(tree))
    ET.ElementTree(merged).write(payload/'tests.xml',encoding='utf8',xml_declaration=True)
    for arm in plan['arms']:
        command=[PYTHON,str(SOURCE/'scripts/boltz_precision_driver.py'),'--pin',sha(SOURCE/'source_manifest.json'),'--arm',arm,'--startup-only']
        write(payload/('startup-'+arm+'-command.json'),{'argv':command,'models_authorized':False})
        subprocess.run(command,check=True,timeout=600)
    return total

def execute(phase,pin):
    start=time.monotonic();op,plan=context(pin);parent=ROOT/phase;own=read(parent/'submission.json')
    require(own['job_id']==os.environ.get('LSB_JOBID') and own['argv']==argv(phase,pin,own['dependency']),
            'Own actual command/resources/source differ')
    storage(op,parent,'entry');payload=parent/'payload';payload.mkdir()
    if phase=='cpu':
        total=cpu(plan,payload);write(payload/'assessment.json',{'status':'passed','cases':total,'job_id':own['job_id'],'source_manifest_sha256':pin});return
    if phase=='after-cpu':
        sub=read(ROOT/'cpu/submission.json');proof=protocol.terminal(ROOT/'cpu',sub['job_id'],'hpc')
        counts=read(SOURCE/'source_manifest.json')['all_cpu_counts'];total=xml(ROOT/'cpu/payload/tests.xml',counts)
        require(read(ROOT/'cpu/payload/assessment.json')['cases']==total,'Actual CPU assessment differs')
        for arm in plan['arms']:
            row=read(ROOT/'cpu/payload'/('startup-'+arm)/'server-startup.json')
            require(row['job_id']==sub['job_id'] and row['model_constructed'] is False and row['updates']==0,'Actual startup missing')
        write(ROOT/'cpu-certificate.json',{'status':'passed','job_id':sub['job_id'],'cases':total,
            'source_manifest_sha256':pin,'same_job_LSF':proof})
        storage(op,parent,'dispatch');gpus={arm:submit('gpu-'+arm,pin) for arm in plan['arms']}
        write(payload/'assessment.json',{'status':'CPU_startups_and_full_shared_storage_passed_three_precision_GPUs_submitted',
            'gpus':gpus,'optimizer_jobs':[]});return
    require(read(ROOT/'cpu-certificate.json')['source_manifest_sha256']==pin,'New allocated CPU qualification required')
    routing=module(Path(plan['chemistry_source']['path'])/'scripts/chemistry_cuda_ownership.py','_original_Boltz_shared_GPU_ownership')
    routing.GPU_REQUEST='num=1:mode=shared:j_exclusive=yes:mps=no'
    routing.run_probe(parent/'cuda-ownership',submission=parent/'submission.json',expected_job_id=own['job_id'],
        input_bindings={'source_manifest':binding(SOURCE/'source_manifest.json'),'plan':binding(SOURCE/'plan.json'),
            'context_identity':plan['context']},deadline=start+600)
    original=read(Path(plan['chemistry_source']['path'])/'plan.json')
    helper=module(Path(original['arms']['protenix']['adapter_source']['path'])/'scripts/full_model_pilot.py','_unchanged_precision_owned_cleanup')
    helper.OUTPUT_CAP=CAP
    original_inventory=helper.output_bytes;next_gate=time.monotonic()+300
    def inventory(path):
        nonlocal next_gate
        budget(plan)
        if time.monotonic()>=next_gate:
            storage(op,parent,'periodic');next_gate=time.monotonic()+300
        return original_inventory(path)
    helper.output_bytes=inventory
    command=[PYTHON,str(SOURCE/'scripts/boltz_precision_driver.py'),'--pin',pin,'--arm',phase.removeprefix('gpu-')]
    write(payload/'actual-command.json',{'argv':command,'optimizer_updates':0})
    with (payload/'execution.log').open('x') as stream:
        with helper._parent_sigterm_cleanup():result=helper.run_isolated_child(command,SOURCE,dict(os.environ),stream,
            max(1,6600-(time.monotonic()-start)),ROOT)
    write(payload/'execution.json',result);context(pin);budget(plan)
    write(payload/'preservation.json',{'status':'inputs_and_sources_preserved','optimizer_updates':0,'can_admit_new_child':False})

protocol.argv=argv
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);parser.add_argument('--pin',required=True)
    args=parser.parse_args();execute(args.phase,args.pin)
