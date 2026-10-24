"""Two independent serial Protenix chains, immutable inputs and exclusive outcomes."""
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from extended_protocol import projection, require_boundary, verify_parent
BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'protenix-extended-02'
SOURCE=BASE/'protenix-extended-release-02/source'
OLD=BASE/'restart-lrreset-04'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
ARMS=('random','continuation')
COUNTS={'random':5,'continuation':4}
START={'random':0,'continuation':10000}
OLD_FULL={'protenix':1493521824,'boltz2':1748021824,'boltz2_chemistry':5144021824}
MODEL='8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
CONTEXT='b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
GIB=1024**3
REVISION='c74c659344123937608077e3e4078f75f09cd743'
STARTUP_VERSION='protenix-actual-command-gpu-premodel-v1'
STARTUP_CASES=28
BASELINE=BASE/'boltz-lr0025-03'
BASELINE_SOURCE=BASE/'boltz-lr0025-release-03/source'
DBMOL=BASE/'boltz-dbmol-pilot-02'
DBMOL_SOURCE=BASE/'boltz-dbmol-pilot-release-02/source'
CONTEXT_READY=OLD/'validate-protenix/payload/study/assessment.json'
CONTEXT_READY_SHA='9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd'
NOISY_ROOT=Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01')
NOISY_RELEASE=Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01')
NOISY_UPLOAD=Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01.tar.gz')
NOISY_MANIFEST='7bb970f48f6f907b7dc05e74a9bc2645a313addf814f08a0eeca11582f3b84ef'
NOISY_STAGE='experiments/brd4-jq1/protenix-noisy-1000/scripts/stage.py'
NOISY_STAGE_SHA='48e3adc68cdb4aed0268b83e9f09508ac7836e0aec14c0113b4b441f947dc6fc'

def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def exclusive(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(p,value):
    p=Path(p);t=p.with_name(p.name+'.tmp');exclusive(t,value);t.replace(p)
def config(arm):
    require(arm in ARMS,'Unknown arm');return SOURCE/'examples/configs'/('extended_'+arm+'.json')
def offset(arm,index):
    require(arm in ARMS and type(index) is int and 0<=index<COUNTS[arm],'Undeclared segment')
    return START[arm]+10000*index
def name(arm,phase,index):offset(arm,index);return f'{arm}-{phase}-{index:04d}'
def decode(parent):
    m=re.fullmatch(r'(random|continuation)-(native|after-native|segment|after-segment)-(\d{4})',Path(parent).name)
    require(m is not None,'Undeclared parent');arm,phase,index=m.group(1),m.group(2),int(m.group(3));offset(arm,index);return arm,phase,index
def resources(phase):
    if phase=='cpu':return ['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00']
    if phase in ('native','segment'):return ['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00' if phase=='native' else '23:00']
    require(phase in ('after-cpu','after-native','after-segment','dispatch'),'Unknown phase')
    return ['-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30']
def command(label,phase,manifest,dependency=None):
    if phase=='dispatch':require(re.fullmatch(r'dispatch-\d{4}',label) is not None,'Dispatch requires a new unique parent')
    elif phase not in ('cpu','after-cpu'):
        arm,actual,index=decode(label);require(actual==phase,'Parent/phase mismatch')
    else:require(label==phase,'CPU parent mismatch')
    parent=ROOT/label;args=['bsub',*resources(phase),'-J','mosaic-pro50k-'+label,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency:args+=['-w',dependency]
    wrapper=SOURCE/'scripts/hpc'/('extended_cpu.lsf' if phase=='cpu' else 'extended_stage.lsf')
    args+=['bash','-l',str(wrapper),str(SOURCE),str(ROOT),manifest]
    if phase!='cpu':args += [phase,str(parent)]
    return args
def submit(label,phase,manifest,dependency=None):
    parent=ROOT/label;args=command(label,phase,manifest,dependency);parent.mkdir(exist_ok=False)
    r={'phase':phase,'args':args,'dependency':dependency,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()};exclusive(parent/'submission-started.json',r)
    try:
        q=subprocess.run(args,capture_output=True,text=True,timeout=60);r.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr);ids=re.findall(r'Job <(\d+)>',q.stdout)
        if q.returncode==0 and len(ids)==1:r.update(outcome='submitted',job_id=ids[0])
    except Exception as e:r['error']=repr(e)
    finally:exclusive(parent/'submission.json',r)
    require(r['outcome']=='submitted','Unknown submission, no blind duplicate')
    jobs=read(ROOT/'jobs.json');jobs[label]=r['job_id'];atomic(ROOT/'jobs.json',jobs)
    q=subprocess.run(['bjobs','-l',r['job_id']],capture_output=True,text=True,timeout=30);(parent/'scheduler-at-submission.txt').write_text(q.stdout+q.stderr)
    return r
def executing(parent,phase,manifest):
    p=Path(parent)/'submission.json'
    for attempt in range(16):
        if p.exists():break
        require(attempt<15,'Missing own actual submission');time.sleep(2)
    r=read(p);require(r['outcome']=='submitted' and r['job_id']==os.environ.get('LSB_JOBID') and r['phase']==phase and r['args']==command(Path(parent).name,phase,manifest,r['dependency']),'Own submission/resources/source differs')
def final_job(parent,queue):
    r=read(parent/'submission.json');job=r['job_id']
    for attempt in range(16):
        q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=q.stdout+q.stderr;p=parent/('scheduler-final-'+str(attempt)+'.txt');p.write_text(raw);compact=re.sub(r'\s+','',raw)
        require(q.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Queue<([^>]+)>',compact)==[queue],'No authentic producer final scheduler')
        status=re.findall(r'Status<([^>]+)>',compact)
        if status==['DONE']:break
        require(status in (['RUN'],['PEND']) and attempt<15,'Actual predecessor failed');time.sleep(2)
    log=parent/('stdout.'+job+'.log');text=log.read_text()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',text,re.M)==[job] and 'Successfully completed.' in text and 'Exited with exit' not in text,'Same-job LSF success absent')
    return [{'path':str(v),'sha256':sha(v)} for v in [p,log,parent/'submission.json']]
def verify_proofs(c):
    require(c['status']=='passed','Failed certificate')
    for p in c['proofs']:require(sha(p['path'])==p['sha256'],'Evidence bytes changed')
def cpu_certificate(manifest):
    p=ROOT/'cpu-certificate.json';c=read(p);require(c['source_manifest_sha256']==manifest and c['suite']=='protenix-serial50k-v1','Wrong CPU scope');verify_proofs(c);return p
def command_path(arm,phase,index):
    offset(arm,index);require(phase in ('native','segment'),'Unknown startup phase')
    return ROOT/(f'startup-command-{arm}-{phase}-{index:04d}.json')

def startup_inputs(arm,phase,index,manifest):
    paths={'config':config(arm),'cpu-certificate':cpu_certificate(manifest),
           'reference':ROOT/'cpu/payload/composition_reference.json','pocket':ROOT/'cpu/payload/pocket_definition.json'}
    context_ready=CONTEXT_READY
    if phase=='segment':
        context_ready,_=native_assessment(arm,index,manifest)
        paths['ready']=context_ready
    else:require(sha(context_ready)==CONTEXT_READY_SHA,'Original frozen context assessment changed')
    paths['context-assessment']=context_ready
    context=read(context_ready)['context_files']
    require(set(context)=={'prepared_context.json','prepared_context.npz'},'Frozen context file scope differs')
    for name,digest in context.items():
        path=context_ready.parent/'context'/name
        require(sha(path)==digest,'Frozen context bytes changed')
        paths['context-'+name]=path
    snapshot=input_snapshot(arm,index)
    if snapshot:
        path=Path(snapshot['path']);require(sha(path)==snapshot['sha256'],'Exact predecessor snapshot changed')
        paths['resume-snapshot']=path
    return {key:{'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)} for key,path in paths.items()}

def startup_args(arm,phase,index,manifest,inputs,output):
    args=[PYTHON,str(SOURCE/'examples/brd4_25_atoms.py'),'--mode','validate' if phase=='native' else 'run',
          '--config',inputs['config']['path'],'--output',output,'--manifest',manifest,
          '--cpu-certificate',inputs['cpu-certificate']['path'],'--cpu-certificate-sha256',inputs['cpu-certificate']['sha256'],
          '--reference',inputs['reference']['path'],'--pocket',inputs['pocket']['path'],'--pocket-sha256',POCKET,
          '--offset',str(offset(arm,index)),'--segment-steps','10000']
    if 'resume-snapshot' in inputs:args+=['--resume-snapshot',inputs['resume-snapshot']['path'],'--resume-sha256',inputs['resume-snapshot']['sha256']]
    if 'ready' in inputs:args+=['--ready',inputs['ready']['path'],'--ready-sha256',inputs['ready']['sha256']]
    return args

def verify_command_capture(arm,phase,index,manifest):
    path=command_path(arm,phase,index);record=read(path);verify_proofs(record)
    r=record['result'];inputs=startup_inputs(arm,phase,index,manifest)
    require(record['source_manifest_sha256']==manifest and record['version']==STARTUP_VERSION
            and record['arm']==arm and record['phase']==phase and record['index']==index,
            'CPU command certificate scope differs')
    require(r['version']==STARTUP_VERSION and r['kind']=='cpu_generated_command_only'
            and r['status']=='command_only_authenticated' and r['queue']=='hpc'
            and r['source']==str(SOURCE) and r['source_manifest_sha256']==manifest and r['git_revision_reference']==REVISION
            and r['arm']==arm and r['phase']==phase and r['index']==index and r['inputs']==inputs
            and r['actual_driver_executed'] is False and r['gpu_startup_passed'] is False
            and r['model_loaded'] is False and r['gradient_passed'] is False and r['scientific_readiness_passed'] is False
            and r['optimizer_updates']==0 and r['saved_results_changed'] is False,'CPU command metadata cannot certify GPU startup/science')
    owner=Path(record['owner_parent']);require(owner.parent==ROOT,'CPU command owner escapes campaign')
    submission=read(owner/'submission.json')
    require(submission['outcome']=='submitted' and submission['job_id']==record['owner_job_id']==r['job_id']
            and submission['phase'] in ('after-cpu','after-native','after-segment','dispatch')
            and submission['args']==command(owner.name,submission['phase'],manifest,submission['dependency']),
            'CPU command owner submission/resources differs')
    expected=[owner/'submission.json',SOURCE/'source_manifest.json',SOURCE/'scripts/extended_startup.py',*[Path(v['path']) for v in inputs.values()]]
    require({p['path']:p['sha256'] for p in record['proofs']}=={str(p):sha(p) for p in expected},'CPU command input/proof coverage differs')
    args=r['args'];require(args.count('--output')==1,'Missing exact disposable output')
    require(args==startup_args(arm,phase,index,manifest,inputs,args[args.index('--output')+1]),'Actual generated CPU command differs')
    return record

def ensure_command_capture(arm,phase,index,manifest,owner):
    path=command_path(arm,phase,index)
    if path.exists():return verify_command_capture(arm,phase,index,manifest)
    started=path.with_name(path.stem+'-started.json');require(not started.exists(),'Unknown command-capture outcome; no replay')
    require(os.environ.get('LSB_QUEUE')=='hpc' and os.environ.get('TMPDIR'),'CPU command capture requires allocated hpc/node TMP')
    owner=Path(owner);submission=read(owner/'submission.json');executing(owner,submission['phase'],manifest)
    exclusive(started,{'outcome':'unknown','owner_job_id':os.environ['LSB_JOBID'],'arm':arm,'phase':phase,'index':index})
    from extended_startup import capture_cpu
    result=capture_cpu(sys.modules[__name__],manifest,Path(os.environ['TMPDIR'])/f'pro-command-{arm}-{phase}-{index:04d}',arm=arm,index=index,phase=phase)
    inputs=startup_inputs(arm,phase,index,manifest)
    paths=[owner/'submission.json',SOURCE/'source_manifest.json',SOURCE/'scripts/extended_startup.py',*[Path(v['path']) for v in inputs.values()]]
    exclusive(path,{'status':'passed','version':STARTUP_VERSION,'source_manifest_sha256':manifest,'arm':arm,'phase':phase,'index':index,
                    'owner_job_id':submission['job_id'],'owner_parent':str(owner),'result':result,
                    'scope':'Actual command/input metadata only; real GPU startup remains pending',
                    'proofs':[{'path':str(p),'sha256':sha(p)} for p in paths]})
    return verify_command_capture(arm,phase,index,manifest)

def verify_gpu_startup(arm,phase,index,manifest,parent):
    cpu=verify_command_capture(arm,phase,index,manifest)
    parent=Path(parent);path=parent/'startup-gpu.json';record=read(path);verify_proofs(record)
    r=record['result'];inputs=startup_inputs(arm,phase,index,manifest);submission=read(parent/'submission.json')
    require(record['source_manifest_sha256']==manifest and record['version']==STARTUP_VERSION
            and record['arm']==arm and record['phase']==phase and record['index']==index,
            'GPU startup certificate applicability differs')
    require(parent==ROOT/name(arm,phase,index) and submission['outcome']=='submitted'
            and submission['job_id']==r['job_id'] and submission['phase']==phase
            and submission['args']==command(parent.name,phase,manifest,submission['dependency']),
            'Actual GPU startup owner/resources/source differs')
    require(r['version']==STARTUP_VERSION and r['kind']=='gpu_startup_only' and r['status']=='startup_only_passed'
            and r['queue']=='gpul40s' and r['source']==str(SOURCE) and r['source_manifest_sha256']==manifest
            and r['git_revision_reference']==REVISION and r['arm']==arm and r['phase']==phase and r['index']==index
            and r['inputs']==inputs and r['actual_driver_executed'] is True and r['gpu_backend_guard_executed_unchanged'] is True
            and r['model_loaded'] is False and r['gradient_passed'] is False and r['scientific_readiness_passed'] is False
            and r['optimizer_updates']==0 and r['saved_results_changed'] is False,'GPU startup cannot certify science or optimization')
    expected=[parent/'submission.json',command_path(arm,phase,index),SOURCE/'source_manifest.json',SOURCE/'scripts/extended_startup.py',*[Path(v['path']) for v in inputs.values()]]
    require({p['path']:p['sha256'] for p in record['proofs']}=={str(p):sha(p) for p in expected},'GPU startup proof coverage differs')
    args=r['args'];require(args.count('--output')==1 and args==startup_args(arm,phase,index,manifest,inputs,args[args.index('--output')+1]),'GPU actual driver command differs')
    summary=r['summary']
    require(summary['status']=='failed' and summary['error']=='ProtenixPremodelBoundaryReached: Intentional Protenix GPU startup-only boundary; no native model loaded'
            and summary['source_manifest_sha256']==manifest and summary['source']['git_revision_reference']==REVISION
            and not summary.get('model_digest') and not summary.get('native_gradient_checks') and not summary.get('cases'),
            'Disposable startup sentinel must remain a failed pre-model summary')
    return record

def ensure_gpu_startup(arm,phase,index,manifest,parent):
    require(os.environ.get('LSB_QUEUE')=='gpul40s' and os.environ.get('TMPDIR'),'Actual GPU allocation/node TMP required')
    parent=Path(parent);executing(parent,phase,manifest);verify_command_capture(arm,phase,index,manifest)
    path=parent/'startup-gpu.json';started=parent/'startup-gpu-started.json'
    require(not path.exists() and not started.exists(),'Inspect existing/unknown GPU startup; no replay')
    exclusive(started,{'outcome':'unknown','owner_job_id':os.environ['LSB_JOBID'],'arm':arm,'phase':phase,'index':index})
    from extended_startup import gpu_startup
    result=gpu_startup(sys.modules[__name__],manifest,Path(os.environ['TMPDIR'])/f'pro-gpu-startup-{arm}-{phase}-{index:04d}',arm=arm,index=index,phase=phase)
    inputs=startup_inputs(arm,phase,index,manifest)
    paths=[parent/'submission.json',command_path(arm,phase,index),SOURCE/'source_manifest.json',SOURCE/'scripts/extended_startup.py',*[Path(v['path']) for v in inputs.values()]]
    exclusive(path,{'status':'passed','version':STARTUP_VERSION,'source_manifest_sha256':manifest,'arm':arm,'phase':phase,'index':index,
                    'result':result,'scope':'Actual unchanged GPU/FP32/provenance startup only; native scientific checks still mandatory',
                    'proofs':[{'path':str(p),'sha256':sha(p)} for p in paths]})
    return verify_gpu_startup(arm,phase,index,manifest,parent)

def external_budgets(full_required):
    peers={};missing=[]
    declarations=[('baseline',BASELINE,BASELINE_SOURCE,('boltz2','boltz2_chemistry'),(3,3),
                   'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'),
                  ('dbmol',DBMOL,DBMOL_SOURCE,('boltz2_dbmol',),(4,),
                   '600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed')]
    for key,root,source,arms,segments,manifest in declarations:
        projection_path=root/'projection.json'
        if not projection_path.exists():
            require(not any(root.glob('*-segment-*')) and not (root/'long-submissions.json').exists(),'Peer optimization/unknown markers lack full projection')
            require(not full_required,'Await authenticated full peer native projections: '+key)
            current=sum(tree(path) for arm in arms for path in root.glob((arm if key=='baseline' else 'pilot')+'-*'))
            budget=len(arms)*GIB;require(0<=current<=budget,'Pending peer native allowance exceeded')
            peers[key]={'full':budget,'current':current,'remaining':budget-current,'scope':'native validation allowance only'}
            missing.append(key);continue
        require(sha(source/'source_manifest.json')==manifest,'Peer scientific source changed')
        cpu_path=root/'cpu-certificate.json';cpu=read(cpu_path);verify_proofs(cpu)
        require(cpu['source_manifest_sha256']==manifest,'Peer CPU source differs')
        base=read(projection_path)
        if key=='baseline':
            require(set(base)==set(arms),'Both baseline projections required')
            units={arm:base[arm]['unit_bytes'] for arm in arms}
        else:
            verify_proofs(base);require(base['source_manifest_sha256']==manifest,'DBMol projection source differs')
            units={arms[0]:base['unit_bytes']}
        current=0;budget=0
        for arm,count in zip(arms,segments):
            observed=[]
            pattern=(arm if key=='baseline' else 'pilot')+'-native-*-certificate.json'
            for path in sorted(root.glob(pattern)):
                cert=read(path);verify_proofs(cert);require(cert['source_manifest_sha256']==manifest,'Peer native source differs')
                producer=root/path.name.removesuffix('-certificate.json');submission=read(producer/'submission.json')
                require(submission['outcome']=='submitted' and submission['phase']=='native','Peer native actual producer missing')
                ready_path=producer/'payload/study/assessment.json';ready=read(ready_path)
                config_path=source/('examples/configs/boltz_serial_'+arm+'.json' if key=='baseline' else 'examples/configs/dbmol_pilot.json')
                require(ready['status']=='passed' and ready['source_manifest_sha256']==manifest
                        and ready['config_sha256']==sha(config_path)
                        and any(p['path']==str(ready_path) and p['sha256']==sha(ready_path) for p in cert['proofs']),
                        'Peer readiness source/config/proofs differ')
                unit=ready['storage_projection']['maximum_pilot_update_bytes'];require(type(unit) is int and unit>0,'Invalid native peer projection unit')
                observed.append(unit)
            require(observed and type(units[arm]) is int and units[arm]==max(observed),'Peer full projection is not all authenticated native maxima')
            budget+=2*max(observed)*(count*10000 if key=='baseline' else 10000)+count*GIB
            current+=sum(tree(path) for path in root.glob((arm if key=='baseline' else 'pilot')+'-*'))
        require(0<=current<=budget,'Peer full output projection exceeded')
        peers[key]={'full':budget,'current':current,'remaining':budget-current,'source_manifest_sha256':manifest,
                    'projection_sha256':sha(projection_path),'scope':'Declared mutable capacity metadata; exact all authenticated native maxima, no lowering'}
    return peers,missing

def dispatch(manifest,parent):
    values=read(ROOT/'projection.json') if (ROOT/'projection.json').exists() else {}
    if set(values)!=set(ARMS):
        exclusive(Path(parent)/'waiting.json',{'status':'waiting_own_native_projections','long_submitted':False});return
    _,missing=external_budgets(False)
    if missing:
        exclusive(Path(parent)/'waiting.json',{'status':'waiting_peer_native_projections','missing':missing,'long_submitted':False,
                    'required_action':'New declared CPU dispatch after authenticated peer projections; never replay callbacks or waive full budgets'});return
    for arm in ARMS:
        for index in range(COUNTS[arm]):
            cert=ROOT/(name(arm,'native',index)+'-certificate.json')
            if not cert.exists():continue
            label=name(arm,'segment',index)
            if (ROOT/label).exists():
                require(read(ROOT/label/'submission.json')['outcome']=='submitted','Unknown segment outcome blocks duplicate');continue
            verify_proofs(read(cert));native_assessment(arm,index,manifest)
            storage_gate(parent,'dispatch-'+arm+'-'+str(index),all_budgets(),existing_full=True)
            ensure_command_capture(arm,'segment',index,manifest,parent)
            sub=submit(label,'segment',manifest);submit(name(arm,'after-segment',index),'after-segment',manifest,'ended('+sub['job_id']+')')


def tree(path):
    import fixed25_lrreset_storage as storage
    return storage.tree_bytes(path)
def storage_gate(parent,phase,full=None,*,existing_full=False):
    import fixed25_lrreset_storage as storage
    p=Path(parent)/('storage-'+phase);p.mkdir(exist_ok=False)
    r={'status':'failed','phase':phase,'reserve_bytes':4*GIB,'file_reserve':100000,
       'scope':'All original remaining +report2GiB/diagnostic512MiB +both Pro02 +current Boltz30k +DBMol10k +bounded noisyPro; no reservations',
       'stale_scope':'Entire analysis BASE includes original/new Pro/Boltz/DBMol roots/releases/uploads; external noise paths additionally charged'}
    try:
        old={a:tree(OLD/('run-'+a)) for a in OLD_FULL};require(all(0<=old[a]<=OLD_FULL[a] for a in OLD_FULL),'Original projection exceeded')
        own={a:sum(tree(item) for item in ROOT.glob(a+'-*')) for a in ARMS}
        budgets={a:GIB for a in ARMS} if full is None else full
        require(set(budgets)==set(ARMS) and all(type(budgets[a]) is int and budgets[a]>0 and 0<=own[a]<=budgets[a] for a in ARMS),'Extended output exceeded full projection')
        peers,missing=external_budgets(existing_full)
        noisy=tree(NOISY_ROOT);require(0<=noisy<=GIB,'Noisy Pro output exceeds bound')
        noise_mode='pending_validation_allowance_only'
        if NOISY_ROOT.exists():
            manifest=NOISY_RELEASE/'source/source_manifest.json';stage=NOISY_RELEASE/'source'/NOISY_STAGE
            bound=manifest.is_file() and sha(manifest)==NOISY_MANIFEST and stage.is_file() and sha(stage)==NOISY_STAGE_SHA and read(manifest)['files'].get(NOISY_STAGE)==NOISY_STAGE_SHA
            require(bound or not existing_full,'Await authenticated noisy Pro persistent-output cap')
            if bound:noise_mode='authenticated_immutable_one_GiB_cap'
        projected=sum(OLD_FULL[a]-old[a] for a in OLD_FULL)+sum(budgets[a]-own[a] for a in ARMS)+sum(v['remaining'] for v in peers.values())+(GIB-noisy)+2*GIB+512*1024**2
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        stale=storage.stale_work_bytes()+tree(NOISY_ROOT)+tree(NOISY_RELEASE)+tree(NOISY_UPLOAD)
        gate=storage.gate_record('work3',raw,projected,stale,shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        r.update(status=gate['status'],old_current=old,own_current=own,full_extended=budgets,
                 peers=peers,missing_full_projections=missing,full_long_admission=existing_full,
                 noisy_protenix={'full':GIB,'current':noisy,'remaining':GIB-noisy,'scope':noise_mode},projected_bytes=projected,work3=gate)
    except Exception as e:r['error']=repr(e);raise
    finally:exclusive(p/'gate.json',r)
    require(r['status']=='passed','Whole extended admission failed');return r
def all_budgets():
    values=read(ROOT/'projection.json');require(set(values)==set(ARMS),'Both full Pro native projections required')
    manifest=sha(SOURCE/'source_manifest.json');budgets={}
    for arm in ARMS:
        observed=[]
        for path in sorted(ROOT.glob(arm+'-native-*-certificate.json')):
            cert=read(path);verify_proofs(cert);require(cert['source_manifest_sha256']==manifest,'Own native source differs')
            _,_,index=decode(path.name.removesuffix('-certificate.json'))
            ready_path,ready=native_assessment(arm,index,manifest)
            require(any(proof['path']==str(ready_path) and proof['sha256']==sha(ready_path) for proof in cert['proofs']),'Readiness not certificate-bound')
            observed.append(ready['storage_projection']['maximum_pilot_update_bytes'])
        require(observed and values[arm]['unit_bytes']==max(observed),'Own projection differs from all authenticated pilot maxima')
        budgets[arm]=projection(max(observed),COUNTS[arm]*10000,COUNTS[arm])
    return budgets
def input_snapshot(arm,index):
    if index==0:
        if arm=='random':return None
        return read(SOURCE/'docs/extended_parent.json')['snapshot']
    p=ROOT/(name(arm,'segment',index-1)+'-certificate.json');c=read(p);verify_proofs(c)
    require(c['end']==offset(arm,index) and c['source_manifest_sha256']==sha(SOURCE/'source_manifest.json'),'Predecessor global/source identity differs')
    snapshot=c['terminal_snapshot'];require(sha(snapshot['path'])==snapshot['sha256'],'Predecessor checkpoint changed');return snapshot
def native_assessment(arm,index,manifest):
    from campaign25_contract import verify_readiness
    p=ROOT/name(arm,'native',index)/'payload/study/assessment.json';a=verify_readiness(p,sha(p),config(arm),manifest);s=read(p.with_name('summary.json'));execution=require_boundary(s,a)
    expected=input_snapshot(arm,index)
    require(execution['start']==offset(arm,index) and execution['end']==offset(arm,index)+10000 and execution['input_snapshot_sha256']==(None if expected is None else expected['sha256']),'Native boundary differs from exact predecessor')
    require(a['model_digest']==MODEL and a['context_digest']==CONTEXT and a['pocket_definition_sha256']==POCKET and a['diffusion_gradient'] is False,'Frozen inputs changed')
    require(s['status']=='passed' and s['native_hard_parity']['status']=='passed' and s['native_gradient_checks']['status']=='passed' and s['frozen_identity_after']=={'model_digest':MODEL,'context_digest':CONTEXT},'Native raw/parity/frozen gate incomplete')
    if arm=='random':
        require(s['chemistry_assay']['status']=='passed','Chemistry nonzero gate failed')
        for label in ['low','high']:require(read(p.parent/(label+'_coupled_checks/report.json'))['status']=='passed','Required coupled gate failed')
    require(a['runtime_gate']['status']=='passed' and a['runtime_gate']['conservative_projected_seconds']<82800,'Segment runtime failed')
    unit=a['storage_projection']['maximum_pilot_update_bytes'];require(type(unit) is int and unit>0 and a['storage_projection']['projected_bytes']==2*unit*10000+1024**3,'Pilot projection changed')
    return p,a
def queue_native(arm,index,manifest,owner):
    ensure_command_capture(arm,'native',index,manifest,owner)
    native=name(arm,'native',index);callback=name(arm,'after-native',index)
    sub=submit(native,'native',manifest);submit(callback,'after-native',manifest,'ended('+sub['job_id']+')')
def certify_cpu(manifest):
    proofs=final_job(ROOT/'cpu','hpc');counts={}
    for group,minimum in [('hpc-contract',444),('legacy',238),('native-mapping',33),('chemistry',58)]:
        xml=ROOT/'cpu/payload'/(group+'.xml');log=xml.with_suffix('.log');t=ET.parse(xml);cases=list(t.iter('testcase'))
        require(len(cases)>minimum and not any(list(t.iter(k)) for k in ['failure','error','skipped']),'Full inherited/new CPU group incomplete')
        require(re.search(r'\b'+str(len(cases))+r'\s+passed\b',log.read_text()),'Raw XML/log count differs')
        if group=='hpc-contract':
            found={part for c in cases for part in c.get('classname','').split('.')};require({'test_random_chem','test_extended_protenix','test_protenix_initialization'}<=found,'Missing required new/inherited attribution')
        counts[group]=len(cases)
        for v in [xml,log]:proofs.append({'path':str(v),'sha256':sha(v)})
    require(counts['legacy']==239 and counts['native-mapping']==34 and counts['chemistry']==59,'Changed inherited suite')
    require(counts['hpc-contract']==500 and sum(counts.values())==832,'Exact original832 CPU rows changed')
    xml=ROOT/'cpu/payload/startup.xml';log=xml.with_suffix('.log');parsed=ET.parse(xml);cases=list(parsed.iter('testcase'))
    require(len(cases)==STARTUP_CASES and not any(list(parsed.iter(k)) for k in ['failure','error','skipped'])
            and re.search(r'\b'+str(STARTUP_CASES)+r'\s+passed\b',log.read_text())
            and all('test_extended_startup' in c.get('classname','').split('.') for c in cases),'Allocated startup fixture group incomplete')
    counts['startup']=len(cases)
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [xml,log]]

    for n in ['composition_reference.json','pocket_definition.json','assessment.json','oracles.json']:
        p=ROOT/'cpu/payload'/n;proofs.append({'path':str(p),'sha256':sha(p)})
    require(sha(ROOT/'cpu/payload/pocket_definition.json')==POCKET,'Pocket changed')
    adoption=verify_parent(read(SOURCE/'docs/extended_parent.json'),sha,read)
    retirement=read(SOURCE/'docs/extended_retirement.json');require(retirement['status']=='intentionally_retired_unstarted','Superseded old job not retired')
    exclusive(ROOT/'parent-adoption.json',adoption)
    for p in [ROOT/'parent-adoption.json',SOURCE/'docs/extended_parent.json',SOURCE/'docs/extended_retirement.json']:proofs.append({'path':str(p),'sha256':sha(p)})
    exclusive(ROOT/'cpu-certificate.json',{'status':'passed','suite':'protenix-serial50k-v1','source_manifest_sha256':manifest,'cases':sum(counts.values()),'groups':counts,'proofs':proofs})
    storage_gate(ROOT/'after-cpu','native-dispatch')
    for arm in ARMS:queue_native(arm,0,manifest,ROOT/'after-cpu')
def after_native(arm,index,manifest):
    producer=ROOT/name(arm,'native',index)
    proofs=final_job(producer,'gpul40s');p,a=native_assessment(arm,index,manifest)
    verify_gpu_startup(arm,'native',index,manifest,producer)
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [p,producer/'startup-gpu.json',command_path(arm,'native',index),producer/'payload/driver-invocation.json']]
    exclusive(ROOT/(name(arm,'native',index)+'-certificate.json'),{'status':'passed','proofs':proofs,'source_manifest_sha256':manifest,'start':offset(arm,index)})
    path=ROOT/'projection.json';values=read(path) if path.exists() else {}
    values[arm]={'unit_bytes':max(values.get(arm,{}).get('unit_bytes',0),a['storage_projection']['maximum_pilot_update_bytes'])}
    atomic(path,values)
    dispatch(manifest,ROOT/name(arm,'after-native',index))
def certify_segment(arm,index,manifest):
    from campaign25_contract import verify_attempt_rates
    parent=ROOT/name(arm,'segment',index);proofs=final_job(parent,'gpul40s');p=parent/'payload/study';s=read(p/'summary.json');execution=require_boundary(s)
    require(s['status']=='passed' and s['terminal_gradient_checks']['status']=='passed' and s['native_gradient_checks']['status']=='passed' and s['native_hard_parity']['status']=='passed','Segment scientific acceptance failed')
    require(s['source_manifest_sha256']==manifest and s['config_sha256']==sha(config(arm)) and s['frozen_identity_after']=={'model_digest':MODEL,'context_digest':CONTEXT},'Segment inputs/frozen differ')
    require(execution['start']==offset(arm,index) and execution['end']==offset(arm,index)+10000,'Segment range differs')
    case=read(p/'case-0/report.json');wrapped=s['cases'][0]
    require({k:v for k,v in wrapped.items() if k not in ('label','initialization')}==case and case['status']=='passed' and len(case['updates'])==10000 and len(case['states'])==10001,'Case/summary/full budget mismatch')
    verify_attempt_rates(case,read(config(arm)))
    for field in ['states','updates']:
        with (p/'case-0'/(field+'.jsonl')).open() as f:
            for n,line in enumerate(f):
                r=json.loads(line);require(line.endswith('\n') and r==case[field][n] and r['step']==n and r['global_step']==offset(arm,index)+n,'Committed contiguous JSONL differs')
            require(n+1==len(case[field]),'Incomplete committed stream')
        fpath=p/'case-0'/(field+'.jsonl');proofs.append({'path':str(fpath),'sha256':sha(fpath)})
    for path in [p/'summary.json',p/'case-0/report.json',p/'terminal_gradient_checks/report.json']:proofs.append({'path':str(path),'sha256':sha(path)})
    verify_gpu_startup(arm,'segment',index,manifest,parent)
    for state in case['states']:
        path=p/'case-0'/state['snapshot']
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to((p/'case-0').resolve()) and sha(path)==state['snapshot_sha256'],'Saved state snapshot changed')
        proofs.append({'path':str(path),'sha256':sha(path)})
    proofs += [{'path':str(v),'sha256':sha(v)} for v in [parent/'startup-gpu.json',command_path(arm,'segment',index),parent/'payload/driver-invocation.json']]

    state=case['states'][-1];snapshot=p/'case-0'/state['snapshot'];require(sha(snapshot)==state['snapshot_sha256'],'Terminal snapshot bytes differ')
    exclusive(ROOT/(name(arm,'segment',index)+'-certificate.json'),{'status':'passed','source_manifest_sha256':manifest,'start':offset(arm,index),'end':offset(arm,index)+10000,'proofs':proofs,'terminal_snapshot':{'path':str(snapshot),'sha256':sha(snapshot),'bytes':snapshot.stat().st_size}})
    if index+1<COUNTS[arm]:
        storage_gate(ROOT/name(arm,'after-segment',index),'next-native',all_budgets(),existing_full=True);queue_native(arm,index+1,manifest,ROOT/name(arm,'after-segment',index))
    else:
        chain=[read(ROOT/(name(arm,'segment',i)+'-certificate.json')) for i in range(COUNTS[arm])]
        for c in chain:verify_proofs(c)
        require([c['start'] for c in chain]==list(range(START[arm],50000,10000)) and chain[-1]['end']==50000,'Full chain incomplete')
        exclusive(ROOT/(arm+'-completion.json'),{'status':'passed','arm':arm,'target_global_updates':50000,'new_updates':COUNTS[arm]*10000,'original_parent_updates':START[arm],'segments':chain,'original_results_unchanged':True})
def produce(parent,phase,arm,index,manifest):
    cpu=cpu_certificate(manifest);snapshot=input_snapshot(arm,index);ready=None
    if phase=='segment':ready,_=native_assessment(arm,index,manifest)
    storage_gate(parent,'entry',all_budgets() if (ROOT/'projection.json').exists() and set(read(ROOT/'projection.json'))==set(ARMS) else None,existing_full=phase=='segment')
    output=parent/'payload/study';output.parent.mkdir(exist_ok=False)
    args=[PYTHON,str(SOURCE/'examples/brd4_25_atoms.py'),'--mode','validate' if phase=='native' else 'run','--config',str(config(arm)),'--output',str(output),'--manifest',manifest,'--cpu-certificate',str(cpu),'--cpu-certificate-sha256',sha(cpu),'--reference',str(ROOT/'cpu/payload/composition_reference.json'),'--pocket',str(ROOT/'cpu/payload/pocket_definition.json'),'--pocket-sha256',POCKET,'--offset',str(offset(arm,index)),'--segment-steps','10000']
    if snapshot:args+=['--resume-snapshot',snapshot['path'],'--resume-sha256',snapshot['sha256']]
    if ready:args+=['--ready',str(ready),'--ready-sha256',sha(ready)]
    exclusive(parent/'payload/driver-invocation.json',{'args':args,'original_saved_runs_read_only':True})
    subprocess.run(args,cwd=SOURCE,check=True,timeout=6600 if phase=='native' else 81000)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--manifest',required=True);p.add_argument('--parent',type=Path);a=p.parse_args()
    require(a.root.resolve()==ROOT and a.source.resolve()==SOURCE and os.environ.get('LSB_JOBID'),'Wrong/unallocated execution')
    parent=ROOT/'cpu' if a.phase=='cpu-entry' else a.parent;require(parent is not None and parent.resolve().is_relative_to(ROOT),'Parent escaped')
    executing(parent,'cpu' if a.phase=='cpu-entry' else a.phase,a.manifest)
    if a.phase=='cpu-entry':storage_gate(parent,'cpu-entry');return
    if a.phase in ('after-cpu','dispatch'):
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if a.phase=='after-cpu':certify_cpu(a.manifest)
            else:cpu_certificate(a.manifest);dispatch(a.manifest,parent)
        return
    arm,phase,index=decode(parent)
    if phase in ('native','segment'):
        verify_command_capture(arm,phase,index,a.manifest)
        ensure_gpu_startup(arm,phase,index,a.manifest,parent)
        produce(parent,phase,arm,index,a.manifest)
    else:
        cpu_certificate(a.manifest)
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if phase=='after-native':after_native(arm,index,a.manifest)
            elif phase=='after-segment':certify_segment(arm,index,a.manifest)
            else:raise ValueError('Unknown phase')
if __name__=='__main__':main()
