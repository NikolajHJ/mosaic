"""Two independent serial Protenix chains, immutable inputs and exclusive outcomes."""
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from extended_protocol import projection, require_boundary, verify_parent
BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'protenix-extended-01'
SOURCE=BASE/'protenix-extended-release-01/source'
OLD=BASE/'restart-lrreset-04'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
ARMS=('random','continuation')
COUNTS={'random':5,'continuation':4}
START={'random':0,'continuation':10000}
OLD_FULL={'protenix':1493521824,'boltz2':1748021824,'boltz2_chemistry':5144021824}
MODEL='8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
CONTEXT='b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
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
    require(phase in ('after-cpu','after-native','after-segment'),'Unknown phase')
    return ['-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30']
def command(label,phase,manifest,dependency=None):
    if phase not in ('cpu','after-cpu'):
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
def tree(path):
    import fixed25_lrreset_storage as storage
    return storage.tree_bytes(path)
def storage_gate(parent,phase,full=None):
    import fixed25_lrreset_storage as storage
    p=Path(parent)/('storage-'+phase);p.mkdir(exist_ok=False)
    r={'status':'failed','phase':phase,'reserve_bytes':4294967296,'file_reserve':100000,'scope':'All original3 remaining +report2GiB/diagnostic512MiB +both full extended remaining; independent output; no reservation'}
    try:
        old={a:tree(OLD/('run-'+a)) for a in OLD_FULL};require(all(0<=old[a]<=OLD_FULL[a] for a in OLD_FULL),'Original projection exceeded')
        own={a:sum(tree(p) for p in ROOT.glob(a+'-*')) for a in ARMS}
        budgets={a:1024**3 for a in ARMS} if full is None else full
        require(set(budgets)==set(ARMS) and all(type(budgets[a]) is int and budgets[a]>0 and own[a]<=budgets[a] for a in ARMS),'Extended output exceeded full projection')
        projected=sum(OLD_FULL[a]-old[a] for a in OLD_FULL)+sum(budgets[a]-own[a] for a in ARMS)+2*1024**3+512*1024**2
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        gate=storage.gate_record('work3',raw,projected,storage.stale_work_bytes(),shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        r.update(status=gate['status'],old_current=old,own_current=own,full_extended=budgets,projected_bytes=projected,work3=gate)
    except Exception as e:r['error']=repr(e);raise
    finally:exclusive(p/'gate.json',r)
    require(r['status']=='passed','Whole extended admission failed');return r
def all_budgets():
    c=read(ROOT/'projection.json');return {a:projection(c[a]['unit_bytes'],COUNTS[a]*10000,COUNTS[a]) for a in ARMS}
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
def queue_native(arm,index,manifest):
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
    for n in ['composition_reference.json','pocket_definition.json','assessment.json','oracles.json']:
        p=ROOT/'cpu/payload'/n;proofs.append({'path':str(p),'sha256':sha(p)})
    require(sha(ROOT/'cpu/payload/pocket_definition.json')==POCKET,'Pocket changed')
    adoption=verify_parent(read(SOURCE/'docs/extended_parent.json'),sha,read)
    retirement=read(SOURCE/'docs/extended_retirement.json');require(retirement['status']=='intentionally_retired_unstarted','Superseded old job not retired')
    exclusive(ROOT/'parent-adoption.json',adoption)
    for p in [ROOT/'parent-adoption.json',SOURCE/'docs/extended_parent.json',SOURCE/'docs/extended_retirement.json']:proofs.append({'path':str(p),'sha256':sha(p)})
    exclusive(ROOT/'cpu-certificate.json',{'status':'passed','suite':'protenix-serial50k-v1','source_manifest_sha256':manifest,'cases':sum(counts.values()),'groups':counts,'proofs':proofs})
    storage_gate(ROOT/'after-cpu','native-dispatch')
    for arm in ARMS:queue_native(arm,0,manifest)
def after_native(arm,index,manifest):
    proofs=final_job(ROOT/name(arm,'native',index),'gpul40s');p,a=native_assessment(arm,index,manifest);proofs += [{'path':str(p),'sha256':sha(p)}]
    exclusive(ROOT/(name(arm,'native',index)+'-certificate.json'),{'status':'passed','proofs':proofs,'source_manifest_sha256':manifest,'start':offset(arm,index)})
    projection_path=ROOT/'projection.json';values=read(projection_path) if projection_path.exists() else {}
    values[arm]={'unit_bytes':max(values.get(arm,{}).get('unit_bytes',0),a['storage_projection']['maximum_pilot_update_bytes'])}
    atomic(projection_path,values)
    if set(values)!=set(ARMS):return
    targets=ARMS if index==0 else [arm]
    for target in targets:
        target_index=index if target==arm else 0
        cert=ROOT/(name(target,'native',target_index)+'-certificate.json')
        if not cert.exists():continue
        label=name(target,'segment',target_index)
        if (ROOT/label).exists():
            require(read(ROOT/label/'submission.json')['outcome']=='submitted','Unknown outcome blocks dispatch');continue
        verify_proofs(read(cert));native_assessment(target,target_index,manifest)
        storage_gate(ROOT/name(arm,'after-native',index),'dispatch-'+target,all_budgets())
        sub=submit(label,'segment',manifest);submit(name(target,'after-segment',target_index),'after-segment',manifest,'ended('+sub['job_id']+')')
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
    state=case['states'][-1];snapshot=p/'case-0'/state['snapshot'];require(sha(snapshot)==state['snapshot_sha256'],'Terminal snapshot bytes differ')
    exclusive(ROOT/(name(arm,'segment',index)+'-certificate.json'),{'status':'passed','source_manifest_sha256':manifest,'start':offset(arm,index),'end':offset(arm,index)+10000,'proofs':proofs,'terminal_snapshot':{'path':str(snapshot),'sha256':sha(snapshot),'bytes':snapshot.stat().st_size}})
    if index+1<COUNTS[arm]:
        storage_gate(ROOT/name(arm,'after-segment',index),'next-native',all_budgets());queue_native(arm,index+1,manifest)
    else:
        chain=[read(ROOT/(name(arm,'segment',i)+'-certificate.json')) for i in range(COUNTS[arm])]
        for c in chain:verify_proofs(c)
        require([c['start'] for c in chain]==list(range(START[arm],50000,10000)) and chain[-1]['end']==50000,'Full chain incomplete')
        exclusive(ROOT/(arm+'-completion.json'),{'status':'passed','arm':arm,'target_global_updates':50000,'new_updates':COUNTS[arm]*10000,'original_parent_updates':START[arm],'segments':chain,'original_results_unchanged':True})
def produce(parent,phase,arm,index,manifest):
    cpu=cpu_certificate(manifest);snapshot=input_snapshot(arm,index);ready=None
    if phase=='segment':ready,_=native_assessment(arm,index,manifest)
    storage_gate(parent,'entry',all_budgets() if (ROOT/'projection.json').exists() and set(read(ROOT/'projection.json'))==set(ARMS) else None)
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
    if a.phase=='after-cpu':
        with (ROOT/'dispatch.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);certify_cpu(a.manifest)
        return
    arm,phase,index=decode(parent)
    if phase in ('native','segment'):produce(parent,phase,arm,index,a.manifest)
    else:
        cpu_certificate(a.manifest)
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if phase=='after-native':after_native(arm,index,a.manifest)
            elif phase=='after-segment':certify_segment(arm,index,a.manifest)
            else:raise ValueError('Unknown phase')
if __name__=='__main__':main()
