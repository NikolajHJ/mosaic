"""Exclusive, serial DBMol pilot with allocated proofs and whole-campaign budget."""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from dbmol_protocol import require,read,sha,SUITE,SEGMENT,TARGET,verify_readiness,verify_identity,frozen,verify_attempt_rates,verify_native_report,scope

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'boltz-dbmol-pilot-01'
SOURCE=BASE/'boltz-dbmol-pilot-release-01/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
WORKER='/work3/s234818/latentmol/env/boltz311/bin/python'
PREPARED='/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared'
EXTENDED=BASE/'protenix-extended-01'
BASELINE=BASE/'boltz-lr0025-02'
OLD=BASE/'restart-lrreset-04'
OLD_FULL={'protenix':1493521824,'boltz2':1748021824,'boltz2_chemistry':5144021824}
NOISE_PATHS=(Path('/work3/s234818/mosaic/runs/20261023-protenix-noisy-01'),
             Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01'),
             Path('/work3/s234818/mosaic/releases/20261023-protenix-noisy-01.tar.gz'))

def exclusive(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def atomic(path,value):
    p=Path(path);q=p.with_name(p.name+'.tmp');exclusive(q,value);q.replace(p)

def config():return SOURCE/'examples/configs/dbmol_pilot.json'

def label(phase,index):
    require(type(index) is int and 0<=index<4,'Undeclared pilot segment')
    require(phase in ['native','after-native','segment','after-segment'],'Undeclared phase')
    return f'pilot-{phase}-{index:04d}'

def decode(name):
    m=re.fullmatch(r'pilot-(native|after-native|segment|after-segment)-(\d{4})',name)
    require(m is not None,'Undeclared parent');phase,index=m.group(1),int(m.group(2));label(phase,index);return phase,index

def resources(phase):
    if phase=='cpu':return ['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00']
    if phase in ['native','segment']:return ['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00' if phase=='native' else '23:00']
    require(phase in ['after-cpu','after-native','after-segment','dispatch'],'Unknown phase')
    return ['-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30']

def command(name,phase,manifest,dependency=None):
    if phase in ['cpu','after-cpu']:require(name==phase,'Wrong parent')
    elif phase=='dispatch':require(re.fullmatch(r'dispatch-\d{4}',name) is not None,'Wrong dispatch label')
    else:require(decode(name)[0]==phase,'Wrong parent/phase')
    p=ROOT/name
    args=['bsub',*resources(phase),'-J','mosaic-dbmol-'+name,'-oo',str(p/'stdout.%J.log'),'-eo',str(p/'stderr.%J.log')]
    if dependency:args+=['-w',dependency]
    args+=['bash','-l',str(SOURCE/'scripts/hpc'/('dbmol_cpu.lsf' if phase=='cpu' else 'dbmol_stage.lsf')),str(SOURCE),str(ROOT),manifest]
    if phase!='cpu':args+=[phase,str(p)]
    return args

def submit(name,phase,manifest,dependency=None):
    p=ROOT/name;p.mkdir(exist_ok=False);args=command(name,phase,manifest,dependency)
    r={'phase':phase,'args':args,'dependency':dependency,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()}
    exclusive(p/'submission-started.json',r)
    try:
        q=subprocess.run(args,capture_output=True,text=True,timeout=60);r.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
        ids=re.findall(r'Job <(\d+)>',q.stdout)
        if q.returncode==0 and len(ids)==1:r.update(outcome='submitted',job_id=ids[0])
    except Exception as e:r['error']=repr(e)
    finally:exclusive(p/'submission.json',r)
    require(r['outcome']=='submitted','Unknown submission outcome; no duplicate')
    jobs=read(ROOT/'jobs.json');jobs[name]=r['job_id'];atomic(ROOT/'jobs.json',jobs)
    q=subprocess.run(['bjobs','-l',r['job_id']],capture_output=True,text=True,timeout=30);(p/'scheduler-at-submission.txt').write_text(q.stdout+q.stderr)
    return r

def executing(parent,phase,manifest):
    for i in range(16):
        if (parent/'submission.json').exists():break
        require(i<15,'Missing own submission');time.sleep(2)
    r=read(parent/'submission.json')
    require(r['outcome']=='submitted' and r['job_id']==os.environ.get('LSB_JOBID')
        and r['phase']==phase and r['args']==command(parent.name,phase,manifest,r['dependency']),'Actual own submission/resources differ')

def final_job(parent,queue):
    job=read(parent/'submission.json')['job_id']
    for i in range(16):
        q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=q.stdout+q.stderr
        p=parent/f'scheduler-final-{i}.txt';p.write_text(raw);compact=re.sub(r'\s+','',raw)
        require(q.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Queue<([^>]+)>',compact)==[queue],'No original actual final scheduler')
        status=re.findall(r'Status<([^>]+)>',compact)
        if status==['DONE']:break
        require(status in (['RUN'],['PEND']) and i<15,'Producer failed');time.sleep(2)
    log=parent/('stdout.'+job+'.log');text=log.read_text()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',text,re.M)==[job] and 'Successfully completed.' in text and 'Exited with exit' not in text,'Same-job LSF success missing')
    return [{'path':str(x),'sha256':sha(x)} for x in [p,log,parent/'submission.json']]

def verify_proofs(c):
    require(c['status']=='passed','Certificate not passed')
    for p in c['proofs']:require(sha(p['path'])==p['sha256'],'Authenticated proof changed')

def cpu_certificate(manifest):
    p=ROOT/'cpu-certificate.json';c=read(p)
    require(c['suite']==SUITE and c['source_manifest_sha256']==manifest,'CPU scope differs');verify_proofs(c);return p

def tree(path):
    import fixed25_lrreset_storage as storage
    return storage.tree_bytes(path)

def external_budgets(full):
    """Authenticate peer native projections; pending allowances are not full budgets."""
    peers={};missing=[]
    for key,root,arms,counts in [('extended',EXTENDED,('random','continuation'),(5,4)),('baseline',BASELINE,('boltz2','boltz2_chemistry'),(3,3))]:
        projection=root/'projection.json'
        if not projection.exists() or set(read(projection))!=set(arms):
            require(not any(root.glob('*-segment-*')),'Peer optimization/unknown marker lacks full projection')
            require(not full or not root.exists() or not (root/'jobs.json').exists(),'Peer full native projections not established')
            peers[key]={'remaining':len(arms)*1024**3,'scope':'native-validation allowance only; no full-long pass'}
            missing.append(key);continue
        c=read(root/'cpu-certificate.json');verify_proofs(c)
        require(c['source_manifest_sha256']==sha((BASE/('protenix-extended-release-01' if key=='extended' else 'boltz-lr0025-release-02'))/'source/source_manifest.json'),'Peer source differs')
        units=read(projection);remaining=0
        for arm,count in zip(arms,counts):
            observed=[]
            for cp in sorted(root.glob(arm+'-native-*-certificate.json')):
                cert=read(cp);verify_proofs(cert)
                require(cert['source_manifest_sha256']==c['source_manifest_sha256'],'Peer native/source differs')
                ready_path=root/cp.name.removesuffix('-certificate.json')/'payload/study/assessment.json'
                ready=read(ready_path)
                peer_source=BASE/('protenix-extended-release-01' if key=='extended' else 'boltz-lr0025-release-02')/'source'
                peer_config=peer_source/'examples/configs'/('extended_'+arm+'.json' if key=='extended' else 'boltz_serial_'+arm+'.json')
                require(any(p['path']==str(ready_path) and p['sha256']==sha(ready_path) for p in cert['proofs'])
                    and ready['status']=='passed' and ready['source_manifest_sha256']==c['source_manifest_sha256']
                    and ready['config_sha256']==sha(peer_config)
                    and ready['arm']==('protenix' if key=='extended' else arm),'Peer native assessment/proof differs')
                observed.append(ready['storage_projection']['maximum_pilot_update_bytes'])
            unit=units[arm]['unit_bytes'];require(type(unit) is int and unit>0,'Peer projection invalid')
            require(observed and unit==max(observed),'Peer full projection differs from authenticated pilot maxima')
            maximum=2*unit*count*10000+count*1024**3
            current=sum(tree(p) for p in root.glob(arm+'-*'))
            require(current<=maximum,'Peer full projection exceeded');remaining+=maximum-current
        peers[key]={'remaining':remaining,'scope':'authenticated complete peer projection','cpu_sha256':sha(root/'cpu-certificate.json'),'projection_sha256':sha(projection)}
    if full:require(not missing,'All prospective peers need full native projections')
    return peers,missing

def storage_gate(parent,name,full=False):
    import fixed25_lrreset_storage as storage
    path=Path(parent)/('storage-'+name);path.mkdir(exist_ok=False)
    r={'status':'failed','reserve_bytes':4294967296,'file_reserve':100000,'scope':'all concurrent original+Proextensions+Boltz30k+DBMol remaining, report/diagnostic; no reservation'}
    try:
        peers,missing=external_budgets(full)
        original={a:tree(OLD/('run-'+a)) for a in OLD_FULL}
        require(all(original[a]<=OLD_FULL[a] for a in OLD_FULL),'Original projection exceeded')
        own=sum(tree(p) for p in ROOT.glob('pilot-*'))
        own_full=own_budget()
        require(type(own_full) is int and 0<=own<=own_full,'DBMol projection exceeded')
        projected=sum(OLD_FULL[a]-original[a] for a in OLD_FULL)+sum(v['remaining'] for v in peers.values())+own_full-own+2*1024**3+512*1024**2+1024**3
        r['additional_prospective_noisy_protenix_allowance']=1024**3
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        noise_stale=sum(tree(p) for p in NOISE_PATHS)
        gate=storage.gate_record('work3',raw,projected,storage.stale_work_bytes()+noise_stale,shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        r['additional_noise_stale_bytes']=noise_stale
        r.update(status=gate['status'],work3=gate,peers=peers,missing_full_projections=missing,original_current=original,own_current=own,own_full=own_full,projected_bytes=projected,full_long_admission=full)
    except Exception as e:r['error']=repr(e);raise
    finally:exclusive(path/'gate.json',r)
    require(r['status']=='passed','Whole campaign capacity failed');return r

def own_budget():
    if not (ROOT/'projection.json').exists():
        require(not any(ROOT.glob('pilot-segment-*')),'Unknown optimization without projection')
        return 1024**3
    r=read(ROOT/'projection.json');verify_proofs(r)
    require(r['source_manifest_sha256']==sha(SOURCE/'source_manifest.json'),'Own projection/source differs')
    observed=[]
    for cp in sorted(ROOT.glob('pilot-native-*-certificate.json')):
        c=read(cp);verify_proofs(c)
        _,index=decode(cp.name.removesuffix('-certificate.json'));_,a=readiness(index,r['source_manifest_sha256'])
        observed.append(a['storage_projection']['maximum_pilot_update_bytes'])
    require(observed and r['unit_bytes']==max(observed) and r['full_projected_bytes']==2*max(observed)*TARGET+4*1024**3,'Own full projection differs from actual pilot maxima')
    return r['full_projected_bytes']

def input_snapshot(index):
    if index==0:return None
    c=read(ROOT/(label('segment',index-1)+'-certificate.json'));verify_proofs(c)
    require(c['end']==index*SEGMENT and c['source_manifest_sha256']==sha(SOURCE/'source_manifest.json'),'Predecessor global/source differs')
    z=c['terminal_snapshot'];require(sha(z['path'])==z['sha256'],'Predecessor bytes changed');return z

def readiness(index,manifest):
    p=ROOT/label('native',index)/'payload/study/assessment.json';z=input_snapshot(index)
    execution=scope(index*SEGMENT,SEGMENT,None if z is None else z['path'],None if z is None else z['sha256'])
    a=verify_readiness(p,sha(p),config(),manifest,execution)
    return p,a

def queue_native(index,manifest):
    r=submit(label('native',index),'native',manifest);submit(label('after-native',index),'after-native',manifest,'ended('+r['job_id']+')')

def certify_cpu(manifest):
    proofs=final_job(ROOT/'cpu','hpc');counts={}
    for group,minimum in [('hpc-contract',499),('legacy',238),('native-mapping',33),('chemistry',58),('dbmol-native',0),('dbmol-transport',0)]:
        p=ROOT/'cpu/payload'/(group+'.xml');log=p.with_suffix('.log');t=ET.parse(p);cases=list(t.iter('testcase'))
        require(len(cases)>minimum and not any(list(t.iter(k)) for k in ['failure','error','skipped']),'Allocated inherited/new CPU cases incomplete')
        require(re.search(r'\b'+str(len(cases))+r'\s+passed\b',log.read_text()),'Actual XML/log counts differ')
        if group=='dbmol-native':require({'test_dbmol_objective','test_dbmol_native_paths'}<={c.get('classname','').split('.')[0] for c in cases},'Missing native component fixture attribution')
        if group=='dbmol-transport':require({'test_dbmol_pilot_contract','test_dbmol_gradient_contract','test_dbmol_bridge'}<={c.get('classname','').split('.')[0] for c in cases},'Missing transport/scope fixture attribution')
        counts[group]=len(cases);proofs += [{'path':str(v),'sha256':sha(v)} for v in [p,log]]
    require([counts[k] for k in ['hpc-contract','legacy','native-mapping','chemistry']]==[500,239,34,59],'Inherited actual suite changed')
    for n in ['composition_reference.json','pocket_definition.json','assessment.json','oracles.json']:
        p=ROOT/'cpu/payload'/n;proofs.append({'path':str(p),'sha256':sha(p)})
    exclusive(ROOT/'cpu-certificate.json',{'status':'passed','suite':SUITE,'source_manifest_sha256':manifest,'cases':sum(counts.values()),'groups':counts,'proofs':proofs})
    storage_gate(ROOT/'after-cpu','native-dispatch');queue_native(0,manifest)

def dispatch(parent,manifest):
    cpu_certificate(manifest)
    for index in range(4):
        cert=ROOT/(label('native',index)+'-certificate.json')
        if not cert.exists():continue
        verify_proofs(read(cert));readiness(index,manifest)
        target=ROOT/label('segment',index)
        if target.exists():require(read(target/'submission.json')['outcome']=='submitted','Unknown outcome blocks duplicate');continue
        try:
            _,missing=external_budgets(False)
            if missing:
                exclusive(parent/f'waiting-peer-projections-{index}.json',{'status':'waiting_peer_native_projections','missing':missing,'no_long_submitted':True})
                queue_peer_dispatch(manifest)
                return
            storage_gate(parent,'dispatch-'+str(index),True)
        except Exception:raise
        r=submit(target.name,'segment',manifest);submit(label('after-segment',index),'after-segment',manifest,'ended('+r['job_id']+')')

def queue_peer_dispatch(manifest):
    """One declared CPU recheck after actual peer callbacks; no callback replay."""
    dependencies=[]
    for root,arms in [(EXTENDED,('random','continuation')),(BASELINE,('boltz2','boltz2_chemistry'))]:
        if not (root/'jobs.json').exists():return
        jobs=read(root/'jobs.json')
        for arm in arms:
            name=arm+'-after-native-0000'
            if name not in jobs:return
            r=read(root/name/'submission.json')
            require(r['outcome']=='submitted' and r['job_id']==jobs[name] and r['phase']=='after-native'
                and r['dependency']=='ended('+jobs[arm+'-native-0000']+')','Peer callback actual identity differs')
            pro=root==EXTENDED
            source=BASE/('protenix-extended-release-01' if pro else 'boltz-lr0025-release-02')/'source'
            peer_manifest=sha(source/'source_manifest.json')
            expected=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30',
                '-J',('mosaic-pro50k-' if pro else 'mosaic-bol30k-')+name,
                '-oo',str(root/name/'stdout.%J.log'),'-eo',str(root/name/'stderr.%J.log'),
                '-w',r['dependency'],'bash','-l',str(source/'scripts/hpc'/('extended_stage.lsf' if pro else 'boltz_serial_stage.lsf')),
                str(source),str(root),peer_manifest,'after-native',str(root/name)]
            require(r['args']==expected,'Peer callback source/resources vector differs')
            dependencies.append('done('+r['job_id']+')')
    dependency=' && '.join(dependencies)
    parent=ROOT/'dispatch-0001'
    if parent.exists():
        require((parent/'submission.json').exists(),'Unknown dispatch outcome blocks duplicates')
        r=read(parent/'submission.json');require(r['outcome']=='submitted' and r['args']==command(parent.name,'dispatch',manifest,dependency),'Mismatched dispatch outcome')
        return r
    return submit('dispatch-0001','dispatch',manifest,dependency)

def after_native(parent,index,manifest):
    proofs=final_job(ROOT/label('native',index),'gpul40s');p,a=readiness(index,manifest);proofs.append({'path':str(p),'sha256':sha(p)})
    cert={'status':'passed','source_manifest_sha256':manifest,'start':index*SEGMENT,'proofs':proofs}
    exclusive(ROOT/(label('native',index)+'-certificate.json'),cert)
    unit=a['storage_projection']['maximum_pilot_update_bytes'];require(type(unit) is int and unit>0,'Invalid native pilot projection')
    old=read(ROOT/'projection.json')['unit_bytes'] if (ROOT/'projection.json').exists() else 0
    unit=max(unit,old)
    atomic(ROOT/'projection.json',{'status':'passed','unit_bytes':unit,'full_projected_bytes':2*unit*TARGET+4*1024**3,'source_manifest_sha256':manifest,'producer_job':read((p.parent.parent.parent)/'submission.json')['job_id'],'proofs':proofs})
    dispatch(parent,manifest)

def certify_segment(parent,index,manifest):
    producer=ROOT/label('segment',index);proofs=final_job(producer,'gpul40s');root=producer/'payload/study';s=read(root/'summary.json');a_path,a=readiness(index,manifest)
    require(s['status']=='passed' and s['config_sha256']==sha(config()) and s['source_manifest_sha256']==manifest and s['execution']==a['execution'],'Segment status/input/global scope differs')
    require(s['native_gradient_checks']['status']==s['terminal_gradient_checks']['status']=='passed','Initial/terminal component native gate failed')
    verify_native_report(s['native_gradient_checks']);verify_native_report(s['terminal_gradient_checks'])
    frozen(s['frozen_identity_after'],a['native_identity'])
    case=read(root/'case-0/report.json');require(s['cases']==[case] and case['status']=='passed' and len(case['updates'])==SEGMENT and len(case['states'])==SEGMENT+1,'Case/full segment differs')
    verify_attempt_rates(case,read(config()))
    for field in ['states','updates']:
        with (root/'case-0'/(field+'.jsonl')).open() as f:
            n=-1
            for n,line in enumerate(f):require(line.endswith('\n') and json.loads(line)==case[field][n] and case[field][n]['step']==n and case[field][n]['global_step']==index*SEGMENT+n,'Committed JSONL/global equality failed')
            require(n+1==len(case[field]),'Committed prefix incomplete')
        p=root/'case-0'/(field+'.jsonl');proofs.append({'path':str(p),'sha256':sha(p)})
    for state in case['states']:
        p=root/'case-0'/state['snapshot'];require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to((root/'case-0').resolve()) and sha(p)==state['snapshot_sha256'],'A saved probability frame changed')
        proofs.append({'path':str(p),'sha256':sha(p)})
    for p in [root/'summary.json',root/'case-0/report.json',root/'terminal_gradient_checks/report.json']:proofs.append({'path':str(p),'sha256':sha(p)})
    state=case['states'][-1];z=root/'case-0'/state['snapshot'];require(sha(z)==state['snapshot_sha256'],'Terminal snapshot differs')
    exclusive(ROOT/(label('segment',index)+'-certificate.json'),{'status':'passed','source_manifest_sha256':manifest,'start':index*SEGMENT,'end':(index+1)*SEGMENT,'proofs':proofs,'terminal_snapshot':{'path':str(z),'sha256':sha(z),'bytes':z.stat().st_size}})
    if index<3:storage_gate(parent,'next-native',True);queue_native(index+1,manifest)
    else:
        chain=[read(ROOT/(label('segment',i)+'-certificate.json')) for i in range(4)]
        for c in chain:verify_proofs(c)
        require([c['start'] for c in chain]==[0,2500,5000,7500] and chain[-1]['end']==10000,'Full requested pilot missing')
        exclusive(ROOT/'completion.json',{'status':'passed','requested_updates':10000,'lr':.0025,'segments':chain,'saved_originals_unchanged':True})

def produce(parent,phase,index,manifest):
    cpu=cpu_certificate(manifest);z=input_snapshot(index);a=None
    if phase=='segment':a,_=readiness(index,manifest)
    storage_gate(parent,'entry',phase=='segment')
    out=parent/'payload/study';out.parent.mkdir(exist_ok=False)
    from dbmol_protocol import CONTEXT_FILE,POCKET
    args=[PYTHON,str(SOURCE/'scripts/dbmol_driver.py'),'--mode','validate' if phase=='native' else 'run','--config',str(config()),'--output',str(out),'--manifest',manifest,'--cpu-certificate',str(cpu),'--cpu-certificate-sha256',sha(cpu),'--reference',str(ROOT/'cpu/payload/composition_reference.json'),'--pocket',str(ROOT/'cpu/payload/pocket_definition.json'),'--pocket-sha256',POCKET,'--prepared',PREPARED,'--context-sha256',CONTEXT_FILE,'--worker-python',WORKER,'--offset',str(index*SEGMENT),'--segment-steps',str(SEGMENT)]
    if z:args+=['--resume-snapshot',z['path'],'--resume-sha256',z['sha256']]
    if a:args+=['--ready',str(a),'--ready-sha256',sha(a)]
    exclusive(parent/'payload/driver-invocation.json',{'args':args,'original_saved_results_read_only':True})
    subprocess.run(args,cwd=SOURCE,check=True,timeout=6600 if phase=='native' else 81000)

def main():
    p=argparse.ArgumentParser()
    for k in ['root','source','parent']:p.add_argument('--'+k,type=Path)
    p.add_argument('--phase',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
    require(a.root.resolve()==ROOT and a.source.resolve()==SOURCE and os.environ.get('LSB_JOBID'),'Unallocated/wrong source/root')
    parent=ROOT/'cpu' if a.phase=='cpu-entry' else a.parent
    require(parent is not None and parent.resolve().is_relative_to(ROOT),'Parent escapes output root')
    executing(parent,'cpu' if a.phase=='cpu-entry' else a.phase,a.manifest)
    if a.phase=='cpu-entry':storage_gate(parent,'cpu-entry');return
    if a.phase in ['native','segment']:
        phase,index=decode(parent.name);produce(parent,phase,index,a.manifest);return
    cpu_certificate(a.manifest) if a.phase!='after-cpu' else None
    with (ROOT/'dispatch.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if a.phase=='after-cpu':certify_cpu(a.manifest)
        elif a.phase=='dispatch':dispatch(parent,a.manifest)
        else:
            phase,index=decode(parent.name)
            if phase=='after-native':after_native(parent,index,a.manifest)
            elif phase=='after-segment':certify_segment(parent,index,a.manifest)
            else:raise ValueError('Unknown callback')

if __name__=='__main__':main()
