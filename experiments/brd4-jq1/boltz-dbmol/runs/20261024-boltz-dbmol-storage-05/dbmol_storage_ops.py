"""Exclusive, serial DBMol pilot with allocated proofs and whole-campaign budget."""
import argparse,fcntl,hashlib,json,os,re,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from dbmol_protocol import require,read,sha,SUITE,SEGMENT,TARGET,verify_readiness,verify_identity,frozen,verify_attempt_rates,verify_native_report,scope

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'boltz-dbmol-pilot-02'
HISTORY_ROOT=BASE/'boltz-dbmol-pilot-01'
SOURCE=BASE/'boltz-dbmol-pilot-release-02/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
WORKER='/work3/s234818/latentmol/env/boltz311/bin/python'
PREPARED='/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared'
EXTENDED=BASE/'protenix-extended-01'
BASELINE=BASE/'boltz-lr0025-03'
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
    require(manifest==SCIENCE_MANIFEST and LAUNCHER_MANIFEST is not None,'Missing exact dual source identities')
    if phase in ['cpu','after-cpu']:require(name==(CPU_PARENT if phase=='cpu' else AFTER_CPU_PARENT),'Wrong adopted CPU parent')
    elif phase=='dispatch':require(re.fullmatch(r'dispatch-\d{4}',name) is not None,'Wrong dispatch label')
    else:require(decode(name)[0]==phase,'Wrong parent/phase')
    p=ROOT/name
    args=['bsub',*resources(phase),'-J','mosaic-dbmol-'+name,'-oo',str(p/'stdout.%J.log'),'-eo',str(p/'stderr.%J.log')]
    if dependency:args+=['-w',dependency]
    args+=['bash','-l',str(LAUNCHER/'scripts/hpc'/('dbmol_storage_cpu.lsf' if phase=='cpu' else 'dbmol_storage_stage.lsf')),str(LAUNCHER),LAUNCHER_MANIFEST,str(SOURCE),str(ROOT),manifest]
    if phase!='cpu':args+=[phase,str(p)]
    return args

def submit(name,phase,manifest,dependency=None):
    startup=None
    if phase in ['native','segment']:
        _,index=decode(name);startup=verify_startup_entry(phase,index,manifest)
    p=ROOT/name;p.mkdir(exist_ok=False);args=command(name,phase,manifest,dependency)
    r={'phase':phase,'args':args,'dependency':dependency,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()}
    if startup is not None:r['startup_smoke']=startup
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
    require(c['suite']==SUITE and c['source_manifest_sha256']==manifest==SCIENCE_MANIFEST and c['launcher_manifest_sha256']==LAUNCHER_MANIFEST and c['launcher_source']==str(LAUNCHER),'CPU dual-source scope differs');verify_launcher(manifest,LAUNCHER_MANIFEST);verify_proofs(c);return p

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
        require(c['source_manifest_sha256']==sha((BASE/('protenix-extended-release-01' if key=='extended' else 'boltz-lr0025-release-03'))/'source/source_manifest.json'),'Peer source differs')
        units=read(projection);remaining=0
        for arm,count in zip(arms,counts):
            observed=[]
            for cp in sorted(root.glob(arm+'-native-*-certificate.json')):
                cert=read(cp);verify_proofs(cert)
                require(cert['source_manifest_sha256']==c['source_manifest_sha256'],'Peer native/source differs')
                ready_path=root/cp.name.removesuffix('-certificate.json')/'payload/study/assessment.json'
                ready=read(ready_path)
                peer_source=BASE/('protenix-extended-release-01' if key=='extended' else 'boltz-lr0025-release-03')/'source'
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
        original_future,terminal_evidence=original_remaining(original)
        own=sum(tree(p) for p in ROOT.glob('pilot-*'))
        own_full=own_budget()
        require(type(own_full) is int and 0<=own<=own_full,'DBMol projection exceeded')
        projected=sum(original_future.values())+sum(v['remaining'] for v in peers.values())+own_full-own+2*1024**3+512*1024**2+1024**3
        r['additional_prospective_noisy_protenix_allowance']=1024**3
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        noise_stale=sum(tree(p) for p in NOISE_PATHS)
        gate=storage.gate_record('work3',raw,projected,storage.stale_work_bytes()+noise_stale,shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        r['additional_noise_stale_bytes']=noise_stale
        r.update(status=gate['status'],work3=gate,peers=peers,missing_full_projections=missing,original_current=original,original_future=original_future,terminal_original_evidence=terminal_evidence,launcher_manifest_sha256=LAUNCHER_MANIFEST,science_manifest_sha256=SCIENCE_MANIFEST,own_current=own,own_full=own_full,projected_bytes=projected,full_long_admission=full)
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
    ensure_startup('native',index,manifest)
    r=submit(label('native',index),'native',manifest);submit(label('after-native',index),'after-native',manifest,'ended('+r['job_id']+')')

def certify_cpu(manifest):
    proofs=final_job(ROOT/CPU_PARENT,'hpc')+verify_launcher(manifest,LAUNCHER_MANIFEST)+verify_failed_old_cpu();counts={}
    for group,minimum in [('hpc-contract',499),('legacy',238),('native-mapping',33),('chemistry',58),('dbmol-native',0),('dbmol-transport',0),('launcher-adoption',0),('startup',0)]:
        p=ROOT/CPU_PARENT/'payload'/(group+'.xml');log=p.with_suffix('.log');t=ET.parse(p);cases=list(t.iter('testcase'))
        require(len(cases)>minimum and not any(list(t.iter(k)) for k in ['failure','error','skipped']),'Allocated inherited/new CPU cases incomplete')
        require(re.search(r'\b'+str(len(cases))+r'\s+passed\b',log.read_text()),'Actual XML/log counts differ')
        if group=='dbmol-native':require({'test_dbmol_objective','test_dbmol_native_paths'}<={part for c in cases for part in c.get('classname','').split('.')},'Missing native component fixture attribution')
        if group=='dbmol-transport':require({'test_dbmol_pilot_contract','test_dbmol_gradient_contract','test_dbmol_bridge'}<={part for c in cases for part in c.get('classname','').split('.')},'Missing transport/scope fixture attribution')
        if group=='launcher-adoption':require({'fixture_storage_adoption'}<={part for c in cases for part in c.get('classname','').split('.')},'Missing allocated launcher fixture attribution')
        if group=='startup':require(len(cases)==STARTUP_CASES and {'test_premodel_startup'}<={part for c in cases for part in c.get('classname','').split('.')},'Missing exact allocated startup contracts')
        counts[group]=len(cases);proofs += [{'path':str(v),'sha256':sha(v)} for v in [p,log]]
    require([counts[k] for k in ['hpc-contract','legacy','native-mapping','chemistry','dbmol-native','dbmol-transport','launcher-adoption']]==[500,239,34,59,83,148,72],'Exact allocated scientific/launcher suite changed')
    for n in ['composition_reference.json','pocket_definition.json','assessment.json','oracles.json']:
        p=ROOT/CPU_PARENT/'payload'/n;proofs.append({'path':str(p),'sha256':sha(p)})
    exclusive(ROOT/'cpu-certificate.json',{'status':'passed','suite':SUITE,'source_manifest_sha256':manifest,'launcher_manifest_sha256':LAUNCHER_MANIFEST,'launcher_source':str(LAUNCHER),'cases':sum(counts.values()),'groups':counts,'proofs':proofs})
    storage_gate(ROOT/AFTER_CPU_PARENT,'native-dispatch');queue_native(0,manifest)

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
        ensure_startup('segment',index,manifest)
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
            source=BASE/('protenix-extended-release-01' if pro else 'boltz-lr0025-release-03')/'source'
            peer_manifest=sha(source/'source_manifest.json')
            expected=['bsub','-q','hpc','-n','2','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:30',
                '-J',('mosaic-pro50k-' if pro else 'mosaic-bol30k-')+name,
                '-oo',str(root/name/'stdout.%J.log'),'-eo',str(root/name/'stderr.%J.log'),
                '-w',r['dependency'],'bash','-l',str(source/'scripts/hpc'/('extended_stage.lsf' if pro else 'boltz_serial_stage.lsf')),
                str(source),str(root),peer_manifest,'after-native',str(root/name)]
            require(r['args']==expected,'Peer callback source/resources vector differs')
            if pro and peer_failed_projection(root,name,r,peer_manifest):return
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
    args=[PYTHON,str(SOURCE/'scripts/dbmol_driver.py'),'--mode','validate' if phase=='native' else 'run','--config',str(config()),'--output',str(out),'--manifest',manifest,'--cpu-certificate',str(cpu),'--cpu-certificate-sha256',sha(cpu),'--reference',str(ROOT/CPU_PARENT/'payload/composition_reference.json'),'--pocket',str(ROOT/CPU_PARENT/'payload/pocket_definition.json'),'--pocket-sha256',POCKET,'--prepared',PREPARED,'--context-sha256',CONTEXT_FILE,'--worker-python',WORKER,'--offset',str(index*SEGMENT),'--segment-steps',str(SEGMENT)]
    if z:args+=['--resume-snapshot',z['path'],'--resume-sha256',z['sha256']]
    if a:args+=['--ready',str(a),'--ready-sha256',sha(a)]
    exclusive(parent/'payload/driver-invocation.json',{'args':args,'original_saved_results_read_only':True,'launcher_source':str(LAUNCHER),'launcher_manifest_sha256':LAUNCHER_MANIFEST,'science_manifest_sha256':manifest})
    subprocess.run(args,cwd=SOURCE,check=True,timeout=6600 if phase=='native' else 81000)

def main():
    p=argparse.ArgumentParser()
    for k in ['root','source','parent']:p.add_argument('--'+k,type=Path)
    p.add_argument('--launcher',type=Path,required=True);p.add_argument('--launcher-manifest',required=True)
    p.add_argument('--phase',required=True);p.add_argument('--manifest',required=True);a=p.parse_args()
    require(a.launcher.resolve()==LAUNCHER,'Wrong launcher source')
    verify_launcher(a.manifest,a.launcher_manifest);verify_failed_old_cpu()
    require(a.root.resolve()==ROOT and a.source.resolve()==SOURCE and os.environ.get('LSB_JOBID'),'Unallocated/wrong source/root')
    parent=ROOT/CPU_PARENT if a.phase=='cpu-entry' else a.parent
    require(parent is not None and parent.resolve().is_relative_to(ROOT),'Parent escapes output root')
    executing(parent,'cpu' if a.phase=='cpu-entry' else a.phase,a.manifest)
    if a.phase=='cpu-entry':storage_gate(parent,'cpu-entry');return
    if a.phase in ['native','segment']:
        phase,index=decode(parent.name);pin=verify_startup_entry(phase,index,a.manifest)
        require(read(parent/'submission.json').get('startup_smoke')==pin,'Actual GPU entry startup proof differs')
        produce(parent,phase,index,a.manifest);return
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

SCIENCE_MANIFEST='600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'
LAUNCHER=BASE/'boltz-dbmol-storage-release-05/source'
LAUNCHER_MANIFEST=None
CPU_PARENT='cpu-05'
AFTER_CPU_PARENT='after-cpu-05'
ORIGINAL_SOURCE=BASE/'restart-lrreset-release-04/source'
ORIGINAL_SOURCE_MANIFEST='c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'
TERMINAL_PINS={
 'protenix':{
  'job_id':'29792670','summary_bytes':298580732,
  'summary_sha256':'b94d4bae58b7c82b92f4cf0b6b0e9804734afff0f54ea0b162b65d825ec2661d',
  'status':'passed','lsf_status':'Done','stdout_bytes':2201,
  'stdout_sha256':'46f7ff9c942cf3ca9447e4d4d2630a3e604d8a6cd42504b56e8a0fcd6bb3170e',
  'submission_bytes':1808,'submission_sha256':'5dc53846d086b2b6522d1d4aeb42c3461e4875f890b547b8362904793f33e58a'},
 'boltz2':{
  'job_id':'29792671','summary_bytes':301971481,
  'summary_sha256':'06a68008a0dfb8b6a66668447bcda4f01c7b54427d9b68f0d84e5926db278cc1',
  'status':'failed','lsf_status':'Exited','stdout_bytes':2199,
  'stdout_sha256':'b40c2741e3d500f74aafa8819a597d260b7372f5372eb2642aeb6c450341f5c0',
  'submission_bytes':1798,'submission_sha256':'ddc416a8ed853307b639e9011fefadbb484d2721dfd3e3f9d054045c1eee8df0'},
}
FAILED_CPU_PINS={
 'cpu/submission.json':'bbc02e915d3cda0d58cfcfe0806b7c5544df9328214bfabb319669ea27c7103d',
 'after-cpu/submission.json':'f22d46432171ff8cf5b0fc3ceae8d05b65309dd5185777fa103338f8e7f558e9',
 'cpu/stdout.29798549.log':'c1b559a85182bf1fd485e987480c43323a2a7538ea22fd58e51f5df21dc3a33e',
 'after-cpu/stdout.29798550.log':'373813d3d4faabc87af19299827c1983ef62de6b2e43fab08dc3542a7cbcd2cf',
 'cpu/storage-cpu-entry/gate.json':'0cb5b96b49ebefd8f483e882d3b796d040554e9c60b7e1f2ee194fceda2e8afd',
}

def configure_launcher(manifest):
    global LAUNCHER_MANIFEST
    require(isinstance(manifest,str) and re.fullmatch(r'[0-9a-f]{64}',manifest),'Malformed launcher manifest')
    require(LAUNCHER_MANIFEST is None or LAUNCHER_MANIFEST==manifest,'Launcher identity changed')
    LAUNCHER_MANIFEST=manifest
    return manifest

def verify_launcher(science_manifest,launcher_manifest):
    require(science_manifest==SCIENCE_MANIFEST,'Scientific source657 identity changed')
    configure_launcher(launcher_manifest)
    proofs=[]
    for root,expected in [(SOURCE,SCIENCE_MANIFEST),(LAUNCHER,launcher_manifest)]:
        mp=root/'source_manifest.json'
        require(mp.is_file() and not mp.is_symlink() and sha(mp)==expected,'Source manifest bytes changed')
        files=read(mp)['files']
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}
        require(actual==set(files)|{'source_manifest.json'},'Source execution member scope changed')
        for p in [root,*root.rglob('*')]:
            require(not p.is_symlink() and (p.is_dir() or p.is_file()) and not(p.stat().st_mode&0o222),'Source is not sealed read-only regular content')
        for name,digest in files.items():
            p=root/name
            require(p.resolve().is_relative_to(root.resolve()) and p.is_file() and sha(p)==digest,'Source member bytes changed')
            if root==LAUNCHER:proofs.append({'path':str(p),'sha256':digest})
        if root==LAUNCHER:proofs.append({'path':str(mp),'sha256':expected})
    return proofs

def terminal_original_evidence(arm):
    require(arm in ('protenix','boltz2') and arm in TERMINAL_PINS,'Only exact terminal original Pro/Bol may receive zero future charge')
    p=TERMINAL_PINS[arm];parent=OLD/('run-'+arm)
    require(sha(ORIGINAL_SOURCE/'source_manifest.json')==ORIGINAL_SOURCE_MANIFEST,'Original source625 manifest changed')
    paths={'summary':parent/'payload/study/summary.json',
           'stdout':parent/('stdout.'+p['job_id']+'.log'),'submission':parent/'submission.json'}
    records={}
    for name,path in paths.items():
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(parent.resolve()),'Original terminal proof missing/nonregular')
        require(path.stat().st_size==p[name+'_bytes'] and sha(path)==p[name+'_sha256'],'Exact original terminal proof changed')
        records[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':p[name+'_sha256']}
    # Authenticate root status using the real sorted JSON, whose large cases
    # precede status. Nested case statuses must never stand in for root status.
    summary=read(paths['summary'])
    require(isinstance(summary,dict) and summary.get('status')==p['status']
        and summary.get('source_manifest_sha256')==ORIGINAL_SOURCE_MANIFEST,'Pinned original terminal summary status/source differs')
    cases=summary.get('cases')
    require(isinstance(cases,list) and len(cases)==1 and isinstance(cases[0],dict),'Original terminal case scope differs')
    case=cases[0]
    require(case.get('status')=='passed' and isinstance(case.get('updates'),list) and len(case['updates'])==10000
        and isinstance(case.get('states'),list) and len(case['states'])==10001,'Original terminal optimization budget incomplete')
    del case,cases,summary
    text=paths['stdout'].read_text(encoding='utf-8')
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(p['job_id'],p['lsf_status'])],'Original same-job LSF terminal status differs')
    if p['lsf_status']=='Done':
        require('Successfully completed.' in text and 'Exited with exit' not in text,'Original Pro LSF success missing')
    else:require(re.search(r'Exited with exit code\s+1\b',text) is not None,'Original Bol scientific failure not preserved')
    sub=read(paths['submission'])
    require(sub.get('job_id')==p['job_id'] and sub.get('outcome')=='submitted'
        and sub.get('source_manifest_sha256')==ORIGINAL_SOURCE_MANIFEST,'Original actual job submission/source differs')
    return {'status':'terminal_original_authenticated','job_id':p['job_id'],
            'scientific_status':p['status'],'persisted_lsf_status':p['lsf_status'],
            'live_scheduler_status_invented':False,'future_output_bytes':0,
            'counts_scope':'Exact pinned completed10000-update/10001-state summary previously audited; no numerical re-evaluation',
            'original_source_manifest_sha256':ORIGINAL_SOURCE_MANIFEST,'proofs':records}

def original_remaining(current):
    require(set(current)==set(OLD_FULL) and all(type(v) is int and v>=0 for v in current.values()),'Malformed original current-byte inventory')
    require(set(TERMINAL_PINS)=={'protenix','boltz2'},'Terminal accounting pin scope differs')
    evidence={arm:terminal_original_evidence(arm) for arm in TERMINAL_PINS}
    require(type(current['boltz2_chemistry']) is int and current['boltz2_chemistry']<=OLD_FULL['boltz2_chemistry'],'Active original chemistry projection exceeded')
    remaining={'protenix':0,'boltz2':0,'boltz2_chemistry':OLD_FULL['boltz2_chemistry']-current['boltz2_chemistry']}
    return remaining,evidence

def verify_failed_old_cpu():
    pins=FAILED_CPU_PINS
    for name,digest in pins.items():require(sha(HISTORY_ROOT/name)==digest,'Historical failed CPU evidence changed')
    require(read(HISTORY_ROOT/'cpu/storage-cpu-entry/gate.json')['status']=='failed','Historical CPU storage failure relabeled')
    for name,job in [('cpu','29798549'),('after-cpu','29798550')]:
        sub=read(HISTORY_ROOT/name/'submission.json');text=(HISTORY_ROOT/name/('stdout.'+job+'.log')).read_text()
        require(sub['job_id']==job and sub['outcome']=='submitted'
            and re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,'Exited')]
            and 'Exited with exit code 1' in text,'Original failed CPU/callback status differs')
    require(not list((HISTORY_ROOT/'cpu/payload').glob('*.xml')),'Original failed CPU unexpectedly reached pytest')
    return [{'path':str(HISTORY_ROOT/name),'sha256':digest} for name,digest in pins.items()]+verify_failed_cpu02()+verify_cpu03_history()+verify_predecessor_history()




FAILED_CPU02_PINS={'cpu-02/submission.json': 'cb0d36f854b15cda59f0261366c209ec89f52f9ee57f6045cdc0758904c0bc0e', 'after-cpu-02/submission.json': '5b79746f9331ebb3a8eb5da7d16fe9798ae3aa32a3de256ed344efdd63071dd1', 'cpu-02/stdout.29798699.log': '981fe2fa535f613a131d31aaafc8bfea733829282065c3307e17bcaa7ecf019d', 'after-cpu-02/stdout.29798700.log': '66d1335c49c5672c82f33b200093cd30a303901a65645631a714b9d20fdf3fa8', 'cpu-02/storage-cpu-entry/gate.json': '7e5a75a5a0b132df3ee2f8eb164b298be95cba30950f9de4d8c7945a96498b49', 'cpu-02/payload/hpc-contract.xml': '00251c0632f2cec8b97e17140899192bb164ba34f97887e96b43c2d6e5fded71', 'cpu-02/payload/hpc-contract.log': 'b074fd8adb9b31db1f9ad34acde6034f96b2e2b0fb47f9e27971cbe7244c07fc', 'cpu-02/payload/legacy.xml': '7e04fb5fac25026b5550273a402f2b2c4c3a1b8734f5e942f5bd345e42622853', 'cpu-02/payload/legacy.log': 'a92649d51ff960fba2d3812290eb16d448c4ed9a9bf3cb0a612b7edbbacd4e45', 'cpu-02/payload/native-mapping.xml': 'e79c101288fb95d8544fbac5a3b379305e7856de92d70772203b88655c42b5fa', 'cpu-02/payload/native-mapping.log': '35d2cf6a71308872d5b4b2af41d250375ce1559b123380fb9623f10f4dc7da01', 'cpu-02/payload/chemistry.xml': '97922866e2a72d497ed385c57aae7a9344176a9ca606c9b484907952293fb6b2', 'cpu-02/payload/chemistry.log': '9e3d662a45940d2dbf867d98dd388b05c0213b4bc1da2973f556788b95cb60c8', 'cpu-02/payload/dbmol-native.xml': 'ca8246b0d0976a0409e9915e094d49552f29fdb86905a80adf8107a2c5212d9f', 'cpu-02/payload/dbmol-native.log': '25723fb6d2b940eddf4016910ea200bb69b8956d9ad02d251f2b5bf13afeb24c', 'cpu-02/payload/dbmol-transport.xml': 'beda98b250233d30431319fe1fd43f2d7d4b5cbcfbc49e43a81cc0f20dba452a', 'cpu-02/payload/dbmol-transport.log': 'e7e6f66f2563a27c789a6a970c4334e2d3d1f2df0d4e631e5a6f16e159c6563f'}
FAILED_CPU02_NAMES=['test_config_rejects_undeclared_policy_changes[extra]', 'test_config_rejects_undeclared_policy_changes[seed]', 'test_config_rejects_undeclared_policy_changes[steps]', 'test_config_rejects_undeclared_policy_changes[temperature]', 'test_peer_dispatch_waits_only_actual_authenticated_own_ended_callback_ids', 'test_raw_checker_rejects_missing_wrong_or_nonfinite_derivative_paths[nonfinite-E_upper]', 'test_raw_checker_rejects_missing_wrong_or_nonfinite_derivative_paths[nonfinite-X]']

def verify_failed_cpu02():
    for name,digest in FAILED_CPU02_PINS.items():
        require(sha(HISTORY_ROOT/name)==digest,'Historical failed CPU02 evidence changed')
    require(read(HISTORY_ROOT/'cpu-02/storage-cpu-entry/gate.json')['status']=='passed','Historical CPU02 capacity pass changed')
    for name,job in [('cpu-02','29798699'),('after-cpu-02','29798700')]:
        submission=read(HISTORY_ROOT/name/'submission.json');text=(HISTORY_ROOT/name/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,'Exited')]
            and 'Exited with exit code 1' in text,'Original CPU02/callback failure differs')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148)]:
        path=HISTORY_ROOT/'cpu-02/payload'/(group+'.xml');tree=ET.parse(path);cases=list(tree.iter('testcase'))
        require(len(cases)==count and not any(list(tree.iter(k)) for k in ['error','skipped']),'Historical CPU02 case inventory differs')
        names=sorted(case.attrib['name'] for case in cases if case.find('failure') is not None)
        require(names==(FAILED_CPU02_NAMES if group=='dbmol-transport' else []),'Historical seven fixture failures changed or relabeled')
    text=(HISTORY_ROOT/'cpu-02/payload/dbmol-transport.log').read_text()
    require(re.search(r'\b7 failed, 141 passed\b',text) is not None,'Historical CPU02 transport failure log differs')
    require(not (HISTORY_ROOT/'cpu-02/payload/launcher-adoption.xml').exists(),'Historical unexecuted launcher group fabricated')
    return [{'path':str(HISTORY_ROOT/name),'sha256':digest} for name,digest in FAILED_CPU02_PINS.items()]



CPU03_HISTORY_PINS={'cpu-03/submission.json': '07fe54c6c16b7b406f200dccaed2d70afa158e0c3df6e4a529e4469924c502ff', 'after-cpu-03/submission.json': '4f05b6a99aee2576eeb7ffc85480ae1a234bc8e30a2a88df4cdb40f74e9aa096', 'cpu-03/stdout.29798769.log': '4c1b70309891b6e174b1f3e0e9f22e3f4915743183a1697953e112c18b2d8fb8', 'after-cpu-03/stdout.29798770.log': '69fa27b15fbef23bdc0bb3321e3de0b826f073b33f1a67524a1da812642e63f3', 'after-cpu-03/stderr.29798770.log': 'b7cc42d085b4f17b11d7828ee3bfcfae524fe4bfc5d87e62b65d881e375e7f89', 'cpu-03/storage-cpu-entry/gate.json': 'a95322fe217f1e23c3eae04c1056d0b82288d6d9746f975a8b9df3a504a179f7', 'cpu-03/payload/hpc-contract.xml': '8b958a79a0df1f05f110f5ffedac218d1ba8f28604611d2d0a3675e4931dee72', 'cpu-03/payload/hpc-contract.log': '89028b1892f91a03e150db35ccdc1800f3892dbc2ade50f3985a4c5a511f9dc7', 'cpu-03/payload/legacy.xml': 'f0ac8c6c28db1834d7dc7428de2d76cf2ac1c6f5a5345b152e5f8cd057980bbd', 'cpu-03/payload/legacy.log': 'faf9a625ac9677b4545e51485a29cef8e21c85f41ed67691210dd8edbd5f2bc8', 'cpu-03/payload/native-mapping.xml': '566f36aadc008cef16803d73abfc0d420625a7cd69de410486394dcdfa722b73', 'cpu-03/payload/native-mapping.log': '59d1a433b95abc37c5cb9a242f2f5b3ead50aecec94f9dec955a84b1cea7ca43', 'cpu-03/payload/chemistry.xml': '6acd47df49f6ebc104e792d087818b50ba29149ac26245176d44965f33280960', 'cpu-03/payload/chemistry.log': '3b3a413b1bcdfa6b0a82cf6a6fff90e0e83d59f9b3673b0c7f3179482cecff47', 'cpu-03/payload/dbmol-native.xml': '44775d744e0e04aa7e400b6e6fbbaea47f65912caaba188913711569989fd6ae', 'cpu-03/payload/dbmol-native.log': '0e94edde5a85ea0e8db7c3c916a03b4d2ebcbab94ce1045bc203b965d07d0f0a', 'cpu-03/payload/dbmol-transport.xml': '8ba540fcc124327840a2c8b16f68a692aa88cccdc6e126ab9beb6aaf186fd6ec', 'cpu-03/payload/dbmol-transport.log': '5b83938cf12b311863026319ba56b99f64aa57f31f879aa9dff19dcea361581d', 'cpu-03/payload/launcher-adoption.xml': 'c49f6263e31ce5a71e5d9de88c1ac1dd79c46c8b7862ebb6aa3159d5c3daef25', 'cpu-03/payload/launcher-adoption.log': 'de35c3ffacfdacfe9d8112b98f18b28c4e2ca1ffb7be7dd0a5e0db257537fd80'}
CPU03_SOURCE=BASE/'boltz-dbmol-storage-release-03/source'
CPU03_MANIFEST='7b67e6221e4d227b6e501478802ab4ebc8536d74272ca3a7d305bb6f15e7ffee'

def verify_cpu03_history():
    manifest=CPU03_SOURCE/'source_manifest.json'
    require(sha(manifest)==CPU03_MANIFEST,'Historical launcher03 source manifest changed')
    for name,digest in CPU03_HISTORY_PINS.items():
        require(sha(HISTORY_ROOT/name)==digest,'Historical CPU03 or callback proof changed')
    require(read(HISTORY_ROOT/'cpu-03/storage-cpu-entry/gate.json')['status']=='passed','Historical CPU03 capacity pass changed')
    for name,job,status in [('cpu-03','29798769','Done'),('after-cpu-03','29798770','Exited')]:
        submission=read(HISTORY_ROOT/name/'submission.json');text=(HISTORY_ROOT/name/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,status)],'Historical CPU03/callback same-job status differs')
        if status=='Done':require('Successfully completed.' in text and 'Exited with exit' not in text,'Historical CPU03 success missing')
        else:require('Exited with exit code 1' in text,'Historical callback03 failure missing')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148),('launcher-adoption',72)]:
        path=HISTORY_ROOT/'cpu-03/payload'/(group+'.xml');tree=ET.parse(path);cases=list(tree.iter('testcase'))
        require(len(cases)==count and not any(list(tree.iter(k)) for k in ['failure','error','skipped']),'Historical CPU03 passing cases changed or relabeled')
        require(re.search(r'\b'+str(count)+r'\s+passed\b',path.with_suffix('.log').read_text()) is not None,'Historical CPU03 XML/log count differs')
    error=(HISTORY_ROOT/'after-cpu-03/stderr.29798770.log').read_text()
    require('Missing transport/scope fixture attribution' in error,'Historical callback03 cause changed')
    return [{'path':str(HISTORY_ROOT/name),'sha256':digest} for name,digest in CPU03_HISTORY_PINS.items()]+[{'path':str(manifest),'sha256':CPU03_MANIFEST}]



STARTUP_CASES=28
PREDECESSOR_PINS={'cpu-certificate.json': '14199ca0444d614385c5bda4863fd449e063a34540085513c3985080e02b5880', 'cpu-04/submission.json': 'ae4a810a1f2a521972d5471abbc65389328f48b1d3027de39a7f3cfe80630af9', 'after-cpu-04/submission.json': '47ba405eb0e7e79b3742f57c9e287ec52f480ef8935c1eaa16057e11480eb58c', 'cpu-04/stdout.29802621.log': '4641013fd63d93446c7ca4e9926bb3becf25fbf4b8e844188533eef0575742af', 'after-cpu-04/stdout.29802622.log': '9961733b8e68675c2972e27850a8517fd84f8d4db163e63634788b52908b0d2b', 'pilot-native-0000/submission.json': 'fbf85f020b240124d11ed7b648cb9a1804f0c51d4d0a9f299d471a43b1826402', 'pilot-after-native-0000/submission.json': '8a8761cec242f79258b9312f2d2d392946da6553ed166dc284a478900e320fad', 'pilot-native-0000/stdout.29802648.log': '21510672bfd17acca29b368bf63e9943d5726f9dcffb225f2c1d1d7071d0187a', 'pilot-after-native-0000/stdout.29802649.log': '9eb42dffbd473f3da997e19eea215014f0b67040a79606461b76b4f2b5844062', 'pilot-native-0000/stderr.29802648.log': 'c86905513718c0a867b23f858a030fff35dbb2924af7009798420c5797296f5c', 'pilot-after-native-0000/stderr.29802649.log': '0f9c9161e31b604e3289ab0a129727d7c63dea61892d855398209ecee3ce4d5f', 'pilot-native-0000/storage-entry/gate.json': '452eb8f545fc3ec3540cc8651f6b2eb2e04045fdfae3c3bce553204c4d487160', 'cpu-04/payload/hpc-contract.xml': '1667583afcd4321d86727f21ee1601518676dc589bbc3028522953daa8fcbc5a', 'cpu-04/payload/hpc-contract.log': 'ebe6197a34366b2a62b9c50c6a7c616f1063a0012e106f45610341a197baf043', 'cpu-04/payload/legacy.xml': '5471b60dd8e066a1eef901913782b36de8fd3fe7066a04c22bd73404506b9353', 'cpu-04/payload/legacy.log': '9a6558a81c4118265b8cc0f14abdf215a58039ac9660cd502cd799ebf938650b', 'cpu-04/payload/native-mapping.xml': '05c158a9dd06025f3d438981e9e8bb7d0bc5a7f3c6ae9ae94d743c739deb35e5', 'cpu-04/payload/native-mapping.log': 'e4dabcd89e7563f28b9681a4663a6609cb5e83ac2ae9c505f2f1ae25403b6db0', 'cpu-04/payload/chemistry.xml': 'ecddbbd63b9bf6b42d1bf72c539e4bb65bdd5a2642432e581995254bea329c5c', 'cpu-04/payload/chemistry.log': 'a09df74910cf1668383f2f2ff6a14f829608f67646543bd8856536189a5fe3b5', 'cpu-04/payload/dbmol-native.xml': '422641f0cf46b700230c9f259733a048987eac8f82287fc390b36dbd9a2f7c77', 'cpu-04/payload/dbmol-native.log': '6a3f01743f23632ecb8faa0a51621b355ccea3e812e0b87c64966af2236a7dfa', 'cpu-04/payload/dbmol-transport.xml': '8a0f0016186bfd2b0392640213b3e55847b2fe5e17c9c21183e09c0236c39a35', 'cpu-04/payload/dbmol-transport.log': 'c9f874b837871ec6ffd0530bebd1bae67f78984a71053e368c3f76cfce9bc48d', 'cpu-04/payload/launcher-adoption.xml': 'bc0a4bf3db85858492be84f4fc8486d5a6406c47a460bfcfb69c977dacf982c1', 'cpu-04/payload/launcher-adoption.log': '4e98ec59a3bf5b8aed330eb491c14ab08a939d5e37147d9cb2b0b944b1612a13'}

def verify_predecessor_history():
    for name,digest in PREDECESSOR_PINS.items():require(sha(HISTORY_ROOT/name)==digest,'Historical CPU04/native failure proof changed')
    certificate=read(HISTORY_ROOT/'cpu-certificate.json');verify_proofs(certificate)
    require(certificate['cases']==1135 and certificate['source_manifest_sha256']=='24c207da5510fa579576a04ad32bf1afc6e1f23f51dcb5251d82a4ebdfa8f436'
        and certificate['launcher_manifest_sha256']=='df89def61047469e0db8ee6f4b235c75344dd4ad9f35a0833b0839f23859e9fa','Historical CPU04 pass scope differs')
    for parent,job,status in [('cpu-04','29802621','Done'),('after-cpu-04','29802622','Done'),('pilot-native-0000','29802648','Exited'),('pilot-after-native-0000','29802649','Exited')]:
        submission=read(HISTORY_ROOT/parent/'submission.json');text=(HISTORY_ROOT/parent/('stdout.'+job+'.log')).read_text()
        require(submission['job_id']==job and submission['outcome']=='submitted'
            and re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)==[(job,status)],'Historical CPU04/native same-job outcome differs')
        require(('Successfully completed.' in text and 'Exited with exit' not in text) if status=='Done' else 'Exited with exit code 1' in text,'Historical actual LSF terminal footer differs')
    for group,count in [('hpc-contract',500),('legacy',239),('native-mapping',34),('chemistry',59),('dbmol-native',83),('dbmol-transport',148),('launcher-adoption',72)]:
        path=HISTORY_ROOT/'cpu-04/payload'/(group+'.xml');tree=ET.parse(path)
        require(len(list(tree.iter('testcase')))==count and not any(list(tree.iter(k)) for k in ['failure','error','skipped']),'Historical CPU04 passing cases relabeled')
    text=(HISTORY_ROOT/'pilot-native-0000/stderr.29802648.log').read_text()
    require("KeyError: 'git_revision_reference'" in text and 'source_identity' in text,'Historical pre-model metadata failure differs')
    study=HISTORY_ROOT/'pilot-native-0000/payload/study'
    require(not list(study.rglob('*')),'Historical pre-model failure unexpectedly has scientific output')
    return [{'path':str(HISTORY_ROOT/name),'sha256':digest} for name,digest in PREDECESSOR_PINS.items()]

def peer_failed_projection(root,name,submission,peer_manifest):
    path=root/name/('stdout.'+submission['job_id']+'.log')
    if not path.is_file():return False
    text=path.read_text();rows=re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$',text,re.M)
    if rows!=[(submission['job_id'],'Exited')]:return False
    require('Exited with exit code 1' in text,'Unclassified failed peer callback')
    record={'status':'awaiting_failed_peer_native_projection','peer_root':str(root),'callback':name,'job_id':submission['job_id'],
        'source_manifest_sha256':peer_manifest,'full_projection_established':False,'gate_waived':False,
        'proofs':[{'path':str(p),'sha256':sha(p)} for p in [path,root/name/'submission.json']]}
    target=ROOT/('awaiting-failed-peer-'+name+'.json')
    if target.exists():require(read(target)==record,'Preserved failed peer decision changed')
    else:exclusive(target,record)
    return True

def startup_path(phase,index):
    require(phase in ['native','segment'] and type(index) is int and index in range(4),'Undeclared startup scope')
    return ROOT/('startup-'+phase+'-'+format(index,'04d')+'.json')

def verify_startup_entry(phase,index,manifest):
    path=startup_path(phase,index);record=read(path);r=record['result']
    require(record['status']=='startup_only_passed' and record['source_manifest_sha256']==manifest==SCIENCE_MANIFEST and record['launcher_manifest_sha256']==LAUNCHER_MANIFEST
        and record['phase']==phase and record['index']==index,'Startup source/phase/index differs')
    require(r['version']=='mosaic-actual-generated-command-premodel-v1' and r['arm']=='boltz2_dbmol'
        and r['status']=='startup_only_passed' and r['model_loaded'] is False and r['scientific_readiness_passed'] is False
        and r['gradient_passed'] is False and r['optimizer_updates']==0 and r['saved_results_changed'] is False
        and r['phase']==phase and r['index']==index and r['source_manifest_sha256']==manifest
        and r['source']==str(SOURCE) and r['git_revision_reference']=='c74c659344123937608077e3e4078f75f09cd743','Actual startup-only result differs')
    require(r['summary']['status']=='failed' and r['summary']['error']=='PremodelBoundaryReached: Intentional startup-only boundary; no native model loaded'
        and r['summary']['source_manifest_sha256']==manifest and not r['summary'].get('native_gradient_checks') and not r['summary'].get('cases'),'Startup cannot certify native readiness or optimization')
    owner=record['owner'];submission=read(owner['path'])
    require(sha(owner['path'])==owner['sha256'] and submission['job_id']==owner['job_id']==r['job_id']
        and submission['outcome']=='submitted' and submission['phase'] in ['after-cpu','after-native','after-segment','dispatch'],'Actual startup callback owner differs')
    parent=Path(owner['path']).parent
    require(parent.parent==ROOT and submission['args']==command(parent.name,submission['phase'],manifest,submission['dependency']),'Actual startup owner resources/source differs')
    require(record['cpu_sha256']==sha(cpu_certificate(manifest)),'Startup CPU proof changed')
    expected_inputs={'config','cpu-certificate','reference','pocket','prepared_context'}
    if phase=='segment':expected_inputs.add('ready')
    if index:expected_inputs.add('resume-snapshot')
    require(set(r['inputs'])==expected_inputs,'Startup authenticated input scope differs')
    for value in r['inputs'].values():require(Path(value['path']).stat().st_size==value['bytes'] and sha(value['path'])==value['sha256'],'Startup original input bytes changed')
    args=r['args'];require(args[:2]==[PYTHON,str(SOURCE/'scripts/dbmol_driver.py')],'Startup driver/interpreter differs')
    def argument(name):
        require(args.count(name)==1,'Missing/duplicate startup argument');return args[args.index(name)+1]
    for flag,value in [('--manifest',manifest),('--mode','validate' if phase=='native' else 'run'),('--config',str(config())),
        ('--cpu-certificate',str(ROOT/'cpu-certificate.json')),('--cpu-certificate-sha256',record['cpu_sha256']),
        ('--offset',str(index*SEGMENT)),('--segment-steps',str(SEGMENT))]:require(argument(flag)==value,'Startup actual driver command differs')
    snapshot=input_snapshot(index)
    require(record['input_snapshot']==snapshot,'Startup exact predecessor differs')
    if phase=='segment':
        ready,_=readiness(index,manifest);require(record['ready']=={'path':str(ready),'sha256':sha(ready)} and argument('--ready')==str(ready) and argument('--ready-sha256')==sha(ready),'Startup actual readiness differs')
    else:require(record['ready'] is None and '--ready' not in args,'Validation cannot invent readiness')
    from dbmol_protocol import CONTEXT_FILE,POCKET
    expected=[PYTHON,str(SOURCE/'scripts/dbmol_driver.py'),'--mode','validate' if phase=='native' else 'run',
        '--config',str(config()),'--output',argument('--output'),'--manifest',manifest,
        '--cpu-certificate',str(ROOT/'cpu-certificate.json'),'--cpu-certificate-sha256',record['cpu_sha256'],
        '--reference',str(ROOT/CPU_PARENT/'payload/composition_reference.json'),
        '--pocket',str(ROOT/CPU_PARENT/'payload/pocket_definition.json'),'--pocket-sha256',POCKET,
        '--prepared',PREPARED,'--context-sha256',CONTEXT_FILE,'--worker-python',WORKER,
        '--offset',str(index*SEGMENT),'--segment-steps',str(SEGMENT)]
    if snapshot:expected+=['--resume-snapshot',snapshot['path'],'--resume-sha256',snapshot['sha256']]
    if phase=='segment':expected+=['--ready',str(ready),'--ready-sha256',sha(ready)]
    require(args==expected,'Complete actual startup driver arguments differ from production')
    expected_paths={'config':str(config()),'cpu-certificate':str(ROOT/'cpu-certificate.json'),
        'reference':str(ROOT/CPU_PARENT/'payload/composition_reference.json'),
        'pocket':str(ROOT/CPU_PARENT/'payload/pocket_definition.json'),'prepared_context':str(Path(PREPARED)/'context.pt')}
    if phase=='segment':expected_paths['ready']=str(ready)
    if snapshot:expected_paths['resume-snapshot']=snapshot['path']
    require({name:value['path'] for name,value in r['inputs'].items()}==expected_paths,'Exact startup input paths differ')
    return {'path':str(path),'sha256':sha(path)}

def ensure_startup(phase,index,manifest):
    path=startup_path(phase,index)
    if path.exists():return verify_startup_entry(phase,index,manifest)
    marker=path.with_name(path.stem+'-started.json')
    require(not marker.exists(),'Unknown startup outcome; no blind replay')
    owner=[]
    for parent in ROOT.iterdir():
        candidate=parent/'submission.json'
        if parent.is_dir() and candidate.is_file() and read(candidate).get('job_id')==os.environ.get('LSB_JOBID'):owner.append(candidate)
    require(len(owner)==1,'Actual allocated startup owner missing/ambiguous')
    exclusive(marker,{'phase':phase,'index':index,'owner':str(owner[0]),'outcome':'unknown'})
    from premodel_startup import run_startup_smoke
    result=run_startup_smoke(sys.modules[__name__],manifest,Path(os.environ['TMPDIR'])/('dbmol-startup-'+phase+'-'+format(index,'04d')),phase=phase,index=index)
    ready=readiness(index,manifest)[0] if phase=='segment' else None
    record={'status':'startup_only_passed','source_manifest_sha256':manifest,'launcher_manifest_sha256':LAUNCHER_MANIFEST,
        'phase':phase,'index':index,'cpu_sha256':sha(ROOT/'cpu-certificate.json'),'input_snapshot':input_snapshot(index),
        'ready':None if ready is None else {'path':str(ready),'sha256':sha(ready)},
        'owner':{'path':str(owner[0]),'sha256':sha(owner[0]),'job_id':os.environ['LSB_JOBID']},'result':result}
    exclusive(path,record);return verify_startup_entry(phase,index,manifest)

if __name__=='__main__':main()
