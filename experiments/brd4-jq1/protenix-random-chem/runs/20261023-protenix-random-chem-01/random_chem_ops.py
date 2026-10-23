"""One new random/chemistry Protenix run; independent immutable guarded orchestration."""
import argparse,fcntl,hashlib,json,math,os,re,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'protenix-random-chem-01'
SOURCE=BASE/'protenix-random-chem-release-01/source'
OLD=BASE/'restart-lrreset-04'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
CONFIG=SOURCE/'examples/configs/protenix_random_delayed_chem_10000.json'
MODEL='8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2'
CONTEXT='b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4'
POCKET='9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
OLD_FULL={'protenix':1493521824,'boltz2':1748021824,'boltz2_chemistry':5144021824}
CONTRACT_MODULES=('test_boltz_submission_controller','test_boltz_hpc_tmpdir','test_campaign25_ops','test_campaign25_release','test_campaign25_pocket_ops','test_pocket_definition','test_protenix_lr03_ops','test_protenix_lr03_storage','test_fixed25_ops','test_fixed25_storage','test_fixed25_lrboost_contract','test_fixed25_lrboost_ops','test_fixed25_lrboost_storage','test_fixed25_lrreset_contract','test_fixed25_lrreset_ops','test_fixed25_lrreset_storage','test_protenix_initialization','test_random_chem')
def require(ok,message):
    if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def exclusive(p,value):
    with Path(p).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def atomic(p,value):
    p=Path(p);t=p.with_name(p.name+'.tmp');exclusive(t,value);t.replace(p)
def resources(phase):
    if phase=='cpu':return ['-q','hpc','-n','4','-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','01:00']
    if phase in ('validate','run'):return ['-q','gpul40s','-n','4','-R','span[hosts=1]','-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00' if phase=='validate' else '23:00']
    require(phase in ('after-cpu','after-ready'),'Unknown phase')
    return ['-q','hpc','-n','1','-R','span[hosts=1]','-R','rusage[mem=2GB]','-W','00:30']
def command(name,phase,manifest,dependency=None):
    parent=ROOT/name
    args=['bsub',*resources(phase),'-J','mosaic-random-prochem-'+name,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency:args+=['-w',dependency]
    wrapper=SOURCE/'scripts/hpc'/('random_chem_cpu.lsf' if phase=='cpu' else 'random_chem_stage.lsf')
    args+=['bash','-l',str(wrapper),str(SOURCE),str(ROOT),manifest]
    if phase!='cpu':args += [phase,str(parent)]
    return args
def submit(name,phase,manifest,dependency=None):
    require((name,phase) in [('cpu','cpu'),('after-cpu','after-cpu'),('native','validate'),('after-ready','after-ready'),('run','run')],'Undeclared single-arm job')
    p=ROOT/name;p.mkdir(exist_ok=False);args=command(name,phase,manifest,dependency)
    record={'phase':phase,'args':args,'dependency':dependency,'outcome':'unknown','utc':datetime.now(timezone.utc).isoformat()};exclusive(p/'submission-started.json',record)
    try:
        q=subprocess.run(args,capture_output=True,text=True,timeout=60);record.update(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr)
        ids=re.findall(r'Job <(\d+)>',q.stdout)
        if q.returncode==0 and len(ids)==1:record.update(outcome='submitted',job_id=ids[0])
    except Exception as error:record['error']=repr(error)
    finally:exclusive(p/'submission.json',record)
    require(record['outcome']=='submitted','Unknown/failed submission; inspect markers, never duplicate')
    q=subprocess.run(['bjobs','-l',record['job_id']],capture_output=True,text=True,timeout=30);(p/'scheduler-at-submission.txt').write_text(q.stdout+q.stderr)
    return record
def executing(parent,phase,manifest):
    p=Path(parent)/'submission.json'
    for attempt in range(16):
        if p.exists():break
        require(attempt<15,'No actual executing submission');time.sleep(2)
    sub=read(p)
    require(sub['outcome']=='submitted' and sub['job_id']==os.environ.get('LSB_JOBID') and sub['phase']==phase and sub['args']==command(Path(parent).name,phase,manifest,sub['dependency']),'Executing job/resources/source differs')
    return sub
def capacity(own_projection,current_own,old_current):
    require(type(own_projection) is int and own_projection>0 and type(current_own) is int and 0<=current_own<=own_projection,'Malformed/exceeded new run projection')
    require(set(old_current)==set(OLD_FULL) and all(type(v) is int and 0<=v<=OLD_FULL[a] for a,v in old_current.items()),'Original projections exceeded/malformed')
    return sum(OLD_FULL[a]-old_current[a] for a in OLD_FULL)+own_projection-current_own+2*1024**3+512*1024**2
def storage_gate(parent,phase,projection=None):
    import fixed25_lrreset_storage as storage
    require(Path(parent).resolve().is_relative_to(ROOT) and ROOT.resolve()==ROOT,'Wrong independent work3 receipt root')
    p=Path(parent)/('storage-'+phase);p.mkdir(exist_ok=False)
    record={'status':'failed','phase':phase,'reserve_bytes':4294967296,'file_reserve':100000,'scope':'Original3remaining +existing analysis allowance+new single run remaining; no reservation or cleanup'}
    try:
        old_current={a:storage.tree_bytes(OLD/('run-'+a)) for a in OLD_FULL}
        own=storage.tree_bytes(ROOT/'run')
        full=1024**3 if projection is None else projection
        projected=capacity(full,own,old_current)
        raw=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,check=True,timeout=60).stdout
        gate=storage.gate_record('work3',raw,projected,storage.stale_work_bytes(),shutil.disk_usage(ROOT).free,storage.inode_observation(os.statvfs(ROOT)))
        record.update(status=gate['status'],old_current=old_current,new_full_projection=full,new_current=own,projected_bytes=projected,work3=gate)
    except Exception as error:record['error']=repr(error);raise
    finally:exclusive(p/'gate.json',record)
    require(record['status']=='passed','Full combined byte/inode admission failed')
    return record
def final_job(job,parent,queue):
    for attempt in range(16):
        q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=q.stdout+q.stderr;p=Path(parent)/('scheduler-final-'+str(attempt)+'.txt');p.write_text(raw)
        compact=re.sub(r'\s+','',raw)
        require(q.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Queue<([^>]+)>',compact)==[queue],'Unknown/wrong job/queue final proof')
        status=re.findall(r'Status<([^>]+)>',compact);require(len(status)==1,'Malformed final scheduler')
        if status[0]=='DONE':break
        require(status[0] in ('RUN','PEND') and attempt<15,'Actual producer failed or no finalDONE');time.sleep(2)
    log=Path(parent)/('stdout.'+job+'.log');text=log.read_text()
    require(re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',text,re.M)==[job] and 'Successfully completed.' in text and 'Exited with exit' not in text,'Same-job LSFsuccess missing')
    return [{'path':str(v),'sha256':sha(v)} for v in [p,log,Path(parent)/'submission.json']]
def cpu_certificate(manifest):
    path=ROOT/'cpu-certificate.json';c=read(path)
    require(c['status']=='passed' and c['source_manifest_sha256']==manifest and c['suite']=='protenix-random-delayed-chem-v1','Wrong new CPU certificate')
    for p in c['proofs']:require(sha(p['path'])==p['sha256'],'CPU proof changed')
    return path
def certify_cpu(manifest):
    jobs=read(ROOT/'jobs.json');proofs=final_job(jobs['cpu'],ROOT/'cpu','hpc');counts={};modules=set()
    for group in ['hpc-contract','legacy','native-mapping','chemistry']:
        xml=ROOT/'cpu/payload'/(group+'.xml');log=xml.with_suffix('.log');x=ET.parse(xml);cases=list(x.iter('testcase'))
        require(cases and not any(list(x.iter(k)) for k in ['failure','error','skipped']),'Full allocated fixture group failed/skipped')
        found={part for c in cases for part in c.get('classname','').split('.')};modules.update(found);counts[group]=len(cases)
        require(re.search(r'\b'+str(len(cases))+r'\s+passed\b',log.read_text()),'Actual XML/log row count differs')
        if group=='hpc-contract':require(set(CONTRACT_MODULES)<=found,'Every inherited/new controller fixture must be attributed')
        for p in [xml,log]:proofs.append({'path':str(p),'sha256':sha(p)})
    require(counts['hpc-contract']>396 and counts['legacy']==239 and counts['native-mapping']==34 and counts['chemistry']==59,'Changed full inherited CPU suite')
    for name in ['oracles.json','composition_reference.json','pocket_definition.json','assessment.json']:
        p=ROOT/'cpu/payload'/name;proofs.append({'path':str(p),'sha256':sha(p)})
    require(sha(ROOT/'cpu/payload/pocket_definition.json')==POCKET,'Original pocket changed')
    c={'status':'passed','suite':'protenix-random-delayed-chem-v1','source_manifest_sha256':manifest,'producer_job_id':jobs['cpu'],'cases':sum(counts.values()),'groups':counts,'proofs':proofs};exclusive(ROOT/'cpu-certificate.json',c)
    storage_gate(ROOT/'after-cpu','native-dispatch')
    jobs['native']=submit('native','validate',manifest)['job_id'];atomic(ROOT/'jobs.json',jobs)
    jobs['after_ready']=submit('after-ready','after-ready',manifest,'ended('+jobs['native']+')')['job_id'];atomic(ROOT/'jobs.json',jobs)
def verify_native(manifest):
    from campaign25_contract import verify_readiness,update_parameters
    p=ROOT/'native/payload/study';a=p/'assessment.json';value=verify_readiness(a,sha(a),CONFIG,manifest);summary=read(p/'summary.json')
    require(value['model_digest']==MODEL and value['context_digest']==CONTEXT and value['pocket_definition_sha256']==POCKET and value['diffusion_gradient'] is False,'Frozen native context/model/pocket changed')
    require(summary['status']=='passed' and summary['native_hard_parity']['status']=='passed' and summary['native_gradient_checks']['status']=='passed' and summary['chemistry_assay']['status']=='passed','New raw/parity/nonzero checks incomplete')
    require(summary['frozen_identity_after']=={'model_digest':MODEL,'context_digest':CONTEXT},'Frozen before/after differs')
    for label in ['low','high']:
        require(read(p/(label+'_coupled_checks/report.json'))['status']=='passed','Required endpoint coupled derivative gate failed')
    full=value['storage_projection'];require(type(full['maximum_pilot_update_bytes']) is int and full['maximum_pilot_update_bytes']>0 and full['projected_bytes']==2*full['maximum_pilot_update_bytes']*10000+1024**3,'Changed full pilot output projection')
    require(value['runtime_gate']['status']=='passed' and value['runtime_gate']['conservative_projected_seconds']<82800,'Runtime guard failed')
    return a,value
def after_ready(manifest):
    jobs=read(ROOT/'jobs.json');proofs=final_job(jobs['native'],ROOT/'native','gpul40s');p,a=verify_native(manifest)
    cpu_certificate(manifest)
    gate=storage_gate(ROOT/'after-ready','long-dispatch',a['storage_projection']['projected_bytes'])
    exclusive(ROOT/'long-dispatch-started.json',{'source_manifest':manifest,'native_job':jobs['native'],'ready_sha256':sha(p),'cpu_sha256':sha(ROOT/'cpu-certificate.json'),'proofs':proofs,'storage':gate})
    jobs['run']=submit('run','run',manifest)['job_id'];atomic(ROOT/'jobs.json',jobs)
def produce(parent,phase,manifest):
    cpu=cpu_certificate(manifest);projection=None;ready=None
    if phase=='run':ready,a=verify_native(manifest);projection=a['storage_projection']['projected_bytes']
    storage_gate(parent,'entry',projection)
    output=Path(parent)/'payload/study';output.parent.mkdir(exist_ok=False)
    args=[PYTHON,str(SOURCE/'examples/brd4_25_atoms.py'),'--mode',phase,'--config',str(CONFIG),'--output',str(output),'--manifest',manifest,'--cpu-certificate',str(cpu),'--cpu-certificate-sha256',sha(cpu),'--reference',str(ROOT/'cpu/payload/composition_reference.json'),'--pocket',str(ROOT/'cpu/payload/pocket_definition.json'),'--pocket-sha256',POCKET]
    if ready:args += ['--ready',str(ready),'--ready-sha256',sha(ready)]
    exclusive(Path(parent)/'payload/driver-invocation.json',{'args':args,'original_saved_runs_read_only':True})
    subprocess.run(args,cwd=SOURCE,check=True)
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['cpu-entry','after-cpu','validate','after-ready','run'],required=True);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--manifest',required=True);parser.add_argument('--parent',type=Path);args=parser.parse_args()
    require(args.root.resolve()==ROOT and args.source.resolve()==SOURCE and bool(os.environ.get('LSB_JOBID')),'Wrong source/root or unallocated execution')
    parent=ROOT/'cpu' if args.phase=='cpu-entry' else args.parent
    require(parent.resolve().is_relative_to(ROOT),'Independent parent escaped')
    executing(parent,'cpu' if args.phase=='cpu-entry' else args.phase,args.manifest)
    if args.phase=='cpu-entry':storage_gate(parent,'cpu-entry')
    elif args.phase in ['after-cpu','after-ready']:
        with (ROOT/'dispatch.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            if args.phase=='after-cpu':certify_cpu(args.manifest)
            else:after_ready(args.manifest)
    else:produce(parent,args.phase,args.manifest)
if __name__=='__main__':main()
