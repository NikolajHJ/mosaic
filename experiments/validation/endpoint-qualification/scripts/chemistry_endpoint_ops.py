"""Authenticated CPU -> two separate diagnostics; no optimizer dispatch branch."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'chemistry-endpoint-recovery-02'
SOURCE=BASE/'chemistry-endpoint-recovery-release-02/source'
PARENT=BASE/'nonchem-endpoint-release-09/source'
PARENT_PIN='f15443b51c5197b95205bf59c90e05192f9ceabb955e6006cccfbbdff257fc9b'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
CAP=2*256*1024**2+64*1024**2
PHASES=('cpu','after-cpu','gpu-protenix','gpu-boltz')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def write(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def binding(path):
    path=Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and all(not p.is_symlink() for p in path.parents),'Ordinary absolute proof required')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify(source,pin,count=None):
    source=Path(source);require(source.is_absolute() and not source.is_symlink()
        and sha(source/'source_manifest.json')==pin,'Manifest identity differs')
    m=read(source/'source_manifest.json');files=m['files']
    require(count is None or len(files)==count,'Source count differs')
    actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file() or p.is_symlink()}
    require(actual==set(files)|{'source_manifest.json'},'Source inventory differs')
    for name,digest in files.items():
        rel=PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name
                and binding(source/name)['sha256']==digest,'Source member changed')
    return m


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def external(manifest):
    require(manifest['diagnostic_only'] is True and manifest['optimizer_updates']==0
            and manifest['original_results_changed'] is False,'Scope changed')
    for row in manifest['parent_sources']:verify(row['path'],row['manifest_sha256'],row['files'])
    plan=read(SOURCE/'plan.json')
    require(plan['version']=='chemistry-endpoint-diagnostic-v1'
            and plan['can_certify_or_restart_original_study'] is False,'Plan scope changed')
    for row in plan['protected_inputs']:require(binding(row['path'])==row,'Original input changed')
    return plan


def storage(parent,phase):
    verify(PARENT,PARENT_PIN,19)
    sys.path.insert(0,str(PARENT/'scripts'))
    original=module(PARENT/'scripts/continuation_ops.py','_chemistry_diagnostic_capacity_parent')
    value=original.storage(parent,'chem-diag-'+phase)
    current=0
    for root in (ROOT,SOURCE.parent,BASE/'chemistry-endpoint-recovery-01',BASE/'chemistry-endpoint-recovery-release-01'):
        for path in root.rglob('*'):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()),'Linked/special own output')
            if path.is_file():current+=path.stat().st_size
    result={'status':'failed','inherited_gate':value,'reserve_bytes':4*1024**3,'file_reserve':100000,
        'additional_complete_group_cap_bytes':CAP,'entire_current_tree_stale_charge_bytes':current,
        'required_bytes':value['required_bytes']+CAP,'available_bytes':value['available_bytes']-current,
        'forecasts_removed':False,'original_gate_changed':False}
    try:
        require(value['status']=='passed' and value['reserve_bytes']==4*1024**3 and value['file_reserve']==100000,
                'Inherited full futures/reserves required')
        require(result['available_bytes']>=result['required_bytes'],'Complete chemistry diagnostic group does not fit')
        require(current<CAP,'New diagnostic group cap exhausted')
        result['status']='passed'
    finally:write(parent/('chemistry-diagnostic-storage-'+phase+'.json'),result)
    return result


def command(phase,pin,dependency=None):
    require(phase in PHASES,'No optimization/continuation phase exists in this capsule')
    gpu=phase.startswith('gpu-');parent=ROOT/phase
    require((dependency is not None)==(phase=='after-cpu'),'Only CPU callback may have a dependency')
    argv=['bsub','-q','gpul40s' if gpu else 'hpc','-n','1' if phase=='after-cpu' else '4',
        '-R','span[hosts=1]','-R','rusage[mem=2GB]' if phase=='after-cpu' else 'rusage[mem=8GB]' if gpu else 'rusage[mem=4GB]',
        '-W','02:00' if gpu else '00:20' if phase=='after-cpu' else '01:00',
        '-J','mosaic-chem-endpoint02-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if gpu:argv+=['-gpu','num=1:mode=exclusive_process']
    if dependency:argv+=['-w',dependency]
    return argv+['bash','-l',str(SOURCE/'scripts/chemistry_endpoint_stage.lsf'),str(SOURCE),pin,phase]


def submit(phase,pin,dependency=None):
    parent=ROOT/phase;argv=command(phase,pin,dependency);parent.mkdir(exist_ok=False)
    row={'status':'unknown','phase':phase,'argv':argv,'dependency':dependency,'source_manifest_sha256':pin,
         'submitted_utc':datetime.now(timezone.utc).isoformat()}
    write(parent/'submission-started.json',row)
    try:
        run=subprocess.run(argv,capture_output=True,text=True,timeout=60)
        row.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
        jobs=re.findall(r'Job <([1-9][0-9]*)> is submitted',run.stdout)
        if run.returncode==0 and len(jobs)==1:row.update(status='submitted',job_id=jobs[0])
    finally:write(parent/'submission.json',row)
    require(row['status']=='submitted','Unknown outcome; inspect, never duplicate')
    with (ROOT/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        path=ROOT/'jobs.json';jobs=read(path) if path.exists() else {}
        require(phase not in jobs,'Existing job identity cannot be replaced')
        jobs[phase]=row['job_id'];temporary=path.with_suffix('.partial');write(temporary,jobs);temporary.replace(path)
    return row


def own(phase,pin):
    parent=ROOT/phase;sub=read(parent/'submission.json')
    dependency=sub.get('dependency')
    require(sub['status']=='submitted' and sub['source_manifest_sha256']==pin
            and sub['job_id']==os.environ.get('LSB_JOBID') and sub['argv']==command(phase,pin,dependency),
            'Actual own job/command differs')
    queue='gpul40s' if phase.startswith('gpu-') else 'hpc'
    require(os.environ.get('LSB_QUEUE')==queue,'Actual allocated queue differs')
    return parent


def xml_cases(path,counts):
    cases=list(ET.parse(path).iter('testcase'))
    require(cases and all(not list(x) for x in cases),'A CPU fixture failed/errored/skipped')
    observed=dict(Counter(x.get('classname') for x in cases))
    require(observed==counts and len({(x.get('classname'),x.get('name')) for x in cases})==len(cases),
            'Exact CPU counts/attribution required')
    return len(cases)


def cpu_pass(pin,manifest):
    parent=ROOT/'cpu';sub=read(parent/'submission.json');job=sub['job_id']
    require(sub['argv']==command('cpu',pin),'Exact CPU command differs')
    for i in range(13):
        q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
        row={'job_id':job,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
        write(ROOT/'after-cpu'/f'cpu-terminal-{i:02d}.json',row)
        compact=''.join(q.stdout.split())
        require(q.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[job]
                and re.findall(r'Queue<([^>]+)>',compact)==['hpc'],'Actual CPU scheduler unavailable')
        state=re.findall(r'Status<([^>]+)>',compact)
        require(state in (['DONE'],['RUN']),'Actual CPU failed or unknown')
        if state==['DONE']:break
        require(i<12,'CPU terminal wait expired');time.sleep(3)
    log=parent/('stdout.'+job+'.log')
    for _ in range(30):
        if log.exists() and 'Successfully completed.' in log.read_text():break
        time.sleep(1)
    text=log.read_text();compact=''.join(text.split())
    require(compact.count('Subject:Job'+job+':')==1 and 'Successfullycompleted.' in compact
            and 'Exitedwithexit' not in compact and 'inqueue<hpc>' in compact,'Same-job CPU LSF success missing')
    payload=parent/'payload';assessment=read(payload/'assessment.json')
    cases=xml_cases(payload/'tests.xml',manifest['cpu_counts'])
    require(assessment['status']=='passed' and assessment['cases']==cases
            and assessment['job_id']==job and assessment['source_manifest_sha256']==pin,'CPU result differs')
    saved=read(payload/'saved/assessment.json')
    require(saved['status']=='diagnostics_completed_original_failure_preserved'
            and saved['protected_inputs_unchanged'] is True,'Saved-forensic preservation failed')
    for arm in ('saved','protenix','boltz'):
        require(read(payload/('startup-'+arm)/'startup.json')['status']=='passed_actual_startup','Actual diagnostic startup failed')
    return {'status':'passed','job_id':job,'source_manifest_sha256':pin,'cases':cases,
        'proofs':[binding(p) for p in (parent/'submission.json',log,payload/'tests.xml',payload/'assessment.json',
            payload/'saved/assessment.json',*(payload/('startup-'+a)/'startup.json' for a in ('saved','protenix','boltz')))]}


def execute(phase,pin):
    manifest=verify(SOURCE,pin);plan=external(manifest);parent=own(phase,pin)
    storage(parent,'entry');payload=parent/'payload';payload.mkdir(exist_ok=False)
    if phase=='after-cpu':
        certificate=cpu_pass(pin,manifest);write(ROOT/'cpu-certificate.json',certificate)
        storage(parent,'dispatch')
        for arm in ('protenix','boltz'):submit('gpu-'+arm,pin)
        write(payload/'assessment.json',{'status':'passed_CPU_and_two_diagnostics_submitted','optimizer_jobs_submitted':False})
        return
    if phase=='cpu':
        base=[PYTHON,str(SOURCE/'scripts/chemistry_endpoint_diagnostic.py'),'--plan',str(SOURCE/'plan.json'),
            '--plan-sha256',sha(SOURCE/'plan.json')]
        commands=[base+['--arm',a,'--output',str(payload/('startup-'+a)),'--startup-only'] for a in ('saved','protenix','boltz')]
        for i,args in enumerate(commands):
            write(payload/f'startup-invocation-{i}.json',{'argv':args})
            subprocess.run(args,check=True,timeout=180)
        args=[PYTHON,'-m','pytest',str(SOURCE/'tests'),'-q','-o','addopts=','-p','no:cacheprovider','--junitxml='+str(payload/'tests.xml')]
        with (payload/'tests.log').open('x') as stream:subprocess.run(args,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
        cases=xml_cases(payload/'tests.xml',manifest['cpu_counts'])
        args=base+['--arm','saved','--output',str(payload/'saved')]
        with (payload/'saved-analysis.log').open('x') as stream:subprocess.run(args,check=True,stdout=stream,stderr=subprocess.STDOUT,timeout=900)
        write(payload/'assessment.json',{'status':'passed','cases':cases,'job_id':os.environ['LSB_JOBID'],
            'source_manifest_sha256':pin,'optimizer_updates':0})
        return
    certificate=read(ROOT/'cpu-certificate.json')
    require(certificate['status']=='passed' and certificate['source_manifest_sha256']==pin,'Allocated CPU certificate required')
    for row in certificate['proofs']:require(binding(row['path'])==row,'CPU proof changed')
    arm=phase.removeprefix('gpu-')
    args=[PYTHON,str(SOURCE/'scripts/chemistry_endpoint_diagnostic.py'),'--plan',str(SOURCE/'plan.json'),
          '--plan-sha256',sha(SOURCE/'plan.json'),'--arm',arm,'--output',str(payload/'native')]
    write(payload/'native-invocation.json',{'argv':args})
    with (payload/'native.log').open('x') as stream:
        run=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=6600)
    require(run.returncode==0,'Native diagnostic failed; raw evidence preserved')
    result=read(payload/'native/assessment.json')
    require(result['status']=='diagnostics_completed_original_failure_preserved'
            and result['can_certify_or_restart_original_study'] is False
            and result['protected_inputs_unchanged'] is True and result['frozen_unchanged'] is True,
            'Native diagnostics or preservation incomplete')
    write(payload/'assessment.json',{'status':'diagnostic_evidence_complete_not_admission','job_id':os.environ['LSB_JOBID'],
        'source_manifest_sha256':pin,'native_assessment':binding(payload/'native/assessment.json'),'optimizer_updates':0})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pin',required=True)
    p.add_argument('--phase',choices=PHASES,required=True);args=p.parse_args(argv)
    execute(args.phase,args.pin)


if __name__=='__main__':main()
