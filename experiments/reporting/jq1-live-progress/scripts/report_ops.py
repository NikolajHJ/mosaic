"""Immutable allocated-CPU report orchestration; no optimizer or native calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

VERSION = 'jq1-live-report-source-v1'
BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE/'nonchem-live-report-08'
SOURCE = BASE/'nonchem-live-report-release-08/source'
ADMISSION = BASE/'nonchem-stride20-startup-release-03/source'
ADMISSION_SHA = '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'
PYTHON = '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
GIB = 1024**3
CAP = GIB


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def binding(path):
    path=Path(path)
    require(path.is_absolute() and not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode),'Nonregular proof')
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def check_binding(row):
    actual=binding(row['path'])
    require(actual==row,'Proof bytes changed: '+str(row['path']))
    return actual


def verify(source, pin):
    source=Path(source)
    require(source.is_dir() and not source.is_symlink(),'Source root missing or linked')
    require(sha(source/'source_manifest.json')==pin,'Manifest identity changed')
    manifest=read(source/'source_manifest.json')
    names=manifest['files']
    for name,digest in names.items():
        p=PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,'Unsafe source member')
        path=source.joinpath(*p.parts)
        require(not path.is_symlink() and path.is_file() and sha(path)==digest,'Source member changed: '+name)
        require(all(not parent.is_symlink() for parent in path.parents if parent!=source.parent),'Linked source parent')
    return manifest


def external(manifest):
    for row in manifest['parent_sources']:
        actual=verify(row['path'],row['manifest_sha256'])
        require(len(actual['files'])==row['files'],'Parent source file count differs')
    for row in manifest['proofs']: check_binding(row)
    plan=read(SOURCE/'plan.json')
    for arm in plan['arms']:
        for segment in arm['segments']:
            for key in ('states','updates','summary','config','retained_index'):
                if key in segment: check_binding(segment[key])
            summary=read(segment['summary']['path'])
            require(summary['status']==segment['scientific_status'],'Original scientific verdict differs')
            stdout=Path(segment['stdout']['path']).read_text()
            compact=''.join(stdout.split())
            require(compact.count('Subject:Job')==1 and 'Subject:Job'+segment['job_id']+':' in compact,
                    'Original same-job terminal LSF identity missing')
            if summary['status']=='failed':
                require('Exitedwithexitcode1.' in compact,'Failed original lacks same-job exit-one LSF')
            else:
                require('Successfullycompleted.' in compact,'Passed original lacks LSF success')
    return {'parent_sources':len(manifest['parent_sources']),'original_proofs':len(manifest['proofs']),
            'original_results_changed':False}


def own_bytes():
    total=0
    for root in (ROOT,SOURCE.parent,BASE/'nonchem-live-report-05',BASE/'nonchem-live-report-release-05',
                 BASE/'nonchem-live-report-07',BASE/'nonchem-live-report-release-07'):
        if not root.exists(): continue
        for directory,dirs,files in os.walk(root,followlinks=False):
            for name in dirs+files:
                path=Path(directory)/name
                info=path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    require(path.resolve(strict=True).is_relative_to(root.resolve()),'External owned link')
                    total+=info.st_size
                elif stat.S_ISREG(info.st_mode): total+=info.st_size
                else: require(stat.S_ISDIR(info.st_mode),'Owned special file')
    return total


def report_budget(prior, charged):
    require(prior.get('status')=='passed' and prior.get('reserve_bytes')==4*GIB and prior.get('file_reserve')==100000
            and prior.get('forecasts_removed') is False and prior.get('original_gate_changed') is False,
            'Complete current futures/reserves must pass')
    require(type(charged) is int and 0<=charged<CAP
            and all(type(prior.get(k)) is int for k in ('required_bytes','available_bytes')),
            'Whole report bytes must be finite integers within its independent cap')
    return {'status':'failed','inherited_current_gate':prior,'additional_report_allowance_bytes':CAP,
            'required_bytes':prior['required_bytes']+CAP,'available_bytes':prior['available_bytes']-charged,
            'reserve_bytes':4*GIB,'file_reserve':100000,'new_owned_bytes':charged,'forecasts_removed':False}


def storage(parent, phase):
    """Retain the complete current continuation guard plus a separate report cap."""
    current = BASE / 'nonchem-precision-release-15/source'
    pin = '5448503965365f4db726d558df4ac91a41a7f711366b6ae47ae936993fd0c8bb'
    require(len(verify(current, pin)['files']) == 35, 'Current complete admission source changed')
    spec = importlib.util.spec_from_file_location('_current_complete_report_admission', current/'scripts/precision_continuation_ops.py')
    op = importlib.util.module_from_spec(spec);sys.modules[spec.name]=op;spec.loader.exec_module(op)
    inherited = Path(parent)/('current-full-'+phase);inherited.mkdir(exist_ok=False)
    prior = op.storage(inherited, 'entry')
    upload = BASE/'nonchem-live-report-release-08.tar.gz'
    old_upload=BASE/'nonchem-live-report-release-05.tar.gz'
    predecessor_upload=BASE/'nonchem-live-report-release-07.tar.gz'
    charged=own_bytes()+sum(p.stat().st_size for p in (upload,old_upload,predecessor_upload) if p.exists())
    gate=report_budget(prior,charged)
    try:
        require(charged<CAP,'Independent report cap exhausted')
        require(gate['available_bytes']>=gate['required_bytes'],'Whole campaign plus independent report does not fit')
        gate['status']='passed'
    finally:write(Path(parent)/('report-storage-'+phase+'.json'),gate)
    return gate


def argv(phase,pin,dependency=None):
    require(phase in ('cpu','after-cpu','render'),'Undeclared report phase')
    require(re.fullmatch('[a-f0-9]{64}',pin or '') is not None,'Malformed source pin')
    parent=ROOT/phase
    args=['bsub','-q','hpc','-n','1' if phase=='after-cpu' else '4',
          '-R','span[hosts=1]','-R','rusage[mem=4GB]','-W','00:20' if phase=='after-cpu' else '01:00',
          '-J','mosaic-live-report08-'+phase,'-oo',str(parent/'stdout.%J.log'),'-eo',str(parent/'stderr.%J.log')]
    if dependency is not None:
        require(phase=='after-cpu' and re.fullmatch(r'ended\([1-9]\d*\)',dependency),'Wrong own-ended dependency')
        args+=['-w',dependency]
    else: require(phase!='after-cpu','Callback missing own-ended dependency')
    return args+['bash','-l',str(SOURCE/'scripts/report_stage.lsf'),str(SOURCE),pin,phase]


def submit(phase,pin,dependency=None):
    parent=ROOT/phase;parent.mkdir(exist_ok=False)
    command=argv(phase,pin,dependency)
    row={'status':'unknown','phase':phase,'manifest_sha256':pin,'argv':command,'dependency':dependency,
         'submitted_utc':datetime.now(timezone.utc).isoformat()}
    if phase=='render':row['cpu_certificate_sha256']=sha(ROOT/'cpu-certificate.json')
    write(parent/'submission-started.json',row)
    try:
        run=subprocess.run(command,capture_output=True,text=True,timeout=60)
        row.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
        jobs=re.findall(r'Job <([1-9]\d*)> is submitted',run.stdout)
        if run.returncode==0 and len(jobs)==1:row.update(status='submitted',job_id=jobs[0])
    except BaseException as error:row['error']=repr(error)
    finally:write(parent/'submission.json',row)
    require(row['status']=='submitted','Unknown submission outcome: do not retry')
    import fcntl
    with (ROOT/'jobs.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else {}
        require(phase not in jobs,'Duplicate report phase')
        jobs[phase]=row['job_id'];tmp=ROOT/('jobs-'+phase+'.partial');write(tmp,jobs);tmp.replace(ROOT/'jobs.json')
    return row


def xml_counts(path):
    root=ET.parse(path).getroot();cases=root.findall('.//testcase')
    require(cases and all(not list(case) for case in cases),'Missing/failed/error/skipped CPU fixture')
    require(len({(c.get('classname'),c.get('name')) for c in cases})==len(cases),'Duplicate fixture identity')
    return dict(Counter(c.get('classname') for c in cases))


def success_lsf(parent,job,phase):
    path=parent/('stdout.'+job+'.log')
    for _ in range(30):
        if path.exists() and 'Successfully completed.' in path.read_text():break
        time.sleep(1)
    raw=path.read_text();text=''.join(raw.split())
    require(text.count('Subject:Job')==1 and 'Subject:Job'+job+':' in text
            and '<mosaic-live-report08-'+phase+'>incluster<dcc>Done' in text
            and 'inqueue<hpc>' in text and text.count('Successfullycompleted.')==1,
            'Actual same-job/name/hpc LSF success missing')
    return binding(path)


def terminal(parent,job):
    queries=[]
    for index in range(13):
        run=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
        row={'job_id':job,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        path=parent/('terminal-query-'+str(index).zfill(3)+'.json');write(path,row);queries.append(binding(path))
        compact=''.join(run.stdout.split())
        require(run.returncode==0 and re.findall(r'Job<(\d+)>',compact)==[job],'Actual scheduler query unavailable')
        status=re.findall(r'Status<([^>]+)>',compact)
        require(len(status)==1 and status[0] in ('DONE','RUN','PEND'),'Actual report predecessor failed')
        if status[0]=='DONE':return queries
        time.sleep(5)
    raise TimeoutError('Actual terminal DONE not yet established')


def verify_certificate(pin,manifest):
    cert=read(ROOT/'cpu-certificate.json')
    require(cert['status']=='passed' and cert['manifest_sha256']==pin,'Missing CPU certificate')
    for row in cert['proofs']:check_binding(row)
    require(xml_counts(ROOT/'cpu/tests.xml')==manifest['cpu_counts'],'Raw CPU attribution changed')
    return cert


def execute(pin,phase):
    manifest=verify(SOURCE,pin);require(manifest['version']==VERSION,'Wrong report release')
    parent=ROOT/phase
    for _ in range(20):
        if (parent/'submission.json').exists():break
        time.sleep(1)
    sub=read(parent/'submission.json')
    require(sub['status']=='submitted' and sub['job_id']==os.environ.get('LSB_JOBID')
            and sub['argv']==argv(phase,pin,sub['dependency']),'Actual allocated command/source mismatch')
    require(os.environ.get('LSB_QUEUE')=='hpc' and os.environ.get('LSB_DJOB_NUMPROC')==('1' if phase=='after-cpu' else '4'),
            'Wrong actual allocated report resources')
    external(manifest);storage(parent,phase)
    if phase=='cpu':
        tests=sorted(str(path) for path in (SOURCE/'tests').glob('test_*.py'))
        command=[PYTHON,'-m','pytest',*tests,'--rootdir',str(SOURCE),'--import-mode=importlib','-o','addopts=',
                 '-o','cache_dir='+str(parent/'cache'),'--basetemp',str(Path(os.environ['TMPDIR'])/'report-tests'),
                 '--junitxml',str(parent/'tests.xml'),'-q']
        write(parent/'command.json',{'argv':command})
        env=os.environ.copy();env['PYTHONPATH']=str(SOURCE/'scripts')
        with (parent/'tests.log').open('x') as stream:
            subprocess.run(command,env=env,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=2400)
        counts=xml_counts(parent/'tests.xml');require(counts==manifest['cpu_counts'],'Actual test attribution/count differs')
        write(parent/'assessment.json',{'status':'passed','job_id':sub['job_id'],'manifest_sha256':pin,
              'cpu_counts':counts,'xml_sha256':sha(parent/'tests.xml'),'original_results_changed':False})
    elif phase=='after-cpu':
        cpu=read(ROOT/'cpu/submission.json');job=cpu['job_id']
        require(sub['dependency']=='ended('+job+')','Wrong CPU predecessor')
        queries=terminal(parent,job);lsf=success_lsf(ROOT/'cpu',job,'cpu')
        assessment=read(ROOT/'cpu/assessment.json')
        require(assessment['status']=='passed' and assessment['job_id']==job
                and assessment['manifest_sha256']==pin and xml_counts(ROOT/'cpu/tests.xml')==manifest['cpu_counts'],
                'CPU actual raw proof failed')
        write(ROOT/'cpu-certificate.json',{'status':'passed','manifest_sha256':pin,'cpu_job_id':job,
              'cpu_counts':manifest['cpu_counts'],'proofs':queries+[lsf]+[binding(ROOT/'cpu'/name) for name in ('submission.json','assessment.json','tests.xml','tests.log','command.json')]})
        submit('render',pin)
        write(parent/'assessment.json',{'status':'passed','submitted_render':read(ROOT/'jobs.json')['render']})
    else:
        cert=verify_certificate(pin,manifest)
        require(sub['cpu_certificate_sha256']==sha(ROOT/'cpu-certificate.json'),'Render CPU pin changed')
        from report_data import render
        output=parent/'viewer';render(read(SOURCE/'plan.json'),output)
        shutil.copyfile(SOURCE/'templates/report.html',output/'report.html')
        shutil.copyfile(SOURCE/'plan.json',output/'input-plan.json')
        shutil.copyfile(ROOT/'cpu-certificate.json',output/'cpu-certificate.json')
        receipt={'status':'passed_saved_only_presentation','job_id':sub['job_id'],'manifest_sha256':pin,
                 'cpu_certificate_sha256':sha(ROOT/'cpu-certificate.json'),'plan_sha256':sha(SOURCE/'plan.json'),
                 'scientific_statuses':{arm['id']:arm['scientific_status'] for arm in read(output/'manifest.json')['arms']},
                 'files':{path.relative_to(output).as_posix():{'bytes':path.stat().st_size,'sha256':sha(path)} for path in sorted(output.rglob('*')) if path.is_file()},
                 'original_results_changed':False,'scientific_success_claim':False}
        write(output/'report-receipt.json',receipt)
        archive=ROOT/'edition-0004.tar.gz'
        with tarfile.open(archive,'x:gz') as target:
            for path in sorted(output.rglob('*')):
                if path.is_file():target.add(path,arcname=path.relative_to(output).as_posix(),recursive=False)
        require(own_bytes()<CAP,'New report cap exceeded')
        write(ROOT/'edition-0004-archive.json',{'status':'passed_saved_only_presentation','job_id':sub['job_id'],
              'manifest_sha256':pin,'archive':binding(archive),'receipt_sha256':sha(output/'report-receipt.json'),
              'cpu_certificate_sha256':sha(ROOT/'cpu-certificate.json'),'scientific_success_claim':False})
        write(parent/'assessment.json',{'status':'passed_saved_only_presentation','job_id':sub['job_id'],
              'manifest_sha256':pin,'archive_receipt_sha256':sha(ROOT/'edition-0004-archive.json')})


if __name__=='__main__':
    require(len(sys.argv)==3,'Require exact source pin and phase')
    execute(*sys.argv[1:])
