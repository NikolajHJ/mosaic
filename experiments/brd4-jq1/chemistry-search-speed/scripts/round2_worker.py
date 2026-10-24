"""Stdlib authenticated CPU/GPU benchmark orchestration; never production."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

BASE=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT=BASE/'chemistry-search-speed-02'
SOURCE=BASE/'chemistry-search-speed-release-02/source'
SCIENCE=BASE/'protenix-extended-release-03/source'
OLD_SOURCE=BASE/'chemistry-search-speed-release-01/source'
OLD_ROOT=BASE/'chemistry-search-speed-01'
ADMISSION=BASE/'nonchem-stride20-startup-release-03/source'
PYTHON='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
LIMIT=1024**3
CPU_COUNTS={'tests.test_speed':79,'tests.test_hybrid':26,'tests.test_gpu_matrix':20,'tests.test_oracle_profile':18}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')


def verify(source,pin):
    source=Path(source);raw=(source/'source_manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==pin
    files=json.loads(raw)['files']
    for rel,digest in files.items():
        p=source/rel
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert p.is_file() and not p.is_symlink() and sha(p)==digest,rel
    return {'manifest_sha256':pin,'files':len(files)}


def external():
    science=verify(SCIENCE,'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562')
    verify(OLD_SOURCE,'a30fd3937ea79935d0c449c600d2ea42f5f26da31906508a31e306f81d1c13b0')
    old=OLD_ROOT/'cpu/payload/assessment.json'
    assert sha(old)=='56d48b762e190b395f5a7c8568b303d090ff60598b5fb1f71f22a982d8230b74'
    a=json.loads(old.read_text());assert a['status']=='passed' and a['cpu_cases']==79
    old_xml=OLD_ROOT/'cpu/payload/tests.xml'
    assert sha(old_xml)==a['cpu_xml_sha256']
    assert xml_cases(old_xml,{'tests.test_speed':79})=={'tests.test_speed':79}
    log=(OLD_ROOT/'cpu/stdout.29809404.log').read_text()
    compact=''.join(log.split())
    assert 'Subject:Job29809404:<mosaic-chem-speed01>incluster<dcc>Done' in compact and 'Successfullycompleted.' in compact
    return science


def storage(parent,phase):
    verify(ADMISSION,'704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab')
    sys.path.insert(0,str(ADMISSION))
    from selected_storage import storage_gate
    original=storage_gate(parent,phase)
    w=original['work3']
    record={'status':'failed','whole_original_gate':original,'additional_allowance_bytes':LIMIT,
            'required_bytes':w['required_bytes']+LIMIT,'available_bytes':w['available_bytes_conservative'],
            'reserve_bytes':4*1024**3,'file_reserve':100000}
    try:
        assert record['available_bytes']>=record['required_bytes'],'Independent CPU+GPU benchmark allowance does not fit'
        record['status']='passed'
    finally:write(Path(parent)/('round2-storage-'+phase+'.json'),record)
    return record


def resources(phase):
    common=['-R','span[hosts=1]']
    if phase=='cpu':return ['-q','hpc','-n','16',*common,'-R','rusage[mem=2GB]','-W','02:00']
    if phase=='after-cpu':return ['-q','hpc','-n','1',*common,'-R','rusage[mem=2GB]','-W','00:20']
    assert phase=='gpu'
    return ['-q','gpul40s','-n','4',*common,'-R','rusage[mem=8GB]','-gpu','num=1:mode=exclusive_process','-W','02:00']


def argv(phase,manifest,dependency=None):
    p=ROOT/phase
    a=['bsub',*resources(phase),'-J','mosaic-chem-speed02-'+phase,'-oo',str(p/'stdout.%J.log'),'-eo',str(p/'stderr.%J.log')]
    if dependency:
        assert phase=='after-cpu' and re.fullmatch(r'ended\(\d+\)',dependency)
        a+=['-w',dependency]
    return a+['bash','-l',str(SOURCE/'scripts/round2_stage.lsf'),str(SOURCE),manifest,phase]


def submit(phase,manifest,dependency=None):
    parent=ROOT/phase;parent.mkdir(exist_ok=False)
    command=argv(phase,manifest,dependency)
    record={'status':'unknown','phase':phase,'argv':command,'dependency':dependency,'manifest_sha256':manifest,
            'submitted_utc':datetime.now(timezone.utc).isoformat()}
    write(parent/'submission-started.json',record)
    try:
        r=subprocess.run(command,capture_output=True,text=True,timeout=60)
        record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        ids=re.findall(r'Job <(\d+)> is submitted',r.stdout)
        if r.returncode==0 and len(ids)==1:record.update(status='submitted',job_id=ids[0])
    except BaseException as error:record['error']=repr(error)
    finally:write(parent/'submission.json',record)
    assert record['status']=='submitted','Unknown submission outcome: never duplicate'
    jobs_path=ROOT/'jobs.json'
    jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    assert phase not in jobs,'Do not replace an actual prior job'
    jobs[phase]=record['job_id']
    temporary=ROOT/('jobs-'+phase+'.partial');write(temporary,jobs);temporary.replace(jobs_path)
    q=subprocess.run(['bjobs','-l',record['job_id']],capture_output=True,text=True,timeout=30)
    (parent/'scheduler-at-submission.txt').write_text(q.stdout+q.stderr)
    return record


def own(phase,manifest):
    p=ROOT/phase/'submission.json'
    for _ in range(20):
        if p.exists():break
        time.sleep(1)
    d=json.loads(p.read_text())
    assert d['status']=='submitted' and d['job_id']==os.environ.get('LSB_JOBID')
    assert d['argv']==argv(phase,manifest,d['dependency']) and d['manifest_sha256']==manifest
    assert os.environ.get('LSB_QUEUE')==('gpul40s' if phase=='gpu' else 'hpc')
    assert int(os.environ['LSB_DJOB_NUMPROC'])=={'cpu':16,'after-cpu':1,'gpu':4}[phase]
    return d


def xml_cases(path,counts):
    cases=ET.parse(path).getroot().findall('.//testcase')
    assert all(not list(v) for v in cases) and len({(v.get('classname'),v.get('name')) for v in cases})==len(cases)
    actual={}
    for v in cases:actual[v.get('classname')]=actual.get(v.get('classname'),0)+1
    assert actual==counts,(actual,counts)
    return actual


def cpu_terminal(manifest):
    sub=json.loads((ROOT/'cpu/submission.json').read_text());job=sub['job_id']
    q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
    write(ROOT/'after-cpu/cpu-final-scheduler.json',{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
    compact=''.join(q.stdout.split())
    assert f'Job<{job}>' in compact and 'Status<DONE>' in compact and 'Queue<hpc>' in compact
    log=ROOT/'cpu'/('stdout.'+job+'.log')
    for _ in range(20):
        if log.exists() and 'Successfully completed.' in log.read_text():break
        time.sleep(1)
    text=''.join(log.read_text().split())
    assert f'Subject:Job{job}:<mosaic-chem-speed02-cpu>incluster<dcc>Done' in text and 'Successfullycompleted.' in text
    a=json.loads((ROOT/'cpu/payload/assessment.json').read_text())
    assert a['status']=='passed' and a['job_id']==job and a['source_manifest_sha256']==manifest
    assert a['cpu_counts']==CPU_COUNTS and xml_cases(ROOT/'cpu/payload/tests.xml',CPU_COUNTS)==CPU_COUNTS
    assert sha(ROOT/'cpu/payload/tests.xml')==a['raw_xml_sha256']
    bench=ROOT/'cpu/payload/benchmark/assessment.json'
    assert sha(bench)==a['benchmark_sha256'] and json.loads(bench.read_text())['status']=='passed_exact_hybrid_saved_input_comparison'
    return {'job_id':job,'assessment_sha256':sha(ROOT/'cpu/payload/assessment.json'),'scheduler':q.stdout,'lsf_sha256':sha(log)}


def run(phase,manifest):
    assert not sys.flags.optimize
    sub=own(phase,manifest);verify(SOURCE,manifest);external()
    parent=ROOT/phase
    if phase=='after-cpu':
        proof=cpu_terminal(manifest)
        gate=storage(parent,'gpu-dispatch')
        gpu=submit('gpu',manifest)
        write(parent/'gpu-submission-complete.json',{'status':'submitted_not_passed','gpu':gpu,'cpu_terminal':proof,'storage':gate})
        print(json.dumps({'gpu_job':gpu['job_id'],'status':'submitted_not_passed'}));return
    payload=parent/'payload';payload.mkdir(exist_ok=False)
    result={'status':'failed','phase':phase,'job_id':sub['job_id'],'source_manifest_sha256':manifest,'original_results_changed':False}
    started=time.monotonic()
    try:
        result['storage']=storage(payload,phase+'-entry')
        plan=json.loads((ROOT/'plan.json').read_text())
        for row in plan['inputs']:
            assert sha(row['path'])==row['sha256'] and sha(row['frozen_reference']['path'])==row['frozen_reference']['sha256']
        env=dict(os.environ,PYTHONPATH=str(SCIENCE/'src'),PYTHONDONTWRITEBYTECODE='1',JAX_ENABLE_X64='0',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',MOSAIC_SPEED_PLAN=str(ROOT/'plan.json'),
                 CUDA_CACHE_PATH=str(payload/'cuda-cache'),CUDA_CACHE_MAXSIZE=str(64*1024**2),
                 XDG_CACHE_HOME=str(payload/'cache'),JAX_ENABLE_COMPILATION_CACHE='false')
        env['JAX_PLATFORMS']='cuda,cpu' if phase=='gpu' else 'cpu'
        if phase=='cpu':env['CUDA_VISIBLE_DEVICES']=''
        modules=['test_speed.py','test_hybrid.py','test_gpu_matrix.py','test_oracle_profile.py'] if phase=='cpu' else ['test_gpu_execution.py']
        command=[PYTHON,'-m','pytest','-q','-o','addopts=','-o','cache_dir='+str(payload/'pytest-cache'),
                 *[str(SOURCE/'tests'/name) for name in modules],'--junitxml='+str(payload/'tests.xml')]
        with (payload/'tests.log').open('xb') as f:
            subprocess.run(command,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
        counts=CPU_COUNTS if phase=='cpu' else {'tests.test_gpu_execution':19}
        result['cpu_counts' if phase=='cpu' else 'gpu_counts']=xml_cases(payload/'tests.xml',counts)
        result['raw_xml_sha256']=sha(payload/'tests.xml')
        if phase=='gpu':
            smi=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=30)
            (payload/'nvidia-smi.txt').write_text(smi.stdout)
        with (payload/'benchmark.log').open('xb') as f:
            subprocess.run([PYTHON,str(SOURCE/'scripts'/('round2_cpu.py' if phase=='cpu' else 'round2_gpu.py')),
                            str(ROOT/'plan.json'),str(payload/'benchmark')],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,
                           check=True,timeout=4500 if phase=='cpu' else 4200)
        assessment=payload/'benchmark/assessment.json'
        expected='passed_exact_hybrid_saved_input_comparison' if phase=='cpu' else 'passed_separate_gpu_matrix_benchmark'
        assert json.loads(assessment.read_text())['status']==expected
        result['benchmark_sha256']=sha(assessment)
        result['original_source_after']=external()
        for row in plan['inputs']:
            assert sha(row['path'])==row['sha256'] and sha(row['frozen_reference']['path'])==row['frozen_reference']['sha256']
        result['status']='passed'
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['output_bytes']=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if result['output_bytes']>=LIMIT:result.update(status='failed',cap_exceeded=True)
        write(payload/'assessment.json',result)
        assert result['output_bytes']<LIMIT,'Joint persistent cap exceeded'


if __name__=='__main__':run(sys.argv[2],sys.argv[1])
