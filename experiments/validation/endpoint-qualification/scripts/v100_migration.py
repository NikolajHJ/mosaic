"""Bounded device comparison, followed by conditional unstarted-job migration.

Smoke states are disposable. Production restarts from the original saved input.
No running job is migrated, and an inconclusive device result keeps L40S.
"""
import argparse
from copy import deepcopy
import csv
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

from exploratory_continuation import bound,capacity,read,require,write
from launch_exploratory_continuations import exclusive,submit
from gpu_policy import profile


def command(argv,timeout=60):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    return {'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}


def observed_job(job):
    raw=command(['bjobs','-json','-o','jobid user stat queue job_name command',job])
    require(raw['returncode']==0,'Scheduler observation failed')
    rows=json.loads(raw['stdout'])['RECORDS'];require(len(rows)==1,'Ambiguous scheduler observation')
    return rows[0]


def eligible(row,smoke):
    """Only device fit and a measured whole-remainder budget; no scientific claim."""
    if smoke.get('status')!='passed_device_smoke':return {'eligible':False,'reason':'device smoke failed'}
    if smoke['memory_headroom_MiB']<1024:return {'eligible':False,'reason':'less than 1 GiB measured GPU headroom'}
    summary=smoke['summary'];times=summary['update_seconds']
    require(len(times)==3 and all(math.isfinite(v) and v>0 for v in times),'Three actual positive update timings required')
    require(summary['applied_updates']==3 and summary['frozen_before']==summary['frozen_after'],'Incomplete smoke/preservation')
    # The first update includes compilation. Predeclared warm estimate uses the
    # slower of updates two/three; this is timing, not gradient-pair selection.
    warm=max(times[1:]);setup=max(0,summary['elapsed_seconds']-sum(times))+max(0,times[0]-warm)
    minutes=max(15,math.ceil((1.25*(setup+warm*row['steps'])+300)/60))
    return {'eligible':minutes<=24*60,'reason':'whole remainder fits V100 queue limit' if minutes<=1440 else 'whole remainder exceeds V100 24-hour limit; retain L40S',
            'wall_minutes':minutes,'warm_update_seconds':warm,'measured_setup_seconds':setup,
            'scope':'three-update timing estimate; future chemistry/runtime may vary'}


def compare(plan,source):
    root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    gate=capacity(plan,'device-comparison');write(root/'capacity-entry.json',gate)
    require(os.environ.get('LSB_QUEUE')=='gpuv100','Allocated V100 comparison only')
    routing=__import__('exploratory_continuation').module(Path(plan['routing_source'])/'scripts/cuda_ownership.py','_v100_routing')
    os.environ['CUDA_VISIBLE_DEVICES']=routing.resolve_visibility(os.environ)['selected_cuda_visible_devices']
    results={};group_started=time.monotonic()
    for arm in plan['arms']:
        require(time.monotonic()-group_started<3000,'Complete device-comparison time budget exceeded')
        folder=root/arm;folder.mkdir(exist_ok=False)
        argv=['bash','-l',str(source/'exploratory_continuation_stage.sh'),str(source),arm,'smoke',str(source/'plan.json')]
        stop=threading.Event();samples=[];sampling_errors=[]
        def sample():
            while not stop.is_set():
                try:
                    raw=command(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],15)
                    require(raw['returncode']==0,'GPU memory observation failed')
                    rows=list(csv.reader(io.StringIO(raw['stdout'])));require(len(rows)==1,'Ambiguous GPU memory observation')
                    samples.append([int(v.strip()) for v in rows[0]])
                except BaseException as error:sampling_errors.append(repr(error))
                stop.wait(.75)
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        process=None
        try:
            with (folder/'device-stdout.log').open('xb') as stdout,(folder/'device-stderr.log').open('xb') as stderr:
                process=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
                try:code=process.wait(timeout=540)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=15)
                    raise TimeoutError('Bounded device child exceeded 540 seconds')
            summary=read(folder/'payload/study/summary.json')
            require(code==0 and summary['status']=='completed_device_smoke','Actual device child failed')
            require(summary['frozen_before']==summary['frozen_after'],'Frozen device input changed')
            require(not sampling_errors and samples,'Complete GPU memory observations required')
            results[arm]={'status':'passed_device_smoke','summary':summary,'memory_samples_MiB':samples,
                'memory_headroom_MiB':min(total-used for used,total in samples),'scope':'finite backward/three actual updates; not derivative qualification'}
        except BaseException as error:
            results[arm]={'status':'failed_device_smoke','error':repr(error),'memory_samples_MiB':samples,'sampling_errors':sampling_errors}
        finally:stop.set();thread.join(timeout=20)
        write(folder/'device-result.json',results[arm])
    write(root/'comparison.json',{'status':'device_comparison_completed','job_id':os.environ['LSB_JOBID'],
        'results':results,'saved_results_changed':False,'smoke_states_adopted':False})


def unstarted(plan,arm,job):
    old=plan['migration'][arm];r=observed_job(job)
    require(r['JOBID']==job and r['USER']=='s234818' and r['JOB_NAME']==old['name'] and r['QUEUE']=='gpul40s', 'Original job identity differs')
    require(shlex.split(r['COMMAND'])==old['command'],'Original job command differs')
    history=command(['bhist','-l',job]);require(history['returncode']==0,'No trustworthy job history')
    text=history['stdout']
    require(not any(token in text for token in ('Dispatched to','Started on','Running with','Done successfully','Exited with')),'Original job has executed')
    require(not (Path(old['parent'])/'payload').exists(),'Original output exists; keep saved progress')
    return r,history


def migrate(plan,source):
    root=Path(plan['root']);comparison=read(root/'comparison.json');submission=read(root/'gpu/submission.json')
    require(comparison['job_id']==submission['job_id'],'Comparison job differs')
    actual=observed_job(submission['job_id']);require(actual['STAT']=='DONE' and actual['QUEUE']=='gpuv100' and actual['USER']=='s234818','Comparison must actually finish')
    lsf=(root/'gpu'/('stdout.'+submission['job_id']+'.log')).read_text()
    require('Successfully completed.' in lsf and 'Job <'+submission['job_id']+'>' in lsf,'Same-job successful LSF required')
    decisions={};production=Path(plan['production_root']);production.mkdir(exist_ok=True)
    for arm,row in plan['arms'].items():
        decision=eligible(row,comparison['results'][arm]);decisions[arm]=decision
        if not decision['eligible']:continue
        parent=root/('migration-'+arm);parent.mkdir(exist_ok=False);held=False;retired=False
        old=plan['migration'][arm];job=old['job_id']
        try:
            current=observed_job(job)
            if current['STAT']!='PEND':decision.update(action='left_original_unchanged',original_state=current['STAT']);continue
            initial,history=unstarted(plan,arm,job);write(parent/'unstarted-before.json',{'scheduler':initial,'history':history})
            child=deepcopy(plan);child.update(root=str(production),purpose='exploratory_continuation',job_name_prefix='mosaic-v10001')
            for name,newrow in child['arms'].items():
                newrow.update(gpu_profile='v100-32gb-v1',queue='gpuv100',gpu_request=profile()['resource'])
                newrow.update(plan['production_caps'][name])
            newrow=child['arms'][arm];minutes=decision['wall_minutes']
            newrow.update(walltime=f'{minutes//60:02}:{minutes%60:02}',internal_seconds=minutes*60-180)
            newrow['predecessor_job_ids']=list(set(newrow.get('predecessor_job_ids',[])+[job]))
            config=production/(arm+'-plan.json');exclusive(config,child);config.chmod(0o444)
            for key in ('config','snapshot','reference','pocket'):bound(row[key])
            write(parent/'capacity-before-retirement.json',capacity(child,'migration-before-retirement'))
            startup=command(['bash','-l',str(source/'exploratory_continuation_stage.sh'),str(source),arm,'startup',str(config)],180)
            write(parent/'startup.json',startup);require(startup['returncode']==0,'Replacement actual metadata startup failed')
            exclusive(parent/'retirement-intent.json',{'job_id':job,'only_if_unstarted':True})
            stop=command(['bstop',job]);write(parent/'stop-response.json',stop);require(stop['returncode']==0,'Could not hold original pending job')
            held=True
            for _ in range(12):
                state=observed_job(job)['STAT']
                if state in ('PSUSP','USUSP','DONE','EXIT'):break
                time.sleep(2)
            if state!='PSUSP':decision.update(action='left_original_unchanged',original_state=state);continue
            current,history=unstarted(plan,arm,job);write(parent/'held-unstarted.json',{'scheduler':current,'history':history})
            kill=command(['bkill',job]);write(parent/'retirement-response.json',kill);require(kill['returncode']==0,'Retirement needs reconciliation')
            for _ in range(12):
                if observed_job(job)['STAT']=='EXIT':break
                time.sleep(2)
            require(observed_job(job)['STAT']=='EXIT','Original retirement not established');retired=True
            write(parent/'retired-unstarted.json',{'job_id':job,'applied_updates':0,'replacement_plan':str(config)})
            submit(child,arm,source,config)
            newjob=read(production/arm/'submission.json')['job_id'];decision.update(action='migrated',old_job=job,new_job=newjob)
        except BaseException as error:decision.update(action='needs_reconciliation',error=repr(error),original_retired=retired)
        finally:
            if held and not retired:
                restore=command(['bresume',job]);write(parent/'restore-original-release.json',restore)
                if restore['returncode']!=0:decision['restore_warning']='Original release needs reconciliation'
            write(parent/'decision.json',decision)
    write(root/'migration-summary.json',{'status':'conditional_migration_finished','decisions':decisions,'saved_results_changed':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('compare','migrate'));parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();plan=read(args.source/'plan.json')
    (compare if args.mode=='compare' else migrate)(plan,args.source)
