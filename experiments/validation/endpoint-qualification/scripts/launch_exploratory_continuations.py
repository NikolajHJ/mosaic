"""One explicit submission per fresh exploratory arm; no legacy gate replay."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from exploratory_continuation import binding,bound,capacity,read,require,validate,write


def exclusive(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def submit(plan,arm,source,plan_path=None):
    from gpu_policy import selected,resource
    gpu=selected(plan["arms"][arm]);plan_path=Path(plan_path) if plan_path else source/"plan.json"
    row=validate(plan['arms'][arm]);root=Path(plan['root']);root.mkdir(parents=True,exist_ok=True)
    parent=root/arm;parent.mkdir(exist_ok=True)
    require(not list(parent.glob('submission*.json')),'Existing or unknown submission; reconcile before retry')
    require(not (parent/'payload').exists(),'Existing output; never overwrite')
    for key in ('config','snapshot','reference','pocket'):bound(row[key])
    account=subprocess.run(['bjobs','-u','s234818','-json','-o','jobid stat job_name command'],capture_output=True,text=True,check=True)
    records=json.loads(account.stdout)['RECORDS']
    live=[r for r in records if r['STAT'] not in ('DONE','EXIT')]
    require(not any(r['JOBID'] in row.get('predecessor_job_ids',[]) for r in live),'Predecessor still active')
    name=plan.get('job_name_prefix','mosaic-explore')+'-'+arm
    require(not any(r['JOB_NAME']==name or str(parent) in r['COMMAND'] for r in live),'Existing active writer')
    gate=capacity(plan,'dispatch');write(parent/'capacity-dispatch.json',gate)
    command=['bash','-l',str(source/'exploratory_continuation_stage.sh'),str(source),arm,'run',str(plan_path)]
    argv=['bsub','-H','-q',gpu['queue'],'-J',name,'-n',str(row['slots']),'-R','span[hosts=1]',
        '-R','rusage[mem='+str(row['memory_per_slot_mb'])+']','-W',row['walltime'],
        '-gpu',resource(row),'-env','all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
        '-o',str(parent/'stdout.%J.log'),'-e',str(parent/'stderr.%J.log'),*command]
    intent={'argv':argv,'job_name':name,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'research_profile':'exploratory','plan':binding(plan_path),'status':'submission_outcome_unknown_until_receipt'}
    exclusive(parent/'submission-intent.json',intent)
    try:r=subprocess.run(argv,capture_output=True,text=True,timeout=90)
    except subprocess.TimeoutExpired as error:
        write(parent/'submission-timeout.json',{'stdout':str(error.stdout),'stderr':str(error.stderr),'unknown_outcome':True})
        raise
    write(parent/'submission-response.json',{'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    matches=re.findall(r'Job <([1-9][0-9]*)> is submitted',r.stdout)
    require(r.returncode==0 and len(matches)==1,'Ambiguous submission; preserve and reconcile')
    job=matches[0];receipt={**intent,'job_id':job,'status':'submitted','dispatch_state':'held'}
    exclusive(parent/'submission.json',receipt)
    registry=root/'jobs.json';jobs=read(registry) if registry.exists() else {};require(arm not in jobs,'Registered arm already exists')
    jobs[arm]=job;write(registry,jobs)
    observations=[]
    for _ in range(12):
        r=subprocess.run(['bjobs','-json','-o','jobid user stat job_name',job],capture_output=True,text=True,check=True)
        rows=json.loads(r.stdout)['RECORDS'];require(len(rows)==1,'Ambiguous held job')
        record=rows[0];observations.append(record)
        require(record['JOBID']==job and record['USER']=='s234818' and record['JOB_NAME']==name,'Held job identity differs')
        if record['STAT']=='PSUSP':break
        require(record['STAT']=='PEND','Unexpected held state; do not resume');time.sleep(5)
    require(observations[-1]['STAT']=='PSUSP','Held state did not settle; no automatic retry')
    write(parent/'held-proof.json',observations)
    r=subprocess.run(['bresume',job],capture_output=True,text=True,timeout=60)
    write(parent/'release.json',{'job_id':job,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
    require(r.returncode==0,'Release outcome needs reconciliation')
    print(json.dumps({'arm':arm,'job_id':job,'status':'submitted_and_released','walltime':row['walltime'],
        'start':row['start'],'target':row['target'],'capacity_margin_bytes':gate['byte_margin']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--arm',required=True)
    args=parser.parse_args();plan=read(args.plan);submit(plan,args.arm,Path(plan['source_root']),args.plan)
