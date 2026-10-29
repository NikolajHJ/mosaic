"""Stdlib local seal/stage/submit/collect for the isolated source144 probe."""
import argparse
import json
import re
import shlex
import time
from pathlib import Path
from embedding_bridge_campaign import ROOT, WORK, SSH, call, save, seal, stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("seal","stage","submit","collect"))
    parser.add_argument("--attempt",default="attempt001")
    parser.add_argument("--stage",choices=("cpu","gpu"),default="cpu")
    args=parser.parse_args()
    if not re.fullmatch(r"attempt[0-9]{3}",args.attempt) or args.attempt=="attempt000":
        raise ValueError("Require a new immutable attempt001..999")
    label="source144-embedding-runtime-probe-"+args.attempt
    folder=ROOT/"outputs/hpc-submissions"/label
    if args.action=="seal":
        seal(folder,args.attempt,number=144,campaign_name="embedding-runtime-probe",
             config_relative="configs/embedding_runtime_probe.json",operator_relative="scripts/hpc/embedding_runtime_probe.py")
        return
    identity=json.loads((folder/"seal.json").read_text())
    if args.action=="stage":
        stage(folder,identity)
        return
    prelude=f'''from pathlib import Path
import json,subprocess,sys,shlex,re
identity={identity!r}; stage={args.stage!r}; root=Path({WORK!r}); release=Path(identity['release']); campaign=Path(identity['campaign'])
sys.path.insert(0,str(release/'scripts/hpc'))
from embedding_operator import authenticate,digest,quota,write_new
authenticate(release,identity['archive_sha256'],identity['plan_sha256'],'configs/embedding_runtime_probe.json')
assert digest(release/'embedding-release-manifest.json')==identity['manifest_sha256']
config=json.loads((release/'configs/embedding_runtime_probe.json').read_text())
control=root/'submissions'/identity['label']
'''
    if args.action=="submit":
        program=prelude+'''control.mkdir(parents=True,exist_ok=True); campaign.mkdir(parents=True,exist_ok=True)
intent=control/(stage+'-intent.json'); receipt=control/(stage+'-receipt.json')
assert not intent.exists() and not receipt.exists() and not (campaign/stage).exists(),'Stage already attempted; no overwrite/ambiguous retry'
fresh=quota(); dependency=None
command=[str(root/'env/boltz311/bin/python'),str(release/'scripts/hpc/embedding_runtime_probe.py'),stage,
 '--config',str(release/'configs/embedding_runtime_probe.json'),'--source-sha256',identity['archive_sha256'],
 '--plan-sha256',identity['plan_sha256'],'--output',str(campaign/stage)]
if stage=='gpu':
 cpu_path=campaign/'cpu/result.json'; cpu=json.loads(cpu_path.read_text()); prior=json.loads((control/'cpu-receipt.json').read_text())
 assert prior['status']=='submitted_not_passed' and cpu['job_id']==prior['job_id']
 assert cpu['status']=='passed' and cpu['source_sha256']==identity['archive_sha256'] and cpu['plan_sha256']==identity['plan_sha256']
 assert cpu['junit']['tests']>0 and all(cpu['junit'][key]==0 for key in ['failures','errors','skipped'])
 assert digest(cpu['junit_asset']['path'])==cpu['junit_asset']['sha256']
 r=subprocess.run(['bjobs','-a','-json','-o','jobid stat job_name',prior['job_id']],text=True,capture_output=True,timeout=40)
 assert r.returncode==0 and not r.stderr.strip()
 state=json.loads(r.stdout); assert state['COMMAND']=='bjobs' and state['JOBS']==1 and len(state['RECORDS'])==1
 row=state['RECORDS'][0]; assert row['JOBID']==prior['job_id']
 expected=identity['label']+'-cpu'
 if row.get('ERROR')=='Job <'+prior['job_id']+'> is not found':
  text=(root/'logs'/(expected+'.'+prior['job_id']+'.out')).read_text()
  assert 'Subject: Job '+prior['job_id']+': <'+expected+'> in cluster <dcc> Done' in text and 'Successfully completed.' in text
 else: assert row.get('STAT')=='DONE' and row.get('JOB_NAME')==expected
 dependency=dict(cpu_result_sha256=digest(cpu_path),cpu_job_id=prior['job_id'],scheduler=dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode))
 command.extend(['--cpu-sha256',dependency['cpu_result_sha256']])
name=identity['label']+'-'+stage
gpu_queue='gpul40s' if config.get('gpu_profile')=='l40s48' else 'gpua100'
bsub=['bsub','-J',name,'-q','hpc' if stage=='cpu' else gpu_queue,'-n','4','-R','span[hosts=1]',
 '-R','rusage[mem=8GB]' if stage=='cpu' else 'rusage[mem=16GB]','-W','0:45' if stage=='cpu' else '1:00',
 '-oo',str(root/'logs'/(name+'.%J.out')),'-eo',str(root/'logs'/(name+'.%J.err'))]
if stage=='gpu':
 if gpu_queue=='gpua100': bsub+=['-R','select[gpu80gb]']
 bsub+=['-gpu','num=1:mode=exclusive_process']
script='#!/bin/bash\\nset -euo pipefail\\ncd '+shlex.quote(str(release))+'\\nexec '+shlex.join(['env','PYTHONDONTWRITEBYTECODE=1','LATENTMOL_SOURCE_ARCHIVE='+str(release)+'.tar.gz','bash','scripts/hpc/boltz_run.sh',*command])+'\\n'
write_new(intent,dict(identity=identity,stage=stage,command=command,bsub=bsub,fresh_quota=fresh,dependency=dependency))
try: r=subprocess.run(bsub,input=script,text=True,capture_output=True,timeout=60)
except Exception as error:
 write_new(receipt,dict(status='ambiguous_submission_no_retry',error=str(error),intent_sha256=digest(intent))); raise
m=re.search(r'Job <([0-9]+)>',r.stdout); ok=r.returncode==0 and m is not None
value=dict(status='submitted_not_passed' if ok else 'submission_failed_no_retry',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,job_id=m.group(1) if m else None,intent_sha256=digest(intent))
write_new(receipt,value); assert ok,'Submission failed; inspect receipt'
print(json.dumps(value))
'''
    else:
        program=prelude+'''out=dict(identity=identity,control={},results={},logs={})
jobs=[]
for p in sorted(control.glob('*.json')):
 assert p.stat().st_size<1024*1024
 value=json.loads(p.read_text()); out['control'][p.name]=value
 if p.name.endswith('-receipt.json') and value.get('job_id'): jobs.append((p.stem.split('-')[0],value['job_id']))
for label,job in jobs:
 for suffix in ['out','err']:
  p=root/'logs'/(identity['label']+'-'+label+'.'+job+'.'+suffix)
  out['logs'][label+'.'+suffix]=p.read_text()[-22000:] if p.exists() else None
if jobs:
 r=subprocess.run(['bjobs','-a','-json','-o','jobid stat job_name',*[job for label,job in jobs]],text=True,capture_output=True,timeout=40)
 out['scheduler']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
for label in ['cpu','gpu']:
 p=campaign/label/'result.json'
 out['results'][label]=json.loads(p.read_text()) if p.exists() else None
 p=campaign/label/'cpu-tests.log'
 if p.exists(): out['logs']['cpu-tests.log']=p.read_text()[-18000:]
p=campaign/'gpu/setup.json'
if p.exists(): out['setup']=json.loads(p.read_text())
print(json.dumps(out))
'''
    stamp=str(time.time_ns())
    reply=call(SSH+["bash -lc "+shlex.quote(WORK+"/env/boltz311/bin/python -")],
        folder/(args.action+'-'+args.stage+'-'+stamp+'.json'),payload=program,timeout=180)
    result=json.loads(reply['stdout'])
    if args.action=='collect':
        target=ROOT/'outputs/hpc-results'/label; target.mkdir(parents=True,exist_ok=True)
        save(target/('status-'+stamp+'.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
