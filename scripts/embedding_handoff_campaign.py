"""Local stdlib packaging and controlled activation of the source143 repair.

Numerical workers retain the immutable source142 release and trained mapping.
This wrapper never edits either HPC release or retries an ambiguous submission.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import time

from embedding_bridge_campaign import ROOT, WORK, SSH, call, save, seal, stage


def invoke(folder, program, action, *, timeout=180):
    return call(SSH + ["bash -lc " + shlex.quote(WORK + "/env/boltz311/bin/python -")],
                folder / (action + "-" + str(time.time_ns()) + ".json"),
                payload=program, timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "stage", "submit-cpu", "collect", "resume"))
    parser.add_argument("--attempt", default="attempt001")
    args = parser.parse_args()
    if not re.fullmatch(r"attempt[0-9]{3}", args.attempt) or args.attempt == "attempt000":
        raise ValueError("Require a new immutable attempt001..attempt999")
    label = "source143-embedding-handoff-recovery-" + args.attempt
    folder = ROOT / "outputs/hpc-submissions" / label
    if args.action == "seal":
        seal(folder, args.attempt, number=143, campaign_name="embedding-handoff-recovery",
             config_relative="configs/embedding_handoff_recovery.json",
             operator_relative="scripts/hpc/embedding_handoff_recovery.py")
        return
    identity = json.loads((folder / "seal.json").read_text())
    if args.action == "stage":
        stage(folder, identity)
        return
    prelude = f'''from pathlib import Path
import json,subprocess,sys,hashlib,shlex,re,time
identity={identity!r}
release=Path(identity['release']); root=Path({WORK!r})
sys.path.insert(0,str(release/'scripts/hpc'))
from embedding_operator import authenticate,digest,quota,write_new
authenticate(release,identity['archive_sha256'],identity['plan_sha256'],'configs/embedding_handoff_recovery.json')
assert digest(release/'embedding-release-manifest.json')==identity['manifest_sha256']
config=json.loads((release/'configs/embedding_handoff_recovery.json').read_text())
control=root/'submissions'/identity['label']; campaign=Path(identity['campaign'])
'''
    if args.action == "submit-cpu":
        program = prelude + '''control.mkdir(parents=True,exist_ok=True)
campaign.mkdir(parents=True,exist_ok=True)
intent=control/'readiness-cpu-intent.json'; receipt=control/'readiness-cpu-receipt.json'
assert not intent.exists() and not receipt.exists() and not (campaign/'readiness').exists(), 'Readiness submission already attempted'
fresh=quota()
argv=[str(root/'env/boltz311/bin/python'),str(release/'scripts/hpc/embedding_handoff_readiness.py'),
 '--config',str(release/'configs/embedding_handoff_recovery.json'),
 '--source-sha256',identity['archive_sha256'],'--manifest-sha256',identity['manifest_sha256'],
 '--plan-sha256',identity['plan_sha256'],'--output',str(campaign/'readiness'),'--seconds','2400']
job_name=identity['label']+'-cpu'
bsub=['bsub','-J',job_name,'-q','hpc','-n','4','-W','0:45','-R','rusage[mem=8GB]','-R','span[hosts=1]',
 '-oo',str(root/'logs'/(job_name+'.%J.out')),'-eo',str(root/'logs'/(job_name+'.%J.err'))]
script='#!/bin/bash\\nset -euo pipefail\\ncd '+shlex.quote(str(release))+'\\nexec '+shlex.join([
 'env','PYTHONDONTWRITEBYTECODE=1','LATENTMOL_SOURCE_ARCHIVE='+str(release)+'.tar.gz',
 'bash','scripts/hpc/boltz_run.sh',*argv])+'\\n'
write_new(intent,dict(identity=identity,command=argv,bsub=bsub,script=script,fresh_quota=fresh,status='intent_not_submitted'))
try:
 r=subprocess.run(bsub,input=script,text=True,capture_output=True,timeout=60)
except Exception as error:
 write_new(receipt,dict(status='ambiguous_submission_no_retry',error=str(error),intent_sha256=digest(intent)))
 raise
m=re.search(r'Job <([0-9]+)>',r.stdout); passed=r.returncode==0 and m is not None
value=dict(status='submitted_not_passed' if passed else 'submission_failed_no_retry',returncode=r.returncode,
 stdout=r.stdout,stderr=r.stderr,job_id=m.group(1) if m else None,intent_sha256=digest(intent))
write_new(receipt,value)
assert passed,'LSF did not confirm submission; do not retry'
print(json.dumps(value))
'''
    elif args.action == "collect":
        program = prelude + '''out=dict(identity=identity,files={})
for path in sorted(control.glob('*.json')):
 assert path.stat().st_size<8*1024*1024
 out['files'][path.name]=json.loads(path.read_text())
proof=campaign/'readiness/readiness.json'
out['readiness']=json.loads(proof.read_text()) if proof.exists() else None
receipt=control/'readiness-cpu-receipt.json'
if receipt.exists():
 job=json.loads(receipt.read_text()).get('job_id')
 if job:
  r=subprocess.run(['bjobs','-a','-json','-o','jobid stat job_name',job],text=True,capture_output=True,timeout=40)
  out['cpu_scheduler']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
  for suffix in ['out','err']:
   p=root/'logs'/(identity['label']+'-cpu.'+job+'.'+suffix)
   out['cpu_'+suffix]=p.read_text()[-16000:] if p.exists() else None
for name in ['cpu-tests.log','actual-pilot-cli.log']:
 p=campaign/'readiness'/name
 out[name]=p.read_text()[-18000:] if p.exists() else None
print(json.dumps(out))
'''
    else:
        program = prelude + '''proof=campaign/'readiness/readiness.json'
receipt=json.loads((control/'readiness-cpu-receipt.json').read_text())
readiness=json.loads(proof.read_text())
assert readiness['status']=='passed' and readiness['job_id']==receipt['job_id']
assert receipt['status']=='submitted_not_passed'
assert receipt['intent_sha256']==digest(control/'readiness-cpu-intent.json')
for key,value in [('source_sha256',identity['archive_sha256']),('manifest_sha256',identity['manifest_sha256']),('plan_sha256',identity['plan_sha256'])]:
 assert readiness[key]==value
parent=config['parent']
argv=[str(root/'env/boltz311/bin/python'),str(release/'scripts/hpc/embedding_handoff_recovery.py'),
 'advance','--after','fit','--release',parent['release'],'--campaign',parent['campaign'],
 '--archive-sha256',parent['source_sha256'],'--manifest-sha256',parent['manifest_sha256'],'--plan-sha256',parent['plan_sha256'],
 '--repair-release',str(release),'--repair-archive-sha256',identity['archive_sha256'],
 '--repair-manifest-sha256',identity['manifest_sha256'],'--repair-plan-sha256',identity['plan_sha256'],
 '--readiness',str(proof),'--readiness-sha256',digest(proof)]
r=subprocess.run(argv,text=True,capture_output=True,timeout=150)
print(json.dumps(dict(command=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
'''
    reply = invoke(folder, program, args.action)
    value = json.loads(reply["stdout"])
    if args.action == "collect":
        target = ROOT / "outputs/hpc-results" / label
        target.mkdir(parents=True, exist_ok=True)
        save(target / ("status-" + str(time.time_ns()) + ".json"), value)
    print(json.dumps(value, indent=2))
    if args.action == "resume" and value.get("returncode") != 0:
        raise RuntimeError("Recovery handoff stopped; inspect the saved receipt before retrying")


if __name__ == "__main__":
    main()
