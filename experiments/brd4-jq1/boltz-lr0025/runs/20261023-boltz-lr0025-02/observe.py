"""One compact read-only SSH query of actual own jobs and committed prefixes."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');root=base/'boltz-lr0025-02'
jobs=json.loads((root/'jobs.json').read_bytes());result={'utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'scheduler':{},'artifacts':{}}
for label,job in jobs.items():
 q=subprocess.run(['bjobs','-l',str(job)],capture_output=True,text=True,timeout=30)
 result['scheduler'][label]={'job':job,'returncode':q.returncode,'raw':q.stdout+q.stderr}
 for suffix in ['payload/study/assessment.json','payload/study/summary.json','payload/driver-invocation.json','storage-entry/gate.json']:
  path=root/label/suffix
  if path.is_file():
   with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   result['artifacts'][str(path)]={'bytes':path.stat().st_size,'sha256':digest}
 for field in ['states','updates']:
  path=root/label/'payload/study/case-0'/(field+'.jsonl')
  if path.is_file():
   count=0;last=None
   with path.open('rb') as stream:
    for line in stream:
     if not line.endswith(b'\n'):break
     last=json.loads(line);count+=1
   result['artifacts'][str(path)]={'newline_committed_rows':count,'last_local_step':None if last is None else last.get('step'),'last_global_step':None if last is None else last.get('global_step')}
for suffix in ['projection.json','boltz2-completion.json','boltz2_chemistry-completion.json']:
 path=root/suffix
 if path.is_file():result['artifacts'][str(path)]=json.loads(path.read_bytes())
print(json.dumps(result))
'''


def main():
    folder=REPO/'results/hpc'/HERE.name/'observations'
    folder.mkdir(parents=True,exist_ok=True)
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc','bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -c '+shlex.quote(REMOTE))]
    result=subprocess.run(args,capture_output=True,timeout=120)
    (folder/(timestamp+'-stdout.bin')).write_bytes(result.stdout)
    (folder/(timestamp+'-stderr.bin')).write_bytes(result.stderr)
    (folder/(timestamp+'-receipt.json')).write_text(json.dumps({'returncode':result.returncode,'args':args})+'\n')
    print(result.stdout.decode()[-16000:]);print(result.stderr.decode()[-1500:])


if __name__=='__main__':
    main()
