"""Retire only the superseded queued random10k pair; preserve every artifact."""
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
REMOTE = r'''
import fcntl,hashlib,json,re,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
old=base/'protenix-random-chem-01';target=base/'protenix-extended-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with (old/'dispatch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 jobs=json.loads((old/'jobs.json').read_bytes())
 assert jobs=={'cpu':'29796731','after_cpu':'29796732','native':'29796780','after_ready':'29796781'},jobs
 assert not any((old/n).exists() for n in ['run','long-dispatch-started.json'])
 assert not target.exists(),'Inspect existing new campaign before recovery'
 before={}
 for job in ['29796781','29796780']:
  q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
  raw=q.stdout+q.stderr;compact=re.sub(r'\s+','',raw)
  assert q.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[job]
  assert re.findall(r'Status<([^>]+)>',compact)==['PEND'],raw
  before[job]=raw
 target.mkdir();out=target/'predecessor-retirement';out.mkdir()
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'scope':'User requested replacement50k; no original molecule/results modified','before':before,'commands':[]}
 (out/'started.json').write_text(json.dumps(receipt,indent=2)+'\n')
 for job in ['29796781','29796780']:
  q=subprocess.run(['bkill',job],capture_output=True,text=True,timeout=30)
  receipt['commands'].append({'args':['bkill',job],'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
  (out/'commands.json').write_text(json.dumps(receipt,indent=2)+'\n')
  assert q.returncode==0,q.stderr
 receipt['after']={}
 for job in ['29796781','29796780']:
  for attempt in range(10):
   q=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
   raw=q.stdout+q.stderr;compact=re.sub(r'\s+','',raw)
   if q.returncode==0 and re.findall(r'Status<([^>]+)>',compact)==['EXIT']:break
   assert attempt<9,raw
   time.sleep(2)
  receipt['after'][job]=raw
 assert not any((old/n).exists() for n in ['run','long-dispatch-started.json'])
 receipt.update(status='intentionally_retired_unstarted',old_jobs_sha256=sha(old/'jobs.json'),original_saved_results_unchanged=True)
 (out/'assessment.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('RETIRE_BEGIN');print(json.dumps(receipt))
'''

def main():
    folder=REPO/'results/hpc'/HERE.name/'predecessor-retirement'
    folder.mkdir(parents=True,exist_ok=False)
    (folder/'started.json').write_bytes((json.dumps({'utc':datetime.now(timezone.utc).isoformat()})+'\n').encode())
    q=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=90)
    (folder/'stdout.bin').write_bytes(q.stdout);(folder/'stderr.bin').write_bytes(q.stderr)
    assert q.returncode==0,q.stderr.decode()[-2000:]
    value=json.loads(q.stdout.split(b'RETIRE_BEGIN\n')[1]);(HERE/'predecessor-retirement.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
    print(value['status'],list(value['after']))

if __name__=='__main__':main()
