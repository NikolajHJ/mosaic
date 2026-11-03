"""Fetch fresh compact audit evidence; never submit or perform model work."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

HELPER = r'C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe'
REPO = Path(__file__).resolve().parents[4]
PROGRAM = r'''
import datetime,hashlib,json,subprocess
from pathlib import Path
root=Path(ROOT)
def file(path):
 if not path.is_file():return dict(path=str(path),exists=False)
 size=path.stat().st_size
 if path.suffix=='.log' and size>65536:
  with path.open('rb') as f:f.seek(-65536,2);raw=f.read()
  return dict(path=str(path),bytes=len(raw),full_bytes=size,tail_only=True,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8'))
 raw=path.read_bytes()
 return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode('utf-8'))
data=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(root),files={})
for name in ['submission-intent.json','submission-response.json','submission.json','submission-unknown.json','held-proof.json','release.json','dispatch-complete.json','capacity-dispatch-raw.json','capacity-entry.json','startup.json','source/checksums.json','source/plan.json','payload/result.json']:
 data['files'][name]=file(root/name)
receipt=root/'submission.json'
if receipt.is_file():
 job=json.loads(receipt.read_text())['job_id']
 for name in ['stdout.'+job+'.log','stderr.'+job+'.log']:data['files'][name]=file(root/name)
 for command in [['bjobs','-a','-w',job],['bhist','-l',job]]:
  p=subprocess.run(command,capture_output=True,text=True,timeout=25)
  data[command[0]]=dict(argv=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 data['committed_prefixes']={}
 for arm in ['original_logits','saved_repair']:
  prefix={}
  for filename in ['states.jsonl','updates.jsonl']:
   path=root/'payload'/arm/'study'/filename;count=0;last=None;contiguous=True
   if path.is_file():
    with path.open('rb') as f:
     for line in f:
      if not line.endswith(b'\n'):break
      value=json.loads(line);step=value.get('step',value.get('global_step'))
      if step!=count:contiguous=False
      count+=1;last=step
   prefix[filename]=dict(count=count,last_step=last,contiguous=contiguous)
  data['committed_prefixes'][arm]=prefix
print(json.dumps(data,allow_nan=False))
'''

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);args=p.parse_args()
 assert all(c.isalnum() or c=='-' for c in args.tag),'Unsafe tag'
 label=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 output=REPO/'results/hpc'/args.tag/('status-'+label);output.mkdir(parents=True,exist_ok=False)
 program='ROOT='+repr('/work3/s234818/mosaic/runs/'+args.tag)+'\n'+PROGRAM
 (output/'program.py').write_text(program,encoding='utf-8',newline='\n')
 result=subprocess.run([HELPER,'exec','--timeout','90',"bash -lc 'python3 -B -'"],input=program,text=True,capture_output=True,timeout=105)
 (output/'stdout.json').write_text(result.stdout,encoding='utf-8');(output/'stderr.txt').write_text(result.stderr,encoding='utf-8')
 if result.returncode:raise RuntimeError('Observation unavailable: '+result.stderr[-1000:])
 packet=json.loads(result.stdout)
 for name,row in packet['files'].items():
  if row.get('exists') is False:continue
  raw=row['text'].encode('utf-8')
  assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'],name
 print('evidence',output)
 print(packet.get('bjobs',{}).get('stdout','No accepted job receipt'))
 scientific=packet['files'].get('payload/result.json',{})
 if 'text' in scientific:
  data=json.loads(scientific['text'])
  print(json.dumps({k:data.get(k) for k in ['status','job_id','native_forward_evaluations','elapsed_seconds','frozen_unchanged','inputs_unchanged','error']}))
  print('saved_points',len(data.get('points',[])))
  for arm in data.get('arms',[]):
   print(json.dumps({k:arm.get(k) for k in ['arm','status','applied_updates','strict_valid_states','elapsed_seconds']}))
 else:print('No native result yet')
 for arm,prefix in packet.get('committed_prefixes',{}).items():print(arm,json.dumps(prefix))

if __name__=='__main__':main()
