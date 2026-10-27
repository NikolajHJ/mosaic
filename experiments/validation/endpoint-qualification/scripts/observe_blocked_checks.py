"""One read-only SSH observation of the current blocked validation attempts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

REPO = Path(__file__).resolve().parents[4]
PROGRAM = r'''
from datetime import datetime, timezone
from pathlib import Path
import base64, hashlib, json, subprocess
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'roots':{},'mutated':False}
ids=set()
for name in ('protenix-endpoint-fix-05','nonchem-endpoint-continuation-07'):
 root=base/name; rows={}; jobs={}
 if (root/'jobs.json').exists():jobs=json.loads((root/'jobs.json').read_bytes());ids.update(jobs.values())
 parents=[p for p in root.iterdir() if p.is_dir() and (p/'submission.json').exists()]
 for parent in parents:
  submission=json.loads((parent/'submission.json').read_bytes())
  ids.add(submission['job_id'])
  for path in parent.rglob('*'):
   if not path.is_file() or path.is_symlink():continue
   rel=path.relative_to(root).as_posix()
   if '/study/' in rel or '/oracle-points/' in rel:continue
   if path.suffix not in ('.json','.xml','.log') and path.name!='tests.log':continue
   size=path.stat().st_size
   with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
   row={'path':str(path),'bytes':size,'sha256':digest}
   if size<=2000000:row['base64']=base64.b64encode(path.read_bytes()).decode()
   else:row['tail']=path.read_text(errors='replace')[-12000:]
   rows[rel]=row
 for path in root.glob('*.json'):
  raw=path.read_bytes()
  rows[path.name]={'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()}
 result['roots'][name]={'jobs':jobs,'files':rows}
query=subprocess.run(['bjobs','-l',*sorted(ids)],capture_output=True,text=True,timeout=45)
result['scheduler']={'job_ids':sorted(ids),'returncode':query.returncode,'stdout':query.stdout,'stderr':query.stderr}
retirement=base/'storage-upload-retirement-01/completion.json'
if retirement.exists():
 raw=retirement.read_bytes();record=json.loads(raw)
 result['retirement']={'path':str(retirement),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
   'record':record,'archive_absent':not Path(record['archive']['path']).exists(),'other_files_match':{}}
 for name,row in record['preserved_other_files'].items():
  p=Path(name)
  with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  result['retirement']['other_files_match'][name]=p.stat().st_size==row['bytes'] and digest==row['sha256']
print(json.dumps(result))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--current',action='store_true')
    args=parser.parse_args()
    assert args.name.replace('-','').isalnum()
    out=REPO/'results/hpc/20261027-blocked-checks'/args.name
    out.mkdir(parents=True,exist_ok=False)
    program=PROGRAM
    if args.current:
        program=program.replace("('protenix-endpoint-fix-05','nonchem-endpoint-continuation-07')",
                                "('protenix-endpoint-fix-07','nonchem-endpoint-continuation-08','nonchem-endpoint-continuation-09')")
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'started_utc':datetime.now(timezone.utc).isoformat(),'argv':argv,
             'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'mutated':False,'status':'unknown'}
    try:
        with (out/'raw.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            run=subprocess.run(argv,input=program.encode(),stdout=stdout,stderr=stderr,timeout=180)
        receipt.update(status='returned',returncode=run.returncode)
    finally:
        receipt['ended_utc']=datetime.now(timezone.utc).isoformat()
        receipt['raw_sha256']=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest()
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    assert run.returncode==0,(out/'stderr.txt').read_text()
    data=json.loads((out/'raw.json').read_bytes())
    compact=re.sub(r'\s+','',data['scheduler']['stdout'])
    states=re.findall(r'Job<(\d+)>.*?Status<([^>]+)>',compact)
    print(json.dumps({'captured_utc':data['captured_utc'],'jobs':data['scheduler']['job_ids'],
                      'states':states,'scheduler_stderr':data['scheduler']['stderr'],'output':str(out)},indent=2))


if __name__=='__main__':main()
