"""One compact read-only actual scheduler/artifact observation; no evaluation."""
from datetime import datetime,timezone
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);args=parser.parse_args()
    assert args.name.replace('-','').isalnum()
    out=LOCAL/args.name;out.mkdir(exist_ok=False)
    program=r'''
from datetime import datetime,timezone
from pathlib import Path
import base64,hashlib,json,subprocess
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-endpoint-continuation-01')
def bound(path):
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
jobs=json.loads((root/'jobs.json').read_bytes())
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'scheduler':{},'parents':{},'mutated':False}
for name,job in jobs.items():
 run=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=20)
 result['scheduler'][job]={'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
 parent=root/name;data={'submission':json.loads((parent/'submission.json').read_bytes()),'files':{}}
 for path in sorted(parent.rglob('*')):
  if not path.is_file() or path.is_symlink():continue
  rel=path.relative_to(parent).as_posix()
  if '/study/' in rel:continue
  if path.suffix in ('.json','.xml') or path.name in ('tests.log',f'stdout.{job}.log',f'stderr.{job}.log'):
   row={'binding':bound(path)}
   if path.stat().st_size<=512000:row['base64']=base64.b64encode(path.read_bytes()).decode()
   else:row['tail']=path.read_text(errors='replace')[-6000:]
   data['files'][rel]=row
 result['parents'][name]=data
for path in sorted(root.glob('*certificate.json')):
 result.setdefault('certificates',{})[path.name]={'binding':bound(path),'data':json.loads(path.read_bytes())}
print(json.dumps(result))
'''
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    with (out/'raw.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
        run=subprocess.run(argv,input=program.encode(),stdout=output,stderr=error,timeout=120)
    receipt={'returncode':run.returncode,'argv':argv,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),
             'raw_sha256':hashlib.sha256((out/'raw.json').read_bytes()).hexdigest(),'mutated':False}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    assert run.returncode==0,(out/'stderr.txt').read_text()
    result=json.loads((out/'raw.json').read_bytes())
    print(result['captured_utc'])
    for name,job in result['jobs'].items():
        actual=result['scheduler'][job]
        print(name,job,actual['returncode'],actual['stdout'][:180].replace('\n',' '))
        for rel,row in result['parents'][name]['files'].items():
            if rel.endswith('assessment.json') and 'base64' in row:
                import base64
                print(rel,base64.b64decode(row['base64']).decode()[:350])


if __name__=='__main__':main()
