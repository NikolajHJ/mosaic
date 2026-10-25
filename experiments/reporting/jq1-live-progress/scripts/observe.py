"""Read-only compact scheduler/receipt collection for this exact report capsule."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'
PROGRAM=r'''
import base64,hashlib,json,os,subprocess
from pathlib import Path
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/nonchem-live-report-03')
source=root.parent/'nonchem-live-report-release-03/source'
jobs=json.loads((root/'jobs.json').read_bytes())
queries={}
for phase,job in jobs.items():
 p=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
 queries[phase]={'job_id':job,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
q=subprocess.run(['bjobs','-a','-u',os.environ['USER'],'-noheader','-o','jobid stat queue job_name:90'],capture_output=True,text=True,timeout=30)
files=[]
for path in sorted(root.rglob('*')):
 if path.is_file() and not path.is_symlink() and not path.is_relative_to(root/'render/viewer') and path.suffix in ('.json','.xml','.log'):
  assert path.stat().st_size<4*1024**2
  raw=path.read_bytes();files.append({'relative':path.relative_to(root).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'base64':base64.b64encode(raw).decode()})
archive=root/'edition-0002-archive.json'
ready=False
if archive.exists() and 'render' in jobs:
 stdout=root/'render'/('stdout.'+jobs['render']+'.log')
 ready=stdout.exists() and 'Successfully completed.' in stdout.read_text() and 'Status <DONE>' in queries['render']['stdout']
print(json.dumps({'jobs':jobs,'queries':queries,'account_queue':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},
 'source_manifest_sha256':hashlib.sha256((source/'source_manifest.json').read_bytes()).hexdigest(),
 'files':files,'publication_ready':ready,'archive_receipt':json.loads(archive.read_bytes()) if archive.exists() else None,'mutated':False}))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--generation',default='03');parser.add_argument('--edition',default='edition-0002');args=parser.parse_args()
    assert args.name.startswith('observation-') and '/' not in args.name and '\\' not in args.name
    assert re.fullmatch('[0-9]{2}',args.generation) and int(args.generation)>=3 and re.fullmatch('edition-[0-9]{4}',args.edition)
    program=PROGRAM.replace('nonchem-live-report-release-03','nonchem-live-report-release-'+args.generation).replace('nonchem-live-report-03','nonchem-live-report-'+args.generation).replace('edition-0002-archive.json',args.edition+'-archive.json')
    out=LOCAL/args.name;out.mkdir(exist_ok=False)
    (out/'program.py').write_text(program,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown','program_sha256':hashlib.sha256(program.encode()).hexdigest()}
    try:
        with (out/'raw.json').open('xb') as stdout,(out/'stderr.txt').open('xb') as stderr:
            run=subprocess.run(command,input=program.encode(),stdout=stdout,stderr=stderr,timeout=120)
        receipt.update(status='returned',returncode=run.returncode);assert run.returncode==0
        value=json.loads((out/'raw.json').read_bytes());safe=out/'files';safe.mkdir()
        for row in value['files']:
            p=PurePosixPath(row['relative']);assert not p.is_absolute() and '..' not in p.parts and '\\' not in row['relative']
            raw=base64.b64decode(row['base64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
            target=safe.joinpath(*p.parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        receipt['files_byte_verified']=len(value['files'])
    finally:
        receipt.update(captured_utc=datetime.now(timezone.utc).isoformat(),raw_sha256=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest())
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'jobs':value['jobs'],'publication_ready':value['publication_ready'],'files':len(value['files']),
                      'scheduler':{k:r['stdout'].splitlines()[0:4] for k,r in value['queries'].items()}},indent=2))


if __name__=='__main__':main()
