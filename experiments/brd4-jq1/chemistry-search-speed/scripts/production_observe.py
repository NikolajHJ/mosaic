"""One-connection read-only compact chemistry production evidence observation.

Only metadata/source bytes are transferred. Native arrays, complete optimizer
study summaries, case reports and loss JSONL remain authenticated HPC references.
Compact validation summaries may be copied as explicit adoption metadata.
There is no launch, recovery, cleanup, inference or numerical evaluation path.
"""
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
LOCAL=REPO/'results/hpc/20261026-chemistry-production-02'
PROGRAM=r'''
import base64,hashlib,json,os,re,stat,subprocess
from datetime import datetime,timezone
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'chemistry-production-02';source=base/'chemistry-production-release-02/source'
limit=64*1024**2
record={'version':'chemistry-production-observation-v1','captured_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(root),'source':str(source),'read_only':True,'files':[],'omitted':[],
        'scope':'Metadata/source bytes only; native arrays/full study summaries/case reports/JSONL remain remote'}
total=0;by_remote={}
def require(value,message):
 if not value:raise ValueError(message)
def collect(path,scope,relative):
 global total
 path=Path(path)
 if str(path) in by_remote:return by_remote[str(path)]
 for attempt in range(3):
  try:
   before=path.lstat();require(stat.S_ISREG(before.st_mode) and not path.is_symlink(),'Linked/special evidence file')
   if before.st_size>8*1024**2:
    record['omitted'].append({'path':str(path),'bytes':before.st_size,'reason':'Compact per-file8MiB limit; remote reference only'})
    return None
   require(total+before.st_size<=limit,'Complete compact evidence budget exceeded')
   data=path.read_bytes();after=path.lstat()
   require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
           (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and len(data)==after.st_size,
           'Metadata changed during exact byte capture')
   break
  except (FileNotFoundError,ValueError):
   if attempt==2:raise
 row={'scope':scope,'relative':relative,'path':str(path),'bytes':len(data),
      'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
 record['files'].append(row);by_remote[str(path)]=row;total+=len(data)
 return row
def ordinary_paths(directory,scope):
 if not directory.exists():return
 require(directory.is_dir() and not directory.is_symlink(),'Root is linked/special')
 for parent,dirs,files in os.walk(directory,followlinks=False):
  for name in list(dirs):
   p=Path(parent)/name
   require(p.is_dir() and not p.is_symlink(),'Evidence directory is linked/special')
   if scope=='root' and (name in ('cache','cuda-cache','pytest-cache','tmp','study','acceleration') or name.startswith('node-tmp')):
    dirs.remove(name)
  for name in sorted(files):
   p=Path(parent)/name
   if scope=='source' or p.suffix in ('.json','.xml','.log','.txt'):
    collect(p,scope,p.relative_to(directory).as_posix())
ordinary_paths(source,'source');ordinary_paths(root,'root')
record['root_present']=root.is_dir();record['source_present']=source.is_dir()
submissions=[];unknown=[]
for row in list(record['files']):
 if row['scope']!='root' or Path(row['relative']).name!='submission.json':continue
 value=json.loads(base64.b64decode(row['base64']))
 if value.get('status')=='submitted' and re.fullmatch('[1-9][0-9]*',str(value.get('job_id',''))):
  submissions.append({'path':row['path'],'job_id':value['job_id'],'phase':value.get('phase'),
                      'arm':value.get('arm'),'index':value.get('index')})
 else:unknown.append({'path':row['path'],'status':value.get('status'),'job_id':value.get('job_id')})
record['actual_submissions']=submissions;record['unknown_submission_records']=unknown
submitted_paths={row['path'] for row in submissions}
for row in list(record['files']):
 if row['scope']=='root' and Path(row['relative']).name=='submission-started.json':
  expected=str(Path(row['path']).with_name('submission.json'))
  if expected not in submitted_paths:
   value=json.loads(base64.b64decode(row['base64']))
   unknown.append({'path':row['path'],'status':value.get('status'),
                   'job_id':value.get('job_id'),'reason':'Exclusive start marker has no authenticated submitted record'})
record['production_phase_entries']=[]
if root.exists():
 for arm in ('protenix','boltz'):
  directory=root/arm
  if not directory.exists():continue
  require(directory.is_dir() and not directory.is_symlink(),'Arm root is linked/special')
  for path in directory.iterdir():
   if re.fullmatch(r'(?:random|boltz2_chemistry)-(?:after-)?(?:native|segment)-[0-9]{4}',path.name):
    record['production_phase_entries'].append({'path':str(path),'directory':path.is_dir(),'symlink':path.is_symlink()})
ids=sorted({row['job_id'] for row in submissions},key=int)
record['scheduler_query']={'job_ids':ids,'invoked':bool(ids)}
if ids:
 result=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=45)
 record['scheduler_query'].update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
# Follow only explicit byte-bound compact metadata from completed adoption
# records. These external files remain references, not an inferred native pass.
references=[]
for row in list(record['files']):
 if row['scope']=='root' and Path(row['relative']).name=='adoption.json':
  value=json.loads(base64.b64decode(row['base64']))
  references.extend(value.get('proofs',[]));references.extend(value.get(key,{}) for key in ('readiness','summary'))
  context=value.get('execution_context',{})
  references.extend(context.get(key,{}) for key in ('qualified_context','native_assessment','native_LSF'))
for binding in references:
 if not isinstance(binding,dict) or not {'path','bytes','sha256'}<=set(binding):continue
 p=Path(binding['path'])
 require(p.is_absolute() and '..' not in p.parts and p.is_relative_to(base),'External metadata binding escapes Mosaic analysis root')
 if p.suffix not in ('.json','.xml','.log','.txt'):
  record['omitted'].append({'path':str(p),'reason':'Nonmetadata scientific reference, not downloaded'});continue
 row=collect(p,'external',format(len([x for x in record['files'] if x['scope']=='external']),'04d')+p.suffix)
 if row is not None:
  require(row['bytes']==binding['bytes'] and row['sha256']==binding['sha256'],'Adoption external metadata bytes differ')
  if p.name=='qualified-context.json':
   qualified=json.loads(base64.b64decode(row['base64']))
   require(qualified.get('version')=='new-full-model-context-adoption-v1','Unexpected qualified-context metadata')
   references.extend(qualified.get('proofs',[]))
record['compact_file_bytes']=total
record['completed_utc']=datetime.now(timezone.utc).isoformat()
print(json.dumps(record))
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('label');args=parser.parse_args()
    if re.fullmatch('[a-z0-9][a-z0-9-]{0,100}',args.label) is None:raise ValueError('Invalid unique observation label')
    if REPO.name!='mosaic':raise ValueError('Wrong authorized repository')
    LOCAL.mkdir(parents=True,exist_ok=True)
    out=LOCAL/('observation-'+args.label);out.mkdir(exist_ok=False)
    (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown_readonly_query_outcome','program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (out/'raw.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
            result=subprocess.run(command,input=PROGRAM.encode(),stdout=output,stderr=error,timeout=180)
        receipt.update(returncode=result.returncode,completed_utc=datetime.now(timezone.utc).isoformat(),
                       stdout_sha256=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest())
        if result.returncode:raise RuntimeError('Single read-only SSH observation failed; no reconnect attempted')
        packet=json.loads((out/'raw.json').read_bytes());records=[];seen=set()
        for row in packet['files']:
            relative=PurePosixPath(row['scope'])/row['relative']
            if relative.is_absolute() or '..' in relative.parts or '\\' in str(relative) or str(relative) in seen:
                raise ValueError('Unsafe/duplicate compact metadata path')
            if row['scope'] not in ('root','source','external'):raise ValueError('Unknown compact metadata scope')
            seen.add(str(relative));data=base64.b64decode(row['base64'],validate=True)
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                raise ValueError('Captured metadata bytes/hash differ')
            target=out/'files'/Path(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            records.append({key:row[key] for key in ('scope','relative','path','bytes','sha256')})
        (out/'byte-verification.json').write_text(json.dumps({'status':'verified_compact_bytes_only',
            'files':records,'native_arrays_downloaded':False,'scientific_pass_claim':False},indent=2)+'\n')
        receipt['status']='readonly_compact_observation_byte_verified'
    except BaseException as error:
        receipt['error']=repr(error);raise
    finally:(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'directory':str(out),'files':len(records),'actual_jobs':packet['actual_submissions'],
                      'unknown_submissions':packet['unknown_submission_records'],'status':receipt['status']},indent=2))


if __name__=='__main__':main()
