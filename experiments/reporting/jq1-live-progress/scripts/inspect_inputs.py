"""One read-only SSH byte/JSON metadata inspection; no numerical evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-jq1-live-report-01'
PROGRAM=r'''
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
from pathlib import Path
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
def binding(path):
 path=Path(path);assert path.is_file() and not path.is_symlink()
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for b in iter(lambda:stream.read(1024**2),b''):h.update(b)
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}
def small(path):
 path=Path(path);assert path.stat().st_size<4*1024**2
 return {'binding':binding(path),'data':json.loads(path.read_bytes())}
specs=[('protenix','protenix-extended-03','continuation',[(0,10000,10000)],'protenix-extended-release-03/source','examples/configs/extended_continuation.json'),
 ('boltz','boltz-lr0025-03','boltz2',[(0,0,10000)],'boltz-lr0025-release-03/source','examples/configs/boltz_serial_boltz2.json'),
 ('dbmol','boltz-dbmol-pilot-02','pilot',[(0,0,2500),(1,2500,2500)],'boltz-dbmol-pilot-release-02/source','examples/configs/dbmol_pilot.json')]
arms=[];proofs=[];sources=[]
for key,rootrel,prefix,segments,sourcerel,configrel in specs:
 root=base/rootrel;source=base/sourcerel;m=small(source/'source_manifest.json')
 sources.append({'path':str(source),'manifest_sha256':m['binding']['sha256'],'files':len(m['data']['files'])})
 arm={'id':key,'root':str(root),'source':sources[-1],'config':binding(source/configrel),'segments':[]}
 for index,offset,requested in segments:
  parent=root/(prefix+'-segment-'+str(index).zfill(4));study=parent/'payload/study';case=study/'case-0'
  sub=small(parent/'submission.json');proofs.append(sub['binding'])
  job=sub['data']['job_id']
  row={'id':parent.name,'global_offset':offset,'requested_updates':requested,'case_directory':str(case),
       'anchor_directory':str(case),'states':binding(case/'states.jsonl'),'updates':binding(case/'updates.jsonl'),
       'summary':binding(study/'summary.json'),'config':arm['config'],'source_manifest_sha256':m['binding']['sha256'],
       'job_id':job,'submission':sub,'stdout':binding(parent/('stdout.'+job+'.log')),'stderr':binding(parent/('stderr.'+job+'.log')),
       'small_reports':{},'study_json_files':[]}
  for file in sorted(study.glob('*.json')):row['study_json_files'].append({'name':file.name,'bytes':file.stat().st_size})
  for file in [case/'report.json',case/'retained-frames.json',study/'assessment.json']:
   if file.exists():
    if file.stat().st_size<4*1024**2:row['small_reports'][str(file.relative_to(study))]=small(file)
    else:row[str(file.name).replace('.json','')+'_binding']=binding(file)
  term=study/'terminal_gradient_checks'
  if term.is_dir():
   for file in sorted(term.rglob('*.json')):
    if file.stat().st_size<4*1024**2:row['small_reports'][str(file.relative_to(study))]=small(file)
  first={};last={}
  for field in ['states','updates']:
   count=0
   with (case/(field+'.jsonl')).open() as stream:
    for line in stream:
     if not line.endswith('\n'):raise ValueError('terminal JSONL has uncommitted suffix')
     value=json.loads(line)
     if count==0:first[field]=value
     last[field]=value;assert value['step']==count;count+=1
  row['first_records']=first;row['last_records']=last
  arm['segments'].append(row)
  proofs.extend([row['stdout'],row['stderr']])
 arms.append(arm)
noise=Path('/work3/s234818/mosaic/runs/20261025-protenix-noisy-05');ready=small(noise/'cpu-ready.json')
noise_summary=small(noise/'gpu/summary.json')
noise_source=None
for proof in ready['data'].get('proofs',[]):
 if proof['path'].endswith('/source_manifest.json'):noise_source=str(Path(proof['path']).parent)
noise_row={'root':str(noise),'cpu_ready':ready,'summary':noise_summary,'source_path':noise_source,
 'states':binding(noise/'gpu/trajectory/states.jsonl'),'updates':binding(noise/'gpu/trajectory/updates.jsonl'),
 'retained_index':binding(noise/'gpu/trajectory/retained-frames.json'),'result':small(noise/'gpu/trajectory/result.json')}
for file in sorted(noise.glob('*.json')):
 if file.stat().st_size<1024**2:noise_row[file.name]=small(file)
if noise_source:
 s=small(Path(noise_source)/'source_manifest.json');sources.append({'path':noise_source,'manifest_sha256':s['binding']['sha256'],'files':len(s['data']['files'])})
 noise_row['source']=sources[-1]
q=subprocess.run(['bjobs','-u',os.environ.get('USER','s234818'),'-a','-noheader','-o','jobid stat queue job_name:90'],capture_output=True,text=True,timeout=30)
controls={}
for rel in ['protenix-extended-03','boltz-lr0025-03','boltz-dbmol-pilot-02','chemistry-full-model-01']:
 path=base/rel/'jobs.json'
 if path.exists():controls[rel]=small(path)
print(json.dumps({'captured_utc':datetime.now(timezone.utc).isoformat(),'arms':arms,'noise':noise_row,'sources':sources,
 'proofs':proofs,'controls':controls,'scheduler':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},
 'node':shutil.which('node'),'mutated':False},allow_nan=False))
'''


def main():
    out=LOCAL/'inputs-01';out.mkdir(exist_ok=False,parents=True)
    (out/'program.py').write_text(PROGRAM,encoding='utf8',newline='\n')
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
             'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    record={'status':'unknown','program_sha256':hashlib.sha256(PROGRAM.encode()).hexdigest()}
    try:
        with (out/'raw.json').open('xb') as output,(out/'stderr.txt').open('xb') as error:
            result=subprocess.run(command,input=PROGRAM.encode(),stdout=output,stderr=error,timeout=180)
        record.update(status='returned',returncode=result.returncode)
    except BaseException as error:
        record.update(status='unknown_partial_preserved',error=repr(error));raise
    finally:
        record.update(completed_utc=datetime.now(timezone.utc).isoformat(),raw_sha256=hashlib.sha256((out/'raw.json').read_bytes()).hexdigest())
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    if result.returncode:
        print((out/'stderr.txt').read_text(errors='replace')[-5000:]);raise SystemExit(result.returncode)
    data=json.loads((out/'raw.json').read_bytes())
    print(json.dumps({'arms':[(a['id'],len(a['segments'])) for a in data['arms']],
                      'noise_source':data['noise']['source_path'],'node':data['node']},indent=2))
    print(data['scheduler']['stdout']);print(data['scheduler']['stderr'])


if __name__=='__main__':main()
