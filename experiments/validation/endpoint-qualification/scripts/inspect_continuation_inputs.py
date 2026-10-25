"""One read-only compact SSH inventory; never load arrays/models or submit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

REPO=Path(__file__).resolve().parents[4]
LOCAL=REPO/'results/hpc/20261025-continuation-admission-01'
PROGRAM=r'''
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,os,subprocess
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
def binding(path):
 path=Path(path);assert path.is_file() and not path.is_symlink()
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
jobs={}
for job in ['29809291','29809292','29809297','29809298','29810971','29810972','29811883','29811884']:
 p=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=20)
 jobs[job]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
p=subprocess.run(['bjobs','-a','-u',os.environ['USER'],'-noheader','-o','jobid stat queue job_name:100'],capture_output=True,text=True,timeout=20)
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'mutated':False,'jobs':jobs,
 'account_queue':{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},'arms':[]}
for arm in PLAN['arms']:
 if arm['id']=='protenix_noisy':continue
 segment=arm['segments'][-1];case=Path(segment['case_directory']);study=Path(segment['summary']['path']).parent
 row={'id':arm['id'],'global_start':arm['global_start'],'global_target':arm['global_target'],'rate':arm['rate'],
      'bindings':{},'raw_terminal':None,'original_summary_status':None}
 for name in ['states','updates','summary','config','stdout','stderr']:
  actual=binding(segment[name]['path']);assert actual==segment[name],name+' input changed';row['bindings'][name]=actual
 summary=json.loads(Path(segment['summary']['path']).read_bytes());row['original_summary_status']=summary['status']
 counts={}
 for field in ['states','updates']:
  raw=Path(segment[field]['path']).read_bytes();boundary=raw.rfind(b'\n')+1
  records=[json.loads(line) for line in raw[:boundary].splitlines()]
  counts[field]={'newline_committed_rows':len(records),'first_step':records[0].get('step'),'last_step':records[-1].get('step'),'uncommitted_suffix_bytes':len(raw)-boundary}
 row['prefixes']=counts
 final=json.loads(Path(segment['states']['path']).read_bytes().splitlines()[-1]);snapshot=SNAPSHOTS[arm['id']]
 assert binding(snapshot['path'])==snapshot
 row['endpoint_snapshot']=snapshot;row['last_state_metadata']={key:final.get(key) for key in ['step','global_step','snapshot_retained','snapshot_sha256','snapshot','key_data'] if key in final}
 for name in ['terminal_gradient_checks/report.json','native_gradient_checks/report.json','assessment.json']:
  path=study/name
  if path.is_file():row[name]={'binding':binding(path),'data':json.loads(path.read_bytes())}
 parent=study.parent.parent
 row['producer_submission']={'binding':binding(parent/'submission.json'),'data':json.loads((parent/'submission.json').read_bytes())}
 row['original_LSF_tail']=Path(segment['stdout']['path']).read_text()[-7000:]
 row['original_stderr_tail']=Path(segment['stderr']['path']).read_text()[-7000:]
 for name in ['continuation-segment-0001','boltz2-segment-0001','pilot-segment-0002']:
  if (parent.parent/name).exists():row.setdefault('later_sibling_markers',[]).append(str(parent.parent/name))
 result['arms'].append(row)
for name in ['nonchem-stride20-02/startup-fix-03/jobs.json','protenix-extended-03/jobs.json','boltz-lr0025-03/jobs.json','boltz-dbmol-pilot-02/jobs.json']:
 path=base/name
 if path.is_file():result.setdefault('controller_jobs',{})[name]={'binding':binding(path),'data':json.loads(path.read_bytes())}
for command in ['getquota_work3.sh','getquota_zhome.sh']:
 p=subprocess.run([command],capture_output=True,text=True,timeout=30)
 result.setdefault('quota_snapshots_not_admission',{})[command]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(result))
'''


def main():
    plan=json.loads((REPO/'results/hpc/20261025-jq1-live-report-01/source04/plan.json').read_bytes())
    viewer=REPO/'results/hpc/20261025-jq1-live-report-01/viewer/editions/edition-0003'
    manifest=json.loads((viewer/'manifest.json').read_bytes());snapshots={}
    for arm in manifest['arms']:
        if arm['id']=='protenix_noisy':continue
        chunk=json.loads((viewer/arm['latest']['chunk']).read_bytes())
        snapshots[arm['id']]=chunk['frames'][arm['latest']['index']]['source_snapshot']
    program='PLAN='+repr(plan)+'\nSNAPSHOTS='+repr(snapshots)+'\n'+PROGRAM
    LOCAL.mkdir(exist_ok=False);(LOCAL/'program.py').write_text(program,encoding='utf8',newline='\n')
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',
          'bash -l -c '+shlex.quote('/work3/s234818/mosaic/envs/protenix-tiny/bin/python -')]
    receipt={'status':'unknown','mutated':False,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'argv':argv}
    try:
        with (LOCAL/'raw.json').open('xb') as out,(LOCAL/'stderr.txt').open('xb') as err:
            run=subprocess.run(argv,input=program.encode(),stdout=out,stderr=err,timeout=150)
        receipt.update(status='returned',returncode=run.returncode)
        assert run.returncode==0,(LOCAL/'stderr.txt').read_text()
    finally:
        receipt['captured_utc']=datetime.now(timezone.utc).isoformat()
        receipt['raw_sha256']=hashlib.sha256((LOCAL/'raw.json').read_bytes()).hexdigest()
        (LOCAL/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8',newline='\n')
    result=json.loads((LOCAL/'raw.json').read_bytes())
    print(json.dumps({'captured_utc':result['captured_utc'],'arms':[{k:a[k] for k in ['id','global_target','original_summary_status','endpoint_snapshot','prefixes']} for a in result['arms']],
                      'queue':result['account_queue']['stdout'],'mutated':False},indent=2))


if __name__=='__main__':main()
