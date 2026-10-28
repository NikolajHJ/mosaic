"""Read-only stdlib capture of the six current saved Mosaic trajectories."""
from pathlib import Path
import importlib.util
import json

REPO = Path(__file__).resolve().parents[4]
LOCAL = REPO / 'results/hpc/20261028-jq1-current-report-05'
PROGRAM = r'''
import json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timezone
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
assert not (base/'nonchem-live-report-05').exists() and not (base/'nonchem-live-report-release-05').exists(),'Existing/unknown report outcome; inspect first'
source=base/'nonchem-live-report-release-04/source'
def binding(path):
 path=Path(path);assert path.is_absolute() and path.is_file() and all(not p.is_symlink() for p in (path,*path.parents))
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest}
assert binding(source/'source_manifest.json')['sha256']=='856f788c5959c6fe5239374d3141808ec011907f4e092deb78a65f28ef9b196d'
old=json.loads((source/'source_manifest.json').read_bytes());plan=json.loads((source/'plan.json').read_bytes())
proofs=[];sources=list(old['parent_sources']);samples={}
def counts(segment):
 result={};sample={}
 for field in ('states','updates'):
  n=0;frames=0;first=last=None
  with Path(segment[field]['path']).open('rb') as stream:
   for line in stream:
    assert line.endswith(b'\n'),'Terminal input contains incomplete suffix'
    row=json.loads(line);assert row.get('step',row.get('update',-1)-1)==n
    if field=='states':
     assert row.get('global_step',segment['global_offset']+n)==segment['global_offset']+n
     assert type(row['snapshot_retained']) is bool;frames+=int(row['snapshot_retained'])
    first=row if n==0 else first;last=row;n+=1
  result[field]=n
  if field=='states':result['retained_frames']=frames
  sample[field]={'first':first,'last':last}
 assert result['states']==result['updates']+1
 segment['expected_counts']={'committed_updates':result['updates'],'scalar_states':result['states'],'retained_frames':result['retained_frames']}
 samples[segment['id']]=sample
def add_segment(parent,offset,requested,science):
 study=parent/'payload/study';case=study/'case-0'
 summary=json.loads((study/'summary.json').read_bytes());sub=json.loads((parent/'submission.json').read_bytes());job=sub['job_id']
 manifest=json.loads((science/'source_manifest.json').read_bytes());pin=binding(science/'source_manifest.json')['sha256']
 assert summary['source_manifest_sha256']==pin and summary['job_id']==job
 candidates=[science/name for name,digest in manifest['files'].items() if name.endswith('.json') and digest==summary['config_sha256']]
 assert len(candidates)==1,'Ambiguous scientific configuration identity'
 row={'id':parent.parent.name+'-'+parent.name,'global_offset':offset,'requested_updates':requested,
  'case_directory':str(case),'anchor_directory':str(case),'source_manifest_sha256':pin,'job_id':job,
  'scientific_status':summary['status'],'config':binding(candidates[0])}
 for key,path in [('summary',study/'summary.json'),('case_report',case/'report.json'),('states',case/'states.jsonl'),
  ('updates',case/'updates.jsonl'),('retained_index',case/'retained-frames.json'),('stdout',parent/('stdout.'+job+'.log')),('stderr',parent/('stderr.'+job+'.log'))]:row[key]=binding(path)
 lsf=''.join(Path(row['stdout']['path']).read_text().split());assert lsf.count('Subject:Job')==1 and 'Subject:Job'+job+':' in lsf
 assert ('Exitedwithexitcode1.' in lsf) if summary['status']=='failed' else ('Successfullycompleted.' in lsf)
 proofs.append(binding(parent/'submission.json'))
 if (parent/'payload/assessment.json').exists():proofs.append(binding(parent/'payload/assessment.json'))
 sources.append({'path':str(science),'manifest_sha256':pin,'files':len(manifest['files'])})
 counts(row);return row
for arm in plan['arms']:
 for segment in arm['segments']:counts(segment)
 if arm['id']=='boltz':
  arm['segments'].append(add_segment(base/'nonchem-endpoint-continuation-09/boltz-segment-10000',10000,10000,base/'boltz-lr0025-release-03/source'))
  arm['status_detail']='20,000 saved updates. The new terminal endpoint check failed after the second segment; continuation to30,000 is blocked. Original failures remain preserved.'
 if arm['id']=='dbmol':
  arm['segments'].append(add_segment(base/'nonchem-endpoint-continuation-08/dbmol-segment-05000',5000,2500,base/'boltz-dbmol-pilot-release-02/source'))
  arm['status_detail']='7,500 saved updates. Original terminal validation failed. Separate finer-scale qualification passed; new full-model continuation validation is queued. The original scientific failures remain failed.'
 if arm['id']=='protenix':
  arm['status_detail']='Absolute20,000 saved updates (10,000 shown since the saved10,000 parent). Original terminal validation failed. Separate FP64 endpoint qualification passed; full-model continuation validation is queued.'
for name,label,parent,target,rate,science,detail in [
 ('protenix_chemistry','Random Protenix + delayed chemistry',base/'chemistry-production-03/protenix/random-segment-0000',50000,.3,base/'protenix-extended-release-03/source','First10,000-update segment passed. The next boundary combined-gradient check and later separate revalidation failed; continuation is blocked. Chemistry rises from0.001 after5,000 toward100 at50,000.'),
 ('boltz_chemistry','Boltz-2 + delayed chemistry',base/'chemistry-production-04/boltz/boltz2_chemistry-segment-0000',30000,.0025,base/'boltz-lr0025-release-03/source','10,000 saved updates; terminal bond-gradient validation failed. Separate native-component recheck passed; combined-loss checks and pilots are still required before continuation. Chemistry rises from0.001 after5,000 toward100 at30,000.')]:
 segment=add_segment(parent,0,10000,science)
 plan['arms'].append({'id':name,'label':label,'kind':'fixed','rate':rate,'global_start':0,'global_target':target,'target_updates':target,
  'scientific_status':'failed' if segment['scientific_status']=='failed' else 'partial','segments':[segment],
  'metric_labels':{},'status_detail':detail})
for arm in plan['arms']:
 n=sum(s['expected_counts']['committed_updates'] for s in arm['segments']);frames=sum(s['expected_counts']['retained_frames'] for s in arm['segments'])-max(0,len(arm['segments'])-1)
 arm['expected_counts']={'committed_updates':n,'scalar_states':n+1,'retained_frames':frames,'latest_global_step':arm['global_start']+n}
 arm['expected_committed_updates']=n;arm['expected_retained_frames']=frames
 for seg in arm['segments']:
  for key in ('states','updates','summary','config','case_report','retained_index','stdout','stderr'):
   if key in seg:assert binding(seg[key]['path'])==seg[key];proofs.append(seg[key])
plan.update(edition='edition-0004',captured_at=datetime.now(timezone.utc).isoformat(),monitor_remains_paused=True,automatic_optimizer_submission=False)
sources={row['path']:row for row in sources};proofs={row['path']:row for row in proofs}
print(json.dumps({'plan':plan,'parent_sources':list(sources.values()),'proofs':list(proofs.values()),'schema_samples':samples,'mutated':False},allow_nan=False))
'''

def main():
    transport = REPO / 'experiments/validation/endpoint-qualification/scripts/manage_protenix_precision.py'
    spec = importlib.util.spec_from_file_location('_report_capture_transport', transport)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    LOCAL.mkdir(parents=True, exist_ok=True); helper.LOCAL = LOCAL
    result = helper.command(PROGRAM, 'capture01', 240)
    with (LOCAL/'fresh-plan.json').open('xb') as stream:stream.write(helper.raw(result['plan']))
    print(json.dumps({'arms':{a['id']:a['expected_counts'] for a in result['plan']['arms']},'mutated':False},indent=2))

if __name__ == '__main__': main()
