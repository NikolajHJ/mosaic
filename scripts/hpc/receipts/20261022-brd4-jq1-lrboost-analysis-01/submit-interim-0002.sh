set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import subprocess,sys,json,re,fcntl,hashlib,datetime
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01/source')
science=Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source')
manifest='214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
python='/work3/s234818/mosaic/envs/protenix-tiny/bin/python'
for base,pin in ((source,manifest),(science,'9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323')):
 subprocess.run([python,str(source/'scripts/verify_stage0_release.py'),str(base),pin],check=True)
sys.path[:0]=[str(source/'scripts'),str(science/'scripts'),str(science/'src')]
import jq1_lrboost_report_ops as ops
import jq1_lrboost_report_contract as contract
assert ops.ROOT==root and ops.SOURCE==source
with (root/'dispatch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 jobs=ops.read(root/'jobs.json')
 assert jobs['report_manifest']==manifest and jobs['native_jobs']=={a:p['job'] for a,p in contract.ARMS.items()}
 assert jobs['active_interim']=='29780829' and jobs['interim_parent']=='interim-0001' and jobs['final']=='29780830'
 assert jobs['diagnostics']=={'protenix':'29780821','boltz2':'29780823'} and jobs['diagnostic_captures']=={'protenix':'29780822','boltz2':'29780824'}
 assert sorted(p.name for p in root.glob('interim-*') if p.is_dir())==['interim-0001'], 'Other interim/unknown parent exists'
 assert not (root/'jobs.json.tmp').exists() and not (root/'interim-0002-dispatch-guard.json').exists()
 cpu=ops.verify_cpu(manifest)
 assert ops.sha(root/'cpu-certificate.json')=='42b2b000ecce5d661da9b75ad85f03a41c3b303010ccfa03f87835a25993ca69'
 inputs={a:contract.authenticate_inputs(a) for a in contract.ARMS}
 observed={}
 for key,job,queue in [('prior_interim',jobs['active_interim'],'hpc'),('final',jobs['final'],'hpc'),('chemistry','29779179','gpul40s')]:
  r=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=r.stdout+r.stderr
  assert r.returncode==0
  compact=re.sub(r'\s+','',raw)
  assert re.findall(r'Job<([^>]+)>',compact)==[job] and re.findall(r'Queue<([^>]+)>',compact)==[queue]
  states=re.findall(r'Status<([^>]+)>',compact);assert len(states)==1
  observed[key]={'job':job,'status':states[0],'raw':raw}
 assert observed['prior_interim']['status']=='DONE' and observed['final']['status']=='PEND'
 assert observed['chemistry']['status'] in ('RUN','DONE','EXIT')
 ops.actual_lsf((root/'interim-0001/stdout.29780829.log').read_text(),'29780829')
 prior=ops.read(root/'interim-0001/payload/report/report_receipt.json')
 assert prior['included_updates']==[10000,10000,9600]
 assert prior['milestone_progress']=={'protenix':10000,'boltz2':10000,'boltz2_chemistry':9500}
 assert ops.sha(root/'interim-0001/payload/report/report_receipt.json')=='e0d07ee7618aacd9dfafa0c374b76530b04c727b2a2b552e2b356468b50e0da7'
 study=Path(contract.ARMS['boltz2_chemistry']['output_parent'])/'payload/study/case-0'
 prefixes={}
 for field,expected,last in [('states',10001,10000),('updates',10000,9999)]:
  p=study/(field+'.jsonl');n=0;tail=0
  with p.open('rb') as stream:
   for row in stream:
    if not row.endswith(b'\n'):tail=len(row);break
    value=json.loads(row);assert value['step']==n;n+=1
  assert n==expected and value['step']==last and tail==0
  prefixes[field]={'rows':n,'last_step':last,'path':str(p),'bytes':p.stat().st_size,'sha256':ops.sha(p)}
 guard={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'name':'interim-0002','source_manifest_sha256':manifest,'cpu_certificate_sha256':ops.sha(root/'cpu-certificate.json'),'published_prior_report_receipt_sha256':ops.sha(root/'interim-0001/payload/report/report_receipt.json'),'published_milestones':prior['milestone_progress'],'new_chemistry_milestone':10000,'chemistry_terminal_validation':'pending unless separate authenticated terminal evidence establishes outcome','prefixes':prefixes,'observed':observed,'authenticated_inputs':inputs,'scope':'Metadata-only new saved report submission; original scientific runs untouched; full allocated entry storage/numerical gates remain mandatory'}
 ops.exclusive(root/'interim-0002-dispatch-guard.json',guard)
 sub=ops.submit('interim-0002','interim',manifest)
 jobs['active_interim']=sub['job_id'];jobs['interim_parent']='interim-0002';ops.atomic(root/'jobs.json',jobs)
 print(json.dumps({'status':'actual_submitted','submission':sub,'jobs':jobs,'guard_sha256':ops.sha(root/'interim-0002-dispatch-guard.json')}))
PY