"""Retire only the authenticated unstarted, superseded29796558 report request."""
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'
LOCAL=REPO/'results/hpc/20261029-shared-writer-repair-23'


def main():
    code=r'''
import json,hashlib,subprocess,time,os
from pathlib import Path
from datetime import datetime,timezone
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
root=base/'lrreset-analysis-03/combined-report';audit=base/'obsolete-report-retirement-25'
assert not audit.exists(),'Existing/unknown retirement; inspect, never repeat'
def write(p,v):
 with p.open('x') as f:json.dump(v,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def binding(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def query():
 argv=['bjobs','-json','-o','jobid stat queue start_time command','29796558'];r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
 assert r.returncode==0,r.stderr
 v={'argv':argv,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};rows=json.loads(r.stdout)['RECORDS'];assert len(rows)==1
 row=rows[0];assert row['JOBID']=='29796558' and row['QUEUE']=='hpc' and row['START_TIME']==''
 return v,row
sub=json.loads((root/'submission.json').read_bytes());assert sub['job_id']=='29796558' and sub['outcome']=='submitted' and sub['returncode']==0
before,row=query();assert row['STAT']=='PEND' and row['COMMAND']==' '.join(sub['args'][sub['args'].index('bash'):])
detail=subprocess.run(['bjobs','-l','29796558'],capture_output=True,text=True,timeout=40)
assert detail.returncode==0 and 'Dependency condition invalid or never satisfied' in detail.stdout
assert not (root/'payload').exists() and not list(root.glob('stdout.*.log')) and not list(root.glob('stderr.*.log'))
original={str(p):binding(p) for p in root.iterdir() if p.is_file()}
audit.mkdir();write(audit/'before.json',before);write(audit/'detail-before.json',{'returncode':detail.returncode,'stdout':detail.stdout,'stderr':detail.stderr})
write(audit/'intent.json',{'job_id':'29796558','reason':'superseded_unstarted_report_with_unsatisfiable_dependencies',
 'user_authorization':'fix remaining blocked continuation launch/resource issues; preserve saved results','original_files':original})
r=subprocess.run(['bkill','29796558'],capture_output=True,text=True,timeout=40)
write(audit/'bkill.json',{'argv':['bkill','29796558'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});assert r.returncode==0
for i in range(16):
 after,row_after=query();write(audit/('after-'+str(i).zfill(2)+'.json'),after)
 assert row_after['COMMAND']==row['COMMAND']
 if row_after['STAT']=='EXIT':break
 assert row_after['STAT']=='PEND' and i<15;time.sleep(1)
assert not (root/'payload').exists()
assert all(binding(Path(p))==v for p,v in original.items())
proofs=[binding(audit/'before.json'),binding(audit/'detail-before.json'),binding(audit/'bkill.json'),binding(audit/('after-'+str(i).zfill(2)+'.json')),binding(root/'submission.json')]
value={'status':'terminal_job_no_unspent_future_execution_retained_files_preserved','job_id':'29796558','output_root':str(root),
 'source_manifest_sha256':'8e3045ddfd44a701127deb7af7a163536dfbf699123ba883f98207d6d57b95d5','files_deleted_or_modified':False,
 'never_started':True,'proofs':proofs,'before_proof':proofs[0],'after_proof':proofs[3],'submission':proofs[4],
 'utc':datetime.now(timezone.utc).isoformat()}
write(audit/'terminal-proof.json',value)
print(json.dumps({'status':'obsolete_unstarted_report_actually_EXIT_retained_all_files','binding':binding(audit/'terminal-proof.json'),'value':value}))
'''
    spec=importlib.util.spec_from_file_location('_report_retirement_transport',OWNER/'scripts/manage_protenix_precision.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.LOCAL=LOCAL
    v=m.command(code,'obsolete-report-retire13',120)
    (LOCAL/'obsolete-report-retire13/observation.json').write_bytes(m.raw(v))
    print(json.dumps({k:v[k] for k in ('status','binding')}))


if __name__=='__main__':main()
