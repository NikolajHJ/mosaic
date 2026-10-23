"""One readonly SSH input-identity snapshot; no array/model calculations."""
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
REMOTE=r'''
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
b=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01');r=b/'restart-lrreset-04'
source=b/'restart-lrreset-release-04/source';launch=b/'restart-lrreset-allwork3-launch-02/source'
def proof(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
out={'utc':datetime.now(timezone.utc).isoformat(),'scientific_source':proof(source/'source_manifest.json'),'launcher_source':proof(launch/'source_manifest.json'),'dispatch':proof(r/'long-dispatch-started.json'),'submissions':proof(r/'long-submissions.json'),'cpu':proof(r/'cpu-certificate.json'),'launcher_cpu':proof(r/'all-work3-launch-02/cpu-certificate.json'),'arms':{}}
jobs=json.loads((r/'long-submissions.json').read_bytes())['jobs']
for sub in jobs:
 a=sub['arm'];p=r/('run-'+a);s=p/'payload/study';pin={'job':sub['job_id'],'rate':sub['learning_rate'],'config':proof(source/('examples/configs/brd4_jq1_25_fixed_lrreset_'+a+'_10000.json')),'ready':proof(r/('validate-'+a)/'payload/study/assessment.json'),'decision':proof(r/('decision-'+a+'.json')),'submission':proof(p/'submission.json'),'output_parent':str(p),'projections':json.loads((r/('decision-'+a+'.json')).read_bytes())['projected_bytes']}
 for key,name in [('driver','payload/driver-invocation.json'),('entry','entry-quota/lrreset-storage-gate.json'),('summary','payload/study/summary.json'),('case','payload/study/case-0/report.json'),('raw_entry','payload/study/native_gradient_checks/report.json'),('coupled_entry','payload/study/case-0/combined_initial_checks/report.json'),('terminal','payload/study/terminal_gradient_checks/report.json'),('stdout','stdout.'+sub['job_id']+'.log')]:
  if (p/name).exists():pin[key]=proof(p/name)
 if (s/'summary.json').exists():
  j=json.loads((s/'summary.json').read_bytes());pin.update(scientific_status=j['status'],model=j['model_digest'],context=j['context_digest'],error=j.get('error'))
 out['arms'][a]=pin
for p in [b/'lrreset-analysis-01',b/'lrreset-analysis-release-01',b/'lrreset-analysis-01.tar.gz']:out[str(p)]=p.exists()
out['scheduler']=subprocess.run(['bjobs','-o','jobid stat queue job_name',*[x['job_id'] for x in jobs]],capture_output=True,text=True).stdout
print('PINS_BEGIN');print(json.dumps(out))
'''


def main():
    output=HERE/'input-pins.json'
    assert not output.exists(),'Inspect existing pins rather than replacing a sealed observation'
    x=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','dtu-hpc',"bash -l -c 'python3 -'"],input=REMOTE.encode(),capture_output=True,timeout=120)
    root=REPO/'results/hpc/20261023-brd4-jq1-lrreset-analysis-01';root.mkdir(parents=True,exist_ok=True)
    (root/'pins-stdout.bin').write_bytes(x.stdout);(root/'pins-stderr.bin').write_bytes(x.stderr)
    assert x.returncode==0,x.stderr.decode()[-1000:]
    value=json.loads(x.stdout.split(b'PINS_BEGIN\n')[1]);output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))


if __name__=='__main__':main()
