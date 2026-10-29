"""One compact read-only CPU/controller receipt collection for an exact capsule."""
import argparse
import importlib.util
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[4]
OWNER=REPO/'experiments/validation/endpoint-qualification'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--label',required=True);a=p.parse_args()
    assert a.tag.replace('-','').isalnum() and a.label.replace('-','').isalnum()
    local=REPO/'results/hpc'/a.tag;source=local/'source01';plan=json.loads((source/'plan.json').read_bytes())
    assert plan['root'].startswith('/work3/s234818/mosaic/runs/')
    code='ROOT='+repr(plan['root'])+'\nSOURCE='+repr(plan['source_root'])+'\n'+r'''
import json,subprocess,hashlib,base64
from pathlib import Path
from datetime import datetime,timezone
r=Path(ROOT);jobs=json.loads((r/'jobs.json').read_bytes());files={}
def keep(p):
 assert p.is_file() and not p.is_symlink() and p.stat().st_size<4000000
 b=p.read_bytes();files[str(p)]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()}
for p in (r/'cpu-certificate.json',r/'cpu/payload/assessment.json',r/'bootstrap-complete.json'):
 if p.exists():keep(p)
for p in (r/'cpu/payload').glob('tests*'):
 if p.is_file() and p.stat().st_size<4000000:keep(p)
if (r/'cpu-certificate.json').exists():
 for row in json.loads((r/'cpu-certificate.json').read_bytes())['proofs']:
  keep(Path(row['path']));assert all(files[row['path']][k]==v for k,v in row.items())
for name,jid in jobs.items():
 for p in (r/name/'submission.json',r/name/('stdout.'+jid+'.log'),r/name/('stderr.'+jid+'.log')):
  if p.exists() and p.stat().st_size<4000000:keep(p)
for p in r.glob('after-cpu/rolling-capacity-*/assessment.json'):keep(p)
for row in json.loads((Path(SOURCE)/'plan.json').read_bytes())['rolling_storage_policy']['known_mosaic_jobs']:
 if 'terminal_proof' in row:
  pin=row['terminal_proof'];keep(Path(pin['path']));assert all(files[pin['path']][k]==x for k,x in pin.items())
  for pin in json.loads(Path(pin['path']).read_bytes()).get('proofs',[]):
   keep(Path(pin['path']));assert all(files[pin['path']][k]==x for k,x in pin.items())
q=subprocess.run(['bjobs','-a','-json','-o','jobid stat queue job_name command',*jobs.values()],capture_output=True,text=True,timeout=40)
print(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'jobs':jobs,'files':list(files.values()),
 'scheduler':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'mutations':False}))
'''
    s=importlib.util.spec_from_file_location('_precision_collection_transport',OWNER/'scripts/manage_protenix_precision.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.LOCAL=local
    v=m.command(code,a.label,180);(local/a.label/'observation.json').write_bytes(m.raw(v))
    rows=json.loads(v['scheduler']['stdout']).get('RECORDS',[]) if v['scheduler']['stdout'] else []
    print(json.dumps({'observed_utc':v['observed_utc'],'jobs':v['jobs'],'states':{r['JOBID']:r['STAT'] for r in rows},'files':len(v['files'])}))


if __name__=='__main__':main()
