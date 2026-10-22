set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,re,subprocess,hashlib,tarfile
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
source=Path('/work3/s234818/mosaic/releases/20261022-brd4-jq1-lrboost-analysis-01/source')
manifest='214c19387649a741a7ad42f1e3e027061631993b4ae33425071aba9005fdcbcc'
jobs=json.loads((root/'jobs.json').read_text());job=jobs['active_interim'];parent=root/jobs['interim_parent']
def digest(b):return hashlib.sha256(b).hexdigest()
result=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30);raw=result.stdout+result.stderr;c=re.sub(r'\s+','',raw)
assert result.returncode==0 and re.findall(r'Job<([^>]+)>',c)==[job] and re.findall(r'Status<([^>]+)>',c)==['DONE'] and re.findall(r'Queue<([^>]+)>',c)==['hpc'], 'Report has not actually finished successfully'
stdout=parent/('stdout.'+job+'.log');log=stdout.read_text()
assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]* Done\s*$',log,re.M)==[job] and 'Successfully completed.' in log and 'Exited with exit' not in log
assessment=json.loads((parent/'payload/assessment.json').read_text());receipt=json.loads((parent/'payload/report/report_receipt.json').read_text())
assert assessment['status']=='passed' and assessment['job_id']==job and assessment['report_source_manifest_sha256']==manifest
assert assessment['scientific_passed'] is False and assessment['campaign_status']=='failed'
assert receipt['input_jobs']==['29779177','29779178','29779179'] and receipt['states'][:2]==[10001,10001] and receipt['included_updates'][:2]==[10000,10000]
assert assessment['report_receipt_sha256']==digest((parent/'payload/report/report_receipt.json').read_bytes())
archive_receipt=Path(str(parent)+'.tar.receipt.json');archive=json.loads(archive_receipt.read_text());q=Path(archive['path'])
assert q==Path(str(parent)+'.tar.gz') and q.stat().st_size==archive['bytes'] and digest(q.read_bytes())==archive['sha256']
dest=root/'interim-delivery-evidence-01';dest.mkdir(exist_ok=False);files={};origin={}
def put(name,data,p=None):
 assert name not in files
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as out:out.write(data)
 files[name]={'bytes':len(data),'sha256':digest(data)}
 if p:origin[name]=str(p)
def add(p):
 p=Path(p);assert p.is_file() and not p.is_symlink()
 name=('analysis/'+p.relative_to(root).as_posix()) if p.is_relative_to(root) else ('source/'+p.relative_to(source).as_posix())
 if name not in files:put(name,p.read_bytes(),p)
put('report-final-scheduler.txt',raw.encode())
for p in (stdout,parent/'submission.json',parent/'submission-started.json',parent/'scheduler-at-submission.txt',parent/'payload/assessment.json',parent/'payload/report/report_receipt.json',archive_receipt,root/'jobs.json',root/'cpu-certificate.json',root/'after-cpu/storage-dispatch/gate.json',source/'source_manifest.json'):
 add(p)
for proof in json.loads((root/'cpu-certificate.json').read_text())['proofs']:
 p=Path(proof['path']);assert digest(p.read_bytes())==proof['sha256'];add(p)
for arm in ('protenix','boltz2'):
 collector=root/('complete-'+arm);j=jobs['collectors'][arm]
 r=subprocess.run(['bjobs','-l',j],capture_output=True,text=True,timeout=30);text=r.stdout+r.stderr;compact=re.sub(r'\s+','',text)
 assert r.returncode==0 and re.findall(r'Job<([^>]+)>',compact)==[j] and re.findall(r'Status<([^>]+)>',compact)==['DONE']
 put('collector-'+arm+'-scheduler.txt',text.encode())
 for name in ('submission.json','stdout.'+j+'.log','payload/assessment.json','payload/snapshots.json','payload/native-stdout.txt','payload/native-scheduler-0.txt'):
  add(collector/name)
 for proof in json.loads((collector/'payload/assessment.json').read_text())['proofs']:
  p=Path(proof['path']);assert digest(p.read_bytes())==proof['sha256']
put('origin-map.json',(json.dumps(origin,indent=2)+'\n').encode())
(dest/'evidence_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
bundle=root/'interim-delivery-evidence-01.tar.gz'
with bundle.open('xb') as out:
 with tarfile.open(fileobj=out,mode='w:gz') as tar:
  for p in sorted(dest.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(dest).as_posix(),recursive=False)
record={'report_archive':archive,'delivery_evidence':{'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest(bundle.read_bytes()),'files':len(files)}}
(root/'interim-delivery-evidence-01.receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
