set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'

from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,subprocess,re,tarfile
root=Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-lrboost-01')
source=Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source')
home=Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-lrboost-01')
dest=root/'delivery-metadata-01';assert not dest.exists();dest.mkdir()
files={};origins={};rehashes=[]
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def add_bytes(name,data,origin=None):
 path=dest/name;path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
 files[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 if origin:origins[name]=str(origin)
def add(path):
 path=Path(path);assert path.is_file() and not path.is_symlink()
 if path.is_relative_to(root):name='campaign/'+path.relative_to(root).as_posix()
 elif path.is_relative_to(home):name='home/'+path.relative_to(home).as_posix()
 elif path.is_relative_to(source):name='source/'+path.relative_to(source).as_posix()
 else:raise ValueError('Unexpected evidence source')
 if name not in files:add_bytes(name,path.read_bytes(),path)
def root_metadata(path):
 # Source writes pretty JSON with exactly two spaces for top-level keys.
 # Stream over the large saved cases, retaining metadata blocks only.
 keys={'status','error','identity_error','mode','job_id','config_sha256','source_manifest_sha256','model_digest','context_digest','pocket_definition_sha256','diffusion_gradient','native_hard_parity','native_gradient_checks','terminal_gradient_checks','optimization_parameters','baseline_initialization_match','initialization_sha256','frozen_identity_after','elapsed_seconds','requested_updates','acceptance_policy','nominal_step_size','counts'}
 found={};key=None;block=[]
 def finish():
  if key is not None:
   data=''.join(block).strip().removesuffix(',')
   found[key]=json.loads(data)
 with path.open(encoding='utf-8') as f:
  for line in f:
   match=re.match(r'^  "([^"\\]+)":\s*(.*)',line)
   if match:
    finish();key=match.group(1) if match.group(1) in keys else None;block=[match.group(2)+'\n'] if key else []
   elif line=='}\n' or line=='}':finish();key=None;block=[]
   elif key is not None:block.append(line)
 finish()
 return {'original_path':str(path),'original_bytes':path.stat().st_size,'original_sha256':sha(path),'metadata':found,'scope':'Saved JSON metadata only; original large case arrays not downloaded'}
def prefix(path):
 observed=path.stat().st_size;digest=hashlib.sha256();count=0;first=None;last=None;expected=0;consumed=0;contiguous=True
 with path.open('rb') as f:
  while consumed<observed:
   line=f.readline(observed-consumed);consumed+=len(line);digest.update(line)
   if not line.endswith(b'\n'):break
   row=json.loads(line)
   if count==0:first=row
   last=row;contiguous=contiguous and row.get('step')==expected;expected+=1;count+=1
 return {'path':str(path),'observed_bytes':observed,'captured_prefix_bytes':consumed,'captured_prefix_sha256':digest.hexdigest(),'committed_rows':count,'steps_contiguous_from_zero':contiguous,'first':first,'last':last,'scope':'Newline committed saved JSON metadata prefix, not a native/model pass'}
for name in ('source_manifest.json',):add(source/name)
for name in ('cpu-certificate.json','native-submissions.json','long-submissions.json','long-dispatch-started.json'):
 add(root/name)
for arm in ('protenix','boltz2','boltz2_chemistry'):
 add(source/('examples/configs/brd4_jq1_25_fixed_lrboost_'+arm+'_10000.json'))
 decision=root/('decision-'+arm+'.json');add(decision);d=json.loads(decision.read_text())
 for proof in d['proofs']:
  p=Path(proof['path']);assert sha(p)==proof['sha256'];add(p)
 ready=root/('validate-'+arm)/'payload/study/assessment.json';assessment=json.loads(ready.read_text());add(ready)
 for proof in assessment['proofs']:
  p=Path(proof['path']);actual=sha(p);assert actual==proof['sha256']
  rehashes.append({'path':str(p),'bytes':p.stat().st_size,'sha256':actual})
  if p.suffix=='.json' and p.stat().st_size<2*1024**2:add(p)
 add(ready.with_name('summary.json'))
 for role in ('validate-','after-ready-'):
  parent=root/(role+arm)
  for p in parent.glob('*.json'):add(p)
  for p in parent.glob('stdout.*.log'):add(p)
  for p in parent.glob('stderr.*.log'):add(p)
summary={'capture_utc':datetime.now(timezone.utc).isoformat(),'arms':{},'readiness_all_proof_byte_rehashes':rehashes}
longs=json.loads((root/'long-submissions.json').read_text())
for row in longs['jobs']:
 arm=row['arm'];job=row['job_id'];parent=Path(row['output_parent']);study=parent/'payload/study';entry={'job_id':job,'learning_rate':row['learning_rate'],'parent':str(parent)}
 for p in parent.glob('*.json'):add(p)
 for p in parent.glob('stdout.*.log'):add(p)
 for p in parent.glob('stderr.*.log'):add(p)
 if (parent/'payload/driver-invocation.json').exists():add(parent/'payload/driver-invocation.json')
 native=parent/'payload/native.log'
 if native.exists():
  size=native.stat().st_size
  with native.open('rb') as f:f.seek(max(0,size-16000));tail=f.read(size-min(size,max(0,size-16000)))
  add_bytes('native-tail-'+arm+'.txt',tail)
  entry['native_log']={'path':str(native),'bytes_observed':size,'tail_only':True}
 for filename in ('summary.json','case-0/report.json'):
  p=study/filename
  if p.exists():entry[filename]=root_metadata(p)
 for filename in ('states.jsonl','updates.jsonl'):
  p=study/'case-0'/filename
  if p.exists():entry[filename]=prefix(p)
 for directory in ('native_gradient_checks','case-0/combined_initial_checks','terminal_gradient_checks'):
  for p in (study/directory).glob('*.json'):
   if p.stat().st_size<2*1024**2:add(p)
 gate=root/('entry-quota-'+arm)
 for p in gate.rglob('*.json'):add(p)
 hist=subprocess.run(['bhist','-l',job],capture_output=True,text=True,timeout=40)
 add_bytes('history-'+arm+'.txt',(hist.stdout+hist.stderr).encode());entry['history_returncode']=hist.returncode
 summary['arms'][arm]=entry
for p in root.glob('quota-dispatch-*/*.json'):add(p)
add_bytes('metadata.json',(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode())
add_bytes('origin-map.json',(json.dumps(origins,indent=2)+'\n').encode())
(dest/'evidence_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
archive=root/'delivery-metadata-01.tar.gz'
with archive.open('xb') as output:
 with tarfile.open(fileobj=output,mode='w:gz') as tar:
  for p in sorted(dest.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(dest).as_posix(),recursive=False)
print(json.dumps({'path':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),'files':len(files),'capture_utc':summary['capture_utc']}))

PY
