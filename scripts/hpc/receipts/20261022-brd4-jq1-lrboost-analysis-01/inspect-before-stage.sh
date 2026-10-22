set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,datetime
tag='20261022-brd4-jq1-lrboost-analysis-01'
root=Path('/work3/s234818/mosaic/runs')/tag
source=Path('/work3/s234818/mosaic/releases')/tag
upload=Path('/work3/s234818/mosaic')/(tag+'.tar.gz')
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'paths':{str(p):p.exists() for p in (root,source,upload)}}
for p in (root/'jobs.json',root/'initial-submissions.json',root/'analysis-dispatch-started.json'):
 if p.exists():record[str(p)]=json.loads(p.read_text())
job='29779179';r=subprocess.run(['bjobs','-l',job],capture_output=True,text=True,timeout=30)
record['chemistry_scheduler']={'returncode':r.returncode,'raw':r.stdout+r.stderr}
study=Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-fixed-lrboost-01/run-boltz2_chemistry/payload/study')
prefix={}
for name in ('states','updates'):
 path=study/'case-0'/(name+'.jsonl');rows=0;last=None
 if path.exists():
  with path.open('rb') as f:
   for line in f:
    if not line.endswith(b'\n'):break
    row=json.loads(line);rows+=1
    last={k:row.get(k) for k in ('step','index','repair_weight','selection_reason','affinity_loss')}
 prefix[name]={'rows':rows,'last':last}
record['chemistry_saved_prefix']=prefix
summary=study/'summary.json'
record['chemistry_summary_exists']=summary.exists()
if summary.exists():
 s=json.loads(summary.read_text());record['chemistry_terminal']={k:s.get(k) for k in ('status','error','job_id')}
print(json.dumps(record,indent=2))
PY
