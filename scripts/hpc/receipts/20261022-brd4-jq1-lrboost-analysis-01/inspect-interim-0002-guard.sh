set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json
r=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
items=[]
for p in sorted(r.glob('interim-*')):
 d={'name':p.name,'is_dir':p.is_dir(),'is_symlink':p.is_symlink()}
 if p.is_dir():
  d['children']=[x.name for x in sorted(p.iterdir())]
  for name in ('submission-started.json','submission.json'):
   if (p/name).exists():d[name]=json.loads((p/name).read_text())
 items.append(d)
print(json.dumps({'items':items,'jobs':json.loads((r/'jobs.json').read_text()),'guard_exists':(r/'interim-0002-dispatch-guard.json').exists(),'jobs_tmp_exists':(r/'jobs.json.tmp').exists()},indent=2))
PY