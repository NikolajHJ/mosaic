set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,datetime,subprocess
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
parent=root/'interim-0001';record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'render_files':{}}
for arm in ('protenix','boltz2','boltz2_chemistry'):
 p=parent/'payload/report'/arm;bins=sorted(p.glob('frames-*.bin'))
 record['render_files'][arm]={'binary_chunks':len(bins),'latest_binary_chunk':bins[-1].name if bins else None,'curves_exists':(p/'curves.json').exists()}
for rel in ('stderr.29780829.log','payload/render.log'):
 p=parent/rel
 if p.exists():record[rel]=p.read_text(errors='replace')[-8000:]
record['archive_exists']=Path(str(parent)+'.tar.gz').exists()
record['assessment_exists']=(parent/'payload/assessment.json').exists()
print(json.dumps(record,indent=2))
PY
