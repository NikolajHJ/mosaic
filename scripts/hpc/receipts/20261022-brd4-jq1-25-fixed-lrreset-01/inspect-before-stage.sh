set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess
analysis=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
tag='20261022-brd4-jq1-25-fixed-lrreset-01'
paths=[analysis/'restart-lrreset-01',analysis/'restart-lrreset-release-01',analysis/(tag+'.tar.gz'),Path('/zhome/95/e/206121/mosaic-runs')/tag]
jobs=json.loads((analysis/'jobs.json').read_text())
ids=[*jobs['diagnostics'].values(),*jobs['diagnostic_captures'].values(),jobs['final'],'29779177','29779178','29779179']
r=subprocess.run(['bjobs','-l',*ids],capture_output=True,text=True,timeout=30)
record={'utc':datetime.now(timezone.utc).isoformat(),'new_paths':{str(p):p.exists() or p.is_symlink() for p in paths},'analysis_jobs':jobs,'scheduler':{'returncode':r.returncode,'raw':r.stdout+r.stderr},'collectors':{},'raw_quota':{}}
for arm in ('protenix','boltz2','boltz2_chemistry'):
 p=analysis/('complete-'+arm)/'payload/assessment.json'
 v=json.loads(p.read_text());record['collectors'][arm]={'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'native_job':v['native_job'],'collector_job':v['collector_job'],'scientific_status':v['scientific_status'],'all_terminal':v['outcome']['all_terminal'],'states':v['states'],'updates':v['updates']}
for pool,command in [('work3','getquota_work3.sh'),('zhome','getquota_zhome.sh')]:
 q=subprocess.run([command],capture_output=True,text=True,timeout=60);record['raw_quota'][pool]={'returncode':q.returncode,'raw':q.stdout+q.stderr,'limitation':'Read-only snapshot, no full storage gate or completed recent home inventory'}
print(json.dumps(record,indent=2))
if any(record['new_paths'].values()):raise SystemExit('New path already exists; inspect before any mutation')
PY
