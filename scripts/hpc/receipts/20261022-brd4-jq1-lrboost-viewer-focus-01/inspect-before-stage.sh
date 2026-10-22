set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,datetime
base=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
paths=[base/'viewer-focus-01',base/'viewer-focus-release-01',base/'20261022-brd4-jq1-lrboost-viewer-focus-01.tar.gz']
observed=[{'path':str(p),'exists':p.exists(),'is_symlink':p.is_symlink(),'resolved':str(p.resolve()),'children':[q.name for q in p.iterdir()] if p.is_dir() else []} for p in paths]
r=subprocess.run(['bjobs','-l','29780821','29780822','29780823','29780824','29780830'],capture_output=True,text=True,timeout=30)
q=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=60)
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_new_paths':observed,'actual_current_analysis_jobs':{'returncode':r.returncode,'raw':r.stdout+r.stderr},'raw_work3_quota':{'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr},'analysis_jobs':json.loads((base/'jobs.json').read_text())},indent=2))
PY