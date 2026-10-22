set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import json,subprocess,os,stat
from datetime import datetime,timezone
paths=[
 '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study/case-0',
 '/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study/case-0',
 '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-protenix-02/run-protenix/payload/study/case-0',
 '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/run-boltz2/payload/study/case-0',
 '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/run-boltz2_chemistry/payload/study/case-0']
record={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only previous-rate artifact inventory; no archive, deletion or storage-gate pass','candidates':{}}
for raw in paths:
 p=Path(raw)
 item={'exists':p.exists(),'safe_identity':p.resolve()==p and not p.is_symlink()}
 if item['exists'] and item['safe_identity']:
  count=0;size=0;bad=[]
  for root,dirs,files in os.walk(p,followlinks=False):
   for name in dirs+files:
    q=Path(root)/name;s=q.lstat()
    if stat.S_ISLNK(s.st_mode) or not (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)):bad.append(str(q))
    if stat.S_ISREG(s.st_mode):count+=1;size+=s.st_size
  item.update(files=count,logical_bytes=size,unsafe_entries=bad)
 record['candidates'][raw]=item
print(json.dumps(record,indent=2))
PY
