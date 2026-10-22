set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess
root=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02')
control=root/'cleanup-recovery-02'
record={'utc':datetime.now(timezone.utc).isoformat(),'artifacts':{},'journal':{}}
for name in ('backup-receipt.json','attempt-01/receipt.json','attempt-01/quota-after.txt','attempt-01/jobs-before-verification.json','attempt-01/jobs-before-removal.json'):
 p=control/name
 if p.is_file():
  b=p.read_bytes();record['artifacts'][name]={'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content':json.loads(b) if p.suffix=='.json' else b.decode()}
p=control/'attempt-01/deletion-journal.jsonl'
if p.is_file():
 counts={};total=0;last=None
 with p.open('rb') as stream:
  for line in stream:
   if not line.endswith(b'\n'):break
   row=json.loads(line);counts[row['event']]=counts.get(row['event'],0)+1
   if row['event']=='removed':total+=row['bytes']
   last=row
 record['journal']={'newline_committed_events':counts,'removed_bytes_so_far':total,'last':last}
r=subprocess.run(['getquota_work3.sh'],capture_output=True,text=True,timeout=60)
record['raw_quota']={'returncode':r.returncode,'raw':r.stdout+r.stderr,'scope':'Observation only; a fresh original full capacity gate remains mandatory'}
print(json.dumps(record,indent=2))
PY
