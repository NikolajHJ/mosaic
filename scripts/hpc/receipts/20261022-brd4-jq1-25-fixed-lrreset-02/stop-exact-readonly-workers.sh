set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import os,signal,subprocess,json,time
targets=[(1806752,'Thu Oct 22 11:57:50 2026','obsolete read-only metadata inventory'),(1878607,'Thu Oct 22 12:07:12 2026','slow read-only byte backup attempt01')]
record={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Terminate only two explicitly identified own read-only helper workers; no model/diagnostic/scientific job affected','workers':[]}
for pid,expected,role in targets:
 p=Path('/proc')/str(pid)
 if not p.exists():record['workers'].append({'pid':pid,'role':role,'status':'already_absent'});continue
 if p.stat().st_uid!=os.getuid() or (p/'cmdline').read_bytes().split(b'\0')[:2]!=[b'/work3/s234818/mosaic/envs/protenix-tiny/bin/python',b'-']:raise ValueError('Worker identity changed')
 start=subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
 if start!=expected:raise ValueError('Worker PID was reused')
 if pid==1878607 and not any(os.readlink(q)=='/work3/s234818/mosaic/20261021-brd4-jq1-25-fixed-02.tar.gz' for q in (p/'fd').iterdir()):raise ValueError('Backup read descriptor identity changed')
 os.kill(pid,signal.SIGTERM)
 for _ in range(50):
  if not p.exists() or (p/'stat').read_text().split()[2]=='Z':break
  time.sleep(.1)
 else:raise ValueError('Read-only worker did not stop; no duplicate allowed')
 record['workers'].append({'pid':pid,'role':role,'start':start,'status':'stopped_SIGTERM'})
out=Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-02/readonly-worker-stop.json')
with out.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps(record,indent=2))
PY
