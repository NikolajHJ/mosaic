set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json
ps=subprocess.run(['ps','-u','s234818','-o','pid,ppid,lstart,args'],capture_output=True,text=True,timeout=15)
processes=[s for s in ps.stdout.splitlines() if 'python -' in s or 'python3 -' in s or 'bash -l -s' in s]
roots=[Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-03/run-protenix/payload/study'),Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-storage-04/run-boltz2/payload/study'),Path('/work3/s234818/mosaic/runs/20261021-brd4-jq1-25-fixed-report-04/interim-0001/payload')]
record={'utc':datetime.now(timezone.utc).isoformat(),'scope':'Read-only metadata options; no deletion/archive/submission','readonly_ssh_python_processes':processes,'targets':{}}
for p in roots:
 v={'exists':p.exists(),'resolved_identity_matches':p.resolve()==p and not p.is_symlink()}
 if v['exists'] and v['resolved_identity_matches']:
  r=subprocess.run(['timeout','45','du','-s','-b','--',str(p)],capture_output=True,text=True,timeout=50)
  v.update(returncode=r.returncode,raw=r.stdout+r.stderr)
 record['targets'][str(p)]=v
print(json.dumps(record,indent=2))
PY
