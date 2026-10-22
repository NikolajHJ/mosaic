set -euo pipefail
/work3/s234818/mosaic/envs/protenix-tiny/bin/python - <<'PY'
from pathlib import Path
import os,json,subprocess
rows=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  command=(p/'cmdline').read_bytes().split(b'\0')
  if command[:2]!=[b'/work3/s234818/mosaic/envs/protenix-tiny/bin/python',b'-']:continue
  meta=subprocess.run(['ps','-p',p.name,'-o','pid,ppid,lstart,etime,time,pcpu,args'],capture_output=True,text=True,timeout=10)
  fds=[]
  for fd in (p/'fd').iterdir():
   try:fds.append(os.readlink(fd))
   except OSError:pass
  rows.append({'pid':p.name,'metadata':meta.stdout,'open_paths':fds})
 except (OSError,PermissionError):pass
print(json.dumps({'scope':'Read-only identification of exact stdin metadata/backup workers; no process killed','processes':rows},indent=2))
PY
