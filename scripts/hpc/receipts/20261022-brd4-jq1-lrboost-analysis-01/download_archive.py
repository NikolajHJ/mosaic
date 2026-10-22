"""One byte transfer to a fresh local file, then safe stdlib verification."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

folder = Path(__file__).resolve().parent; repo = folder.parents[3]
receipt = json.loads((folder / sys.argv[1]).read_text(encoding='utf-8')); label = sys.argv[2]
assert re.fullmatch(r'[a-z0-9-]+', label)
remote = receipt['path']
assert remote.startswith('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/') and remote.endswith('.tar.gz')
out = repo / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01' / (label + '.tar.gz')
out.parent.mkdir(parents=True, exist_ok=True); assert not out.exists()
command = ['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc:' + remote, str(out)]
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command, 'receipt': receipt, 'outcome': 'unknown'}
with (folder / (label + '-download-started.json')).open('x') as f: json.dump(record, f, indent=2)
try:
    r = subprocess.run(command, capture_output=True, timeout=180)
    (folder / (label + '-download-stdout.txt')).write_bytes(r.stdout); (folder / (label + '-download-stderr.txt')).write_bytes(r.stderr)
    record['returncode'] = r.returncode
    assert r.returncode == 0 and out.stat().st_size == receipt['bytes'] and hashlib.sha256(out.read_bytes()).hexdigest() == receipt['sha256']
    record['outcome'] = 'byte_verified'
    subprocess.run([sys.executable, str(repo / 'scripts/verify_hpc_evidence.py'), str(out), receipt['sha256']], check=True)
    record['outcome'] = 'safe_extraction_and_all_files_verified'
finally:
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (folder / (label + '-download-result.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
