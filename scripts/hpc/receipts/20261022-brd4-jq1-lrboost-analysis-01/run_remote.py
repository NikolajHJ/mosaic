"""Single SSH connection attempt with complete byte-preserved local receipts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

folder = Path(__file__).resolve().parent
script = folder / sys.argv[1]
label = sys.argv[2]
timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 180
command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc', 'bash -l -s']
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
          'script': script.name, 'script_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'outcome': 'unknown'}
with (folder / (label + '-started.json')).open('x', encoding='utf-8') as f: json.dump(record, f, indent=2)
try:
    r = subprocess.run(command, input=script.read_bytes(), capture_output=True, timeout=timeout)
    (folder / (label + '-stdout.txt')).write_bytes(r.stdout); (folder / (label + '-stderr.txt')).write_bytes(r.stderr)
    record.update(returncode=r.returncode, outcome='completed' if r.returncode == 0 else 'failed')
    print(r.stdout.decode('utf-8', errors='replace')); print(r.stderr.decode('utf-8', errors='replace'), file=sys.stderr)
except subprocess.TimeoutExpired as error:
    (folder / (label + '-stdout.txt')).write_bytes(error.stdout or b''); (folder / (label + '-stderr.txt')).write_bytes(error.stderr or b'')
    record.update(outcome='timeout_unknown', error=repr(error))
finally:
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (folder / (label + '-result.json')).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
if record['outcome'] != 'completed': raise SystemExit(1)
