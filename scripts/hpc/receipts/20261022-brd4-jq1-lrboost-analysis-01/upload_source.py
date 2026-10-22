"""Exclusive byte transfer of the newly sealed tooling archive, one attempt."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

folder = Path(__file__).resolve().parent
repo = folder.parents[3]
record = json.loads((folder / 'local-static-verification.json').read_text())
archive = repo / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/source01/source.tar.gz'
data = archive.read_bytes()
assert len(data) == record['archive_bytes'] and hashlib.sha256(data).hexdigest() == record['archive_sha256']
remote = '/work3/s234818/mosaic/20261022-brd4-jq1-lrboost-analysis-01.tar.gz'
code = ('import pathlib,sys,hashlib,json; p=pathlib.Path(' + repr(remote) + '); '
        'f=p.open("xb"); data=sys.stdin.buffer.read(); f.write(data); f.close(); '
        'assert len(data)==' + str(len(data)) + '; assert hashlib.sha256(data).hexdigest()==' + repr(record['archive_sha256']) + '; '
        'print(json.dumps({"path":str(p),"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}))')
command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1', 'dtu-hpc',
           '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -c ' + shlex.quote(code)]
attempt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'outcome': 'unknown', 'command': command,
           'archive_sha256': record['archive_sha256'], 'archive_bytes': len(data)}
with (folder / 'upload-01-started.json').open('x') as f: json.dump(attempt, f, indent=2)
try:
    r = subprocess.run(command, input=data, capture_output=True, timeout=90)
    (folder / 'upload-01-stdout.txt').write_bytes(r.stdout); (folder / 'upload-01-stderr.txt').write_bytes(r.stderr)
    attempt.update(returncode=r.returncode, outcome='completed' if r.returncode == 0 else 'failed')
    print(r.stdout.decode(errors='replace'))
finally:
    (folder / 'upload-01-result.json').write_text(json.dumps(attempt, indent=2) + '\n')
assert attempt['outcome'] == 'completed', 'Inspect remote exact upload before any recovery; unknown never duplicated'
