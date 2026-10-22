"""Resume one confirmed stopped transfer, authenticating prefix and remote bytes.

This avoids SFTP per-request latency. It never changes remote files or replaces
local existing bytes. Inspect the prior exact process/session before invoking.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

folder = Path(__file__).resolve().parent; repo = folder.parents[3]
receipt = json.loads((folder / 'report-archive-01.json').read_text())
out = repo / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/interim-0001.tar.gz'
previous = json.loads((folder / 'interim-0001-download-result.json').read_text())
assert previous['outcome'] != 'safe_extraction_and_all_files_verified', 'No duplicate completed download'
offset = out.stat().st_size
assert 0 < offset < receipt['bytes']
prefix = hashlib.sha256(out.read_bytes()).hexdigest()
code = ('from pathlib import Path; import hashlib,sys,shutil; p=Path(' + repr(receipt['path']) + '); '
        'assert p.stat().st_size==' + str(receipt['bytes']) + '; '
        'assert hashlib.sha256(p.read_bytes()).hexdigest()==' + repr(receipt['sha256']) + '; '
        'f=p.open("rb"); first=f.read(' + str(offset) + '); assert hashlib.sha256(first).hexdigest()==' + repr(prefix) + '; '
        'shutil.copyfileobj(f,sys.stdout.buffer,1024*1024); sys.stdout.buffer.flush()')
command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=2', 'dtu-hpc',
           '/work3/s234818/mosaic/envs/protenix-tiny/bin/python -c ' + shlex.quote(code)]
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'offset': offset,
          'prefix_sha256': prefix, 'remote': receipt, 'outcome': 'unknown', 'command': command}
with (folder / 'report-resume-01-started.json').open('x') as f: json.dump(record, f, indent=2)
process = None
try:
    with (folder / 'report-resume-01-stderr.txt').open('xb') as error:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error)
        record['ssh_pid'] = process.pid
        (folder / 'report-resume-01-process.json').write_text(json.dumps(record, indent=2) + '\n')
        assert out.stat().st_size == offset and hashlib.sha256(out.read_bytes()).hexdigest() == prefix
        with out.open('ab') as stream:
            while block := process.stdout.read(1024 * 1024):
                assert stream.tell() + len(block) <= receipt['bytes'], 'Unexpected excess transfer bytes'
                stream.write(block); stream.flush()
        code = process.wait(timeout=30); record['returncode'] = code
        assert code == 0 and out.stat().st_size == receipt['bytes'] and hashlib.sha256(out.read_bytes()).hexdigest() == receipt['sha256']
    record['outcome'] = 'complete_archive_byte_verified'
    subprocess.run([sys.executable, str(repo / 'scripts/verify_hpc_evidence.py'), str(out), receipt['sha256']], check=True)
    record['outcome'] = 'safe_extraction_and_all_files_verified'
finally:
    if process is not None and process.poll() is None:
        process.terminate(); process.wait(timeout=30)
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (folder / 'report-resume-01-result.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
