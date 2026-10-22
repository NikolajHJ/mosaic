"""Fresh authenticated SSH byte transfer and safe stdlib evidence verification.

Existing files or markers prohibit retries. Inspect any interrupted transfer
before selecting a separately declared recovery; this helper never resumes.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

folder = Path(__file__).resolve().parent
repo = folder.parents[3]
receipt = json.loads((folder / sys.argv[1]).read_text(encoding='utf-8'))
label = sys.argv[2]
assert repo.name == 'mosaic' and re.fullmatch(r'[a-z0-9-]+', label)
remote = receipt['path']
assert re.fullmatch(r'/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/[a-z0-9-]+\.tar\.gz', remote)
assert type(receipt['bytes']) is int and receipt['bytes'] > 0
assert re.fullmatch(r'[0-9a-f]{64}', receipt['sha256'])
output = repo / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01' / (label + '.tar.gz')
marker = folder / (label + '-stream-started.json')
result = folder / (label + '-stream-result.json')
assert not output.exists() and not marker.exists() and not result.exists()
remote_code = '''import hashlib,os,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert p.is_file() and not p.is_symlink()
with p.open('rb') as stream:
 assert os.fstat(stream.fileno()).st_size==int(sys.argv[2])
 assert hashlib.file_digest(stream,'sha256').hexdigest()==sys.argv[3]
 stream.seek(0)
 for block in iter(lambda:stream.read(1048576),b''):sys.stdout.buffer.write(block)
 sys.stdout.buffer.flush()
'''
remote_command = ' '.join(shlex.quote(value) for value in [
    '/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', remote_code,
    remote, str(receipt['bytes']), receipt['sha256']])
command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ConnectionAttempts=1', 'dtu-hpc', remote_command]
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
          'receipt': receipt, 'outcome': 'unknown', 'fresh_file': str(output)}
with marker.open('x', encoding='utf-8') as stream:
    json.dump(record, stream, indent=2)
process = None
try:
    with output.open('xb') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.PIPE)
        record['pid'] = process.pid
        try:
            _, error = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            process.kill()
            _, error = process.communicate(timeout=20)
            record['outcome'] = 'timeout_stopped_partial_preserved'
            raise
        finally:
            (folder / (label + '-stream-stderr.txt')).write_bytes(error)
        record['returncode'] = process.returncode
    assert process.returncode == 0 and output.stat().st_size == receipt['bytes']
    with output.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['sha256']
    record['outcome'] = 'complete_bytes_verified'
    subprocess.run([sys.executable, str(repo / 'scripts/verify_hpc_evidence.py'),
                    str(output), receipt['sha256']], check=True)
    record['outcome'] = 'safe_extraction_and_all_files_verified'
finally:
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    record['observed_bytes'] = output.stat().st_size if output.exists() else 0
    with result.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
print(json.dumps(record, indent=2))
