"""Byte-check every served local report file; no scientific calculations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

folder = Path(__file__).resolve().parent; repo = folder.parents[3]
viewer = repo / 'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/viewer'
base = 'http://127.0.0.1:8778/'
files = sorted(p for p in viewer.rglob('*') if p.is_file())
def check(path):
    relative = path.relative_to(viewer).as_posix(); assert not path.is_symlink()
    local = path.read_bytes()
    with urllib.request.urlopen(base + urllib.parse.quote(relative), timeout=30) as response:
        remote = response.read(); assert response.status == 200
    assert remote == local, 'Served bytes differ: ' + relative
    return {'path': relative, 'bytes': len(local), 'sha256': hashlib.sha256(local).hexdigest()}
with ThreadPoolExecutor(max_workers=12) as pool: records = list(pool.map(check, files))
result = {'status': 'passed', 'utc': datetime.now(timezone.utc).isoformat(), 'url': base + 'report.html',
          'verified_served_files': len(records), 'scope': 'HTTP200 and exact local/server bytes for every served file; no local scientific recomputation', 'files': records}
out = folder / 'http-verification-0002.json'
with out.open('x', encoding='utf-8') as stream: json.dump(result, stream, indent=2); stream.write('\n')
print(json.dumps({'status': 'passed', 'verified_served_files': len(records), 'receipt_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))
