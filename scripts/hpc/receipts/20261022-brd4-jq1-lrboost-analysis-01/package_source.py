"""Standard-library immutable byte/AST packaging; no tests/model imports."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

TAG = '20261022-brd4-jq1-lrboost-analysis-01'
REPO = Path(__file__).resolve().parents[4]
OUT = REPO / 'results/hpc' / TAG / 'source01'
RECEIPTS = Path(__file__).resolve().parent
FILES = ['scripts/' + name for name in ('jq1_lrboost_report_contract.py', 'jq1_lrboost_report_ops.py',
    'lrboost_endpoint_diagnostics.py', 'capture_jq1_lrboost_completion.py', 'render_jq1_lrboost_report.py',
    'render_jq1_lrboost_progress.py', 'publish_jq1_lrboost_report.py', 'verify_stage0_release.py')]
FILES += ['scripts/templates/jq1_lrboost_report.html', 'scripts/hpc/jq1_lrboost_analysis_stage.lsf',
          'docs/brd4_jq1_25_endpoint_diagnostics.md']
FILES += ['tests/' + name for name in ('test_jq1_lrboost_report_contract.py', 'test_jq1_lrboost_ops.py',
    'test_jq1_lrboost_render.py', 'test_jq1_lrboost_publish.py', 'test_lrboost_endpoint_diagnostics.py')]

def digest(data): return hashlib.sha256(data).hexdigest()
def write(path, obj): path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())

files = {}
for name in FILES:
    value = subprocess.check_output(['git', 'show', ':' + name], cwd=REPO)
    assert value == (REPO / name).read_bytes(), 'Index/current bytes differ: ' + name
    if name.endswith('.py'): ast.parse(value.decode('utf-8'), filename=name)
    assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
    files[name] = value
OUT.mkdir(parents=True, exist_ok=False)
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'git_revision_reference': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'scope': 'Separate read-only endpoint diagnostics and saved-only presentation; no original scientific edits/restarts',
    'external_scientific_source_manifest_sha256': '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323',
    'files': {n: digest(b) for n, b in sorted(files.items())}}
write(OUT / 'source_manifest.json', manifest); files['source_manifest.json'] = (OUT / 'source_manifest.json').read_bytes()
with tarfile.open(OUT / 'source.tar.gz', 'w:gz') as archive:
    for name, data in sorted(files.items()):
        member = tarfile.TarInfo(name); member.size = len(data); member.mode = 0o444; member.mtime = 0
        archive.addfile(member, io.BytesIO(data))
archive_bytes = (OUT / 'source.tar.gz').read_bytes()
record = {'status': 'passed', 'scope': 'Local static AST/byte/JSON/hash packaging only; allocated tests pending',
    'source_files': len(FILES), 'archive_members': len(files), 'manifest_sha256': digest(files['source_manifest.json']),
    'archive_bytes': len(archive_bytes), 'archive_sha256': digest(archive_bytes), 'git_index_and_current_bytes_equal': True}
write(RECEIPTS / 'source-manifest.json', manifest); write(RECEIPTS / 'local-static-verification.json', record)
write(OUT / 'local-static-verification.json', record)
print(json.dumps(record, indent=2))
