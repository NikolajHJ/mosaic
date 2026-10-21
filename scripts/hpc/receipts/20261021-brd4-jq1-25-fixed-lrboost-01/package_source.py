"""Standard-library byte/AST packaging only; no scientific execution."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

TAG = '20261021-brd4-jq1-25-fixed-lrboost-01'
REPO = Path(__file__).resolve().parents[4]
BASE = REPO / 'results/hpc/20261021-brd4-jq1-25-fixed-02/source02'
OUT = REPO / 'results/hpc' / TAG / 'source01'
RECEIPTS = Path(__file__).resolve().parent
OVERLAYS = ['.gitattributes', 'scripts/campaign25_contract.py', 'examples/brd4_25_atoms.py',
    'docs/brd4_jq1_25_fixed_lrboost.md', 'scripts/fixed25_split_storage.py',
    'scripts/fixed25_recover_storage.py', 'scripts/fixed25_lrboost_ops.py',
    'scripts/fixed25_lrboost_storage.py', 'scripts/hpc/fixed25_lrboost_cpu.lsf',
    'scripts/hpc/fixed25_lrboost_stage.lsf',
    *['examples/configs/brd4_jq1_25_fixed_lrboost_' + arm + '_10000.json'
      for arm in ('protenix', 'boltz2', 'boltz2_chemistry')],
    *['tests/test_fixed25_lrboost_' + part + '.py' for part in ('contract', 'ops', 'storage')]]

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, obj):
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + '\n').encode())

assert digest((BASE / 'source.tar.gz').read_bytes()) == 'c024ac03ddcf6091e35f1a666117194cd70b05024ee4563c6d8c965d75ed5798'
base_manifest_bytes = (BASE / 'source_manifest.json').read_bytes()
assert digest(base_manifest_bytes) == '1e3a8a21e2fa68e24d1c9229238999c42a4a9b44bcc78344e73e188c87d1ee64'
base_manifest = json.loads(base_manifest_bytes)
with tarfile.open(BASE / 'source.tar.gz') as archive:
    members = archive.getmembers()
    files = {}
    for member in members:
        if member.isdir():
            continue
        assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        name = member.name.removeprefix('source/')
        assert name not in files
        files[name] = archive.extractfile(member).read()
assert files.pop('source_manifest.json') == base_manifest_bytes
assert len(files) == 597 and {n: digest(b) for n, b in files.items()} == base_manifest['files']
original = dict(files)
for name in OVERLAYS:
    data = subprocess.check_output(['git', 'show', ':' + name], cwd=REPO)
    current = (REPO / name).read_bytes()
    assert data == current, 'Index/current bytes differ: ' + name
    if name.endswith('.py'):
        ast.parse(data.decode('utf-8'), filename=name)
    if name.endswith('.json'):
        json.loads(data)
    files[name] = data
assert len(files) == 610
assert all(files[n] == b for n, b in original.items() if n.startswith('src/'))
for tag, filename, helper in [
    ('20261021-brd4-jq1-25-fixed-storage-03', 'source01', 'scripts/fixed25_split_storage.py'),
    ('20261021-brd4-jq1-25-fixed-storage-04', 'source03', 'scripts/fixed25_recover_storage.py')]:
    manifest = json.loads((REPO / 'results/hpc' / tag / filename / 'source/source_manifest.json').read_bytes())
    assert digest(files[helper]) == manifest['files'][helper]
for arm, rate in [('protenix', 3.0), ('boltz2', .05), ('boltz2_chemistry', .05)]:
    new = json.loads(files[f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json'])
    old = json.loads(files[f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json'])
    differences = {key for key in new.keys() | old.keys() if new.get(key) != old.get(key)}
    assert differences == {'version', 'initial_step_size', 'label'} and new['initial_step_size'] == rate
    assert new['max_backtracks'] == 1 and new['acceptance_policy'] == 'fixed_step'
OUT.mkdir(parents=True, exist_ok=False)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'git_revision_reference': revision,
    'identity_policy': 'Authenticated scientific source02 plus declared versioned rate/launcher/config/fixture overlays from exact Git index bytes; commit reference only',
    'parent_manifest_sha256': digest(base_manifest_bytes),
    'declared_overlays': OVERLAYS,
    'files': {n: digest(b) for n, b in sorted(files.items())}}
write_json(OUT / 'source_manifest.json', manifest)
files['source_manifest.json'] = (OUT / 'source_manifest.json').read_bytes()
with tarfile.open(OUT / 'source.tar.gz', 'w:gz') as archive:
    for name, data in sorted(files.items()):
        member = tarfile.TarInfo('source/' + name)
        member.size = len(data)
        member.mode = 0o444
        member.mtime = 0
        archive.addfile(member, io.BytesIO(data))
archive_bytes = (OUT / 'source.tar.gz').read_bytes()
record = {'status': 'passed', 'scope': 'Local standard-library AST/JSON/byte/hash packaging only; no tests or numerical execution',
    'source_files': 610, 'archive_members': 611,
    'manifest_sha256': digest(files['source_manifest.json']),
    'archive_bytes': len(archive_bytes), 'archive_sha256': digest(archive_bytes),
    'source_root': f'/work3/s234818/mosaic/releases/{TAG}/source',
    'unchanged_src_files': sum(n.startswith('src/') for n in original),
    'modified_base_files': [n for n in original if files[n] != original[n]],
    'added_files': [n for n in manifest['files'] if n not in original],
    'git_index_and_current_bytes_equal': True, 'pure_storage_helpers_match_original_authenticated_manifests': True}
write_json(OUT / 'source-comparison.json', record)
write_json(RECEIPTS / 'source-manifest.json', manifest)
write_json(RECEIPTS / 'source-comparison.json', record)
print(json.dumps(record, indent=2))
