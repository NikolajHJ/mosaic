"""Immutable byte/AST packaging from authenticated source610; no numerical work."""
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TAG = HERE.name
BASE = REPO / 'results/hpc/20261021-brd4-jq1-25-fixed-lrboost-01/source01'
OUT = REPO / 'results/hpc' / TAG / 'source01'
OVERLAYS = ['.gitattributes', 'scripts/campaign25_contract.py', 'examples/brd4_25_atoms.py',
    'docs/brd4_jq1_25_fixed_lrreset.md', 'scripts/fixed25_lrreset_ops.py',
    'scripts/fixed25_lrreset_storage.py', 'scripts/hpc/fixed25_lrreset_cpu.lsf',
    'scripts/hpc/fixed25_lrreset_stage.lsf', 'scripts/hpc/lrreset_predecessors.json',
    *['examples/configs/brd4_jq1_25_fixed_lrreset_' + a + '_10000.json'
      for a in ('protenix','boltz2','boltz2_chemistry')],
    *['tests/test_fixed25_lrreset_' + p + '.py' for p in ('contract','ops','storage')]]
def sha(b): return hashlib.sha256(b).hexdigest()
def put(path, v):
    with path.open('xb') as f: f.write((json.dumps(v, indent=2, allow_nan=False) + '\n').encode())
base_bytes = (BASE / 'source_manifest.json').read_bytes()
assert sha(base_bytes) == '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
assert sha((BASE / 'source.tar.gz').read_bytes()) == 'e1a100aebf44698515ba161c1a001e338a902b250bf1a737f888df33f55098d1'
manifest = json.loads(base_bytes)
with tarfile.open(BASE / 'source.tar.gz') as archive:
    files = {}
    for m in archive.getmembers():
        assert m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts
        name = m.name.removeprefix('source/')
        assert name not in files
        files[name] = archive.extractfile(m).read()
assert files.pop('source_manifest.json') == base_bytes
assert len(files) == 610 and {n:sha(b) for n,b in files.items()} == manifest['files']
original = dict(files)
for name in OVERLAYS:
    b = subprocess.check_output(['git','show',':'+name], cwd=REPO)
    current = (REPO / name).read_bytes()
    if name in ('.gitattributes', 'docs/brd4_jq1_25_fixed_lrreset.md'):
        assert current.replace(b'\r\n',b'\n') == b
    else:
        assert current == b, 'Current/index byte mismatch: '+name
    if name.endswith('.py'): ast.parse(b.decode(), filename=name)
    if name.endswith('.json'): json.loads(b)
    files[name] = b
assert len(files) == 622
assert all(files[n] == b for n,b in original.items() if n.startswith('src/'))
for arm, rate in [('protenix',.3),('boltz2',.005),('boltz2_chemistry',.005)]:
    new = json.loads(files[f'examples/configs/brd4_jq1_25_fixed_lrreset_{arm}_10000.json'])
    old = json.loads(files[f'examples/configs/brd4_jq1_25_fixed_lrboost_{arm}_10000.json'])
    expected = {'version','initial_step_size','label'} | ({'repair_schedule'} if arm == 'boltz2_chemistry' else set())
    assert {k for k in new.keys() | old.keys() if new.get(k) != old.get(k)} == expected
    assert new['initial_step_size'] == rate and new['max_backtracks'] == 1 and new['acceptance_policy'] == 'fixed_step'
    if arm == 'boltz2_chemistry':
        assert new['repair_schedule'] == {**old['repair_schedule'], 'final':100.0}
OUT.mkdir(parents=True, exist_ok=False)
record = {'created_utc':datetime.now(timezone.utc).isoformat(),
    'git_revision_reference':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'identity_policy':'Authenticated source610 plus explicit lower-rate/coefficient and launcher overlays from Git index',
    'parent_manifest_sha256':sha(base_bytes), 'declared_overlays':OVERLAYS,
    'files':{n:sha(b) for n,b in sorted(files.items())}}
put(OUT / 'source_manifest.json',record)
files['source_manifest.json'] = (OUT / 'source_manifest.json').read_bytes()
with tarfile.open(OUT / 'source.tar.gz','w:gz') as archive:
    for name,b in sorted(files.items()):
        m=tarfile.TarInfo('source/'+name);m.size=len(b);m.mode=0o444;m.mtime=0
        archive.addfile(m,io.BytesIO(b))
b=(OUT/'source.tar.gz').read_bytes()
result={'status':'passed','scope':'Local stdlib byte/hash/AST/JSON packaging only; no tests or numerical execution',
    'source_files':622,'archive_members':623,'manifest_sha256':sha(files['source_manifest.json']),
    'archive_bytes':len(b),'archive_sha256':sha(b),
    'source_root':'/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-01/source',
    'unchanged_scientific_library_files':sum(n.startswith('src/') for n in original),
    'modified_base_files':[n for n in original if files[n] != original[n]],
    'added_files':[n for n in record['files'] if n not in original],
    'exact_index_executable_and_configuration_bytes':True}
put(OUT/'source-comparison.json',result)
put(HERE/'source-manifest.json',record)
put(HERE/'source-comparison.json',result)
print(json.dumps(result,indent=2))
