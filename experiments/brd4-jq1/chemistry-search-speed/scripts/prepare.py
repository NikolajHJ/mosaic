"""Local AST/bytes only; no numerical execution or remote action."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
OUT = REPO / 'results/hpc/20261025-chemistry-speed-01/source01'


def main():
    OUT.mkdir(exist_ok=False)
    members = {}
    for name in ['README.md','scripts/README.md','tests/README.md','scripts/exact_reuse.py',
                 'scripts/parallel_search.py','scripts/benchmark.py','scripts/worker.py',
                 'scripts/stage.lsf','tests/test_speed.py']:
        raw = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
        members[name] = raw
    doc = {'version': 'chemistry-search-speed-v1', 'scientific_production_installed': False,
           'files': {k: hashlib.sha256(v).hexdigest() for k,v in sorted(members.items())}}
    raw = (json.dumps(doc, indent=2, sort_keys=True)+'\n').encode()
    (OUT / 'source_manifest.json').write_bytes(raw)
    members['source_manifest.json'] = raw
    with tarfile.open(OUT / 'source.tar.gz','x:gz') as f:
        for name, data in sorted(members.items()):
            i = tarfile.TarInfo(name);i.size=len(data);i.mode=0o444;i.mtime=0
            f.addfile(i,io.BytesIO(data))
    with tarfile.open(OUT / 'source.tar.gz') as f:
        assert {r.name for r in f} == set(members)
        for r in f.getmembers():
            assert r.isfile() and f.extractfile(r).read() == members[r.name]
    archive = (OUT / 'source.tar.gz').read_bytes()
    receipt = {'status':'local_ast_byte_package_only','manifest_sha256':hashlib.sha256(raw).hexdigest(),
               'archive_sha256':hashlib.sha256(archive).hexdigest(),'archive_bytes':len(archive),
               'files':len(members)-1,'allocated_tests_passed':False}
    (OUT / 'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
