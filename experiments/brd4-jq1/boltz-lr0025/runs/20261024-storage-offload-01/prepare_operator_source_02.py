"""Preserved-failure successor: timestamp roots and a constructible symlink fixture."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
OVERLAYS = {'offload_protocol.py': 'offload_protocol_02.py',
    'verify_stream_backup.py': 'verify_stream_backup_02.py',
    'fixture_backup_contract.py': 'fixture_backup_contract_02.py',
    'cleanup_cpu.py': 'cleanup_cpu_02.py', 'cleanup_cpu.lsf': 'cleanup_cpu_02.lsf'}


def main():
    original = REPO / 'results/hpc/20261024-storage-offload-01/operator-source-01/source.tar.gz'
    assert hashlib.file_digest(original.open('rb'), 'sha256').hexdigest() == 'f0d9a3f6175a4c5128c92bd4872b802bf9c5ec656d1a8c951957abac122ad382'
    with tarfile.open(original) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar}
    old = json.loads(files.pop('source-manifest.json'))
    assert len(files) == 11
    for name, digest in old['files'].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest
    for name, local in OVERLAYS.items():
        files[name] = (HERE / local).read_bytes()
    for name, raw in files.items():
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    manifest = {**old, 'allocated_fixture_rows_expected': 31, 'parent_manifest_sha256': '984e39e117ac6de6940d61b9d875b36608f69a60cb9c28eff92b768adf97754b',
        'correction': 'Actual timestamp-shaped legacy-root acceptance with invalid timestamp regressions; synthetic short-link tar fixture; independent new source/CPU parent. Original18pass/onefixture failure preserved.',
        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    raw_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-source-02'
    folder.mkdir(exist_ok=False)
    archive = folder / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in {**files, 'source-manifest.json': raw_manifest}.items():
            member = tarfile.TarInfo(name); member.size, member.mode, member.mtime = len(raw), 0o444, 0
            tar.addfile(member, io.BytesIO(raw))
    result = {'status': 'sealed_unvalidated', 'source_files': len(files),
        'manifest_sha256': hashlib.sha256(raw_manifest).hexdigest(),
        'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
        'archive_bytes': archive.stat().st_size, 'archive': str(archive)}
    for path in (folder / 'source-manifest.json', HERE / 'operator-source-manifest-02.json'):
        with path.open('xb') as stream: stream.write(raw_manifest)
    with (HERE / 'operator-package-02.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
