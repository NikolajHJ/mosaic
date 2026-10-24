"""Seal the byte/metadata-only offload operators; never execute tests or SSH."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
NAMES = ('offload_protocol.py', 'stream_backup.py', 'delete_verified.py',
         'fresh_stream_backup.py', 'verify_stream_backup.py',
         'fixture_backup_contract.py', 'cleanup_cpu.py', 'cleanup_cpu.lsf',
         'fixed25_storage.py', 'fixed25_split_storage.py', 'fixed25_recover_storage.py')


def main():
    folder = REPO / 'results/hpc/20261024-storage-offload-01/operator-source-01'
    folder.mkdir(exist_ok=False)
    files = {name: (HERE / name).read_bytes() for name in NAMES}
    for name, raw in files.items():
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
    manifest = {'version': 'mosaic-offload-operator-source-v1',
        'purpose': 'Exact file backup, independent byte verification, guarded literal unlink; no scientific execution.',
        'allocated_fixture_rows_expected': 19,
        'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    archive = folder / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in {**files, 'source-manifest.json': manifest_raw}.items():
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(raw), 0o444, 0
            tar.addfile(member, io.BytesIO(raw))
    result = {'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
              'archive_sha256': hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
              'archive_bytes': archive.stat().st_size, 'source_files': len(files),
              'archive': str(archive), 'status': 'sealed_unvalidated',
              'local_validation': 'Standard-library AST and bytes only; no fixtures executed.'}
    for path in (folder / 'source-manifest.json', HERE / 'operator-source-manifest-01.json'):
        with path.open('xb') as stream:
            stream.write(manifest_raw)
    with (HERE / 'operator-package-01.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
