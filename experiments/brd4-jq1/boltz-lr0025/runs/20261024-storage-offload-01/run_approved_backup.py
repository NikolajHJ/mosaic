"""Bind the exact validated client to its versioned validated verifier alias."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    manifest_path = HERE / 'operator-source-manifest-03.json'
    assert sha(manifest_path) == '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
    manifest = json.loads(manifest_path.read_bytes())
    audit = json.loads((HERE / 'operator-verification-03.json').read_bytes())
    assert audit['status'] == 'passed' and audit['cases'] == 32 and audit['job_id'] == '29806228'
    assert audit['source_manifest_sha256'] == sha(manifest_path)
    verifier = HERE / 'verify_stream_backup_02.py'
    client = HERE / 'fresh_stream_backup.py'
    assert sha(verifier) == manifest['files']['verify_stream_backup.py']
    assert sha(client) == manifest['files']['fresh_stream_backup.py']
    # Immutable archives use the shared import name. The working run keeps
    # older failed-version files, so supply that same validated module under
    # its archive name. No path guard, plan check or implementation is patched.
    spec = importlib.util.spec_from_file_location('verify_stream_backup', verifier)
    module = importlib.util.module_from_spec(spec)
    sys.modules['verify_stream_backup'] = module
    spec.loader.exec_module(module)
    sys.argv[0] = str(client)
    runpy.run_path(str(client), run_name='__main__')


if __name__ == '__main__':
    main()
