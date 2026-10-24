"""Allocated byte/metadata-only cleanup operator checks. No scientific admission."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
LIMIT = 128 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def verify_source(manifest_sha):
    assert HERE == ROOT / 'operator-source-02'
    manifest = HERE / 'source-manifest.json'
    assert sha(manifest) == manifest_sha
    members = json.loads(manifest.read_bytes())['files']
    assert {p.name for p in HERE.iterdir()} == set(members) | {'source-manifest.json'}
    for name, digest in members.items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222
        assert sha(path) == digest


def admission(output, phase):
    # Exact inherited standard-library quota helpers, imported only after hashes.
    from fixed25_recover_storage import gate_record
    from fixed25_split_storage import inode_observation, tree_bytes
    output.mkdir(exist_ok=False)
    raw = subprocess.run(['getquota_work3.sh'], check=True, capture_output=True,
                         text=True, timeout=60).stdout
    # Account for the entire current Mosaic run/release/upload scope again.
    # This disposable 128MiB cleanup phase does not admit any scientific output,
    # lower a scientific projection or change the failed production guard.
    mosaic = Path('/work3/s234818/mosaic')
    stale_roots = [mosaic / 'runs', mosaic / 'releases']
    stale_roots += sorted(mosaic.glob('*.tar.gz'))
    stale = {str(p): tree_bytes(p) for p in stale_roots}
    observation = inode_observation(os.statvfs(ROOT))
    gate = gate_record('work3', raw, LIMIT, sum(stale.values()),
                       shutil.disk_usage(ROOT).free, observation)
    save(output / 'storage-gate.json', {
        'version': 'mosaic-offload-disposable-metadata-cpu-v1',
        'utc': datetime.now(timezone.utc).isoformat(),
        'phase': phase,
        'scientific_admission': False, 'scientific_projection_changed': False,
        'maximum_persistent_bytes': LIMIT, 'stale_roots': stale, 'gate': gate,
        'scope': 'Owned small metadata fixtures only; all production guards remain unchanged.'})
    assert gate['status'] == 'passed', 'Cleanup metadata allocation plus unchanged reserve/headroom does not fit'


def main():
    manifest_sha, parent = sys.argv[1], Path(sys.argv[2])
    assert os.environ.get('LSB_JOBID') and os.environ.get('LSB_QUEUE') == 'hpc'
    assert parent == ROOT / 'cpu-02'
    verify_source(manifest_sha)
    from fixed25_split_storage import tree_bytes
    output = parent / 'payload'
    admission(output, 'allocated-entry')
    temp = parent / 'owned-fixtures'
    assert not temp.exists()
    args = [sys.executable, '-m', 'pytest', '--rootdir=' + str(HERE),
            str(HERE / 'fixture_backup_contract.py'), '--basetemp=' + str(temp),
            '-q', '-p', 'no:cacheprovider', '--junitxml=' + str(output / 'operator-contract.xml')]
    save(output / 'invocation.json', {'args': args, 'job_id': os.environ['LSB_JOBID'],
                                     'queue': os.environ['LSB_QUEUE'],
                                     'source_manifest_sha256': manifest_sha})
    with (output / 'operator-contract.log').open('xb') as stdout:
        result = subprocess.run(args, stdout=stdout, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=600)
    assert result.returncode == 0, 'Allocated cleanup operator fixtures failed'
    import xml.etree.ElementTree as ET
    rows = list(ET.parse(output / 'operator-contract.xml').iter('testcase'))
    assert len(rows) == 31 and all(not list(row) for row in rows)
    assert all(row.attrib['classname'].split('.')[-1] == 'fixture_backup_contract' for row in rows)
    assert sum(tree_bytes(p) for p in (parent, HERE)) <= LIMIT
    save(output / 'assessment.json', {'status': 'passed', 'cases': len(rows),
        'suite': 'mosaic-offload-operator-contract-v1',
        'job_id': os.environ['LSB_JOBID'], 'queue': os.environ['LSB_QUEUE'],
        'source_manifest_sha256': manifest_sha,
        'proofs': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                   for p in sorted(output.iterdir()) if p.is_file()],
        'scientific_tests_or_admission': False, 'deleted_original_files': 0})


if __name__ == '__main__':
    main()
