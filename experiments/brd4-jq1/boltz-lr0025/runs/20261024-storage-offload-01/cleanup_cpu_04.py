"""Allocated byte/metadata-only cleanup operator checks. No scientific admission."""
import hashlib
import json
import os
import re
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
    assert HERE == ROOT / 'operator-source-04'
    manifest = HERE / 'source-manifest.json'
    assert sha(manifest) == manifest_sha
    members = json.loads(manifest.read_bytes())['files']
    assert {p.name for p in HERE.iterdir()} == set(members) | {'source-manifest.json'}
    for name, digest in members.items():
        path = HERE / name
        assert path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222
        assert sha(path) == digest
    verify_parent(json.loads(manifest.read_bytes()))



def verify_parent(manifest):
    """Authenticate unchanged03 source and actual32-case allocated success."""
    import xml.etree.ElementTree as ET
    pin = manifest['parent_adoption']
    assert pin['source_manifest_sha256'] == '9af242392c14b3fc09804f0868b49f0d650245a1f075d43c3b132d994545230d'
    source = ROOT / 'operator-source-03'
    path = source / 'source-manifest.json'
    assert not path.is_symlink() and sha(path) == pin['source_manifest_sha256']
    old = json.loads(path.read_bytes())
    assert {p.name for p in source.iterdir()} == set(old['files']) | {'source-manifest.json'}
    for name, digest in old['files'].items():
        path = source / name
        assert path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222 and sha(path) == digest
    for name, proof in pin['proofs'].items():
        path = Path(name)
        assert path.is_file() and not path.is_symlink() and path.stat().st_size == proof['bytes'] and sha(path) == proof['sha256']
    parent = ROOT / 'cpu-03'
    rows = list(ET.parse(parent / 'payload/operator-contract.xml').iter('testcase'))
    assert len(rows) == 32 and all(not list(row) for row in rows)
    assert all(row.attrib['classname'].split('.')[-1] == 'fixture_backup_contract' for row in rows)
    assessment = json.loads((parent / 'payload/assessment.json').read_bytes())
    assert assessment['status'] == 'passed' and assessment['cases'] == 32
    assert assessment['job_id'] == '29806228' and assessment['queue'] == 'hpc'
    assert assessment['source_manifest_sha256'] == pin['source_manifest_sha256']
    log = (parent / 'stdout.29806228.log').read_text()
    assert re.findall(r'^Subject:\s*Job\s+(\d+):[^\r\n]*\b(Done|Exited)\s*$', log, re.M) == [('29806228', 'Done')]
    assert 'Successfully completed.' in log
    submission = json.loads((parent / 'submission.json').read_bytes())
    expected = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]', '-W', '00:30',
                '-J', 'mosaic-storage-offload-cpu-03', '-oo', str(parent / 'stdout.%J.log'),
                '-eo', str(parent / 'stderr.%J.log'), 'bash', '-l', str(source / 'cleanup_cpu.lsf'), pin['source_manifest_sha256']]
    assert submission['args'] == expected and submission['job_id'] == '29806228'
    assert submission['returncode'] == 0 and submission['source_manifest_sha256'] == pin['source_manifest_sha256']
    assert json.loads((parent / 'payload/storage-gate.json').read_bytes())['gate']['status'] == 'passed'


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
    assert parent == ROOT / 'cpu-04'
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
    assert len(rows) == 44 and all(not list(row) for row in rows)
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
