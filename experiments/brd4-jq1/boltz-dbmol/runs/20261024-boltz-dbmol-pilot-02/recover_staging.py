"""Distinct recovery after authenticated pre-command SSH rejection; staging only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def main():
    folder = REPO / 'results/hpc' / HERE.name
    previous = json.loads((folder / 'staging-result.json').read_bytes())
    assert previous['returncode'] == 255 and not (folder / 'staging-stdout.bin').read_bytes()
    assert hashlib.sha256((folder / 'staging-stderr.bin').read_bytes()).hexdigest() == 'dd5abe512547a2d013ab0b4b53f566c336259151633db6dd7cdf8151ef54b9cc'
    inspection = REPO / 'results/hpc/20261024-boltz-startup-recovery-02/inspection'
    assert json.loads((inspection / 'result.json').read_bytes())['returncode'] == 0
    observed = json.loads((inspection / 'stdout.bin').read_bytes())
    assert all(not value['lexists'] for value in observed['targets'].values())
    tree = ast.parse((HERE / 'submit_science.py').read_bytes())
    remote = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'REMOTE' for target in node.targets)]
    assert len(remote) == 1
    package = json.loads((HERE.parent / '20261024-boltz-dbmol-storage-05/science-package.json').read_bytes())
    archive = folder / 'source01/source.tar.gz'
    assert archive.stat().st_size == package['archive_bytes']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == package['archive_sha256']
    attempt = folder / 'staging-recovery-02'
    attempt.mkdir(exist_ok=False)
    command = ' '.join(shlex.quote(x) for x in ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', remote[0],
                        package['archive_sha256'], str(package['archive_bytes']), package['manifest_sha256']])
    args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'dtu-hpc', 'bash -l -c ' + shlex.quote(command)]
    (attempt / 'started.json').write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'args': args,
        'inspection_sha256': hashlib.sha256((inspection / 'stdout.bin').read_bytes()).hexdigest(),
        'prior_authentication_marker_preserved': True}, indent=2) + '\n', encoding='utf8')
    try:
        result = subprocess.run(args, input=archive.read_bytes(), capture_output=True, timeout=240)
    except subprocess.TimeoutExpired as error:
        (attempt / 'timeout.json').write_text(json.dumps({'outcome': 'unknown', 'error': repr(error)}) + '\n')
        raise
    (attempt / 'stdout.bin').write_bytes(result.stdout)
    (attempt / 'stderr.bin').write_bytes(result.stderr)
    (attempt / 'result.json').write_text(json.dumps({'returncode': result.returncode,
        'stdout_sha256': hashlib.sha256(result.stdout).hexdigest(),
        'stderr_sha256': hashlib.sha256(result.stderr).hexdigest()}, indent=2) + '\n')
    print(result.stdout.decode(errors='replace')[-3000:])
    print(result.stderr.decode(errors='replace')[-1500:])
    assert result.returncode == 0, 'Preserve failed/unknown recovery; inspect before any further action'


if __name__ == '__main__':
    main()
