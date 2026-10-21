"""Capture final three-arm submission/quota receipts, without model execution."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

BASE = Path('/zhome/95/e/206121/mosaic-runs')
root = BASE / '20261021-brd4-jq1-25-protenix-02'
first = BASE / '20261021-brd4-jq1-25-01'
output = BASE / '20261021-brd4-jq1-25-final-submissions-03'
output.mkdir(exist_ok=False)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


paths = set(root.glob('*.json'))
paths.update(p for p in (root / 'cross-campaign-quota-review').rglob('*') if p.is_file())
for folder in ('after-ready-protenix', 'run-protenix'):
    paths.update(p for p in (root / folder).glob('*') if p.is_file())
for name in ('summary.json', 'assessment.json'):
    paths.add(root / 'validate-protenix/payload/study' / name)
assert sha(root / 'validate-protenix/payload/study/assessment.json') == '70fa96f9009a0e7316fff341e8c091bfd5339f9c9e2f4456f39d35b6a2e13cf8'
for path in paths:
    target = output / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)

jobs = [('protenix', root, '29770179'), ('boltz2', first, '29770093'), ('boltz2_chemistry', first, '29770124')]
records = []
for arm, campaign, job in jobs:
    receipt_path = campaign / ('run-' + arm) / 'submission.json'
    receipt = json.loads(receipt_path.read_text())
    args = receipt['args']
    assert receipt['job_id'] == job and receipt['returncode'] == 0 and '-w' not in args
    for option, value in (('-q', 'gpul40s'), ('-n', '4'), ('-W', '23:00'), ('-gpu', 'num=1:mode=exclusive_process')):
        assert args[args.index(option) + 1] == value
    assert 'rusage[mem=8GB]' in args
    shutil.copyfile(receipt_path, output / ('submission-' + arm + '.json'))
    raw = subprocess.run(['bjobs', '-l', job], check=True, capture_output=True, text=True).stdout
    (output / ('scheduler-' + job + '.txt')).write_text(raw)
    normalized = re.sub(r'\s+', '', raw)
    assert f'Job<{job}>' in normalized and 'Queue<gpul40s>' in normalized
    assert 'rusage[mem=8192.00]' in normalized and '1380.0min' in normalized
    status = re.search(r'Status<([^>]+)>', normalized)[1]
    assert status in ('PEND', 'RUN')
    records.append({'arm': arm, 'job_id': job, 'actual_status': status, 'source_root': receipt['args'][-7], 'submission_sha256': sha(receipt_path)})
raw = subprocess.run(['bjobs', '-l', '29770091'], check=True, capture_output=True, text=True).stdout
(output / 'scheduler-29770091.txt').write_text(raw)
assert 'Status<DONE>' in re.sub(r'\s+', '', raw)
assert 'Successfully completed.' in (root / 'after-ready-protenix/stdout.29770091.log').read_text()
(output / 'actual-submissions.json').write_text(json.dumps({'jobs': records, 'scope': 'Actual submissions, not completed 10000-update trajectories'}, indent=2) + '\n')
manifest = {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(output.rglob('*')) if p.is_file()}
(output / 'evidence_manifest.json').write_text(json.dumps({'files': manifest}, indent=2) + '\n')
archive = output.with_name(output.name + '.tar.gz')
with tarfile.open(archive, 'w:gz') as handle:
    for path in sorted(output.rglob('*')):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive), 'files': len(manifest), 'manifest_sha256': sha(output / 'evidence_manifest.json')}
archive.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'receipt': receipt, 'jobs': records}))
