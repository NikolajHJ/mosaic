"""Read-only native proof authentication and conservative shared artifact budget.

No models, chemistry or numerical tests run here. This additional metadata gate
precedes release of the held submission callback; it never resumes a job itself.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path('/zhome/95/e/206121/mosaic-runs')
FIRST = ROOT / '20261021-brd4-jq1-25-01'
RETRY = ROOT / '20261021-brd4-jq1-25-protenix-02'
RELEASES = Path('/work3/s234818/mosaic/releases')
SOURCE = RELEASES / RETRY.name / 'source'
sys.path.insert(0, str(SOURCE / 'scripts'))
from campaign25_contract import sha, verify_cpu_certificate, verify_readiness
from campaign25_pocket_ops import parse_final
from hpc.zhome_quota_preflight import parse_home


def read(path):
    return json.loads(path.read_text())


output = RETRY / 'cross-campaign-quota-review' / 'final-audit-01'
output.mkdir(exist_ok=False)
checks = [
    ('protenix', RETRY, '29770090', '40ab2d4532c477b5074d6631587ae15188941d5167016b7421f993257a064438'),
    ('boltz2', FIRST, '29770007', '2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c'),
    ('boltz2_chemistry', FIRST, '29770009', '2503c2382b429ec6b79d7a256b2a2fccd09f7739a609e97a876d604d00dc063c'),
]
record = {'status': 'failed', 'utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Saved proof/schema/byte authentication and shared storage arithmetic only',
          'auditor_sha256': sha(__file__), 'native_arms': []}
try:
    for arm, campaign, job, manifest in checks:
        source = RELEASES / campaign.name / 'source'
        subprocess.run([sys.executable, str(source / 'scripts/verify_stage0_release.py'), str(source), manifest], check=True)
        dispatch = read(campaign / 'dispatch-started.json')
        verify_cpu_certificate(campaign / 'cpu-certificate.json', dispatch['cpu_certificate_sha256'], manifest)
        raw = subprocess.run(['bjobs', '-l', job], check=True, capture_output=True, text=True).stdout
        (output / f'scheduler-{job}.txt').write_text(raw)
        assert parse_final(raw, job) == ('DONE', 'gpul40s')
        parent = campaign / ('validate-' + arm)
        stdout = parent / f'stdout.{job}.log'
        assert f'Subject: Job {job}:' in stdout.read_text() and 'Successfully completed.' in stdout.read_text()
        path = parent / 'payload/study/assessment.json'
        assessment = verify_readiness(path, sha(path), source / f'examples/configs/brd4_jq1_25_{arm}_10000.json', manifest)
        summary = read(path.with_name('summary.json'))
        assert summary['status'] == summary['native_hard_parity']['status'] == summary['native_gradient_checks']['status'] == 'passed'
        assert summary['runtime_gate']['status'] == summary['pocket_validation_status'] == 'passed'
        assert summary['diffusion_gradient'] is False
        for identity in ('model_digest', 'context_digest'):
            assert summary['frozen_identity_after'][identity] == summary[identity] == assessment[identity]
        if arm == 'protenix':
            assert summary['native_gradient_checks']['primal_every_step'] is True
        if arm == 'boltz2_chemistry':
            assert summary['chemistry_assay']['status'] == 'passed'
            for label in ('low', 'high'):
                assert read(path.parent / (label + '_coupled_checks/report.json'))['status'] == 'passed'
        projection = assessment['storage_projection']['projected_bytes']
        assert type(projection) is int and projection > 0
        record['native_arms'].append({'arm': arm, 'job': job, 'assessment': str(path),
            'assessment_sha256': sha(path), 'projected_bytes': projection,
            'stdout_sha256': sha(stdout), 'source_manifest_sha256': manifest,
            'model_digest': assessment['model_digest'], 'context_digest': assessment['context_digest']})
    hold = subprocess.run(['bjobs', '-l', '29770091'], check=True, capture_output=True, text=True).stdout
    (output / 'held-callback.txt').write_text(hold)
    normalized = re.sub(r'\s+', '', hold)
    assert 'Job<29770091>' in normalized and 'Status<PSUSP>' in normalized
    quota = subprocess.run(['getquota_zhome.sh'], check=True, capture_output=True, text=True).stdout
    (output / 'quota.txt').write_text(quota)
    parsed = parse_home(quota)
    sizes = {str(campaign): sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file()) for campaign in (FIRST, RETRY)}
    available = min(parsed['hard_bytes_lower'] - parsed['used_bytes_upper'], shutil.disk_usage(RETRY).free) - sum(sizes.values())
    required = sum(arm['projected_bytes'] for arm in record['native_arms']) + 4 * 1024**3
    record.update(available_bytes_conservative=available, required_bytes=required,
                  reserve_bytes=4 * 1024**3, campaign_bytes_subtracted=sizes,
                  quota=parsed, limitation='Cached quota is not a reservation; both roots subtracted; original callback repeats fresh quota')
    assert available >= required, 'Whole-three-arm artifact projection exceeds conservative shared quota'
    record['status'] = 'passed'
finally:
    (output / 'assessment.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
