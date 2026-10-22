"""Independent source/config/CPU/native proof metadata audit; stdlib only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ROOT = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-04'
MANIFEST = 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert args.name.isascii() and all(c.isalnum() or c == '-' for c in args.name)
    folder = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04' / args.name
    index = json.loads((folder / 'evidence-index.json').read_bytes())
    by_remote = {}
    for row in index['files']:
        p = folder / 'payload' / row['path']
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        by_remote[row['remote_path']] = p

    def read(relative):
        return json.loads((folder / 'payload' / relative).read_bytes())

    def bound(proof):
        p = by_remote[proof['path']]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == proof['sha256']
        return p

    def final(proofs, job, queue):
        sched = lsf = False
        for proof in proofs:
            p = bound(proof)
            if p.suffix not in ['.txt', '.log']:
                continue
            raw = p.read_text(); compact = re.sub(r'\s+', '', raw)
            if re.findall(r'Job<([^>]+)>', compact) == [job] and re.findall(r'Status<([^>]+)>', compact) == ['DONE'] and re.findall(r'Queue<([^>]+)>', compact) == [queue]:
                sched = True
            if 'Subject: Job ' + job + ':' in raw and 'Successfully completed.' in raw and not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I):
                lsf = True
        assert sched and lsf, (job, queue)

    source_manifest = folder / 'payload/source/source_manifest.json'
    assert hashlib.sha256(source_manifest.read_bytes()).hexdigest() == MANIFEST
    source = json.loads(source_manifest.read_bytes())
    assert len(source['files']) == 625
    archive = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/source02/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '4d6d297451d9820543abc3298622e6599014c4b96bc999e104dd732825f56e4e'
    with tarfile.open(archive, 'r:gz') as t:
        members = {m.name.removeprefix('source/'): m for m in t.getmembers()}
        assert set(members) == set(source['files']) | {'source_manifest.json'}
        for name, expected in source['files'].items():
            assert hashlib.sha256(t.extractfile(members[name]).read()).hexdigest() == expected
    cert = read('work/cpu-certificate.json')
    assert cert['status'] == 'passed' and cert['source_manifest_sha256'] == MANIFEST
    assert cert['producer_job_id'] == '29792389' and cert['tests'] == 728
    final(cert['proofs'], '29792389', 'hpc')
    xmls = {}
    for proof in cert['proofs']:
        p = bound(proof)
        if p.suffix != '.xml':
            continue
        tree = ET.parse(p)
        assert not any(list(tree.iter(tag)) for tag in ['failure', 'error', 'skipped'])
        rows = list(tree.iter('testcase')); xmls[p.name] = len(rows)
        if p.name == 'hpc-contract.xml':
            assert sum('test_protenix_initialization' in x.get('classname', '').split('.') for x in rows) == 4
    assert xmls == {'hpc-contract.xml': 396, 'legacy.xml': 239, 'native-mapping.xml': 34, 'chemistry.xml': 59}
    native = read('work/native-submissions.json')['jobs']
    assert len(native) == 3 and {x['arm'] for x in native} == {'protenix', 'boltz2', 'boltz2_chemistry'}
    for row in native:
        arm, own, callback = row['arm'], row['native'], row['callback']
        assert own['outcome'] == callback['outcome'] == 'submitted'
        assert own['source_manifest_sha256'] == MANIFEST
        assert own['learning_rate'] == (0.3 if arm == 'protenix' else 0.005)
        assert own['cpu_certificate_sha256'] == hashlib.sha256((folder / 'payload/work/cpu-certificate.json').read_bytes()).hexdigest()
        assert own['args'][:5] == ['bsub', '-q', 'gpul40s', '-n', '4']
        assert own['args'][own['args'].index('-W') + 1] == '02:00'
        assert own['args'][own['args'].index('-gpu') + 1] == 'num=1:mode=exclusive_process'
        assert callback['args'][callback['args'].index('-w') + 1] == 'ended(' + own['job_id'] + ')'
    result = {'status': 'passed', 'scope': 'Independent standard-library bytes/source/XML/JSON/same-job LSF/scheduler audit, no numerical evaluation',
              'captured_utc': index['captured_utc'], 'files': len(by_remote), 'source_files': 625,
              'source_manifest_sha256': MANIFEST, 'cpu_cases': 728, 'raw_xml_cases': xmls,
              'cpu_certificate_sha256': hashlib.sha256((folder / 'payload/work/cpu-certificate.json').read_bytes()).hexdigest(),
              'native_jobs': {x['arm']: x['native']['job_id'] for x in native}, 'native_passes_verified': {},
              'evidence_index_sha256': hashlib.sha256((folder / 'evidence-index.json').read_bytes()).hexdigest()}
    if index['mode'] != 'cpu':
        for row in native:
            arm = row['arm']; parent = 'work/validate-' + arm + '/payload/study/'
            decision = read('work/decision-' + arm + '.json')
            summary = read(parent + 'summary.json'); assessment = read(parent + 'assessment.json')
            assert decision['status'] == summary['status'] == assessment['status'] == 'passed'
            assert decision['native_job'] == summary['job_id'] == row['native']['job_id']
            final(decision['proofs'], row['native']['job_id'], 'gpul40s')
            assert summary['source_manifest_sha256'] == assessment['source_manifest_sha256'] == MANIFEST
            assert summary['config_sha256'] == assessment['config_sha256'] == row['native']['config_sha256']
            assert summary['optimization_parameters'] == {'step_size': row['native']['learning_rate'], 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
            assert summary['native_hard_parity']['status'] == summary['native_gradient_checks']['status'] == 'passed'
            assert summary['frozen_identity_after']['model_digest'] == summary['model_digest'] == assessment['model_digest']
            assert summary['frozen_identity_after']['context_digest'] == summary['context_digest'] == assessment['context_digest']
            assert assessment['runtime_gate']['status'] == 'passed'
            for proof in assessment['proofs']:
                bound(proof)
            if arm == 'protenix':
                assert summary['baseline_initialization_match']['status'] == 'passed'
                assert summary['baseline_initialization_match']['sha256'] == '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'
                assert summary['initialization_execution']['version'] == 'protenix-fp32-cpu-atom-softmax-v1'
            for name in ['early_pilot'] + (['high_pilot'] if arm == 'boltz2_chemistry' else []):
                pilot = read(parent + name + '/report.json')
                assert pilot['status'] == 'passed' and len(pilot['updates']) == pilot['counts']['applied'] == 10
                assert all(len(x['attempts']) == 1 and x['accepted_step_size'] == row['native']['learning_rate'] for x in pilot['updates'])
            result['native_passes_verified'][arm] = {'job': row['native']['job_id'], 'assessment_sha256': decision['assessment_sha256'], 'proofs': len(assessment['proofs'])}
    with (folder / 'verification.json').open('xb') as f:
        f.write(json.dumps(result, indent=2).encode() + b'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
