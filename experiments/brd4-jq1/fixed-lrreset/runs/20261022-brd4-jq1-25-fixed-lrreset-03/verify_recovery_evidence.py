"""Independent stdlib audit of delivered job/CPU/cleanup metadata and bytes."""
import hashlib
import json
from pathlib import Path
import re
import tarfile
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[5]
FOLDER = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-03/evidence01'
BASE = FOLDER / 'unpacked'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    manifest = read(BASE / 'evidence-manifest.json')
    by_remote = {p['remote_path']: BASE / p['path'] for p in manifest['files']}
    for row in manifest['files']:
        p = BASE / row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']
    groups = {'lower': ('29792104', 724, 'ff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567'),
              'core': ('29792111', 282, '6d547929ce3227302729b022628cc26612d9829e72c0a70861870023463dac3a'),
              'launcher': ('29792154', 12, '99292b7eed1662f5975a704aa4adfca5a93c1754ff3717c3d75f24473f8ef219')}
    results = {}
    for label, (job, count, source) in groups.items():
        cert_path = BASE / label / 'cpu-certificate.json'; cert = read(cert_path)
        assert cert['status'] == 'passed' and cert['producer_job_id'] == job
        assert cert['tests'] == count and cert['source_manifest_sha256'] == source
        scheduler_ok = False; lsf_ok = False
        for proof in cert['proofs']:
            p = by_remote[proof['path']]; assert sha(p) == proof['sha256']
            if p.suffix == '.txt' and p.name.startswith('scheduler-final-'):
                compact = re.sub(r'\s+', '', p.read_text())
                assert re.findall(r'Job<([^>]+)>', compact) == [job]
                assert re.findall(r'Status<([^>]+)>', compact) == ['DONE']
                assert re.findall(r'Queue<([^>]+)>', compact) == ['hpc']; scheduler_ok = True
            if p.name == 'stdout.' + job + '.log':
                raw = p.read_text(); assert 'Subject: Job ' + job + ':' in raw and 'Successfully completed.' in raw
                assert not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I); lsf_ok = True
        assert scheduler_ok and lsf_ok
        counts = {}
        for xml in (BASE / label / 'cpu/payload').glob('*.xml'):
            tree = ET.parse(xml); rows = list(tree.iter('testcase'))
            assert not any(list(tree.iter(t)) for t in ('failure', 'error', 'skipped'))
            counts[xml.name] = len(rows)
        assert sum(counts.values()) == count
        results[label] = {'cpu_job': job, 'cases': count, 'by_raw_xml': counts, 'certificate_sha256': sha(cert_path)}
    assert results['lower']['by_raw_xml'] == {'hpc-contract.xml': 392, 'legacy.xml': 239, 'native-mapping.xml': 34, 'chemistry.xml': 59}
    assert (BASE / 'lower/cpu/payload/pocket_definition.json').is_file()
    assert sha(BASE / 'lower/cpu/payload/pocket_definition.json') == '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
    assert sha(BASE / 'lower/cpu/payload/composition_reference.json') == '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'
    jobs = read(BASE / 'lower/native-submissions.json')['jobs']
    assert len(jobs) == 3
    for item in jobs:
        arm = item['arm']; native = item['native']; callback = item['callback']; args = native['args']
        assert native['outcome'] == callback['outcome'] == 'submitted'
        assert native['learning_rate'] == (0.3 if arm == 'protenix' else 0.005)
        assert args[args.index('-q') + 1] == 'gpul40s' and args[args.index('-n') + 1] == '4'
        assert args[args.index('-W') + 1] == '02:00' and args[args.index('-gpu') + 1] == 'num=1:mode=exclusive_process'
        assert 'rusage[mem=8GB]' in args and '-w' not in args
        assert callback['args'][callback['args'].index('-w') + 1] == 'ended(' + native['job_id'] + ')'
    diag = read(BASE / 'launcher/diagnostic-submissions.json')
    assert diag['native']['phase'] == 'diagnostic' and diag['native']['arm'] == 'protenix'
    assert diag['capture']['dependency'] == 'ended(' + diag['native']['job_id'] + ')'
    assert diag['native']['core_manifest_sha256'] == groups['core'][2]
    plan = read(BASE / 'cleanup/plan.json'); completion = read(BASE / 'cleanup/completion.json')
    assert completion['status'] == 'completed' and completion['removed_files'] == len(plan['files']) == 1212
    assert completion['removed_logical_bytes'] == sum(p['bytes'] for p in plan['files']) == 852317966
    assert completion['plan_sha256'] == sha(BASE / 'cleanup/plan.json') == 'bfe10342b9d8378f7c5842ca3a2eb87806a46cdb08501d9e0813f81776ca7e9f'
    assert completion['journal_sha256'] == sha(BASE / 'cleanup/journal.jsonl')
    events = [json.loads(line) for line in (BASE / 'cleanup/journal.jsonl').read_text().splitlines()]
    removed = [e for e in events if e['status'] == 'removed_exact_backed_up_copy']
    assert len(removed) == 1212 and {e['path'] for e in removed} == {e['remote_path'] for e in plan['files']}
    for proof in plan['local_verified_backups']:
        assert sha(proof['path']) == proof['sha256']
    report = {'status': 'passed', 'scope': 'Independent stdlib bytes/JSON/XML/scheduler/LSF/resource/cleanup verification; no numerical model checks',
              'cpu': results, 'native_jobs': {x['arm']: x['native']['job_id'] for x in jobs},
              'corrected_diagnostic_job': diag['native']['job_id'], 'corrected_capture_job': diag['capture']['job_id'],
              'cleanup_files': 1212, 'cleanup_bytes': 852317966, 'files_byte_verified': len(manifest['files']),
              'native_readiness_or_long_run_pass_claimed': False, 'saved_original_studies_modified': False}
    output = FOLDER / 'independent-verification.json'
    assert not output.exists()
    output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
