"""Independent stdlib evidence audit; never loads scientific NPZ arrays."""
import hashlib
import json
from pathlib import Path
import re


def main():
    repo = Path(__file__).resolve().parents[5]
    folder = repo / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/probe-diagnostic-evidence01'
    index = json.loads((folder / 'evidence-index.json').read_bytes())
    files = {}
    for row in index['files']:
        p = folder / 'payload' / row['path']
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']
        files[row['remote_path']] = p
    assert len(files) == 416
    for path, row in index['remote_reference_rehash'].items():
        if row['downloaded']:
            assert path in files and hashlib.sha256(files[path].read_bytes()).hexdigest() == row['sha256']

    def read(relative):
        return json.loads((folder / 'payload' / relative).read_bytes())

    for job, queue, relative in [('29792346', 'hpc', 'probe/cpu'), ('29792347', 'gpul40s', 'probe/gpu'),
                                  ('29792158', 'gpul40s', 'diagnostic/native'), ('29792159', 'hpc', 'diagnostic/capture')]:
        blocks = [b for b in re.split(r'(?=Job <\d+>)', index['scheduler']['stdout'])
                  if 'Job<' + job + '>' in re.sub(r'\s+', '', b)]
        assert len(blocks) == 1
        compact = re.sub(r'\s+', '', blocks[0])
        assert re.findall(r'Status<([^>]+)>', compact) == ['DONE']
        assert re.findall(r'Queue<([^>]+)>', compact) == [queue]
        raw = (folder / 'payload' / relative / ('stdout.' + job + '.log')).read_text()
        assert 'Subject: Job ' + job + ':' in raw and 'Successfully completed.' in raw
        assert not re.search(r'Exited\s+(?:with exit|by signal)', raw, re.I)
    probe = read('probe/gpu/payload/assessment.json')
    assert probe['status'] == 'diagnostic_completed' and probe['job_id'] == '29792347'
    assert probe['baseline_sha256'] == '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'
    assert probe['fixture_one_ulp_detection'] == 'passed'
    variants = probe['variants']
    assert variants['cpu-saved_logits']['fields']['X']['equal']
    assert variants['gpu-saved_logits']['fields']['E_upper']['equal']
    for v in variants.values():
        assert all(v['fields'][k]['equal'] for k in ['atom_logits', 'bond_logits', 'key_data'])
        pf = v['proof']; assert hashlib.sha256(files[pf['path']].read_bytes()).hexdigest() == pf['sha256']
    capture = read('diagnostic/capture/payload/assessment.json')
    raw_path = folder / 'payload/diagnostic/native/payload/assessment.json'
    copied = folder / 'payload/diagnostic/capture/payload/endpoint-diagnostics.json'
    diagnostic = json.loads(raw_path.read_bytes())
    assert raw_path.read_bytes() == copied.read_bytes()
    assert hashlib.sha256(copied.read_bytes()).hexdigest() == capture['diagnostic_report']['sha256'] == '2045d6e8537ff86c176e6619725d206b32029058df351ce639cc88f90da9c1db'
    assert capture['status'] == 'passed' and capture['scientific_passed'] is False
    assert (capture['native_job'], capture['job_id'], capture['capture_job']) == ('29779177', '29792158', '29792159')
    assert capture['report_source_manifest_sha256'] == diagnostic['diagnostic_manifest_sha256'] == '6d547929ce3227302729b022628cc26612d9829e72c0a70861870023463dac3a'
    assert capture['cpu_certificate_sha256'] == '5d31b9a130f5a904c5da4b0f2e8707c610ff49654acddf1b89b8945aa05cba6d'
    assert diagnostic['status'] == 'diagnostics_completed_original_failure_preserved'
    assert diagnostic['can_certify_or_restart_original_study'] is False
    assert diagnostic['selected_saved_steps'] == [0, 9900, 10000]
    assert diagnostic['frozen_before'] == diagnostic['frozen_after']
    for proof in capture['proofs']:
        assert index['remote_reference_rehash'][proof['path']]['sha256'] == proof['sha256']
    for point in diagnostic['points']:
        assert point['frozen_after_point'] == diagnostic['frozen_before']
        for proof in point['proofs']:
            assert proof['path'] in files
            assert hashlib.sha256(files[proof['path']].read_bytes()).hexdigest() == proof['sha256']
    terminal = next(p for p in diagnostic['points'] if p['step'] == 10000)
    direction = terminal['directions'][0]
    assert (direction['field'], direction['row'], direction['source'], direction['target']) == ('X', 3, 1, 4)
    assert all(not p['geometry']['usable_diagnostic_points'] and p['evaluation'] is None
               for p in direction['probes'] if p['epsilon'] in [0.01, 0.003, 0.001])
    result = {'status': 'passed', 'scope': 'Independent stdlib byte/schema/scheduler/LSF/proof-chain audit only; no local array calculations',
              'captured_utc': index['captured_utc'], 'files': len(files), 'remote_reference_hashes_verified': len(index['remote_reference_rehash']),
              'remote_only_references': [p for p, x in index['remote_reference_rehash'].items() if not x['downloaded']],
              'probe_jobs': ['29792346', '29792347'], 'diagnostic_job': '29792158', 'capture_job': '29792159',
              'original_scientific_passed': False, 'original_studies_modified': False,
              'diagnostic_report_sha256': capture['diagnostic_report']['sha256'],
              'terminal_atom_symmetric_margin_reported': direction['probes'][0]['geometry']['symmetric_pair_margin'],
              'terminal_atom_diagnostic_only_probes': [{'epsilon': p['epsilon'], 'evaluation': p['evaluation']} for p in direction['probes']],
              'evidence_index_sha256': hashlib.sha256((folder / 'evidence-index.json').read_bytes()).hexdigest()}
    target = folder / 'verification.json'
    with target.open('xb') as f:
        f.write(json.dumps(result, indent=2).encode() + b'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'terminal_atom_diagnostic_only_probes'}, indent=2))


if __name__ == '__main__':
    main()
