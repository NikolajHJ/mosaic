"""Independent saved-metadata/byte audit; no scientific array evaluation."""
import hashlib
import json
from pathlib import Path
import re
import tarfile


def main():
    repo = Path(__file__).resolve().parents[5]
    folder = repo / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-03/news-20261022-2012'
    target = folder / 'verification.json'
    if target.exists():
        raise FileExistsError(target)
    index_path = folder / 'evidence-index.json'
    index = json.loads(index_path.read_bytes())
    files = {}
    for row in index['files']:
        p = folder / row['local_file']
        b = p.read_bytes()
        assert len(b) == row['bytes'] and hashlib.sha256(b).hexdigest() == row['sha256']
        files[row['remote_path']] = p
    assert len(files) == 108
    proofs = {p['path']: p for p in index['remote_proof_rehash']}
    assert len(proofs) == 188 and all(p['matches'] and p['expected_sha256'] == p['actual_sha256'] for p in proofs.values())
    root = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-03'
    manifest = 'ff82857ecd4a985d0ba6cc16a87f1bf7b7bb2c7ffdbde18a183bbd63670ff567'

    def read(path):
        return json.loads(files[path].read_bytes())

    def terminal(parent, job, status):
        lsf = files[parent + '/stdout.' + job + '.log'].read_text()
        assert 'Subject: Job ' + job + ':' in lsf
        assert ('Successfully completed.' in lsf) if status == 'DONE' else ('Exited with exit code 1.' in lsf)
        blocks = [b for b in re.split(r'(?=Job <\d+>)', index['scheduler']['stdout'])
                  if 'Job<' + job + '>' in re.sub(r'\s+', '', b)]
        assert len(blocks) == 1
        c = re.sub(r'\s+', '', blocks[0])
        assert re.findall(r'Status<([^>]+)>', c) == [status]
        assert re.findall(r'Queue<([^>]+)>', c) == ['gpul40s']

    archive = repo / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-03/source01/source.tar.gz'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '31809b975f0387db4ad15b6530c443afc1220fd501b6b8ac62e147bbcf739b01'
    with tarfile.open(archive, 'r:gz') as t:
        def source_bytes(relative):
            matches = [m for m in t.getmembers() if m.name == relative or m.name.endswith('/' + relative)]
            assert len(matches) == 1 and matches[0].isfile()
            return t.extractfile(matches[0]).read()
        configs = {a: source_bytes('examples/configs/brd4_jq1_25_fixed_lrreset_' + a + '_10000.json')
                   for a in ('protenix', 'boltz2', 'boltz2_chemistry')}
        driver = source_bytes('examples/brd4_25_atoms.py').decode()
    assert driver.index('np.testing.assert_array_equal(previous[field], current[field])') < driver.index('report["native_gradient_checks"] = raw_check(')
    passed = {}
    for arm, job, count in [('boltz2', '29792129', 52), ('boltz2_chemistry', '29792131', 128)]:
        parent = root + '/validate-' + arm
        decision = read(root + '/decision-' + arm + '.json')
        assessment = read(parent + '/payload/study/assessment.json')
        summary = read(parent + '/payload/study/summary.json')
        own = read(parent + '/submission.json')
        terminal(parent, job, 'DONE')
        assert own['job_id'] == decision['native_job'] == summary['job_id'] == job
        assert decision['status'] == assessment['status'] == summary['status'] == 'passed'
        assert summary['mode'] == 'validate' and summary['source_manifest_sha256'] == assessment['source_manifest_sha256'] == own['source_manifest_sha256'] == manifest
        assert summary['config'] == json.loads(configs[arm])
        assert summary['config_sha256'] == assessment['config_sha256'] == own['config_sha256'] == hashlib.sha256(configs[arm]).hexdigest()
        assert summary['optimization_parameters'] == {'step_size': 0.005, 'max_backtracks': 1, 'acceptance_policy': 'fixed_step'}
        assert summary['native_hard_parity']['status'] == summary['native_gradient_checks']['status'] == 'passed'
        assert summary['frozen_identity_after']['status'] == assessment['runtime_gate']['status'] == 'passed'
        assert summary['frozen_identity_after']['model_digest'] == summary['model_digest'] == assessment['model_digest']
        assert summary['frozen_identity_after']['context_digest'] == summary['context_digest'] == assessment['context_digest']
        assert summary['diffusion_gradient'] is assessment['diffusion_gradient'] is False
        assert len(assessment['proofs']) == count
        for proof in assessment['proofs'] + decision['proofs']:
            assert proofs[proof['path']]['actual_sha256'] == proof['sha256']
        for pilot in ['early_pilot'] + (['high_pilot'] if arm == 'boltz2_chemistry' else []):
            p = read(parent + '/payload/study/' + pilot + '/report.json')
            assert p['status'] == 'passed' and p['counts']['applied'] == p['requested_updates'] == len(p['updates']) == 10
            assert len(p['states']) == 11 and p['acceptance_policy'] == 'fixed_step' and p['nominal_step_size'] == 0.005
        if arm == 'boltz2_chemistry':
            assert summary['chemistry_assay']['status'] == 'passed'
            for name, weight in [('low_coupled_checks', 0.001), ('high_coupled_checks', 100.0)]:
                p = read(parent + '/payload/study/' + name + '/report.json')
                assert p['status'] == 'passed' and p['repair_weight'] == weight
        passed[arm] = {'native_job': job, 'proofs_rehashed_on_hpc': count,
                       'assessment_sha256': decision['assessment_sha256']}
    pro = read(root + '/validate-protenix/payload/study/summary.json')
    terminal(root + '/validate-protenix', '29792127', 'EXIT')
    assert pro['status'] == read(root + '/decision-protenix.json')['status'] == 'failed'
    assert pro['job_id'] == '29792127' and 'Mismatched elements: 150 / 150' in pro['error']
    assert 'Max absolute difference among violations: 5.9604645e-08' in pro['error']
    assert 'native_gradient_checks' not in pro
    assert not index['long_submissions_exists'] and not index['long_dispatch_marker_exists']
    for job in ['29792158', '29792159']:
        blocks = [b for b in re.split(r'(?=Job <\d+>)', index['scheduler']['stdout'])
                  if 'Job<' + job + '>' in re.sub(r'\s+', '', b)]
        assert len(blocks) == 1 and 'Status<PEND>' in re.sub(r'\s+', '', blocks[0])
    report = {'status': 'passed', 'scope': 'Independent saved bytes, JSON metadata, actual scheduler/LSF and immutable source audit; no numerical re-evaluation',
              'captured_utc': index['captured_utc'], 'metadata_files': len(files),
              'hpc_remote_proof_hashes_matched': len(proofs), 'boltz_readiness_passed': passed,
              'protenix_readiness': 'failed exact initialization identity before gradient checks',
              'long_runs_submitted': False, 'corrected_protenix_diagnostic': '29792158 PEND; capture29792159 PEND',
              'evidence_index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
              'raw_npz_proofs_downloaded': False, 'original_studies_modified': False}
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
