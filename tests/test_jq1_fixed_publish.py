"""Allocated stdlib fixtures for immutable, byte-only fixed-report delivery."""

import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import publish_jq1_fixed_report as publisher


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Report fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(',', ':')) + '\n', encoding='utf-8')


def repack(root):
    manifest = root / 'evidence_manifest.json'
    manifest.unlink(missing_ok=True)
    files = {path.relative_to(root).as_posix(): {'bytes': path.stat().st_size, 'sha256': publisher.digest(path)}
             for path in sorted(root.rglob('*')) if path.is_file()}
    put(manifest, {'files': files})
    archive = root.with_name(root.name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as handle:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                handle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    put(root.with_name(root.name + '-download.json'), {'archive': str(archive), 'sha256': publisher.digest(archive), 'verified_files': len(files), 'scope': 'stdlib byte integrity only; no local numerical evaluation'})


def saved(root, change=None):
    report = root / 'interim-0001/payload/report'
    payload = report.parent
    report.mkdir(parents=True)
    studies = []
    for index, (arm, pin) in enumerate(publisher.ARMS.items()):
        entry = {'arm': arm, 'job_id': pin['job'], 'learning_rate': pin['rate'], 'selection_policy': 'fixed_step',
                 'source_manifest_sha256': pin['manifest'], 'config_sha256': pin['config'], 'readiness_sha256': pin['ready'],
                 'native_status': 'RUN' if index == 0 else 'PEND', 'status': 'running' if index == 0 else 'queued',
                 'states': 2 if index == 0 else 0, 'latest_update': 1 if index == 0 else None, 'chunks': [],
                 'counts': {'applied': 1, 'forced_fallback': 0, 'strict_improving': 0, 'scalar_increased': 1, 'scalar_equal': 0, 'scalar_decreased': 0, 'no_op': 0} if index == 0 else {},
                 'progress_evidence': {'state': 'Saved fixture only'}}
        if index == 0:
            destination = report / arm
            destination.mkdir()
            (destination / 'frames-00000.bin').write_bytes(b'\x00' * (2 * 1650 * 4))
            put(destination / 'frames-00000.json', [{'step': 0}, {'step': 1}])
            put(destination / 'curves.json', [{'step': 0}, {'step': 1}])
            entry['chunks'] = [{'start': 0, 'count': 2, 'binary': 'frames-00000.bin', 'binary_sha256': publisher.digest(destination / 'frames-00000.bin'), 'metadata': 'frames-00000.json', 'metadata_sha256': publisher.digest(destination / 'frames-00000.json')}]
        studies.append(entry)
    data = {'version': publisher.VERSION, 'selection_policy': 'fixed_step', 'n_atoms': 25, 'frame_stride_floats': 1650, 'chunk_size': 100, 'complete': False, 'studies': studies}
    if change:
        change(data)
    (report / 'report.html').write_text('<!doctype html><script id="manifest" type="application/json">' + json.dumps(data) + '</script>', encoding='utf-8')
    (report / 'loss-overlays.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding='utf-8')
    bindings = [{k: study[k] for k in ('arm', 'job_id', 'learning_rate', 'selection_policy', 'native_status', 'source_manifest_sha256', 'config_sha256', 'readiness_sha256')} for study in studies]
    receipt = {'version': publisher.VERSION, 'report_version': publisher.VERSION, 'input_jobs': [s['job_id'] for s in studies],
               'n_atoms': 25, 'complete': data['complete'], 'chunk_size': 100, 'frame_stride_floats': 1650,
               'states': [s['states'] for s in studies], 'included_updates': [s['latest_update'] for s in studies],
               'milestone_progress': {s['arm']: 0 for s in studies}, 'arm_bindings': bindings,
               'html_bytes': (report / 'report.html').stat().st_size, 'html_sha256': publisher.digest(report / 'report.html')}
    put(report / 'report_receipt.json', receipt)
    put(payload / 'report-source-manifest.json', {'fixture': 'saved publication bytes only'})
    cpu_dir = payload / 'cpu-proof'
    proof_dir = cpu_dir / 'evidence'
    proof_dir.mkdir(parents=True)
    (proof_dir / 'report-tests.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"><testcase name="fixed_report_fixture"/></testsuite></testsuites>', encoding='utf-8')
    (proof_dir / 'scheduler-final-1.txt').write_text('Job <29790000>, Status <DO\n                          NE>, Que\n                          ue <hp\n                          c>', encoding='utf-8')
    (proof_dir / 'stdout.log').write_text('Subject: Job 29790000: report fixture\nSuccessfully completed.\n', encoding='utf-8')
    proofs = [{'path': '/fixture/report/cpu/' + p.name, 'sha256': publisher.digest(p)} for p in sorted(proof_dir.iterdir())]
    put(cpu_dir / 'assessment.json', {'status': 'passed', 'source_manifest_sha256': publisher.digest(payload / 'report-source-manifest.json'),
                                     'producer_job_id': '29790000', 'tests': 1, 'suite': 'jq1-fixed-saved-report-contract-v1', 'proofs': proofs})
    put(cpu_dir / 'bundled-proofs.json', [{'original_path': p['path'], 'local_path': 'cpu-proof/evidence/' + Path(p['path']).name, 'sha256': p['sha256']} for p in proofs])
    put(payload / 'assessment.json', {'status': 'passed', 'job_id': '29790001', 'report_source_manifest_sha256': publisher.digest(payload / 'report-source-manifest.json'),
                                      'cpu_certificate_sha256': publisher.digest(cpu_dir / 'assessment.json'), 'complete': data['complete'],
                                      'input_jobs': receipt['input_jobs'], 'states': receipt['states'], 'included_updates': receipt['included_updates'],
                                      'n_atoms': 25, 'report_receipt_sha256': publisher.digest(report / 'report_receipt.json')})
    repack(root)
    return report, data


def test_publish_keeps_exact_served_bytes_and_explicit_empty_arms(tmp_path):
    extracted = tmp_path / 'evidence'
    report, data = saved(extracted)
    viewer = tmp_path / 'fixed-viewer'
    latest = publisher.publish(extracted, viewer, 'edition-0001')
    assert latest['input_jobs'] == ['29777709', '29778203', '29778204']
    assert latest['states'] == [2, 0, 0] and latest['included_updates'] == [1, None, None]
    assert not latest['complete'] and publisher.read(viewer / 'latest.json') == latest
    for source in report.rglob('*'):
        if source.is_file():
            assert (viewer / 'edition-0001' / source.relative_to(report)).read_bytes() == source.read_bytes()
    assert 'setInterval(refresh,60000)' in (viewer / 'report.html').read_text(encoding='utf-8')
    assert data['studies'][1]['chunks'] == []


def test_existing_edition_and_previous_pointer_are_immutable(tmp_path):
    extracted = tmp_path / 'evidence'
    saved(extracted)
    viewer = tmp_path / 'fixed-viewer'
    publisher.publish(extracted, viewer, 'edition-0001')
    before = {p.relative_to(viewer).as_posix(): p.read_bytes() for p in viewer.rglob('*') if p.is_file()}
    with pytest.raises(ValueError, match='already exists'):
        publisher.publish(extracted, viewer, 'edition-0001')
    assert before == {p.relative_to(viewer).as_posix(): p.read_bytes() for p in viewer.rglob('*') if p.is_file()}
    publisher.publish(extracted, viewer, 'edition-0002')
    for name, content in before.items():
        if name.startswith('edition-0001/'):
            assert (viewer / name).read_bytes() == content
    assert publisher.read(viewer / 'latest.json')['path'] == 'edition-0002/report.html'


@pytest.mark.parametrize('damage', ['old_jobs', 'rate', 'policy', 'source', 'config', 'ready', 'sparse', 'count', 'fallback', 'empty_frame', 'fake_final', 'chunk_start', 'chunk_count', 'asset_escape'])
def test_wrong_current_science_or_frame_schema_never_publishes(tmp_path, damage):
    def change(data):
        study = data['studies'][0]
        if damage == 'old_jobs': study['job_id'] = '29770179'
        elif damage == 'rate': study['learning_rate'] = .03
        elif damage == 'policy': study['selection_policy'] = 'best_primal'
        elif damage == 'source': study['source_manifest_sha256'] = '0' * 64
        elif damage == 'config': study['config_sha256'] = '0' * 64
        elif damage == 'ready': study['readiness_sha256'] = '0' * 64
        elif damage == 'sparse': study['snapshot_steps'] = [0, 500]
        elif damage == 'count': study['latest_update'] = 2
        elif damage == 'fallback': study['counts']['forced_fallback'] = 1
        elif damage == 'empty_frame': data['studies'][1]['latest_update'] = 0
        elif damage == 'fake_final': data['complete'] = True
        elif damage == 'chunk_start': study['chunks'][0]['start'] = 1
        elif damage == 'chunk_count': study['chunks'][0]['count'] = 101
        elif damage == 'asset_escape': study['chunks'][0]['binary'] = '../outside.bin'
    extracted = tmp_path / 'evidence'
    saved(extracted, change)
    viewer = tmp_path / 'fixed-viewer'
    with pytest.raises((ValueError, FileNotFoundError)):
        publisher.publish(extracted, viewer, 'edition-0001')
    assert not viewer.exists()


@pytest.mark.parametrize('edition', ['../edition-0001', 'edition-0001/elsewhere', 'edition-NEW', 'edition-', 'edition-00001', 'edition-final/other', 'edition-0001\\other'])
def test_unsafe_edition_name_is_rejected_before_writing(tmp_path, edition):
    with pytest.raises(ValueError, match='safe immutable'):
        publisher.publish(tmp_path / 'missing', tmp_path / 'viewer', edition)
    assert not (tmp_path / 'viewer').exists()


def test_historical_or_unmarked_existing_viewer_is_preserved(tmp_path):
    extracted = tmp_path / 'evidence'
    saved(extracted)
    for viewer in (tmp_path / '20261021-brd4-jq1-25-report/viewer', tmp_path / 'another-existing-viewer'):
        viewer.mkdir(parents=True)
        old = viewer / 'report.html'
        old.write_text('preserve historical report', encoding='utf-8')
        with pytest.raises(ValueError):
            publisher.publish(extracted, viewer, 'edition-0001')
        assert old.read_text(encoding='utf-8') == 'preserve historical report'
        assert not (viewer / 'latest.json').exists()


@pytest.mark.parametrize('damage', ['asset', 'extra', 'manifest', 'archive', 'marker'])
def test_modified_or_unverified_download_is_rejected(tmp_path, damage):
    extracted = tmp_path / 'evidence'
    report, _ = saved(extracted)
    if damage == 'asset': (report / 'loss-overlays.svg').write_text('changed', encoding='utf-8')
    elif damage == 'extra': (report / 'extra.html').write_text('unverified', encoding='utf-8')
    elif damage == 'manifest': put(extracted / 'evidence_manifest.json', {'files': {}})
    elif damage == 'archive': extracted.with_name('evidence.tar.gz').write_bytes(b'changed archive')
    elif damage == 'marker': extracted.with_name('evidence-download.json').unlink()
    with pytest.raises(ValueError):
        publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')
    assert not (tmp_path / 'viewer').exists()


@pytest.mark.parametrize('damage', ['assessment', 'source', 'certificate', 'cpu_failed', 'cpu_skip', 'cpu_count', 'cpu_scheduler', 'cpu_lsf', 'proof_path', 'proof_digest'])
def test_scientific_assessment_and_allocated_cpu_proofs_remain_mandatory(tmp_path, damage):
    extracted = tmp_path / 'evidence'
    report, _ = saved(extracted)
    payload, cpu_dir = report.parent, report.parent / 'cpu-proof'
    if damage == 'assessment':
        value = publisher.read(payload / 'assessment.json');value['status'] = 'failed';put(payload / 'assessment.json', value)
    elif damage == 'source':
        put(payload / 'report-source-manifest.json', {'different': True})
    elif damage == 'certificate':
        value = publisher.read(cpu_dir / 'assessment.json');value['suite'] = 'old-backtracking-fixtures';put(cpu_dir / 'assessment.json', value)
    elif damage == 'cpu_failed':
        (cpu_dir / 'evidence/report-tests.xml').write_text('<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>', encoding='utf-8')
    elif damage == 'cpu_skip':
        (cpu_dir / 'evidence/report-tests.xml').write_text('<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>', encoding='utf-8')
    elif damage == 'cpu_count':
        value = publisher.read(cpu_dir / 'assessment.json');value['tests'] = 2;put(cpu_dir / 'assessment.json', value)
    elif damage == 'cpu_scheduler':
        (cpu_dir / 'evidence/scheduler-final-1.txt').write_text('Job <29790000>, Status <RUN>, Queue <hpc>', encoding='utf-8')
    elif damage == 'cpu_lsf':
        (cpu_dir / 'evidence/stdout.log').write_text('Subject: Job 29790000\nExited with exit code 1.', encoding='utf-8')
    elif damage == 'proof_path':
        value = publisher.read(cpu_dir / 'bundled-proofs.json');value[0]['local_path'] = '../escaped';put(cpu_dir / 'bundled-proofs.json', value)
    elif damage == 'proof_digest':
        value = publisher.read(cpu_dir / 'bundled-proofs.json');value[0]['sha256'] = '0' * 64;put(cpu_dir / 'bundled-proofs.json', value)
    if damage in ('cpu_failed', 'cpu_skip', 'cpu_count', 'cpu_scheduler', 'cpu_lsf', 'certificate'):
        # Rebind changed fixture bytes so this exercises the semantic gate,
        # rather than merely failing the earlier byte-integrity check.
        cpu = publisher.read(cpu_dir / 'assessment.json')
        for proof in cpu['proofs']:
            proof['sha256'] = publisher.digest(cpu_dir / 'evidence' / Path(proof['path']).name)
        put(cpu_dir / 'assessment.json', cpu)
        put(cpu_dir / 'bundled-proofs.json', [{'original_path': proof['path'], 'local_path': 'cpu-proof/evidence/' + Path(proof['path']).name, 'sha256': proof['sha256']} for proof in cpu['proofs']])
        assessment = publisher.read(payload / 'assessment.json')
        assessment['cpu_certificate_sha256'] = publisher.digest(cpu_dir / 'assessment.json')
        put(payload / 'assessment.json', assessment)
    repack(extracted)
    with pytest.raises((ValueError, FileNotFoundError)):
        publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')
    assert not (tmp_path / 'viewer').exists()


def test_embedded_and_receipt_prefixes_must_match(tmp_path):
    extracted = tmp_path / 'evidence'
    report, _ = saved(extracted)
    value = publisher.read(report / 'report_receipt.json');value['states'] = [10001, 0, 0];put(report / 'report_receipt.json', value)
    repack(extracted)
    with pytest.raises(ValueError, match='prefixes'):
        publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')


def test_fixed_template_keeps_all_arm_selection_and_empty_view_guards():
    template = (Path(__file__).resolve().parents[1] / 'scripts/templates/jq1_fixed_report.html').read_text(encoding='utf-8')
    assert 'option.disabled=s.states===0' not in template
    assert 'No saved snapshot' in template and 'clearFrame()' in template
    assert "byId('step').max=available?study.states-1:0" in template
    assert 'if(!hasFrames(s)||!Number.isFinite(value))return' in template
    assert 'stop();++request;clearFrame()' in template
    assert 'Fixed LR: Protenix 0.3; both Boltz-2 arms 0.01' in template
    assert 'scalar_increased' in template and 'scalar_equal' in template and 'no_op' in template
    assert 'Latest captured' in template and 'probabilities and losses move' in template


def test_convenient_duplicate_xml_copy_does_not_double_count_original_proofs(tmp_path):
    extracted = tmp_path / 'evidence'
    report, _ = saved(extracted)
    cpu_dir = report.parent / 'cpu-proof'
    (cpu_dir / 'report-tests.xml').write_bytes((cpu_dir / 'evidence/report-tests.xml').read_bytes())
    repack(extracted)
    latest = publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')
    assert latest['states'] == [2, 0, 0]
