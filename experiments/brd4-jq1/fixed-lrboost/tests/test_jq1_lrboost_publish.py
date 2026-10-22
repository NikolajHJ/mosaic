"""Allocated stdlib fixtures for immutable, byte-only fixed-report delivery."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(_layout_path(_repository_root(__file__) / 'scripts')))
import publish_jq1_lrboost_report as publisher


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
                 'readiness_status': 'passed', 'terminal_status': 'pending', 'terminal_error': None,
                 'overall_status': 'running' if index == 0 else 'queued',
                 'budget_status': 'partial' if index == 0 else 'not_started',
                 'scientific_passed': False, 'all_terminal': False,
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
    data.update(publisher.outcome_scope(studies, False))
    data['diagnostics'] = [{'arm': arm, 'original_job': pin['job'], 'status': 'pending'} for arm, pin in publisher.ARMS.items()]
    if change:
        change(data)
    (report / 'report.html').write_text('<!doctype html><script id="manifest" type="application/json">' + json.dumps(data) + '</script>', encoding='utf-8')
    (report / 'loss-overlays.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding='utf-8')
    bindings = [{k: study[k] for k in ('arm', 'job_id', 'learning_rate', 'selection_policy', 'native_status', 'source_manifest_sha256', 'config_sha256', 'readiness_sha256', *publisher.OUTCOME_FIELDS)} for study in studies]
    receipt = {'version': publisher.VERSION, 'report_version': publisher.VERSION, 'input_jobs': [s['job_id'] for s in studies],
               'n_atoms': 25, 'complete': data['complete'], 'chunk_size': 100, 'frame_stride_floats': 1650,
               **{k: data[k] for k in ('presentation_complete', 'all_terminal', 'scientific_passed', 'strict_scientific_campaign_passed', 'overall_success', 'campaign_status')},
               'states': [s['states'] for s in studies], 'included_updates': [s['latest_update'] for s in studies],
               'milestone_progress': {s['arm']: 0 for s in studies}, 'arm_bindings': bindings,
               'diagnostics': data['diagnostics'],
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
                                     'producer_job_id': '29790000', 'tests': 1, 'suite': 'jq1-lrboost-saved-report-contract-v1', 'proofs': proofs})
    put(cpu_dir / 'bundled-proofs.json', [{'original_path': p['path'], 'local_path': 'cpu-proof/evidence/' + Path(p['path']).name, 'sha256': p['sha256']} for p in proofs])
    put(payload / 'assessment.json', {'status': 'passed', 'job_id': '29790001', 'report_source_manifest_sha256': publisher.digest(payload / 'report-source-manifest.json'),
                                      'cpu_certificate_sha256': publisher.digest(cpu_dir / 'assessment.json'), 'complete': data['complete'],
                                      'input_jobs': receipt['input_jobs'], 'states': receipt['states'], 'included_updates': receipt['included_updates'],
                                      'n_atoms': 25, 'scientific_passed': receipt['scientific_passed'], 'campaign_status': receipt['campaign_status'],
                                      'report_receipt_sha256': publisher.digest(report / 'report_receipt.json')})
    repack(root)
    return report, data


def test_publish_keeps_exact_served_bytes_and_explicit_empty_arms(tmp_path):
    extracted = tmp_path / 'evidence'
    report, data = saved(extracted)
    viewer = tmp_path / 'fixed-viewer'
    latest = publisher.publish(extracted, viewer, 'edition-0001')
    assert latest['input_jobs'] == ['29779177', '29779178', '29779179']
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
    for viewer in (tmp_path / '20261021-brd4-jq1-25-report/viewer', tmp_path / '20261021-brd4-jq1-25-fixed-report/viewer', tmp_path / 'another-existing-viewer'):
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
    template = (_layout_path(_repository_root(__file__) / 'scripts/templates/jq1_lrboost_report.html')).read_text(encoding='utf-8')
    assert 'option.disabled=s.states===0' not in template
    assert 'No saved snapshot' in template and 'clearFrame()' in template
    assert "byId('step').max=available?study.states-1:0" in template
    assert 'if(!hasFrames(s)||!Number.isFinite(value))return' in template
    assert 'stop();++request;clearFrame()' in template
    assert 'Fixed LR: Protenix 3.0; both Boltz-2 arms 0.05' in template
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


def test_failed_science_can_be_published_as_honest_saved_presentation(tmp_path):
    def change(data):
        study = data['studies'][0]
        study.update(native_status='FAILED_PERSISTED', overall_status='failed',
                     terminal_status='failed', terminal_error='Saved native terminal FD failed',
                     all_terminal=True)
        data.update(publisher.outcome_scope(data['studies'], False))
    extracted = tmp_path / 'evidence'
    _, data = saved(extracted, change)
    latest = publisher.publish(extracted, tmp_path / 'higher-rate-viewer', 'edition-0001')
    assert latest['campaign_status'] == 'failed' and not latest['scientific_passed']
    assert not latest['complete'] and not latest['all_terminal']
    assert data['studies'][0]['native_status'] == 'FAILED_PERSISTED'


def test_final_presentation_may_preserve_failed_partial_and_unstarted_outcomes(tmp_path):
    def change(data):
        for study in data['studies']:
            study.update(native_status='FAILED_PERSISTED', overall_status='failed',
                         terminal_status='not_reached', terminal_error=None, all_terminal=True)
        data['complete'] = True
        data.update(publisher.outcome_scope(data['studies'], True))
    extracted = tmp_path / 'evidence'
    saved(extracted, change)
    latest = publisher.publish(extracted, tmp_path / 'higher-rate-viewer', 'edition-final')
    assert latest['complete'] and latest['presentation_complete'] and latest['all_terminal']
    assert latest['campaign_status'] == 'failed' and not latest['scientific_passed']
    assert latest['states'] == [2, 0, 0]


@pytest.mark.parametrize('damage', ['science_success', 'hide_failure', 'full_budget', 'terminal_missing_error', 'all_terminal'])
def test_presentation_fields_cannot_promote_or_hide_scientific_failure(tmp_path, damage):
    def change(data):
        study = data['studies'][0]
        if damage == 'science_success': study['scientific_passed'] = True
        elif damage == 'hide_failure': study.update(terminal_status='failed', terminal_error='FD failed')
        elif damage == 'full_budget': study['budget_status'] = 'full'
        elif damage == 'terminal_missing_error': study.update(overall_status='failed', terminal_status='failed')
        elif damage == 'all_terminal': study['all_terminal'] = True
    extracted = tmp_path / 'evidence'
    saved(extracted, change)
    with pytest.raises(ValueError):
        publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')


def test_new_viewer_marker_cannot_overwrite_original_fixed_family(tmp_path):
    extracted = tmp_path / 'evidence'
    saved(extracted)
    viewer = tmp_path / 'viewer'
    viewer.mkdir()
    put(viewer / 'viewer-kind.json', {'version': 'jq1-fixed-step-viewer-v1', 'report_version': 'jq1-25-fixed-saved-report-v1'})
    old = viewer / 'report.html'
    old.write_text('preserve old fixed report', encoding='utf-8')
    with pytest.raises(ValueError):
        publisher.publish(extracted, viewer, 'edition-0001')
    assert old.read_text(encoding='utf-8') == 'preserve old fixed report'


def test_template_does_not_equate_presentation_completion_to_scientific_success():
    template = (_layout_path(_repository_root(__file__) / 'scripts/templates/jq1_lrboost_report.html')).read_text(encoding='utf-8')
    assert 'PRESENTATION COMPLETE' in template and 'scientific campaign' in template
    assert 'terminal_error' in template and 'scientific_passed' in template
    assert 'all three authenticated 10,000-update trajectories' not in template
    assert 'Fixed LR: Protenix 3.0; both Boltz-2 arms 0.05' in template


def attach_diagnostic_fixture(extracted, report, data, *, damage=None):
    arm, pin = next(iter(publisher.ARMS.items()))
    directory = report / 'diagnostics'
    directory.mkdir()
    rawpath = directory / (arm + '-report.json')
    raw = {'version': 'jq1-lrboost-endpoint-diagnostics-v1', 'arm': arm,
           'status': 'diagnostics_completed_original_failure_preserved',
           'original_job': pin['job'], 'diagnostic_job': '29790020',
           'scientific_manifest_sha256': pin['manifest'],
           'diagnostic_manifest_sha256': 'd' * 64,
           'can_certify_or_restart_original_study': False, 'points': []}
    if damage == 'promote': raw['can_certify_or_restart_original_study'] = True
    elif damage == 'original_job': raw['original_job'] = '29770179'
    put(rawpath, raw)
    capturepath = directory / (arm + '-capture.json')
    capture = {'status': 'passed', 'arm': arm, 'native_job': pin['job'], 'job_id': '29790021',
               'report_source_manifest_sha256': publisher.digest(report.parent / 'report-source-manifest.json'),
               'cpu_certificate_sha256': publisher.digest(report.parent / 'cpu-proof/assessment.json'),
               'diagnostic_report': {'path': '/fixture/endpoint-diagnostics.json',
                                     'sha256': publisher.digest(rawpath), 'bytes': rawpath.stat().st_size}}
    if damage == 'capture_source': capture['report_source_manifest_sha256'] = '0' * 64
    put(capturepath, capture)
    entry = {'arm': arm, 'original_job': pin['job'], 'status': 'available',
             'diagnostic_status': raw['status'], 'diagnostic_job': raw['diagnostic_job'],
             'assets': {kind: {'path': path.relative_to(report).as_posix(),
                               'sha256': publisher.digest(path), 'bytes': path.stat().st_size}
                        for kind, path in [('report', rawpath), ('capture', capturepath)]}}
    data['diagnostics'][0] = entry
    if damage == 'asset_hash': rawpath.write_text('changed bytes', encoding='utf-8')
    (report / 'report.html').write_text('<!doctype html><script id="manifest" type="application/json">' + json.dumps(data) + '</script>', encoding='utf-8')
    receipt = publisher.read(report / 'report_receipt.json')
    receipt.update(diagnostics=data['diagnostics'], html_bytes=(report / 'report.html').stat().st_size,
                   html_sha256=publisher.digest(report / 'report.html'))
    put(report / 'report_receipt.json', receipt)
    assessment = publisher.read(report.parent / 'assessment.json')
    assessment['report_receipt_sha256'] = publisher.digest(report / 'report_receipt.json')
    put(report.parent / 'assessment.json', assessment)
    repack(extracted)
    return rawpath.read_bytes()


def test_optional_diagnostic_json_is_published_verbatim_without_native_pass(tmp_path):
    extracted = tmp_path / 'evidence'
    report, data = saved(extracted)
    expected = attach_diagnostic_fixture(extracted, report, data)
    viewer = tmp_path / 'viewer'
    latest = publisher.publish(extracted, viewer, 'edition-0001')
    assert (viewer / 'edition-0001/diagnostics/protenix-report.json').read_bytes() == expected
    assert not latest['scientific_passed'] and latest['campaign_status'] == 'in_progress'


@pytest.mark.parametrize('damage', ['promote', 'original_job', 'capture_source', 'asset_hash'])
def test_optional_diagnostics_cannot_change_original_run_or_proof_scope(tmp_path, damage):
    extracted = tmp_path / 'evidence'
    report, data = saved(extracted)
    attach_diagnostic_fixture(extracted, report, data, damage=damage)
    with pytest.raises(ValueError):
        publisher.publish(extracted, tmp_path / 'viewer', 'edition-0001')


@pytest.mark.parametrize('native_status,passed', [('SUCCESS_PERSISTED', True), ('FAILED_PERSISTED', False)])
def test_persisted_native_outcomes_keep_explicit_scope_without_fabricated_done(native_status, passed):
    study = {'native_status': native_status, 'readiness_status': 'passed',
             'terminal_status': 'passed' if passed else 'failed',
             'terminal_error': None if passed else 'Original terminal derivative gate failed',
             'overall_status': 'passed' if passed else 'failed', 'budget_status': 'full',
             'scientific_passed': passed, 'all_terminal': True, 'states': 10001, 'latest_update': 10000}
    scope = publisher.outcome_scope([study] * 3, True)
    assert scope['presentation_complete'] and scope['all_terminal']
    assert scope['scientific_passed'] is passed
    assert scope['campaign_status'] == ('passed' if passed else 'failed')
    assert study['native_status'] == native_status


def test_persisted_failure_alias_cannot_be_relabelled_scientific_success():
    study = {'native_status': 'FAILED_PERSISTED', 'readiness_status': 'passed',
             'terminal_status': 'passed', 'terminal_error': None, 'overall_status': 'passed',
             'budget_status': 'full', 'scientific_passed': True, 'all_terminal': True,
             'states': 10001, 'latest_update': 10000}
    with pytest.raises(ValueError, match='Scientific success'):
        publisher.outcome_scope([study] * 3, True)
