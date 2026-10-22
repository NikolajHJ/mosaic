"""Allocated byte/provenance fixtures for one presentation-only event fix.

Tiny fake data files isolate metadata integrity; they do not stand in for the
required original 30,003-frame render or its original278-case CPU evidence.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer_focus as focus


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Viewer-focus fixtures require allocated DTU HPC')


def html():
    return (b'<!doctype html><script id="manifest" type="application/json">'
            b'{"scientific_passed":false,"original_job":"29779177"}</script><script>\n'
            b"function stop(){playing=false;playEpoch++;byId('play').textContent='Play';}\n"
            + focus.ANCHOR + b'\n</script>')


def test_focus_fix_is_one_exact_listener_before_typing_and_keeps_manifest_bytes():
    original = html()
    fixed = focus.patch_html(original)
    assert fixed.replace(focus.FOCUS, b'', 1) == original
    assert fixed.count(focus.FOCUS) == 1
    assert fixed.index(focus.FOCUS) < fixed.index(focus.ANCHOR)
    assert fixed[fixed.index(b'<script id="manifest"'):fixed.index(b'</script>')] == original[
        original.index(b'<script id="manifest"'):original.index(b'</script>')]


@pytest.mark.parametrize('damage', ['absent', 'duplicate', 'already_fixed', 'stop_changed'])
def test_unknown_or_already_fixed_html_fails_closed(damage):
    original = html()
    if damage == 'absent':
        original = original.replace(focus.ANCHOR, b'')
    elif damage == 'duplicate':
        original += focus.ANCHOR
    elif damage == 'already_fixed':
        original = focus.patch_html(original)
    else:
        original = original.replace(b'playEpoch++;', b'playEpoch+=2;')
    with pytest.raises(ValueError):
        focus.patch_html(original)


def delta_fixture(tmp_path):
    original = tmp_path / 'original'
    report = original / 'payload/report'; report.mkdir(parents=True)
    (report / 'report.html').write_bytes(html())
    # Match the number of unchanged real served assets without fabricating any
    # scientific arrays. Each file is explicitly only a byte-identity fixture.
    for index in range(610):
        (report / ('saved-byte-fixture-' + str(index) + '.dat')).write_bytes(str(index).encode())
    old_receipt = {'html_bytes': len(html()), 'html_sha256': focus.sha(report / 'report.html'),
                   'scientific_passed': False, 'campaign_status': 'failed', 'states': [10001] * 3,
                   'included_updates': [10000] * 3, 'readiness_source': 'original-unchanged'}
    focus.put(report / 'report_receipt.json', old_receipt)
    old_assessment = {'status': 'passed', 'job_id': '29781041',
                      'report_source_manifest_sha256': focus.ANALYSIS_MANIFEST,
                      'cpu_certificate_sha256': focus.ANALYSIS_CPU,
                      'report_receipt_sha256': focus.sha(report / 'report_receipt.json'),
                      'scientific_passed': False, 'scope': 'Original saved-only allocated rendering'}
    focus.put(report.parent / 'assessment.json', old_assessment)
    proof = tmp_path / 'job-proof'; proof.mkdir()
    focus.put(proof / 'proof.json', {'original': 'raw fixture bytes'})
    cpu = tmp_path / 'cpu-proof'; cpu.mkdir()
    focus.put(cpu / 'assessment.json', {'fixture': 'separate unit writer only'})
    source = tmp_path / 'source_manifest.json'
    focus.put(source, {'files': {'viewer_focus.py': focus.sha(Path(focus.__file__))}})
    baseline = {'report': report, 'receipt': old_receipt, 'assessment': old_assessment,
                'job_proofs': focus.tree_records(proof),
                'archive_evidence_manifest_sha256': 'e' * 64}
    delta = tmp_path / 'delta'
    focus._write_delta(baseline, original, delta,
        original_pin={'job_id': '29781041', 'archive_bytes': 123, 'archive_sha256': 'a' * 64},
        original_job_proof_dir=proof, patch_source_manifest_sha256=focus.sha(source),
        patch_source_manifest_path=source, patch_job_id='29799999',
        patch_cpu_certificate_sha256=focus.sha(cpu / 'assessment.json'), patch_cpu_proof_dir=cpu)
    return report, delta


def test_tiny_delta_preserves_original610_assets_and_explicit_renderer_provenance(tmp_path):
    report, delta = delta_fixture(tmp_path)
    original_before = focus.tree_records(report.parent)
    adoption = focus.read(delta / 'focus-adoption.json')
    checked = focus.validate_delta_metadata(report, delta, adoption)
    assert checked['unchanged_served_assets'] == 610
    assert not any(path.name.startswith('saved-byte-fixture-') for path in delta.iterdir())
    assert focus.tree_records(report.parent) == original_before
    assessment = focus.read(delta / 'assessment.json')
    assert assessment['job_id'] == '29799999' and assessment['original_render_job'] == '29781041'
    assert assessment['report_source_manifest_sha256'] == focus.ANALYSIS_MANIFEST
    assert assessment['cpu_certificate_sha256'] == focus.ANALYSIS_CPU
    assert not assessment['scientific_passed'] and not assessment['scientific_render_reexecuted']
    assert focus.read(delta / 'original-assessment.json') == focus.read(report.parent / 'assessment.json')


@pytest.mark.parametrize('damage', ['asset', 'html', 'receipt_scope', 'render_cpu', 'new_cpu', 'adoption_scope', 'extra_file'])
def test_changed_science_assets_or_adoption_is_rejected(tmp_path, damage):
    report, delta = delta_fixture(tmp_path)
    adoption = focus.read(delta / 'focus-adoption.json')
    if damage == 'asset':
        (report / 'saved-byte-fixture-1.dat').write_bytes(b'changed')
    elif damage == 'html':
        (delta / 'report.html').write_bytes((delta / 'report.html').read_bytes() + b'changed')
    elif damage == 'receipt_scope':
        value = focus.read(delta / 'report_receipt.json'); value['scientific_passed'] = True
        focus._replace_json(delta / 'report_receipt.json', value)
    elif damage in ('render_cpu', 'new_cpu'):
        value = focus.read(delta / 'assessment.json')
        value['cpu_certificate_sha256' if damage == 'render_cpu' else 'patch_cpu_certificate_sha256'] = 'f' * 64
        focus._replace_json(delta / 'assessment.json', value)
    elif damage == 'adoption_scope':
        adoption['scientific_data_changed'] = True
    else:
        (delta / 'unexpected.json').write_bytes(b'{}')
    with pytest.raises(ValueError):
        focus.validate_delta_metadata(report, delta, adoption)


def cpu_fixture(tmp_path):
    folder = tmp_path / 'cpu'; evidence = folder / 'evidence'; evidence.mkdir(parents=True)
    job = '29799998'
    (evidence / 'scheduler-final-0.txt').write_text(
        'Job <' + job + '>, Status <DO\n NE>, Queue <hp\n c>', encoding='utf-8')
    (evidence / ('stdout.' + job + '.log')).write_text(
        'Subject: Job ' + job + ': focus-cpu Done\nSuccessfully completed.\n', encoding='utf-8')
    counts = dict(focus.OLD_COUNTS) | {'test_viewer_focus': 2, 'test_viewer_focus_ops': 3}
    root = ET.Element('testsuites', tests=str(sum(counts.values())), failures='0', errors='0', skipped='0')
    suite = ET.SubElement(root, 'testsuite', tests=str(sum(counts.values())), failures='0', errors='0', skipped='0')
    for module, number in counts.items():
        for index in range(number):
            ET.SubElement(suite, 'testcase', classname=module, name='case_' + str(index))
    (evidence / 'report-tests.xml').write_bytes(ET.tostring(root))
    (evidence / 'report-tests.log').write_text(
        'collected ' + str(sum(counts.values())) + ' items\n' + str(sum(counts.values())) + ' passed\n', encoding='utf-8')
    records = [{'path': '/fixture/cpu/' + p.name, 'sha256': focus.sha(p)} for p in sorted(evidence.iterdir())]
    focus.put(folder / 'assessment.json', {'status': 'passed', 'suite': focus.SUITE,
        'source_manifest_sha256': 'a' * 64, 'producer_job_id': job, 'tests': sum(counts.values()),
        'tests_by_module': counts, 'proofs': records})
    focus.put(folder / 'bundled-proofs.json', [{'original_path': p['path'],
        'local_path': 'evidence/' + Path(p['path']).name, 'sha256': p['sha256']} for p in records])
    return folder


def test_new_cpu_certificate_requires_all278_original_cases_and_both_new_modules(tmp_path):
    folder = cpu_fixture(tmp_path)
    actual = focus.verify_patch_cpu(folder, 'a' * 64, focus.sha(folder / 'assessment.json'))
    assert set(actual['counts']) == focus.CPU_MODULES
    assert sum(actual['counts'][key] for key in focus.OLD_COUNTS) == 278


@pytest.mark.parametrize('damage', ['proof', 'certificate_source', 'raw_lsf', 'wrong_collection', 'path_escape'])
def test_changed_new_cpu_proofs_or_source_are_rejected(tmp_path, damage):
    folder = cpu_fixture(tmp_path)
    cert = focus.read(folder / 'assessment.json')
    if damage == 'certificate_source':
        cert['source_manifest_sha256'] = 'b' * 64
        focus._replace_json(folder / 'assessment.json', cert)
    elif damage == 'path_escape':
        bundles = focus.read(folder / 'bundled-proofs.json'); bundles[0]['local_path'] = '../outside'
        focus._replace_json(folder / 'bundled-proofs.json', bundles)
    else:
        name = {'proof': 'report-tests.xml', 'raw_lsf': 'stdout.29799998.log',
                'wrong_collection': 'report-tests.log'}[damage]
        (folder / 'evidence' / name).write_bytes(b'changed raw proof')
    with pytest.raises((ValueError, FileNotFoundError)):
        focus.verify_patch_cpu(folder, 'a' * 64, focus.sha(folder / 'assessment.json'))


def test_production_patch_refuses_nonallocated_execution_before_any_output(tmp_path, monkeypatch):
    monkeypatch.delenv('LSB_JOBID', raising=False)
    with pytest.raises(ValueError, match='allocated'):
        focus.create_patch(tmp_path / 'original', tmp_path / 'out', publisher=None,
            original_pin={}, original_archive=tmp_path / 'missing', original_job_proof_dir=tmp_path / 'proof',
            patch_source_manifest_sha256='a' * 64, patch_source_manifest_path=tmp_path / 'source',
            patch_job_id='123', patch_cpu_certificate_sha256='b' * 64, patch_cpu_proof_dir=tmp_path / 'cpu')
    assert not (tmp_path / 'out').exists()


def patch_job_fixture(tmp_path):
    folder = tmp_path / 'patch-job'; folder.mkdir()
    job, manifest = '29799997', 'a' * 64
    parent = focus.FOCUS_ROOT / 'patch'
    args = ['bsub', '-q', 'hpc', '-n', '2', '-R', 'span[hosts=1]', '-R', 'rusage[mem=4GB]',
            '-W', '00:30', '-J', 'mosaic-viewer-focus-patch', '-oo', str(parent / 'stdout.%J.log'),
            '-eo', str(parent / 'stderr.%J.log'), 'bash', '-l', str(focus.FOCUS_SOURCE / 'stage.lsf'),
            str(focus.FOCUS_SOURCE), str(focus.FOCUS_ROOT), manifest, 'patch', str(parent)]
    (folder / 'scheduler.txt').write_text('Job <' + job + '>, Status <DONE>, Queue <hpc>', encoding='utf-8')
    (folder / 'stdout.log').write_text('Subject: Job ' + job + ': focus-patch Done\nSuccessfully completed.\n', encoding='utf-8')
    started = {'phase': 'patch', 'args': args, 'dependency': None, 'outcome': 'unknown', 'utc': 'fixture-time'}
    focus.put(folder / 'submission-started.json', started)
    focus.put(folder / 'submission.json', started | {'outcome': 'submitted', 'job_id': job,
        'returncode': 0, 'stdout': 'Job <' + job + '> is submitted to queue <hpc>.', 'stderr': ''})
    return folder, job, manifest


def test_additive_producer_requires_actual_exact_patch_submission_and_exclusive_start(tmp_path):
    folder, job, manifest = patch_job_fixture(tmp_path)
    assert set(focus.verify_patch_submission(folder, job, manifest)) == {
        'scheduler.txt', 'stdout.log', 'submission.json', 'submission-started.json'}
    for field, bad in [('phase', 'diagnostic'), ('args', ['bsub', '-q', 'hpc', '-n', '4'])]:
        original = focus.read(folder / 'submission.json')
        focus._replace_json(folder / 'submission.json', original | {field: bad})
        with pytest.raises(ValueError):
            focus.verify_patch_submission(folder, job, manifest)
        focus._replace_json(folder / 'submission.json', original)
    started = focus.read(folder / 'submission-started.json')
    focus._replace_json(folder / 'submission-started.json', started | {'outcome': 'submitted'})
    with pytest.raises(ValueError, match='exclusive|Exclusive'):
        focus.verify_patch_submission(folder, job, manifest)
