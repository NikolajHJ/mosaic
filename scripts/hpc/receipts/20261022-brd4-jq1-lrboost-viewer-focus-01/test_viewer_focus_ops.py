"""CPU contracts for a viewer-only, independently authenticated patch job."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import viewer_focus_ops as ops


def collected_fixture():
    counts = {**ops.OLD_COUNTS, 'test_viewer_focus': 2, 'test_viewer_focus_ops': 3}
    lines = [f'/release/{name}.py::test_case_{i}' for name, count in counts.items() for i in range(count)]
    return counts, '\n'.join(lines) + f'\n{sum(counts.values())} tests collected in 0.01s\n'


def xml_fixture(path, counts):
    root = ET.Element('testsuites'); suite = ET.SubElement(root, 'testsuite', tests=str(sum(counts.values())), failures='0', errors='0', skipped='0')
    for name, count in counts.items():
        for index in range(count):
            ET.SubElement(suite, 'testcase', classname='release.' + name, name='test_' + str(index))
    ET.ElementTree(root).write(path)


@pytest.mark.parametrize('current', [{'protenix': 0, 'boltz2': 0}, {'protenix': 100, 'boltz2': 200}, ops.FULL_NATIVE])
def test_whole_budget_always_preserves_original_full_native_analysis_and_reserves(current):
    b = ops.whole_budget(current, 100 * 1024**3, 90 * 1024**3, 123)
    assert b['required_bytes'] == sum(ops.FULL_NATIVE.values()) - sum(current.values()) + (2 * 1024**3 + 512 * 1024**2) + 4 * 1024**3
    assert b['available_bytes_conservative'] == 90 * 1024**3 - 123
    assert b['file_reserve'] == 100000


@pytest.mark.parametrize('current', [{'protenix': -1, 'boltz2': 0}, {'protenix': True, 'boltz2': 0},
                                  {'protenix': ops.FULL_NATIVE['protenix'] + 1, 'boltz2': 0}, {'protenix': 0}])
def test_invalid_or_exceeded_projection_fails_closed(current):
    with pytest.raises(ValueError): ops.whole_budget(current, 100, 100, 0)


@pytest.mark.parametrize('capacity', [(-1, 20, 0), (20, True, 0), (20, 20, -1)])
def test_malformed_capacity_cannot_pass(capacity):
    with pytest.raises(ValueError): ops.whole_budget(dict.fromkeys(ops.FULL_NATIVE, 0), *capacity)


def test_new_and_original_analysis_roots_releases_and_uploads_are_stale_charged():
    paths = ops.stale_paths()
    assert len(paths) == len(set(paths))
    for path in (ops.ANALYSIS_ROOT, ops.ANALYSIS.parent, ops.SCIENCE_ROOT):
        assert path in paths
    for nested in (ops.ROOT, ops.SOURCE.parent, ops.UPLOAD):
        assert nested.is_relative_to(ops.ANALYSIS_ROOT)
        assert sum(nested.is_relative_to(p) for p in paths) == 1
    assert Path('/work3/s234818/mosaic/' + ops.ANALYSIS_TAG + '.tar.gz') in paths


@pytest.mark.parametrize('phase,n,mem,time', [('cpu', '4', '4GB', '01:00'), ('certify-cpu', '1', '2GB', '00:30'), ('patch', '2', '4GB', '00:30')])
def test_only_declared_cpu_resources(phase, n, mem, time):
    args = ops.resources(phase)
    assert args[args.index('-q') + 1] == 'hpc' and '-gpu' not in args
    assert args[args.index('-n') + 1] == n and 'rusage[mem=' + mem + ']' in args
    assert args[args.index('-W') + 1] == time


@pytest.mark.parametrize('phase', ['native', 'diagnostic', 'run', 'final', 'capture', 'interim', 'bootstrap'])
def test_no_scientific_diagnostic_or_old_controller_branch(phase):
    with pytest.raises(ValueError): ops.resources(phase)


@pytest.mark.parametrize('name,phase', [('run-protenix', 'patch'), ('../escape', 'cpu'), ('combined-report', 'patch'), ('patch-final', 'patch'), ('patch', 'cpu')])
def test_undeclared_parent_or_phase_rejected_before_mutation(tmp_path, monkeypatch, name, phase):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    with pytest.raises(ValueError): ops.submit(name, phase, 'a' * 64)
    assert not list(tmp_path.iterdir())


def test_callback_dependency_is_only_own_ended_cpu():
    args = ops.command('after-cpu', 'certify-cpu', 'a' * 64, 'ended(123)')
    assert args[args.index('-w') + 1] == 'ended(123)'
    for phase, name, dependency in [('patch', 'patch', 'ended(123)'), ('certify-cpu', 'after-cpu', 'done(123)'),
                                   ('certify-cpu', 'after-cpu', None), ('cpu', 'cpu', 'ended(123)')]:
        with pytest.raises(ValueError): ops.command(name, phase, 'a' * 64, dependency)


def test_unknown_submission_receipt_prevents_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=255, stdout='', stderr='unknown outcome'))
    with pytest.raises(ValueError): ops.submit('patch', 'patch', 'a' * 64)
    assert ops.read(tmp_path / 'patch/submission.json')['outcome'] == 'unknown'
    with pytest.raises(FileExistsError): ops.submit('patch', 'patch', 'a' * 64)


def test_receipt_exclusivity_and_atomic_jobs(tmp_path):
    path = tmp_path / 'proof.json'; ops.exclusive(path, {'status': 'unknown'})
    with pytest.raises(FileExistsError): ops.exclusive(path, {'status': 'passed'})
    jobs = tmp_path / 'jobs.json'; ops.atomic(jobs, {'cpu': '1'}); ops.atomic(jobs, {'cpu': '1', 'patch': '2'})
    assert ops.read(jobs) == {'cpu': '1', 'patch': '2'} and not (tmp_path / 'jobs.json.tmp').exists()


def test_executing_job_requires_own_exact_submission_vector(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path); monkeypatch.setenv('LSB_JOBID', '123')
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='Job <123> is submitted', stderr=''))
    record = ops.submit('patch', 'patch', 'b' * 64); parent = tmp_path / 'patch'
    assert ops.validate_submission(parent, 'patch', 'b' * 64)['job_id'] == '123'
    for key, value in [('outcome', 'unknown'), ('job_id', '999'), ('phase', 'cpu')]:
        bad = {**record, key: value}; (parent / 'submission.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): ops.validate_submission(parent, 'patch', 'b' * 64)
    bad = dict(record); bad['args'] = record['args'].copy(); bad['args'][bad['args'].index('-n') + 1] = '4'
    (parent / 'submission.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError): ops.validate_submission(parent, 'patch', 'b' * 64)


def test_collection_authenticates_original_278_and_dynamic_new_case_counts():
    counts, raw = collected_fixture()
    assert ops.collection_counts(raw) == counts and sum(ops.OLD_COUNTS.values()) == 278


@pytest.mark.parametrize('defect', ['missing_new', 'duplicate', 'wrong_total', 'unrelated', 'changed_old'])
def test_collection_cannot_omit_or_invent_cases(defect):
    counts, raw = collected_fixture()
    if defect == 'missing_new': raw = raw.replace('/release/test_viewer_focus.py::test_case_0\n', '').replace('/release/test_viewer_focus.py::test_case_1\n', '')
    elif defect == 'duplicate': raw = raw.replace('/release/test_viewer_focus.py::test_case_1', '/release/test_viewer_focus.py::test_case_0')
    elif defect == 'wrong_total': raw = raw.replace(str(sum(counts.values())) + ' tests collected', '999 tests collected')
    elif defect == 'unrelated': raw = raw.replace('/release/test_viewer_focus.py', '/release/test_other.py')
    else: raw = raw.replace('/release/test_jq1_lrboost_ops.py::test_case_0\n', '')
    with pytest.raises(ValueError): ops.collection_counts(raw)


def test_raw_xml_exact_module_counts_and_headers(tmp_path):
    counts, _ = collected_fixture(); path = tmp_path / 'tests.xml'; xml_fixture(path, counts)
    assert len(ops.xml_cases(path, counts)) == sum(counts.values())
    tree = ET.parse(path); tree.getroot()[0].set('tests', '999'); tree.write(path)
    with pytest.raises(ValueError): ops.xml_cases(path, counts)


@pytest.mark.parametrize('defect', ['failure', 'error', 'skipped', 'unrelated', 'missing'])
def test_raw_xml_failures_skips_or_attribution_fail_closed(tmp_path, defect):
    counts, _ = collected_fixture(); path = tmp_path / 'tests.xml'; xml_fixture(path, counts)
    tree = ET.parse(path); suite = tree.getroot()[0]
    if defect in ('failure', 'error', 'skipped'): ET.SubElement(suite[0], defect)
    elif defect == 'unrelated': suite[0].set('classname', 'test_unrelated')
    else: suite.remove(suite[0]); suite.set('tests', str(len(suite)))
    tree.write(path)
    with pytest.raises(ValueError): ops.xml_cases(path, counts)


def test_source_authentication_precedes_module_import_and_rejects_extra_bytes(tmp_path):
    root = tmp_path / 'source'; root.mkdir(); (root / 'tool.py').write_bytes(b'value=1\n')
    manifest = {'files': {'tool.py': hashlib.sha256((root / 'tool.py').read_bytes()).hexdigest()}}
    (root / 'source_manifest.json').write_text(json.dumps(manifest)); pin = ops.sha(root / 'source_manifest.json')
    assert ops.authenticate_source(root, pin) == manifest['files']
    (root / 'extra.py').write_text('bad=1')
    with pytest.raises(ValueError): ops.authenticate_source(root, pin)
    (root / 'extra.py').unlink(); (root / 'tool.py').write_text('value=2')
    with pytest.raises(ValueError): ops.authenticate_source(root, pin)


def test_absent_cpu_proof_cannot_create_patch_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError): ops.verify_cpu('a' * 64)
    assert not (tmp_path / 'patch').exists() and not (tmp_path / 'patch-dispatch-started.json').exists()


def test_bootstrap_submits_only_cpu_and_its_own_ended_certificate(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path / 'new')
    monkeypatch.setattr(ops, 'authenticate_source', lambda *a: None)
    monkeypatch.setattr(ops, 'authenticate_external', lambda: None)
    gates = []; monkeypatch.setattr(ops, 'check_storage', lambda parent, phase: gates.append((parent.name, phase)))
    submitted = []
    def submit(name, phase, manifest, **kw):
        submitted.append((name, phase, kw.get('dependency')))
        return {'job_id': str(123 + len(submitted) - 1)}
    monkeypatch.setattr(ops, 'submit', submit)
    assert ops.bootstrap('a' * 64) == {'cpu': '123', 'after_cpu': '124'}
    assert submitted == [('cpu', 'cpu', None), ('after-cpu', 'certify-cpu', 'ended(123)')]
    assert gates == [('preflight', 'preflight')]
    assert ops.read(ops.ROOT / 'initial-submissions-complete.json') == {'cpu': '123', 'after_cpu': '124'}
    with pytest.raises(FileExistsError): ops.bootstrap('a' * 64)
    assert len(submitted) == 2
