"""Allocated orchestration contracts: resource scope, evidence and failed studies."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

import jq1_lrboost_report_ops as ops
import capture_jq1_lrboost_completion as capture


@pytest.mark.parametrize('phase', ['preflight', 'cpu', 'dispatch', 'diagnostic', 'diagnostic-capture', 'capture', 'interim', 'final'])
def test_budget_retains_entire_remaining_native_projection_and_reserve(phase):
    current = {'protenix': 100, 'boltz2': 200}
    b = ops.report_budget(current, 100 * 1024**3, 90 * 1024**3, 500, phase)
    remaining = sum(ops.FULL_NATIVE.values()) - 300
    allowance = 1024**3 if phase in ('preflight', 'cpu') else (2 * 1024**3 + 512 * 1024**2)
    assert b['required_bytes'] == remaining + allowance + 4 * 1024**3
    assert b['available_bytes_conservative'] == 90 * 1024**3 - 500
    assert b['diagnostic_per_arm_hard_output_bytes'] == 256 * 1024**2


@pytest.mark.parametrize('bad', [{'protenix': -1, 'boltz2': 0}, {'protenix': True, 'boltz2': 0},
                                 {'protenix': ops.FULL_NATIVE['protenix'] + 1, 'boltz2': 0}, {'protenix': 0}])
def test_projection_mismatch_cannot_become_capacity_pass(bad):
    with pytest.raises(ValueError): ops.report_budget(bad, 10, 10, 0, 'dispatch')


@pytest.mark.parametrize('values', [(-1, 10, 0), (10, True, 0), (10, 10, -1)])
def test_budget_rejects_malformed_quota(values):
    with pytest.raises(ValueError): ops.report_budget(dict.fromkeys(ops.FULL_NATIVE, 0), *values, 'cpu')


def test_exclusive_receipt_prevents_overwriting_unknown_submission(tmp_path):
    path = tmp_path / 'submission-started.json'
    ops.exclusive(path, {'outcome': 'unknown'})
    with pytest.raises(FileExistsError): ops.exclusive(path, {'outcome': 'submitted'})
    assert json.loads(path.read_text()) == {'outcome': 'unknown'}


def test_atomic_jobs_update_preserves_complete_json(tmp_path):
    path = tmp_path / 'jobs.json'; path.write_text('{"cpu":"123"}\n')
    ops.atomic(path, {'cpu': '123', 'diagnostics': {'protenix': '124'}})
    assert json.loads(path.read_text())['diagnostics'] == {'protenix': '124'}
    assert not path.with_name('jobs.json.tmp').exists()


def test_only_two_diagnostics_and_three_readonly_collectors_are_declared():
    plan = ops.submission_plan()
    assert [(a, p) for _, p, a, _ in plan if p == 'diagnostic'] == [('protenix', 'diagnostic'), ('boltz2', 'diagnostic')]
    assert {a for _, p, a, _ in plan if p == 'capture'} == set(ops.ARMS)
    assert [d for _, p, a, d in plan if p == 'capture' and a == 'boltz2_chemistry'] == ['ended(29779179)']
    assert all(d is None for _, _, a, d in plan if a != 'boltz2_chemistry')


def test_gpu_diagnostics_and_cpu_render_have_separate_resources():
    gpu = ops.resources('diagnostic'); render = ops.resources('final')
    assert gpu[gpu.index('-q') + 1] == 'gpul40s'
    assert gpu[gpu.index('-W') + 1] == '02:00'
    assert gpu[gpu.index('-gpu') + 1] == 'num=1:mode=exclusive_process'
    assert render[render.index('-q') + 1] == 'hpc' and '-gpu' not in render
    assert 'rusage[mem=8GB]' in render and render[render.index('-n') + 1] == '4'


@pytest.mark.parametrize('queue', ['hpc', 'gpul40s'])
def test_actual_scheduler_accepts_line_wrapping_only_with_exact_identity(queue):
    ops.actual_scheduler('Job <123>, Status <DONE>, Queue <' + queue[:2] + '\n ' + queue[2:] + '>', '123', queue)
    for raw in ['Job <124>, Status <DONE>, Queue <' + queue + '>',
                'Job <123>, Status <RUN>, Queue <' + queue + '>',
                'Job <123>, Job <123>, Status <DONE>, Queue <' + queue + '>']:
        with pytest.raises(ValueError): ops.actual_scheduler(raw, '123', queue)


def test_lsf_success_cannot_relabel_original_failure():
    ops.actual_lsf('Subject: Job 123: example Done\nSuccessfully completed.\n', '123')
    for raw in ['Subject: Job 123: example Exited\nExited with exit code 1.\n',
                'Subject: Job 124: example Done\nSuccessfully completed.\n',
                'Subject: Job 123: example Done\nSuccessfully completed.\nExited by signal 15\n']:
        with pytest.raises(ValueError): ops.actual_lsf(raw, '123')


def xml_fixture(path, *, omit=None, failure=False, extra=False):
    names = [Path(n).stem for n in ops.TESTS + ops.LEGACY_TESTS if n != omit]
    if extra: names.append('test_unrelated')
    root = ET.Element('testsuites'); suite = ET.SubElement(root, 'testsuite', tests=str(len(names)), failures=str(int(failure)), errors='0', skipped='0')
    for i, name in enumerate(names):
        c = ET.SubElement(suite, 'testcase', classname='releases.source.tests.' + name, name='test_' + str(i))
        if failure and i == 0: ET.SubElement(c, 'failure', message='fixture failed')
    ET.ElementTree(root).write(path)


def test_raw_cpu_xml_requires_all_modules_without_waiving_failures(tmp_path):
    path = tmp_path / 'tests.xml'; xml_fixture(path)
    assert len(ops.xml_cases(path)) == len(ops.TESTS + ops.LEGACY_TESTS)
    for kw in [{'omit': ops.TESTS[0]}, {'failure': True}, {'extra': True}]:
        xml_fixture(path, **kw)
        with pytest.raises(ValueError): ops.xml_cases(path)


def test_cpu_xml_totals_must_match_actual_rows(tmp_path):
    path = tmp_path / 'tests.xml'; xml_fixture(path)
    tree = ET.parse(path); tree.getroot()[0].set('tests', '999'); tree.write(path)
    with pytest.raises(ValueError): ops.xml_cases(path)


def test_submission_rejects_scientific_or_unsafe_job_names_before_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    for name, phase, arm in [('run-protenix', 'diagnostic', 'protenix'), ('../escape', 'interim', 'none'),
                              ('diagnostic-boltz2_chemistry', 'diagnostic', 'boltz2_chemistry')]:
        with pytest.raises(ValueError): ops.submit(name, phase, 'a' * 64, arm=arm)
    assert not list(tmp_path.iterdir())


def test_unknown_bsub_outcome_is_persisted_and_second_attempt_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=255, stdout='', stderr='connection lost'))
    with pytest.raises(ValueError): ops.submit('interim-0001', 'interim', 'a' * 64)
    assert json.loads((tmp_path / 'interim-0001/submission.json').read_text())['outcome'] == 'unknown'
    with pytest.raises(FileExistsError): ops.submit('interim-0001', 'interim', 'a' * 64)


@pytest.mark.parametrize('status', ['FAILED_PERSISTED', 'EXIT', 'DONE'])
def test_collector_keeps_truthful_terminal_scheduler_scope(tmp_path, monkeypatch, status):
    monkeypatch.setattr(capture.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='Job is not found\n', stderr=''))
    monkeypatch.setattr(capture, 'status_or_completion', lambda *a: status)
    observed, proof = capture.terminal_scheduler('protenix', tmp_path, tmp_path)
    assert observed == status
    assert proof.read_text() == 'Job is not found\n'


def test_cpu_gate_validates_allocated_scopes_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError): ops.verify_cpu('a' * 64)
    assert not (tmp_path / 'analysis-dispatch-started.json').exists()


def test_executing_job_must_match_its_exact_submission_and_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path); monkeypatch.setenv('LSB_JOBID', '321')
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='Job <321> is submitted\n', stderr=''))
    record = ops.submit('interim-0001', 'interim', 'a' * 64)
    parent = tmp_path / 'interim-0001'
    assert ops.validate_submission(parent, 'interim', 'none', 'a' * 64)['job_id'] == '321'
    for field, value in [('job_id', '999'), ('phase', 'diagnostic'), ('arm', 'protenix'), ('outcome', 'unknown')]:
        changed = dict(record); changed[field] = value
        (parent / 'submission.json').write_text(json.dumps(changed))
        with pytest.raises(ValueError): ops.validate_submission(parent, 'interim', 'none', 'a' * 64)
    changed = dict(record); changed['args'] = record['args'].copy(); changed['args'][changed['args'].index('-n') + 1] = '1'
    (parent / 'submission.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError): ops.validate_submission(parent, 'interim', 'none', 'a' * 64)


def test_diagnostic_failure_does_not_become_capture_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'ROOT', tmp_path)
    original = tmp_path / 'diagnostic-protenix'; original.mkdir()
    (original / 'submission.json').write_text(json.dumps({'job_id': '123', 'phase': 'diagnostic', 'arm': 'protenix', 'outcome': 'submitted'}))
    callback = tmp_path / 'diagnostic-capture-protenix'; callback.mkdir()
    import campaign25_pocket_ops
    monkeypatch.setattr(campaign25_pocket_ops, 'final_job', lambda *a: ('EXIT', 'gpul40s', []))
    with pytest.raises(ValueError): ops.diagnostic_capture(callback, 'protenix', 'a' * 64)
    assert not (callback / 'payload/assessment.json').exists()
