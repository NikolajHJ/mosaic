"""Allocated fixtures for append-only prefixes and sparse interim presentation."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import jq1_report_contract as contract
import render_jq1_25_interim as interim
from test_jq1_25_report import saved, put


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Report fixtures require allocated DTU HPC')


def test_partial_final_json_line_is_excluded_but_committed_corruption_fails(tmp_path):
    p = tmp_path / 'states.jsonl'
    p.write_bytes(b'{"step":0}\n{"step":1')
    rows, receipt = interim.prefix(p)
    assert rows == [{'step': 0}] and receipt['bytes'] == len(b'{"step":0}\n')
    assert receipt['uncommitted_tail_bytes'] > 0
    p.write_bytes(b'{"step":0}\nbad\n')
    with pytest.raises(json.JSONDecodeError):
        interim.prefix(p)


def test_only_shared_committed_prefix_is_captured(saved):
    root, case, _, _ = saved
    for name in ('states', 'updates'):
        (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in case[name]))
    captured, proofs = interim.committed_case(root)
    assert captured == {'states': case['states'], 'updates': case['updates']} and len(proofs) == 2
    (root / 'updates.jsonl').write_bytes(b'{"step":0')
    captured, _ = interim.committed_case(root)
    assert len(captured['states']) == 1 and not captured['updates']


@pytest.mark.parametrize('rows', [[{'step': 1}], [{'step': 0}, {'step': 2}]])
def test_noncontiguous_live_records_are_not_hidden(tmp_path, rows):
    p = tmp_path / 'states.jsonl'
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='Noncontiguous'):
        interim.prefix(p)


def test_milestones_keep_exact_requested_steps_and_latest_without_duplicates():
    assert interim.selected_steps(1499) == [0, 1499]
    assert interim.selected_steps(1500) == [0, 1500]
    assert interim.selected_steps(3600) == [0, 1500, 2500, 3500, 3600]
    assert len(interim.selected_steps(10000)) == 11
    assert interim.milestone(1499) == 0 and interim.milestone(2501) == 2500


def test_sparse_assets_keep_exact_float_bytes_raw_graph_and_every_scalar(saved, tmp_path):
    root, case, x, e = saved
    chunks, curves, selected, counts = interim.sparse_assets(root, case, tmp_path / 'sparse', 'contact')
    assert selected == [0, 1] and len(curves) == 2 and counts['applied'] == 1
    array = np.frombuffer((tmp_path / 'sparse/frames.bin').read_bytes(), dtype='<f4').reshape(2, 1650)
    np.testing.assert_array_equal(array[1, :150], x.ravel())
    np.testing.assert_array_equal(array[1, 150:], e.ravel())
    assert contract.read(tmp_path / 'sparse/frames.json')[1]['readout']['disconnected']
    studies = [{'arm': 'fixture', 'job_id': 'allocated-fixture', 'label': 'Fixture', 'color': '#246bb2',
                'metric': 'contact', 'chunks': chunks, 'curves': curves, 'snapshot_steps': selected,
                'states': 2, 'latest_update': 1}]
    interim.assemble(studies, tmp_path / 'sparse', interim=True)
    receipt = contract.read(tmp_path / 'sparse/report_receipt.json')
    assert receipt['complete'] is False and receipt['included_updates'] == [1]
    assert receipt['snapshot_steps'] == [[0, 1]]
    assert 'IN PROGRESS' in (tmp_path / 'sparse/report.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('status', ['RUN', 'PEND', 'DONE'])
def test_current_live_status_never_converts_running_to_done(status):
    assert interim.live_status(f'Job <29770179>, Status <{status}>, Queue <gpu\nl40s>', '29770179') == status


@pytest.mark.parametrize('raw', ['Job <29770179> is not found', 'Job <29770179>, Status <EXIT>, Queue <gpul40s>', 'Job <12>, Status <RUN>, Queue <gpul40s>'])
def test_failed_wrong_or_missing_job_cannot_be_shown_as_healthy(raw):
    with pytest.raises(ValueError):
        interim.live_status(raw, '29770179')


@pytest.mark.parametrize('filename', ['scheduler-final-1.txt', 'native-scheduler-0.txt'])
def test_aged_out_record_needs_pinned_success_proofs(tmp_path, filename):
    with pytest.raises(FileNotFoundError):
        interim.status_or_completion('Job <29770179> is not found', 'protenix', tmp_path)
    proof = tmp_path / filename
    proof.write_text('Job <29770179>, Status <DONE>, Queue <gpul40s>')
    put(tmp_path / 'complete-protenix/payload/assessment.json', {'status': 'passed', 'native_job': '29770179',
        'proofs': [{'path': str(proof), 'sha256': contract.sha(proof)}]})
    assert interim.status_or_completion('not found', 'protenix', tmp_path) == 'DONE'
    proof.write_text('Job <29770179>, Status <EXIT>, Queue <gpul40s>')
    with pytest.raises(ValueError, match='changed'):
        interim.status_or_completion('not found', 'protenix', tmp_path)
