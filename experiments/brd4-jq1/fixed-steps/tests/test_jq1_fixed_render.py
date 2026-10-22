"""Allocated CPU presentation fixtures; synthetic saved arrays never certify a run."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import copy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(_layout_path(_repository_root(__file__) / 'scripts')))
import jq1_fixed_report_contract as contract
import render_jq1_fixed_report as report
import render_jq1_fixed_interim as interim
import capture_jq1_fixed_completion as collector


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Fixed report fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')


def saved_case(path, *, count=2, deltas=None, no_ops=None, strict=None, rate=0.3):
    """Synthetic presentation data, isolated from production identity validators."""
    root = path / 'case-0'
    (root / 'snapshots').mkdir(parents=True)
    # Binary fractions and a bounded two-state change avoid accumulated FP32
    # normalization drift in long indexing fixtures.  This remains synthetic
    # presentation data; no production simplex tolerance is changed.
    x = np.full((25, 6), .0625, np.float32)
    x[:, 0] = .6875
    probability_phase = False
    e = np.zeros((300, 5), np.float32)
    e[:, 0] = 1
    pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
    states, updates, arrays = [], [], []
    loss = -.1
    for step in range(count):
        if step:
            noop = False if no_ops is None else no_ops[step - 1]
            if not noop:
                probability_phase = not probability_phase
                offset = np.float32(2**-16 if probability_phase else 0.)
                x[0, 0] = np.float32(.6875) + offset
                x[0, 1] = np.float32(.0625) - offset
            delta = -.001 if deltas is None else deltas[step - 1]
            before = loss
            loss += delta
            improving = delta < -1e-7 if strict is None else strict[step - 1]
            updates.append({'step': step - 1, 'fixed_before': before, 'fixed_after': loss,
                'selection_policy': 'fixed_step', 'selected_trial': 0,
                'accepted_step_size': rate, 'fallback_used': False,
                'no_op': noop, 'fixed_increased': loss > before,
                'repair_weight': 0., 'contact_before': before, 'contact_after': loss,
                'attempts': [{'trial': 0, 'rate': rate, 'strict': improving,
                    'autodiff_surrogate': loss, 'independent_primal_surrogate': loss,
                    'components': {'contact_loss': loss}, 'repair_weight': 0.}]})
        snapshot = root / f'snapshots/step-{step:04d}.npz'
        np.savez_compressed(snapshot, X=x, E_upper=e, contact_primal=loss,
                            repair_weight=0., refreshed_surrogate=loss, autodiff_surrogate=loss)
        states.append({'step': step, 'snapshot': snapshot.relative_to(root).as_posix(),
            'snapshot_sha256': contract.sha(snapshot), 'contact_loss': loss,
            'repair_weight': 0., 'refreshed_surrogate': loss, 'autodiff_surrogate': loss,
            'previous_fixed_surrogate_after': None if step == 0 else loss,
            'components': {'contact_loss': loss}, 'preparation_identity': 'allocated-synthetic-presentation-fixture',
            'chemistry_preparation': [], 'readout': {'element_indices': [0] * 25,
                'bond_upper_indices': [0] * 300, 'pairs': pairs,
                'sanitized': False, 'disconnected': True, 'canonical_smiles': None,
                'conversion_error': 'Saved invalid synthetic fixture'}})
        arrays.append((x.copy(), e.copy()))
    return root, {'states': states, 'updates': updates}, arrays


@pytest.fixture
def saved(tmp_path):
    return saved_case(tmp_path / 'native')


def edit_snapshot(root, state, **fields):
    path = root / state['snapshot']
    with np.load(path, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    np.savez_compressed(path, **(arrays | fields))
    state['snapshot_sha256'] = contract.sha(path)


def study_entry(case, chunks=None, curves=None, *, arm='protenix', native_status='RUN'):
    pin = contract.ARMS[arm]
    latest = len(case['updates']) if case['states'] else None
    return {'arm': arm, 'job_id': pin['job'], 'label': pin['label'], 'color': pin['color'],
        'metric': pin['metric'], 'learning_rate': pin['rate'], 'rate': pin['rate'],
        'selection_policy': 'fixed_step', 'acceptance_policy': 'fixed_step',
        'native_status': native_status, 'status': 'Allocated synthetic presentation fixture',
        'chunks': chunks or [], 'curves': curves or [], 'states': len(case['states']),
        'latest_update': latest, 'milestone_steps': [] if latest is None else interim.selected_steps(latest),
        'counts': contract.fixed_counts(case['updates'], pin['rate']),
        'model_digest': pin['model'], 'context_digest': pin['context'],
        'source_manifest_sha256': pin['manifest'], 'config_sha256': pin['config'],
        'readiness_sha256': pin['ready'], 'progress_evidence': {'scope': 'synthetic presentation fixture'}}


def test_saved_fp32_frames_are_exact_and_raw_invalid_graphs_are_retained(saved, tmp_path):
    root, case, arrays = saved
    chunks, curves = report.write_frames(root, case, tmp_path / 'frames', 'contact', rate=.3, chunk_size=1)
    assert [(c['start'], c['count']) for c in chunks] == [(0, 1), (1, 1)]
    for index, chunk in enumerate(chunks):
        path = tmp_path / 'frames' / chunk['binary']
        raw = np.frombuffer(path.read_bytes(), dtype='<f4')
        assert path.stat().st_size == 6600 and raw.size == 1650
        np.testing.assert_array_equal(raw[:150].reshape(25, 6), arrays[index][0])
        np.testing.assert_array_equal(raw[150:].reshape(300, 5), arrays[index][1])
        row = contract.read(tmp_path / 'frames' / chunk['metadata'])[0]
        assert row['snapshot_sha256'] == case['states'][index]['snapshot_sha256']
        assert row['readout']['disconnected'] and not row['readout']['sanitized']
        assert row['readout']['canonical_smiles'] is None
        assert contract.sha(path) == chunk['binary_sha256']
    assert curves[0]['fixed_delta'] is None and curves[0]['binder'] is None
    previous = contract.read(tmp_path / 'frames/frames-00001.json')[0]['previous_update']
    assert previous == {**{k: case['updates'][0][k] for k in (
        'selection_policy', 'selected_trial', 'accepted_step_size', 'fixed_before',
        'fixed_after', 'fixed_increased', 'no_op', 'fallback_used')}, 'strict': True, 'attempt_count': 1}


@pytest.mark.parametrize('count', [99, 100, 101])
def test_every_frame_is_present_across_bounded_chunk_boundary(tmp_path, count):
    root, case, _ = saved_case(tmp_path / 'native', count=count)
    chunks, curves, counts = interim.progress_assets(root, case, tmp_path / 'frames', 'contact', rate=.3)
    assert [(c['start'], c['count']) for c in chunks] == [(0, min(count, 100))] + ([(100, 1)] if count == 101 else [])
    rows = [r for c in chunks for r in contract.read(tmp_path / 'frames' / c['metadata'])]
    assert [r['step'] for r in rows] == list(range(count))
    assert len(curves) == count and counts['applied'] == count - 1
    assert all((tmp_path / 'frames' / c['binary']).stat().st_size == 6600 * c['count'] for c in chunks)


def test_uphill_equality_noop_and_nonstrict_decrease_stay_distinct(tmp_path):
    root, case, _ = saved_case(tmp_path / 'native', count=5,
        deltas=[.01, 0., 0., -1e-8], no_ops=[False, False, True, False],
        strict=[False, False, False, False])
    chunks, curves, counts = interim.progress_assets(root, case, tmp_path / 'frames', 'contact', rate=.3)
    assert counts == dict(applied=4, strict_improving=0, forced_fallback=0, no_op=1,
                          scalar_increased=1, scalar_equal=2, scalar_decreased=1)
    assert curves[1]['fixed_increased'] and not curves[1]['fallback']
    assert curves[2]['scalar_equal'] and not curves[2]['no_op']
    assert curves[3]['scalar_equal'] and curves[3]['no_op']
    assert curves[4]['scalar_decreased'] and not curves[4]['strict']
    rows = contract.read(tmp_path / 'frames' / chunks[0]['metadata'])
    assert rows[1]['previous_update']['fixed_increased']
    assert rows[3]['previous_update']['no_op']
    assert rows[4]['previous_update']['strict'] is False


@pytest.mark.parametrize('damage', ['hash', 'shape', 'dtype', 'nonfinite', 'simplex',
                                  'readout', 'bond_readout', 'pairs', 'scalar', 'nonscalar'])
def test_bad_saved_snapshot_cannot_be_rendered_as_a_verified_frame(saved, damage):
    root, case, arrays = saved
    state, (x, e) = case['states'][0], arrays[0]
    if damage == 'hash': state['snapshot_sha256'] = '0' * 64
    elif damage == 'shape': edit_snapshot(root, state, X=x[:6])
    elif damage == 'dtype': edit_snapshot(root, state, X=x.astype(np.float64))
    elif damage == 'nonfinite':
        x[0, 0] = np.nan
        edit_snapshot(root, state, X=x)
    elif damage == 'simplex': edit_snapshot(root, state, E_upper=e * .5)
    elif damage == 'readout': state['readout']['element_indices'][0] = 1
    elif damage == 'bond_readout': state['readout']['bond_upper_indices'][0] = 1
    elif damage == 'pairs': state['readout']['pairs'][0] = [2, 3]
    elif damage == 'scalar': edit_snapshot(root, state, contact_primal=0.)
    elif damage == 'nonscalar': edit_snapshot(root, state, contact_primal=np.asarray([state['contact_loss']]))
    with pytest.raises(ValueError):
        report.frame_arrays(root, state, 'contact')


def test_missing_or_escaped_snapshot_fails_instead_of_skipping_a_frame(saved, tmp_path):
    root, case, _ = saved
    original = root / case['states'][0]['snapshot']
    external = tmp_path / 'outside.npz'
    external.write_bytes(original.read_bytes())
    case['states'][0]['snapshot'] = '../../outside.npz'
    with pytest.raises(ValueError, match='path/hash'):
        report.frame_arrays(root, case['states'][0], 'contact')
    case['states'][0]['snapshot'] = 'snapshots/missing.npz'
    with pytest.raises(FileNotFoundError):
        report.frame_arrays(root, case['states'][0], 'contact')


@pytest.mark.parametrize('damage', ['gap', 'extra_update', 'huge_chunk', 'zero_chunk', 'halved_rate',
                                  'multiple_attempts', 'changed_policy', 'wrong_noop', 'wrong_before',
                                  'wrong_native_after', 'wrong_fixed_after'])
def test_frame_writer_rejects_scope_policy_and_state_update_correspondence_drift(saved, tmp_path, damage):
    root, case, _ = saved
    chunk_size = 100
    if damage == 'gap': case['states'][1]['step'] = 2
    elif damage == 'extra_update': case['updates'].append(copy.deepcopy(case['updates'][0]))
    elif damage == 'huge_chunk': chunk_size = 101
    elif damage == 'zero_chunk': chunk_size = 0
    elif damage == 'halved_rate': case['updates'][0]['accepted_step_size'] = .15
    elif damage == 'multiple_attempts': case['updates'][0]['attempts'].append(copy.deepcopy(case['updates'][0]['attempts'][0]))
    elif damage == 'changed_policy': case['updates'][0]['selection_policy'] = 'best_primal'
    elif damage == 'wrong_noop': case['updates'][0]['no_op'] = True
    elif damage == 'wrong_before': case['states'][0]['refreshed_surrogate'] = 0.
    elif damage == 'wrong_native_after': case['updates'][0]['contact_after'] = 0.
    elif damage == 'wrong_fixed_after': case['states'][1]['previous_fixed_surrogate_after'] = 0.
    with pytest.raises(ValueError):
        report.write_frames(root, case, tmp_path / 'frames', 'contact', rate=.3, chunk_size=chunk_size)


def test_only_unfinished_last_jsonl_line_is_excluded(tmp_path):
    path = tmp_path / 'states.jsonl'
    path.write_bytes(b'{"step":0}\n{"step":1')
    rows, receipt = interim.prefix(path)
    assert rows == [{'step': 0}] and receipt['bytes'] == len(b'{"step":0}\n')
    assert receipt['uncommitted_tail_bytes'] > 0
    path.write_bytes(b'{"step":0}\nbad\n')
    with pytest.raises(json.JSONDecodeError):
        interim.prefix(path)


@pytest.mark.parametrize('rows', [[{'step': 1}], [{'step': 0}, {'step': 2}], [{'step': True}], [None]])
def test_noncontiguous_or_wrongly_typed_committed_rows_are_not_hidden(tmp_path, rows):
    path = tmp_path / 'states.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(ValueError, match='Noncontiguous'):
        interim.prefix(path)


def test_common_prefix_uses_exact_included_byte_hash_and_does_not_outrun_updates(saved):
    root, case, _ = saved
    for name in ('states', 'updates'):
        (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in case[name]), encoding='utf-8')
    captured, proofs = interim.committed_case(root)
    assert captured == case and [p['included_rows'] for p in proofs] == [2, 1]
    assert all(p['included_sha256'] == p['sha256'] for p in proofs)
    (root / 'updates.jsonl').write_bytes(b'{"step":0')
    captured, proofs = interim.committed_case(root)
    assert captured == {'states': case['states'][:1], 'updates': []}
    assert proofs[0]['included_rows'] == 1 and proofs[0]['included_bytes'] < proofs[0]['bytes']
    assert proofs[1]['included_bytes'] == 0


def test_mutation_of_a_previously_committed_prefix_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'case'
    root.mkdir()
    states = root / 'states.jsonl'
    states.write_bytes(b'{"step":0}\n')
    (root / 'updates.jsonl').write_bytes(b'')
    original = interim.prefix
    def mutate_after_read(path):
        value = original(path)
        if Path(path).name == 'updates.jsonl':
            states.write_bytes(b'{"step":9}\n')
        return value
    monkeypatch.setattr(interim, 'prefix', mutate_after_read)
    with pytest.raises(ValueError, match='changed while capturing'):
        interim.committed_case(root)


@pytest.mark.parametrize('status,case,label', [
    ('PEND', None, 'queued; no saved molecule yet'),
    ('RUN', None, 'allocated/preparing; no authenticated saved molecule yet'),
    ('EXIT', None, 'failed before authenticated saved trajectory'),
    ('EXIT', {'states': [1]}, 'failed partial trajectory'),
])
def test_all_arm_statuses_remain_explicit_without_inventing_a_molecule(status, case, label):
    assert interim.status_label(status, case, {}) == label


def test_failed_case_is_not_presented_as_healthy_running():
    assert interim.status_label('RUN', {'states': [1]}, {'native_case_status': 'failed'}) == 'failed partial trajectory'


def test_milestones_are_per_arm_and_preserve_exact_endpoints():
    assert interim.selected_steps(1499) == [0, 1499]
    assert interim.selected_steps(3600) == [0, 1500, 2500, 3500, 3600]
    assert interim.selected_steps(10000) == [0, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10000]
    assert report.milestone(None) == 0 and report.milestone(1499) == 0
    assert report.milestone(2501) == 2500 and report.milestone(10000) == 10000


def test_schedule_keeps_declared_5000_hold_and_terminal_50():
    assert interim.expected_weight('boltz2', 10000) == 0.
    assert interim.expected_weight('boltz2_chemistry', 0) == .001
    assert interim.expected_weight('boltz2_chemistry', 4999) == .001
    assert interim.expected_weight('boltz2_chemistry', 5000) > .001
    assert interim.expected_weight('boltz2_chemistry', 9999) == 50.
    assert interim.expected_weight('boltz2_chemistry', 10000) == 50.


def test_html_receipt_keeps_fixed_schema_exact_jobs_all_empty_arms_and_immutable_capture(saved, tmp_path):
    root, case, _ = saved
    output = tmp_path / 'report'
    output.mkdir()
    chunks, curves = report.write_frames(root, case, output / 'protenix', 'contact', rate=.3)
    visible = study_entry(case, chunks, curves)
    visible['label'] = '</script><img src=x onerror=alert(1)>'
    empty = {'states': [], 'updates': []}
    entries = [visible, study_entry(empty, arm='boltz2', native_status='RUN'),
               study_entry(empty, arm='boltz2_chemistry', native_status='PEND')]
    stamp = '2026-10-21T20:00:00+00:00'
    report.assemble(entries, output, interim=True, captured_utc=stamp)
    html = (output / 'report.html').read_text(encoding='utf-8')
    assert visible['label'] not in html
    embedded = html.split('<script id="manifest" type="application/json">')[1].split('</script>')[0]
    data = json.loads(embedded)
    assert data['version'] == report.VERSION and data['acceptance_policy'] == 'fixed_step'
    assert data['captured_utc'] == stamp and data['complete'] is False
    assert [s['job_id'] for s in data['studies']] == ['29777709', '29778203', '29778204']
    assert data['studies'][1]['states'] == 0 and data['studies'][1]['latest_update'] is None
    assert data['studies'][1]['chunks'] == [] and 'curves' not in data['studies'][0]
    assert 'id="stepInput"' in html and 'getFloat32(offset,true)' in html
    receipt = contract.read(output / 'report_receipt.json')
    assert receipt['states'] == [2, 0, 0] and receipt['included_updates'] == [1, None, None]
    assert receipt['milestone_progress'] == dict(protenix=0, boltz2=0, boltz2_chemistry=0)
    assert receipt['html_bytes'] == (output / 'report.html').stat().st_size
    assert receipt['html_sha256'] == contract.sha(output / 'report.html')
    assert (output / 'loss-overlays.svg').is_file()


def test_completion_cannot_omit_full_frame_budget_or_policy(tmp_path):
    path = tmp_path / 'complete-protenix/payload/assessment.json'
    put(path, {'status': 'passed', 'native_job': '29777709', 'arm': 'protenix',
               'report_version': report.VERSION, 'states': 10000, 'updates': 9999,
               'n_atoms': 25, 'learning_rate': .3, 'selection_policy': 'fixed_step',
               'all_probability_frames_verified': True, 'proofs': []})
    with pytest.raises(ValueError, match='incomplete'):
        report.completion_assessment('protenix', tmp_path)


def test_collector_requires_same_job_lsf_success_and_uses_actual_done_only(tmp_path):
    path = tmp_path / 'native-stdout.txt'
    path.write_text('Subject: Job 29777709: completed\nSuccessfully completed.\n', encoding='utf-8')
    assert 'Successfully completed.' in collector.lsf_success(path, '29777709')
    with pytest.raises(ValueError, match='same-job'):
        collector.lsf_success(path, '29778203')
    assert collector._is_done('Job <29777709>, Status <DONE>, Queue <gpul40s>', '29777709')
    assert not collector._is_done('Job <29777709> is not found', '29777709')
    assert not collector._is_done('Job <29777709>, Status <EXIT>, Queue <gpul40s>', '29777709')
