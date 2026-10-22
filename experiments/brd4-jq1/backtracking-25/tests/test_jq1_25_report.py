"""Allocated report fixtures: exact saved frames, failures and bounded assets."""

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
import jq1_report_contract as contract
import render_jq1_25_report as report


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Report fixtures require allocated DTU HPC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'case'
    (root / 'snapshots').mkdir(parents=True)
    x = np.full((25, 6), 1 / 6, np.float32)
    x[:, 0] += .5
    x[:, 1:] *= .4
    x[:, 0] = 1 - x[:, 1:].sum(-1)
    e = np.zeros((300, 5), np.float32)
    e[:, 0] = 1
    pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
    states = []
    for step in range(2):
        path = root / f'snapshots/step-{step:04d}.npz'
        loss = -.1 - step * .01
        np.savez_compressed(path, X=x, E_upper=e, contact_primal=loss, repair_weight=0., refreshed_surrogate=loss, autodiff_surrogate=loss)
        states.append({'step': step, 'snapshot': path.relative_to(root).as_posix(), 'snapshot_sha256': contract.sha(path),
            'contact_loss': loss, 'repair_weight': 0., 'refreshed_surrogate': loss, 'autodiff_surrogate': loss,
            'components': {'contact_loss': loss}, 'preparation_identity': 'saved-fixture', 'chemistry_preparation': [],
            'readout': {'element_indices': [0] * 25, 'bond_upper_indices': [0] * 300, 'pairs': pairs,
                'sanitized': False, 'disconnected': True, 'canonical_smiles': None, 'conversion_error': 'saved invalid fixture'}})
    case = {'status': 'passed', 'requested_updates': 1, 'states': states, 'initial_checks': [{'field': 'atoms'}, {'field': 'bonds'}],
        'updates': [{'step': 0, 'fixed_before': -.1, 'fixed_after': -.11, 'fallback_used': False, 'no_op': False, 'accepted_step_size': .03}],
        'counts': {'applied': 1, 'strict_improving': 1, 'forced_fallback': 0, 'no_op': 0}}
    return root, case, x, e


def edit_snapshot(root, state, **fields):
    path = root / state['snapshot']
    with np.load(path) as saved:
        arrays = {k: saved[k] for k in saved.files}
    np.savez_compressed(path, **(arrays | fields))
    state['snapshot_sha256'] = contract.sha(path)


def test_25_slot_frames_are_bit_exact_and_keep_invalid_disconnected_readouts(saved, tmp_path):
    root, case, x, e = saved
    chunks, curves = report.write_frames(root, case, tmp_path / 'frames', 'contact', chunk_size=1)
    assert len(chunks) == 2 and len(curves) == 2
    for index, chunk in enumerate(chunks):
        raw = np.frombuffer((tmp_path / 'frames' / chunk['binary']).read_bytes(), dtype='<f4')
        assert raw.size == 1650
        np.testing.assert_array_equal(raw[:150].reshape(25, 6), x)
        np.testing.assert_array_equal(raw[150:].reshape(300, 5), e)
        rows = contract.read(tmp_path / 'frames' / chunk['metadata'])
        assert rows[0]['step'] == index and rows[0]['readout']['disconnected']
        assert not rows[0]['readout']['sanitized'] and rows[0]['readout']['canonical_smiles'] is None
        assert rows[0]['snapshot_sha256'] == case['states'][index]['snapshot_sha256']
    assert curves[0]['binder'] is None and curves[0]['repair'] is None
    assert curves[1]['fixed_delta'] == case['updates'][0]['fixed_after'] - case['updates'][0]['fixed_before']


@pytest.mark.parametrize('damage', ['hash', 'shape', 'dtype', 'nonfinite', 'simplex', 'readout', 'pairs', 'scalar'])
def test_bad_saved_data_cannot_be_presented_as_an_authenticated_frame(saved, damage):
    root, case, x, e = saved
    state = case['states'][0]
    if damage == 'hash': state['snapshot_sha256'] = '0' * 64
    elif damage == 'shape': edit_snapshot(root, state, X=x[:6])
    elif damage == 'dtype': edit_snapshot(root, state, X=x.astype(np.float64))
    elif damage == 'nonfinite':
        x[0, 0] = np.nan
        edit_snapshot(root, state, X=x)
    elif damage == 'simplex': edit_snapshot(root, state, E_upper=e * .5)
    elif damage == 'readout': state['readout']['element_indices'][0] = 1
    elif damage == 'pairs': state['readout']['pairs'][0] = [2, 3]
    elif damage == 'scalar': edit_snapshot(root, state, contact_primal=0.)
    with pytest.raises(ValueError):
        report.frame_arrays(root, state, 'contact')


def test_snapshot_path_cannot_escape_native_case(saved, tmp_path):
    root, case, _, _ = saved
    external = tmp_path / 'outside.npz'
    external.write_bytes((root / case['states'][0]['snapshot']).read_bytes())
    case['states'][0]['snapshot'] = '../outside.npz'
    with pytest.raises(ValueError, match='path/hash'):
        report.frame_arrays(root, case['states'][0], 'contact')


@pytest.mark.parametrize('damage', ['missing', 'oversize_chunk'])
def test_frame_asset_budget_is_contiguous_and_bounded(saved, tmp_path, damage):
    root, case, _, _ = saved
    if damage == 'missing': case['states'][1]['step'] = 2
    with pytest.raises(ValueError):
        report.write_frames(root, case, tmp_path / 'out', 'contact', chunk_size=101 if damage == 'oversize_chunk' else 100)


def test_html_loads_exact_chunks_instead_of_embedding_all_probabilities(saved, tmp_path):
    root, case, _, _ = saved
    out = tmp_path / 'report'
    out.mkdir()
    chunks, curves = report.write_frames(root, case, out / 'fixture', 'contact')
    label = '</script><img src=x onerror=alert(1)>'
    report.assemble([{'arm': 'fixture', 'label': label, 'job_id': 'synthetic-allocated-fixture',
        'metric': 'contact', 'color': '#246bb2', 'chunks': chunks, 'curves': curves, 'states': 2}], out)
    html = (out / 'report.html').read_text(encoding='utf-8')
    assert label not in html and 'getFloat32(offset,true)' in html
    assert 'cache.size>3' in html and 'No repaired target' in html
    embedded = html.split('<script id="manifest" type="application/json">')[1].split('</script>')[0]
    parsed = json.loads(embedded)
    assert parsed['studies'][0]['label'] == label and 'curves' not in parsed['studies'][0]
    assert len(html) < 30000 and (out / 'loss-overlays.svg').is_file()
    assert contract.read(out / 'report_receipt.json')['states'] == [2]


def test_case_requires_jsonl_equality_and_exact_budget(saved, tmp_path):
    _, case, _, _ = saved
    study = tmp_path / 'study'
    put(study / 'case-0/report.json', case)
    put(study / 'case-0/combined_initial_checks/report.json', {'status': 'passed', 'directions': case['initial_checks']})
    for name in ('states', 'updates'):
        (study / f'case-0/{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in case[name]))
    summary = {'status': 'passed', 'cases': [copy.deepcopy(case)]}
    assert contract.verify_case(study, summary, steps=1) == case
    with pytest.raises(ValueError, match='budget'):
        contract.verify_case(study, summary)
    (study / 'case-0/states.jsonl').write_text(json.dumps(case['states'][0]) + '\n')
    with pytest.raises(ValueError, match='JSONL'):
        contract.verify_case(study, summary, steps=1)


@pytest.mark.parametrize('status', ['DONE', 'D\n ONE', 'EXIT'])
def test_actual_scheduler_wrapping_is_allowed_without_fabricating_done(status):
    assert contract.scheduler_outcome(f'Job <29770179>, Status <{status}>, Queue <gpu\nl40s>', '29770179') in ('DONE', 'EXIT')


@pytest.mark.parametrize('raw', ['Job <29770179> is not found', 'Job <29770179>, Status <RUN>, Queue <gpul40s>', 'Job <29770179>, Status <DONE>, Queue <hpc>', 'Job <123>, Status <DONE>, Queue <gpul40s>'])
def test_missing_live_outcome_is_not_a_completed_native_study(raw):
    with pytest.raises(ValueError):
        contract.scheduler_outcome(raw, '29770179')
