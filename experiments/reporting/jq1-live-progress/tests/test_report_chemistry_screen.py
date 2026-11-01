"""Saved-array adapter fixtures; execute on allocated report CPU only."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import report_data as data
import report_chemistry_screen as adapter
import quick_inputs as inputs
import quick_runtime as runtime

spec = importlib.util.spec_from_file_location('_screen_saved_fixture', Path(__file__).with_name('test_report_data.py'))
saved = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saved)


def fixture(root):
    name = 'constant1-s17'
    run = root / 'run'
    parent = run / name / 'payload'
    parent.mkdir(parents=True)
    old_arm, old, states, updates = saved.make_segment(parent, name='study')
    study = parent / 'study'
    config = {'version': adapter.VERSION, 'updates': 20, 'temperature': 1.,
              'chemistry': {'mode': 'constant', 'coefficient': 1.},
              'sampling_interval': 20, 'sampling_draws': 128}
    identity = {'model_digest': 'a' * 64, 'context_digest': 'b' * 64}
    source = root / 'source'
    source.mkdir()
    science = root / 'science'
    science.mkdir()
    scientific = saved.save(science / 'source_manifest.json', {'files': {}})
    row = {'screen': config, 'rate': .3, 'name': name, 'model_key': 17,
           'config': old['config'], 'frozen_identity': identity,
           'science_source': {'path': str(science), 'manifest_sha256': scientific['sha256']}}
    plan = saved.save(source / 'plan.json', {'root': str(run), 'source_root': str(source), 'arms': {name: row}})
    driver = source / 'protenix_chemistry_screen.py'
    driver.write_text('# fixture producer identity\n')
    saved.save(source / 'checksums.json', {'protenix_chemistry_screen.py': data.sha(driver),
                                          'plan.json': plan['sha256']})
    production = run / 'production' / name
    production.mkdir(parents=True)
    saved.save(production / 'submission.json', {'kind': 'run', 'arm': name, 'job_id': '123', 'plan': plan})
    (production / 'stdout.123.log').write_text('Subject: Job 123: fixture\nSuccessfully completed.\n')
    (production / 'stderr.123.log').write_text('')
    for state in states:
        state.update(temperature=1., chemistry_loss=.125, binding_weight=1.)
        state.pop('autodiff_surrogate')
        state.pop('previous_fixed_surrogate_after')
        if state['snapshot_retained']:
            path = study / state['snapshot']
            with np.load(path, allow_pickle=False) as z:
                arrays = {n: z[n].copy() for n in z.files if n not in ('refreshed_surrogate', 'autodiff_surrogate', 'repair_weight')}
            np.savez(path, **arrays, step=state['step'], temperature=np.asarray(1., np.float32))
            state.update(snapshot=str(path), snapshot_sha256=data.sha(path))
    for update in updates:
        update.update(rate=.3, candidate_count=1, binding_weight=1.)
        for key in ('attempts', 'selection_policy', 'selected_trial', 'accepted_step_size', 'fallback_used'):
            update.pop(key)
    saved.save_rows(study / 'states.jsonl', states)
    saved.save_rows(study / 'updates.jsonl', updates)
    samples = []
    (study / 'samples').mkdir()
    for step in (0, 20):
        raw = study / 'samples' / ('samples-' + str(step) + '.npz')
        np.savez(raw, atom_indices=np.zeros((128, 25), np.int16), bond_indices=np.zeros((128, 300), np.int16))
        samples.append({'step': step, 'draws': 128, 'raw_samples': saved.bind(raw),
                        'strict_valid_connected': 0, 'native_model_key_changed': False,
                        'binding_scores_evaluated_for_samples': False})
    saved.save_rows(study / 'sampled-validity.jsonl', samples)
    summary = {'version': adapter.VERSION, 'status': 'completed_exploratory_updates',
               'arm': name, 'job_id': '123', 'configuration': config, 'applied_updates': 20,
               'actual_steps': 20, 'full_budget_complete': True, 'probe_not_adopted': False,
               'derivative_qualified': False, 'original_results_changed': False,
               'frozen_before': identity, 'frozen_after': identity, 'report_path': str(study / 'report.json')}
    saved.save(parent / 'summary.json', summary)
    saved.save(study / 'report.json', {**summary, 'initial_state': states[0], 'last_state': states[-1]})
    profile = {'allowed_roots': [str(root)]}
    segment, _, _ = adapter.capture_segment({'id': name}, profile, parent, summary)
    arm = {**old_arm, 'id': name, 'label': 'Constant chemistry weight1', 'scientific_status': 'unqualified',
           'primary_metric': 'contact_loss', 'segments': [segment],
           'schema_renderer_sha256': inputs.sha(SCRIPTS / 'report_chemistry_screen.py'),
           'expected_counts': {'committed_updates': 20, 'scalar_states': 21, 'retained_frames': 2, 'latest_global_step': 20}}
    return arm, segment, summary, states, updates


def test_screen_reuses_assembler_preserves_scalars_and_unqualified_scope(tmp_path):
    arm, segment, _, states, _ = fixture(tmp_path)
    output = tmp_path / 'out'
    output.mkdir()
    original = data.collect_segment
    result = adapter.collect_arm(arm, output)
    assert data.collect_segment is original and result['scientific_status'] == 'unqualified'
    assert result['optimization_complete'] and result['committed_updates'] == 20
    assert result['frames']['steps'] == [0, 20]
    rows = data.load(output / result['scalars']['path'])['rows']
    assert [r['state'] for r in rows] == states
    assert all(r['losses']['chemistry_loss'] == .125 for r in rows)
    assert result['segment_proofs'][0]['sampled_validity'][1]['step'] == 20
    assert data.load(segment['summary']['path'])['status'] == 'completed_exploratory_updates'


@pytest.mark.parametrize('change', [
    {'status': 'running'}, {'applied_updates': 19}, {'full_budget_complete': False},
    {'probe_not_adopted': True}, {'derivative_qualified': True}, {'frozen_after': {}}])
def test_screen_incomplete_smoke_or_mutation_rejected(tmp_path, change):
    _, _, summary, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        adapter.header({**summary, **change})


def test_screen_changed_saved_frame_rejected(tmp_path):
    arm, _, _, states, _ = fixture(tmp_path)
    path = Path(states[-1]['snapshot'])
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA changed'):
        adapter.collect_segment(arm['segments'][0], arm)


def test_screen_changed_within_step_coefficient_rejected(tmp_path):
    arm, segment, _, _, updates = fixture(tmp_path)
    updates[1]['repair_weight'] = 2.
    segment['updates'] = saved.save_rows(Path(segment['updates']['path']), updates)
    with pytest.raises(ValueError, match='Within-step'):
        adapter.collect_segment(segment, arm)


def test_screen_sampling_binding_participates_in_cache_identity(tmp_path):
    arm, _, _, _, _ = fixture(tmp_path)
    changed = deepcopy(arm)
    changed['segments'][0]['sampled_validity']['sha256'] = 'f' * 64
    assert inputs.science_key(arm, 'a' * 64) != inputs.science_key(changed, 'a' * 64)


def test_runtime_routes_screen_and_keeps_original_frame_renderer(tmp_path):
    arm, _, _, _, _ = fixture(tmp_path)
    result = runtime.render_plan({'edition': 'edition-0001', 'captured_at': 'fixture', 'arms': [arm]},
                                 tmp_path / 'out', 'f' * 64, [])
    assert result['arms'][0]['scientific_status'] == 'unqualified'
    assert result['arms'][0]['committed_updates'] == 20


def test_screen_suite_and_existing_seven_fit_one_bounded_report():
    assert 7 + 22 <= inputs.MAX_RUNS == 32
