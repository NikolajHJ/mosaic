"""Focused conditional-tail controls; no frozen scorer or HPC submission."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

OWNER = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(OWNER/'scripts'), str(OWNER)]
import chemistry_screen_control as control
import chemistry_screen_ops as ops


def parent_plan(root):
    screen = {'version': 'protenix-chemistry-screen-v1', 'updates': 50000, 'temperature': .7,
              'temperature_schedule': {'version': 'linear-temperature-v1', 'initial': .7, 'final': .4, 'total_updates': 50000},
              'chemistry': {'mode': 'ramp_hold', 'initial': 1., 'final': 30., 'hold_updates': 20000, 'ramp_end': 30000},
              'logit_noise': {'version': 'binding-logit-noise-linear-v1', 'initial_ratio': .4, 'zero_at_update': 30000, 'seed': 230017}}
    return {'root': str(root), 'source_root': str(root/'source'), 'code_source_root': str(root/'source'),
            'arms': {'parent': {'name': 'parent', 'screen': screen, 'rate': .3, 'model_key': 17,
                                'gpu_profile': 'v100-32gb-v1', 'queue': 'gpuv100',
                                'frozen_identity': {'model_digest': 'model', 'context_digest': 'context'},
                                'output_cap_bytes': 2*1024**3, 'file_cap': 7000}},
            'conditional_tail': {'parent_arm': 'parent', 'target_updates': 70000, 'coefficient': 100.,
                                 'wall_minutes': 255, 'output_cap_bytes': 1024**3, 'file_cap': 3500}}


def final_proof(valid):
    return {'parent_job_id': '123', 'parent_arm': 'parent', 'global_step': 50000,
            'state': {'step': 50000, 'global_step': 50000, 'temperature': .4,
                      'repair_weight': 30., 'noise_state': {'next_update': 50000, 'rng_state': {'stream': 'preserved'}}},
            'snapshot': {'path': '/parent/step-50000.npz', 'sha256': 'snapshot', 'bytes': 100},
            'reference': {'path': '/parent/step-50000.json.gz', 'sha256': 'reference', 'bytes': 200},
            'strict_valid_connected': valid}


def test_tail_preserves_schedule_rng_and_inputs_changes_only_requested_parameters(tmp_path):
    plan = parent_plan(tmp_path); untouched = deepcopy(plan); proof = final_proof(False)
    child, name = control.make_tail_plan(plan, proof['state'], proof['snapshot'], proof['reference'], '123')
    assert plan == untouched
    assert name == 'parent-repair20k' and list(child['arms']) == [name]
    row = child['arms'][name]
    assert row['screen']['updates'] == 70000 and row['screen']['chemistry'] == {'mode': 'constant', 'coefficient': 100.}
    for key in ('temperature_schedule', 'logit_noise'):
        assert row['screen'][key] == untouched['arms']['parent']['screen'][key]
    assert row['resume']['offset'] == 50000 and row['resume']['policy_change'] is True
    assert row['resume']['state'] == proof['state'] and row['resume']['snapshot'] == proof['snapshot']
    assert row['resume']['reference'] == proof['reference'] and row['predecessor_job_ids'] == ['123']
    assert row['output_cap_bytes'] == 1024**3 and row['file_cap'] == 3500
    assert row['internal_seconds'] == 255*60-600
    assert 'conditional_tail' not in child and child['root'] == plan['root']
    for key in ('rate', 'model_key', 'frozen_identity', 'gpu_profile', 'queue'):
        assert row[key] == untouched['arms']['parent'][key]


@pytest.mark.parametrize('change', [{'target_updates': 80000}, {'coefficient': 30.}, {'wall_minutes': 0}, {'arm_name': '..'}, {'arm_name': 'parent'}])
def test_unrequested_or_escaping_tail_rejected(tmp_path, change):
    plan = parent_plan(tmp_path); plan['conditional_tail'].update(change); proof = final_proof(False)
    with pytest.raises(ValueError):
        control.make_tail_plan(plan, proof['state'], proof['snapshot'], proof['reference'], '123')


@pytest.mark.parametrize('rows,expected,error', [
    ([{'global_step': 0}, {'global_step': 1}], 2, None),
    ([{'global_step': 0}, {'global_step': 2}], 2, 'Noncontiguous'),
    ([{'global_step': 0}], 2, 'not reached'),
    ([{'global_step': 0}, {'global_step': 1}], 1, 'extra'),
])
def test_only_complete_contiguous_terminal_prefix_accepted(tmp_path, rows, expected, error):
    path = tmp_path/'prefix.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if error:
        with pytest.raises(ValueError, match=error): control.read_committed_prefix(path, expected)
    else:
        assert control.read_committed_prefix(path, expected) == rows[-1]


def test_terminal_partial_line_is_not_silently_ignored(tmp_path):
    path = tmp_path/'prefix.jsonl'; path.write_text('{"global_step":0}\n{"global_step":1}')
    with pytest.raises(ValueError, match='Incomplete'):
        control.read_committed_prefix(path, 2)


@pytest.mark.parametrize('valid', [True, False])
def test_callback_skips_valid_endpoint_and_submits_one_invalid_tail(tmp_path, monkeypatch, valid):
    plan = parent_plan(tmp_path); path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    proof = final_proof(valid); calls = []
    monkeypatch.setenv('LSB_JOBID', '456')
    monkeypatch.setattr(control, 'verify_completed_parent', lambda *_: proof)
    monkeypatch.setattr(control, 'capacity', lambda *_: {'status': 'passed'})
    monkeypatch.setattr(ops, 'command', lambda *args: {'returncode': 0, 'argv': args[0], 'stdout': '', 'stderr': ''})
    monkeypatch.setattr(control, 'submit', lambda *args: calls.append(args) or '789')
    control.conditional(plan, path)
    result = json.loads((tmp_path/'conditional/assessment.json').read_bytes())
    assert result['saved_results_changed'] is False
    if valid:
        assert result['status'] == 'skipped_endpoint_already_valid' and calls == []
        assert not (tmp_path/'conditional/child-plan.json').exists()
    else:
        assert result['status'] == 'invalid_endpoint_tail_submitted' and len(calls) == 1
        assert result['job_id'] == '789' and calls[0][1:] == ('run', 'parent-repair20k', 255)
        child = json.loads((tmp_path/'conditional/child-plan.json').read_bytes())
        assert child['arms']['parent-repair20k']['resume']['policy_change'] is True
    with pytest.raises(FileExistsError): control.conditional(plan, path)
    assert len(calls) == int(not valid)


def test_incomplete_parent_never_dispatches_child(tmp_path, monkeypatch):
    monkeypatch.setenv('LSB_JOBID', '456')
    def reject(*args): raise ValueError('Incomplete parent')
    monkeypatch.setattr(control, 'verify_completed_parent', reject)
    monkeypatch.setattr(control, 'submit', lambda *args: pytest.fail('Must not submit incomplete parent'))
    with pytest.raises(ValueError, match='Incomplete parent'):
        control.conditional(parent_plan(tmp_path), tmp_path/'plan.json')
    assert json.loads((tmp_path/'conditional/failure.json').read_bytes())['status'] == 'failed_no_blind_retry'


def save_endpoint(path, invalid=False):
    # Numerical cases are run only by the allocated CPU controls job.
    import jax
    import jax.numpy as jnp
    import numpy as np
    import protenix_chemistry_screen as screen
    config = {'initialization': dict(screen.CHAIN_MIX), 'init_seed': 17, 'temperature': .4}
    point = screen.initialize(config, None, screen.logits_class())
    if invalid:
        atoms = point.atoms.at[1].set(jnp.log(jnp.asarray([.002, .002, .002, .002, .99, .002], dtype=jnp.float32)))
        point = type(point)(atoms, point.bonds, .4)  # Interior fluorine has two chain bonds.
    graph = point.graph()
    np.savez_compressed(path, X=np.asarray(graph.X), E_upper=np.asarray(graph.E_upper),
                        atom_logits=np.asarray(point.atoms), bond_logits=np.asarray(point.bonds),
                        key_data=np.asarray(jax.random.key_data(jax.random.key(17))),
                        step=np.asarray(50000), temperature=np.asarray(.4, dtype=np.float32))


@pytest.mark.parametrize('invalid', [False, True])
def test_saved_endpoint_readout_checks_actual_graph_not_teacher_or_probability_threshold(tmp_path, invalid):
    path = tmp_path/'endpoint.npz'; save_endpoint(path, invalid)
    readout = control.inspect_endpoint_snapshot(path, 50000, .4, 17)
    valid = readout.get('sanitized') is True and readout.get('fragment_count') == 1
    assert valid is not invalid
    with pytest.raises(ValueError, match='key'):
        control.inspect_endpoint_snapshot(path, 50000, .4, 23)


def test_complete_endpoint_verifier_accepts_saved_FP32_temperature(tmp_path, monkeypatch):
    import gzip
    import shutil
    from types import SimpleNamespace
    import numpy as np
    import protenix_chemistry_screen as screen
    plan = parent_plan(tmp_path); source = tmp_path/'source'; source.mkdir()
    (source/'checksums.json').write_text('{}')
    original = tmp_path/'original.json'; original.write_text('{}')
    for name in ('config', 'snapshot', 'reference', 'pocket', 'context_ready'):
        plan['arms']['parent'][name] = screen.binding(original)
    plan_path = tmp_path/'plan.json'; plan_path.write_text(json.dumps(plan)); parent = tmp_path/'production/parent'; parent.mkdir(parents=True)
    sub = {'job_id': '123', 'kind': 'run', 'arm': 'parent', 'plan': screen.binding(plan_path),
           'job_name': 'new-parent', 'argv': ['bsub', 'bash', '-l', '/stage']}
    (parent/'submission.json').write_text(json.dumps(sub)); (parent/'release.json').write_text('{"returncode":0}')
    (parent/'stdout.123.log').write_text('Subject: Job 123: <new-parent> in cluster <dcc> Done\nSuccessfully completed.\n')
    (tmp_path/'jobs.json').write_text('{"parent":"123"}'); own = tmp_path/'jobs/conditional'; own.mkdir(parents=True)
    (own/'submission.json').write_text(json.dumps({'job_id': '456', 'kind': 'conditional',
        'plan': screen.binding(plan_path), 'argv': ['bsub', '-w', 'done(123)']}))
    study = tmp_path/'parent/payload/study'; (study/'snapshots').mkdir(parents=True); (study/'references').mkdir()
    point = study/'snapshots/step-50000.npz'; save_endpoint(point)
    shutil.copyfile(point, study/'latest-state.npz')
    readout = control.inspect_endpoint_snapshot(point, 50000, .4, 17)
    state = {'step': 50000, 'global_step': 50000, 'temperature': float(np.float32(.4)), 'repair_weight': 30.,
             'snapshot_retained': True, 'snapshot': str(point), 'snapshot_sha256': screen.binding(point)['sha256'],
             'latest_snapshot': screen.binding(study/'latest-state.npz'), 'readout': readout, 'strict_valid_connected': True,
             'noise_state': {'next_update': 50000, 'policy': plan['arms']['parent']['screen']['logit_noise']},
             'preparation_identity': 'teacher', 'neighbor_preparation_identity': None}
    for name in ('states.jsonl', 'updates.jsonl'): (study/name).write_text(json.dumps(state)+'\n')
    (study/'report.json').write_text(json.dumps({'status': 'completed_exploratory_updates', 'applied_updates': 50000, 'last_state': state}))
    with gzip.open(study/'references/step-50000.json.gz', 'wt') as stream: json.dump({}, stream)
    (study.parent/'summary.json').write_text(json.dumps({'job_id': '123', 'arm': 'parent',
        'status': 'completed_exploratory_updates', 'applied_updates': 50000, 'actual_steps': 50000,
        'full_budget_complete': True, 'probe_not_adopted': False, 'configuration': plan['arms']['parent']['screen'],
        'frozen_before': plan['arms']['parent']['frozen_identity'], 'frozen_after': plan['arms']['parent']['frozen_identity']}))
    monkeypatch.setenv('LSB_JOBID', '456')
    monkeypatch.setattr(ops, 'command', lambda *_: {'returncode': 0, 'stdout': json.dumps({'RECORDS': [{
        'JOBID': '123', 'USER': 's234818', 'STAT': 'DONE', 'QUEUE': 'gpuv100', 'JOB_NAME': 'new-parent', 'COMMAND': 'bash -l /stage'}]})})
    # Prefix continuity/counts are separately tested above; here isolate the complete metadata/readout path.
    monkeypatch.setattr(control, 'read_committed_prefix', lambda path, count: deepcopy(state))
    monkeypatch.setattr(screen, 'restore_teacher', lambda *_: (SimpleNamespace(identity='teacher'), None))
    result = control.verify_completed_parent(plan, plan_path)
    assert result['strict_valid_connected'] is True and result['global_step'] == 50000
    assert result['state']['temperature'] == float(np.float32(.4)) and result['scorer_evaluations'] == 0
