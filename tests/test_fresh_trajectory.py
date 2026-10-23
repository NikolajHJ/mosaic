"""HPC-only fixtures for selection, denominators, immutable results and CLI budgets."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from latentmol.boltz.fresh_trajectory import load_protocol, native_command, execute_request
from latentmol.boltz.trajectory_selection import artifact_tree, verify_tree, planned_roles, _pilot_candidates
from latentmol.boltz.trajectory_report import build_report
from latentmol.experiments import file_sha256


def test_protocol_rejects_silent_cli_budget_or_selection_typos(tmp_path):
    base = yaml.safe_load(Path('configs/boltz_fresh_trajectory.yaml').read_text())
    for group, key, value in [('native', 'affinity_recycles', 0), ('native', 'precision', 'float32'),
                              ('native', 'mw_correction', True), ('selection', 'random_sample_draws', 9),
                              ('selection', 'checkpoints', [0, 100, 50]), ('native', 'pose_seeds', [17, 17])]:
        changed = deepcopy(base); changed[group][key] = value
        path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError):
            load_protocol(path)


def test_all_planned_roles_exist_even_without_parent_outputs(tmp_path):
    protocol = load_protocol(Path('configs/boltz_fresh_trajectory.yaml'))
    def asset(name, value):
        path = tmp_path / name; path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': file_sha256(path)}
    targets = asset('targets.json', {'targets': [{'id': f'p{i}'} for i in range(10)]})
    protocol['parents']['panel']['protocol'] = asset('panel.yaml', {'counts': list(range(6, 16)), 'targets': targets})
    protocol['parents']['random']['protocol'] = asset('random.yaml', {'seeds': list(range(10))})
    roles = planned_roles(protocol)
    assert len(roles) == 122
    assert sum(r['cohort'] == 'panel' for r in roles) == 100
    assert sum(r['cohort'] == 'panel' and r['draws'] == 4 for r in roles) == 20
    assert sum((r['draws'] + 1) * 6 for r in roles) == 1788


def selection_fixture():
    role = {'id': 'a', 'cohort': 'random', 'arm': 'active_set', 'pilot_status': 'failed', 'full_budget_verified': False}
    members = []
    for step, draw, status, request, smiles in [(0, 0, 'invalid', None, None), (0, 1, 'valid', 'startdraw', 'CC'),
                                               (100, 0, 'valid', 'final', 'CO'), (100, 1, 'valid', 'final', 'CO')]:
        members.append({'id': f'a/{step}/{draw}', 'role': 'a', 'cohort': 'random', 'target': 't', 'step': step,
                        'draw': draw, 'kind': 'argmax' if draw == 0 else 'categorical', 'status': status,
                        'candidate': {'smiles': smiles, 'heavy_atoms': 2}, 'failure': 'invalid' if not smiles else None,
                        'requests': {'17': request} if request else {}})
    return {'roles': [role], 'members': members, 'requests': [{'id': 'startdraw'}, {'id': 'final'}]}


def test_invalid_start_stays_unpaired_duplicates_retain_sampling_weight():
    selection = selection_fixture()
    result = {'status': 'completed', 'retention_eligible': True, 'identity': {}, 'failure': None,
              'affinity': {'affinity_probability_binary': .6, 'affinity_pred_value': 1.2}}
    summary, rows = build_report(selection, {'startdraw': result, 'final': result}, [17])
    assert summary['planned_candidate_memberships'] == 4
    assert summary['unique_native_requests'] == 2
    assert not any(r['paired_eligible'] for r in rows)
    assert len([r for r in rows if r['request_id'] == 'final']) == 2
    assert rows[-1]['draw_start_delta_binary'] == 0
    assert not rows[-1]['full_budget_verified']
    assert summary['all_requests_terminal']


def test_changed_identity_score_is_reported_raw_but_excluded_from_retention():
    selection = selection_fixture()
    result = {'status': 'identity_ineligible', 'retention_eligible': False,
              'identity': {'failure': 'charge changed'}, 'failure': 'charge changed',
              'affinity': {'affinity_probability_binary': .99, 'affinity_pred_value': -2}}
    summary, rows = build_report(selection, {'final': result}, [17])
    assert rows[-1]['native_binary'] == .99
    assert not rows[-1]['retention_eligible']
    assert not summary['all_requests_terminal']
    with pytest.raises(ValueError, match='Unexpected'):
        build_report(selection, {'other': result, 'final': result}, [17])


def test_terminal_file_set_detects_added_deleted_and_modified_artifacts(tmp_path):
    file = tmp_path / 'affinity.json'; file.write_text('original')
    seal = artifact_tree(tmp_path)
    verify_tree(tmp_path, seal)
    file.write_text('modified')
    with pytest.raises(ValueError, match='file set'):
        verify_tree(tmp_path, seal)
    file.write_text('original')
    extra = tmp_path / 'new.json'; extra.write_text('added')
    with pytest.raises(ValueError):
        verify_tree(tmp_path, seal)
    extra.unlink(); file.unlink()
    with pytest.raises(ValueError):
        verify_tree(tmp_path, seal)


def test_native_command_uses_explicit_structure_affinity_and_msa_budgets(tmp_path):
    protocol = load_protocol(Path('configs/boltz_fresh_trajectory.yaml'))
    command = native_command(tmp_path / 'input.yaml', tmp_path / 'output', tmp_path / 'cache',
                             {'native_settings': protocol['native'], 'seed': 31})
    for key, expected in [('--sampling_steps', '100'), ('--sampling_steps_affinity', '100'),
                          ('--recycling_steps', '0'), ('--num_subsampled_msa', '128'), ('--max_msa_seqs', '8192'), ('--seed', '31')]:
        assert command[command.index(key) + 1] == expected
    assert '--affinity_mw_correction' not in command


def test_native_cli_failure_keeps_exact_requested_input_and_logs(tmp_path, monkeypatch):
    import latentmol.boltz.fresh_trajectory as module
    protocol = load_protocol(Path('configs/boltz_fresh_trajectory.yaml'))
    request = {'id': 'a' * 64, 'seed': 17, 'smiles': 'CO', 'native_settings': protocol['native']}
    target = {'target_sequence_sha256': hashlib.sha256(b'AA').hexdigest(), 'msa_sha256': 'b' * 64,
              'protein_sequence': 'AA', 'msa_path': '/msa.a3m',
              'template': {'version': 1, 'sequences': [{'protein': {'id': 'A', 'sequence': 'AA', 'msa': '/msa.a3m'}},
                                                    {'ligand': {'id': 'B', 'smiles': 'CC'}}],
                           'properties': [{'affinity': {'binder': 'B'}}]}}
    monkeypatch.setattr(module, 'run_logged', lambda *a, **k: {'returncode': 1, 'timed_out': False})
    result = execute_request(request, target, tmp_path / 'cache', tmp_path / 'attempt', protocol)
    assert result['status'] == 'native_failed' and not result['retention_eligible']
    parsed = yaml.safe_load(next((tmp_path / 'attempt').glob('*.yaml')).read_text())
    assert parsed['sequences'][1]['ligand']['smiles'] == 'CO'
    assert target['template']['sequences'][1]['ligand']['smiles'] == 'CC'


def test_failed_parent_gate_without_recorded_candidates_is_retained(tmp_path):
    config = {'optimizer': {'steps': 100}, 'diagnostics': {'hard_samples': 8}, 'arms': ['active_set']}
    record = {'status': 'failed', 'source_archive_sha256': 'a' * 64, 'config': config,
              'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    (tmp_path / 'run.json').write_text(json.dumps(record))
    observed, rows, proxies, warnings = _pilot_candidates(tmp_path, 'a' * 64)
    assert observed['status'] == 'failed' and not rows and not proxies


def test_partial_parent_retains_failed_gate_but_never_scores_its_states(tmp_path, monkeypatch):
    import latentmol.boltz.trajectory_selection as module
    config = {'optimizer': {'steps': 100}, 'diagnostics': {'hard_samples': 8}, 'arms': ['active_set']}
    record = {'status': 'failed', 'source_archive_sha256': 'a' * 64, 'config': config,
              'graph_spec_sha256': 'b' * 64,
              'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    (tmp_path / 'run.json').write_text(json.dumps(record))
    folder = tmp_path / 'active_set'; folder.mkdir()
    (folder / 'candidates.jsonl').write_text('retained')
    (folder / 'gates.json').write_text(json.dumps({
        'step-0000': {'passed': True, 'native_seed_parity': True, 'context_sha256': 'c' * 64},
        'step-0010': {'passed': False, 'native_seed_parity': True, 'context_sha256': 'd' * 64}}))
    for step, context in [(0, 'c'), (10, 'd')]:
        row = {'step': step, 'context_sha256': context * 64, 'score': {'raw': .1},
               'natural_selection': {'count': 2}, 'selection': {'count': 2}}
        (folder / f'metrics-{step:04d}.json').write_text(json.dumps(row))
    monkeypatch.setattr(module, '_read_candidates', lambda *a: [{'step': 0}, {'step': 10}])
    _, rows, proxies, warnings = _pilot_candidates(tmp_path, 'a' * 64)
    assert len(rows['active_set']) == 2
    assert set(proxies['active_set']) == {0}
    assert any(w.get('unverified_step') == 10 for w in warnings)


def test_runtime_cannot_start_native_requests_outside_hpc(tmp_path, monkeypatch):
    from latentmol.boltz.fresh_trajectory import run
    monkeypatch.delenv('LSB_JOBID', raising=False)
    with pytest.raises(ValueError, match='DTU allocation'):
        run(tmp_path / 'config.yaml', 'a' * 64)


def test_controller_resume_does_not_retry_terminal_failures_and_rejects_tampering(tmp_path, monkeypatch):
    """Exercise durable controller boundaries using zero-cost native fixtures."""
    import latentmol.boltz.fresh_trajectory as module
    protocol = load_protocol(Path('configs/boltz_fresh_trajectory.yaml'))
    config = tmp_path / 'protocol.yaml'
    cache = tmp_path / 'cache'; cache.mkdir()
    for kind, key in [('aff', 'affinity'), ('conf', 'confidence')]:
        p = cache / f'boltz2_{kind}.ckpt'; p.write_text(kind)
        protocol['checkpoints'][key] = file_sha256(p)
    protocol.update(cache=str(cache), output_root=str(tmp_path / 'study'))
    asset = tmp_path / 'asset.yaml'; asset.write_text('fixture')
    for parent in protocol['parents'].values():
        for key in ('protocol', 'template'):
            if key in parent:
                parent[key] = {'path': str(asset), 'sha256': file_sha256(asset)}
    config.write_text(yaml.safe_dump(protocol))
    (tmp_path / 'release_manifest.json').write_text(json.dumps({'sha256': 'source-fixture'}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('LSB_JOBID', 'fixture')
    monkeypatch.setenv('LATENTMOL_SOURCE_SHA256', 'a' * 64)
    monkeypatch.setattr(module, 'source_manifest', lambda path: {'sha256': 'source-fixture'})
    monkeypatch.setattr(module, 'installed_boltz_provenance', lambda: {'commit': 'fixture'})
    selection = selection_fixture()
    msa = tmp_path / 'msa.a3m'; msa.write_text('>q\nAA\n')
    selection.update(parent_seals={}, targets={'t': {'msa_path': str(msa), 'msa_sha256': file_sha256(msa)}})
    for request in selection['requests']:
        request['target_key'] = 't'
    monkeypatch.setattr(module, 'freeze_selection', lambda _: selection)
    calls = []
    def execute(request, target, cache, attempt, protocol):
        calls.append(request['id']); attempt.mkdir(parents=True)
        (attempt / 'native.log').write_text('retained fixture failure')
        return {'id': request['id'], 'status': 'native_failed', 'failure': 'fixture',
                'retention_eligible': False, 'affinity': None, 'identity': None,
                'native_process': {'seconds': .1}}
    monkeypatch.setattr(module, 'execute_request', execute)
    root = module.run(config, file_sha256(config))
    assert calls == ['startdraw', 'final']
    assert json.loads((root / 'summary.json').read_text())['all_requests_terminal']
    module.run(config, file_sha256(config), resume=True)
    assert len(calls) == 2  # Failed native requests are a denominator, not retries.
    next((root / 'requests').rglob('native.log')).write_text('changed')
    with pytest.raises(ValueError, match='Sealed file set changed'):
        module.run(config, file_sha256(config), resume=True)
