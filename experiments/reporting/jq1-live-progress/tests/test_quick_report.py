"""Reusable command, cache corruption, scope and real CLI controls; allocated CPU."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import quick_inputs as inputs
import quick_runtime as runtime
import report as cli

SPEC = importlib.util.spec_from_file_location('_quick_report_saved_fixture', Path(__file__).with_name('test_report_data.py'))
saved = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(saved)


def fixture_arm(root):
    arm, segment, states, updates = saved.make_segment(root)
    arm['primary_metric'] = 'contact_loss'
    arm['expected_counts'] = {'committed_updates': 20, 'scalar_states': 21, 'retained_frames': 2, 'latest_global_step': 20}
    return arm, segment, states, updates


def make_seed(root, arm):
    output = root / 'prior'
    value = runtime.render_plan({'edition': 'edition-0001', 'captured_at': '2026-01-01T00:00:00Z', 'arms': [arm]}, output, 'f'*64, [])
    rows = {p.name: {'bytes': p.stat().st_size, 'sha256': inputs.sha(p)} for p in output.iterdir()}
    return {'root': output, 'plan': {'arms': [deepcopy(arm)]}, 'manifest': value,
            'receipt': {'files': rows}, 'renderer_sha256': 'f'*64}


def test_selection_key_ignores_capture_time_and_legacy_missing_metric(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    legacy = deepcopy(arm)
    del legacy['primary_metric']
    assert inputs.selection_key([arm], 'f'*64) == inputs.selection_key([legacy], 'f'*64)


@pytest.mark.parametrize('field,value', [('rate', .1), ('global_target', 50), ('scientific_status', 'passed')])
def test_scientific_changes_never_reuse_old_cache(tmp_path, field, value):
    arm, _, _, _ = fixture_arm(tmp_path)
    key = inputs.science_key(arm, 'f'*64)
    arm[field] = value
    assert inputs.science_key(arm, 'f'*64) != key


def test_changed_original_input_hash_invalidates_cache(tmp_path):
    arm, segment, _, _ = fixture_arm(tmp_path)
    changed = deepcopy(arm)
    changed['segments'][0]['states']['sha256'] = 'a'*64
    assert inputs.science_key(arm, 'f'*64) != inputs.science_key(changed, 'f'*64)


def test_display_wording_does_not_invalidate_scientific_assets(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    changed = deepcopy(arm)
    changed['label'] = 'A clearer title'
    assert inputs.science_key(arm, 'f'*64) == inputs.science_key(changed, 'f'*64)
    assert inputs.selection_key([arm], 'f'*64) != inputs.selection_key([changed], 'f'*64)


@pytest.mark.parametrize('count', [1, 2, 3, 7])
def test_any_supported_cohort_renders_with_same_saved_frame_validation(tmp_path, count):
    arms = []
    for i in range(count):
        root = tmp_path / str(i)
        root.mkdir()
        arm, _, _, _ = fixture_arm(root)
        arm['id'] = 'run-' + str(i)
        arms.append(arm)
    value = runtime.render_plan({'edition': 'edition-0001', 'captured_at': '2026-01-01T00:00:00Z', 'arms': arms}, tmp_path / 'out', 'f'*64, [])
    assert len(value['arms']) == count
    assert all(a['committed_updates'] == 20 and a['frames']['count'] == 2 and a['scientific_status'] == 'failed' for a in value['arms'])


def test_cache_hit_skips_array_revalidation_but_rechecks_every_original_byte(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    seed = make_seed(tmp_path, arm)
    import report_data
    with patch.object(report_data, 'collect_arm', side_effect=AssertionError('Unchanged arm was rebuilt')):
        value = runtime.render_plan({'edition': 'edition-0002', 'captured_at': '2026-01-02T00:00:00Z', 'arms': [arm]}, tmp_path / 'new', 'f'*64, [seed])
    assert value['arms'] == seed['manifest']['arms']
    assert inputs.load(tmp_path / 'new/assessment.json')['cached_arms'] == ['protenix']


def test_cache_cannot_hide_changed_snapshot_bytes(tmp_path):
    arm, segment, states, _ = fixture_arm(tmp_path)
    seed = make_seed(tmp_path, arm)
    path = Path(segment['anchor_directory']) / states[0]['snapshot']
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='length changed|SHA changed'):
        runtime.render_plan({'edition': 'edition-0002', 'captured_at': 'now', 'arms': [arm]}, tmp_path / 'new', 'f'*64, [seed])


def test_cache_cannot_hide_changed_asset_bytes(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    seed = make_seed(tmp_path, arm)
    path = seed['root'] / seed['manifest']['arms'][0]['scalars']['path']
    path.write_bytes(b'corrupt')
    out = tmp_path / 'new'
    out.mkdir()
    with pytest.raises(ValueError, match='asset bytes changed'):
        runtime.cached_arm(arm, 'f'*64, [seed], out)


def test_other_renderer_cannot_reuse_old_cache(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    seed = make_seed(tmp_path, arm)
    out = tmp_path / 'new'
    out.mkdir()
    assert runtime.cached_arm(arm, 'e'*64, [seed], out) is None


def test_cache_keeps_new_display_status_without_changing_original_verdict(tmp_path):
    arm, _, _, _ = fixture_arm(tmp_path)
    seed = make_seed(tmp_path, arm)
    arm['continuation_status'] = 'blocked'
    out = tmp_path / 'new'
    out.mkdir()
    value = runtime.cached_arm(arm, 'f'*64, [seed], out)
    assert value['continuation_status'] == 'blocked' and value['scientific_status'] == 'failed'


@pytest.mark.parametrize('phase', ['gpu', 'training', 'optimizer', 'after-cpu'])
def test_only_two_CPU_report_phases_can_be_submitted(tmp_path, phase):
    with pytest.raises(ValueError):
        runtime.bsub_argv({'python': '/python'}, '/source', 'f'*64, phase, tmp_path, tmp_path/'request.json')


@pytest.mark.parametrize('phase', ['validate', 'render'])
def test_real_startup_argv_requires_exact_allocation_and_source(tmp_path, monkeypatch, phase):
    source = tmp_path / 'source'
    source.mkdir()
    profile = {'python': '/python'}
    runtime.write(source/'profile.json', profile)
    parent = tmp_path/'validation' if phase == 'validate' else tmp_path/'request'
    parent.mkdir()
    request = parent/'request.json'
    runtime.write(request, {'selection': {}})
    args = runtime.bsub_argv(profile, source, 'f'*64, phase, parent, request)
    runtime.write(parent/'submission.json', {'job_id': '123', 'argv': args})
    monkeypatch.setenv('LSB_JOBID', '123')
    monkeypatch.setenv('LSB_QUEUE', 'hpc')
    monkeypatch.setenv('LSB_DJOB_NUMPROC', '4')
    with patch.object(runtime, 'verify_source'), patch.object(runtime, phase) as action:
        runtime.main([phase, '--source', str(source), '--pin', 'f'*64, '--request', str(request)])
    action.assert_called_once()
    monkeypatch.setenv('LSB_QUEUE', 'gpul40s')
    with patch.object(runtime, 'verify_source'), pytest.raises(ValueError, match='command/resources'):
        runtime.main([phase, '--source', str(source), '--pin', 'f'*64, '--request', str(request)])


def test_submission_unknown_marker_prevents_duplicate(tmp_path):
    profile = {'python': '/python'}
    with patch.object(runtime.subprocess, 'run', side_effect=TimeoutError('unknown')):
        with pytest.raises(TimeoutError):
            runtime.submit(profile, '/source', 'f'*64, 'render', tmp_path, tmp_path/'request.json')
    assert (tmp_path/'submission-started.json').exists() and not (tmp_path/'submission.json').exists()
    with patch.object(runtime.subprocess, 'run') as submit:
        with pytest.raises(FileExistsError):
            runtime.submit(profile, '/source', 'f'*64, 'render', tmp_path, tmp_path/'request.json')
    submit.assert_not_called()


def test_actual_local_CLI_dry_run_never_connects_or_submits(tmp_path):
    profile = tmp_path/'profile.json'
    profile.write_text('{}')
    run = subprocess.run([sys.executable, str(SCRIPTS/'report.py'), '--dry-run', '--run', '/a/saved/study',
                          '--profile', str(profile), '--output', str(tmp_path/'viewer')], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    value = json.loads(run.stdout)
    assert value['status'] == 'dry_run_no_SSH_or_submission' and value['selection']['runs'] == ['/a/saved/study']
    assert not (tmp_path/'viewer').exists()


def test_generated_remote_actions_are_real_python_not_shell_interpolation(tmp_path):
    import ast
    connection = cli.Connection({'ssh_host': 'dtu-hpc', 'python': '/python'}, tmp_path)
    programs = []
    connection.run = lambda code, label, timeout=300: programs.append(code)
    for action in ('setup', 'certify', 'capture', 'submit', 'recover_submission', 'status', 'delivery'):
        connection.action('/a/source', 'f'*64, action, {'label': "literal $(not-a-command) and ' quote"})
    assert len(programs) == 7
    for code in programs:
        ast.parse(code)


def test_existing_different_evidence_is_never_overwritten(tmp_path):
    path = tmp_path/'proof.json'
    cli.record(path, {'same': True})
    cli.record(path, {'same': True})
    with pytest.raises(ValueError):
        cli.record(path, {'same': False})


def test_terminal_capture_rejects_uncommitted_suffix_and_noncontiguous_rows(tmp_path):
    states = tmp_path/'states'
    updates = tmp_path/'updates'
    states.write_bytes(b'{"step":0,"snapshot_retained":true}\n{"step":1}')
    updates.write_bytes(b'')
    with pytest.raises(ValueError, match='incomplete'):
        inputs.counts(states, updates, 0)
    states.write_bytes(b'{"step":1,"snapshot_retained":true}\n')
    with pytest.raises(ValueError, match='contiguous'):
        inputs.counts(states, updates, 0)


@pytest.mark.parametrize('status,ending', [('passed', 'Successfully completed.'), ('failed', 'Exited with exit code 1.')])
def test_original_scientific_verdict_requires_its_own_LSF(tmp_path, status, ending):
    path = tmp_path/'stdout.log'
    path.write_text('Subject: Job 123: result\n' + ending)
    inputs.check_lsf(path, '123', status)
    with pytest.raises(ValueError):
        inputs.check_lsf(path, '124', status)
    with pytest.raises(ValueError):
        inputs.check_lsf(path, '123', 'failed' if status == 'passed' else 'passed')


def test_relative_or_linked_saved_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        inputs.regular(Path('relative.json'))
    path = tmp_path/'real'
    path.write_text('x')
    linked = tmp_path/'linked'
    linked.symlink_to(path)
    with pytest.raises(ValueError, match='not regular|Linked'):
        inputs.binding(linked)


def delta_fixture(root, malicious=False):
    files = {'manifest.json': inputs.encoded({'edition': 'edition-0001', 'arms': [{'id': 'a', 'committed_updates': 1, 'frames': {'count': 1}}]}),
             'report.html': b'<html></html>', 'input-plan.json': b'{}', 'assessment.json': b'{}',
             'cpu-certificate.json': b'{"status":"passed"}', 'a-scalars.json': b'{"rows":[]}'}
    import hashlib
    sha = lambda b: hashlib.sha256(b).hexdigest()
    receipt = {'engine_pin': 'f'*64, 'job_id': '123', 'cpu_certificate_sha256': sha(files['cpu-certificate.json']),
               'files': {n: {'bytes': len(b), 'sha256': sha(b)} for n, b in files.items()}}
    files['report-receipt.json'] = inputs.encoded(receipt)
    cache = root/'cached-scalar'
    cache.write_bytes(files['a-scalars.json'])
    archive = root/'delta.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name, raw in files.items():
            if name == 'a-scalars.json':
                continue
            item = tarfile.TarInfo('../escape' if malicious and name == 'report.html' else name)
            item.size = len(raw)
            bundle.addfile(item, io.BytesIO(raw))
    delivery = {'edition': 'edition-0001', 'receipt_sha256': sha(files['report-receipt.json']), 'job_id': '123',
                'arms': {'a': {'updates': 1, 'frames': 1}}, 'omitted_verified_local_assets': {'a-scalars.json': receipt['files']['a-scalars.json']}}
    return archive, delivery, {sha(files['a-scalars.json']): cache}, receipt['cpu_certificate_sha256']


def test_delta_reconstructs_verified_cache_before_atomic_publication(tmp_path):
    archive, delivery, cache, cpu_sha = delta_fixture(tmp_path)
    viewer = tmp_path/'viewer'
    pointer = cli.publish_delta(archive, delivery, viewer, cache, 'f'*64, cpu_sha)
    assert pointer == inputs.load(viewer/'latest.json')
    assert (viewer/'editions/edition-0001/a-scalars.json').read_bytes() == b'{"rows":[]}'
    with pytest.raises(ValueError, match='already exists'):
        cli.publish_delta(archive, delivery, viewer, cache, 'f'*64, cpu_sha)


@pytest.mark.parametrize('problem', ['traversal', 'missing-cache', 'changed-cache', 'wrong-CPU'])
def test_bad_delta_never_updates_latest_pointer(tmp_path, problem):
    archive, delivery, cache, cpu_sha = delta_fixture(tmp_path, malicious=problem == 'traversal')
    if problem == 'missing-cache':
        cache = {}
    if problem == 'changed-cache':
        next(iter(cache.values())).write_bytes(b'corrupt')
    if problem == 'wrong-CPU':
        cpu_sha = 'a'*64
    viewer = tmp_path/'viewer'
    with pytest.raises(ValueError):
        cli.publish_delta(archive, delivery, viewer, cache, 'f'*64, cpu_sha)
    assert not (viewer/'latest.json').exists()


def registered_fixture(root):
    arm, _, _, _ = fixture_arm(root)
    seed = make_seed(root, arm)
    source = root/'engines'/('f'*64)/'source'
    (source/'scripts').mkdir(parents=True)
    (source/'scripts/report_data.py').write_bytes(saved.SOURCE.read_bytes())
    parent = root/'requests/request-fixture'
    parent.mkdir(parents=True)
    import shutil
    shutil.copytree(seed['root'], parent/'viewer')
    plan = {'edition': 'edition-0001', 'arms': [arm]}
    runtime.write(parent/'viewer/input-plan.json', plan)
    cert = {'status': 'passed', 'cpu_job_id': '88', 'engine_pin': 'f'*64}
    runtime.write(parent/'viewer/cpu-certificate.json', cert)
    runtime.write(parent/'submission.json', {'job_id': '123'})
    runtime.write(parent/'terminal-scheduler.json', {'returncode': 0, 'stdout': 'Job <123> Status <DONE> Queue <hpc>'})
    (parent/'stdout.123.log').write_text('Subject: Job 123: report\nSuccessfully completed.\n')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': inputs.sha(p)} for p in (parent/'viewer').iterdir()}
    receipt = {'status': 'passed_saved_only_presentation', 'engine_pin': 'f'*64, 'job_id': '123', 'cpu_job_id': '88',
               'cpu_certificate_sha256': inputs.sha(parent/'viewer/cpu-certificate.json'), 'files': files}
    runtime.write(parent/'viewer/report-receipt.json', receipt)
    runtime.write(parent/'delivery.json', {'job_id': '123', 'receipt_sha256': inputs.sha(parent/'viewer/report-receipt.json')})
    return source, parent, cert


def test_successful_new_reports_become_verified_reusable_cache_without_profile_edits(tmp_path):
    source, parent, cert = registered_fixture(tmp_path)
    profile = {'remote_root': str(tmp_path)}
    with patch.object(runtime, 'verify_source'), patch.object(runtime, 'certificate', return_value=cert):
        runtime.register_cache(profile, source, 'f'*64, parent)
        runtime.register_cache(profile, source, 'f'*64, parent)
        seeds = runtime.registered_seeds(profile)
    assert len((tmp_path/'cache-index.jsonl').read_text().splitlines()) == 1
    assert len(seeds) == 1 and seeds[0]['root'] == parent/'viewer'
    assert seeds[0]['manifest']['arms'][0]['scientific_status'] == 'failed'


def test_failed_or_running_reports_never_become_cache_entries(tmp_path):
    source, parent, cert = registered_fixture(tmp_path)
    (parent/'terminal-scheduler.json').write_bytes(inputs.encoded({'returncode': 0, 'stdout': 'Job <123> Status <RUN> Queue <hpc>'}))
    with patch.object(runtime, 'certificate', return_value=cert), pytest.raises(ValueError):
        runtime.register_cache({'remote_root': str(tmp_path)}, source, 'f'*64, parent)
    assert not (tmp_path/'cache-index.jsonl').exists()


def test_registered_cache_rejects_later_provenance_mutation(tmp_path):
    source, parent, cert = registered_fixture(tmp_path)
    profile = {'remote_root': str(tmp_path)}
    with patch.object(runtime, 'verify_source'), patch.object(runtime, 'certificate', return_value=cert):
        runtime.register_cache(profile, source, 'f'*64, parent)
        (parent/'viewer/input-plan.json').write_bytes(b'changed')
        with pytest.raises(ValueError, match='plan/manifest changed'):
            runtime.registered_seeds(profile)
