"""Allocated pure-metadata/inventory adapter fixtures; no model calculations."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location('rolling_capacity_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/rolling_capacity.py')
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def raw_scheduler(rows):
    return json.dumps({'JOBS': len(rows), 'RECORDS': rows})


def scheduler(jid='12', state='PEND', command='external command'):
    return {'JOBID': jid, 'STAT': state, 'QUEUE': 'hpc', 'COMMAND': command}


def basic_policy():
    return {'version': r.VERSION, 'control_cap_bytes': r.CONTROL_CAP,
            'control_cap_files': r.CONTROL_FILES, 'release_cap_bytes': r.RELEASE_CAP,
            'release_cap_files': r.RELEASE_FILES, 'known_mosaic_jobs': [], 'deferred_goals': []}


def test_parse_actual_complete_scheduler_json_and_unique_ids():
    rows = [scheduler('1'), scheduler('2', 'RUN', r.c.ROOT + '/own')]
    assert r.parse_scheduler(raw_scheduler(rows)) == rows
    for malformed in ({}, {'RECORDS': rows, 'JOBS': 3}, {'RECORDS': rows + rows},
                      {'RECORDS': [scheduler(state='UNKWN')]}):
        with pytest.raises(ValueError):
            r.parse_scheduler(json.dumps(malformed))


def test_known_exact_independent_and_own_active_jobs_are_classified():
    teacher = scheduler('1', 'RUN', r.c.ROOT + '/teacher command')
    own = scheduler('2', 'RUN', r.c.ROOT + '/release/source stage')
    external = scheduler('3')
    actual, non_mosaic = r.classify_jobs([teacher, own, external],
        [{'job_id': '1', 'command': teacher['COMMAND']}], {'2'}, r.c.ROOT + '/release/source')
    assert set(actual) == {'1', '2'}
    assert non_mosaic == ['3']


@pytest.mark.parametrize('state', ['PEND', 'RUN', 'SSUSP', 'PSUSP'])
def test_unknown_mosaic_job_blocks_even_when_only_queued(state):
    with pytest.raises(ValueError, match='Unknown nonterminal Mosaic'):
        r.classify_jobs([scheduler(state=state, command=r.c.ROOT + '/unknown')], [], set(), '/own')


@pytest.mark.parametrize('state', ['RUN', 'SSUSP', 'USUSP'])
def test_unbounded_external_active_writer_blocks_not_claimed_reserved(state):
    with pytest.raises(ValueError, match='Unrelated active account writer'):
        r.classify_jobs([scheduler(state=state)], [], set(), '/own')


def test_queued_external_jobs_are_disclosed_and_terminal_jobs_need_no_cap():
    rows = [scheduler('1', 'PEND'), scheduler('2', 'DONE'), scheduler('3', 'EXIT')]
    assert r.classify_jobs(rows, [], set(), '/own') == ({}, ['1'])


def test_independent_job_id_without_exact_command_is_rejected():
    with pytest.raises(ValueError, match='command changed'):
        r.classify_jobs([scheduler('1', 'RUN', r.c.ROOT + '/changed')],
                        [{'job_id': '1', 'command': r.c.ROOT + '/original'}], set(), '/own')


@pytest.mark.parametrize('key,bad', [('version', 'legacy'), ('control_cap_bytes', 1),
    ('control_cap_files', 1), ('release_cap_bytes', 1), ('release_cap_files', 1),
    ('known_mosaic_jobs', None), ('deferred_goals', None)])
def test_changed_or_partial_adapter_policy_fails(key, bad):
    p = basic_policy()
    p[key] = bad
    with pytest.raises(ValueError):
        r.policy({'rolling_storage_policy': p})


def fixture_op(tmp_path, monkeypatch):
    # Explicit unit-only owned metadata root; production policy has no root flag.
    monkeypatch.setattr(r.c, 'ROOT', str(tmp_path))
    root, source = tmp_path / 'runs/own', tmp_path / 'releases/own/source'
    root.mkdir(parents=True)
    source.mkdir(parents=True)
    (source / 'source_manifest.json').write_text('{}\n')
    plan = {'rolling_storage_policy': basic_policy()}
    (source / 'plan.json').write_text(json.dumps(plan) + '\n')
    arms = {'protenix': {'offsets': [20000, 22500], 'segment_steps': 2500, 'global_target': 25000},
            'dbmol': {'offsets': [7500, 8750], 'segment_steps': 1250, 'global_target': 10000}}
    op = SimpleNamespace(ROOT=root, SOURCE=source, ARMS=arms, DISPATCH_ARMS=tuple(arms), read=r.read)
    op.label = lambda phase, arm, offset: arm + '-' + phase + '-' + str(offset).zfill(5)
    op.parent_path = lambda phase, arm, offset: root / op.label(phase, arm, offset)
    op.verify_ready = lambda *args: None
    op.verify_result = lambda *args: {'projections': {'maximum_pilot_update_bytes': 19}}
    return op, plan, r.binding(source / 'source_manifest.json')['sha256']


def test_before_new_pilots_only_control_release_and_deferred_segments(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    scopes, goals, claim = r.build_scopes(op, plan, pin, 'preflight', op.ROOT / 'preflight',
                                         r.binding(op.SOURCE / 'plan.json'), {})
    assert claim is None
    assert all(s['state'] == 'deferred' for s in scopes if s['role'] == 'segment')
    assert [g['next_updates'] for g in goals] == [0, 0]
    assert [g['later_updates_deferred'] for g in goals] == [5000, 2500]
    assert sum(s['cap_bytes'] or 0 for s in scopes) == r.CONTROL_CAP + r.RELEASE_CAP
    assert scopes[0]['excluded_roots'] == [str(op.ROOT), str(op.SOURCE.parent)]


def prepare_ready(op, arm='protenix', offset=20000):
    path = op.ROOT / (op.label('validate', arm, offset) + '-certificate.json')
    path.write_text('{"status":"passed"}\n')
    (op.ROOT / 'projections.json').write_text(json.dumps({arm: {
        'unit_bytes': 19, 'remaining_offsets': op.ARMS[arm]['offsets']}}) + '\n')
    return path


def test_after_actual_new_validation_proposes_only_own_next_segment(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    prepare_ready(op)
    parent = op.parent_path('after-validate', 'protenix', 20000)
    scopes, goals, claim = r.build_scopes(op, plan, pin, 'dispatch', parent,
                                         r.binding(op.SOURCE / 'plan.json'), {})
    assert claim['updates'] == 2500
    assert claim['cap_bytes'] == r.c.sparse_segment_cap(19, 2500)
    assert goals[0]['next_updates'] == 2500 and goals[0]['later_updates_deferred'] == 2500
    assert goals[1]['next_updates'] == 0
    assert len([s for s in scopes if s['state'] == 'requested']) == 1
    assert not (op.ROOT / 'storage-claims').exists()  # Building is not admission.


def test_after_segment_only_next_validation_not_optimizer_is_admitted(tmp_path, monkeypatch):
    op, _, _ = fixture_op(tmp_path, monkeypatch)
    assert r.candidate_from_parent(op, op.parent_path('after-segment', 'protenix', 20000), 'dispatch') is None
    assert r.candidate_from_parent(op, op.parent_path('after-validate', 'protenix', 20000), 'entry') is None


def test_missing_new_projection_cannot_admit_from_historical_guessed_unit(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError):
        r.build_scopes(op, plan, pin, 'dispatch', op.parent_path('after-validate', 'protenix', 20000),
                       r.binding(op.SOURCE / 'plan.json'), {})


def test_wrong_actual_pilot_remaining_offsets_fail(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    prepare_ready(op)
    (op.ROOT / 'projections.json').write_text(json.dumps({'protenix': {'unit_bytes': 19, 'remaining_offsets': [22500]}}))
    with pytest.raises(ValueError, match='remaining offsets'):
        r.build_scopes(op, plan, pin, 'dispatch', op.parent_path('after-validate', 'protenix', 20000),
                       r.binding(op.SOURCE / 'plan.json'), {})


def inventory_scope(root, state='retained', excluded=()):
    return {'id': root.name, 'root': str(root), 'state': state, 'excluded_roots': [str(p) for p in excluded]}


def test_inventory_partitions_nested_root_without_duplicating_actual_bytes(tmp_path):
    child = tmp_path / 'child'
    child.mkdir()
    (tmp_path / 'one').write_bytes(b'abc')
    (child / 'two').write_bytes(b'12345')
    rows = r.inventory_selections([inventory_scope(tmp_path, excluded=[child]), inventory_scope(child, 'requested')])
    assert [v['actual_files'] for v in rows] == [1, 1]
    assert [v['logical_bytes'] for v in rows] == [3, 5]
    assert all(v['complete'] and v['nofollow'] and len(v['metadata_rows_sha256']) == 64 for v in rows)


def test_retained_link_is_recorded_without_traversal_but_active_link_fails(tmp_path):
    retained = tmp_path / 'retained'
    retained.mkdir()
    (tmp_path / 'not-owned').write_bytes(b'x' * 10000)
    (retained / 'alias').symlink_to(tmp_path / 'not-owned')
    row = r.inventory_selections([inventory_scope(retained)])[0]
    assert row['actual_files'] == 1
    assert row['retained_symlinks_not_traversed'] == 1
    assert row['logical_bytes'] < 10000
    with pytest.raises(ValueError, match='linked/special active'):
        r.inventory_selections([inventory_scope(retained, 'admitted')])


def test_linked_inventory_root_or_parent_fails(tmp_path):
    owned = tmp_path / 'owned'
    owned.mkdir()
    linked = tmp_path / 'linked'
    linked.symlink_to(owned, target_is_directory=True)
    with pytest.raises(ValueError, match='root linked'):
        r.inventory_selections([inventory_scope(linked)])
    with pytest.raises(ValueError, match='linked parent'):
        r.inventory_selections([inventory_scope(linked / 'new')])


def test_new_absent_scope_is_zero_and_inventory_timeout_is_not_partial_pass(tmp_path):
    absent = r.inventory_selections([inventory_scope(tmp_path / 'new')])[0]
    assert absent['actual_bytes'] == 0 and absent['actual_files'] == 0
    with pytest.raises(ValueError, match='bounded deadline'):
        r.inventory_selections([inventory_scope(tmp_path)], timeout=-1)


def test_binding_rejects_symlink_instead_of_authenticating_target(tmp_path, monkeypatch):
    monkeypatch.setattr(r.c, 'ROOT', str(tmp_path))
    (tmp_path / 'proof').write_text('{}\n')
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path / 'proof')
    with pytest.raises(ValueError, match='linked/special'):
        r.binding(alias)


def test_actual_metadata_command_failure_cannot_supply_capacity(monkeypatch):
    monkeypatch.setattr(r.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout='', stderr='denied'))
    with pytest.raises(ValueError, match='Actual metadata command failed'):
        r._command(['bjobs'])


def live_fixture(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    def write(path, value):
        with Path(path).open('x', encoding='utf8') as stream:
            json.dump(value, stream)
            stream.write('\n')
    op.write = write
    def command(argv, timeout=60):
        raw = raw_scheduler([]) if argv[0] == 'bjobs' else ' s234818|306121|| 1 GiB| 100 GiB|| 100| 20000000\n'
        return {'argv': argv, 'returncode': 0, 'stdout': raw, 'stderr': ''}
    monkeypatch.setattr(r, '_command', command)
    monkeypatch.setattr(r.os, 'statvfs', lambda path: SimpleNamespace(
        f_files=0, f_ffree=0, f_favail=0, f_bavail=200 * 1024**3, f_frsize=1))
    return op, plan, pin


def test_live_preflight_has_fresh_raw_inventory_quota_and_no_optimizer_claim(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    parent = op.ROOT / 'preflight'
    parent.mkdir()
    result = r.storage(op, parent, 'preflight')
    assert result['status'] == 'passed'
    assert result['historical_lifetime_allowance_chain_used'] is False
    assert result['all_retained_mosaic_bytes_charged_once'] is True
    assert result['future_admitted_bytes'] + result['entire_tree_stale_charge_bytes'] >= r.CONTROL_CAP + r.RELEASE_CAP
    assert not (op.ROOT / 'storage-claims').exists()
    assert (parent / 'continuation-storage-preflight.json').is_file()


def test_live_dispatch_claim_is_immutable_and_periodic_uses_unique_receipts(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    prepare_ready(op)
    parent = op.parent_path('after-validate', 'protenix', 20000)
    parent.mkdir()
    first = r.storage(op, parent, 'dispatch')
    claim_path = op.ROOT / 'storage-claims/protenix-segment-20000.json'
    claim_before = claim_path.read_bytes()
    claim = r.read(claim_path)
    assert first['status'] == 'passed' and claim['updates'] == 2500
    assert claim['admission_policy'] == r.c.VERSION
    assert r.binding(claim['admission_receipt']['path']) == claim['admission_receipt']
    one = r.storage(op, parent, 'periodic')
    two = r.storage(op, parent, 'periodic')
    assert one['status'] == two['status'] == 'passed'
    assert claim_path.read_bytes() == claim_before
    assert len(list(parent.glob('rolling-capacity-periodic-*'))) == 2
    assert not (parent / 'continuation-storage-periodic.json').exists()


def test_live_capacity_failure_preserves_exact_receipt_and_submits_nothing(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    parent = op.ROOT / 'preflight'
    parent.mkdir()
    monkeypatch.setattr(r.os, 'statvfs', lambda path: SimpleNamespace(
        f_files=0, f_ffree=0, f_favail=0, f_bavail=1, f_frsize=1))
    with pytest.raises(ValueError, match='capacity failed'):
        r.storage(op, parent, 'preflight')
    attempts = list(parent.glob('rolling-capacity-preflight-*'))
    assert len(attempts) == 1
    assert r.read(attempts[0] / 'assessment.json')['status'] == 'failed'
    assert r.read(attempts[0] / 'failure.json')['partial_inventory_used'] is False
    assert not (op.ROOT / 'storage-claims').exists()
