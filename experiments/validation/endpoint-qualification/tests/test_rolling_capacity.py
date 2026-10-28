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


def terminal_fixture(tmp_path, monkeypatch):
    path = tmp_path/'gpu.out'
    path.write_text('Subject: Job 14: old\n in queue <gpul40s>\n Exited with exit code 1.\n')
    row = {'job_id':'14','root':r.c.ROOT+'/runs/old','source_manifest':{'sha256':'a'*64},
           'terminal_proof':{'path':r.c.ROOT+'/proof.json','bytes':1,'sha256':'b'*64}}
    value={'status':'terminal_job_no_unspent_future_execution_retained_files_preserved','job_id':'14',
           'output_root':row['root'],'source_manifest_sha256':'a'*64,'files_deleted_or_modified':False,
           'proofs':[{'path':path.as_posix(),'bytes':path.stat().st_size,'sha256':'c'*64}]}
    monkeypatch.setattr(r,'check_binding',lambda p:p)
    monkeypatch.setattr(r,'read',lambda p:value)
    return row,value,path


def test_actual_terminal_proof_can_retire_only_the_unspent_execution_allowance(tmp_path,monkeypatch):
    row,_,_=terminal_fixture(tmp_path,monkeypatch)
    assert r.independent_terminal(row)==row['terminal_proof']


@pytest.mark.parametrize('bad',['job','root','source','lsf'])
def test_terminal_scope_requires_original_owner_source_and_same_job_lsf(tmp_path,monkeypatch,bad):
    row,value,path=terminal_fixture(tmp_path,monkeypatch)
    if bad=='job':value['job_id']='15'
    if bad=='root':value['output_root']+='/other'
    if bad=='source':value['source_manifest_sha256']='d'*64
    if bad=='lsf':path.write_text('other job, no terminal evidence')
    with pytest.raises(ValueError):r.independent_terminal(row)


def test_only_freshly_authenticated_foreign_family_ids_receive_shared_budget_exemption():
    vae = scheduler('11','RUN','exact authenticated VAE family command')
    assert r.classify_jobs([vae],[],set(),'/own',accounted_external={'11'}) == ({},[])
    with pytest.raises(ValueError,match='Unrelated active account writer'):
        r.classify_jobs([vae,scheduler('12','RUN','unknown foreign writer')],[],set(),'/own',accounted_external={'11'})


def test_independent_job_id_without_exact_command_is_rejected():
    with pytest.raises(ValueError, match='command changed'):
        r.classify_jobs([scheduler('1', 'RUN', r.c.ROOT + '/changed')],
                        [{'job_id': '1', 'command': r.c.ROOT + '/original'}], set(), '/own')


def validator_group(source_manifest=None):
    source = str(Path((source_manifest or {'path': r.c.ROOT + '/releases/source16/source/source_manifest.json'})['path']).parent)
    return {'job_id': '16', 'command': 'bash ' + source + '/stage.lsf ' + source + ' validate protenix 20000',
            'additional_jobs': [{'job_id': '18', 'command': 'bash ' + source + '/stage.lsf ' + source + ' validate dbmol 7500'}],
            'root': r.c.ROOT + '/runs/source16', 'cap_bytes': 2 * 1024**3, 'cap_files': 30000,
            'role': 'validation', 'release_root': None, 'release_cap_bytes': None, 'release_cap_files': None,
            'source_manifest': source_manifest or {'path': source + '/source_manifest.json', 'bytes': 10, 'sha256': 'a' * 64}}


def test_two_exact_old_validator_jobs_share_one_bounded_source_owned_tree():
    group = validator_group()
    value = basic_policy()
    value['known_mosaic_jobs'] = [group]
    assert r.policy({'rolling_storage_policy': value}) == value
    rows = [scheduler(job['job_id'], 'RUN', job['command']) for job in r.allocation_jobs(group)]
    actual, external = r.classify_jobs(rows, [group], set(), '/new-source')
    assert set(actual) == {'16', '18'} and external == []


def test_grouped_validator_identity_cannot_be_duplicated_or_added_implicitly():
    group = validator_group()
    value = basic_policy()
    value['known_mosaic_jobs'] = [group]
    group['additional_jobs'].append({'job_id': '16', 'command': group['command']})
    with pytest.raises(ValueError, match='Duplicate/invalid'):
        r.policy({'rolling_storage_policy': value})
    with pytest.raises(ValueError, match='Unknown nonterminal Mosaic'):
        r.classify_jobs([scheduler('99', 'PEND', group['command'])], [group], set(), '/new-source')


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
    study = op.parent_path('validate', arm, offset) / 'payload/study'
    study.mkdir(parents=True)
    (study / 'summary.json').write_text('{"status":"passed"}\n')
    (study / 'raw-proof.json').write_text('{"separate_reference":"complete"}\n')
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


def test_full_remaining_in_one_job_uses_actual_dynamic_steps_without_small_chunk_limit(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    op.ARMS['protenix'].update(offsets=[20000], segment_steps=30000, global_target=50000)
    op.ARMS['dbmol'].update(offsets=[7500], segment_steps=2500, global_target=10000)
    prepare_ready(op)
    scopes, goals, claim = r.build_scopes(op, plan, pin, 'dispatch',
        op.parent_path('after-validate', 'protenix', 20000), r.binding(op.SOURCE / 'plan.json'), {})
    assert claim['updates'] == 30000
    assert claim['cap_bytes'] == r.c.sparse_segment_cap(19, 30000)
    assert claim['cap_files'] == 7500
    assert claim['file_capacity']['retained_streams'] == 3
    assert claim['validation_inventory']['actual_files'] == 2
    assert goals[0]['saved_updates'] == 20000 and goals[0]['next_updates'] == 30000
    assert goals[0]['target_updates'] == 50000 and goals[0]['later_updates_deferred'] == 0
    assert goals[1]['saved_updates'] == 7500 and goals[1]['later_updates_deferred'] == 2500
    assert len([s for s in scopes if s['role'] == 'segment']) == 2


def test_old_validator_group_budget_is_paid_once_and_both_jobs_recognized(tmp_path, monkeypatch):
    op, plan, pin = fixture_op(tmp_path, monkeypatch)
    manifest = tmp_path / 'releases/source16/source/source_manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{}\n')
    group = validator_group(r.binding(manifest))
    plan['rolling_storage_policy']['known_mosaic_jobs'] = [group]
    observed = {job['job_id']: scheduler(job['job_id'], 'RUN', job['command']) for job in r.allocation_jobs(group)}
    scopes, _, claim = r.build_scopes(op, plan, pin, 'preflight', op.ROOT / 'preflight',
                                     r.binding(op.SOURCE / 'plan.json'), observed)
    old = [s for s in scopes if s['role'] == 'validation']
    assert claim is None and len(old) == 1
    assert old[0]['cap_bytes'] == 2 * 1024**3 and old[0]['state'] == 'active'


@pytest.mark.parametrize('updates', [30000, 30001])
def test_file_capacity_funds_all_three_actual_retained_file_streams(updates):
    # Independent metadata enumeration of the authenticated retention contract:
    # snapshot/reference(step), trial(update+1), true initial/final boundaries.
    retained_steps = {0, updates, *range(20, updates + 1, 20)}
    snapshots = {'snapshots/step-' + str(step) + '.npz' for step in retained_steps}
    references = {'references/step-' + str(step) + '.json' for step in retained_steps}
    trials = {'trials/step-' + str(step - 1) + '.npz' for step in retained_steps if step > 0}
    count = len(snapshots | references | trials)
    capacity = r.segment_file_capacity(updates, 2)
    assert count == 3 * ((updates + 19) // 20) + 2
    assert capacity['cap_files'] - count == 2998
    if updates == 30000:
        assert count == 4502 and capacity['cap_files'] == 7500
        assert count > (updates + 19) // 20 + 3000  # Preserved source17 defect.


def test_large_verified_validation_cohort_expands_fixed_boundary_allowance():
    capacity = r.segment_file_capacity(30000, 2000)
    assert capacity['fixed_boundary_and_metadata_files'] == 4100
    assert capacity['cap_files'] == 8600
    assert capacity['policy'] == r.FILE_CAPACITY_POLICY


@pytest.mark.parametrize('updates,validation_files', [(0, 1), (True, 1), (30000, 0), (30000, False)])
def test_file_capacity_requires_finite_complete_new_validation_count(updates, validation_files):
    with pytest.raises(ValueError):
        r.segment_file_capacity(updates, validation_files)


def test_missing_passed_validation_tree_cannot_supply_file_cap_basis(tmp_path, monkeypatch):
    op, _, _ = fixture_op(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='study tree missing'):
        r.validation_inventory(op, 'protenix', 20000)


def test_completed_validation_identity_survives_different_host_device_inode_and_blocks(tmp_path, monkeypatch):
    op, _, _ = fixture_op(tmp_path, monkeypatch)
    prepare_ready(op)
    before = r.validation_inventory(op, 'protenix', 20000)
    old_lstat, old_fstat = Path.lstat, r.os.fstat
    def other_host(info):
        names = ('st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_dev', 'st_ino')
        value = {key: getattr(info, key) for key in names}
        value.update(st_dev=value['st_dev'] + 1234567, st_ino=value['st_ino'] + 7654321,
                     st_blocks=999999999, st_atime_ns=1)
        return SimpleNamespace(**value)
    monkeypatch.setattr(Path, 'lstat', lambda path, *args, **kwargs: other_host(old_lstat(path, *args, **kwargs)))
    monkeypatch.setattr(r.os, 'fstat', lambda descriptor: other_host(old_fstat(descriptor)))
    after = r.validation_inventory(op, 'protenix', 20000)
    assert after == before
    assert before['identity_policy'] == r.VALIDATION_INVENTORY_POLICY
    assert all(set(row) == {'path', 'type', 'bytes', 'sha256'} for row in before['entries'])


def test_windows_handle_ctime_difference_does_not_weaken_posix_race_checks():
    before = SimpleNamespace(st_dev=1, st_ino=2, st_mode=0o100644, st_size=35, st_mtime_ns=4, st_ctime_ns=5)
    opened = SimpleNamespace(**{**before.__dict__, 'st_ctime_ns': 6})
    assert r._local_file_signature(before, windows=True) == r._local_file_signature(opened, windows=True)
    assert r._local_file_signature(before, windows=False) != r._local_file_signature(opened, windows=False)
    opened.st_mtime_ns = 7
    assert r._local_file_signature(before, windows=True) != r._local_file_signature(opened, windows=True)


def test_completed_validation_identity_rejects_same_length_content_or_path_change(tmp_path, monkeypatch):
    op, _, _ = fixture_op(tmp_path, monkeypatch)
    prepare_ready(op)
    before = r.validation_inventory(op, 'protenix', 20000)
    study = op.parent_path('validate', 'protenix', 20000) / 'payload/study'
    old = study / 'raw-proof.json'
    data = old.read_bytes()
    old.write_bytes(b'X' * len(data))
    content = r.validation_inventory(op, 'protenix', 20000)
    assert content['actual_files'] == before['actual_files'] and content['actual_bytes'] == before['actual_bytes']
    assert content['semantic_rows_sha256'] != before['semantic_rows_sha256']
    old.write_bytes(data)
    old.rename(study / 'renamed-proof.json')
    renamed = r.validation_inventory(op, 'protenix', 20000)
    assert renamed['actual_files'] == before['actual_files']
    assert renamed['semantic_rows_sha256'] != before['semantic_rows_sha256']


def test_completed_validation_missing_or_linked_input_is_not_a_retry(tmp_path, monkeypatch):
    op, _, _ = fixture_op(tmp_path, monkeypatch)
    prepare_ready(op)
    study = op.parent_path('validate', 'protenix', 20000) / 'payload/study'
    (study / 'linked.json').symlink_to(study / 'summary.json')
    with pytest.raises(ValueError, match='linked/special'):
        r.validation_inventory(op, 'protenix', 20000)


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


def retry_scope(op, state='admitted'):
    return {'id': 'protenix-segment-20000',
            'root': str(op.parent_path('segment', 'protenix', 20000) / 'payload/study'),
            'excluded_roots': [], 'state': state, 'role': 'segment'}


def test_disappearing_owned_atomic_file_restarts_entire_inventory_without_partial_totals(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    attempt = op.ROOT / 'retry'
    attempt.mkdir()
    scope = retry_scope(op)
    missing = Path(scope['root']) / 'case-0/references/latest-reference.json.tmp'
    calls = []
    def inventory(scopes, *, timeout):
        calls.append(list(scopes))
        if len(calls) == 1:
            raise FileNotFoundError(2, 'atomic rename', str(missing))
        return [{'complete': True, 'actual_bytes': 123, 'actual_files': 4}]
    monkeypatch.setattr(r, 'inventory_selections', inventory)
    result = r.complete_capacity_inventory(op, [scope], attempt)
    assert len(calls) == 2 and calls[0] == calls[1] == [scope]
    assert result == [{'complete': True, 'actual_bytes': 123, 'actual_files': 4}]
    failed = r.read(attempt / 'incomplete-capacity-inventory-01.json')
    assert failed['recognized_owned_atomic_temporary'] is True and failed['retry_will_occur'] is True
    assert failed['partial_totals_used'] is False and failed['files_skipped'] is False


@pytest.mark.parametrize('state,name', [('terminal', 'report.json.tmp'), ('retained', 'report.json.tmp'),
    ('admitted', 'summary.json'), ('admitted', 'unknown.tmp')])
def test_missing_immutable_or_unrecognized_file_is_fatal_without_retry(tmp_path, monkeypatch, state, name):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    attempt = op.ROOT / 'retry'
    attempt.mkdir()
    scope = retry_scope(op, state)
    missing = Path(scope['root']) / name
    calls = []
    def inventory(scopes, *, timeout):
        calls.append(1)
        raise FileNotFoundError(2, 'not an owned atomic temporary', str(missing))
    monkeypatch.setattr(r, 'inventory_selections', inventory)
    with pytest.raises(FileNotFoundError):
        r.complete_capacity_inventory(op, [scope], attempt)
    assert len(calls) == 1
    failed = r.read(attempt / 'incomplete-capacity-inventory-01.json')
    assert failed['retry_will_occur'] is False and failed['files_skipped'] is False


def test_repeated_owned_atomic_race_stops_at_three_full_failed_attempts(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    attempt = op.ROOT / 'retry'
    attempt.mkdir()
    scope = retry_scope(op)
    missing = Path(scope['root']) / 'case-0/best-observed.partial'
    calls = []
    def inventory(scopes, *, timeout):
        calls.append(1)
        raise FileNotFoundError(2, 'atomic rename', str(missing))
    monkeypatch.setattr(r, 'inventory_selections', inventory)
    with pytest.raises(FileNotFoundError):
        r.complete_capacity_inventory(op, [scope], attempt)
    assert len(calls) == 3
    records = sorted(attempt.glob('incomplete-capacity-inventory-*.json'))
    assert len(records) == 3
    assert r.read(records[-1])['retry_will_occur'] is False
    assert all(r.read(path)['partial_totals_used'] is False for path in records)


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
    assert r.binding(claim['validation_inventory_evidence']['path']) == claim['validation_inventory_evidence']
    assert r.read(claim['validation_inventory_evidence']['path']) == claim['validation_inventory']
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


def test_admitted_file_cap_rejects_changed_completed_validation_inventory(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    prepare_ready(op)
    parent = op.parent_path('after-validate', 'protenix', 20000)
    parent.mkdir()
    r.storage(op, parent, 'dispatch')
    study = op.parent_path('validate', 'protenix', 20000) / 'payload/study'
    (study / 'unexpected-after-admission.json').write_text('{}\n')
    with pytest.raises(ValueError, match='file-cap basis changed'):
        r.storage(op, parent, 'periodic')


def test_admitted_file_cap_rejects_changed_immutable_inventory_evidence(tmp_path, monkeypatch):
    op, _, _ = live_fixture(tmp_path, monkeypatch)
    prepare_ready(op)
    parent = op.parent_path('after-validate', 'protenix', 20000)
    parent.mkdir()
    r.storage(op, parent, 'dispatch')
    claim = r.read(op.ROOT / 'storage-claims/protenix-segment-20000.json')
    Path(claim['validation_inventory_evidence']['path']).write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='Metadata proof changed'):
        r.storage(op, parent, 'periodic')
