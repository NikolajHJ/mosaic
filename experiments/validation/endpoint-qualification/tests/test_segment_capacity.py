"""Allocated metadata contracts for new rolling admission, never legacy waiver.

Fixtures are pure accounting metadata.  Full live proof authentication belongs
to the sealed caller, independently from these arithmetic/schema contracts.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('segment_capacity_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/segment_capacity.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)
GIB = 1024**3
ZERO = {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}


def proof(name):
    return {'path': c.ROOT + '/proofs/' + name + '.json', 'bytes': 13, 'sha256': 'a' * 64}


def scope(sid='next-pro', *, state='requested', role='segment', actual=0, files=0,
          cap=GIB, cap_files=500, root=None, included=False):
    return {'id': sid, 'root': root or c.ROOT + '/runs/rolling/' + sid, 'excluded_roots': [],
            'state': state, 'role': role, 'actual_bytes': actual, 'actual_files': files,
            'cap_bytes': cap if state in c.FUNDED else None,
            'cap_files': cap_files if state in c.FUNDED else None,
            'inventory_proof': proof(sid + '-inventory'), 'liveness_proof': proof(sid + '-state'),
            'quota_inclusion_proof': proof(sid + '-included') if included else None}


def goal(gid='protenix', sid='next-pro', saved=20000, target=50000, next_updates=2500):
    return {'id': gid, 'saved_updates': saved, 'target_updates': target,
            'next_updates': next_updates, 'next_scope_id': sid,
            'later_updates_deferred': target - saved - next_updates,
            'readmission_before_later_execution': True}


def ledger():
    return {'version': c.VERSION, 'filesystem': 'work3', 'phase': 'entry',
            'account': {'name': 's234818', 'id': 306121}, 'source_manifest_sha256': 'b' * 64,
            'observation': {'complete': True, 'nofollow': True, 'untracked_mosaic_writers': [],
                            'inventory_proof': proof('all-inventory'), 'scheduler_proof': proof('all-scheduler')},
            'scopes': [scope()], 'goals': [goal()],
            'shared_account': {'non_mosaic_job_ids': ['29838765', '29838737', '29839755'],
                               'their_future_bytes_reserved': False, 'whole_account_reservation': False,
                               'entry_and_periodic_recheck_required': True}}


def quota(used='41.98 GiB', hard='100.00 GiB', files=371063, hard_files=20000000):
    return ' s234818|306121|| ' + used + '| ' + hard + '|| ' + str(files) + '| ' + str(hard_files) + '\n'


def assess(value=None, raw=None, disk=200 * GIB, inodes=None):
    return c.assess(value or ledger(), raw or quota(), disk, ZERO if inodes is None else inodes)


def test_next_bounded_segment_and_original_goal_are_separate():
    result = assess()
    assert result['status'] == 'passed'
    assert result['required_bytes'] == 5 * GIB
    assert result['lifetime_targets_preserved'] is True
    assert result['later_segments_admitted'] is False
    assert result['ledger']['goals'][0]['later_updates_deferred'] == 27500
    assert result['historical_gate_results_changed'] is False
    assert result['scientific_settings_or_guards_changed'] is False
    assert result['saved_results_changed'] is False
    assert result['ledger']['shared_account']['whole_account_reservation'] is False


def test_sparse_projection_is_one_stride20_segment_not_lifetime_or_new_stride():
    assert c.sparse_segment_cap(23182, 2500) == 2 * 23182 * (125 + 3) + GIB
    assert c.sparse_segment_cap(41425, 1250) == 2 * 41425 * (63 + 3) + GIB
    assert c.sparse_segment_cap(100, 21, 500) == 2 * 100 * (2 + 3) + 500


@pytest.mark.parametrize('unit,steps,text', [(0, 1, 1), (1, 0, 1), (True, 1, 1), (1, 1, -1)])
def test_projection_rejects_invalid_metadata(unit, steps, text):
    with pytest.raises(ValueError):
        c.sparse_segment_cap(unit, steps, text)


def test_whole_mutable_tree_and_future_pay_full_cap_once_without_credit():
    empty = assess()
    value = ledger()
    value['scopes'][0].update(actual_bytes=12345, actual_files=12)
    written = assess(value)
    assert written['available_bytes'] == empty['available_bytes'] - 12345
    assert written['required_bytes'] == empty['required_bytes'] - 12345
    assert written['byte_margin'] == empty['byte_margin']
    assert written['file_margin'] == empty['file_margin']
    assert written['entire_tree_stale_charge_bytes'] + written['future_admitted_bytes'] == GIB


def test_all_concurrently_admitted_writers_and_release_are_funded():
    value = ledger()
    value['scopes'] += [scope('next-db', cap=2 * GIB), scope('teacher', role='defog', state='active',
                        actual=GIB, files=100, cap=4 * GIB, cap_files=20000),
                        scope('teacher-release', role='release', cap=64 * 1024**2)]
    value['goals'].append(goal('dbmol', 'next-db', 7500, 10000, 1250))
    result = assess(value)
    assert result['file_reserve'] == 130000
    assert result['future_admitted_bytes'] + result['entire_tree_stale_charge_bytes'] == 7 * GIB + 64 * 1024**2
    assert result['reserve_bytes'] == 4 * GIB


def test_terminal_history_retires_future_not_actual_bytes():
    value = ledger()
    value['scopes'].append(scope('old-failed', state='terminal', role='history', actual=2 * GIB, files=1000))
    result = assess(value)
    assert result['entire_tree_stale_charge_bytes'] == 2 * GIB
    assert result['future_admitted_bytes'] == GIB
    assert result['ledger']['scopes'][1]['actual_bytes'] == 2 * GIB
    assert result['ledger']['scopes'][1]['liveness_proof'] == proof('old-failed-state')


def test_exact_unchanged_quota_included_history_is_not_charged_twice():
    value = ledger()
    value['scopes'].append(scope('old-kept', state='retained', role='history', actual=20 * GIB,
                                 files=40000, included=True))
    result = assess(value)
    assert result['entire_tree_stale_charge_bytes'] == 0
    assert result['scope_accounting'][1]['actual_bytes'] == 20 * GIB
    assert result['scope_accounting'][1]['quota_inclusion_proved'] is True


def test_deferred_scopes_cannot_have_outputs_or_execution_cap():
    value = ledger()
    value['scopes'].append(scope('future-pro', state='deferred'))
    result = assess(value)
    assert result['scope_accounting'][1]['remaining_admitted_cap_bytes'] == 0
    for key, bad in [('actual_bytes', 1), ('actual_files', 1), ('cap_bytes', 1),
                     ('quota_inclusion_proof', proof('invented-inclusion'))]:
        edited = copy.deepcopy(value)
        edited['scopes'][1][key] = bad
        with pytest.raises(ValueError):
            assess(edited)


def test_deferred_entire_arm_keeps_target_but_gets_no_execution_admission():
    value = ledger()
    value['goals'].append(goal('boltz', None, 20000, 30000, 0))
    result = assess(value)
    assert result['ledger']['goals'][1]['later_updates_deferred'] == 10000
    assert result['future_admitted_bytes'] == GIB


@pytest.mark.parametrize('key,bad', [('target_updates', 49999), ('later_updates_deferred', 0),
    ('readmission_before_later_execution', False), ('next_scope_id', 'missing'), ('next_updates', True)])
def test_goal_changes_or_unfunded_execution_fail(key, bad):
    value = ledger()
    value['goals'][0][key] = bad
    with pytest.raises(ValueError):
        assess(value)


def test_two_goals_cannot_silently_share_one_segment_budget():
    value = ledger()
    value['goals'].append(goal('other', 'next-pro', 0, 10000, 2500))
    with pytest.raises(ValueError, match='uniquely funded'):
        assess(value)


@pytest.mark.parametrize('key,bad', [('complete', False), ('nofollow', False),
    ('untracked_mosaic_writers', ['unknown-running-job']), ('inventory_proof', None), ('scheduler_proof', {})])
def test_partial_or_unknown_inventory_never_admits(key, bad):
    value = ledger()
    value['observation'][key] = bad
    with pytest.raises(ValueError):
        assess(value)


@pytest.mark.parametrize('key,bad', [('actual_bytes', GIB + 1), ('actual_files', 501),
    ('actual_bytes', False), ('actual_files', -1), ('cap_bytes', float('inf')),
    ('quota_inclusion_proof', proof('mutable-credited')), ('liveness_proof', None)])
def test_cap_breaches_invalid_metadata_or_mutable_credit_fail(key, bad):
    value = ledger()
    value['scopes'][0][key] = bad
    with pytest.raises(ValueError):
        assess(value)


def test_duplicate_identity_and_overlapping_roots_fail():
    for duplicate in (scope(), scope('different', root=ledger()['scopes'][0]['root']),
                      scope('nested', root=ledger()['scopes'][0]['root'] + '/child')):
        value = ledger()
        value['scopes'].append(duplicate)
        with pytest.raises(ValueError):
            assess(value)


def test_parent_control_inventory_can_explicitly_exclude_funded_child():
    value = ledger()
    control = scope('control', role='control', root=c.ROOT + '/runs/rolling')
    control['excluded_roots'] = [value['scopes'][0]['root']]
    value['scopes'].append(control)
    result = assess(value)
    assert result['future_admitted_bytes'] == 2 * GIB
    assert result['scope_accounting'][1]['excluded_roots'] == control['excluded_roots']


def test_whole_mosaic_root_is_retained_once_with_explicit_mutable_exclusion():
    value = ledger()
    retained = scope('retained-mosaic', state='retained', role='history', actual=20 * GIB,
                     root=c.ROOT)
    retained['excluded_roots'] = [value['scopes'][0]['root']]
    value['scopes'].append(retained)
    result = assess(value)
    assert result['entire_tree_stale_charge_bytes'] == 20 * GIB
    assert result['future_admitted_bytes'] == GIB
    retained['state'] = 'requested'
    retained.update(cap_bytes=40 * GIB, cap_files=200000)
    with pytest.raises(ValueError, match='only as retained-mosaic'):
        assess(value)


def test_excluded_subtree_without_its_own_ledger_cannot_hide_bytes():
    value = ledger()
    value['scopes'][0]['excluded_roots'] = [value['scopes'][0]['root'] + '/hidden']
    with pytest.raises(ValueError, match='own ledger scope'):
        assess(value)


@pytest.mark.parametrize('root', ['/work3/s234818/mosaic/', '/work3/s234818/other/run',
    '/work3/s234818/mosaic/runs/../escape', '/work3/s234818/mosaic/runs//double', 'relative'])
def test_wrong_or_noncanonical_scope_paths_fail(root):
    value = ledger()
    value['scopes'][0]['root'] = root
    with pytest.raises(ValueError):
        assess(value)


def test_all_finite_pools_and_conservative_decimal_rounding():
    raw = quota('41.00000000001 GiB', '100.999999999 GiB') + quota('10.01 GiB', '20.99 GiB', 99990, 200000)
    q = c.parse_work3_quota(raw, ledger()['account'])
    assert len(q['pools']) == 2
    assert q['pools'][0]['used_bytes_upper'] == 42 * GIB
    assert q['pools'][0]['hard_bytes_lower'] == 100 * GIB
    assert q['available_bytes_lower'] == 9 * GIB
    assert q['available_files_lower'] == 100010


def test_byte_finite_file_unlimited_and_reverse_are_both_enforced():
    raw = quota(files=0, hard_files='unlimited') + quota('0 Byte', 'unlimited', files=10, hard_files=200000)
    result = assess(raw=raw)
    assert result['quota']['available_bytes_lower'] == 58 * GIB
    assert result['quota']['available_files_lower'] == 199990


@pytest.mark.parametrize('raw', ['', ' s234818|306121|| 1 GiB| 100 GiB|| 1| 200000',
    ' s234818|306121|| NaN GiB| 100 GiB|| 1| 200000\n',
    ' s234818|306122|| 1 GiB| 100 GiB|| 1| 200000\n',
    ' s234818|306121|| 0 Byte| unlimited|| 0| unlimited\n',
    ' s234818|306121|| 1 GiB| 100 GiB|| bad| 200000\n'])
def test_missing_malformed_nonfinite_or_unbounded_quota_fails(raw):
    with pytest.raises(ValueError):
        c.parse_work3_quota(raw, ledger()['account'])


def test_physical_disk_minimum_is_mandatory_and_failure_receipt_preserved():
    result = assess(disk=4 * GIB)
    assert result['status'] == 'failed'
    assert result['required_bytes'] == 5 * GIB
    assert result['byte_margin'] == -GIB
    assert result['ledger']['source_manifest_sha256'] == 'b' * 64


def test_all_zero_inodes_honestly_use_finite_quota_without_independent_claim():
    result = assess()
    assert result['filesystem_inode_reporting'] == 'unreported_all_zero'
    assert 'No independent filesystem inode evidence' in result['file_headroom_scope']
    assert result['required_files'] == 100500


def test_positive_filesystem_inodes_take_lower_bound_and_fail_low_capacity():
    result = assess(inodes={'f_files': 1000000, 'f_ffree': 100400, 'f_favail': 100200})
    assert result['status'] == 'failed'
    assert result['available_files'] == 100200
    assert result['filesystem_inode_reporting'] == 'reported_positive'


@pytest.mark.parametrize('inodes', [{'f_files': 0, 'f_ffree': 1, 'f_favail': 1},
    {'f_files': 100, 'f_ffree': 101, 'f_favail': 1},
    {'f_files': 100, 'f_ffree': 10, 'f_favail': 11},
    {'f_files': 100, 'f_ffree': 10, 'f_favail': False}])
def test_incoherent_or_malformed_filesystem_inodes_fail(inodes):
    with pytest.raises(ValueError):
        assess(inodes=inodes)


@pytest.mark.parametrize('key,bad', [('whole_account_reservation', True),
    ('their_future_bytes_reserved', True), ('entry_and_periodic_recheck_required', False),
    ('non_mosaic_job_ids', ['1', '1']), ('non_mosaic_job_ids', ['unknown'])])
def test_scope_never_claims_unrelated_jobs_are_reserved(key, bad):
    value = ledger()
    value['shared_account'][key] = bad
    with pytest.raises(ValueError):
        assess(value)


def test_closed_fields_fail_instead_of_ignoring_unknown_policy_override():
    value = ledger()
    value['reserve_bytes'] = 0
    with pytest.raises(ValueError, match='fields differ'):
        assess(value)
