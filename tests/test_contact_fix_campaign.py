"""Allocated CPU metadata fixtures for strict conditional pilot submission."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/hpc'))
import contact_fix_campaign as campaign


def protocol(tmp_path, version=1):
    validation = {'path': str(tmp_path / 'validation.json'), 'sha256': 'a' * 64}
    pilot = tmp_path / 'pilot.json'
    pilot.write_text(json.dumps({'version': {1: 2, 2: 3, 3: 4}[version],
        'saved_state_readiness': {'plan': validation, 'output_root': '/work3/u/runs/validation'}}))
    return {'version': version, 'release': '/work3/u/releases/source105', 'work_root': '/work3/u',
        'source_archive_sha256': 'b' * 64, 'source_manifest_sha256': 'c' * 64,
        'validation_plan': validation, 'pilot_plan': {'path': str(pilot), 'sha256': 'd' * 64},
        'receipts': str(tmp_path / 'receipts'), 'allocation_root': '/work3/u/runs/allocations'}


def identity(p):
    return {**{k: p[k] for k in ('source_archive_sha256', 'source_manifest_sha256')},
            'protocol': '/work3/u/protocol.json', 'protocol_sha256': 'e' * 64}


def receipt(p, who):
    root = Path(p['receipts']); root.mkdir()
    intent = {'identity': who, 'cwd': p['release']}
    (root / 'readiness-intent.json').write_text(json.dumps(intent))
    (root / 'readiness-receipt.json').write_text(json.dumps({'job_id': '123', 'intent': intent}))


def test_cpu_then_readiness_then_promotion_without_prequeued_pilot(tmp_path):
    p = protocol(tmp_path)
    rows = campaign.workflow(p, Path('/work3/u/protocol.json'), 'e' * 64)
    assert [r['stage'] for r in rows] == ['cpu', 'readiness', 'promotion']
    assert rows[1]['dependency'] == 'done:cpu'
    assert rows[2]['dependency'] == 'ended:readiness'
    assert '6:00' in rows[1]['arguments']
    assert all('latentmol.repair_guidance.matched_pilot' not in r['job_command'] for r in rows)


@pytest.mark.parametrize('reason', ['scheduler_exit', 'failed_gate', 'wrong_allocation'])
def test_no_pilot_submission_when_predecessor_evidence_fails(tmp_path, monkeypatch, reason):
    p, submitted = protocol(tmp_path), []
    who = identity(p); receipt(p, who)
    monkeypatch.setattr(campaign, 'submit', lambda *args: submitted.append(args))
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *args, **kw:
        SimpleNamespace(returncode=0, stdout='EXIT' if reason == 'scheduler_exit' else 'DONE', stderr=''))
    import latentmol.boltz.contact_precision_validation as validator
    def readiness(binding, source_identity):
        assert set(source_identity) == {'source_archive_sha256', 'source_manifest_sha256'}
        if reason == 'failed_gate': raise ValueError('native contact failed')
        return {'passed': True, 'allocation_job_id': '999'}
    monkeypatch.setattr(validator, 'require_readiness', readiness)
    with pytest.raises(ValueError): campaign.advance(p, who)
    assert submitted == []
    decision = json.loads((Path(p['receipts']) / 'promotion-decision.json').read_text())
    assert decision['status'] == 'stopped' and decision['pilot_submitted'] is False


def test_receipt_identity_cannot_be_rebound_to_another_campaign(tmp_path):
    p = protocol(tmp_path); who = identity(p)
    receipt(p, {**who, 'protocol_sha256': 'f' * 64})
    with pytest.raises(ValueError, match='another immutable campaign'):
        campaign.advance(p, who)


def test_quota_uses_user_limit_not_global_space_and_rejects_unknown():
    official = 'u|123|| 88.16 GiB| 100 GiB|| 42|1000\nu|123|| 0 Byte|unlimited||0|unlimited'
    assert str(campaign.available_quota_gib(official, 'u')) == '11.84'
    with pytest.raises(ValueError, match='missing'):
        campaign.available_quota_gib('some other user', 'u')
    with pytest.raises((ValueError, KeyError)):
        campaign.available_quota_gib('u|123|| 1 Unknown| 100 GiB||0|100', 'u')


@pytest.mark.parametrize('version', [2, 3, 4])
def test_branch_pilot_reserves_two_full_children_and_readiness_binding(tmp_path, version):
    from latentmol.repair_guidance.matched_pilot import BRANCH_BUDGETS
    reserve = 180 + 2 * sum(BRANCH_BUDGETS[k] for k in
        ('optimization_timeout_seconds', 'repair_timeout_seconds', 'rescore_timeout_seconds')) + 120
    assert reserve < BRANCH_BUDGETS['allocation_seconds'] < 12 * 3600
    from submit_defog_matched_pilot import workflow
    path = tmp_path / 'plan.json'; path.write_text(json.dumps({'version': version}))
    cpu, native = workflow(Path('/work3/u/releases/source105'), Path('/work3/u'), path,
                           'a' * 64, 'b' * 64, 'c' * 64)
    assert native['dependency'] == 'done:cpu' and '12:00' in native['arguments']


def test_additive_readiness_waits_for_actual_diagnostic_and_cpu(tmp_path, monkeypatch):
    p = protocol(tmp_path, version=2)
    monkeypatch.setattr(campaign, 'diagnostic_predecessor', lambda protocol, validation=None: '29728602')
    rows = campaign.workflow(p, Path('/work3/u/protocol.json'), 'e' * 64)
    assert [row['stage'] for row in rows] == ['cpu', 'readiness', 'promotion']
    assert rows[1]['dependency'] == 'done:cpu'
    assert rows[1]['external_dependencies'] == [{'kind': 'done', 'job_id': '29728602'}]
    assert '3:15' in rows[1]['arguments']
    assert 'latentmol.boltz.additive_readiness' in rows[1]['job_command']
    assert rows[2]['dependency'] == 'ended:readiness'
    assert all('latentmol.repair_guidance.matched_pilot' not in row['job_command'] for row in rows)
    assert all('external_dependencies' not in row for row in (rows[0], rows[2]))


def test_pointwise_focused_readiness_does_not_queue_a_new_broad_observer(tmp_path, monkeypatch):
    p = protocol(tmp_path, version=3)
    def prohibited(*args, **kwargs):
        raise AssertionError('Pointwise readiness cannot depend on the broad observer')
    monkeypatch.setattr(campaign, 'diagnostic_predecessor', prohibited)
    rows = campaign.workflow(p, Path('/work3/u/protocol.json'), 'e' * 64)
    assert [row['stage'] for row in rows] == ['cpu', 'readiness', 'promotion']
    assert rows[1]['dependency'] == 'done:cpu' and '0:45' in rows[1]['arguments']
    assert 'latentmol.boltz.pointwise_readiness' in rows[1]['job_command']
    assert rows[2]['dependency'] == 'ended:readiness'
    assert all('external_dependencies' not in row for row in rows)
    assert all('latentmol.repair_guidance.matched_pilot' not in row['job_command'] for row in rows)


@pytest.mark.parametrize('version,policy,pilot_version', [(1, 'branch_local_v2', 2), (2, 'additive_branch_v3', 3),
    (3, 'pointwise_native_ad_v4', 4)])
def test_protocol_rejects_cross_contract_or_cross_source_plan(tmp_path, version, policy, pilot_version):
    from copy import deepcopy
    p = protocol(tmp_path, version=version)
    validation = {'objective_arithmetic': 'contact_head_and_assembly_float64_v1',
                  'fd_policy': policy, 'output_root': '/work3/u/runs/validation'}
    pilot = {'version': pilot_version, 'derivative_contract': policy,
        'saved_state_readiness': {'plan': p['validation_plan'], 'output_root': validation['output_root']},
        'source_archive_sha256': p['source_archive_sha256'], 'source_manifest_sha256': p['source_manifest_sha256']}
    campaign.validate_plan_policies(p, validation, pilot)
    for key, value in [('version', 3 if version == 1 else 2), ('derivative_contract', 'local_v1'),
                       ('source_archive_sha256', 'f' * 64), ('saved_state_readiness', {})]:
        changed = deepcopy(pilot); changed[key] = value
        with pytest.raises(ValueError, match='not bound'):
            campaign.validate_plan_policies(p, validation, changed)
    changed = dict(validation, fd_policy='local_v1')
    with pytest.raises(ValueError, match='not bound'):
        campaign.validate_plan_policies(p, changed, pilot)


@pytest.mark.parametrize('version', [True, 0, 4, '2', None])
def test_unknown_protocol_contract_is_not_guessed(version):
    with pytest.raises(ValueError, match='Unsupported'):
        campaign.protocol_settings(version)


def diagnostic_fixture(tmp_path, monkeypatch):
    """Bound fake observer artifacts; this fixture never constructs a model."""
    import hashlib
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    release = tmp_path / 'releases/source106'
    release.mkdir(parents=True)
    archive = Path(str(release) + '.tar.gz'); archive.write_bytes(b'fixture archive')
    archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan = write(tmp_path / 'submissions/observer/plan.json', {'output_root': str(tmp_path / 'runs/observer')})
    identity = {'archive_sha256': archive_sha, 'manifest_sha256': 'c' * 64, 'plan_sha256': plan['sha256']}
    import shlex
    command = ['python', '-m', 'latentmol.boltz.trajectory_gradient_diagnostic', '--plan', plan['path'],
               '--sha256', plan['sha256'], '--archive-sha256', archive_sha, '--manifest-sha256', 'c' * 64]
    intent = {'cwd': str(release), 'identity': identity, 'command': ['bsub', shlex.join(command)]}
    root = tmp_path / 'submissions/observer/receipts'
    write(root / 'replay-intent.json', intent)
    ref = write(root / 'replay-receipt.json', {'job_id': '29728602', 'status': 'submitted_not_passed', 'intent': intent})
    validation = {'diagnostic_reference': {'plan': plan, 'output_root': str(tmp_path / 'runs/observer')}}
    monkeypatch.setattr(campaign, 'source_manifest', lambda root: {'sha256': 'c' * 64})
    return {'work_root': str(tmp_path), 'diagnostic_job_receipt': ref}, validation, intent


def test_external_observer_job_receipt_binds_source_intent_and_actual_plan(tmp_path, monkeypatch):
    p, validation, intent = diagnostic_fixture(tmp_path, monkeypatch)
    assert campaign.diagnostic_predecessor(p, validation) == '29728602'
    # A byte-authenticated but semantically transplanted observer is rejected.
    changed = json.loads(json.dumps(validation))
    changed['diagnostic_reference']['output_root'] += '-other'
    with pytest.raises(ValueError, match='different observer'):
        campaign.diagnostic_predecessor(p, changed)
    path = Path(p['diagnostic_job_receipt']['path']).with_name('replay-intent.json')
    path.write_text(json.dumps(dict(intent, cwd='different')))
    with pytest.raises(ValueError, match='receipt/intent'):
        campaign.diagnostic_predecessor(p, validation)


@pytest.mark.parametrize('reason', ['wrong_policy', 'wrong_source', 'failed_certificate', 'low_quota'])
def test_additive_promotion_cannot_use_wrong_certificate_or_quota(tmp_path, monkeypatch, reason):
    p, submitted = protocol(tmp_path, version=2), []
    who = identity(p); receipt(p, who)
    monkeypatch.setattr(campaign, 'submit', lambda *args: submitted.append(args))
    source = {key: p[key] for key in ('source_archive_sha256', 'source_manifest_sha256')}
    proof = {'passed': reason != 'failed_certificate', 'allocation_job_id': '123',
             'fd_policy': 'branch_local_v2' if reason == 'wrong_policy' else 'additive_branch_v3',
             'source_identity': {} if reason == 'wrong_source' else source}
    monkeypatch.setattr(campaign.importlib, 'import_module', lambda name: SimpleNamespace(require_readiness=lambda *args: proof))
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(returncode=0, stdout='u|123|| 99 GiB| 100 GiB||0|100' if args[0][0] == 'getquota_work3.sh' else 'DONE', stderr=''))
    with pytest.raises(ValueError):
        campaign.advance(p, who)
    assert submitted == []
    decision = json.loads((Path(p['receipts']) / 'promotion-decision.json').read_text())
    assert decision['status'] == 'stopped' and decision['pilot_submitted'] is False


@pytest.mark.parametrize('version,policy', [(1, 'branch_local_v2'), (2, 'additive_branch_v3'),
    (3, 'pointwise_native_ad_v4')])
def test_promotion_submits_only_pilot_after_actual_matching_readiness_and_quota(tmp_path, monkeypatch, version, policy):
    p, submitted = protocol(tmp_path, version=version), []
    who = identity(p); receipt(p, who)
    source = {key: p[key] for key in ('source_archive_sha256', 'source_manifest_sha256')}
    proof = {'passed': True, 'allocation_job_id': '123', 'fd_policy': policy, 'source_identity': source}
    expected_module = campaign.protocol_settings(version)['readiness_module']
    def module(name):
        assert name == expected_module
        return SimpleNamespace(require_readiness=lambda binding, identity: proof)
    monkeypatch.setattr(campaign.importlib, 'import_module', module)
    def scheduler(command, **kwargs):
        return SimpleNamespace(returncode=0, stderr='',
            stdout='u|123|| 88.16 GiB| 100 GiB||0|100' if command[0] == 'getquota_work3.sh' else 'DONE\n')
    monkeypatch.setattr(campaign.subprocess, 'run', scheduler)
    def submit(rows, root, identity):
        submitted.append((rows, root, identity)); return {'pilot': '456'}
    monkeypatch.setattr(campaign, 'submit', submit)
    decision = campaign.advance(p, who)
    assert decision['status'] == 'submitted_not_passed' and decision['pilot_submitted'] is True
    assert decision['jobs'] == {'pilot': '456'} and len(submitted) == 1
    rows, root, bound = submitted[0]
    assert len(rows) == 1 and rows[0]['stage'] == 'pilot' and rows[0]['dependency'] is None
    assert '12:00' in rows[0]['arguments'] and bound['readiness'] == proof
    assert root == Path(p['receipts']) / 'pilot'
