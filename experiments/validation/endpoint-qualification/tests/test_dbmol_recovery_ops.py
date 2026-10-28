"""Allocated metadata controls for the exact preserved historical empty lock."""
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import re
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('dbmol_empty_lock_ops_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/dbmol_recovery_ops.py')
ops = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ops)


@pytest.fixture(autouse=True)
def allocated():
    assert str(os.environ.get('LSB_JOBID', '')).isdigit(), 'Fixtures require allocated DTU HPC'


def require(value, message):
    if not value: raise ValueError(message)


def empty_lock():
    return {'path': ops.HISTORICAL_EMPTY_LOCK, 'bytes': 0, 'sha256': ops.EMPTY_SHA256}


def parent(returned=None, failure=None):
    calls = {'positive': [], 'binding': []}
    def binding(path):
        calls['binding'].append(path)
        if failure is not None: raise ValueError(failure)
        return deepcopy(empty_lock() if returned is None else returned)
    def positive(row):
        calls['positive'].append(deepcopy(row))
        require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}
            and isinstance(row['path'], str) and type(row['bytes']) is int and row['bytes'] > 0
            and isinstance(row['sha256'], str) and re.fullmatch('[a-f0-9]{64}', row['sha256']),
            'Inherited positive proof validation failed')
        return row
    return SimpleNamespace(require=require, binding=binding, check_binding=positive), calls


def test_exact_historical_empty_regular_lock_is_checked_without_positive_bypass():
    op, calls = parent(); row = empty_lock()
    assert ops.check_preserved_binding(op, row) == row
    assert calls == {'positive': [], 'binding': [ops.HISTORICAL_EMPTY_LOCK]}


def test_nonempty_proof_uses_the_unchanged_inherited_validator():
    op, calls = parent(); row = {'path': '/work3/owned/proof.json', 'bytes': 128, 'sha256': 'a' * 64}
    assert ops.check_preserved_binding(op, row) is row
    assert calls == {'positive': [row], 'binding': []}


@pytest.mark.parametrize('value', [False, True, 0.0, '0', None, -1])
def test_nonliteral_or_invalid_byte_counts_are_not_empty_lock_authority(value):
    op, calls = parent(); row = empty_lock(); row['bytes'] = value
    with pytest.raises(ValueError): ops.check_preserved_binding(op, row)
    assert not calls['binding']


@pytest.mark.parametrize('path', [
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-11/jobs.lock',
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-12/jobs.lock',
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-10/jobs.json',
    'dbmol-endpoint-recovery-10/jobs.lock',
    '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/dbmol-endpoint-recovery-10/../dbmol-endpoint-recovery-10/jobs.lock'])
def test_all_other_empty_paths_are_rejected_before_filesystem_validation(path):
    op, calls = parent(); row = empty_lock(); row['path'] = path
    with pytest.raises(ValueError, match='Only the exact authenticated'):
        ops.check_preserved_binding(op, row)
    assert not calls['binding']


@pytest.mark.parametrize('digest', ['', '0' * 64, 'E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855'])
def test_only_the_exact_empty_sha_is_accepted(digest):
    op, calls = parent(); row = empty_lock(); row['sha256'] = digest
    with pytest.raises(ValueError, match='Only the exact authenticated'):
        ops.check_preserved_binding(op, row)
    assert not calls['binding']


@pytest.mark.parametrize('mutation', ['extra', 'missing_path', 'missing_bytes', 'missing_sha256'])
def test_closed_binding_schema_is_required(mutation):
    op, calls = parent(); row = empty_lock()
    if mutation == 'extra': row['allowed'] = True
    else: row.pop(mutation.removeprefix('missing_'))
    with pytest.raises(ValueError): ops.check_preserved_binding(op, row)
    assert not calls['binding']


@pytest.mark.parametrize('row', [None, [], {}, 'jobs.lock'])
def test_nonbinding_inputs_keep_the_inherited_failure(row):
    op, calls = parent()
    with pytest.raises(ValueError, match='Inherited positive proof'):
        ops.check_preserved_binding(op, row)
    assert not calls['binding']


@pytest.mark.parametrize('key,value', [('bytes', 1), ('sha256', 'a' * 64), ('path', '/work3/other/jobs.lock')])
def test_current_metadata_must_equal_the_authentic_empty_receipt(key, value):
    row = empty_lock(); current = empty_lock(); current[key] = value
    op, calls = parent(returned=current)
    with pytest.raises(ValueError, match='Historical empty lock is changed'):
        ops.check_preserved_binding(op, row)
    assert calls['binding'] == [ops.HISTORICAL_EMPTY_LOCK]


@pytest.mark.parametrize('failure', ['Input is nonregular', 'Input is a symlink', 'Parent directory is linked'])
def test_absolute_regular_unlinked_parent_validation_is_not_suppressed(failure):
    op, calls = parent(failure=failure)
    with pytest.raises(ValueError, match=failure):
        ops.check_preserved_binding(op, empty_lock())
    assert calls['binding'] == [ops.HISTORICAL_EMPTY_LOCK]


def test_external_adoption_routes_each_protected_proof_through_strict_helper(monkeypatch):
    import sys
    policy = {'closed': 'fixture'}
    monkeypatch.setitem(sys.modules, 'dbmol_endpoint_precision', SimpleNamespace(finer_policy_spec=lambda: policy))
    valid = {'path': '/work3/owned/proof.json', 'bytes': 128, 'sha256': 'a' * 64}
    op, calls = parent()
    manifest = {'version': 'dbmol-separate-affinity-reference-finer-source-v1',
                'cpu_counts': ops.COUNTS, 'parent_sources': []}
    plan = {'version': 'dbmol-separate-affinity-reference-finer-v1', 'root': str(ops.ROOT),
            'original_results_changed': False, 'optimizer_updates': 0,
            'precision_policy': policy, 'protected_inputs': [valid, empty_lock()]}
    op.verify = lambda *args: manifest; op.read = lambda *args: plan
    assert ops.external(op, 'sealed-fixture-pin') == (manifest, plan)
    assert calls == {'positive': [valid], 'binding': [ops.HISTORICAL_EMPTY_LOCK]}


def test_external_adoption_rejects_an_unrelated_zero_byte_proof(monkeypatch):
    import sys
    policy = {'closed': 'fixture'}
    monkeypatch.setitem(sys.modules, 'dbmol_endpoint_precision', SimpleNamespace(finer_policy_spec=lambda: policy))
    op, calls = parent(); invalid = empty_lock(); invalid['path'] = '/work3/other/jobs.lock'
    manifest = {'version': 'dbmol-separate-affinity-reference-finer-source-v1',
                'cpu_counts': ops.COUNTS, 'parent_sources': []}
    plan = {'version': 'dbmol-separate-affinity-reference-finer-v1', 'root': str(ops.ROOT),
            'original_results_changed': False, 'optimizer_updates': 0,
            'precision_policy': policy, 'protected_inputs': [invalid]}
    op.verify = lambda *args: manifest; op.read = lambda *args: plan
    with pytest.raises(ValueError, match='Only the exact authenticated'):
        ops.external(op, 'sealed-fixture-pin')
    assert not calls['binding']
