"""Allocated unit controls for private precision hooks and live-producer replay."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('precision_hook_fixture',
    Path(__file__).resolve().parents[1] / 'scripts/precision_continuation_hooks.py')
h = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(h)


@pytest.fixture(autouse=True)
def allocated():
    assert str(os.environ.get('LSB_JOBID', '')).isdigit(), 'All fixtures execute on allocated DTU HPC'


class Base:
    def __init__(self, arm, name, output, offset):
        self.arm = arm; self.name = name; self.output = Path(output); self.offset = offset
        self.records = []; self.logit_records = []; self.transforms = []; self.delegate_calls = []
    def _new_plan(self, point, objective=None):
        self.delegate_calls.append(('plan', objective)); return {'untouched': objective}
    def _callbacks(self, loss, key, directory, frozen, template, objective=None):
        return loss[0], loss[1], frozen
    def _replacement(self, point, prospective, legacy, original, directory, callbacks):
        def call(context):
            self.delegate_calls.append(('replacement', directory.name))
            self.records.append({'stage': point['stage'], 'status': 'passed', 'can_admit_new_child': True})
            return {'endpoint_policy': h.RAW_POLICY, 'nonraw': 'unchanged'}
        return call
    def combined(self, original): return original
    def dbmol(self, module, original): return original


def cfg(arm):
    return {'command_prefix': ['/sealed/python', '/sealed/reference.py'], 'environment': {},
            'policy_kind': 'protenix-original-fp64' if arm == 'protenix' else 'dbmol-finer-fp64-affinity',
            'policy_module': {'path': '/sealed/policy.py', 'sha256': 'a' * 64},
            'expected_identity': {'frozen': {'model': 'fixed'}},
            'historical_global': 20000 if arm == 'protenix' else 7500,
            'historical_original_directions': {'X': [{'row': 3, 'source': 1, 'target': 2}]
                                               if arm == 'protenix' else [{'row': 1, 'source': 0, 'target': 4}],
                                               'E_upper': []},
            'max_session_bytes': h.SESSION_CAP}


def setup(arm='protenix'):
    configuration = {'arms': {arm: cfg(arm)}}
    plan = {'arms': [{'id': arm, 'start_global': configuration['arms'][arm]['historical_global']}]}
    return SimpleNamespace(EndpointHooks=Base), plan, configuration


def test_class_installation_restores_original_on_success_and_exception():
    driver, plan, configuration = setup()
    with h.install_precision_hooks(driver, plan, configuration):
        assert driver.EndpointHooks is not Base and issubclass(driver.EndpointHooks, Base)
    assert driver.EndpointHooks is Base
    with pytest.raises(RuntimeError):
        with h.install_precision_hooks(driver, plan, configuration): raise RuntimeError('native failure')
    assert driver.EndpointHooks is Base


def test_nested_hook_installation_cannot_silently_replace_an_active_class():
    driver, plan, configuration = setup()
    with h.install_precision_hooks(driver, plan, configuration):
        with pytest.raises(ValueError, match='Nested'):
            with h.install_precision_hooks(driver, plan, configuration): pass


@pytest.mark.parametrize('mutation', ['arm', 'historical', 'budget', 'kind', 'identity'])
def test_undeclared_reference_configuration_is_rejected(mutation):
    driver, plan, configuration = setup()
    value = configuration['arms']['protenix']
    if mutation == 'arm': configuration['arms']['boltz'] = configuration['arms'].pop('protenix')
    elif mutation == 'historical': value['historical_global'] = 10000
    elif mutation == 'budget': value['max_session_bytes'] = h.SESSION_CAP + 1
    elif mutation == 'kind': value['policy_kind'] = 'raw-fallback'
    else: value['expected_identity'] = {}
    with pytest.raises(ValueError):
        with h.install_precision_hooks(driver, plan, configuration): pass


@pytest.mark.parametrize('arm,stage,offset,expected_scope', [
    ('protenix', 'native_gradient_checks', 20000, 'saved_failed_endpoint'),
    ('protenix', 'terminal_gradient_checks', 20000, 'not_preexisting_at_new_endpoint'),
    ('protenix', 'native_gradient_checks', 30000, 'not_preexisting_at_new_endpoint'),
    ('dbmol', 'native_gradient_checks', 7500, 'saved_failed_endpoint'),
    ('dbmol', 'terminal_gradient_checks', 7500, 'not_preexisting_at_new_endpoint')])
def test_future_boundaries_cannot_fabricate_historical_contrasts(arm, stage, offset, expected_scope):
    driver, plan, configuration = setup(arm); state = Base(plan['arms'][0], arm, '/new/owned', offset)
    directions, scope = h._original_contrasts(state, {'stage': stage}, configuration['arms'][arm])
    assert scope == expected_scope
    assert directions == (configuration['arms'][arm]['historical_original_directions']
                          if scope == 'saved_failed_endpoint' else {'X': [], 'E_upper': []})


@pytest.mark.parametrize('objective', ['dbmol_contact_loss', 'dbmol_confidence_loss', 'dbmol_total_loss'])
def test_other_dbmol_component_policies_and_pilot_hook_remain_original(objective, tmp_path):
    driver, plan, configuration = setup('dbmol')
    with h.install_precision_hooks(driver, plan, configuration):
        state = driver.EndpointHooks(plan['arms'][0], 'dbmol', tmp_path, 7500)
        point = {'stage': 'native_gradient_checks'}
        assert state._new_plan(point, objective) == {'untouched': objective}
        result = state._replacement(point, {}, None, None, tmp_path / objective, (None, None, None))({})
        assert result == {'endpoint_policy': h.RAW_POLICY, 'nonraw': 'unchanged'}
        assert state.records[-1]['objective'] == objective
        assert state.records[-1]['endpoint_policy'] == h.RAW_POLICY
        assert state.records[-1]['reference_session'] is None
        unchanged = lambda: 'unchanged-pilot'
        assert state.combined(unchanged) is unchanged


def test_original_finite_nonzero_fp32_raw_contrast_requirement_is_preserved():
    import numpy as np
    values = {f: np.tile(np.asarray([.25, .25, .5], np.float32), (4, 1)) for f in ('X', 'E_upper')}
    gradients = {f: np.tile(np.asarray([1, 2, 3], np.float32), (4, 1)) for f in values}
    h._nonzero({'gx': gradients['X'], 'ge': gradients['E_upper']}, {'values': values})
    for invalid in (np.zeros_like(gradients['X']), gradients['X'].astype(np.float64),
                    np.full_like(gradients['X'], np.nan)):
        with pytest.raises(ValueError, match='Original finite/nonzero'):
            h._nonzero({'gx': invalid, 'ge': gradients['E_upper']}, {'values': values})


def _linear_fixture(monkeypatch, tmp_path, *, during_change=False, after_change=False,
                    reference_wrong=False, session_fails=False):
    import numpy as np
    phase = {'active': False, 'finished': False, 'sessions': 0}
    values = {f: np.tile(np.asarray([.25, .25, .5], np.float32), (4, 1)) for f in ('X', 'E_upper')}
    weights = {f: np.tile(np.asarray([1., 2., 3.], np.float64), (4, 1)) for f in values}
    def primal(v): return np.float32(sum(np.sum(v[f].astype(np.float64) * weights[f]) for f in v))
    def derivative(v):
        factor = 2. if (during_change and phase['active']) or (after_change and phase['finished']) else 1.
        return primal(v), {f: (w * factor).astype(np.float32) for f, w in weights.items()}
    def frozen(): return {'model': 'original', 'context': 'original'}
    class ToySession:
        def __init__(self, config, directory, key_data):
            if session_fails: raise ValueError('Exclusive GPU co-residency unavailable')
            self.directory = Path(directory); self.directory.mkdir()
            self.requests = 0; self.closed = False; phase['active'] = True; phase['sessions'] += 1
            self.identity = {'status': 'ready_full_double_reference', 'frozen': {'model': 'reference'},
                             'optimizer_updates': 0, 'precision': h.PRECISION}
            h.write(self.directory / 'invocation.json', {'argv': config['command_prefix'], 'environment': {}, 'optimizer_updates': 0})
            h.write(self.directory / 'identity.json', self.identity)
        def evaluate(self, v, gradient):
            assert phase['active']
            self.requests += 1
            scalar = np.float64(sum(np.sum(v[f] * weights[f]) for f in v))
            return (scalar, {f: w * (1.5 if reference_wrong else 1.) for f, w in weights.items()}) if gradient else scalar
        def finish(self):
            self.closed = True; phase.update(active=False, finished=True)
            h.write(self.directory / 'client-raw-point-index.json', {'status': 'allocated_toy_session_only',
                    'requests': self.requests, 'points': [], 'optimizer_updates': 0})
            h.write(self.directory / 'close.json', {'status': 'closed', 'frozen': self.identity['frozen'],
                'audit': {'status': 'passed_full_FP64_reference_audit', 'floating_operations_float64': True,
                          'narrow_float_avals': [], 'frozen_unchanged': True, 'original_results_changed': False,
                          'optimizer_updates': 0}})
        def abort(self): phase['active'] = False
    parent_policy = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/protenix-endpoint-fix-release-09/source/scripts/protenix_precision_admission.py')
    policy = h._module({'path': str(parent_policy), 'sha256': 'ae549f19a8af11453b0bb01154bde61cd63f4fa8ad1676759c7427e9049ddf68'},
                       'precision_hook_real_allocated_toy_policy')
    monkeypatch.setattr(h, 'Session', ToySession)
    monkeypatch.setattr(h, '_policy', lambda *args: (policy, h.PRO_POLICY))
    monkeypatch.setattr(h, '_key_data', lambda key: list(key))
    driver, plan, configuration = setup()
    return driver, plan, configuration, values, weights, (primal, derivative, frozen), phase


def _replacement_fixture(driver, plan, configuration, values, weights, callbacks, tmp_path):
    import numpy as np
    state = driver.EndpointHooks(plan['arms'][0], 'protenix', tmp_path, 20000)
    point = {'stage': 'native_gradient_checks', 'values': values}
    directory = tmp_path / 'native_gradient_checks-prospective'; directory.mkdir()
    actual = state._callbacks(callbacks[:2], [0, 17], directory, callbacks[2], SimpleNamespace())
    prospective = state._new_plan(point)
    def legacy(): pass
    replacement = state._replacement(point, prospective,
        compile("raise ValueError('original coarse failure preserved')", '<old-coarse>', 'exec'),
        legacy, directory, actual)
    context = {'gx': weights['X'].astype(np.float32), 'ge': weights['E_upper'].astype(np.float32)}
    return state, directory, replacement, context


def test_full_precision_hook_measures_live_production_during_residency_and_restores_it(monkeypatch, tmp_path):
    args = _linear_fixture(monkeypatch, tmp_path)
    driver, plan, configuration, values, weights, callbacks, phase = args
    with h.install_precision_hooks(driver, plan, configuration):
        state, directory, replacement, context = _replacement_fixture(*args[:3], values, weights, callbacks, tmp_path)
        result = replacement(context)
    assert phase['sessions'] == 1 and phase['finished'] and not phase['active']
    assert result['endpoint_policy'] == h.PRO_POLICY
    assert result['prospective_endpoint_admission']['status'] == 'passed'
    assert result['original_coarse_observation']['status'] == 'failed_original_coarse_observation_preserved'
    assert not result['original_failed_parent_recertified']
    record = state.records[0]
    assert record['objective'] == 'protenix_contact_loss' and record['endpoint_policy'] == h.PRO_POLICY
    assert set(record['reference_session']) == {'invocation', 'identity', 'close', 'residency'}
    for row in record['reference_session'].values(): assert h.binding(row['path']) == row
    residency = json.loads(Path(record['reference_session']['residency']['path']).read_bytes())
    assert residency['status'] == 'passed_live_production_reference_co_residency'
    assert residency['production_measurements_before'] == residency['production_measurements_during'] == residency['production_measurements_after']
    assert residency['frozen_before'] == residency['frozen_during'] == residency['frozen_after']
    assert residency['input_FP32_baseline_unchanged'] and residency['reference_closed_before_return']
    assert residency['no_runtime_fallback'] and residency['production_optimizer_updates'] == 0
    assert all(row['new_reference_requests'] == 0 for row in json.loads((directory / 'negative-controls.json').read_bytes()))


@pytest.mark.parametrize('failure', ['during', 'after', 'reference', 'exclusive'])
def test_live_production_or_reference_failure_blocks_without_fallback(failure, monkeypatch, tmp_path):
    args = _linear_fixture(monkeypatch, tmp_path, during_change=failure == 'during', after_change=failure == 'after',
                           reference_wrong=failure == 'reference', session_fails=failure == 'exclusive')
    driver, plan, configuration, values, weights, callbacks, phase = args
    with h.install_precision_hooks(driver, plan, configuration):
        state, directory, replacement, context = _replacement_fixture(*args[:3], values, weights, callbacks, tmp_path)
        with pytest.raises(ValueError): replacement(context)
    assert not phase['active']
    assert not any(record['status'] == 'passed' for record in state.records)


def test_reference_configuration_mutation_after_installation_is_not_accepted(monkeypatch, tmp_path):
    args = _linear_fixture(monkeypatch, tmp_path)
    driver, plan, configuration, values, weights, callbacks, phase = args
    with h.install_precision_hooks(driver, plan, configuration):
        state, directory, replacement, context = _replacement_fixture(*args[:3], values, weights, callbacks, tmp_path)
        configuration['arms']['protenix']['environment']['JAX_ENABLE_X64'] = '0'
        with pytest.raises(ValueError, match='configuration mutated'): replacement(context)
    assert phase['sessions'] == 0


def test_all_reference_output_is_charged_and_no_oversize_proof_is_discarded(tmp_path):
    file = tmp_path / 'large-proof'; file.write_bytes(b'keep')
    with file.open('r+b') as stream: stream.truncate(h.SESSION_CAP + 1)
    with pytest.raises(ValueError, match='cap exceeded'): h._budget(tmp_path, cfg('protenix'))
    assert file.stat().st_size == h.SESSION_CAP + 1


def test_reference_budget_rejects_symlinked_output_without_following_it(tmp_path):
    target = tmp_path / 'point'; target.write_text('exact point', encoding='utf8')
    (tmp_path / 'alias').symlink_to(target)
    with pytest.raises(ValueError, match='linked session output'): h._budget(tmp_path, cfg('protenix'))


@pytest.mark.parametrize('actual,expected,accept', [
    ({'frozen': {'model': 'a', 'context': 'b'}}, {'frozen': {'model': 'a'}}, True),
    ({'frozen': {'model': 'a'}}, {'frozen': {'model': 'b'}}, False),
    ({'updates': False}, {'updates': 0}, False),
    ({'updates': 1}, {'updates': 0}, False)])
def test_reference_identity_subset_is_typed_and_cannot_accept_wrong_frozen_state(actual, expected, accept):
    assert h._subset(actual, expected) is accept


def test_policy_module_bytes_are_authenticated_before_import(tmp_path):
    p = tmp_path / 'policy.py'; p.write_text("raise RuntimeError('must not import corrupted code')", encoding='utf8')
    with pytest.raises(ValueError, match='Policy module bytes differ'):
        h._module({'path': str(p), 'sha256': '0' * 64}, 'must_not_import')


def rpc_session(monkeypatch, tmp_path, *, response_mutation=None, close_mutation=None, identity_mutation=None):
    import numpy as np
    monkeypatch.setenv('LSB_QUEUE', 'gpul40s')
    configuration = cfg('protenix'); point = {'X': np.ones((4, 3), np.float64), 'E_upper': np.ones((4, 3), np.float64)}
    digest = hashlib.sha256(b''.join(point[f].tobytes() for f in ('X', 'E_upper'))).hexdigest()
    identity = {'status': 'ready_full_double_reference', 'frozen': {'model': 'fixed'},
                'optimizer_updates': 0, 'precision': h.PRECISION}
    response = {'id': 0, 'graph_sha256': digest, 'key_data': [0, 17], 'value': 1.0,
                'gradients': {f: np.ones((4, 3), np.float64).tolist() for f in point}}
    closing = {'status': 'closed', 'frozen': {'model': 'fixed'}, 'audit': {
        'status': 'passed_full_FP64_reference_audit', 'floating_operations_float64': True,
        'narrow_float_avals': [], 'frozen_unchanged': True, 'original_results_changed': False, 'optimizer_updates': 0}}
    if identity_mutation: identity.update(identity_mutation)
    if response_mutation: response.update(response_mutation)
    if close_mutation: closing['audit'].update(close_mutation)
    packets = [identity, response, closing]; calls = []
    process = SimpleNamespace(stdin=io.StringIO(), returncode=None)
    process.wait = lambda timeout: setattr(process, 'returncode', 0)
    process.poll = lambda: process.returncode
    process.terminate = lambda: setattr(process, 'returncode', -15)
    process.kill = lambda: setattr(process, 'returncode', -9)
    def popen(command, **kwargs): calls.append((command, kwargs)); return process
    monkeypatch.setattr(h.subprocess, 'Popen', popen)
    monkeypatch.setattr(h.Session, '_read', lambda self: packets.pop(0))
    return configuration, point, process, calls


def test_rpc_binds_actual_key_graph_gradient_flag_and_closes_before_return(monkeypatch, tmp_path):
    configuration, point, process, calls = rpc_session(monkeypatch, tmp_path)
    session = h.Session(configuration, tmp_path / 'session', [0, 17])
    scalar, gradients = session.evaluate(point, True)
    session.finish()
    sent = [json.loads(row) for row in process.stdin.getvalue().splitlines()]
    assert sent[0]['id'] == 0 and sent[0]['gradient'] is True and sent[0]['key_data'] == [0, 17]
    assert sent[0]['X'] == point['X'].tolist() and set(gradients) == {'X', 'E_upper'}
    assert sent[1] == {'action': 'close'} and session.closed and process.returncode == 0
    assert calls[0][0][-2:] == ['--output', str(tmp_path / 'session/worker')]
    index = json.loads((tmp_path / 'session/client-raw-point-index.json').read_bytes())
    assert index['status'] == 'complete_actual_reference_points' and index['requests'] == 1
    assert index['cached_or_interpolated_points'] == 0 and len(index['points']) == 1
    raw = index['points'][0]['raw']; assert h.binding(raw['path']) == raw
    import numpy as np
    with np.load(raw['path'], allow_pickle=False) as saved:
        for field in ('X', 'E_upper'):
            assert saved[field].dtype == np.float64 and saved[field].tobytes() == point[field].tobytes()
            assert saved['gradient_' + field].dtype == np.float64
        assert saved['key_data'].tolist() == [0, 17] and saved['gradient'].shape == ()


@pytest.mark.parametrize('mutation', [{'id': True}, {'id': 1}, {'graph_sha256': '0' * 64}, {'key_data': [0, 18]}])
def test_rpc_cannot_accept_wrong_request_graph_or_caller_key(mutation, monkeypatch, tmp_path):
    configuration, point, process, calls = rpc_session(monkeypatch, tmp_path, response_mutation=mutation)
    session = h.Session(configuration, tmp_path / 'session', [0, 17])
    try:
        with pytest.raises(ValueError): session.evaluate(point, True)
    finally: session.abort()
    assert not session.closed


@pytest.mark.parametrize('mutation', [{'frozen': {'model': 'different'}}, {'optimizer_updates': 1},
    {'precision': {'final_reduction_only': True}}, {'status': 'ready_cpu_fallback'}])
def test_rpc_startup_cannot_accept_wrong_identity_dtype_updates_or_cpu_fallback(mutation, monkeypatch, tmp_path):
    configuration, point, process, calls = rpc_session(monkeypatch, tmp_path, identity_mutation=mutation)
    with pytest.raises(ValueError): h.Session(configuration, tmp_path / 'session', [0, 17])
    assert process.returncode is not None


@pytest.mark.parametrize('mutation', [{'floating_operations_float64': False}, {'narrow_float_avals': ['float32']},
    {'frozen_unchanged': False}, {'original_results_changed': True}, {'optimizer_updates': 1}])
def test_rpc_close_audit_failure_cannot_be_reported_as_completed(mutation, monkeypatch, tmp_path):
    configuration, point, process, calls = rpc_session(monkeypatch, tmp_path, close_mutation=mutation)
    session = h.Session(configuration, tmp_path / 'session', [0, 17]); session.evaluate(point, True)
    try:
        with pytest.raises(ValueError): session.finish()
    finally: session.abort()
    assert not session.closed and not (tmp_path / 'session/close.json').exists()


def test_only_own_plan_sha_argument_is_expanded_without_mutating_sealed_configuration():
    config = cfg('protenix'); config['command_prefix'] += [
        '--reference-plan', '/sealed/plan.json', '--reference-plan-sha256', '${OWN_PLAN_SHA256}']
    original = deepcopy(config)
    result = h.expanded_command(config, 'a' * 64)
    assert result[-1] == 'a' * 64 and result[:-1] == original['command_prefix'][:-1]
    assert config == original


@pytest.mark.parametrize('command', [
    ['/python', '${OTHER}'],
    ['/python', '${OWN_PLAN_SHA256}'],
    ['/python', '--reference-plan-sha256', 'prefix${OWN_PLAN_SHA256}'],
    ['/python', '--reference-plan-sha256', '${OWN_PLAN_SHA256}', '${OWN_PLAN_SHA256}'],
    ['/python', '--reference-plan-sha256', '${OWN_PLAN_SHA256}', '--reference-plan-sha256', 'a' * 64],
    ['/python', '--reference-plan-sha256']])
def test_misplaced_duplicate_partial_or_unknown_command_placeholders_are_rejected(command):
    config = cfg('protenix'); config['command_prefix'] = command
    with pytest.raises(ValueError): h.expanded_command(config, 'a' * 64)


@pytest.mark.parametrize('digest', [None, '', 'A' * 64, 'a' * 63])
def test_own_plan_expansion_requires_the_actual_lower_hex_digest(digest):
    config = cfg('protenix'); config['command_prefix'] += ['--reference-plan-sha256', '${OWN_PLAN_SHA256}']
    with pytest.raises(ValueError, match='SHA256 required'):
        h.expanded_command(config, digest)


def test_a_different_literal_own_plan_hash_is_not_accepted():
    config = cfg('protenix'); config['command_prefix'] += ['--reference-plan-sha256', 'b' * 64]
    with pytest.raises(ValueError, match='plan SHA differs'):
        h.expanded_command(config, 'a' * 64)


def test_environment_templates_are_never_substituted_or_accepted():
    config = cfg('protenix'); config['environment'] = {'XLA_FLAGS': '${OWN_PLAN_SHA256}'}
    with pytest.raises(ValueError, match='placeholders are forbidden'):
        h.expanded_command(config, 'a' * 64)


def test_dbmol_literal_reference_command_has_no_plan_hash_substitution():
    config = cfg('dbmol')
    assert h.expanded_command(config, 'a' * 64) == config['command_prefix']
