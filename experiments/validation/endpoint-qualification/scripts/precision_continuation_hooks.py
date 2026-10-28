"""Private continuation hooks for authenticated separate full-FP64 references.

Only the raw-FD replacement is specialized. The authenticated driver's original
component, native-oracle, null, pilot, runtime, finite and frozen checks remain.
The original optimizer always evaluates its original production model and dtype.
Every reference session must coexist with that live producer on the assigned GPU,
close successfully, and leave exact production measurements and identities intact.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys


VERSION = 'precision-reference-continuation-hooks-v1'
PRO_POLICY = 'protenix-separate-precision-admission-v1'
DB_POLICY = 'dbmol-separate-precision-finer-admission-v1'
RAW_POLICY = 'prospective-endpoint-admission-v3'
PRECISION = {'full_model_float64': True, 'parameters_float64': True,
             'native_features_float64': True, 'native_operations_float64': True,
             'final_reduction_only': False}
SESSION_CAP = 64 * 1024**2
POLICIES = {
    'protenix': {'protenix_contact_loss': PRO_POLICY},
    'dbmol': {'dbmol_affinity_loss': DB_POLICY, 'dbmol_contact_loss': RAW_POLICY,
              'dbmol_confidence_loss': RAW_POLICY, 'dbmol_total_loss': RAW_POLICY}}


def require(value, message):
    if not value: raise ValueError(message)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def binding(path):
    p = Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink()
            and all(not part.is_symlink() for part in p.parents), 'Ordinary absolute proof file required')
    with p.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest}


def _budget(directory, config):
    require(type(config.get('max_session_bytes')) is int and config['max_session_bytes'] == SESSION_CAP,
            'Exact per-boundary64MiB reference budget required')
    total = 0
    for path in Path(directory).rglob('*'):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Special or linked session output')
        if path.is_file(): total += path.stat().st_size
    require(total <= SESSION_CAP, 'Complete boundary/reference metadata cap exceeded; no output may be dropped')
    return total


def _numpy():
    require(str(os.environ.get('LSB_JOBID', '')).isdigit(), 'Allocated DTU numerical execution required')
    import numpy as np
    return np


def _key_data(key):
    np = _numpy()
    import jax
    values = np.asarray(jax.random.key_data(key))
    require(values.dtype == np.uint32 and values.shape == (2,), 'Exact two-word native PRNG key required')
    return values.tolist()


def _record(value):
    np = _numpy(); array = np.asarray(value)
    require(np.isfinite(array).all(), 'Nonfinite production/reference measurement')
    return {'dtype': str(array.dtype), 'shape': list(array.shape),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def _snapshot(values):
    require(set(values) == {'X', 'E_upper'}, 'Both original raw graph fields are required')
    return {field: _record(values[field]) for field in ('X', 'E_upper')}


def _measurement(callbacks, values):
    np = _numpy(); primal, autodiff, frozen = callbacks
    p = primal(values); scalar, gradients = autodiff(values)
    require(set(gradients) == {'X', 'E_upper'}, 'Both original production raw gradients required')
    return {'primal': _record(np.asarray(p)), 'autodiff_scalar': _record(np.asarray(scalar)),
            'gradients': {field: _record(gradients[field]) for field in ('X', 'E_upper')},
            'frozen': json.loads(_json(frozen()))}


def _module(row, name):
    require(isinstance(row, dict) and isinstance(row.get('path'), str)
            and isinstance(row.get('sha256'), str), 'Exact policy-module binding required')
    observed = binding(row['path'])
    require(observed['sha256'] == row['sha256'] and ('bytes' not in row or observed['bytes'] == row['bytes']),
            'Policy module bytes differ')
    spec = importlib.util.spec_from_file_location(name, row['path'])
    require(spec is not None and spec.loader is not None, 'Policy-module loader unavailable')
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def _policy(config, arm):
    source = _module(config['policy_module'], '_precision_continuation_policy_' + arm)
    if arm == 'protenix':
        require(config['policy_kind'] == 'protenix-original-fp64'
                and source.VERSION == PRO_POLICY
                and source.EPSILONS == (.003, .001, .0003, .0001, .00003),
                'Original Protenix separate-reference policy differs')
        return source, PRO_POLICY
    require(arm == 'dbmol' and config['policy_kind'] == 'dbmol-finer-fp64-affinity'
            and source.FINER_POLICY_VERSION == DB_POLICY, 'Authorized DBMol finer policy differs')
    original = _module({'path': str(source.POLICY_SOURCE), 'sha256': source.POLICY_SOURCE_SHA256},
                       '_precision_continuation_dbmol_parent_policy')
    return source.FinerPolicy({'precision_policy': deepcopy(config['precision_policy'])}, original), DB_POLICY


def _subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _subset(actual[key], value)
                                                for key, value in expected.items())
    return type(actual) is type(expected) and actual == expected


def expanded_command(config, plan_sha256):
    """Resolve only the owning-plan hash argument, avoiding a circular plan hash.

    The token is data in the sealed plan, never shell interpolation. No other
    argv or environment placeholder is accepted or substituted.
    """
    command = config.get('command_prefix')
    require(isinstance(command, list) and command and all(isinstance(x, str) and x for x in command),
            'Exact reference command prefix required')
    command = list(command); token = '${OWN_PLAN_SHA256}'
    positions = [index for index, part in enumerate(command) if '${' in part]
    if positions:
        require(len(positions) == 1 and command[positions[0]] == token
                and command.count('--reference-plan-sha256') == 1
                and positions[0] == command.index('--reference-plan-sha256') + 1,
                'Only the sole owning-plan SHA argument may contain the exact placeholder')
        require(isinstance(plan_sha256, str) and re.fullmatch('[a-f0-9]{64}', plan_sha256),
                'Authenticated owning-plan SHA256 required for expansion')
        command[positions[0]] = plan_sha256
    if '--reference-plan-sha256' in command:
        require(command.count('--reference-plan-sha256') == 1
                and command.index('--reference-plan-sha256') + 1 < len(command),
                'Unique owning-plan SHA option/value required')
        actual = command[command.index('--reference-plan-sha256') + 1]
        require(isinstance(plan_sha256, str) and re.fullmatch('[a-f0-9]{64}', plan_sha256)
                and actual == plan_sha256, 'Literal or resolved reference plan SHA differs')
    require(not any('${' in part for part in command)
            and not any('${' in key or '${' in value for key, value in config.get('environment', {}).items()),
            'Unresolved argv/environment placeholders are forbidden')
    return command


class Session:
    """One strict JSON-RPC reference process; no retry or alternate-device path."""
    def __init__(self, config, directory, key_data):
        require(os.environ.get('LSB_QUEUE') == 'gpul40s'
                and str(os.environ.get('LSB_JOBID', '')).isdigit(), 'Real allocated GPU session required')
        self.config = config; self.directory = Path(directory); self.key_data = list(key_data)
        require(len(self.key_data) == 2 and all(type(x) is int and 0 <= x < 2**32 for x in self.key_data),
                'Actual two-word key data required')
        self.directory.mkdir(exist_ok=False)
        self.process = None; self.closed = False; self.sequence = 0; self.requests = 0; self.points = []
        command = list(config['command_prefix'])
        require(command and all(isinstance(part, str) and part for part in command), 'Exact reference command required')
        require(not any('${' in part for part in command), 'Reference command must already have its exact plan hash resolved')
        if config['policy_kind'] == 'protenix-original-fp64':
            command += ['--output', str(self.directory / 'worker')]
        environment = dict(os.environ); environment.update(config.get('environment', {}))
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()),
                'Reference environment must be explicit string metadata')
        write(self.directory / 'invocation.json', {'version': VERSION, 'argv': command,
              'environment': config.get('environment', {}), 'job_id': os.environ['LSB_JOBID'],
              'queue': os.environ['LSB_QUEUE'], 'key_data': self.key_data,
              'client_raw_point_index_path': str(self.directory / 'client-raw-point-index.json'),
              'live_production_required': True, 'co_residency_required': True,
              'runtime_fallback_allowed': False, 'production_optimizer_updates': 0, 'optimizer_updates': 0})
        self.log = (self.directory / 'stderr.log').open('x')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, bufsize=1, env=environment)
            self.identity = self._read()
            require(self.identity.get('status') == 'ready_full_double_reference'
                    and self.identity.get('optimizer_updates') == 0
                    and isinstance(self.identity.get('frozen'), dict) and self.identity['frozen'],
                    'Separate GPU reference failed to initialize while production remained live')
            expected = config.get('expected_identity')
            require(isinstance(expected, dict) and expected and _subset(self.identity, expected),
                    'Actual reference model/context/source/device identity differs')
            if config['policy_kind'] == 'protenix-original-fp64':
                require(self.identity.get('precision') == PRECISION,
                        'Reference does not declare complete FP64 execution')
            write(self.directory / 'identity.json', self.identity)
            _budget(self.directory.parent, config)
        except BaseException:
            self.abort(); raise

    def _read(self):
        require(self.process is not None and self.process.stdout is not None, 'Reference process is unavailable')
        with selectors.DefaultSelector() as wait:
            wait.register(self.process.stdout, selectors.EVENT_READ)
            require(bool(wait.select(1200)), 'Reference response timeout; no fallback permitted')
        line = self.process.stdout.readline()
        require(bool(line), 'Reference process exited; preserve its original stderr')
        def invalid(value): raise ValueError('Nonfinite JSON reference value: ' + value)
        response = json.loads(line, parse_constant=invalid)
        require(isinstance(response, dict), 'Reference response must be a closed object')
        return response

    def evaluate(self, values, gradient):
        np = _numpy()
        require(not self.closed and type(gradient) is bool, 'Closed session or nonliteral gradient flag')
        require(set(values) == {'X', 'E_upper'} and all(np.asarray(values[f]).dtype == np.float64
                and np.isfinite(values[f]).all() for f in values), 'Exact finite FP64 reference fields required')
        digest = hashlib.sha256(b''.join(np.asarray(values[f]).tobytes() for f in ('X', 'E_upper'))).hexdigest()
        request = {'id': self.sequence, 'gradient': gradient, 'graph_sha256': digest,
                   'key_data': self.key_data, **{f: np.asarray(values[f]).tolist() for f in ('X', 'E_upper')}}
        self.process.stdin.write(_json(request) + '\n'); self.process.stdin.flush()
        response = self._read()
        response_path = self.directory / ('response-' + str(self.sequence).zfill(4) + '.json')
        write(response_path, response)
        require(type(response.get('id')) is int and response['id'] == self.sequence
                and response.get('graph_sha256') == digest, 'Reference request/graph identity changed')
        if self.config['policy_kind'] == 'protenix-original-fp64':
            require(response.get('key_data') == self.key_data, 'Reference evaluated a different native key')
        self.sequence += 1; self.requests += 1
        _budget(self.directory.parent, self.config)
        scalar = np.asarray(response['value'], dtype=np.float64)
        require(scalar.shape == () and bool(np.isfinite(scalar)), 'Invalid independent reference scalar')
        data = {f: np.asarray(values[f], dtype=np.float64) for f in ('X', 'E_upper')}
        data.update(value=scalar, key_data=np.asarray(self.key_data, dtype=np.uint32), gradient=np.asarray(gradient))
        gradients = None
        if gradient:
            require(set(response.get('gradients', {})) == {'X', 'E_upper'}, 'Missing full reference raw gradients')
            gradients = {field: np.asarray(response['gradients'][field], dtype=np.float64) for field in ('X', 'E_upper')}
            require(all(gradients[f].shape == np.asarray(values[f]).shape and np.isfinite(gradients[f]).all()
                        for f in gradients), 'Invalid independent reference gradient shape/finite state')
            data.update({'gradient_' + field: value for field, value in gradients.items()})
        path = self.directory / ('point-client-' + str(request['id']).zfill(4) + '.npz')
        with path.open('xb') as stream: np.savez_compressed(stream, **data)
        self.points.append({'id': request['id'], 'gradient': gradient, 'graph_sha256': digest,
                            'key_data': self.key_data, 'raw': binding(path), 'response': binding(response_path)})
        _budget(self.directory.parent, self.config)
        return (scalar, gradients) if gradient else scalar

    def finish(self):
        require(not self.closed and self.requests > 0, 'Reference must execute actual fixed-point evaluations')
        self.process.stdin.write('{"action":"close"}\n'); self.process.stdin.flush()
        closing = self._read(); self.process.wait(timeout=60)
        require(closing.get('status') == 'closed' and self.process.returncode == 0
                and closing.get('frozen') == self.identity['frozen'], 'Reference close/frozen identity failed')
        audit = closing.get('audit', {})
        if self.config['policy_kind'] == 'protenix-original-fp64':
            require(audit.get('status') == 'passed_full_FP64_reference_audit'
                    and audit.get('floating_operations_float64') is True
                    and audit.get('narrow_float_avals') == [] and audit.get('frozen_unchanged') is True
                    and audit.get('original_results_changed') is False and audit.get('optimizer_updates') == 0,
                    'Full Protenix reference operation/frozen audit failed')
        else:
            require(audit.get('status') == 'passed_full_operation_FP64_audit'
                    and audit.get('narrow_floating_operations') == 0
                    and audit.get('production_execution_changed') is False,
                    'Full DBMol reference operation audit failed')
        require(len(self.points) == self.requests, 'Reference raw point inventory incomplete')
        write(self.directory / 'client-raw-point-index.json', {'status': 'complete_actual_reference_points',
              'requests': self.requests, 'points': self.points, 'optimizer_updates': 0,
              'input_transport': 'exact_current_FP64_graph_and_actual_native_key',
              'reference_frozen': self.identity['frozen'], 'cached_or_interpolated_points': 0})
        write(self.directory / 'close.json', closing)
        self.closed = True; self.log.close()
        _budget(self.directory.parent, self.config)
        return closing

    def abort(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=20)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=20)
        if hasattr(self, 'log') and not self.log.closed: self.log.close()


def _original_contrasts(state, point, config):
    historical = point['stage'] == 'native_gradient_checks' and state.offset == config['historical_global']
    if historical:
        expected = ({'X': [{'row': 3, 'source': 1, 'target': 2}], 'E_upper': []}
                    if state.name == 'protenix' else {'X': [{'row': 1, 'source': 0, 'target': 4}], 'E_upper': []})
        require(config['historical_original_directions'] == expected
                and state.offset == state.arm['start_global'], 'Historical contrasts or first endpoint changed')
        return deepcopy(expected), 'saved_failed_endpoint'
    return {'X': [], 'E_upper': []}, 'not_preexisting_at_new_endpoint'


def _nonzero(context, point):
    np = _numpy()
    if 'evaluation' in context:
        gradients = context['evaluation'].gradients; pairs = (('X', gradients.X), ('E_upper', gradients.E_upper))
    elif 'raw' in context:
        gradients = context['raw'].gradients; pairs = (('X', gradients.X), ('E_upper', gradients.E_upper))
    else: pairs = (('X', context['gx']), ('E_upper', context['ge']))
    for field, value in pairs:
        array = np.asarray(value)
        require(array.dtype == np.float32 and array.shape == np.asarray(point['values'][field]).shape
                and np.isfinite(array).all() and float((array.max(-1) - array.min(-1)).max()) > 1e-7,
                'Original finite/nonzero raw contrast failed: ' + field)


def _configuration(plan, configuration):
    require(isinstance(configuration, dict) and set(configuration) == {'arms'}
            and isinstance(configuration['arms'], dict) and configuration['arms'], 'Explicit reference configuration required')
    require(set(configuration['arms']) <= set(POLICIES), 'No undeclared precision-reference arm')
    for name, cfg in configuration['arms'].items():
        require(isinstance(cfg, dict) and isinstance(cfg.get('command_prefix'), list)
                and isinstance(cfg.get('environment', {}), dict)
                and cfg.get('policy_kind') == ('protenix-original-fp64' if name == 'protenix' else 'dbmol-finer-fp64-affinity')
                and type(cfg.get('historical_global')) is int
                and cfg['historical_global'] == (20000 if name == 'protenix' else 7500)
                and isinstance(cfg.get('historical_original_directions'), dict)
                and type(cfg.get('max_session_bytes')) is int and cfg['max_session_bytes'] == SESSION_CAP
                and isinstance(cfg.get('expected_identity'), dict) and cfg['expected_identity'],
                'Closed reference arm configuration differs')
        rows = [row for row in plan.get('arms', []) if row.get('id') == name]
        require(len(rows) == 1 and rows[0]['start_global'] == cfg['historical_global'],
                'Reference first-boundary identity differs from continuation plan')
    return configuration


@contextmanager
def install_precision_hooks(driver, plan, reference_configuration):
    """Temporarily specialize the driver's class without editing its module bytes."""
    config = _configuration(plan, reference_configuration)
    config_seal = _json(config); original = driver.EndpointHooks
    require(not getattr(original, '_precision_reference_hook', False), 'Nested precision-hook installation is forbidden')

    class PrecisionEndpointHooks(original):
        _precision_reference_hook = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            require(self.name in config['arms'], 'This arm has no declared precision-reference configuration')
            self._precision_contexts = {}

        def _precision_objective(self, objective=None):
            return 'protenix_contact_loss' if self.name == 'protenix' else objective

        def _new_plan(self, point, objective=None):
            selected = self.name == 'protenix' or objective == 'dbmol_affinity_loss'
            if not selected: return super()._new_plan(point, objective)
            cfg = config['arms'][self.name]; contrasts, scope = _original_contrasts(self, point, cfg)
            return {'version': POLICIES[self.name][self._precision_objective(objective)],
                    'scope': scope, 'original_directions': contrasts, 'baseline': _snapshot(point['values']),
                    'selection': 'full_FP64_baseline_AD_and_geometry_before_perturbed_callbacks',
                    'reference_configuration_sha256': hashlib.sha256(_json(cfg).encode()).hexdigest(),
                    'original_failed_parent_recertified': False}

        def _callbacks(self, loss, key, directory, frozen, template, objective=None):
            callbacks = super()._callbacks(loss, key, directory, frozen, template, objective=objective)
            self._precision_contexts[str(Path(directory))] = {'key_data': _key_data(key),
                    'objective': self._precision_objective(objective), 'callbacks': callbacks}
            return callbacks

        def _replacement(self, point, prospective, legacy_code, original_function, policy_dir, callbacks):
            objective = 'protenix_contact_loss' if self.name == 'protenix' else Path(policy_dir).name
            if self.name == 'dbmol' and objective != 'dbmol_affinity_loss':
                delegated = super()._replacement(point, prospective, legacy_code, original_function, policy_dir, callbacks)
                def unchanged(context):
                    before = len(self.records)
                    try: return delegated(context)
                    finally:
                        for row in self.records[before:]:
                            row.update(endpoint_policy=RAW_POLICY, objective=objective, reference_session=None,
                                       reference_applicability='not_applicable_unchanged_raw_v3')
                return unchanged

            def replacement(context):
                np = _numpy(); require(_json(config) == config_seal, 'Reference configuration mutated')
                _nonzero(context, point)
                cfg = config['arms'][self.name]
                current = self._precision_contexts[str(Path(policy_dir))]
                require(current['objective'] == objective
                        and current['callbacks'][1] is callbacks[1] and current['callbacks'][2] is callbacks[2],
                        'Original live callback/key/objective identity differs')
                baseline = {f: np.asarray(point['values'][f]).copy() for f in ('X', 'E_upper')}
                require(all(v.dtype == np.float32 for v in baseline.values()), 'Original FP32 graph fields required')
                before_input = _snapshot(baseline); before = _measurement(callbacks, baseline)
                policy, version = _policy(cfg, self.name)
                session = None; result = None; primary = None; proof = None
                try:
                    session_config = deepcopy(cfg)
                    session_config['command_prefix'] = expanded_command(cfg, plan.get('_sha256'))
                    session = Session(session_config, Path(policy_dir) / 'reference-session', current['key_data'])
                    during = _measurement(callbacks, baseline)
                    require(during == before, 'Live production changed while the FP64 GPU process coexisted')
                    cache = {}; cursors = Counter()
                    def ref(values, kind):
                        digest = _json(_snapshot(values)); key = (digest, kind)
                        if key not in cache:
                            count = 3 if _snapshot(values) == _snapshot({f: v.astype(np.float64) for f, v in baseline.items()}) else 1
                            cache[key] = [session.evaluate(values, kind == 'ad') for _ in range(count)]
                        index = cursors[key]; cursors[key] += 1
                        return cache[key][min(index, len(cache[key]) - 1)]
                    contrasts, scope = _original_contrasts(self, point, cfg)
                    frozen_ref = {'production': before['frozen'], 'reference': session.identity['frozen']}
                    def frozen():
                        require(callbacks[2]() == before['frozen'], 'Original live production frozen identity changed')
                        return deepcopy(frozen_ref)
                    def evaluate(label, factor=1.):
                        cursors.clear()
                        def production(values):
                            scalar, gradients = callbacks[1](values)
                            return scalar, {f: (np.asarray(g) * np.float32(factor)).astype(np.float32)
                                            for f, g in gradients.items()}
                        value = policy.evaluate(baseline, callbacks[0], production,
                            lambda v: ref(v, 'primal'), lambda v: ref(v, 'ad'), frozen,
                            original_directions=contrasts, scope=scope, reference_precision=PRECISION,
                            original_observations=cfg.get('historical_failure'),
                            seal_callback=lambda sealed: write(Path(policy_dir) / (label + '-pair-seal.json'), sealed))
                        write(Path(policy_dir) / (label + '-admission.json'), value)
                        return value
                    result = evaluate('native')
                    negatives = []
                    for label, factor in (('zero', 0.), ('reversed', -1.), ('scaled', 1.5)):
                        calls = session.requests; control = evaluate(label, factor)
                        require(not control['can_admit_new_child'] and session.requests == calls,
                                'Wrong-gradient control passed or issued undeclared reference probes')
                        negatives.append({'label': label, 'rejected': True, 'new_reference_requests': 0})
                    write(Path(policy_dir) / 'negative-controls.json', negatives)
                    session.finish()
                    after = _measurement(callbacks, baseline)
                    require(after == before and _snapshot(baseline) == before_input,
                            'Exact production scalar/gradient/frozen/input replay changed after reference exit')
                    residency = {'version': VERSION, 'status': 'passed_live_production_reference_co_residency',
                                 'production_measurements_before': before, 'production_measurements_during': during,
                                 'production_measurements_after': after, 'input_FP32_baseline': before_input,
                                 'frozen_before': before['frozen'], 'frozen_during': during['frozen'],
                                 'frozen_after': after['frozen'],
                                 'input_FP32_baseline_unchanged': True, 'production_optimizer_updates': 0,
                                 'reference_closed_before_return': session.closed, 'no_runtime_fallback': True,
                                 'reference_client_raw_points': binding(session.directory / 'client-raw-point-index.json'),
                                 'scope': 'Actual live original producer evaluated before/during/after separate authenticated GPU reference; no optimizer or dtype change.'}
                    write(Path(policy_dir) / 'precision-residency.json', residency)
                    _budget(policy_dir, cfg)
                    proof = {'invocation': binding(session.directory / 'invocation.json'),
                             'identity': binding(session.directory / 'identity.json'),
                             'close': binding(session.directory / 'close.json'),
                             'residency': binding(Path(policy_dir) / 'precision-residency.json')}
                except BaseException as error:
                    primary = error
                finally:
                    if session is not None and not session.closed: session.abort()
                if primary is not None:
                    write(Path(policy_dir) / 'precision-execution-failure.json', {'status': 'failed',
                          'error': repr(primary), 'optimizer_updates': 0, 'runtime_fallback': False,
                          'original_failed_parent_recertified': False})
                    raise primary
                require(result is not None and proof is not None, 'Complete separate-reference evidence missing')
                write(Path(policy_dir) / 'admission.json', result)
                self.records.append({'stage': point['stage'], 'path': str(Path(policy_dir) / 'admission.json'),
                    'status': result['status'], 'can_admit_new_child': result['can_admit_new_child'],
                    'endpoint_policy': version, 'objective': objective, 'reference_session': proof})
                legacy = {'status': 'running', 'directions': [], 'rows': []}; local = dict(context); local['report'] = legacy
                legacy_dir = Path(policy_dir) / 'original-coarse'; legacy_dir.mkdir()
                if 'directory' in local: local['directory'] = legacy_dir
                try:
                    exec(legacy_code, original_function.__globals__, local)
                    legacy['status'] = 'passed_original_coarse_observation_only'
                except Exception as error:
                    legacy.update(status='failed_original_coarse_observation_preserved', error=repr(error))
                write(Path(policy_dir) / 'original-coarse.json', legacy)
                _budget(policy_dir, cfg)
                require(result['status'] == 'passed' and result['can_admit_new_child'] is True,
                        'Independent precision admission failed; original observations preserved')
                return {'endpoint_policy': version, 'prospective_endpoint_admission': result,
                        'reference_session': proof, 'original_coarse_observation': legacy,
                        'original_failed_run_preserved': True, 'original_failed_parent_recertified': False,
                        'original_raw_fd_gate_replaced_for_new_child_only': True}
            return replacement

        def dbmol(self, module, function):
            wrapped = super().dbmol(module, function)
            def mixed(*args, **kwargs):
                result = wrapped(*args, **kwargs)
                require(result['policy'] == 'dbmol-prospective-endpoint-v1', 'Unexpected original outer component report')
                result['policy'] = 'dbmol-mixed-precision-endpoint-v1'
                result['policy_by_objective'] = deepcopy(POLICIES['dbmol'])
                directory = kwargs.get('directory', args[3] if len(args) > 3 else None)
                require(directory is not None, 'Original component report directory missing')
                (Path(directory) / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf8')
                return result
            return mixed

    driver.EndpointHooks = PrecisionEndpointHooks
    try: yield PrecisionEndpointHooks
    finally: driver.EndpointHooks = original
