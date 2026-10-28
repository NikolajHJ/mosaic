"""Allocated independent affinity reference at the untouched saved7500 endpoint."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import types

BASE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01')
ROOT = BASE / 'dbmol-endpoint-recovery-13'
CAP = 448 * 1024**2
POLICY_SOURCE = BASE / 'protenix-endpoint-fix-release-07/source/scripts/protenix_precision_admission.py'
POLICY_SOURCE_SHA256 = 'ae549f19a8af11453b0bb01154bde61cd63f4fa8ad1676759c7427e9049ddf68'
POLICY_PARENT_VERSION = 'protenix-separate-precision-admission-v1'
POLICY_PARENT_EPSILONS = (.003, .001, .0003, .0001, .00003)
FINER_POLICY_VERSION = 'dbmol-separate-precision-finer-admission-v1'
FINER_EPSILONS = (.003, .001, .0003, .0001, .00003, .00001, .000003, .000001)


def finer_policy_spec():
    """Closed, predeclared policy; this function performs no numerical work."""
    return {'version': FINER_POLICY_VERSION, 'epsilons': list(FINER_EPSILONS),
            'original_epsilons': [.01, .003, .001], 'relative_tolerance': .15,
            'locality_tolerance': .15, 'required_adjacent_passes': 2,
            'selection': 'finest_geometry_and_baseline_signal_pair_before_perturbed_callbacks',
            'after_result_fallback': False, 'original_failed_outcomes_preserved': True,
            'parent_policy_source_sha256': POLICY_SOURCE_SHA256,
            'changed_engine_fields': ['VERSION', 'EPSILONS']}


def _policy_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_finer_policy(plan):
    require(_policy_json(plan.get('precision_policy')) == _policy_json(finer_policy_spec()),
            'Closed predeclared finer precision policy differs')


def _clone_policy_source(source, filename):
    """Clone authenticated code, proving only the version and ladder changed.

    The immutable parent source is never written or monkey-patched. All original
    geometry, signal, FD, parity, locality, nonzero and frozen checks execute in
    the new private namespace, including the original two-adjacent-pair selector.
    """
    require(hashlib.sha256(source).hexdigest() == POLICY_SOURCE_SHA256,
            'Original precision policy source bytes differ')
    original = ast.parse(source, filename=filename)
    changed = deepcopy(original)
    replacements = {'VERSION': FINER_POLICY_VERSION, 'EPSILONS': FINER_EPSILONS}
    expected = {'VERSION': POLICY_PARENT_VERSION, 'EPSILONS': POLICY_PARENT_EPSILONS}
    old = {}; counts = Counter()
    for node in changed.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in replacements:
                require(ast.literal_eval(node.value) == expected[name], 'Parent policy declaration differs: ' + name)
                counts[name] += 1; old[name] = deepcopy(node.value)
                node.value = ast.parse(repr(replacements[name]), mode='eval').body
    require(counts == Counter({'VERSION': 1, 'EPSILONS': 1}), 'Unique policy declarations required')
    restored = deepcopy(changed)
    for node in restored.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in old: node.value = deepcopy(old[node.targets[0].id])
    require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
            'Policy transformation changed more than the declared version and epsilon tuple')
    private = types.ModuleType('_dbmol_finer_precision_private')
    private.__file__ = filename
    exec(compile(ast.fix_missing_locations(changed), filename + ':finer-private', 'exec'), private.__dict__)
    scalar = private._scalar
    def observed_scalar(value, dtype):
        # Identical DBMol transport adapter to the authenticated original wrapper:
        # the real production scalar is FP64 while production raw gradients stay FP32.
        return scalar(value, 'float64' if dtype == 'float32' else dtype)
    private._scalar = observed_scalar
    derivation = {'status': 'authenticated_two_assignment_AST_clone',
                  'parent_path': filename, 'parent_sha256': POLICY_SOURCE_SHA256,
                  'changed_assignments': ['VERSION', 'EPSILONS'], 'AST_reversal_exact': True,
                  'private_AST_sha256': hashlib.sha256(ast.dump(changed, include_attributes=False).encode()).hexdigest(),
                  'production_scalar_transport': 'original_FP64_scalar_FP32_raw_gradients',
                  'parent_source_modified': False, 'spec': finer_policy_spec()}
    return private, derivation


class FinerPolicy:
    """Guard the predeclared private policy before and after each evaluation."""
    def __init__(self, plan, original):
        validate_finer_policy(plan)
        require(Path(original.__file__) == POLICY_SOURCE and not POLICY_SOURCE.is_symlink(),
                'Authenticated original precision policy location differs')
        require(original.VERSION == POLICY_PARENT_VERSION and original.EPSILONS == POLICY_PARENT_EPSILONS,
                'Original policy namespace already mutated')
        self._plan = plan; self._plan_policy = _policy_json(plan['precision_policy'])
        self._original = original
        self._module, self.derivation = _clone_policy_source(POLICY_SOURCE.read_bytes(), str(POLICY_SOURCE))
        self._constants = {name: deepcopy(value) for name, value in vars(self._module).items()
                           if name.isupper() or name in ('_FIELDS', '_PRECISION')}
        self._functions = {name: (value, value.__code__) for name, value in vars(self._module).items()
                           if isinstance(value, types.FunctionType)}

    def _guard(self):
        validate_finer_policy(self._plan)
        require(_policy_json(self._plan['precision_policy']) == self._plan_policy,
                'Finer policy plan mutated after construction')
        require(self._original.VERSION == POLICY_PARENT_VERSION
                and self._original.EPSILONS == POLICY_PARENT_EPSILONS,
                'Original policy namespace changed')
        require(hashlib.sha256(POLICY_SOURCE.read_bytes()).hexdigest() == POLICY_SOURCE_SHA256,
                'Original policy file changed during evaluation')
        for name, value in self._constants.items():
            current = getattr(self._module, name)
            require(type(current) is type(value) and _policy_json(current) == _policy_json(value),
                    'Private finer policy constant mutated: ' + name)
        for name, (function, code) in self._functions.items():
            require(getattr(self._module, name) is function and function.__code__ is code,
                    'Private finer policy function mutated: ' + name)

    def evaluate(self, *args, **kwargs):
        self._guard()
        result = self._module.evaluate(*args, **kwargs)
        self._guard()
        require(result['version'] == FINER_POLICY_VERSION
                and result['policy']['epsilons'] == list(FINER_EPSILONS),
                'Finer evaluation receipt identifies a different policy')
        return result


def require(value, message):
    if not value: raise ValueError(message)


def binding(path):
    p = Path(path); require(p.is_file() and not p.is_symlink(), 'Regular authenticated input required')
    with p.open('rb') as f: sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def verify(plan, output, startup=False):
    require(plan.get('version') == 'dbmol-separate-affinity-reference-finer-v1'
            and plan.get('root') == str(ROOT) and plan.get('optimizer_updates') == 0
            and plan.get('original_results_changed') is False, 'Closed recovery plan identity differs')
    validate_finer_policy(plan)
    expected = ROOT / ('cpu/payload/startup' if startup else 'gpu/payload/native')
    require(Path(output) == expected, 'Output escapes new owned scope')
    for row in plan['protected_inputs']: require(binding(row['path']) == row, 'Protected input changed')
    require(plan['global_step'] == 7500 and plan['target_global'] == 10000 and plan['rate'] == .0025,
            'Original DBMol scientific settings differ')


class ReferenceClient:
    def __init__(self, command, log):
        self.log = Path(log).open('x'); self.sequence = 0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        self.identity = self.read()
        require(self.identity.get('status') == 'ready_full_double_reference', 'Full FP64 reference did not initialize')
    def read(self):
        with selectors.DefaultSelector() as s:
            s.register(self.process.stdout, selectors.EVENT_READ)
            require(bool(s.select(1200)), 'Reference response timed out')
        line = self.process.stdout.readline()
        require(bool(line), 'Reference worker exited; raw log preserved')
        return json.loads(line)
    def evaluate(self, values, gradient):
        self.sequence += 1
        digest = hashlib.sha256(b''.join(values[f].tobytes() for f in ('X', 'E_upper'))).hexdigest()
        request = {'id': self.sequence, 'gradient': gradient, 'graph_sha256': digest,
                   **{f: v.tolist() for f, v in values.items()}}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n'); self.process.stdin.flush()
        result = self.read()
        require(result.get('id') == self.sequence and result.get('graph_sha256') == digest,
                'Reference response changed point/identity')
        return result
    def finish(self):
        self.process.stdin.write('{"action":"close"}\n'); self.process.stdin.flush()
        result = self.read(); self.process.wait(timeout=30)
        require(result.get('status') == 'closed' and self.process.returncode == 0,
                'Reference frozen/audit handshake failed')
        return result
    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try: self.process.wait(timeout=20)
            except subprocess.TimeoutExpired: self.process.kill(); self.process.wait(timeout=20)
        self.log.close()


def run(plan, output):
    import numpy as np
    import dbmol_reference as reference
    science = Path(plan['science_source']['path'])
    sys.path[:0] = [str(science / 'src'), str(science / 'scripts')]
    os.environ['PYTHONPATH'] = os.pathsep.join((str(science / 'src'), str(science / 'scripts'),
        str(Path(__file__).parent), os.environ.get('PYTHONPATH', '')))
    from dbmol_bridge import DBMolWorker
    from dbmol_protocol import verify_identity, frozen as verify_frozen
    with np.load(plan['boundary']['path'], allow_pickle=False) as z:
        values = {f: z[f].copy() for f in ('X', 'E_upper')}
    original = json.loads(Path(plan['original_admission']['path']).read_bytes())
    for f in values:
        raw = __import__('base64').b64decode(original['plan']['baseline'][f]['base64'])
        require(values[f].dtype == np.float32 and values[f].tobytes() == raw,
                'Saved snapshot and original failed terminal input differ')
    before = {b['path']: b for b in plan['protected_inputs']}
    production = None; ref = None; points = []; failed = None
    try:
        production = DBMolWorker(plan['production_argv'], stderr=output / 'production.log')
        verify_identity(production.identity)
        production_identity = production.identity.copy()
        write(output / 'production-identity.json', production_identity)
        base_records = []
        for number in range(3):
            primal, _, _ = production.evaluate(values['X'], values['E_upper'], gradient=False, objective='dbmol_affinity_loss')
            ad, gx, ge = production.evaluate(values['X'], values['E_upper'], gradient=True, objective='dbmol_affinity_loss')
            row = {'primal': np.float64(primal[0]), 'value': np.float64(ad[0]), 'gradients': {'X': gx, 'E_upper': ge}}
            path = output / ('production-' + str(number) + '.npz')
            np.savez_compressed(path, **values, primal=row['primal'], ad_value=row['value'], gradient_X=gx, gradient_E_upper=ge)
            base_records.append(row); points.append(binding(path))
        after = production.request('verify_identity'); verify_frozen(after, production_identity)
        write(output / 'production-frozen-after.json', after)
        production.close(); production = None
        ref = ReferenceClient(plan['reference_argv'], output / 'reference.log')
        write(output / 'reference-identity.json', ref.identity)
        policy = FinerPolicy(plan, reference.original_metadata()); cache = {}; cursor = Counter()
        write(output / 'precision-policy-derivation.json', policy.derivation)
        def native(values64):
            require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) < CAP - 8 * 1024**2,
                    'Reference persistent output cap exhausted')
            primal = ref.evaluate(values64, False); ad = ref.evaluate(values64, True)
            gradients = {f: np.asarray(ad['gradients'][f], dtype=np.float64) for f in values64}
            row = {'primal': np.float64(primal['value']), 'value': np.float64(ad['value']), 'gradients': gradients}
            path = output / ('reference-' + str(len(points)).zfill(4) + '.npz')
            np.savez_compressed(path, **values64, primal=row['primal'], ad_value=row['value'],
                                **{'gradient_' + f: v for f, v in gradients.items()})
            points.append(binding(path)); return row
        baseline = {f: v.astype(np.float64) for f, v in values.items()}
        def reference_call(v, kind):
            digest = hashlib.sha256(b''.join(v[f].tobytes() for f in ('X', 'E_upper'))).hexdigest()
            if digest not in cache:
                repeats = 3 if all(np.array_equal(v[f], baseline[f]) for f in v) else 1
                cache[digest] = [native(v) for _ in range(repeats)]
            rows = cache[digest]; n = cursor[(digest, kind)]; cursor[(digest, kind)] += 1
            row = rows[min(n, len(rows) - 1)]
            return row['primal'] if kind == 'primal' else (row['value'], row['gradients'])
        def production_call(v, kind):
            for f in v: np.testing.assert_array_equal(v[f], values[f])
            n = cursor[('production', kind)]; cursor[('production', kind)] += 1
            row = base_records[n % 3]
            return row['primal'] if kind == 'primal' else (row['value'], row['gradients'])
        precision = {'full_model_float64': True, 'parameters_float64': True, 'native_features_float64': True,
                     'native_operations_float64': True, 'final_reduction_only': False}
        def evaluate(label, factor=1.):
            cursor.clear()
            def ad32(v):
                scalar, gradients = production_call(v, 'ad')
                return scalar, {f: (g * np.float32(factor)).astype(np.float32) for f, g in gradients.items()}
            result = policy.evaluate(values, lambda v: production_call(v, 'primal'), ad32,
                lambda v: reference_call(v, 'primal'), lambda v: reference_call(v, 'ad'),
                lambda: ref.identity['frozen'], original_directions=plan['original_directions'],
                reference_precision=precision, original_observations=plan['original_admission'],
                seal_callback=lambda p: write(output / (label + '-selection.json'), p))
            write(output / (label + '-admission.json'), result); return result
        result = evaluate('native')
        negatives = []
        for label, factor in (('zero', 0.), ('reversed', -1.), ('scaled', 1.5)):
            n = len(points); control = evaluate(label, factor)
            require(control['can_admit_new_child'] is False and len(points) == n,
                    'Wrong production gradient admitted or new native points added')
            negatives.append({'label': label, 'rejected': True, 'native_calls_added': 0})
        closing = ref.finish(); require(closing['frozen'] == ref.identity['frozen'], 'Reference frozen identity changed')
        write(output / 'reference-operation-audit.json', closing)
        write(output / 'oracle-points.json', {'points': points, 'negative_controls': negatives,
              'endpoint_cache_replays_are_independent_repeats': False})
        require(result['can_admit_new_child'] is True, 'Independent full-FP64 affinity measurement did not qualify')
    except BaseException as error:
        failed = error
    finally:
        if production is not None: production.close()
        if ref is not None: ref.close()
        after = {b['path']: binding(b['path']) for b in plan['protected_inputs']}
        preserved = before == after
        write(output / 'preservation.json', {'status': 'passed' if preserved else 'failed', 'before': before, 'after': after})
        write(output / 'assessment.json', {'status': 'failed' if failed or not preserved else 'passed_affinity_reference_only',
              'error': None if failed is None else repr(failed), 'original_results_changed': not preserved,
              'optimizer_updates': 0, 'can_submit_continuation': False,
              'remaining_admission': 'all_components_raw_null_parity_pilot_runtime_startup_full_storage'})
    if failed: raise failed
    require(preserved, 'Saved original results changed')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', required=True)
    p.add_argument('--plan-sha256', required=True); p.add_argument('--output', required=True)
    p.add_argument('--startup-only', action='store_true'); a = p.parse_args()
    require(str(os.environ.get('LSB_JOBID', '')).isdigit(), 'Allocated execution required')
    require(binding(a.plan)['sha256'] == a.plan_sha256, 'Plan bytes differ')
    plan = json.loads(Path(a.plan).read_bytes()); output = Path(a.output); verify(plan, output, a.startup_only)
    output.mkdir(exist_ok=False)
    if a.startup_only:
        write(output / 'assessment.json', {'status': 'passed_metadata_only', 'optimizer_updates': 0,
              'numerical_setup_entered': False, 'plan_sha256': a.plan_sha256})
        return
    require(os.environ.get('LSB_QUEUE') == 'gpul40s', 'Real GPU allocation required')
    def expired(*args): raise TimeoutError('Bounded native reference deadline exhausted')
    signal.signal(signal.SIGALRM, expired); signal.alarm(5300)
    try: run(plan, output)
    finally: signal.alarm(0)


if __name__ == '__main__': main()
