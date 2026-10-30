"""Read-only instrumentation of the actual Protenix entry checker, never updates.

Use the authenticated original producer/checker, including its whole FP64
cohort. Capture existing calls rather than adding a replacement numerical gate.
Stop immediately after entry qualification if it returns, before any pilot.
"""
import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

CAP = 128 * 1024**2
STOP = 'READ_ONLY_DIAGNOSTIC_NATIVE_BOUNDARY_REACHED'
EXPECTED = {
    'Live production changed while the FP64 GPU process coexisted',
    'Exact production scalar/gradient/frozen/input replay changed after reference exit',
    'Independent precision admission failed; original observations preserved',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def numpy():
    require(os.environ.get('LSB_JOBID', '').isdigit(), 'Allocated numerical execution required')
    import numpy as np
    return np


def bounded(path):
    total = 0
    for item in Path(path).rglob('*'):
        require(not item.is_symlink() and (item.is_file() or item.is_dir()), 'Special diagnostic output')
        if item.is_file():
            total += item.stat().st_size
    require(total < CAP, 'Complete funded diagnostic cap exhausted')
    return total


def stopped_tree(tree):
    """Add one post-native sentinel and independently block optimizer calls."""
    tree = deepcopy(tree)
    changed = 0
    forbidden = ('run_chemical_updates', 'run_updates', 'update_parameters')
    class Stop(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal changed
            node = self.generic_visit(node)
            if len(node.targets) == 1:
                target = node.targets[0]
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id == 'report' and isinstance(target.slice, ast.Constant)
                        and target.slice.value == 'native_gradient_checks'):
                    changed += 1
                    return [node, ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError', ctx=ast.Load()),
                        args=[ast.Constant(STOP)], keywords=[]), cause=None)]
            return node
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id in forbidden:
                return ast.copy_location(ast.Call(func=ast.Name(id='_diagnostic_forbidden_update', ctx=ast.Load()),
                                                  args=[], keywords=[]), node)
            return node
    tree = Stop().visit(tree)
    require(changed == 1, 'Exactly one authenticated native boundary assignment required')
    return ast.fix_missing_locations(tree)


def forbidden_update():
    raise RuntimeError('Optimizer execution is forbidden in this diagnostic')


def expected_observation(error):
    return str(error) == STOP or type(error) is ValueError and str(error) in EXPECTED


@contextmanager
def captured_measurements(precision, output):
    """Save the original call's arrays; preserve its exact digest comparison."""
    np = numpy()
    output.mkdir()
    original = precision._measurement
    records = []
    first = None
    def measurement(callbacks, values):
        nonlocal first
        captured = {}
        primal, autodiff, frozen = callbacks
        def p(point):
            value = primal(point)
            captured['primal'] = np.asarray(value).copy()
            return value
        def ad(point):
            scalar, gradients = autodiff(point)
            captured['autodiff_scalar'] = np.asarray(scalar).copy()
            captured.update({'gradient_' + k: np.asarray(v).copy() for k, v in gradients.items()})
            return scalar, gradients
        result = original((p, ad, frozen), values)
        if first is None:
            first = {k: v.copy() for k, v in captured.items()}
        difference = {}
        for key, value in captured.items():
            a = first[key]
            require(a.shape == value.shape and a.dtype == value.dtype, 'Original measurement schema changed')
            difference[key] = {'exact_equal': bool(np.array_equal(a, value)),
                'maximum_absolute_difference': float(np.max(np.abs(a.astype(np.float64) - value.astype(np.float64))))}
        path = output / ('measurement-' + str(len(records)).zfill(3) + '.npz')
        np.savez_compressed(path, **values, **captured)
        records.append({'index': len(records), 'path': str(path), 'bytes': path.stat().st_size,
                        'sha256': sha(path), 'measurement': result, 'versus_first': difference})
        write(output / ('measurement-' + str(len(records) - 1).zfill(3) + '.json'), records[-1])
        bounded(output.parent)
        return result
    precision._measurement = measurement
    try:
        yield records
    finally:
        precision._measurement = original


def plain(plan, output):
    current = Path(plan['capacity_source']['path'])
    sys.path.insert(0, str(current / 'scripts'))
    adapter = module(current / 'scripts/precision_continuation_driver.py', '_diagnostic_actual_adapter')
    original = read(current / 'plan.json')
    original['_sha256'] = sha(current / 'plan.json')
    original['_manifest'] = plan['capacity_source']['manifest_sha256']
    adapter.verify_owned_source(current, original['_manifest'])
    base = adapter.load_base(original)
    precision = module(current / 'scripts/precision_continuation_hooks.py', '_diagnostic_actual_precision')
    arm = next(row for row in original['arms'] if row['id'] == 'protenix')
    old_argv = sys.argv[:]
    sys.argv = [str(current / 'scripts/continuation_driver.py'), '--plan', str(current / 'plan.json'),
        '--plan-sha256', original['_sha256'], '--manifest', original['_manifest'], '--arm', 'protenix',
        '--phase', 'validate', '--offset', '20000', '--output', str(output / 'study'),
        '--snapshot', arm['parent_snapshot']['path'], '--snapshot-sha256', arm['parent_snapshot']['sha256']]
    try:
        with captured_measurements(precision, output / 'measurements') as records:
            with adapter.adapted_driver(base, original, precision):
                def run_path(filename, *, run_name):
                    require(Path(filename) == Path(arm['source']['path']) / 'examples/brd4_25_atoms.py'
                            and run_name == 'prospective_original_driver', 'Unexpected original entrypoint')
                    tree = adapter.short_segment_driver_tree(Path(filename).read_bytes(), filename, 'protenix')
                    namespace = {'__name__': run_name, '__file__': str(filename),
                                 '_diagnostic_forbidden_update': forbidden_update}
                    exec(compile(stopped_tree(tree), str(filename), 'exec'), namespace)
                    return namespace
                base.runpy = SimpleNamespace(run_path=run_path)
                try:
                    base.main()
                except BaseException as error:
                    require(expected_observation(error), 'New diagnostic execution error: ' + repr(error))
                    return records, str(error)
    finally:
        sys.argv = old_argv
    raise ValueError('Actual checker did not reach its bounded diagnostic stop')


def chemistry(plan, output):
    source = Path(plan['chemistry_source']['path'])
    sys.path.insert(0, str(source / 'scripts'))
    driver = module(source / 'scripts/chemistry_continuation_driver.py', '_diagnostic_actual_chemistry')
    original = driver.authenticated_plan(source / 'plan.json', sha(source / 'plan.json'))
    base, precision = driver.modules()
    driver.modules = lambda: (base, precision)
    old_tree = driver.actual_coefficient_tree
    driver.actual_coefficient_tree = lambda tree: stopped_tree(old_tree(tree))
    # Unknown update names are also unreachable after the mandatory sentinel.
    # A removed known update name would fail before evaluation of its arguments.
    with captured_measurements(precision, output / 'measurements') as records:
        try:
            driver.execute(original, 'protenix', 'validate', output / 'study')
        except BaseException as error:
            require(expected_observation(error), 'New diagnostic execution error: ' + repr(error))
            return records, str(error)
    raise ValueError('Actual chemistry checker did not reach its bounded diagnostic stop')


def child(plan, output, kind):
    output.mkdir()
    result = {'status': 'failed', 'optimizer_updates': 0, 'can_admit_new_child': False,
              'can_certify_or_restart_original_study': False, 'arm': kind}
    try:
        records, observation = (plain if kind == 'plain' else chemistry)(plan, output)
        require(records, 'Actual entry checker produced no measurement evidence')
        result.update(status='actual_checker_observed_original_failures_preserved',
                      observed_boundary_outcome=observation, measurements=records,
                      all_measurements_exactly_equal=all(
                          d['exact_equal'] for r in records for d in r['versus_first'].values()))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        result['output_bytes'] = bounded(output)
        write(output / 'assessment.json', result)


def gpu(plan, output):
    output.mkdir()
    chemistry_plan = read(Path(plan['chemistry_source']['path']) / 'plan.json')
    adapter = Path(chemistry_plan['arms']['protenix']['adapter_source']['path'])
    sys.path.insert(0, str(adapter / 'scripts'))
    pilot = module(adapter / 'scripts/full_model_pilot.py', '_owned_diagnostic_process_cleanup')
    pilot.OUTPUT_CAP = CAP
    rows = []
    for kind in ('plain', 'chemistry'):
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', plan['_path'], '--pin',
                   plan['_pin'], '--mode', kind, '--output', str(output / kind)]
        write(output / (kind + '-command.json'), {'argv': command, 'optimizer_updates': 0})
        with (output / (kind + '.log')).open('x') as stream:
            with pilot._parent_sigterm_cleanup():
                execution = pilot.run_isolated_child(command, Path(__file__).resolve().parents[1],
                    dict(os.environ), stream, 2500, output)
            write(output / (kind + '-execution.json'), execution)
        rows.append(read(output / kind / 'assessment.json'))
        bounded(output)
    write(output / 'assessment.json', {'status': 'diagnostics_completed_original_failures_preserved',
        'observations': rows, 'optimizer_updates': 0, 'can_admit_new_child': False,
        'can_certify_or_restart_original_study': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu', 'gpu', 'plain', 'chemistry'), required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.pin and os.environ.get('LSB_JOBID', '').isdigit(),
            'Actual allocation and authenticated plan required')
    plan = read(args.plan)
    plan.update(_path=args.plan, _pin=args.pin)
    if args.mode == 'cpu':
        args.output.mkdir()
        report = read(plan['boltz_logit_failure']['path'])
        require(report['status'] == 'failed' and report['frozen_unchanged'] is True,
                'Original failed frozen Boltz cohort required')
        rows = [{k: row[k] for k in ('field', 'row', 'source', 'destination', 'analytic',
                 'selected_local_pair', 'pair_passed', 'finer_resolved_disagreements')} |
                {'probes': [{k: p[k] for k in ('epsilon', 'finite_difference', 'analytic', 'relative_error',
                            'eligible_before_FD', 'outcome')} for p in row['probes']]}
                for row in report['directions']]
        write(args.output / 'assessment.json', {'status': 'diagnostics_completed_original_failures_preserved',
            'observations': rows, 'original_error': report['error'], 'optimizer_updates': 0,
            'can_admit_new_child': False, 'can_certify_or_restart_original_study': False})
    elif args.mode == 'gpu':
        gpu(plan, args.output)
    else:
        child(plan, args.output, args.mode)


if __name__ == '__main__':
    main()
