"""Read-only instrumentation of the actual Protenix entry checker, never updates.

Use the authenticated original producer/checker, including its whole FP64
cohort. Capture existing calls rather than adding a replacement numerical gate.
Stop immediately after entry qualification if it returns, before any pilot.
"""
import argparse
import ast
from contextlib import contextmanager, nullcontext
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
REPEATS = 16
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


def study_path(plan, output):
    """Use the original driver's payload/study layout inside exact own scopes."""
    root, output = Path(plan['root']), Path(output)
    allowed = {root / 'cpu/payload/startup' / arm for arm in ('plain', 'chemistry')}
    allowed.update(root / ('gpu-' + arm) / 'payload/replay' / arm for arm in ('plain', 'chemistry'))
    require(root.is_absolute() and output in allowed and '..' not in output.parts,
            'Exact owned diagnostic startup/replay output scope required')
    return output / 'payload' / 'study'


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


def repeat_first(evaluate):
    """Predeclared diagnostics only: keep the first result, never select a pass."""
    require(REPEATS == 16, 'Exactly16 predeclared diagnostic replicas required')
    first = None
    for index in range(REPEATS):
        result = evaluate()
        if index == 0:
            first = result
    return first


@contextmanager
def repeated_measurements(precision, output):
    """Measure repeated existing callbacks with unchanged graph/key/runtime.

    This intentionally adds diagnostic evaluations; none is an optimizer step.
    All replicas pass through the existing raw capture. The original check uses
    replica0 even if subsequent replicas differ, and unexpected errors propagate.
    """
    output.mkdir()
    original = precision._measurement
    groups = []
    def repeated(callbacks, values):
        replicas = []
        def evaluate():
            result = original(callbacks, values)
            replicas.append(result)
            return result
        first = repeat_first(evaluate)
        row = {'native_measurement_index': len(groups), 'replicas': replicas,
               'replicas_requested': REPEATS, 'replicas_completed': len(replicas),
               'original_gate_uses_replica': 0, 'selected_for_agreement': False,
               'optimizer_updates': 0, 'can_admit_new_child': False}
        write(output / ('group-' + str(len(groups)).zfill(3) + '.json'), row)
        groups.append(row)
        return first
    precision._measurement = repeated
    try:
        yield groups
    finally:
        precision._measurement = original


def plain(plan, output, *, startup=False):
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
    study = study_path(plan, output)
    old_argv = sys.argv[:]
    sys.argv = [str(current / 'scripts/continuation_driver.py'), '--plan', str(current / 'plan.json'),
        '--plan-sha256', original['_sha256'], '--manifest', original['_manifest'], '--arm', 'protenix',
        '--phase', 'validate', '--offset', '20000', '--output', str(study),
        '--snapshot', arm['parent_snapshot']['path'], '--snapshot-sha256', arm['parent_snapshot']['sha256']]
    if startup:
        sys.argv.append('--startup-only')
    try:
        capture = nullcontext([]) if startup else captured_measurements(precision, output / 'measurements')
        repeats = nullcontext([]) if startup else repeated_measurements(precision, output / 'repeat-groups')
        with capture as records, repeats:
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
                    if startup:
                        raise
                    require(expected_observation(error), 'New diagnostic execution error: ' + repr(error))
                    return records, str(error)
                if startup:
                    return read(study.parent / 'endpoint-startup.json')
    finally:
        sys.argv = old_argv
    raise ValueError('Actual checker did not reach its bounded diagnostic stop')


def chemistry(plan, output, *, startup=False):
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
    study = study_path(plan, output)
    study.parent.mkdir(exist_ok=False)
    capture = nullcontext([]) if startup else captured_measurements(precision, output / 'measurements')
    repeats = nullcontext([]) if startup else repeated_measurements(precision, output / 'repeat-groups')
    with capture as records, repeats:
        try:
            driver.execute(original, 'protenix', 'validate', study, startup=startup)
        except BaseException as error:
            if startup:
                raise
            require(expected_observation(error), 'New diagnostic execution error: ' + repr(error))
            return records, str(error)
        if startup:
            return read(study.parent / 'endpoint-startup.json')
    raise ValueError('Actual chemistry checker did not reach its bounded diagnostic stop')


def premodel(plan, output):
    """Execute both original argparse/metadata sentinels on allocated CPU."""
    require(os.environ.get('LSB_QUEUE') == 'hpc', 'Actual allocated CPU startup checks required')
    output.mkdir()
    rows = []
    for kind, action in (('plain', plain), ('chemistry', chemistry)):
        child = output / kind
        child.mkdir()
        value = action(plan, child, startup=True)
        require(value['status'] == 'passed_metadata_only' and value['numerical_setup_entered'] is False
                and value['updates'] == 0, 'Actual original pre-model startup sentinel failed')
        rows.append({'arm': kind, 'startup': value})
    references = []
    for kind in ('plain', 'chemistry'):
        source = Path(plan['capacity_source' if kind == 'plain' else 'chemistry_source']['path'])
        filename = 'protenix_reference_worker.py' if kind == 'plain' else 'chemistry_reference_worker.py'
        path = source / 'scripts' / filename
        expected = read(source / 'source_manifest.json')['files']['scripts/' + filename]
        require(sha(path) == expected, 'Original reference worker source changed')
        worker = module(path, '_actual_reference_metadata_' + kind)
        slots = native_reference_slots(path.read_text(), kind)
        metadata = worker.validate_metadata(source / 'plan.json', sha(source / 'plan.json'))
        parent = output / ('reference-' + kind) / 'reference-session'
        parent.mkdir(parents=True)
        native_output = parent / 'worker'
        require(worker.validate_output(native_output) == native_output,
                'Actual original reference output ownership rejected')
        owner = read(source / 'plan.json')
        precision = module(Path(plan['capacity_source']['path']) / 'scripts/precision_continuation_hooks.py',
                           '_diagnostic_reference_premodel_expansion_' + kind)
        command = precision.expanded_command(owner['reference_configuration']['arms']['protenix'],
                                             sha(source / 'plan.json'))
        require(command[:3] == [sys.executable, '-u', str(path)], 'Actual original reference CLI differs')
        old_output = worker.ROOT / 'cpu/payload/protenix-reference-startup.json'
        before = old_output.read_bytes() if old_output.exists() else None
        new_output = parent.parent / 'reference-startup.json'
        original_write = worker._write
        def isolated_receipt(destination, value):
            require(Path(destination) == old_output and value.get('status') == 'passed_metadata_only'
                    and value.get('numerical_setup_entered') is False and value.get('optimizer_updates') == 0,
                    'Only the original expected pre-model metadata receipt may be redirected')
            write(new_output, value)
        command += ['--startup-only', '--startup-output', str(old_output)]
        worker._write = isolated_receipt
        try:
            worker.main(command[3:])
        finally:
            worker._write = original_write
        require((old_output.read_bytes() if old_output.exists() else None) == before,
                'Historical reference startup receipt changed')
        startup = read(new_output)
        require(startup['job_id'] == os.environ['LSB_JOBID'], 'Actual reference CPU startup identity differs')
        references.append({'arm': kind, 'native_GPU_CPU_slots': slots,
            'worker': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': expected},
            'native_output': str(native_output), 'owner_plan_sha256': sha(source / 'plan.json'),
            'actual_reference_metadata_passed': bool(metadata), 'model_constructed': False,
            'startup_command': command, 'actual_worker_startup': startup,
            'single_startup_metadata_receipt_redirected_before_write': True,
            'historical_receipt_preserved': True})
    write(output / 'assessment.json', {'status': 'passed_actual_original_pre_model_startups',
        'startups': rows, 'optimizer_updates': 0, 'model_constructed': False,
        'can_admit_new_child': False, 'reference_metadata': references})


def native_reference_slots(source, kind):
    """Read the actual immutable worker's closed native allocation predicate."""
    require(kind in ('plain', 'chemistry'), 'Unknown reference kind')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'allocated_environment']
    require(len(functions) == 1, 'One original allocated_environment function required')
    nodes = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Compare)
             and isinstance(n.left, ast.Name) and n.left.id == 'declared']
    expected = ast.parse('declared == "4"' if kind == 'plain'
                         else 'declared == ("4" if startup else "8")', mode='eval').body
    require(len(nodes) == 1 and ast.dump(nodes[0]) == ast.dump(expected),
            'Original reference native slot predicate differs')
    return 4 if kind == 'plain' else 8


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


def gpu(plan, output, kind):
    output.mkdir()
    chemistry_plan = read(Path(plan['chemistry_source']['path']) / 'plan.json')
    adapter = Path(chemistry_plan['arms']['protenix']['adapter_source']['path'])
    sys.path.insert(0, str(adapter / 'scripts'))
    pilot = module(adapter / 'scripts/full_model_pilot.py', '_owned_diagnostic_process_cleanup')
    pilot.OUTPUT_CAP = CAP
    rows = []
    for kind in (kind,):
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', plan['_path'], '--pin',
                   plan['_pin'], '--mode', kind, '--output', str(output / kind)]
        write(output / (kind + '-command.json'), {'argv': command, 'optimizer_updates': 0})
        with (output / (kind + '.log')).open('x') as stream:
            with pilot._parent_sigterm_cleanup():
                execution = pilot.run_isolated_child(command, Path(__file__).resolve().parents[1],
                    dict(os.environ), stream, 2500, Path(plan['root']))
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
    parser.add_argument('--mode', choices=('cpu', 'gpu-plain', 'gpu-chemistry', 'startup', 'plain', 'chemistry'), required=True)
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
    elif args.mode == 'startup':
        premodel(plan, args.output)
    elif args.mode.startswith('gpu-'):
        gpu(plan, args.output, args.mode.removeprefix('gpu-'))
    else:
        child(plan, args.output, args.mode)


if __name__ == '__main__':
    main()
