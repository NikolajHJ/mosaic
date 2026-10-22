"""Standalone allocated diagnostics of two preserved failed native endpoint checks.

This never performs an optimizer update, repairs a graph, prepares a new pose or
changes a saved study.  Finer scales and alternative directions are explicitly
diagnostic evidence; they cannot replace the original failed scientific gates.
Scientific imports are lazy and resolve only from the authenticated source610.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys


VERSION = 'jq1-lrboost-endpoint-diagnostics-v1'
SOURCE = Path('/work3/s234818/mosaic/releases/20261021-brd4-jq1-25-fixed-lrboost-01/source')
SOURCE_MANIFEST = '9605a5295b9fe8dfe29f15410b3a029038ea8a18feb16907f043abd5be13c323'
JOBS = {'protenix': '29779177', 'boltz2': '29779178'}
ORIGINAL_EPSILONS = (0.01, 0.003, 0.001)
FINE_EPSILONS = (0.0003, 0.0001, 0.00003, 0.00001)
NEARBY_CHOICES = (9500, 9900, 9999)
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024**2


def diagnostic_output_path(source, arm):
    source = Path(source)
    require(arm in JOBS and source.name == 'source'
            and source.parent.parent == Path('/work3/s234818/mosaic/releases')
            and re.fullmatch(r'20261022-brd4-jq1-lrboost-analysis-[0-9]{2}', source.parent.name),
            'Diagnostics require a separate declared analysis release/output root')
    return Path('/work3/s234818/mosaic/runs') / source.parent.name / ('diagnostic-' + arm) / 'payload'


def output_budget(path, maximum):
    require(type(maximum) is int and 0 < maximum <= DEFAULT_MAX_OUTPUT_BYTES,
            'Diagnostic persistent allowance must be positive and at most declared 256 MiB')
    files = list(Path(path).rglob('*'))
    require(not any(p.is_symlink() for p in files), 'Diagnostic output contains an unexpected symlink')
    size = sum(p.stat().st_size for p in files if p.is_file())
    require(size <= maximum, 'Diagnostic persistent output exceeded its declared allowance')
    return size


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def hashed(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def selected_steps(nearby=None):
    require(nearby is None or type(nearby) is int and nearby in NEARBY_CHOICES,
            'At most one predeclared nearby saved control is permitted')
    return [0, *([] if nearby is None else [nearby]), 10000]


def array_sha(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(value.tobytes())
    return digest.hexdigest()


def original_direction(gradient, *, arm):
    import numpy as np
    values = np.asarray(gradient, dtype=np.float64 if arm == 'protenix' else np.float32)
    require(values.ndim == 2 and np.isfinite(values).all(), 'Malformed raw native gradient')
    row = int(np.argmax(values.max(-1) - values.min(-1)))
    return {'row': row, 'source': int(values[row].argmin()), 'target': int(values[row].argmax())}


def original_report_direction(record, field):
    rows = record.get('directions', record.get('rows', []))
    row = next((r for r in rows if r['field'] == field), None)
    return None if row is None else {'row': row['row'], 'source': row['source'],
        'target': row.get('destination', row.get('target'))}


def feasible_alternative(values, gradient, original):
    """One deterministic contrast with both original ±0.01 endpoints feasible."""
    import numpy as np
    candidates = []
    for row in range(values.shape[0]):
        for source in range(values.shape[1]):
            for target in range(source + 1, values.shape[1]):
                if min(float(values[row, source]), float(values[row, target])) <= max(ORIGINAL_EPSILONS):
                    continue
                pair = {'row': row, 'source': source, 'target': target}
                if pair == original or pair == (original | {'source': original['target'], 'target': original['source']}):
                    continue
                difference = float(gradient[row, target]) - float(gradient[row, source])
                if difference < 0:
                    pair = pair | {'source': target, 'target': source}
                candidates.append((abs(difference), -row, -source, -target, pair))
    return None if not candidates else max(candidates, key=lambda x: x[:4])[-1]


def raw_endpoints(values, direction, epsilon):
    """Create diagnostic FP32 points only; never clip, renormalize or project."""
    import numpy as np
    require(values.dtype == np.float32 and values.ndim == 2 and 0 < epsilon <= .01,
            'Raw probe requires original FP32 probabilities and bounded scale')
    row, source, target = (direction[k] for k in ('row', 'source', 'target'))
    require(0 <= row < values.shape[0] and 0 <= source < values.shape[1]
            and 0 <= target < values.shape[1] and source != target, 'Invalid categorical contrast')
    points, observations = [], []
    for sign in (-1, 1):
        point = values.copy()
        point[row, source] = np.float32(point[row, source] - np.float32(sign * epsilon))
        point[row, target] = np.float32(point[row, target] + np.float32(sign * epsilon))
        delta = point - values
        both_changed = bool(point[row, source] != values[row, source]
                            and point[row, target] != values[row, target])
        simplex = bool(np.isfinite(point).all() and point.min() >= 0 and point.max() <= 1
                       and np.allclose(point.sum(-1), 1, atol=1e-6, rtol=0))
        interior = bool(point[row, source] > 0 and point[row, target] > 0)
        observation = {'sign': sign, 'minimum': float(point.min()), 'maximum': float(point.max()),
            'maximum_row_sum_error': float(np.max(np.abs(point.sum(-1) - 1))),
            'both_affected_entries_changed': both_changed, 'simplex_feasible': simplex,
            'affected_entries_strictly_positive': interior,
            'source_delta': float(delta[row, source]), 'target_delta': float(delta[row, target]),
            'maximum_absolute_delta': float(np.max(np.abs(delta))),
            'point_sha256': array_sha(point)}
        observations.append(observation)
        points.append(point)
    pair_margin = min(float(values[row, source]), float(values[row, target]))
    usable = epsilon < pair_margin and all(o['simplex_feasible'] and o['affected_entries_strictly_positive']
                                         and o['both_affected_entries_changed'] for o in observations)
    return points, {'epsilon': epsilon, 'original_strict_interior_precheck': epsilon < pair_margin,
        'symmetric_pair_margin': pair_margin, 'usable_diagnostic_points': usable, 'endpoints': observations}


def fd_evidence(minus, plus, epsilon, analytic, baseline):
    import numpy as np
    require(minus and plus and all(np.isfinite(x) for x in [*minus, *plus, analytic, baseline]),
            'Nonfinite independent diagnostic forwards')
    left, right = float(np.mean(minus)), float(np.mean(plus))
    fd = (right - left) / (2 * epsilon)
    scale = max(abs(fd), abs(analytic))
    error = abs(fd - analytic)
    spread = max(max(minus) - min(minus), max(plus) - min(plus))
    signal = abs(right - left)
    relative_agreement = bool(abs(analytic) > 1e-7 and fd * analytic > 0 and error <= .15 * scale)
    protenix_agreement = relative_agreement and bool(np.isclose(fd, analytic, atol=.002, rtol=.15))
    return {'minus_repeats': minus, 'plus_repeats': plus, 'finite_difference': fd,
        'analytic': analytic, 'absolute_error': error,
        'relative_error': None if scale == 0 else error / scale,
        'same_sign': bool(fd * analytic > 0),
        'agrees_diagnostic_only': relative_agreement,
        'original_protenix_scalar_criterion_diagnostic_only': protenix_agreement,
        'original_boltz_scalar_criterion_diagnostic_only': relative_agreement,
        'maximum_repeat_spread': spread, 'endpoint_difference_signal': signal,
        'spread_over_signal': None if signal == 0 else spread / signal,
        'central_second_difference': (right - 2 * baseline + left) / epsilon**2,
        'plus_one_sided_derivative': (right - baseline) / epsilon,
        'minus_one_sided_derivative': (baseline - left) / epsilon,
        'certifies_original_gradient_gate': False}


def classify_direction(probes):
    """Describe evidence without promoting a favorable diagnostic scale."""
    findings = []
    original = [p for p in probes if p['epsilon'] in ORIGINAL_EPSILONS]
    if any(not p['geometry']['original_strict_interior_precheck'] for p in original):
        findings.append('Original chosen contrast does not admit every predeclared symmetric interior scale')
    evaluated = [p for p in probes if p.get('evaluation') is not None]
    if any(p['evaluation']['endpoint_difference_signal'] <= p['evaluation']['maximum_repeat_spread'] for p in evaluated):
        findings.append('At least one FD signal is no larger than observed repeat variation; that scale is unresolved')
    original_evaluated = [p for p in evaluated if p['epsilon'] in ORIGINAL_EPSILONS]
    fine = [p for p in evaluated if p['epsilon'] in FINE_EPSILONS]
    if (any(not p['evaluation']['agrees_diagnostic_only'] for p in original_evaluated)
            and any(p['evaluation']['agrees_diagnostic_only'] for p in fine)):
        findings.append('Scale dependence observed; compatible with local curvature, nonsmoothness or cancellation, not a demonstrated root cause')
    if not findings:
        findings.append('See measured primal/AD, repeat, feasibility and scale evidence; no unique mechanism established')
    return findings


def load_snapshot(study, state, metric):
    import numpy as np
    path = (study / 'case-0' / state['snapshot']).resolve(strict=True)
    require(path.is_relative_to((study / 'case-0').resolve()) and sha(path) == state['snapshot_sha256'],
            'Saved endpoint snapshot escaped or changed')
    with np.load(path, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    for name, shape in (('X', (25, 6)), ('E_upper', (300, 5)),
                        ('atom_logits', (25, 6)), ('bond_logits', (300, 5))):
        require(data[name].dtype == np.float32 and data[name].shape == shape
                and np.isfinite(data[name]).all(), 'Original saved FP32 snapshot shape/value differs')
    for name in ('X', 'E_upper'):
        require(data[name].min() >= 0 and data[name].max() <= 1
                and np.allclose(data[name].sum(-1), 1, atol=1e-6, rtol=0), 'Original saved simplex differs')
    for original, field in ((metric + '_primal', metric + '_loss'), ('repair_weight', 'repair_weight'),
                            ('refreshed_surrogate', 'refreshed_surrogate'), ('autodiff_surrogate', 'autodiff_surrogate')):
        require(data[original].shape == () and float(data[original]) == state[field], 'Saved scalar/JSON correspondence differs')
    require(data['key_data'].tolist() == [0, 17]
            and data['X'].argmax(-1).tolist() == state['readout']['element_indices']
            and data['E_upper'].argmax(-1).tolist() == state['readout']['bond_upper_indices']
            and state['readout']['pairs'] == [[i, j] for i in range(25) for j in range(i + 1, 25)],
            'Original key/argmax identity differs')
    for name, shape in ((metric + '_gradient_X', (25, 6)), (metric + '_gradient_E_upper', (300, 5))):
        require(data[name].dtype == np.float32 and data[name].shape == shape
                and np.isfinite(data[name]).all(), 'Original saved raw gradient shape/precision/finiteness differs')
    return data, hashed(path)


def native_owner(arm, study, summary, pin, output):
    """Construct only the original frozen native objective, with no preparation."""
    import jax
    from mosaic.small_molecule.gradient_checks import array_digest
    if arm == 'protenix':
        from mosaic.small_molecule.io import load_context
        from mosaic.small_molecule.protenix import FrozenProtenixTiny
        from mosaic.small_molecule.losses import NativeContactLoss
        from mosaic.small_molecule.pocket import map_protenix_pocket
        model = FrozenProtenixTiny.load(cache_dir=Path('/work3/s234818/mosaic/cache/mosaic/protenix'))
        context = load_context(study / 'context', summary['context_files'])
        tokens = map_protenix_pocket(context, read(study / 'pocket_definition.json'))
        loss = model.build_loss(context=context, loss=NativeContactLoss(8.0, tokens))
        def frozen():
            return {'model_digest': array_digest(model), 'context_digest': array_digest(context)}
        close = lambda: None
    else:
        from mosaic.small_molecule.boltz2.bridge import AffinityWorker, affinity_loss
        prepared = Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared')
        worker = AffinityWorker(['/work3/s234818/latentmol/env/boltz311/bin/python', '-u',
            '-m', 'mosaic.small_molecule.boltz2.worker', '--context', str(prepared / 'context.pt'),
            '--context-sha256', summary['context_file_sha256'],
            '--checkpoint', '/work3/s234818/latentmol/cache/boltz2/boltz2_aff.ckpt'],
            stderr=output / 'worker.log')
        try:
            require(worker.identity['n_atoms'] == 25 and worker.identity['pocket_definition_sha256'] == pin['pocket']
                    and worker.identity['diffusion_gradient'] is False, 'Diagnostic worker native pocket/input identity differs')
            loss = affinity_loss(worker)
        except Exception:
            worker.close()
            raise
        def frozen():
            previous_timeout = worker.timeout
            worker.timeout = 60
            try:
                reply = worker.request('verify_identity')
            finally:
                worker.timeout = previous_timeout
            require(reply['status'] == 'passed', 'Diagnostic worker frozen verification failed')
            return {k: reply[k] for k in ('model_digest', 'context_digest')}
        close = worker.close
    expected = {'model_digest': pin['model'], 'context_digest': pin['context']}
    try:
        require(frozen() == expected, 'Native frozen input/model identity differs before diagnostics')
    except Exception:
        close()
        raise
    return loss, frozen, close


def save_native(path, graph, *, key, value, gradient=None):
    import jax
    import numpy as np
    arrays = {'X': np.asarray(graph.X), 'E_upper': np.asarray(graph.E_upper),
              'key_data': np.asarray(jax.random.key_data(key)), 'native_scalar': np.asarray(value)}
    if gradient is not None:
        arrays.update(gradient_X=np.asarray(gradient.X), gradient_E_upper=np.asarray(gradient.E_upper))
    require(all(np.isfinite(v).all() for v in arrays.values()), 'Nonfinite diagnostic native evidence')
    np.savez_compressed(path, **arrays)
    return hashed(path)


def independent_value(loss, graph, key, path):
    import equinox as eqx
    import numpy as np
    value, aux = loss(graph, key=key)
    require(np.isfinite(float(value)) and all(np.isfinite(np.asarray(x)).all() for x in aux.values() if eqx.is_array(x)),
            'Nonfinite independent native primal/aux output')
    return float(value), save_native(path, graph, key=key, value=value)


def diagnose_point(loss, graph, logits, key, original_raw, arm, step, output, *, original_data=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.optimizers import evaluate_graph
    from mosaic.small_molecule.state import RelaxedGraph
    output.mkdir(exist_ok=False)
    primals, autodiffs, gradients, proofs = [], [], [], []
    for repeat in range(3):
        evaluation = evaluate_graph(loss, graph, key)
        autodiffs.append(float(evaluation.value))
        gradients.append(evaluation.gradients)
        proofs.append(save_native(output / f'baseline-ad-{repeat}.npz', graph, key=key,
                                  value=evaluation.value, gradient=evaluation.gradients))
        scalar, proof = independent_value(loss, graph, key, output / f'baseline-primal-{repeat}.npz')
        primals.append(scalar); proofs.append(proof)
    record = {'step': step, 'primal_repeats': primals, 'autodiff_repeats': autodiffs,
        'primal_repeat_spread': max(primals) - min(primals),
        'autodiff_repeat_spread': max(autodiffs) - min(autodiffs),
        'maximum_primal_ad_scalar_difference': max(abs(a - p) for a in autodiffs for p in primals),
        'native_input_sha256': array_sha(np.asarray(graph.X), np.asarray(graph.E_upper)),
        'directions': [], 'logit_controls': [], 'proofs': proofs,
        'original_scientific_outcome': 'FAILED endpoint gate remains unchanged',
        'diagnostic_scope': 'Primal/AD and perturbation evidence only; no optimizer update or replacement certification'}
    require(record['maximum_primal_ad_scalar_difference'] <= 2e-5, 'Diagnostic baseline primal/AD parity failed')
    for field in ('X', 'E_upper'):
        values, gradient = np.asarray(getattr(graph, field)), np.asarray(getattr(gradients[0], field))
        require(gradient.dtype == np.float32 and np.isfinite(gradient).all(), 'Native gradient is not finite FP32')
        selected = original_report_direction(original_raw, field) or original_direction(gradient, arm=arm)
        degenerate = selected['source'] == selected['target']
        if degenerate:
            selected = selected | {'target': (selected['source'] + 1) % values.shape[1]}
        alternative = feasible_alternative(values, gradient, selected)
        plans = [('original_recorded_or_original_algorithm', selected)]
        if alternative is not None:
            plans.append(('one_other_feasible_original_scale_contrast', alternative))
        for index, (label, direction) in enumerate(plans):
            row, source, target = (direction[k] for k in ('row', 'source', 'target'))
            analytic = float(gradient[row, target]) - float(gradient[row, source])
            result = {'field': field, **direction, 'selection': label, 'analytic': analytic,
                'original_algorithm_was_zero_contrast': degenerate,
                'fresh_original_algorithm_direction': original_direction(gradient, arm=arm),
                'gradient_repeat_max_absolute_difference': max(float(np.max(np.abs(
                    np.asarray(getattr(g, field)) - gradient))) for g in gradients), 'probes': []}
            if original_data is not None:
                metric = 'contact' if arm == 'protenix' else 'affinity'
                saved_gradient = original_data[metric + '_gradient_' + field]
                result['maximum_absolute_difference_from_original_saved_gradient'] = float(np.max(np.abs(gradient - saved_gradient)))
                result['original_saved_gradient_sha256'] = array_sha(saved_gradient)
            scales = ORIGINAL_EPSILONS + (FINE_EPSILONS if index == 0 and step != 0 else ())
            for epsilon in scales:
                points, geometry = raw_endpoints(values, direction, epsilon)
                probe = {'epsilon': epsilon, 'geometry': geometry, 'evaluation': None,
                         'scale_kind': 'original' if epsilon in ORIGINAL_EPSILONS else 'diagnostic_finer'}
                for sign, point in zip((-1, 1), points):
                    point_graph = RelaxedGraph(jnp.asarray(point), graph.E_upper) if field == 'X' else RelaxedGraph(graph.X, jnp.asarray(point))
                    path = output / f'{field}-{index}-{epsilon:g}-{sign:+d}-input.npz'
                    # Infeasible inputs are preserved as input-only geometry
                    # evidence and are never evaluated by a native model.
                    np.savez_compressed(path, X=np.asarray(point_graph.X), E_upper=np.asarray(point_graph.E_upper),
                                        key_data=np.asarray(jax.random.key_data(key)))
                    record['proofs'].append(hashed(path))
                if geometry['usable_diagnostic_points']:
                    endpoint_values = []
                    for sign, point in zip((-1, 1), points):
                        point_graph = RelaxedGraph(jnp.asarray(point), graph.E_upper) if field == 'X' else RelaxedGraph(graph.X, jnp.asarray(point))
                        repeats = []
                        for repeat in range(2):
                            scalar, proof = independent_value(loss, point_graph, key,
                                output / f'{field}-{index}-{epsilon:g}-{sign:+d}-primal-{repeat}.npz')
                            repeats.append(scalar); record['proofs'].append(proof)
                        endpoint_values.append(repeats)
                    probe['evaluation'] = fd_evidence(*endpoint_values, epsilon, analytic, float(np.mean(primals)))
                    if arm == 'protenix':
                        probe['evaluation']['agrees_diagnostic_only'] = probe['evaluation']['original_protenix_scalar_criterion_diagnostic_only']
                result['probes'].append(probe)
            result['evidence_interpretation'] = classify_direction(result['probes'])
            record['directions'].append(result)
        record['logit_controls'].append(logit_control(loss, graph, logits, key, field, selected, output, record['proofs']))
        write(output / 'report.json', record)
    return record


def logit_control(loss, graph, logits, key, field, direction, output, proofs):
    """Frozen-native representation control at original saved logits, not an update."""
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry.state import GraphLogits
    restored = logits.graph()
    require(np.array_equal(np.asarray(restored.X), np.asarray(graph.X))
            and np.array_equal(np.asarray(restored.E_upper), np.asarray(graph.E_upper)),
            'Original saved logits do not reproduce exact diagnostic graph probabilities')
    value, gradient = eqx.filter_value_and_grad(lambda z: loss(z.graph(), key=key)[0])(logits)
    name = 'atoms' if field == 'X' else 'bonds'
    values, grad = np.asarray(getattr(logits, name)), np.asarray(getattr(gradient, name))
    row, source, target = (direction[k] for k in ('row', 'source', 'target'))
    analytic = float(grad[row, target]) - float(grad[row, source])
    require(np.isfinite(values).all() and np.isfinite(grad).all() and np.isfinite(float(value)),
            'Nonfinite saved-logit representation control')
    original = output / (field + '-logit-autodiff.npz')
    np.savez_compressed(original, atom_logits=np.asarray(logits.atoms), bond_logits=np.asarray(logits.bonds),
                        gradient_atom_logits=np.asarray(gradient.atoms), gradient_bond_logits=np.asarray(gradient.bonds),
                        native_scalar=np.asarray(value))
    proofs.append(hashed(original))
    result = {'field': field, **direction, 'scope': 'Saved-logit coordinate derivative only; no proposed optimizer step',
              'autodiff_scalar': float(value), 'analytic': analytic, 'probes': []}
    for epsilon in ORIGINAL_EPSILONS:
        endpoints, changed = [], []
        for sign in (-1, 1):
            moved = values.copy()
            moved[row, source] -= np.float32(sign * epsilon)
            moved[row, target] += np.float32(sign * epsilon)
            point = GraphLogits(jnp.asarray(moved), logits.bonds) if field == 'X' else GraphLogits(logits.atoms, jnp.asarray(moved))
            point_graph = point.graph()
            represented = not (np.array_equal(np.asarray(point_graph.X), np.asarray(graph.X))
                               and np.array_equal(np.asarray(point_graph.E_upper), np.asarray(graph.E_upper)))
            changed.append(represented)
            repeats = []
            if represented:
                for repeat in range(2):
                    scalar, proof = independent_value(loss, point_graph, key,
                        output / f'{field}-logit-{epsilon:g}-{sign:+d}-{repeat}.npz')
                    repeats.append(scalar); proofs.append(proof)
            endpoints.append(repeats)
        result['probes'].append({'epsilon': epsilon, 'both_points_representably_changed': all(changed),
            'evaluation': fd_evidence(*endpoints, epsilon, analytic, float(value)) if all(changed) else None,
            'certifies_original_raw_probability_gate': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=JOBS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--nearby-step', type=int, choices=NEARBY_CHOICES)
    parser.add_argument('--max-output-bytes', type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument('--max-seconds', type=int, default=5400)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'All endpoint diagnostics require allocated DTU GPU')
    own_source = Path(__file__).resolve().parents[1]
    require(sha(own_source / 'source_manifest.json') == args.manifest, 'Diagnostic release manifest changed')
    file_pin = read(own_source / 'source_manifest.json')['files']['scripts/lrboost_endpoint_diagnostics.py']
    require(sha(Path(__file__).resolve()) == (file_pin['sha256'] if isinstance(file_pin, dict) else file_pin),
            'Diagnostic script differs from its sealed release')
    require(args.output.resolve() == diagnostic_output_path(own_source, args.arm),
            'Diagnostic output must be fresh and outside every original scientific study')
    from jq1_lrboost_report_contract import ARMS, POCKET, authenticate_inputs, authenticate_study
    require(sha(SOURCE / 'source_manifest.json') == SOURCE_MANIFEST, 'Original scientific source610 changed')
    provenance = authenticate_inputs(args.arm)
    study, summary, case = authenticate_study(args.arm, allow_failed=True)
    pin = ARMS[args.arm] | {'pocket': POCKET}
    require(pin['job'] == JOBS[args.arm] and summary['status'] == 'failed'
            and case['status'] == 'passed' and len(case['states']) == 10001,
            'Diagnostics require exact preserved failed-terminal full-budget studies')
    steps = selected_steps(args.nearby_step)
    selected = [case['states'][step] for step in steps]
    native_metadata = ({'context_files': summary['context_files']} if args.arm == 'protenix'
                       else {'context_file_sha256': summary['context_file_sha256']})
    original_raw = {0: read(study / 'native_gradient_checks/report.json'),
                    10000: read(study / 'terminal_gradient_checks/report.json')}
    require(original_raw[0]['status'] == 'passed' and original_raw[10000]['status'] == 'failed',
            'Original initial/failed-terminal applicability changed')
    del case
    del summary
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts')]
    import jax
    import jax.numpy as jnp
    import mosaic
    from mosaic.small_molecule.state import RelaxedGraph
    from mosaic.small_molecule.chemistry.state import GraphLogits
    from mosaic.small_molecule.provenance import environment_identity
    require(Path(mosaic.__file__).resolve().is_relative_to(SOURCE / 'src'), 'Scientific package was shadowed')
    require(not jax.config.jax_enable_x64 and all(d.platform == ('gpu' if args.arm == 'protenix' else 'cpu') for d in jax.devices()),
            'Diagnostic native FP32 JAX backend differs from original arm')
    jax.config.update('jax_default_matmul_precision', 'highest')
    args.output.mkdir(parents=True, exist_ok=False)
    require(120 < args.max_seconds <= 5400, 'Diagnostics must retain the declared at-most-90-minute internal budget')
    output_budget(args.output, args.max_output_bytes)
    record = {'version': VERSION, 'status': 'failed', 'arm': args.arm, 'original_job': JOBS[args.arm],
        'diagnostic_job': os.environ['LSB_JOBID'], 'diagnostic_manifest_sha256': args.manifest,
        'scientific_manifest_sha256': SOURCE_MANIFEST, 'captured_utc': datetime.now(timezone.utc).isoformat(),
        'execution_scope': {'precision': 'FP32 native graphs/models',
            'jax_backend': 'gpu' if args.arm == 'protenix' else 'cpu',
            'native_backend': 'JAX CUDA' if args.arm == 'protenix' else 'isolated native PyTorch CUDA worker',
            'context': 'Exact original saved frozen context; no preparation/diffusion',
            'gradients_through_diffusion': False},
        'environment': environment_identity(),
        'original_outcome': 'FAILED terminal raw-gradient gate; permanently preserved',
        'can_certify_or_restart_original_study': False, 'selected_saved_steps': steps,
        'provenance': provenance, 'source_summary': hashed(study / 'summary.json'),
        'source_terminal_report': hashed(study / 'terminal_gradient_checks/report.json'), 'points': [],
        'persistent_output_allowance_bytes': args.max_output_bytes,
        'maximum_internal_seconds': args.max_seconds, 'cleanup_reserve_seconds': 120}
    write(args.output / 'assessment.json', record)
    def timeout_handler(signum, frame):
        raise TimeoutError('Diagnostic evaluation budget exhausted; original saved studies remain unchanged')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, args.max_seconds - 120)
    frozen = close = None
    expected = {'model_digest': pin['model'], 'context_digest': pin['context']}
    try:
        inputs = []
        for state in selected:
            data, proof = load_snapshot(study, state, pin['metric'])
            copied = args.output / ('original-step-' + str(state['step']) + '.npz')
            shutil.copyfile(proof['path'], copied)
            require(sha(copied) == proof['sha256'], 'Original diagnostic input copy changed')
            inputs.append((state, data, proof))
        record['input_snapshots'] = [proof for _, _, proof in inputs]
        write(args.output / 'assessment.json', record)
        loss, frozen, close = native_owner(args.arm, study, native_metadata, pin, args.output)
        record['frozen_before'] = frozen()
        for state, data, proof in inputs:
            graph = RelaxedGraph(jnp.asarray(data['X']), jnp.asarray(data['E_upper']))
            logits = GraphLogits(jnp.asarray(data['atom_logits']), jnp.asarray(data['bond_logits']))
            key = jax.random.key(17)
            result = diagnose_point(loss, graph, logits, key, original_raw.get(state['step'], {}),
                                    args.arm, state['step'], args.output / ('step-' + str(state['step'])),
                                    original_data=data)
            result['original_snapshot'] = proof
            result['frozen_after_point'] = frozen()
            require(result['frozen_after_point'] == expected, 'Native identity changed during diagnostics')
            record['points'].append(result)
            record['persistent_output_bytes_observed'] = output_budget(args.output, args.max_output_bytes)
            write(args.output / 'assessment.json', record)
        record['status'] = 'diagnostics_completed_original_failure_preserved'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        record['budget_exhausted'] = isinstance(error, TimeoutError)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            if frozen is not None and not record.get('budget_exhausted'):
                record['frozen_after'] = frozen()
                require(record['frozen_after'] == expected, 'Native frozen identity changed after diagnostics')
            elif frozen is not None:
                record['frozen_after'] = {'status': 'not_verified_after_evaluation_timeout'}
        except Exception as error:
            record['status'] = 'failed'
            record['frozen_error'] = repr(error)
            raise
        finally:
            if close is not None:
                close()
            write(args.output / 'assessment.json', record)
            output_budget(args.output, args.max_output_bytes)


if __name__ == '__main__':
    main()
