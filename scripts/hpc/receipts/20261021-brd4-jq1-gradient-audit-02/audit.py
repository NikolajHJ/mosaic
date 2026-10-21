"""Allocated, saved-only comparison of gradient scale and actual update size."""

import os
if not os.environ.get('LSB_JOBID'):
    raise RuntimeError('Numerical saved-data analysis requires allocated DTU HPC')

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from jq1_report_contract import ARMS, RELEASES, read, require, sha, study_path


output = Path(sys.argv[1])
capture = Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-report-04/interim-0003/payload/native-prefix-proofs/capture.json')
receipt = capture.parent.parent / 'report/report_receipt.json'
require(sha(capture) == '676b5612bc1651a43a6f8bb41b6212d78220352e26dadedba8be9fe5d87b6f40', 'Wrong fixed capture bytes')
require(sha(receipt) == '72f9340ebf057f62419e86047489b1d44015e59913c798f37b920b575deefed4', 'Wrong report receipt bytes')
require(read(receipt)['included_updates'] == [7175, 1570, 1439], 'Wrong captured prefixes')


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.size > 0 and np.isfinite(a).all(), 'Nonfinite or empty audit statistic')
    return dict(minimum=float(a.min()), median=float(np.median(a)),
                mean=float(a.mean()), maximum=float(a.max()))


def norm(a):
    return float(np.linalg.norm(a.astype(np.float64).ravel()))


def rms(a):
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


def combined(a, b):
    return np.concatenate((a.ravel(), b.ravel())).astype(np.float64)


def vector_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


record = {
    'status': 'recorded', 'job_id': os.environ['LSB_JOBID'],
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Read-only saved FP32 gradient/step analysis. No model calls, new derivatives, repair, candidate steps or optimization; not a new native readiness result.',
    'capture_sha256': sha(capture), 'report_receipt_sha256': sha(receipt),
    'definitions': {
        'raw': 'Native dL/dp in the saved probability coordinates; losses have different units.',
        'tangent': 'Diagnostic Euclidean row-mean removal only; the native run is not modified.',
        'pullback': 'p * (raw - sum(p * raw)), computed descriptively from saved values.',
        'total': 'Actual saved dL/dlogits used by the optimizer, including any active chemistry.',
        'step': 'Measured consecutive saved logit/probability differences, not LR times a separately aggregated gradient median.',
        'rms': 'sqrt(mean(component**2)); atom150 and unordered-bond1500 components; total1650.',
        'matching': 'All arms share the exact initial X/E; later matched update indices are different molecular probability states.',
        'limits': 'Numerical scale comparison is not a comparison of model quality, experimental affinity or identical objectives.',
    }, 'arms': [],
}
initials = {}
flat_rows = []
for entry in read(capture):
    arm = entry['arm']; pin = ARMS[arm]; latest = entry['latest_update']
    require(entry['job_id'] == pin['job'] and entry['model_digest'] == pin['model']
            and entry['context_digest'] == pin['context']
            and entry['source_manifest_sha256'] == pin['manifest']
            and entry['config_sha256'] == pin['config'], 'Frozen input identity differs')
    root = study_path(arm) / 'case-0'
    rows = {}
    for proof in entry['progress_evidence']['jsonl_prefixes']:
        path = Path(proof['path'])
        with path.open('rb') as stream:
            raw = stream.read(proof['bytes'])
        require(hashlib.sha256(raw).hexdigest() == proof['sha256'], 'Captured JSONL prefix changed')
        rows[path.stem] = [json.loads(line) for line in raw.splitlines()]
    states, updates = rows['states'][:latest + 1], rows['updates'][:latest]
    require([s['step'] for s in states] == list(range(latest + 1))
            and [u['step'] for u in updates] == list(range(latest)), 'Noncontiguous prefix')
    require(read(root.parent / 'native_gradient_checks/report.json')['status'] == 'passed'
            and read(root / 'combined_initial_checks/report.json')['status'] == 'passed', 'Original entry gates failed')
    cache = {}

    def snapshot(step):
        if step not in cache:
            state = states[step]; path = (root / state['snapshot']).resolve(strict=True)
            require(path.is_relative_to(root.resolve()) and sha(path) == state['snapshot_sha256'], 'Snapshot path/hash differs')
            with np.load(path, allow_pickle=False) as f:
                cache[step] = {k: f[k].copy() for k in f.files}
            a = cache[step]
            require(a['X'].shape == (25, 6) and a['E_upper'].shape == (300, 5), 'Wrong occupied slot scope')
            require(all(np.isfinite(x).all() for x in a.values()), 'Nonfinite saved audit array')
            require(a['X'].argmax(-1).tolist() == state['readout']['element_indices']
                    and a['E_upper'].argmax(-1).tolist() == state['readout']['bond_upper_indices'], 'Saved raw readout differs')
            require(float(a[pin['metric'] + '_primal']) == state[pin['metric'] + '_loss'], 'Saved native scalar differs')
        return cache[step]

    def observation(step):
        a = snapshot(step)
        result = {'step': step, 'native_loss': states[step][pin['metric'] + '_loss'],
                  'repair_weight': states[step]['repair_weight'],
                  'chemistry_loss': states[step]['components'].get('chemistry_loss')}
        pulls, gradients = [], []
        for block, field, key in [('atom', 'X', 'atom'), ('bond', 'E_upper', 'bond')]:
            p = a[field].astype(np.float64)
            g = a[pin['metric'] + '_gradient_' + field].astype(np.float64)
            tangent = g - g.mean(-1, keepdims=True)
            pulled = p * (g - (p * g).sum(-1, keepdims=True))
            total = a['gradient_' + key + '_logits'].astype(np.float64)
            pulls.append(pulled); gradients.append(total)
            for name, vector in [('raw_native', g), ('tangent_native', tangent),
                                 ('pulled_native', pulled), ('total_logit', total)]:
                result[block + '_' + name + '_rms'] = rms(vector)
                result[block + '_' + name + '_l2'] = norm(vector)
            result[block + '_softmax_norm_ratio'] = norm(pulled) / norm(tangent)
            result[block + '_total_minus_native_pullback_l2'] = norm(total - pulled)
            result[block + '_winning_probability_mean'] = float(p.max(-1).mean())
            top = np.sort(p, axis=-1)[:, -2:]
            result[block + '_argmax_margin_minimum'] = float((top[:, 1] - top[:, 0]).min())
        gtotal = combined(*gradients)
        result['total_logit_gradient_l2'] = norm(gtotal)
        result['total_logit_gradient_rms'] = rms(gtotal)
        result['native_pullback_l2'] = norm(combined(*pulls))
        if step < latest:
            b = snapshot(step + 1); update = updates[step]; rate = update['accepted_step_size']
            da = b['atom_logits'].astype(np.float64) - a['atom_logits'].astype(np.float64)
            db = b['bond_logits'].astype(np.float64) - a['bond_logits'].astype(np.float64)
            dx = b['X'].astype(np.float64) - a['X'].astype(np.float64)
            de = b['E_upper'].astype(np.float64) - a['E_upper'].astype(np.float64)
            result.update(accepted_rate=rate, line_search_trials=len(update['attempts']),
                          actual_logit_step_l2=norm(combined(da, db)),
                          actual_logit_step_rms=rms(combined(da, db)),
                          predicted_eta_g_l2=norm(rate * gtotal),
                          full_nominal_eta_g_l2=norm(0.03 * gtotal),
                          actual_atom_probability_step_rms=rms(dx),
                          actual_bond_probability_step_rms=rms(de),
                          actual_probability_step_rms=rms(combined(dx, de)),
                          actual_probability_step_max=float(np.abs(combined(dx, de)).max()),
                          step_replay_max_abs_residual=float(np.abs(combined(da, db) + rate * gtotal).max()),
                          strict_decrease=bool(update['fixed_after'] < update['fixed_before']),
                          no_op=bool(update['no_op']), fallback=bool(update['fallback_used']))
        return result

    initial = snapshot(0)
    initials[arm] = (initial['X'], initial['E_upper'])
    points = [observation(s) for s in sorted({0, 100, 250, 500, 750, 1000, 1300, 1400, latest})]
    windows = {}
    for label, start, end in [('first100', 0, 100), ('900_999', 900, 1000),
                              ('1300_1399', 1300, 1400), ('latest100', latest - 100, latest)]:
        measured = [observation(s) for s in range(start, end)]
        numeric = {k: stats([r[k] for r in measured]) for k in measured[0]
                   if k not in ('step', 'chemistry_loss') and type(measured[0][k]) in (int, float)}
        windows[label] = {'first': start, 'end_exclusive': end, 'count': len(measured),
                          'metrics': numeric, 'accepted_rates': dict(Counter(str(r['accepted_rate']) for r in measured)),
                          'strict_decreases': sum(r['strict_decrease'] for r in measured),
                          'fallbacks': sum(r['fallback'] for r in measured),
                          'no_ops': sum(r['no_op'] for r in measured)}
        flat_rows.extend(dict(arm=arm, window=label, **r) for r in measured)
    code = RELEASES / pin['tag'] / 'source'
    code_proofs = {f: sha(code / f) for f in ('examples/brd4_25_atoms.py',
        'src/mosaic/small_molecule/losses.py', 'src/mosaic/small_molecule/chemistry/trajectory.py')}
    record['arms'].append({'arm': arm, 'native_job': pin['job'], 'through_update': latest,
        'source_manifest_sha256': pin['manifest'], 'configuration_sha256': pin['config'],
        'model_digest': pin['model'], 'context_digest': pin['context'],
        'scientific_code_proofs': code_proofs, 'points': points, 'windows': windows})

require(all(np.array_equal(initials['protenix'][i], initials[arm][i])
            for arm in initials for i in (0, 1)), 'Arms do not share exact initial probability arrays')
record['initial_graph_arrays_equal'] = True
record['initial_graph_sha256'] = [vector_sha(x) for x in initials['protenix']]
record['comparisons_to_protenix'] = {}
for label in ('first100', '900_999', '1300_1399', 'latest100'):
    baseline = record['arms'][0]['windows'][label]['metrics']
    ratios = {}
    for item in record['arms'][1:]:
        data = item['windows'][label]['metrics']
        ratios[item['arm']] = {metric: data[metric]['median'] / baseline[metric]['median'] for metric in
            ('total_logit_gradient_l2', 'accepted_rate', 'actual_logit_step_l2',
             'actual_atom_probability_step_rms', 'actual_bond_probability_step_rms', 'actual_probability_step_rms')}
    record['comparisons_to_protenix'][label] = ratios

(output / 'analysis.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
(output / 'per_update.json').write_text(json.dumps(flat_rows, allow_nan=False) + '\n')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
items = record['arms']; labels = ['Protenix', 'Boltz-2', 'Boltz + chem']; colors = [ARMS[x['arm']]['color'] for x in items]
for ax, metric, title in zip(axes,
        ('total_logit_gradient_l2', 'accepted_rate', 'actual_logit_step_l2', 'actual_probability_step_rms'),
        ('Gradient L2 in logits', 'Accepted learning rate', 'Measured logit-step L2', 'Measured probability-step RMS')):
    values = [x['windows']['1300_1399']['metrics'][metric]['median'] for x in items]
    ax.bar(labels, values, color=colors); ax.set_yscale('log'); ax.set_title(title, fontsize=10)
    ax.tick_params(axis='x', labelrotation=25, labelsize=8); ax.grid(axis='y', alpha=.2)
    for i, value in enumerate(values):
        ax.annotate(f'{value:.3g}', (i, value), ha='center', xytext=(0, 5), textcoords='offset points', fontsize=8)
fig.suptitle('Saved update mechanics: median over updates 1300–1399\nDifferent objectives and units; no model reruns or parameter changes', fontsize=11)
fig.savefig(output / 'gradient_step_comparison.png', dpi=160)
plt.close(fig)
print(json.dumps({'status': 'recorded', 'job_id': os.environ['LSB_JOBID'],
                  'comparisons_to_protenix': record['comparisons_to_protenix']}, indent=2))
