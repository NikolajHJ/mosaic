"""Allocated saved-only JQ1 report with bounded, exact FP32 frame assets.

The three studies contain 30,003 frames. Probabilities are losslessly packed in
small external chunks so opening the HTML does not allocate the whole dataset.
No model, RDKit conversion, repair, inference or resampling is performed.
"""

import argparse
import json
import os
from pathlib import Path

from jq1_report_contract import ARMS, authenticate_study, read, require, sha, study_path


def frame_arrays(root, state, metric):
    import numpy as np

    path = (root / state['snapshot']).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and sha(path) == state['snapshot_sha256'], 'Snapshot path/hash differs')
    with np.load(path, allow_pickle=False) as saved:
        x, e = saved['X'], saved['E_upper']
        require(x.shape == (25, 6) and e.shape == (300, 5), 'Wrong 25-slot probability shape')
        for values in (x, e):
            require(values.dtype == np.float32 and np.isfinite(values).all(), 'Nonfinite/non-FP32 saved probabilities')
            require(values.min() >= -1e-6 and values.max() <= 1 + 1e-6 and np.allclose(values.sum(-1), 1, atol=1e-6, rtol=0), 'Saved simplex differs')
        for field, snapshot in ((metric + '_loss', metric + '_primal'), ('repair_weight', 'repair_weight'), ('refreshed_surrogate', 'refreshed_surrogate'), ('autodiff_surrogate', 'autodiff_surrogate')):
            require(np.isfinite(saved[snapshot]).all() and float(saved[snapshot]) == state[field], 'Saved snapshot scalar differs: ' + field)
        require(x.argmax(-1).tolist() == state['readout']['element_indices'] and e.argmax(-1).tolist() == state['readout']['bond_upper_indices'], 'Saved raw argmax differs')
        expected_pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
        require(state['readout']['pairs'] == expected_pairs, 'Saved unordered pair correspondence differs')
        return np.concatenate((x.ravel(), e.ravel())).astype('<f4', copy=False).tobytes()


def write_frames(root, case, destination, metric, *, chunk_size=100):
    """Pure saved-data presentation, also exercised by small allocated fixtures."""
    destination.mkdir(parents=True, exist_ok=False)
    chunks, curves = [], []
    states, updates = case['states'], case['updates']
    require([s['step'] for s in states] == list(range(len(states))), 'Missing/noncontiguous frames')
    require(type(chunk_size) is int and 1 <= chunk_size <= 100, 'Unbounded frame chunk')
    for start in range(0, len(states), chunk_size):
        rows, binary = [], bytearray()
        for index in range(start, min(start + chunk_size, len(states))):
            state = states[index]
            binary.extend(frame_arrays(root, state, metric))
            row = dict(state)
            row['readout'] = {k: v for k, v in state['readout'].items() if k != 'pairs'}
            update = updates[index - 1] if index else None
            row['previous_update'] = None if update is None else {k: update[k] for k in ('fixed_before', 'fixed_after', 'fallback_used', 'no_op', 'accepted_step_size')}
            rows.append(row)
            components = state['components']
            chemical = components.get('chemistry_loss')
            curves.append({'step': state['step'], 'native': state[metric + '_loss'],
                'binder': components.get('affinity_probability_binary'),
                'repair': chemical, 'weighted_repair': None if chemical is None else state['repair_weight'] * chemical,
                'weight': state['repair_weight'],
                'fixed_delta': None if update is None else update['fixed_after'] - update['fixed_before'],
                'fallback': False if update is None else update['fallback_used'],
                'sanitized_connected': bool(state['readout']['sanitized'] and not state['readout']['disconnected'])})
        prefix = f'frames-{start:05d}'
        data_path, json_path = destination / (prefix + '.bin'), destination / (prefix + '.json')
        data_path.write_bytes(binary)
        json_path.write_text(json.dumps(rows, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        chunks.append({'start': start, 'count': len(rows), 'binary': data_path.name, 'binary_sha256': sha(data_path), 'metadata': json_path.name, 'metadata_sha256': sha(json_path)})
    (destination / 'curves.json').write_text(json.dumps(curves, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return chunks, curves


def plot_overlays(studies, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()
    titles = ['Protenix pocket contact (lower is better)', 'Boltz fixed-pose affinity (lower is better)',
              'Logged binder probability (not optimized)', 'Chemistry coefficient (saved state)',
              'Raw repair guidance', 'Weighted repair contribution',
              'Accepted update: loss change with its target fixed', 'Saved argmax sanitized and connected']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Saved update')
        ax.grid(alpha=.2)
    for study in studies:
        rows, color, label = study['curves'], study['color'], study['label']
        if not rows:
            continue
        x = [r['step'] for r in rows]
        axes[0 if study['metric'] == 'contact' else 1].plot(x, [r['native'] for r in rows], color=color, label=label)
        for panel, field in ((2, 'binder'), (3, 'weight'), (4, 'repair'), (5, 'weighted_repair'), (6, 'fixed_delta'), (7, 'sanitized_connected')):
            points = [(r['step'], r[field]) for r in rows if r[field] is not None]
            if points and (panel != 3 or study['arm'] == 'boltz2_chemistry'):
                axes[panel].plot([p[0] for p in points], [p[1] for p in points], color=color, label=label, linewidth=1)
        fallback = [r for r in rows if r['fallback']]
        if fallback:
            axes[6].scatter([r['step'] for r in fallback], [r['fixed_delta'] for r in fallback], marker='x', c=color, s=12)
    axes[3].set_yscale('log')
    axes[1].set_ylabel('Native affinity_pred_value')
    axes[2].set_ylim(-.02, 1.02)
    axes[7].set_ylim(-.05, 1.05)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    figure.savefig(output / 'loss-overlays.svg')
    plt.close(figure)


def assemble(studies, output, *, interim=False):
    """Write only a small manifest into HTML; exact frames load on demand."""
    plot_overlays(studies, output)
    public = [{k: v for k, v in s.items() if k != 'curves'} for s in studies]
    data = {'version': 'jq1-25-saved-report-v1', 'n_atoms': 25, 'frame_stride_floats': 1650,
        'chunk_size': 100, 'studies': public, 'complete': not interim,
        'scope': 'Saved probabilities/readouts only. Contact is not affinity. No diffusion gradients.'}
    template = Path(__file__).with_name('templates') / 'jq1_25_report.html'
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    (output / 'report.html').write_text(template.read_text(encoding='utf-8').replace('__SAVED_REPORT_MANIFEST__', encoded), encoding='utf-8')
    (output / 'report_receipt.json').write_text(json.dumps({'complete': not interim,
        'input_jobs': [s['job_id'] for s in studies], 'states': [s['states'] for s in studies],
        'included_updates': [s.get('latest_update', s['states'] - 1) for s in studies],
        'snapshot_steps': [s.get('snapshot_steps') for s in studies],
        'n_atoms': 25, 'probabilities': 'Lossless little-endian FP32, bounded external chunks',
        'html_bytes': (output / 'report.html').stat().st_size, 'html_sha256': sha(output / 'report.html')}, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completions', type=Path, required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Scientific rendering requires allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    studies = []
    for arm, pin in ARMS.items():
        completion = read(args.completions / ('complete-' + arm) / 'payload/assessment.json')
        require(completion['status'] == 'passed' and completion['native_job'] == pin['job'], 'Native completion evidence failed')
        for proof in completion['proofs']:
            require(sha(proof['path']) == proof['sha256'], 'Native completion proof changed')
        study, summary, case = authenticate_study(arm)
        chunks, curves = write_frames(study / 'case-0', case, args.output / arm, pin['metric'])
        studies.append({'arm': arm, 'job_id': pin['job'], 'label': pin['label'], 'color': pin['color'],
            'metric': pin['metric'], 'chunks': chunks, 'states': len(case['states']), 'curves': curves,
            'counts': case['counts'], 'model_digest': pin['model'], 'context_digest': pin['context'],
            'summary_sha256': sha(study / 'summary.json'), 'source_manifest_sha256': pin['manifest']})
    assemble(studies, args.output)


if __name__ == '__main__':
    main()
