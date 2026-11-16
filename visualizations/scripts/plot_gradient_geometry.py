"""Show where saved bond gradients point and where long edges enter a run."""
import argparse
import json
from pathlib import Path

from plot_pair import plt, np, read, digest


def plot(runs, output, labels, title):
    if not 2 <= len(runs) <= 4 or len(labels) != len(runs):
        raise ValueError('Provide two to four runs with one label each')
    configs = [read(run / 'stage2_config.json') for run in runs]
    for field in ('context_sha256',):
        values = [config.get(field) for config in configs]
        if not values[0] or len(set(values)) != 1:
            raise ValueError(f'Require the same identified {field}')
    checkpoints = [config.get('worker_identity', {}).get('checkpoint_sha256') for config in configs]
    if not checkpoints[0] or len(set(checkpoints)) != 1:
        raise ValueError('Require the same identified checkpoint')
    objectives = [config.get('optimization_objective', 'affinity_pred_value') for config in configs]
    if len(set(objectives)) != 1 or any(config.get('hardening') for config in configs):
        raise ValueError('Compare the same relaxed objective; hardening derivatives need separate analysis')
    initial, sources, audits, steps = [], [], [], []
    for run in runs:
        trace_path = run / 'trajectory/trace.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        arrays_path = run / 'trajectory' / trace[0]['evaluated_arrays']
        with np.load(arrays_path, allow_pickle=False) as arrays:
            initial.append({key: arrays[key].copy() for key in ('X', 'E_upper', 'gradient_X', 'gradient_E_upper')})
        audit_path = run / 'gradient_geometry_per_step.json'
        summary_path = run / 'gradient_geometry_audit.json'
        audits.append(read(audit_path))
        steps.append(read(run / 'research_per_step.json'))
        summary = read(summary_path)
        if summary['trace_sha256'] != digest(trace_path) or summary['context_sha256'] != configs[0]['context_sha256']:
            raise ValueError('Saved gradient audit does not match the current trace/context')
        sources.append({'run': str(run.resolve()), 'config_sha256': digest(run / 'stage2_config.json'),
                        'trace_sha256': digest(trace_path), 'initial_arrays_sha256': digest(arrays_path),
                        'audit_sha256': digest(audit_path), 'summary_sha256': digest(summary_path),
                        'per_step_sha256': digest(run / 'research_per_step.json')})
    if any(not np.array_equal(initial[0][key], item[key]) for item in initial for key in initial[0]):
        raise ValueError('Require identical initial arrays and gradients')
    colors = ['#246f8b', '#b75b35', '#208568', '#955f9b']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    summaries = []
    for index, (rows, states, config, label, color) in enumerate(zip(audits, steps, configs, labels, colors)):
        update = np.array([row['step'] for row in rows])
        near = np.array([row['short_pair_gradient_rms'] for row in rows], dtype=float)
        far = np.array([row['long_pair_gradient_rms'] for row in rows], dtype=float)
        ratio = np.divide(far, near, out=np.full_like(far, np.nan), where=near > 0)
        legend = f"{label} ({config['eta_X']:.3g}/{config['eta_E']:.3g})"
        axes[0, 0].plot(update, ratio, color=color, lw=1.2, label=legend)
        for field, style in [('gradient', '-'), ('operators', '--')]:
            counts = np.array([row['long_edge_activations_' + field] for row in rows])
            axes[0, 1].plot(update + 1, np.cumsum(counts), color=color, ls=style, lw=1.25)
        axes[1, 0].plot([row['step'] for row in states],
                        [row.get('maximum_frozen_bond_length_A', np.nan) for row in states], color=color, lw=1.15)
        # Show rates of local operator/model disagreement, rather than treating
        # a projection that opposes descent as a failed optimization update.
        axes[1, 1].plot(update, [row['bond_operator_descent_alignment'] for row in rows], color=color, lw=1.1)
        tail = ratio[len(ratio)//2:]
        summaries.append({'label': label, 'median_tail_far_near_per_pair_gradient_rms_ratio':
                          float(np.nanmedian(tail)) if np.isfinite(tail).any() else None,
                          'long_no_bond_to_bond_events_gradient': sum(row['long_edge_activations_gradient'] for row in rows),
                          'long_no_bond_to_bond_events_operators': sum(row['long_edge_activations_operators'] for row in rows)})
    axes[0, 0].set(title='Bond-gradient strength per pair', ylabel='RMS at >2.5 Å / RMS at ≤2.5 Å')
    axes[0, 0].axhline(1, color='#64748b', lw=.8, alpha=.5)
    axes[0, 1].set(title='Long NO_BOND → bond events', ylabel='Cumulative events (repeats included)')
    axes[0, 1].plot([], [], color='#475569', label='Gradient + simplex')
    axes[0, 1].plot([], [], color='#475569', ls='--', label='Subsequent operators')
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[1, 0].set(title='Longest hard-graph bond in the fixed pose', ylabel='Distance (Å)')
    axes[1, 0].axhline(2.5, color='#a04435', ls=':', lw=1)
    axes[1, 1].set(title='Operator displacement / model descent agreement', ylabel='Cosine', ylim=(-1.05, 1.05))
    axes[1, 1].axhline(0, color='#64748b', lw=.8, alpha=.5)
    for axis in axes.flat:
        axis.set_xlabel('Update')
        axis.grid(alpha=.17)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, x=.07, ha='left', fontsize=16, y=.98)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=2,
               frameon=False, fontsize=9, title='Run (atom / bond learning rate)', title_fontsize=9)
    fig.text(.07, .035, 'Same initial probabilities/gradients, context, checkpoint and relaxed objective; rates and methods may differ.\n'
             'Near/far RMS is normalized per pair. Edge events are local before/after argmax changes, not unique molecules or causal score effects.\n'
             'Distance uses frozen coordinates; the broad 0.8–2.5 Å screen does not establish a realistic pose or binding.', fontsize=8.5, color='#475569')
    fig.subplots_adjust(left=.09, right=.98, top=.81, bottom=.18, hspace=.42, wspace=.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'title': title, 'sources': sources, 'summary': summaries,
        'script_sha256': digest(__file__), 'plot_helper_sha256': digest(Path(__file__).with_name('plot_pair.py'))}, indent=2) + '\n')
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', action='append')
    parser.add_argument('--title', default='Gradients, connectivity and frozen geometry')
    args = parser.parse_args()
    plot(args.runs, args.out, args.label or [run.name for run in args.runs], args.title)
