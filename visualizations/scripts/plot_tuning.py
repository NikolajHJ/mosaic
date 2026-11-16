"""Plot observed study objectives and feasible frontiers from collected runs."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from plot_pair import plt, np, read, digest

# Preserve the original study gates without importing experiment runners.
def frozen_constraints(metrics):
    bad = metrics.get("median_tail_frozen_bad_bond_fraction")
    # A graded distance helps rank infeasible trials when all binary geometry
    # gates fail. Missing geometry is infeasible, never an observed zero error.
    return [.5 - (metrics.get("tail_frozen_supported_valid_fraction") or 0.),
            1 - metrics.get("unique_supported_changed_valid_smiles", 0),
            bad if bad is not None else 1.]

def native_constraints(metrics):
    proposal = metrics.get("proposal_metrics", {})
    screened = metrics.get("native_screened_valid_candidates", 0) if metrics.get("native_pose_capture_required") is True else 0
    return [.5-proposal.get("tail_valid_connected_fraction", 0.), 1-screened]

def frontier(rows):
    eligible = [row for row in rows if row['feasible'] and row['observed_pair']]
    return [row for row in eligible if not any(
        other['x'] >= row['x'] and other['y'] <= row['y'] and
        (other['x'] > row['x'] or other['y'] < row['y']) for other in eligible)]


def plot(root, output, state_path=None, startup_trials=6):
    state_path = state_path or root / 'controller_state.json'
    state = read(state_path)
    rows, missing, sources = [], [], []
    reference_arrays, common_proposal_protocol, common_native_protocol = None, None, None
    for job in state['jobs']:
        if job['group'] not in ('bayesian', 'pose-bayesian') or job['status'] != 'complete':
            continue
        run = root / 'runs' / job['id']
        metric_path, spec_path = run / 'research_metrics.json', run / 'trial_spec.json'
        metrics = read(metric_path) if metric_path.exists() else {}
        if not metrics.get('complete') or not spec_path.exists():
            missing.append(job['id'])
            continue
        spec = read(spec_path)
        native = job['group'] == 'pose-bayesian'
        proposal = spec['proposal'] if native else spec
        directory = run / 'proposal' if native else run
        config_path, trace_path = directory / 'stage2_config.json', directory / 'trajectory/trace.jsonl'
        if not config_path.exists() or not trace_path.exists():
            missing.append(job['id'])
            continue
        config = read(config_path)
        worker = config.get('worker_identity', {})
        protocol = {'context_sha256': config.get('context_sha256'),
                    'checkpoint_sha256': worker.get('checkpoint_sha256'), 'model_digest': worker.get('model_digest'),
                    'recycling_steps': worker.get('recycling_steps'), 'seed': config.get('seed')}
        if any(value is None for value in protocol.values()):
            raise ValueError('Missing proposal model/context provenance')
        if common_proposal_protocol is not None and protocol != common_proposal_protocol:
            raise ValueError('Do not pool different proposal model/context protocols')
        common_proposal_protocol = protocol
        if config.get('hardening') or (not native and config.get('score_metric') != 'affinity_pred_value'):
            raise ValueError('This figure compares relaxed proposals and quantitative frozen scores')
        initial = json.loads(trace_path.read_text().splitlines()[0])
        array_path = (directory / 'trajectory' / initial['evaluated_arrays']).resolve()
        if not array_path.is_relative_to((directory / 'trajectory').resolve()):
            raise ValueError('Initial arrays escape the source trajectory')
        with np.load(array_path, allow_pickle=False) as arrays:
            first = {key: arrays[key].copy() for key in ('X', 'E_upper')}
        if reference_arrays is not None and any(not np.array_equal(first[key], reference_arrays[key]) for key in first):
            raise ValueError('Collected study trials have different starting probabilities')
        reference_arrays = first
        x = metrics.get('native_best_binder_probability' if native else 'tail_frozen_supported_valid_fraction')
        y = metrics.get('native_loss_at_best_binder' if native else 'best_supported_changed_valid_hard_loss')
        observed = x is not None and y is not None and np.isfinite(x) and np.isfinite(y)
        if x is not None and not 0 <= x <= 1:
            raise ValueError('Objective fraction/probability outside [0,1]')
        violations = (native_constraints if native else frozen_constraints)(metrics)
        native_execution = run / 'native_validation/execution.json'
        if native and observed:
            native_protocol = read(native_execution).get('native_protocol')
            if not native_protocol or (common_native_protocol is not None and native_protocol != common_native_protocol):
                raise ValueError('Missing or different rebuilt-native protocol')
            common_native_protocol = native_protocol
        phase = 'fixed startup' if spec.get('startup_anchor') else 'random startup' if job['trial_number'] < startup_trials else 'adaptive TPE'
        row = {'id': job['id'], 'study': 'native' if native else 'frozen', 'trial': job['trial_number'],
               'phase': phase, 'method': proposal.get('method_bundle'), 'mode': proposal['parameters']['mode'],
               'eta_X': proposal['parameters']['eta_X'], 'eta_E': proposal['parameters']['eta_E'],
               'proposal_head': proposal.get('experimental', {}).get('objective', 'quantitative_affinity'),
               'x': x, 'y': y, 'observed_pair': bool(observed),
               'feasible': bool(observed and all(value <= 0 for value in violations)), 'constraints': violations}
        rows.append(row)
        sources.append({'id': job['id'], 'metrics_sha256': digest(metric_path), 'spec_sha256': digest(spec_path),
                        'config_sha256': digest(config_path), 'initial_arrays_sha256': digest(array_path),
                        'native_execution_sha256': digest(native_execution) if native and observed else None})
    styles = {'fixed startup': ('D', '#bc8030'), 'random startup': ('s', '#64748b'), 'adaptive TPE': ('o', '#246f8b')}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fronts = {}
    for axis, study, title, xlabel, ylabel in zip(axes, ('frozen', 'native'),
            ('Frozen-context study', 'Rebuilt native study'),
            ('Valid + supported tail fraction', 'Mean trained-head binder output'),
            ('Best sampled supported hard loss (lower)', 'Quantitative prediction at the same candidate (lower)')):
        subset = [row for row in rows if row['study'] == study]
        for row in subset:
            if not row['observed_pair']:
                continue
            marker, color = styles[row['phase']]
            axis.scatter(row['x'], row['y'], marker=marker, s=62, edgecolors=color,
                         facecolors=color if row['feasible'] else 'none', linewidths=1.5, zorder=3)
        front = sorted(frontier(subset), key=lambda row: (row['x'], row['y']))
        fronts[study] = [row['id'] for row in front]
        axis.plot([row['x'] for row in front], [row['y'] for row in front], ls='--', color='#208568', lw=1, zorder=1)
        for row in front:
            axis.annotate(('N' if study == 'native' else 'F') + str(row['trial']), (row['x'], row['y']),
                          xytext=(6, -12 if row['x'] > .95 else 6), textcoords='offset points', fontsize=9)
        unscored = [str(row['trial']) for row in subset if not row['observed_pair']]
        detail = f"{len(subset)} complete and collected"
        if unscored:
            detail += '\nNo observed objective pair: ' + ', '.join(unscored)
        axis.text(.03, .97, detail, transform=axis.transAxes, va='top', fontsize=9, color='#475569')
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(-.03, 1.08 if study == 'frozen' else 1.03))
        axis.set_xticks([0, .25, .5, .75, 1])
        axis.grid(alpha=.17)
        axis.spines[['top', 'right']].set_visible(False)
        axis.margins(y=.25)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=marker, color=color, ls='', label=phase) for phase, (marker, color) in styles.items()]
    handles += [Line2D([], [], marker='o', markerfacecolor='none', color='#475569', ls='', label='Open: infeasible'),
                Line2D([], [], color='#208568', ls='--', label='Feasible frontier')]
    fig.suptitle('Tuning objectives and the feasibility tradeoff', x=.07, ha='left', fontsize=16, y=.98)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93), ncol=5, frameon=False, fontsize=9)
    timestamp = datetime.fromtimestamp(state['updated'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    footer = ('Snapshot: ' + timestamp + '. Missing predictions are not plotted as penalty values. Numbers label feasible-frontier trials.\n'
              'Frontier uses both objectives; confirmation selection separately prioritizes feasibility/tail rate (frozen) or binder output (native).\n'
              'Frozen: same start/context, sampled hard scores. Native: rebuilt features/pose, up to two selected identities per trial.\n'
              'These are development objectives. Neither broad geometry support nor a model prediction establishes binding.')
    if missing:
        footer += '\nCompleted but not yet collected: ' + ', '.join(missing)
    fig.text(.07, .025, footer, fontsize=8.3, color='#475569')
    fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.27 if not missing else .3, wspace=.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    receipt = {'state_sha256': digest(state_path), 'state_source': str(state_path.resolve()), 'startup_trials': startup_trials,
               'scheduler_updated_utc': timestamp, 'sources': sources,
               'matched_initialization_run_count': len(rows), 'common_proposal_protocol': common_proposal_protocol,
               'common_native_protocol': common_native_protocol,
               'completed_but_uncollected': missing, 'rows': rows, 'feasible_frontiers': fronts,
               'script_sha256': digest(__file__), 'plot_helper_sha256': digest(Path(__file__).with_name('plot_pair.py')),
               'frozen_constraints_source_sha256': digest(__file__),
               'native_constraints_source_sha256': digest(__file__)}
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ['id'])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'figure': str(output.resolve()), 'frontiers': fronts, 'uncollected': missing}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--state', type=Path, help='Optional separately saved scheduler snapshot; raw metrics must still be collected')
    parser.add_argument('--startup-trials', type=int, default=6)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.root, args.out, args.state, args.startup_trials)
