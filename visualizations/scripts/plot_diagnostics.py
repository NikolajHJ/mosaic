"""Compare saved gradient movement and feasibility without pooling model losses."""
import argparse
import json
from pathlib import Path
import textwrap

from plot_pair import plt, np, read, digest
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


def plot(runs, output, labels, title):
    if len(runs) < 2 or len(labels) != len(runs):
        raise ValueError('Provide at least two runs and one label per run')
    configs = [read(run / 'stage2_config.json') for run in runs]
    for field in ('context_sha256',):
        values = [config.get(field) for config in configs]
        if not values[0] or any(value != values[0] for value in values):
            raise ValueError(f'Require the same identified {field}')
    checkpoints = [config.get('worker_identity', {}).get('checkpoint_sha256') for config in configs]
    if not checkpoints[0] or len(set(checkpoints)) != 1:
        raise ValueError('Require the same identified checkpoint')
    first, rows, sources = [], [], []
    for run in runs:
        trace = run / 'trajectory/trace.jsonl'
        initial = json.loads(trace.read_text().splitlines()[0])
        path = run / 'trajectory' / initial['evaluated_arrays']
        with np.load(path, allow_pickle=False) as arrays:
            first.append({key: arrays[key].copy() for key in ('X', 'E_upper')})
        sidecar = run / 'research_per_step.json'
        rows.append(read(sidecar))
        sources.append({'run': str(run.resolve()), 'trace_sha256': digest(trace),
                        'initial_arrays_sha256': digest(path), 'config_sha256': digest(run / 'stage2_config.json'),
                        'per_step_sha256': digest(sidecar)})
    if any(not np.array_equal(first[0][key], item[key]) for item in first for key in first[0]):
        raise ValueError('Initial probability arrays differ')
    last = max(point['step'] for series in rows for point in series)
    colors = ['#246f8b', '#b75b35', '#208568', '#955f9b', '#817130', '#5a65a7']
    if len(runs) > len(colors):
        raise ValueError('Use at most six runs to keep the figure readable')
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), gridspec_kw={'width_ratios': [1.05, 1.2, 1.2]})
    colormap = ListedColormap(['#dba59d', '#65ac91'])
    colormap.set_bad('#e5e7eb')
    for axis, geometry in [(axes[0, 0], False), (axes[1, 0], True)]:
        matrix = np.full((len(rows), last + 1), np.nan)
        for index, series in enumerate(rows):
            for point in series:
                valid = point.get('valid_connected')
                frozen = point.get('frozen_bond_length_pass')
                value = (bool(valid) and bool(frozen)) if geometry and valid is not None and frozen is not None else valid if not geometry else None
                if value is not None:
                    matrix[index, point['step']] = float(value)
        axis.imshow(matrix, aspect='auto', interpolation='nearest', cmap=colormap, vmin=0, vmax=1,
                    extent=(-.5, last + .5, len(rows) - .5, -.5))
        axis.set(yticks=range(len(rows)), yticklabels=[textwrap.fill(label, 17) for label in labels], xlabel='Update',
                 title='Graph valid + connected\nand broad frozen-bond screen' if geometry else 'Graph valid and connected')
        axis.tick_params(axis='y', labelsize=9)
    handles = []
    for series, config, label, color in zip(rows, configs, labels, colors):
        steps = [point['step'] for point in series]
        value = lambda field: [point.get(field) if point.get(field) is not None else np.nan for point in series]
        for column, suffix in [(1, 'X'), (2, 'E')]:
            axes[0, column].plot(steps, value('actual_step_' + suffix), color=color, lw=1.25)
            axes[0, column].plot(steps, value('simplex_step_' + suffix), color=color, lw=.9, ls='--', alpha=.65)
            axes[1, column].plot(steps, value('descent_alignment_' + suffix), color=color, lw=1.1)
        handles.append(Line2D([], [], color=color, label=f"{label} ({config['eta_X']:.4g}/{config['eta_E']:.4g})"))
    for column, field in [(1, 'Atom'), (2, 'Bond')]:
        axes[0, column].set(title=field + ' probability movement', ylabel='L2 change (log scale above 0.001)')
        axes[0, column].set_yscale('symlog', linthresh=.001)
        axes[1, column].set(title=field + ' descent / movement agreement', ylabel='Cosine', xlabel='Update', ylim=(-1.05, 1.05))
        axes[1, column].axhline(0, color='#64748b', alpha=.4, lw=.8)
        for axis in axes[:, column]:
            axis.set_xlim(0, last)
            axis.grid(alpha=.17)
            axis.spines[['top', 'right']].set_visible(False)
            axis.tick_params(labelsize=9)
    fig.suptitle(title, x=.055, ha='left', fontsize=16, y=.985)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .945), ncol=2, frameon=False, fontsize=9,
               title='Run (atom / bond learning rate)', title_fontsize=9)
    fig.text(.055, .05, 'Indicators: green yes, red no, grey unknown. Solid movement includes chemistry corrections; dashed is gradient + simplex only.\n'
             'Same initial probabilities, model and context. Hardening agreement uses surrogate derivatives. States are correlated.\n'
             'The broad 0.8–2.5 Å geometry screen is necessary, not pose or binding validation. Model losses with different units are not pooled.', fontsize=8.5, color='#475569')
    for axis in axes.flat:
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    fig.subplots_adjust(left=.12, right=.98, top=.82, bottom=.18, hspace=.6, wspace=.4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps({'sources': sources, 'title': title, 'labels': labels,
        'script_sha256': digest(__file__), 'shared_plot_helper_sha256': digest(Path(__file__).with_name('plot_pair.py'))}, indent=2) + '\n')
    print(output.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', action='append')
    parser.add_argument('--title', default='Gradient movement and molecular feasibility')
    args = parser.parse_args()
    plot(args.runs, args.out, args.label or [run.name for run in args.runs], args.title)
