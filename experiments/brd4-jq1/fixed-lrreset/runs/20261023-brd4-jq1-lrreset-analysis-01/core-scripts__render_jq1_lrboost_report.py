"""Allocated higher-rate saved-only JQ1 report with exact bounded FP32 assets.

This presentation consumes the three pinned scientific runs.  It never invokes
a model, converter, sampler or repair routine.  Every included saved probability
frame is retained, including invalid/disconnected raw argmax graphs.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from jq1_lrboost_report_contract import (
    ARMS, authenticate_study, fixed_counts, read, require, saved_outcome, sha, study_path,
)


VERSION = 'jq1-25-lrreset-saved-report-v1'
MILESTONES = tuple(range(1500, 10000, 1000)) + (10000,)
FRAME_FLOATS = 1650
FRAME_BYTES = FRAME_FLOATS * 4


def milestone(latest):
    require(latest is None or type(latest) is int and 0 <= latest <= 10000,
            'Invalid included update for milestone')
    return max((m for m in MILESTONES if latest is not None and m <= latest), default=0)


def frame_arrays(root, state, metric):
    """Validate and pack one original saved snapshot on allocated CPU."""
    import numpy as np

    root = Path(root).resolve(strict=True)
    relative = Path(state['snapshot'])
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Snapshot path/hash differs')
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root) and sha(path) == state['snapshot_sha256'],
            'Snapshot path/hash differs')
    with np.load(path, allow_pickle=False) as saved:
        x, e = saved['X'], saved['E_upper']
        require(x.shape == (25, 6) and e.shape == (300, 5), 'Wrong 25-slot probability shape')
        for values in (x, e):
            require(values.dtype == np.float32 and np.isfinite(values).all(),
                    'Nonfinite/non-FP32 saved probabilities')
            require(values.min() >= -1e-6 and values.max() <= 1 + 1e-6
                    and np.allclose(values.sum(-1), 1, atol=1e-6, rtol=0),
                    'Saved simplex differs')
        for field, snapshot in (
                (metric + '_loss', metric + '_primal'),
                ('repair_weight', 'repair_weight'),
                ('refreshed_surrogate', 'refreshed_surrogate'),
                ('autodiff_surrogate', 'autodiff_surrogate')):
            scalar = saved[snapshot]
            require(scalar.shape == () and np.isfinite(scalar).all()
                    and float(scalar) == state[field], 'Saved snapshot scalar differs: ' + field)
        require(x.argmax(-1).tolist() == state['readout']['element_indices']
                and e.argmax(-1).tolist() == state['readout']['bond_upper_indices'],
                'Saved raw argmax differs')
        expected_pairs = [[i, j] for i in range(25) for j in range(i + 1, 25)]
        require(state['readout']['pairs'] == expected_pairs, 'Saved unordered pair correspondence differs')
        packed = np.concatenate((x.ravel(), e.ravel())).astype('<f4', copy=False).tobytes()
        require(len(packed) == FRAME_BYTES, 'Saved packed frame size differs')
        return packed


def previous_update(update):
    """Retain policy and outcome fields instead of a descent/search shorthand."""
    if update is None:
        return None
    selected = update['attempts'][update['selected_trial']]
    fields = ('selection_policy', 'selected_trial', 'accepted_step_size',
              'fixed_before', 'fixed_after', 'fixed_increased', 'no_op', 'fallback_used')
    return {**{k: update[k] for k in fields}, 'strict': selected['strict'],
            'attempt_count': len(update['attempts'])}


def verify_transition(previous, state, update, previous_probabilities, probabilities, metric):
    """Bind saved update scalar/no-op bookkeeping to its adjacent saved states."""
    import numpy as np

    require(update['fixed_before'] == previous['refreshed_surrogate']
            and update['fixed_after'] == state['previous_fixed_surrogate_after']
            and update[metric + '_before'] == previous[metric + '_loss']
            and update[metric + '_after'] == state[metric + '_loss'],
            'Saved state/update scalar correspondence differs')
    same = np.array_equal(np.frombuffer(previous_probabilities, dtype='<f4'),
                          np.frombuffer(probabilities, dtype='<f4'))
    require(update['no_op'] == bool(same), 'Saved probability no-op correspondence differs')


def write_frames(root, case, destination, metric, *, rate, chunk_size=100):
    """Write every saved frame, with exact scientific validation and no synthesis."""
    states, updates = case['states'], case['updates']
    require(states and all(type(s.get('step')) is int for s in states)
            and [s['step'] for s in states] == list(range(len(states))),
            'Missing/noncontiguous frames')
    require(len(updates) == len(states) - 1, 'Committed state/update prefix differs')
    require(type(chunk_size) is int and 1 <= chunk_size <= 100, 'Unbounded frame chunk')
    fixed_counts(updates, rate)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    chunks, curves, previous_probabilities = [], [], None
    for start in range(0, len(states), chunk_size):
        rows, binary = [], bytearray()
        for index in range(start, min(start + chunk_size, len(states))):
            state = states[index]
            probabilities = frame_arrays(root, state, metric)
            binary.extend(probabilities)
            row = dict(state)
            row['readout'] = {k: v for k, v in state['readout'].items() if k != 'pairs'}
            update = updates[index - 1] if index else None
            if update is not None:
                verify_transition(states[index - 1], state, update,
                                  previous_probabilities, probabilities, metric)
            previous_probabilities = probabilities
            outcome = previous_update(update)
            row['previous_update'] = outcome
            rows.append(row)
            components = state['components']
            chemical = components.get('chemistry_loss')
            delta = None if update is None else update['fixed_after'] - update['fixed_before']
            curves.append({
                'step': state['step'], 'native': state[metric + '_loss'],
                'binder': components.get('affinity_probability_binary'),
                'repair': chemical,
                'weighted_repair': None if chemical is None else state['repair_weight'] * chemical,
                'weight': state['repair_weight'], 'fixed_delta': delta,
                'selection_policy': 'fixed_step',
                'accepted_step_size': None if update is None else update['accepted_step_size'],
                'strict': False if outcome is None else outcome['strict'],
                'fixed_increased': False if outcome is None else outcome['fixed_increased'],
                'scalar_increased': False if delta is None else delta > 0,
                'scalar_equal': False if delta is None else delta == 0,
                'scalar_decreased': False if delta is None else delta < 0,
                'no_op': False if outcome is None else outcome['no_op'],
                'fallback': False,
                'sanitized_connected': bool(state['readout']['sanitized']
                                            and not state['readout']['disconnected']),
            })
        prefix = f'frames-{start:05d}'
        data_path, json_path = destination / (prefix + '.bin'), destination / (prefix + '.json')
        require(len(binary) == FRAME_BYTES * len(rows), 'Chunk probability byte size differs')
        data_path.write_bytes(binary)
        json_path.write_text(json.dumps(rows, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        chunks.append({'start': start, 'count': len(rows), 'binary': data_path.name,
                       'binary_sha256': sha(data_path), 'metadata': json_path.name,
                       'metadata_sha256': sha(json_path)})
    (destination / 'curves.json').write_text(
        json.dumps(curves, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return chunks, curves


def plot_overlays(studies, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(14, 16), constrained_layout=True)
    axes = axes.ravel()
    titles = ['Protenix pocket contact (lower is better)',
              'Boltz fixed-pose affinity (lower is better)',
              'Logged binder probability (separate from optimized metric)',
              'Chemistry coefficient (saved state)', 'Raw repair guidance',
              'Weighted repair contribution',
              'One fixed update: scalar change with its target fixed',
              'Saved argmax sanitized and connected']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Saved update')
        ax.grid(alpha=.2)
    for study in studies:
        rows, color = study['curves'], study['color']
        label = study['label'] + ' (fixed LR ' + str(study['learning_rate']) + ')'
        if not rows:
            continue
        x = [r['step'] for r in rows]
        axes[0 if study['metric'] == 'contact' else 1].plot(
            x, [r['native'] for r in rows], color=color, label=label)
        for panel, field in ((2, 'binder'), (3, 'weight'), (4, 'repair'),
                             (5, 'weighted_repair'), (6, 'fixed_delta'),
                             (7, 'sanitized_connected')):
            points = [(r['step'], r[field]) for r in rows if r[field] is not None]
            if points and (panel != 3 or study['arm'] == 'boltz2_chemistry'):
                axes[panel].plot([p[0] for p in points], [p[1] for p in points],
                                 color=color, label=label, linewidth=1)
        for flag, marker, name in (('scalar_increased', '^', 'scalar increase'),
                                   ('scalar_equal', 'o', 'scalar equality'),
                                   ('no_op', 'x', 'probability no-op')):
            points = [r for r in rows if r[flag]]
            if points:
                axes[6].scatter([r['step'] for r in points], [r['fixed_delta'] for r in points],
                                marker=marker, c=color, s=10, label=study['label'] + ': ' + name)
    if any(r['weight'] > 0 for s in studies for r in s['curves']):
        axes[3].set_yscale('log')
    axes[1].set_ylabel('Native affinity_pred_value')
    axes[2].set_ylim(-.02, 1.02)
    axes[7].set_ylim(-.05, 1.05)
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    figure.savefig(Path(output) / 'loss-overlays.svg')
    plt.close(figure)


def collect_diagnostics(root, output, *, report_manifest=None):
    """Copy authenticated endpoint JSON verbatim; never evaluate its probes."""
    entries = []
    for arm, pin in ARMS.items():
        entry = {'arm': arm, 'original_job': pin['job'], 'status': 'pending', 'requested': arm == 'boltz2',
                 'scope': 'Separate endpoint diagnostics never change the original scientific outcome'}
        capture = None if root is None else Path(root) / ('diagnostic-capture-' + arm) / 'payload/assessment.json'
        if capture is None or not capture.exists():
            entries.append(entry)
            continue
        assessment = read(capture)
        manifest = (Path(report_manifest) if report_manifest is not None
                    else Path(__file__).resolve().parents[1] / 'source_manifest.json')
        cpu = Path(root) / 'cpu-certificate.json'
        require(assessment['status'] == 'passed' and assessment['arm'] == arm
                and assessment['native_job'] == pin['job']
                and assessment['report_source_manifest_sha256'] == sha(manifest)
                and assessment['cpu_certificate_sha256'] == sha(cpu)
                and read(cpu)['status'] == 'passed'
                and read(cpu)['source_manifest_sha256'] == sha(manifest),
                'Endpoint diagnostic capture source/CPU/original job differs')
        report = assessment['diagnostic_report']
        path = Path(report['path'])
        require(path.resolve(strict=True).is_relative_to(capture.parent.resolve(strict=True))
                and not path.is_symlink() and path.is_file()
                and type(report['bytes']) is int and path.stat().st_size == report['bytes']
                and sha(path) == report['sha256'], 'Endpoint diagnostic JSON bytes/path differ')
        require(assessment.get('proofs'), 'Missing endpoint diagnostic capture proof bindings')
        for proof in assessment['proofs']:
            require(sha(proof['path']) == proof['sha256'], 'Endpoint diagnostic capture proof changed')
        raw = read(path)
        require(raw['version'] == 'jq1-lrreset-endpoint-diagnostics-v1'
                and raw['arm'] == arm and raw['original_job'] == pin['job']
                and raw['scientific_manifest_sha256'] == pin['manifest']
                and raw['can_certify_or_restart_original_study'] is False,
                'Separate diagnostic evidence changed its original-run scope')
        destination = Path(output) / 'diagnostics'
        destination.mkdir(parents=True, exist_ok=True)
        assets = {}
        for kind, original in (('report', path), ('capture', capture)):
            target = destination / (arm + '-' + kind + '.json')
            require(not target.exists(), 'Diagnostic output is immutable')
            shutil.copyfile(original, target)
            require(target.read_bytes() == original.read_bytes(), 'Diagnostic verbatim JSON copy differs')
            assets[kind] = {'path': 'diagnostics/' + target.name, 'bytes': target.stat().st_size,
                            'sha256': sha(target)}
        entry.update(status='available', diagnostic_status=raw['status'],
                     diagnostic_job=raw['diagnostic_job'], assets=assets,
                     capture_job=str(assessment['job_id']),
                     diagnostic_manifest_sha256=raw['diagnostic_manifest_sha256'])
        entries.append(entry)
    return entries


def assemble(studies, output, *, interim=False, captured_utc=None, diagnostics=None):
    """Embed a small immutable edition manifest; exact frames load on demand."""
    output = Path(output)
    captured_utc = captured_utc or datetime.now(timezone.utc).isoformat()
    plot_overlays(studies, output)
    public = [{k: v for k, v in s.items() if k != 'curves'} for s in studies]
    all_terminal = all(s['all_terminal'] for s in studies)
    scientific_passed = all(s['scientific_passed'] for s in studies)
    campaign_status = ('failed' if any(s['overall_status'] == 'failed' for s in studies)
                       else 'passed' if scientific_passed else 'in_progress')
    require(interim or all_terminal, 'A final presentation requires all native outcomes')
    outcome = {'presentation_complete': not interim, 'all_terminal': all_terminal,
               'scientific_passed': scientific_passed,
               'strict_scientific_campaign_passed': scientific_passed,
               'overall_success': scientific_passed, 'campaign_status': campaign_status}
    diagnostics = diagnostics if diagnostics is not None else collect_diagnostics(None, output)
    data = {'version': VERSION, 'report_schema': VERSION, 'n_atoms': 25, 'frame_stride_floats': FRAME_FLOATS,
            'chunk_size': 100, 'studies': public, 'complete': not interim,
            'diagnostics': diagnostics,
            **outcome,
            'captured_utc': captured_utc, 'selection_policy': 'fixed_step', 'acceptance_policy': 'fixed_step',
            'scope': 'Saved probabilities/readouts only; exactly one fixed-rate candidate. '
                     'Presentation completion is separate from scientific validation. '
                     'Contact is not affinity. No diffusion gradients.'}
    template = Path(__file__).with_name('templates') / 'jq1_lrboost_report.html'
    encoded = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    html = template.read_text(encoding='utf-8')
    require(html.count('__SAVED_REPORT_MANIFEST__') == 1, 'Fixed template manifest token differs')
    (output / 'report.html').write_text(html.replace('__SAVED_REPORT_MANIFEST__', encoded), encoding='utf-8')
    receipt = {
        'version': VERSION, 'report_version': VERSION, 'report_schema': VERSION, 'complete': not interim,
        'captured_utc': captured_utc, 'selection_policy': 'fixed_step', 'acceptance_policy': 'fixed_step',
        **outcome,
        'diagnostics': diagnostics,
        'input_jobs': [s['job_id'] for s in studies],
        'states': [s['states'] for s in studies],
        'included_updates': [s['latest_update'] for s in studies],
        'milestone_progress': {s['arm']: milestone(s['latest_update']) for s in studies},
        'arm_bindings': [{k: s[k] for k in ('arm', 'job_id', 'learning_rate', 'selection_policy',
                         'native_status', 'source_manifest_sha256', 'config_sha256', 'readiness_sha256',
                         'readiness_status', 'terminal_status', 'terminal_error', 'overall_status',
                         'budget_status', 'scientific_passed', 'all_terminal')}
                         for s in studies],
        'snapshot_steps': [s.get('snapshot_steps') for s in studies],
        'n_atoms': 25, 'chunk_size': 100, 'frame_stride_floats': FRAME_FLOATS,
        'probabilities': 'Every included saved state, lossless little-endian FP32, bounded chunks',
        'html_bytes': (output / 'report.html').stat().st_size,
        'html_sha256': sha(output / 'report.html'),
    }
    (output / 'report_receipt.json').write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def completion_assessment(arm, completions):
    pin = ARMS[arm]
    assessment = read(Path(completions) / ('complete-' + arm) / 'payload/assessment.json')
    require(assessment['status'] == 'passed' and assessment['native_job'] == pin['job']
            and assessment['arm'] == arm and assessment['report_version'] == VERSION
            and type(assessment['states']) is int and 1 <= assessment['states'] <= 10001
            and assessment['updates'] == assessment['states'] - 1
            and assessment['n_atoms'] == 25 and assessment['learning_rate'] == pin['rate']
            and assessment['selection_policy'] == 'fixed_step'
            and assessment['all_probability_frames_verified'] is True
            and assessment['presentation_complete'] is True
            and assessment['scientific_status'] in ('passed', 'failed'),
            'Presentation completion certificate failed or incomplete')
    require(assessment['scientific_status'] != 'passed' or assessment['updates'] == 10000,
            'A scientific pass needs the entire update budget')
    manifest = Path(__file__).resolve().parents[1] / 'source_manifest.json'
    cpu_path = Path(completions) / 'cpu-certificate.json'
    cpu = read(cpu_path)
    require(assessment['report_source_manifest_sha256'] == sha(manifest)
            and assessment['cpu_certificate_sha256'] == sha(cpu_path)
            and cpu['status'] == 'passed' and cpu['source_manifest_sha256'] == sha(manifest),
            'Fixed completion report source/CPU provenance differs')
    for proof in assessment['proofs']:
        require(sha(proof['path']) == proof['sha256'], 'Fixed completion proof changed')
    # A passed presentation collector may preserve a failed scientific outcome.
    # The authenticated original summary, terminal report and same-job native
    # LSF remain authoritative; the collector never promotes a scientific pass.
    outcome = saved_outcome(arm)
    require(outcome['all_terminal'] is True
            and assessment['scientific_status'] == ('passed' if outcome['scientific_passed'] else 'failed'),
            'Presentation certificate disagrees with the original native outcome')
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completions', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Scientific rendering requires allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    studies = []
    for arm, pin in ARMS.items():
        certificate = completion_assessment(arm, args.completions)
        study, summary, case = authenticate_study(arm, allow_failed=True)
        outcome = saved_outcome(arm)
        require(certificate['states'] == len(case['states'])
                and certificate['updates'] == len(case['updates']),
                'Presentation certificate saved budget differs')
        chunks, curves = write_frames(study / 'case-0', case, args.output / arm,
                                     pin['metric'], rate=pin['rate'])
        studies.append({
            'arm': arm, 'job_id': pin['job'], 'label': pin['label'], 'color': pin['color'],
            'metric': pin['metric'], 'learning_rate': pin['rate'], 'rate': pin['rate'],
            'selection_policy': 'fixed_step', 'acceptance_policy': 'fixed_step',
            **{k: outcome[k] for k in ('readiness_status', 'terminal_status', 'terminal_error',
               'overall_status', 'budget_status', 'scientific_passed', 'all_terminal')},
            'native_status': outcome['native_status'],
            'status': ('all native checks passed; every saved frame checked' if outcome['scientific_passed']
                       else 'scientific validation failed; every captured saved frame checked'),
            'chunks': chunks, 'states': len(case['states']), 'curves': curves,
            'latest_update': len(case['updates']),
            'milestone_steps': [0, *(m for m in MILESTONES if m <= len(case['updates']))],
            'counts': fixed_counts(case['updates'], pin['rate']),
            'model_digest': pin['model'], 'context_digest': pin['context'],
            'summary_sha256': sha(study / 'summary.json'),
            'source_manifest_sha256': pin['manifest'], 'config_sha256': pin['config'],
            'readiness_sha256': pin['ready'],
            'progress_evidence': {'terminal': outcome,
                                  'presentation': 'Every captured original saved frame checked; native outcome unchanged',
                                  'collector': certificate['collector_job']},
            'frame_sampling': 'Every saved state; no interpolation',
        })
    assemble(studies, args.output, diagnostics=collect_diagnostics(args.diagnostics, args.output))


if __name__ == '__main__':
    main()
