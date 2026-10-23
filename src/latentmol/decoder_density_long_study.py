"""Allocated100epoch dense-chemistry continuation with authenticated milestones.

The encoder/original reference stay frozen. Epoch25/50/75 are declared
diagnostics;100 is the sole final endpoint. No test-selected checkpoint is
used. Only the horizon changes from the protected10epoch source077 prefix.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import torch

from latentmol.decoder_continuation import merge_epoch_history, resume_envelope
from latentmol.decoder_checkpoint_storage import archive_panel_rows, materialized_panel_rows
from latentmol.decoder_density_long_protocol import read_long_protocol, authenticate_base, segments
from latentmol.decoder_density_study import assets, evaluate
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel, assess_advantage
from latentmol.decoder_quality_evaluation import paired_compare
from latentmol.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.train import state_digest


def read(path):
    return json.loads(Path(path).read_text())


def check_unchanged_scientific_modules(protocol, source_root):
    """Only trainer/resume execution changes are allowed in existing model code."""
    manifest = read(Path(protocol['base_release']) / 'release_manifest.json')
    exceptions = {'src/latentmol/decoder_finetune.py', 'src/latentmol/decoder_resume.py'}
    for name, sha in manifest['files'].items():
        if name.startswith('src/') and name.endswith('.py') and name not in exceptions:
            if file_sha256(source_root / name) != sha:
                raise ValueError('Scientific module differs from source077: ' + name)


def completed_parent(fit, history, archive, expected_epoch):
    """Require the exact completed training boundary, independently of evaluation."""
    envelope = resume_envelope(fit, history, archive)
    record = read(envelope['parent_record'])
    if record.get('completed_epochs') != expected_epoch:
        raise ValueError('Parent training ended at another cumulative epoch')
    rows = [json.loads(line) for line in Path(history).read_text().splitlines() if line.strip()]
    if [row.get('epoch') for row in rows] != list(range(1, expected_epoch + 1)):
        raise ValueError('Parent completed-epoch history is not contiguous')
    return envelope


def initial_prefix(protocol, case_label):
    """Reuse completed epoch10 only; restart missing/failed prefixes explicitly.

    Once a completed10 prefix is declared, any corruption fails closed. The
    scheduler ensures all original jobs ended before this policy is applied.
    A failed prefix never supplies partially updated parameters or random state.
    """
    summary_path = Path(protocol['base_campaign_root']) / case_label / 'summary.json'
    if not summary_path.is_file():
        return None, None, None, {'actual_prefix_epochs': 0, 'reason': 'epoch10 summary missing; restart original'}
    parent = read(summary_path)
    fit = parent.get('fit')
    if fit:
        record = read(Path(fit['run_dir']) / 'run.json')
        if record.get('status') == 'completed' and record.get('completed_epochs') == 10:
            metrics = Path(fit['run_dir']) / 'metrics.jsonl'
            envelope = completed_parent(fit, metrics, protocol['base_archive_sha256'], 10)
            return envelope, metrics, {'from_epochs': 10, 'to_epochs': protocol['endpoint_epochs']}, {
                'actual_prefix_epochs': 10, 'reason': 'authenticated completed epoch10 reused'}
        if parent.get('status') == 'completed':
            raise ValueError('Completed epoch10 summary conflicts with its training record')
    elif parent.get('status') == 'completed':
        raise ValueError('Completed epoch10 summary lacks a training checkpoint')
    return None, None, None, {'actual_prefix_epochs': 0,
        'reason': 'epoch10 training failed/incomplete; restart original', 'parent_status': parent.get('status'),
        'parent_summary': str(summary_path), 'parent_summary_sha256': file_sha256(summary_path)}


def compare_panels(reference, candidate):
    """Authenticate and temporarily restore archived rows for unchanged scoring."""
    with materialized_panel_rows(reference), materialized_panel_rows(candidate):
        return paired_compare(reference, candidate, bootstrap_seed=92000)


def native_gate(protocol, base, root, source_root, device):
    """Actual protected decoder: uninterrupted3 vs1→2→3, including tail1.

    This checks epoch extension and TWO successive resumes on the same GPU,
    including AdamW/parameters and all4dedicated streams. It is not efficacy.
    """
    directory = root / 'gate'
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data, reference, _ = assets(base, device)
    latent = reference.config.latent
    del reference
    common = dict(initializer=Path(base['initializer']), train_data=data[0], val_data=data[1],
        output_root=directory / 'training', seed=17, coefficient=1., train_limit=513, val_limit=64,
        validation_panel=make_panel(32, latent, 81000), device=device, source_root=source_root,
        budget_seconds=protocol['gate_budget_seconds'] / 4, profile_updates=9,
        checkpoint_keep_epochs=[1, 2, 3], checkpoint_keep_recent=protocol['checkpoint_keep_recent'],
        archive_validation_rows=protocol['archive_evaluation_rows'])
    final_config = FineTuneConfig(**{**base['fine_tune'], 'epochs': 3})
    full = fit_case(**common, config=final_config)
    prefix = fit_case(**common, config=FineTuneConfig(**{**base['fine_tune'], 'epochs': 1}))
    archive = os.environ['LATENTMOL_SOURCE_SHA256']
    prefix_metrics = Path(prefix['run_dir']) / 'metrics.jsonl'
    first = fit_case(**common, config=final_config,
        resume=completed_parent(prefix, prefix_metrics, archive, 1), stop_after_epoch=2,
        epoch_extension={'from_epochs': 1, 'to_epochs': 3})
    history = directory / 'epochs-001-002.jsonl'
    merge_epoch_history(prefix_metrics, Path(first['run_dir']) / 'metrics.jsonl', history, 2)
    last = fit_case(**common, config=final_config,
        resume=completed_parent(first, history, archive, 2), stop_after_epoch=3)
    merge_epoch_history(history, Path(last['run_dir']) / 'metrics.jsonl', directory / 'epochs-001-003.jsonl', 3)
    left, right = [torch.load(result['checkpoint'], map_location='cpu', weights_only=False) for result in (full, last)]
    if state_digest(left['model_state']) != state_digest(right['model_state']):
        raise ValueError('Native repeated-resume parameters differ from uninterrupted training')
    torch.testing.assert_close(left['decoder_finetune']['optimizer_state'], right['decoder_finetune']['optimizer_state'], rtol=0, atol=0)
    if full['terminal_rng'] != last['terminal_rng'] or full['encoder_sha256'] != last['encoder_sha256']:
        raise ValueError('Native repeated-resume RNG or encoder differs')
    if first['endpoint_completed'] or not last['endpoint_completed']:
        raise ValueError('Native intermediate/final endpoint bookkeeping differs')
    for result in (full, prefix, first, last):
        if ([row['real_points'] for row in result['update_profile']] != [64] * 8 + [1]
                or any(row['real_points'] != row['fresh_prior_points']
                    or row['real_points'] != row['posterior_chemistry_points'] for row in result['update_profile'])):
            raise ValueError('Native repeated-resume dense/tail quota differs')
    write_json(directory / 'passed.json', {'status': 'passed', 'protocol': protocol,
        'base_protocol': base, 'source_archive_sha256': archive,
        'parameter_and_optimizer_equality': True, 'four_stream_equality': True,
        'frozen_encoder_equality': True, 'cases': [full, prefix, first, last],
        'elapsed_seconds': time.monotonic() - started,
        'runtime': {'torch': str(torch.__version__), 'torch_cuda': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(device), 'lsf_job_id': os.environ['LSB_JOBID']},
        'scope': 'Full-width deterministic same-device extension/repeated-resume engineering proof; no efficacy result'})


def run_segment(protocol, base, root, source_root, index, device):
    declaration = segments(base, protocol)[index]
    case_label = f"seed{declaration['seed']}-a{declaration['coefficient']:g}"
    directory = root / case_label / f"e{declaration['stop_epoch']:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'protocol.json', protocol)
    started = time.monotonic()
    fit, model, history, rec, prior = None, None, None, None, None
    lineage = None
    phase = 'parent_authentication'
    try:
        gate = read(root / 'gate/passed.json')
        archive = os.environ['LATENTMOL_SOURCE_SHA256']
        if gate['status'] != 'passed' or gate['protocol'] != protocol or gate['source_archive_sha256'] != archive:
            raise ValueError('Long native gate belongs to another immutable campaign')
        if declaration['segment_index'] == 0:
            resume, parent_metrics, extension, lineage = initial_prefix(protocol, case_label)
            declaration['start_epoch'] = lineage['actual_prefix_epochs']
        else:
            parent = read(root / case_label / f"e{declaration['start_epoch']:03d}" / 'summary.json')
            if not parent.get('training_completed'):
                raise ValueError('Previous segment did not complete its training boundary')
            parent_fit = parent['fit']
            parent_metrics = Path(parent['history']['path'])
            if file_sha256(parent_metrics) != parent['history']['sha256']:
                raise ValueError('Previous cumulative history changed after sealing')
            parent_archive, extension = archive, None
            resume = completed_parent(parent_fit, parent_metrics, parent_archive, declaration['start_epoch'])
            lineage = parent['prefix_lineage']
        phase = 'protected_assets'
        data, reference, checkpoint = assets(base, device)
        latent = reference.config.latent
        del reference
        phase = 'training'
        fit = fit_case(Path(base['initializer']), data[0], data[1], directory / 'training',
            FineTuneConfig(**{**base['fine_tune'], 'epochs': protocol['endpoint_epochs']}),
            seed=declaration['seed'], coefficient=declaration['coefficient'],
            train_limit=base['train_limit'], val_limit=base['validation_limit'],
            validation_panel=make_panel(base['validation_prior_count'], latent, 81000),
            device=device, source_root=source_root, budget_seconds=protocol['segment_budget_seconds'],
            resume=resume, stop_after_epoch=declaration['stop_epoch'], epoch_extension=extension, profile_updates=8,
            checkpoint_keep_epochs=protocol['milestones'], checkpoint_keep_recent=protocol['checkpoint_keep_recent'],
            archive_validation_rows=protocol['archive_evaluation_rows'])
        phase = 'history_sealing'
        history = merge_epoch_history(parent_metrics, Path(fit['run_dir']) / 'metrics.jsonl',
            directory / 'completed-epochs.jsonl', declaration['stop_epoch'])
        phase = 'milestone_evaluation'
        model, _ = load_model(Path(fit['checkpoint']), device, expected_spec=data[0].spec)
        rec, prior = evaluate(model, data, checkpoint, base, directory / 'prior', device)
        phase = 'paired_comparison'
        base_root = Path(protocol['base_campaign_root'])
        original = read(base_root / 'gate/original.json')
        comparison = compare_panels(base_root / 'gate/original-prior', directory / 'prior')
        summary = {**declaration, 'status': 'completed_final_endpoint' if fit['endpoint_completed'] else 'completed_epoch_segment',
            'training_completed': True, 'fit': fit, 'history': history, 'parent': resume, 'prefix_lineage': lineage,
            'reconstruction': rec, 'prior': prior, 'vs_original': comparison,
            'source_archive_sha256': archive, 'scope': 'Predeclared milestone;100 is the sole final endpoint',
            'elapsed_seconds': time.monotonic() - started}
        if declaration['coefficient'] != 0:
            control_path = root / f"seed{declaration['seed']}-a0" / directory.name / 'summary.json'
            control = read(control_path) if control_path.is_file() else {}
            if control.get('status') in ('completed_epoch_segment', 'completed_final_endpoint'):
                if control['fit']['terminal_rng'] != fit['terminal_rng'] or control['fit']['encoder_sha256'] != fit['encoder_sha256']:
                    raise ValueError('Same-epoch control/candidate draw or encoder parity failed')
                summary['vs_control'] = compare_panels(control_path.parent / 'prior', directory / 'prior')
                summary['advantage_assessment'] = assess_advantage(original['reconstruction'], control['reconstruction'],
                    rec, comparison, summary['vs_control'])
            else:
                summary['matched_control_status'] = 'unavailable; no matched advantage claim'
        # Optional10epoch evaluation may fail independently of a good prefix.
        try:
            old = read(base_root / case_label / 'summary.json')
            if old.get('status') == 'completed':
                summary['vs_epoch10'] = compare_panels(base_root / case_label / 'final-prior', directory / 'prior')
            else:
                summary['epoch10_comparison_status'] = 'evaluation unavailable; see explicit prefix lineage'
        except Exception as error:
            summary['epoch10_comparison_error'] = repr(error)
        phase = 'lossless_panel_storage'
        summary['panel_row_storage'] = archive_panel_rows(directory / 'prior')
        write_json(directory / 'summary.json', summary)
    except Exception as error:
        write_json(directory / 'summary.json', {**declaration, 'status': 'failed_or_time_censored',
            'failure_phase': phase, 'training_completed': fit is not None, 'fit': fit, 'history': history,
            'prefix_lineage': lineage,
            'reconstruction': rec, 'prior': prior, 'error': repr(error),
            'run_dir': fit['run_dir'] if fit is not None else getattr(error, 'experiment_run_dir', None),
            'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        if model is not None:
            del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--stage', choices=('gate', 'segment'), required=True)
    parser.add_argument('--segment', type=int, choices=range(16))
    args = parser.parse_args()
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise RuntimeError('Long density numerical work requires immutable DTU allocation')
    protocol = read_long_protocol(args.protocol, args.sha256)
    base = authenticate_base(protocol)
    check_unchanged_scientific_modules(protocol, Path.cwd())
    if args.stage == 'gate':
        native_gate(protocol, base, args.output_root, Path.cwd(), 'cuda')
    else:
        if args.segment is None:
            parser.error('Segment stage requires a declared index')
        run_segment(protocol, base, args.output_root, Path.cwd(), args.segment, 'cuda')


if __name__ == '__main__':
    main()
