"""Metadata health gates for serial, exact large-triangle VAE continuation.

No tensor/model imports or scheduler mutations belong here. Successful metadata
checks do not replace the sealed trainer's checkpoint/Adam/RNG compatibility
checks. Reconstruction/validity need not improve monotonically: that would add
an unrequested scientific early-stopping policy while KL is still warming up.
"""
from __future__ import annotations

import json
import math


def strict_json(text):
    """Reject ambiguous keys and nonstandard NaN/Infinity, including overflow."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    finite_tree(value)
    return value


def finite_tree(value, label='root'):
    """Optional conditional summaries may be None; actual numbers must be finite."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite telemetry: ' + label)
    if isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, label + '.' + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_tree(item, label + '[' + str(index) + ']')


def number(value, label, *, lower=None, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Missing/nonfinite numeric field: ' + label)
    if (lower is not None and value < lower) or (positive and value <= 0):
        raise ValueError('Out-of-range numeric field: ' + label)
    return value


def validate_health(run, rows, latest, *, previous_epoch, expected_job, identity,
                    config, protocol, runtime, expected_resume=None):
    """Check a completed allocation's entire epoch log, not just its last row.

    Paths/checkpoint byte hashes and scheduler evidence are checked by the I/O
    controller before calling this pure function. Values returned here are an
    engineering continuation decision, not a chemical-quality certificate.
    """
    finite_tree(run)
    finite_tree(rows)
    finite_tree(latest)
    epoch = run.get('completed_epoch')
    if (run.get('status') != 'completed' or type(epoch) is not int
            or not previous_epoch < epoch <= 100 or run.get('planned_epochs') != 100
            or run.get('environment', {}).get('job_id') != expected_job
            or run.get('source', {}).get('sha256') != identity['source_sha256']
            or run.get('source_archive_sha256') != identity['source_archive_sha256']
            or run.get('config') != config or run.get('runtime_identity') != runtime):
        raise ValueError('Completed run/epoch/job/source/config/runtime identity differs')
    manifest = run.get('dataset_manifest', {})
    if (manifest.get('dataset_sha256') != identity['dataset_sha256']
            or manifest.get('split_sha256') != identity['split_sha256']
            or run.get('training_molecules') != protocol['requested_counts']['train']
            or run.get('parameter_gate') != protocol['expected_parameters']
            or run.get('large_corpus_protocol') != protocol
            or run.get('joint_encoder_decoder_training') is not True
            or run.get('extra_random_prior_training_points') != 0
            or run.get('chemistry_training_weight') != 0):
        raise ValueError('Dataset/training scope/architecture differs')
    complete = epoch == 100
    reason = 'planned_horizon' if complete else 'wall_budget_epoch_boundary'
    if run.get('planned_horizon_complete') is not complete or run.get('stop_reason') != reason:
        raise ValueError('Allocation did not end at a declared healthy epoch boundary')
    resume = run.get('resume')
    if previous_epoch == 0:
        if resume is not None or expected_resume is not None:
            raise ValueError('First allocation unexpectedly resumed another checkpoint')
    elif (not isinstance(resume, dict) or expected_resume is None
          or any(resume.get(key) != value for key, value in expected_resume.items())
          or resume.get('completed_epoch') != previous_epoch):
        raise ValueError('Continuation parent checkpoint/epoch differs')
    if not rows or [row.get('epoch') for row in rows] != list(range(previous_epoch + 1, epoch + 1)):
        raise ValueError('Epoch log is missing, duplicated, reordered or non-progressing')
    if {k: v for k, v in rows[-1].items() if k != 'at'} != latest:
        raise ValueError('Latest epoch differs from the final complete metric event')
    updates_per_epoch = math.ceil(protocol['requested_counts']['train'] / config['batch_size'])
    updates = previous_epoch * updates_per_epoch
    for row in rows:
        if type(row.get('epoch_updates')) is not int or row['epoch_updates'] != updates_per_epoch:
            raise ValueError('Epoch has incomplete or unexpected optimizer update count')
        updates += updates_per_epoch
        if type(row.get('updates')) is not int or row['updates'] != updates:
            raise ValueError('Cumulative optimizer updates differ')
        for name in ('seconds', 'seconds_including_extended_diagnostics', 'training_seconds',
                     'training_molecules_per_second', 'training_updates_per_second', 'learning_rate'):
            number(row.get(name), name, positive=True)
        for name in ('evaluation_seconds', 'checkpoint_seconds', 'beta', 'gradient_norm_max',
                     'gradient_norm_mean_per_update', 'cumulative_training_seconds',
                     'cumulative_evaluation_seconds'):
            number(row.get(name), name, lower=0)
        if (type(row.get('clipped_updates')) is not int or not 0 <= row['clipped_updates'] <= updates_per_epoch
                or row.get('gradient_clipping_rate') != row['clipped_updates'] / updates_per_epoch
                or row.get('training_latent_policy') != 'posterior_sample'
                or row.get('training_kl_used_in_objective') is not True):
            raise ValueError('Clipping or VAE objective telemetry is inconsistent')
        for group in ('train', 'val', 'train_probe', 'val_probe'):
            metrics = row.get(group, {})
            for key in ('atom_ce', 'bond_ce', 'charge_ce', 'reconstruction'):
                number(metrics.get(key), group + '.' + key, lower=0)
            if group != 'train':
                if (not 0 <= number(metrics.get('validity'), group + '.validity') <= 1
                        or not 0 <= number(metrics.get('exact_reconstruction'), group + '.exact') <= 1
                        or number(metrics.get('chemistry_samples'), group + '.samples', positive=True) < 1):
                    raise ValueError('Invalid reconstruction cohort telemetry')
        number(row['train'].get('kl'), 'train.kl', lower=0)
        number(row['train'].get('total'), 'train.total', lower=0)
        extended = row.get('extended_diagnostics', {})
        for key in ('peak_allocated_vram_bytes_training_and_standard_validation',
                    'peak_reserved_vram_bytes_training_and_standard_validation'):
            number(extended.get(key), key, positive=True)
        host = extended.get('peak_host_resident_bytes_process_lifetime')
        if host is not None:
            number(host, 'peak_host_resident_bytes_process_lifetime', positive=True)
        number(extended.get('pre_clip_gradient_diagnostics_observations'), 'gradient observations', positive=True)
        for group in ('encoder', 'decoder', 'triangle'):
            number(extended.get('pre_clip_group_gradient_means', {}).get(group), group + ' gradient', lower=0)
        for index in range(config['model']['refinement_rounds']):
            for group in ('atoms', 'pairs'):
                name = f'round{index}/{group}'
                number(extended.get('sparse_activation_absmax', {}).get(name), name, lower=0)
        number(extended.get('posterior_mu_absmax'), 'posterior_mu_absmax', lower=0)
        low = number(extended.get('posterior_logvar_min'), 'posterior_logvar_min')
        high = number(extended.get('posterior_logvar_max'), 'posterior_logvar_max')
        if low > high:
            raise ValueError('Posterior log-variance bounds are reversed')
        due = row['epoch'] % protocol['monitor_every'] == 0 or row['epoch'] in protocol['deep_monitor_epochs']
        if due:
            panel = extended.get('panel', {})
            if panel.get('split') != 'val' or not isinstance(panel.get('reconstruction'), dict):
                raise ValueError('Scheduled validation monitor is missing')
            if row['epoch'] in protocol['deep_monitor_epochs'] and any(
                    key not in panel for key in ('prior', 'sampled_reconstruction', 'neighborhood', 'interpolation')):
                raise ValueError('Scheduled deep monitor is incomplete')
            number(extended.get('panel_seconds'), 'panel_seconds', positive=True)
    if run.get('updates') != updates:
        raise ValueError('Run/update log disagree')
    return {'status': 'passed', 'from_epoch': previous_epoch, 'through_epoch': epoch,
            'completed_epochs_in_allocation': len(rows), 'updates': updates,
            'clipped_updates_in_allocation': sum(row['clipped_updates'] for row in rows),
            'last_validation_reconstruction': rows[-1]['val']['reconstruction'],
            'last_validation_validity': rows[-1]['val']['validity'],
            'stop_reason': reason, 'next_action': 'final' if complete else 'segment',
            'scope': 'Metadata/finite-progress gates; no monotonic-quality early stopping or checkpoint tensor certification'}
