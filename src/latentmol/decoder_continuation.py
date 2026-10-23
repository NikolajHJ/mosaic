"""Metadata-only epoch-history and resume envelopes for allocated training.

Completed validation receipts, not checkpoint filenames, define a boundary.
Histories are copied into new immutable files; segment records stay intact.
"""
import json
from pathlib import Path

from latentmol.experiments import file_sha256


def merge_epoch_history(parent_metrics, current_metrics, destination, expected_epoch):
    """Copy contiguous completed epochs1..N without modifying either input.

    Scientific quota/RNG checks are additionally enforced by restore_training.
    A duplicate/gap or checkpoint lacking its completed metric is rejected.
    """
    if type(expected_epoch) is not int or expected_epoch < 1:
        raise ValueError("Expected completed epoch must be a positive integer")
    payload = b''
    for path in (parent_metrics, current_metrics):
        if path is not None:
            for line in Path(path).read_bytes().splitlines():
                if line.strip():
                    payload += line + b'\n'
    rows = [json.loads(line) for line in payload.splitlines()]
    if [row.get('epoch') for row in rows] != list(range(1, expected_epoch + 1)):
        raise ValueError("Epoch history has gaps, overlaps or an incomplete boundary")
    destination = Path(destination)
    with destination.open('xb') as handle:
        handle.write(payload)
    return {'path': str(destination), 'sha256': file_sha256(destination),
            'completed_epochs': expected_epoch}


def resume_envelope(fit, metrics, archive_sha256):
    """Bind checkpoint, completed run and full history to their exact bytes."""
    run = Path(fit['run_dir'])
    record = json.loads((run / 'run.json').read_text())
    if (record.get('status') != 'completed' or record.get('source_archive_sha256') != archive_sha256
            or file_sha256(Path(fit['checkpoint'])) != fit['sha256']):
        raise ValueError("Parent training/checkpoint is not authenticated and completed")
    return {'checkpoint': fit['checkpoint'], 'checkpoint_sha256': fit['sha256'],
        'parent_record': str(run / 'run.json'), 'parent_record_sha256': file_sha256(run / 'run.json'),
        'parent_metrics': str(metrics), 'parent_metrics_sha256': file_sha256(Path(metrics)),
        'parent_archive_sha256': archive_sha256}
