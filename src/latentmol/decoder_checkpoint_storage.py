"""Lossless panel-row storage and explicit retention of NEW run checkpoints.

Scientific arrays, panel metrics and original byte hashes stay unchanged.
Existing experiments are never pruned. Restore archived rows temporarily
before calling the unchanged paired evaluator, then remove only that copy.
"""
from contextlib import contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from latentmol.experiments import file_sha256, write_json


def _stream_hash(handle):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def archive_panel_rows(directory):
    """Verify lossless gzip before removing the bulky raw rows; keep every draw."""
    directory = Path(directory).resolve(strict=True)
    raw, zipped, receipt = [directory / name for name in ('rows.jsonl', 'rows.jsonl.gz', 'row-storage.json')]
    if raw.is_symlink() or zipped.is_symlink() or receipt.is_symlink():
        raise ValueError('Panel storage does not follow artifact symlinks')
    metrics = json.loads((directory / 'metrics.json').read_text())
    expected = metrics['artifacts']['rows.jsonl']['sha256']
    if file_sha256(raw) != expected:
        raise ValueError('Raw panel rows differ from evaluated artifact')
    with raw.open('rb') as source, gzip.open(zipped, 'xb', compresslevel=6) as output:
        shutil.copyfileobj(source, output, 1024 * 1024)
    with gzip.open(zipped, 'rb') as restored:
        if _stream_hash(restored) != expected:
            raise ValueError('Panel gzip changed original bytes; raw retained')
    record = {'version': 1, 'raw': 'rows.jsonl', 'compressed': 'rows.jsonl.gz',
        'raw_sha256': expected, 'compressed_sha256': file_sha256(zipped),
        'raw_bytes': raw.stat().st_size, 'compressed_bytes': zipped.stat().st_size,
        'scope': 'Lossless storage only; all hard draws/denominators preserved'}
    write_json(receipt, record)
    raw.unlink()
    return record


@contextmanager
def materialized_panel_rows(directory):
    """Restore authenticated bytes for existing evaluator, preserve old raw files."""
    directory = Path(directory).resolve(strict=True)
    raw = directory / 'rows.jsonl'
    if raw.is_symlink():
        raise ValueError('Panel rows must not be a symlink')
    created = False
    try:
        if not raw.exists():
            record = json.loads((directory / 'row-storage.json').read_text())
            expected = json.loads((directory / 'metrics.json').read_text())['artifacts']['rows.jsonl']['sha256']
            if (record.get('version') != 1 or record['raw'] != 'rows.jsonl'
                    or record['compressed'] != 'rows.jsonl.gz' or record['raw_sha256'] != expected):
                raise ValueError('Archived panel identity/path differs')
            zipped = directory / 'rows.jsonl.gz'
            if zipped.is_symlink() or file_sha256(zipped) != record['compressed_sha256']:
                raise ValueError('Archived panel bytes changed')
            with gzip.open(zipped, 'rb') as source, raw.open('xb') as output:
                created = True
                shutil.copyfileobj(source, output, 1024 * 1024)
            if file_sha256(raw) != expected:
                raise ValueError('Restored panel differs from original evaluated bytes')
        yield directory
    finally:
        if created:
            raw.unlink()


def prune_owned_checkpoints(run_directory, epoch, milestones, keep_recent):
    """Prune ONLY newly generated files in this run after validated epoch save.

    Keep milestone checkpoints and at least two recent completed boundaries.
    Older runs/parents are never traversed. Deleted identities are retained.
    """
    milestones = list(milestones)
    if (type(epoch) is not int or epoch < 1 or type(keep_recent) is not int or keep_recent < 2
            or any(type(item) is not int or item < 1 for item in milestones)
            or len(set(milestones)) != len(milestones)):
        raise ValueError('Retention needs a completed epoch, distinct milestones and at least two recent boundaries')
    directory = Path(run_directory)
    if directory.is_symlink():
        raise ValueError('Checkpoint run directory must not be a symlink')
    root = directory.resolve(strict=True)
    keep = set(milestones) | set(range(max(1, epoch - keep_recent + 1), epoch + 1))
    deleted, paths = [], []
    for path in sorted(root.glob('epoch-*.pt')):
        match = re.fullmatch(r'epoch-([0-9]+)\.pt', path.name)
        if match is None or path.is_symlink() or path.resolve().parent != root:
            raise ValueError('Checkpoint retention refuses unexpected path/symlink')
        number = int(match[1])
        if number <= epoch and number not in keep:
            deleted.append({'file': path.name, 'sha256': file_sha256(path), 'bytes': path.stat().st_size})
            paths.append(path)
    journal = root / 'checkpoint-retention.jsonl'
    if journal.is_symlink():
        raise ValueError('Checkpoint retention journal must not be a symlink')
    # Persist the identities BEFORE deletion. Quota errors must fail without
    # losing checkpoints or their hashes; prevalidate every path first too.
    with journal.open('a') as handle:
        handle.write(json.dumps({'completed_epoch': epoch, 'keep_milestones': sorted(milestones),
            'keep_recent': keep_recent, 'planned_deletions': deleted}, allow_nan=False) + '\n')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    for path in paths:
        path.unlink()
    with journal.open('a') as handle:
        handle.write(json.dumps({'completed_epoch': epoch, 'deleted': deleted}, allow_nan=False) + '\n')
    return deleted
