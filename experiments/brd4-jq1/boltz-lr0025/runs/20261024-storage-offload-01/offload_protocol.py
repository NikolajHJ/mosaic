"""Exact-file archive/remove contracts. Standard library, no scientific work.

The plan and flat protection manifest are authored and approved by the parent
operator. This module never discovers deletion candidates or broadens roots.
"""
from __future__ import annotations

import fnmatch
from datetime import datetime
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import time

CLEANUP_ROOT = Path('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
MOSAIC_ROOTS = (Path('/work3/s234818/mosaic'), Path('/zhome/95/e/206121/mosaic-runs'))
LEGACY_RUNS_ROOT = Path('/work3/s234818/latentmol/runs')
FORBIDDEN_COMPONENTS = frozenset(('env', 'envs', 'cache', 'caches', 'data', 'datasets', 'source', 'sources', 'proof', 'proofs'))
MANIFEST_MEMBER = 'offload-manifest.json'
PLAN_VERSION = 'mosaic-storage-offload-plan-v1'
STREAM_VERSION = 'mosaic-storage-offload-stream-v1'
VERIFY_VERSION = 'mosaic-storage-offload-local-verification-v1'
MAX_PLAN_BYTES = 64 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError('Nonfinite JSON constant ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def canonical_path(value):
    require(isinstance(value, str) and value.startswith('/') and '\\' not in value
            and '\x00' not in value and '\n' not in value and '\r' not in value,
            'Only explicit canonical absolute POSIX paths are permitted')
    path = PurePosixPath(value)
    require(str(path) == value and '..' not in path.parts and '.' not in path.parts
            and not any(c in value for c in '*?[]'), 'Noncanonical or globbed path')
    return Path(value)


def below(path, root):
    return path != root and root in path.parents


def permitted_candidate_root(root):
    """Only Mosaic or explicit dated/source legacy runs; never a shared tree."""
    if any(below(root, anchor) for anchor in MOSAIC_ROOTS):
        return True
    if not below(root, LEGACY_RUNS_ROOT):
        return False
    direct_run = root.relative_to(LEGACY_RUNS_ROOT).parts[0]
    if re.fullmatch(r'source-?[0-9]{3}[A-Za-z0-9]*-[A-Za-z0-9][A-Za-z0-9._-]*', direct_run):
        return True
    if re.fullmatch(r'20[0-9]{6}(?:[-_][A-Za-z0-9][A-Za-z0-9._-]*)?', direct_run) is None:
        return False
    try:
        datetime.strptime(direct_run[:8], '%Y%m%d')
    except ValueError:
        return False
    return True


def nofollow_parent(path):
    """Open the literal parent by directory descriptors; no linked ancestor."""
    path = canonical_path(str(path))
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open('/', flags)
    try:
        for part in path.parts[1:-1]:
            new = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = new
        return fd, path.name
    except BaseException:
        os.close(fd)
        raise


def metadata(st):
    return {'device': st.st_dev, 'inode': st.st_ino, 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns}


def checked_open(record):
    path = canonical_path(record['path'])
    parent, name = nofollow_parent(path)
    fd = None
    try:
        entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
        require(stat.S_ISREG(entry.st_mode), 'Candidate is not a regular file')
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        opened = os.fstat(fd)
        require(stat.S_ISREG(opened.st_mode) and metadata(entry) == metadata(opened)
                == {key: record[key] for key in ('device', 'inode', 'bytes', 'mtime_ns')},
                'Candidate changed before open')
        # Keep the parent descriptor until the post-read / pre-unlink checks.
        return parent, name, fd, opened
    except BaseException:
        if fd is not None:
            os.close(fd)
        os.close(parent)
        raise


def unchanged(record, parent, name, fd, before):
    after = os.fstat(fd)
    entry = os.stat(name, dir_fd=parent, follow_symlinks=False)
    literal = os.lstat(record['path'])
    expected = {key: record[key] for key in ('device', 'inode', 'bytes', 'mtime_ns')}
    require(all(stat.S_ISREG(item.st_mode) for item in (before, after, entry, literal))
            and metadata(before) == metadata(after) == metadata(entry) == metadata(literal) == expected
            and before.st_ctime_ns == after.st_ctime_ns == entry.st_ctime_ns == literal.st_ctime_ns,
            'Candidate path, inode or content metadata changed during read')


def protection_document(record):
    path = canonical_path(record['path'])
    require(below(path, CLEANUP_ROOT), 'Protection manifest must belong to cleanup control root')
    parent, name = nofollow_parent(path)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'Protection document is not a regular file')
            raw = stream.read(MAX_PLAN_BYTES + 1)
        require(len(raw) <= MAX_PLAN_BYTES and hashlib.sha256(raw).hexdigest() == record['sha256'],
                'Protection manifest identity changed')
    finally:
        os.close(parent)
    value = strict_json(raw)
    require(value['version'] == 'mosaic-storage-offload-protection-v1'
            and isinstance(value['exact'], list) and isinstance(value['subtrees'], list)
            and isinstance(value['negative'], list), 'Malformed flat protection manifest')
    for item in value['exact'] + value['subtrees']:
        canonical_path(item)
    require(all(isinstance(p, str) and p.startswith('/') and '\x00' not in p for p in value['negative']),
            'Malformed negative guard scope')
    return value


def is_protected(path, protections):
    text = str(path)
    return (text in protections['exact']
            or any(path == Path(root) or below(path, Path(root)) for root in protections['subtrees'])
            or any(fnmatch.fnmatchcase(text, pattern) for pattern in protections['negative']))


def validate_plan(raw, expected_sha256):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_PLAN_BYTES
            and re.fullmatch(r'[0-9a-f]{64}', expected_sha256)
            and hashlib.sha256(raw).hexdigest() == expected_sha256, 'Exact approved plan bytes differ')
    plan = strict_json(raw)
    require(plan['version'] == PLAN_VERSION and plan['status'] == 'approved_archive_then_remove',
            'Plan is not explicitly approved for archive-then-remove')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', plan['operation_id'])
            and plan['cleanup_root'] == str(CLEANUP_ROOT), 'Wrong operation/control scope')
    require(type(plan['timeout_seconds']) is int and 1 <= plan['timeout_seconds'] <= 3600,
            'Unbounded archive/remove operation')
    for key in ('audited_active_job_ids', 'candidate_owner_job_ids'):
        require(isinstance(plan[key], list) and len(plan[key]) == len(set(plan[key]))
                and all(isinstance(x, str) and re.fullmatch(r'\d+', x) for x in plan[key]),
                'Malformed explicit audited/owner job IDs')
    require(not set(plan['audited_active_job_ids']) & set(plan['candidate_owner_job_ids']),
            'A candidate owner is still active')
    roots = [canonical_path(x) for x in plan['approved_candidate_roots']]
    require(roots and len(roots) == len(set(roots)), 'Missing/duplicate exact candidate roots')
    for root in roots:
        require(permitted_candidate_root(root)
                and not (root == CLEANUP_ROOT or below(root, CLEANUP_ROOT)),
                'Candidate root outside Mosaic or exact dated/source legacy run scope')
        require(not set(part.casefold() for part in root.parts) & FORBIDDEN_COMPONENTS,
                'Environment/cache/data/source/proof root cannot be retired')
    protections = protection_document(plan['protection_manifest'])
    files = plan['files']
    require(isinstance(files, list) and files, 'No approved exact files')
    paths = set()
    for index, record in enumerate(files):
        path = canonical_path(record['path'])
        require(any(below(path, root) for root in roots), 'File outside approved exact candidate roots')
        require(path != CLEANUP_ROOT and not below(path, CLEANUP_ROOT),
                'Cleanup control files cannot be archived for removal')
        require(str(path) not in paths and not is_protected(path, protections), 'Duplicate or protected current dependency')
        require(not set(part.casefold() for part in path.parts[:-1]) & FORBIDDEN_COMPONENTS,
                'Environment/cache/data/source/proofs cannot be retired')
        require(record['member'] == 'files/' + format(index, '08d'), 'Archive member order/scope differs')
        require(re.fullmatch(r'[0-9a-f]{64}', record['sha256'])
                and all(type(record[k]) is int and record[k] >= 0 for k in ('device', 'inode', 'bytes', 'mtime_ns')),
                'Missing exact file identity/stat proof')
        paths.add(str(path))
    return plan


def active_jobs():
    result = subprocess.run(['bjobs', '-w', '-noheader', '-o', 'jobid stat'],
                            capture_output=True, text=True, timeout=30)
    raw = result.stdout + result.stderr
    require(result.returncode == 0, 'Fresh whole-account active scheduler query failed')
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'\s*(\d+)\s+(\S+)\s*', line)
        require(match is not None, 'Malformed active scheduler row')
        if match.group(2) not in ('DONE', 'EXIT'):
            rows.append({'job_id': match.group(1), 'status': match.group(2)})
    require(len(rows) == len({x['job_id'] for x in rows}), 'Duplicate active scheduler IDs')
    return {'query': result.args, 'raw': raw, 'jobs': rows, 'utc_unix_ns': time.time_ns()}


def check_active(plan, observation):
    ids = {x['job_id'] for x in observation['jobs']}
    require(ids <= set(plan['audited_active_job_ids']), 'An unaudited active job appeared; stop cleanup')
    require(not ids & set(plan['candidate_owner_job_ids']), 'Candidate owner became active')
    return observation


class CountingSHAWriter:
    def __init__(self, output):
        self.output, self.count, self.digest = output, 0, hashlib.sha256()

    def write(self, value):
        view = memoryview(value)
        total = 0
        while total < len(view):
            count = self.output.write(view[total:])
            require(isinstance(count, int) and count > 0, 'Archive stream write stopped')
            self.digest.update(view[total:total + count])
            total += count
            self.count += count
        return total

    def flush(self):
        self.output.flush()

    def tell(self):
        return self.count


class HashReader:
    def __init__(self, stream):
        self.stream, self.count, self.digest = stream, 0, hashlib.sha256()

    def read(self, size=-1):
        raw = self.stream.read(size)
        self.count += len(raw)
        self.digest.update(raw)
        return raw


def pack_stream(plan, raw, output):
    """No remote archive file: tar->gzip->counted stdout, with real file hashes."""
    require(strict_json(raw) == plan, 'Stream plan is not the exact embedded approved plan')
    writer = CountingSHAWriter(output)
    compressor = gzip.GzipFile(filename='', mode='wb', fileobj=writer, mtime=0)
    archive = tarfile.open(fileobj=compressor, mode='w|', format=tarfile.USTAR_FORMAT)
    completed = False
    try:
        info = tarfile.TarInfo(MANIFEST_MEMBER)
        info.size, info.mode, info.mtime = len(raw), 0o444, 0
        archive.addfile(info, io.BytesIO(raw))
        for record in plan['files']:
            parent, name, fd, before = checked_open(record)
            try:
                with os.fdopen(fd, 'rb', closefd=False) as source:
                    reader = HashReader(source)
                    info = tarfile.TarInfo(record['member'])
                    info.size, info.mode = record['bytes'], 0o444
                    info.mtime = record['mtime_ns'] // 10**9
                    archive.addfile(info, reader)
                    require(reader.count == record['bytes'] and reader.digest.hexdigest() == record['sha256'],
                            'Streamed file differs from approved hash/length')
                unchanged(record, parent, name, fd, before)
            finally:
                os.close(fd)
                os.close(parent)
        archive.close()
        compressor.close()
        writer.flush()
        completed = True
    finally:
        # On failure do not append a valid gzip trailer or block while closing
        # an interrupted pipe. Partial streams remain explicitly unverified.
        if not completed:
            archive.fileobj = None
            compressor.fileobj = None
    return {'archive_bytes': writer.count, 'archive_sha256': writer.digest.hexdigest(),
            'manifest_member': MANIFEST_MEMBER, 'member_count': len(plan['files']) + 1,
            'files_count': len(plan['files']), 'logical_bytes': sum(x['bytes'] for x in plan['files']),
            'all_files_verified': True}


def verify_delete_authority(plan, raw, completed, verified):
    require(strict_json(raw) == plan, 'Deletion plan differs from the exact backed-up approved plan')
    plan_sha = hashlib.sha256(raw).hexdigest()
    require(completed['status'] == 'streamed_complete' and completed['version'] == STREAM_VERSION
            and completed['operation_id'] == plan['operation_id'] and completed['plan_sha256'] == plan_sha
            and completed['plan_bytes'] == len(raw) and completed['all_files_verified'] is True,
            'Remote stream never completed for this exact approved plan')
    require(verified['status'] == 'verified_all_members' and verified['version'] == VERIFY_VERSION
            and verified['operation_id'] == plan['operation_id'] and verified['plan_sha256'] == plan_sha
            and verified['all_members_verified'] is True and verified['unexpected_members'] == []
            and verified['missing_members'] == [], 'Independent local full-member verification is missing')
    for key in ('archive_sha256', 'archive_bytes', 'member_count', 'files_count', 'logical_bytes'):
        require(verified[key] == completed[key], 'Local archive and remote completed stream differ')
    require(completed['files_count'] == len(plan['files']) and completed['member_count'] == len(plan['files']) + 1
            and completed['logical_bytes'] == sum(x['bytes'] for x in plan['files']), 'Backup budget/member scope differs')
    expected = [{'path': x['path'], 'member': x['member'], 'bytes': x['bytes'], 'sha256': x['sha256']}
                for x in plan['files']]
    require(verified['files'] == expected, 'Local verifier did not authenticate every original planned member')
    return True


def append_journal(stream, item):
    stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def delete_exact(plan, raw, completed, verified, journal, observe_active=active_jobs):
    """Literal unlink only; unchanged-hash proof required, changed files retained."""
    verify_delete_authority(plan, raw, completed, verified)
    check_active(plan, observe_active())
    removed, removed_bytes, changed, absent = 0, 0, 0, 0
    checked_at = time.monotonic()
    for index, record in enumerate(plan['files']):
        if index % 100 == 0 or time.monotonic() - checked_at >= 10:
            observation = check_active(plan, observe_active())
            append_journal(journal, {'event': 'active_jobs_rechecked', 'observation': observation})
            checked_at = time.monotonic()
        parent = fd = None
        unlinked = False
        try:
            parent, name, fd, before = checked_open(record)
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            require(digest == record['sha256'], 'Candidate content changed since backup')
            unchanged(record, parent, name, fd, before)
            append_journal(journal, {'event': 'unlink_intent', 'path': record['path'],
                                     'bytes': record['bytes'], 'sha256': digest})
            # Recheck the literal directory entry after the durable intent.
            unchanged(record, parent, name, fd, before)
            os.unlink(name, dir_fd=parent)
            unlinked = True
            removed += 1
            removed_bytes += record['bytes']
            append_journal(journal, {'event': 'unlinked_verified_copy', 'path': record['path'],
                                     'bytes': record['bytes'], 'sha256': digest})
        except FileNotFoundError:
            require(not unlinked, 'File unlinked but post-unlink durability is uncertain; inspect journal')
            absent += 1
            append_journal(journal, {'event': 'retained_or_already_absent', 'path': record['path'],
                                     'reason': 'Original literal path missing; no bytes claimed removed'})
        except (ValueError, OSError) as error:
            if unlinked:
                raise RuntimeError('File unlinked but journal durability is uncertain; inspect journal') from error
            changed += 1
            append_journal(journal, {'event': 'retained_changed_or_unreadable', 'path': record['path'],
                                     'reason': repr(error)})
        finally:
            if fd is not None:
                os.close(fd)
            if parent is not None:
                os.close(parent)
    check_active(plan, observe_active())
    return {'status': 'exact_deletion_completed_with_retention' if changed or absent else 'exact_deletion_completed',
            'removed_files': removed, 'removed_logical_bytes': removed_bytes,
            'retained_changed_or_unreadable': changed, 'already_absent_no_removal_claim': absent,
            'recursive_delete_used': False, 'scientific_settings_or_saved_retained_results_changed': False}
