"""Verify an exact streamed backup using only bytes, tar metadata and JSON.

No extraction, NumPy/model imports, network connection or deletion is provided.
The root operator supplies the separately authenticated remote receipt hash.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile

PLAN_VERSION = 'mosaic-storage-offload-plan-v1'
STREAM_VERSION = 'mosaic-storage-offload-stream-v1'
VERIFICATION_VERSION = 'mosaic-storage-offload-local-verification-v1'
MANIFEST_MEMBER = 'offload-manifest.json'
MAX_PLAN_BYTES = 64 * 1024 ** 2
REMOTE_ROOTS = (PurePosixPath('/work3/s234818/mosaic'), PurePosixPath('/zhome/95/e/206121/mosaic-runs'))
LEGACY_RUNS_ROOT = PurePosixPath('/work3/s234818/latentmol/runs')
CLEANUP_ROOT = PurePosixPath('/work3/s234818/mosaic/runs/20261024-storage-offload-01')
FORBIDDEN_COMPONENTS = frozenset(('env', 'envs', 'cache', 'caches', 'data', 'datasets', 'source', 'sources', 'proof', 'proofs'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_record(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked local proof')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest}


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda name: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant: ' + name)))


def sha_string(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def absolute_path(value):
    require(isinstance(value, str) and value and '\\' not in value and '\n' not in value and '\r' not in value,
            'Invalid absolute remote path')
    path = PurePosixPath(value)
    require(path.is_absolute() and path.as_posix() == value and '..' not in path.parts
            and not any(character in value for character in '*?[]\x00'), 'Noncanonical remote path')
    return path


def permitted_candidate_root(root):
    """Only named Mosaic roots or explicitly shaped legacy run descendants."""
    if any(root != anchor and root.is_relative_to(anchor) for anchor in REMOTE_ROOTS):
        return True
    if root == LEGACY_RUNS_ROOT or not root.is_relative_to(LEGACY_RUNS_ROOT):
        return False
    first = root.relative_to(LEGACY_RUNS_ROOT).parts[0]
    if re.fullmatch(r'source-?[0-9]{3}[A-Za-z0-9]*-[A-Za-z0-9][A-Za-z0-9._-]*', first):
        return True
    dated = re.fullmatch(r'(20[0-9]{6})(T[0-9]{6}Z)?(?:[-_][A-Za-z0-9][A-Za-z0-9._-]*)?', first)
    if dated is None:
        return False
    try:
        value = dated.group(1) + (dated.group(2) or '')
        datetime.strptime(value, '%Y%m%dT%H%M%SZ' if dated.group(2) else '%Y%m%d')
    except ValueError:
        return False
    return True


def load_plan(raw, expected_sha):
    """Independent client-side shape check, never a remote inactivity decision."""
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_PLAN_BYTES
            and sha_string(expected_sha) and hash_bytes(raw) == expected_sha, 'Approved plan bytes changed')
    plan = strict_json(raw)
    require(plan['version'] == PLAN_VERSION and plan['status'] == 'approved_archive_then_remove', 'Unapproved plan scope')
    require(isinstance(plan['operation_id'], str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}', plan['operation_id']),
            'Invalid operation identity')
    cleanup = absolute_path(plan['cleanup_root'])
    require(cleanup == CLEANUP_ROOT, 'Cleanup receipts must remain inside the exact declared Mosaic control root')
    require(type(plan['timeout_seconds']) is int and 1 <= plan['timeout_seconds'] <= 3600, 'Invalid transfer timeout')
    roots = [absolute_path(value) for value in plan['approved_candidate_roots']]
    require(roots and len(roots) == len(set(roots)) and all(len(value.parts) >= 4 for value in roots), 'Missing/duplicate candidate roots')
    require(all(permitted_candidate_root(root)
                and root != cleanup and not root.is_relative_to(cleanup)
                and not set(part.casefold() for part in root.parts) & FORBIDDEN_COMPONENTS for root in roots),
            'Candidate root outside declared Mosaic scope or protected component')
    protection = plan['protection_manifest']
    protection_path = absolute_path(protection['path'])
    require(protection_path != cleanup and protection_path.is_relative_to(cleanup),
            'Protection manifest escapes cleanup control root')
    require(sha_string(protection['sha256']), 'Missing current protection manifest hash')
    ids = plan['audited_active_job_ids']
    require(isinstance(ids, list) and len(ids) == len(set(ids))
            and all(isinstance(job, str) and re.fullmatch(r'\d+', job) for job in ids), 'Invalid audited job IDs')
    owners = plan['candidate_owner_job_ids']
    require(isinstance(owners, list) and len(owners) == len(set(owners))
            and all(isinstance(job, str) and re.fullmatch(r'\d+', job) for job in owners), 'Missing candidate owner-job audit')
    require(not set(ids) & set(owners), 'A candidate owner is active in the reviewed plan')
    rows = plan['files']
    require(isinstance(rows, list) and rows, 'No exact original file records')
    paths, members = set(), set()
    for index, row in enumerate(rows):
        path = absolute_path(row['path'])
        require(any(path != root and path.is_relative_to(root) for root in roots)
                and path != cleanup and not path.is_relative_to(cleanup)
                and not set(part.casefold() for part in path.parts[:-1]) & FORBIDDEN_COMPONENTS,
                'Original file escapes approved roots/protected components')
        require(row['path'] not in paths and row['member'] not in members, 'Duplicate original/member path')
        require(row['member'] == 'files/' + format(index, '08d'), 'Member numbering/scope differs')
        require(type(row['bytes']) is int and row['bytes'] >= 0 and sha_string(row['sha256']), 'Missing exact original length/SHA')
        require(type(row['device']) is int and row['device'] >= 0
                and type(row['inode']) is int and row['inode'] >= 0
                and type(row['mtime_ns']) is int and row['mtime_ns'] >= 0, 'Missing original lstat identity')
        paths.add(row['path']); members.add(row['member'])
    return plan


def validate_stream_receipt(receipt, plan, raw, archive_record):
    require(receipt['version'] == STREAM_VERSION and receipt['status'] == 'streamed_complete', 'Remote stream not complete')
    require(receipt['operation_id'] == plan['operation_id'] and receipt['plan_sha256'] == hash_bytes(raw)
            and receipt['plan_bytes'] == len(raw), 'Remote plan binding differs')
    require(receipt['archive_bytes'] == archive_record['bytes']
            and receipt['archive_sha256'] == archive_record['sha256'], 'Remote compressed stream length/hash differs')
    require(receipt['manifest_member'] == MANIFEST_MEMBER and receipt['member_count'] == len(plan['files']) + 1
            and receipt['files_count'] == len(plan['files'])
            and receipt['logical_bytes'] == sum(row['bytes'] for row in plan['files'])
            and receipt['all_files_verified'] is True, 'Remote original member coverage differs')
    require(not receipt.get('deleted', False) and not receipt.get('scientific_passed', False),
            'Streaming receipt must not claim deletion or science')


def verify_archive(archive_path, raw, expected_plan_sha, receipt):
    plan = load_plan(raw, expected_plan_sha)
    before = Path(archive_path).stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    archive_record = file_record(archive_path)
    validate_stream_receipt(receipt, plan, raw, archive_record)
    expected = {row['member']: row for row in plan['files']}
    seen = set(); verified = []; manifest_seen = False
    with gzip.open(archive_path, 'rb') as decoded, tarfile.open(fileobj=decoded, mode='r|', bufsize=512) as archive:
        for index, member in enumerate(archive):
            name = member.name; path = PurePosixPath(name)
            require(not path.is_absolute() and path.as_posix() == name and '..' not in path.parts
                    and '\\' not in name and ':' not in name and name not in seen,
                    'Unsafe/duplicate archive member')
            require(member.isfile() and not member.issparse() and not member.linkname,
                    'Only complete regular archive files are accepted')
            seen.add(name)
            source = archive.extractfile(member)
            if index == 0:
                require(name == MANIFEST_MEMBER and member.size == len(raw), 'Original manifest must be the first exact member')
                require(source.read() == raw, 'Original manifest bytes differ from approved plan')
                manifest_seen = True
                continue
            require(name in expected, 'Undeclared archive member')
            require(index <= len(plan['files']) and name == plan['files'][index - 1]['member'],
                    'Original numbered member order differs')
            row = expected[name]
            require(member.size == row['bytes'], 'Original member length differs')
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
            require(digest == row['sha256'], 'Original member bytes differ')
            verified.append({'path': row['path'], 'member': name, 'bytes': member.size, 'sha256': digest})
        # Tar iteration stops at its first zero header. Consume the complete
        # remaining gzip stream to validate its CRC/EOF and reject hidden
        # appended tar members instead of silently ignoring them.
        trailing = 0
        limit = tarfile.RECORDSIZE + 512
        while True:
            block = archive.fileobj.read(min(4096, limit + 1 - trailing))
            if not block:
                break
            require(not any(block), 'Nonzero data after the declared tar end')
            trailing += len(block)
            require(trailing <= limit, 'Excessive zero data after declared tar end')
        require(trailing >= 512, 'Incomplete tar end-of-archive markers')
    require(manifest_seen and seen == set(expected) | {MANIFEST_MEMBER}
            and len(verified) == len(expected), 'Incomplete original member inventory')
    after = Path(archive_path).stat()
    require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) == identity,
            'Local archive changed while being verified')
    return {'version': VERIFICATION_VERSION, 'status': 'verified_all_members', 'operation_id': plan['operation_id'],
            'plan_sha256': hash_bytes(raw), 'plan_bytes': len(raw), 'archive_sha256': archive_record['sha256'],
            'archive_bytes': archive_record['bytes'], 'archive': archive_record,
            'manifest_member': MANIFEST_MEMBER, 'member_count': len(seen), 'files_count': len(verified),
            'logical_bytes': sum(item['bytes'] for item in verified), 'all_members_verified': True,
            'unexpected_members': [], 'missing_members': [], 'files': verified,
            'deleted': False, 'scientific_passed': False,
            'scope': 'Every original archive byte/member verified against exact approved plan and separately supplied remote completion receipt; no extraction, array decoding, scientific evaluation or deletion.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--remote-receipt', type=Path, required=True)
    parser.add_argument('--remote-receipt-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not os.path.lexists(args.output), 'Existing verification outcome; never overwrite')
    raw = args.plan.read_bytes()
    receipt_proof = file_record(args.remote_receipt)
    require(receipt_proof['sha256'] == args.remote_receipt_sha256, 'Authenticated remote receipt bytes changed')
    result = verify_archive(args.archive, raw, args.plan_sha256, strict_json(args.remote_receipt.read_bytes()))
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(), remote_stream_receipt=receipt_proof,
                  approved_plan=file_record(args.plan))
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({key: result[key] for key in ('status', 'operation_id', 'files_count', 'logical_bytes', 'archive_sha256')}))


if __name__ == '__main__':
    main()
