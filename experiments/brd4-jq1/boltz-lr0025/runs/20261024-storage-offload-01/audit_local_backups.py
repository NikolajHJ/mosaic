"""Read-only local byte audit. No remote connections or scientific operations."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
WORK = '/work3/s234818/mosaic/runs/'
HOME = '/zhome/95/e/206121/mosaic-runs/'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def evidence(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def existing_cleanup():
    paths = [REPO / 'results/hpc/20261021-mosaic-storage-cleanup-01/cleanup-audit-02/attempt-01/deletion-journal.jsonl',
             REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-02/native-delivery-01/cleanup/attempt-02/deletion-journal.jsonl']
    removed, journals = {}, []
    for path in paths:
        count = total = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['event'] == 'removed':
                removed[row['path']] = row
                count += 1; total += row['bytes']
        journals.append({**evidence(path), 'removed_files': count, 'removed_logical_bytes': total})
    return removed, journals


def declared_archives():
    records = []
    for number in (1, 2):
        verification = REPO / f'scripts/hpc/receipts/20261022-brd4-jq1-lrboost-analysis-01/report-delivery-verification-{number:04d}.json'
        value = read(verification)
        assert value['status'] == 'passed'
        records.append({'id': f'higher-rate-report-{number:04d}',
            'archive': REPO / f'results/hpc/20261022-brd4-jq1-lrboost-analysis-01/interim-{number:04d}.tar.gz',
            'pin': value['report_archive'], 'base': WORK, 'verification': verification})
    folder = REPO / 'results/hpc/20261023-brd4-jq1-lrreset-analysis-03/delivery-interim-0001'
    assert read(folder / 'independent-verification.json')['status'] == 'passed'
    records.append({'id': 'lower-rate-report-0001',
        'archive': folder.parent / 'interim-0001.tar.gz', 'pin': read(folder / 'archive-receipt.json'),
        'base': WORK, 'verification': folder / 'independent-verification.json'})
    for folder in sorted((REPO / 'results/hpc/20261021-brd4-jq1-25-report').glob('edition-*')):
        for archive in sorted(folder.glob('interim-*.tar.gz')):
            receipt = archive.with_name(archive.name.removesuffix('.gz') + '.receipt.json')
            records.append({'id': 'original-jq1-' + folder.name, 'archive': archive,
                'pin': read(receipt), 'base': HOME, 'verification': folder / 'verification.json'})
    historical = [
        ('historical-brd4-best-primal', '20261017-brd4-parallel/complete-report-01', '20261017-brd4-best-primal-report-02', 'study-verification.json'),
        ('historical-brd4-delayed-chemistry', '20261019-brd4-delayed-chemistry-report/edition-02', '20261019-brd4-chemistry-ramp-01-report', 'pair-verification.json'),
        ('historical-brd4-chemistry', '20261019-brd4-chemistry/complete-report-01', '20261019-brd4-chemistry-05-repackage', 'study-verification.json')]
    for key, subfolder, name, verification in historical:
        folder = REPO / 'results/hpc' / subfolder
        records.append({'id': key, 'archive': folder / (name + '.tar.gz'),
            'pin': read(folder / (name + '.tar.receipt.json')), 'base': HOME,
            'verification': folder / verification})
    return records


def classification(name):
    if '/payload/report/' in name and '/pytest-' not in name:
        return 'report_export_exact_bytes_only'
    if '/payload/study/' in name and '/pytest-' not in name and '/case-' in name:
        if '/snapshots/' in name and name.endswith('.npz'):
            return 'original_saved_npz_exact_bytes'
        if name.endswith(('/states.jsonl', '/updates.jsonl', '/report.json')):
            return 'original_saved_case_metadata_exact_bytes'
    if '/payload/study/' in name and '/pytest-' not in name and name.endswith('/summary.json'):
        return 'original_saved_summary_exact_bytes'
    return 'other_archived_proof_or_fixture_exact_bytes'


def audit_archive(record, removed):
    archive, pin = record['archive'], record['pin']
    assert archive.is_file() and archive.stat().st_size == pin['bytes']
    archive_sha = sha(archive)
    assert archive_sha == pin['sha256'], ('Archive digest differs', archive)
    observed, manifest = {}, None
    with tarfile.open(archive, 'r|gz') as handle:
        for member in handle:
            name = member.name
            pure = PurePosixPath(name)
            assert not pure.is_absolute() and '..' not in pure.parts and '\\' not in name and ':' not in name
            assert pure.as_posix() == name and name not in observed
            assert member.isfile(), ('Nonregular archive member', name)
            stream = handle.extractfile(member)
            if name == 'evidence_manifest.json':
                data = stream.read(); manifest = json.loads(data)
                observed[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            else:
                observed[name] = {'bytes': member.size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    assert manifest and isinstance(manifest['files'], dict)
    expected = manifest['files']
    assert set(observed) == set(expected) | {'evidence_manifest.json'}
    assert len(expected) == pin['files']
    for name, item in expected.items():
        assert observed[name] == {'bytes': item['bytes'], 'sha256': item['sha256']}, ('Member differs', name)
    totals, counts = {}, {}
    candidate_count = candidate_bytes = removed_count = removed_bytes = 0
    inventory = HERE / ('backup-members-' + record['id'] + '.jsonl.gz')
    with inventory.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
            for name, item in sorted(expected.items()):
                remote = record['base'] + name
                category = classification(name)
                previously_removed = remote in removed
                if previously_removed:
                    assert removed[remote]['bytes'] == item['bytes'] and removed[remote]['sha256'] == item['sha256']
                    removed_count += 1; removed_bytes += item['bytes']
                eligible_kind = category != 'other_archived_proof_or_fixture_exact_bytes'
                if eligible_kind and not previously_removed:
                    candidate_count += 1; candidate_bytes += item['bytes']
                totals[category] = totals.get(category, 0) + item['bytes']
                counts[category] = counts.get(category, 0) + 1
                row = {'remote_path': remote, **item, 'local_archive': str(archive.resolve()), 'archive_member': name,
                       'classification': category, 'previously_removed_exact_receipt': previously_removed,
                       'potential_offload_candidate_after_remote_dependency_and_hash_check': eligible_kind and not previously_removed,
                       'remote_existence_established_now': False, 'deletion_approved_by_this_audit': False}
                compressed.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
    return {'id': record['id'], 'status': 'complete_archive_and_every_member_rehashed_passed',
        'local_archive': evidence(archive), 'remote_archive_path': pin['path'],
        'remote_archive_existing_now': False, 'remote_archive_previously_removed': pin['path'] in removed,
        'member_inventory': evidence(inventory), 'original_delivery_verification': evidence(record['verification']),
        'embedded_manifest_sha256': observed['evidence_manifest.json']['sha256'],
        'listed_files': len(expected), 'all_safe_members': len(observed), 'logical_member_bytes': sum(v['bytes'] for v in expected.values()),
        'category_files': counts, 'category_bytes': totals,
        'already_removed_member_files': removed_count, 'already_removed_logical_bytes': removed_bytes,
        'potential_saved_or_report_member_files': candidate_count, 'potential_saved_or_report_member_bytes': candidate_bytes,
        'scope': 'Exact bytes in named archive members only; report export JSON does not back native NPZ/JSONL or another path unless its exact SHA matches.'}


def exact_original_summary(removed):
    folder = REPO / 'results/hpc/20261021-brd4-jq1-25-report/protenix-completion-01'
    path = folder / 'native-summary.json'; value = evidence(path)
    assert value['bytes'] == 297952692 and value['sha256'] == '942b260ea5b1b60271c818e67dc4169c646cb1216d10baeab0d76c20bb3ee0ee'
    assessment = read(folder / 'assessment.json')
    assert assessment['status'] == 'passed' and assessment['native_job'] == '29770179'
    proof = next(x for x in assessment['proofs'] if x['sha256'] == value['sha256'])
    return {'local': value, 'remote_path': proof['path'], 'assessment': evidence(folder / 'assessment.json'),
            'already_removed': proof['path'] in removed, 'remote_existence_established_now': False,
            'scope': 'Only original summary.json bytes; no case JSON/JSONL or NPZ backup is implied.'}


def main():
    removed, journals = existing_cleanup()
    result = {'version': 'local-exact-backup-offload-audit-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Local stdlib byte/hash/tar/JSON only; no SSH/deletion/numerics. Remote existence, inactivity and dependency exclusions must be authenticated by the root operator before any removal.',
        'prior_cleanup_journals': journals, 'prior_removed_unique_files': len(removed),
        'prior_removed_unique_logical_bytes': sum(x['bytes'] for x in removed.values()), 'archives': [], 'errors': []}
    for record in declared_archives():
        try:
            value = audit_archive(record, removed); result['archives'].append(value)
            print(json.dumps({k: value[k] for k in ('id', 'status', 'logical_member_bytes', 'potential_saved_or_report_member_bytes')}), flush=True)
        except Exception as error:
            result['errors'].append({'id': record['id'], 'archive': str(record['archive']), 'error': repr(error)})
            print(json.dumps(result['errors'][-1]), flush=True)
    result['exact_single_original_summary'] = exact_original_summary(removed)
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    result['status'] = 'passed' if not result['errors'] else 'partial_preserving_errors'
    with (HERE / 'existing-backup-inventory.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write('\n')
    lines = ['# Exact local backup audit', '', result['scope'], '',
        '| Backup | Files | Exact member bytes | Potential saved/report bytes before remote checks |',
        '| --- | ---: | ---: | ---: |']
    for item in result['archives']:
        lines.append(f"| {item['id']} | {item['listed_files']} | {item['logical_member_bytes']} | {item['potential_saved_or_report_member_bytes']} |")
    lines += ['', 'Previous cleanup already removed ' + str(result['prior_removed_unique_logical_bytes']) + ' logical bytes; those exact paths are excluded.',
        '', 'Each compressed JSONL inventory gives exact remote path, archive/member, length and SHA. These are candidate mappings, not a claim the remote copies still exist or can be removed without checking current dependencies.',
        '', 'Report probability JSON, plots and exported chunk assets are backed only as those report files. They do not back original NPZ or JSONL. The separately listed original Protenix summary is a single exact original file backup.']
    with (HERE / 'existing-backup-inventory.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
