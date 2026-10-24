"""Reconcile completed cleanup03 and root's fresh read-only existence proof."""
from datetime import datetime, timezone
import copy
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def read(path):
    return json.loads(Path(path).read_bytes())


def record(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest}


def main():
    original = HERE / 'existing-backup-inventory.json'
    original_proof = record(original)
    assert original_proof['sha256'] == 'af1c810dafa9ea5ef17af82e8687a6ab171b1a7a1fbb008f6b0b5c90a7dcb169'
    value = copy.deepcopy(read(original))
    completion_path = REPO / 'experiments/brd4-jq1/fixed-lrreset/runs/20261022-brd4-jq1-25-fixed-lrreset-03/cleanup-completion.json'
    plan_path = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-03/cleanup-plan.json'
    journal_path = plan_path.parent / 'evidence01/unpacked/cleanup/journal.jsonl'
    completion = read(completion_path)
    assert completion['status'] == 'completed'
    assert record(plan_path)['sha256'] == completion['plan_sha256'] == 'bfe10342b9d8378f7c5842ca3a2eb87806a46cdb08501d9e0813f81776ca7e9f'
    assert record(journal_path)['sha256'] == completion['journal_sha256'] == 'c1e82fc70a1a80d71c0d8bc8ed0ac380f5a962a033c7c6c21763434fbd3ed756'
    planned = {item['remote_path']: item for item in read(plan_path)['files']}
    removed = {}
    for line in journal_path.read_text().splitlines():
        item = json.loads(line)
        if item['status'] == 'removed_exact_backed_up_copy':
            assert item['path'] not in removed
            assert (item['bytes'], item['sha256']) == (planned[item['path']]['bytes'], planned[item['path']]['sha256'])
            removed[item['path']] = item
    assert len(removed) == completion['removed_files'] == 1212
    assert sum(item['bytes'] for item in removed.values()) == completion['removed_logical_bytes'] == 852317966

    observation_folder = REPO / 'results/hpc/20261024-storage-offload-01/duplicate-existence-01'
    observed, result = read(observation_folder / 'stdout.bin'), read(observation_folder / 'result.json')
    assert result['returncode'] == 0 and result['read_only'] is True
    assert record(observation_folder / 'stdout.bin')['sha256'] == result['stdout_sha256']
    assert record(observation_folder / 'stderr.bin')['sha256'] == result['stderr_sha256']
    assert observed['read_only'] is True and not observed['changed']
    matching = {row['path']: row for row in observed['matching']}
    absent = set(observed['absent'])
    assert len(matching) == 6 and len(absent) == 1210 and not matching.keys() & absent
    assert sum(row['bytes'] for row in matching.values()) == 4778979
    covered = set()
    for archive in value['archives']:
        previous = archive['member_inventory']
        assert record(previous['path']) == previous
        target = HERE / ('backup-members-reconciled02-' + archive['id'] + '.jsonl.gz')
        removed_count = removed_bytes = candidate_count = candidate_bytes = actual_count = actual_bytes = unknown_count = unknown_bytes = 0
        with gzip.open(previous['path'], 'rt') as source, target.open('xb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as destination:
                for line in source:
                    row = json.loads(line); path = row['remote_path']
                    if path in removed:
                        assert (row['bytes'], row['sha256']) == (removed[path]['bytes'], removed[path]['sha256'])
                        row['previously_removed_exact_receipt'] = True
                        row['potential_offload_candidate_after_remote_dependency_and_hash_check'] = False
                    if row['previously_removed_exact_receipt']:
                        removed_count += 1; removed_bytes += row['bytes']
                    if path in matching:
                        assert (row['bytes'], row['sha256']) == (matching[path]['bytes'], matching[path]['sha256'])
                        row['current_remote_observation'] = {'status': 'exists_exact_bytes', 'utc': observed['utc'], 'proof': str(observation_folder)}
                        row['remote_existence_established_now'] = True; covered.add(path)
                    elif path in absent:
                        row['current_remote_observation'] = {'status': 'absent', 'utc': observed['utc'], 'proof': str(observation_folder)}
                        row['remote_existence_established_now'] = True; covered.add(path)
                    else:
                        row['current_remote_observation'] = {'status': 'not_inspected_by_this_observation'}
                    if row['potential_offload_candidate_after_remote_dependency_and_hash_check']:
                        candidate_count += 1; candidate_bytes += row['bytes']
                        if path in matching: actual_count += 1; actual_bytes += row['bytes']
                        elif path not in absent: unknown_count += 1; unknown_bytes += row['bytes']
                    destination.write((json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode())
        archive.update(member_inventory=record(target), previous_member_inventory=previous,
            already_removed_member_files=removed_count, already_removed_logical_bytes=removed_bytes,
            potential_saved_or_report_member_files=candidate_count, potential_saved_or_report_member_bytes=candidate_bytes,
            actually_existing_candidate_files_in_root_observation=actual_count,
            actually_existing_candidate_bytes_in_root_observation=actual_bytes,
            uninspected_potential_candidate_files=unknown_count, uninspected_potential_candidate_bytes=unknown_bytes)
        if archive['remote_archive_path'] in removed:
            row = removed[archive['remote_archive_path']]
            assert (row['bytes'], row['sha256']) == (archive['local_archive']['bytes'], archive['local_archive']['sha256'])
            archive['remote_archive_previously_removed'] = True
        archive['remote_archive_current_observation'] = 'not_inspected; prior exact removal receipt applies if flagged'
    assert covered == set(matching) | absent
    value.update(version='local-exact-backup-offload-audit-v2-reconciled',
        reconciled_utc=datetime.now(timezone.utc).isoformat(), previous_inventory_preserved=original_proof,
        additional_prior_cleanup={'completion': record(completion_path), 'plan': record(plan_path), 'journal': record(journal_path),
                                 'removed_files': len(removed), 'removed_logical_bytes': 852317966},
        prior_removed_unique_files=value['prior_removed_unique_files'] + len(removed),
        prior_removed_unique_logical_bytes=value['prior_removed_unique_logical_bytes'] + 852317966,
        root_read_only_current_observation={'stdout': record(observation_folder / 'stdout.bin'),
            'result': record(observation_folder / 'result.json'), 'utc': observed['utc'],
            'inspected_files': 1216, 'matching_files': 6, 'matching_logical_bytes': 4778979,
            'absent_files': 1210, 'changed_files': 0,
            'scope': 'Only the two higher-rate report export trees were inspected; home archives, lower-rate report and original summaries remain uninspected.'})
    destination = HERE / 'existing-backup-inventory-02.json'
    with destination.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
    lines = ['# Reconciled exact local backup audit', '',
        'Revision02 preserves the original audit and adds the completed22October report retirement journal plus the root operator\'s24October read-only observation.', '',
        '**The two higher-rate report exports are already retired except six small files totaling4,778,979bytes. Their tar copies also have exact prior removal receipts. No storage gain is claimed for those absent copies.**', '',
        '| Backup | Previously removed member bytes | Existing matching candidate bytes actually observed | Candidate bytes still uninspected |',
        '| --- | ---: | ---: | ---: |']
    for item in value['archives']:
        lines.append(f"| {item['id']} | {item['already_removed_logical_bytes']} | {item['actually_existing_candidate_bytes_in_root_observation']} | {item['uninspected_potential_candidate_bytes']} |")
    lines += ['', 'Home archives, old native snapshots and the lower-rate report are exact local backup mappings only; their remote existence/inactivity/current dependencies have not been established. Report exports do not back original NPZ/JSONL bytes. No remote mutation was performed by this audit.']
    with (HERE / 'existing-backup-inventory-02.md').open('x', encoding='utf8') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(record(destination)))


if __name__ == '__main__':
    main()
