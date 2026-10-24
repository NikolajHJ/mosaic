"""Stdlib-only independent audit of the fresh full-source/no-jobs failure bundle."""
import ast
import base64
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
ROOT = BASE + '/protenix-extended-03'
SOURCE = BASE + '/protenix-extended-release-03/source'
MANIFEST = 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562'
ARCHIVE = 'bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef'
GIB = 1024 ** 3


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    output = HERE / 'preflight-failure-verification-02.json'
    require(not output.exists(), 'Preserve earlier verification output')
    folder = REPO / 'results/hpc/20261024-protenix-extended-03/evidence-02'
    stdout = (folder / 'stdout.bin').read_bytes()
    stderr = (folder / 'stderr.bin').read_bytes()
    result = read(folder / 'result.json')
    require(result == {'returncode': 0, 'remote_mutations': False,
                       'stdout_sha256': sha(stdout), 'stderr_sha256': sha(stderr)},
            'Fresh read-only collection failed or byte hashes differ')

    # Authenticate the exact successful command, including its full per-member
    # remote hash and read-only checks, without importing or executing it here.
    collector = HERE / 'collect_pro03.py'
    collector_tree = ast.parse(collector.read_text(encoding='utf8'))
    remote_values = [ast.literal_eval(node.value) for node in collector_tree.body
                     if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'REMOTE' for t in node.targets)]
    require(len(remote_values) == 1, 'Collector remote command not unique')
    remote = remote_values[0]
    ast.parse(remote)
    command = ' '.join(shlex.quote(x) for x in
                       ['/work3/s234818/mosaic/envs/protenix-tiny/bin/python', '-c', remote])
    expected_args = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                     '-o', 'ConnectionAttempts=1', 'dtu-hpc',
                     'bash -l -c ' + shlex.quote(command)]
    started = read(folder / 'started.json')
    require(started['args'] == expected_args, 'Executed collector command differs')
    require("for rel,digest in files.items():assert sha(source/rel)==digest" in remote
            and "assert not p.is_symlink()" in remote
            and "not(p.stat().st_mode&0o222)" in remote
            and "==set(files)|{'source_manifest.json'}" in remote,
            'Remote full seal checks missing')
    calls = [node for node in ast.walk(ast.parse(remote)) if isinstance(node, ast.Call)]
    require(not any(isinstance(node.func, ast.Attribute)
                    and node.func.attr in {'write_text', 'write_bytes', 'mkdir', 'unlink',
                                          'rmdir', 'rename', 'replace', 'chmod'} for node in calls),
            'Collector unexpectedly mutates remote files')
    process_calls = [node for node in calls if isinstance(node.func, ast.Attribute)
                     and isinstance(node.func.value, ast.Name)
                     and node.func.value.id == 'subprocess']
    require(len(process_calls) == 1 and process_calls[0].func.attr == 'run'
            and isinstance(process_calls[0].args[0], ast.List)
            and [ast.literal_eval(n) for n in process_calls[0].args[0].elts[:2]] == ['bjobs', '-l'],
            'Collector subprocess scope exceeds read-only scheduler observation')

    raw = json.loads(stdout)
    metadata = read(folder / 'metadata.json')
    expected_paths = {ROOT + '/staging-01/started.json',
                      ROOT + '/staging-01/storage-preflight/gate.json',
                      SOURCE + '/source_manifest.json'}
    require(set(raw['files']) == set(metadata['files']) == expected_paths,
            'Unexpected files or omitted early-failure proofs')
    decoded = {}
    for remote_path, item in raw['files'].items():
        data = base64.b64decode(item['base64'], validate=True)
        require(set(item) == {'bytes', 'sha256', 'base64'}
                and type(item['bytes']) is int and len(data) == item['bytes']
                and sha(data) == item['sha256'], 'Raw delivered file hash differs')
        saved = metadata['files'][remote_path]
        local = PurePosixPath(saved['local_path'])
        require(not local.is_absolute() and '..' not in local.parts
                and local.parts[0] == 'files' and len(local.parts) == 3,
                'Unsafe extracted evidence path')
        local_path = folder.joinpath(*local.parts)
        require(not local_path.is_symlink() and local_path.read_bytes() == data
                and saved == {'bytes': len(data), 'sha256': sha(data),
                              'local_path': local.as_posix()}, 'Local evidence differs')
        decoded[remote_path] = json.loads(data)
        item.pop('base64')
        item['local_path'] = local.as_posix()
    require(raw == metadata, 'Decoded metadata differs from authenticated stdout')
    require(metadata['roots'] == {ROOT: {
        'jobs': {}, 'jobs_file_present': False,
        'absence_scope': 'No jobs are inferred when staging/preflight failed before jobs.json'}}
        and metadata['scheduler'] == {} and metadata['external_files'] == {},
        'Jobs, scheduler outcomes or large scientific files were fabricated')
    require(metadata['releases'] == {SOURCE: {'manifest_sha256': MANIFEST,
        'source_files': 643, 'all_members_rehashed': True, 'readonly': True}},
        'Remote full-source verification did not complete')

    archive_path = REPO / 'results/hpc/20261024-protenix-extended-03/source01/source.tar.gz'
    require(archive_path.stat().st_size == 39644558 and sha(archive_path.read_bytes()) == ARCHIVE,
            'Sealed archive changed')
    members = {}
    with tarfile.open(archive_path) as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            require(member.isfile() and path.parts[0] == 'source' and len(path.parts) > 1
                    and '..' not in path.parts and not path.is_absolute()
                    and member.name not in members and not member.mode & 0o222,
                    'Unsafe source archive member')
            members[member.name] = tar.extractfile(member).read()
    require(len(members) == 644, 'Source archive member count differs')
    manifest_bytes = members.pop('source/source_manifest.json')
    manifest = json.loads(manifest_bytes)
    require(sha(manifest_bytes) == MANIFEST
            and manifest == decoded[SOURCE + '/source_manifest.json']
            and len(manifest['files']) == 643
            and set(members) == {'source/' + rel for rel in manifest['files']}
            and all(sha(members['source/' + rel]) == digest
                    for rel, digest in manifest['files'].items()),
            'Full local source archive and remotely rehashed manifest differ')
    require(manifest['git_revision_reference'] == 'c74c659344123937608077e3e4078f75f09cd743',
            'Authenticated provenance field differs')

    old_stdout = read(REPO / 'results/hpc/20261024-protenix-extended-03/failed-preflight-evidence-01/stdout.bin')
    for path in expected_paths - {SOURCE + '/source_manifest.json'}:
        earlier = old_stdout['files'][path]
        require(earlier['sha256'] == metadata['files'][path]['sha256']
                and earlier['bytes'] == metadata['files'][path]['bytes'],
                'Original failed staging/gate proof was overwritten')
    attempt = decoded[ROOT + '/staging-01/started.json']
    original_proofs = read(REPO / 'results/hpc/20261024-boltz-startup-recovery-02/peer-projection-inspection/stdout.bin')
    prefix = BASE + '/protenix-extended-01/'
    pins = {path.removeprefix(prefix): {'bytes': item['bytes'], 'sha256': item['sha256']}
            for path, item in original_proofs['files'].items()}
    require(attempt['source_manifest_sha256'] == MANIFEST
            and attempt['old_source_manifest_sha256'] == 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00'
            and attempt['old_failure_proofs'] == pins and len(pins) == 12
            and attempt['continuation_snapshot_sha256'] == 'edf15f06ba1b6591d62cbc3e04e19ecd93b553a0627da38c370fdeed3c1b780f',
            'Old failed native and exact continuation provenance differs')

    gate = decoded[ROOT + '/staging-01/storage-preflight/gate.json']
    work = gate['work3']
    require(gate['status'] == work['status'] == 'failed' and gate['phase'] == 'preflight'
            and gate['full_long_admission'] is False and gate['missing_full_projections'] == []
            and gate['reserve_bytes'] == work['reserve_bytes'] == 4 * GIB
            and gate['file_reserve'] == work['file_reserve'] == 100000,
            'Original failed admission/reserves changed')
    require(gate['own_current'] == {'random': 0, 'continuation': 0}
            and gate['full_extended'] == {'random': GIB, 'continuation': GIB},
            'Pro validation allowance mislabeled full production projection')
    original_full = {'protenix': 1493521824, 'boltz2': 1748021824, 'boltz2_chemistry': 5144021824}
    require(set(gate['old_current']) == set(original_full)
            and all(type(gate['old_current'][arm]) is int
                    and 0 <= gate['old_current'][arm] <= full for arm, full in original_full.items()),
            'Original remaining accounting changed')
    original_remaining = sum(full - gate['old_current'][arm] for arm, full in original_full.items())
    peers = gate['peers']
    require(set(peers) == {'baseline', 'dbmol'}
            and peers['baseline']['source_manifest_sha256'] == 'c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6'
            and peers['dbmol']['source_manifest_sha256'] == '600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed'
            and all(type(p['full']) is int and type(p['current']) is int
                    and 0 <= p['current'] <= p['full'] and p['remaining'] == p['full'] - p['current']
                    for p in peers.values()), 'Full peer projection accounting differs')
    controls = gate['prospective_adoption_metadata_controls']
    require(controls['per_control_bytes'] == GIB and controls['remaining_bytes'] == 2 * GIB
            and controls['current'] == {BASE + '/boltz-pro02-adoption-04': 0,
                                       BASE + '/dbmol-pro02-adoption-06': 0},
            'Future metadata control allowances changed')
    inventory = gate['noisy_capsule_inventory']
    require(inventory['unknown_scopes'] == []
            and inventory['stale_bytes'] == sum(p['bytes'] for p in inventory['capsules'])
            and all(p['scope'] == 'authenticated_noisy01' for p in inventory['capsules']),
            'Unregistered noisy scope admitted')
    noisy = gate['noisy_protenix']
    require(noisy['full'] == GIB and 0 <= noisy['current'] <= GIB
            and noisy['remaining'] == GIB - noisy['current']
            and noisy['scope'] == 'authenticated_immutable_one_GiB_cap', 'Noisy bound differs')
    projected = (original_remaining + 2 * GIB + sum(p['remaining'] for p in peers.values())
                 + 2 * GIB + noisy['remaining'] + 2 * GIB + 512 * 1024 ** 2)
    require(gate['projected_bytes'] == work['projected_bytes'] == projected == 37348701956
            and work['required_bytes'] == projected + 4 * GIB == 41643669252,
            'Projection or reserve arithmetic differs')
    rows = re.findall(r's234818\|306121\|\|\s*([0-9.]+)\s+GiB\|\s*([0-9.]+)\s+GiB\|\|\s*(\d+)\|\s*(\d+)', work['raw_quota'])
    require(len(rows) == 1, 'Raw finite quota row missing/ambiguous')
    used, hard, used_files, hard_files = rows[0]
    used_upper = int(Decimal(used).to_integral_value(rounding=ROUND_CEILING)) * GIB
    hard_lower = int(Decimal(hard).to_integral_value(rounding=ROUND_FLOOR)) * GIB
    remaining_files = int(hard_files) - int(used_files)
    require(work['quota']['finite_pools'] == [{'used_bytes_upper': used_upper,
        'hard_bytes_lower': hard_lower, 'remaining_bytes': hard_lower - used_upper,
        'remaining_files': remaining_files}]
        and work['quota']['available_bytes_lower'] == hard_lower - used_upper
        and work['available_bytes_conservative'] == min(hard_lower - used_upper - work['stale_bytes_subtracted'],
                                                        work['disk_free_bytes']) == 27472908456
        and work['available_files_lower'] == work['quota']['available_files_lower'] == remaining_files >= 100000,
        'Conservative quota rounding/accounting differs')
    require(work['filesystem_inode_observation'] == {'f_files': 0, 'f_ffree': 0, 'f_favail': 0}
            and work['filesystem_inode_reporting'] == 'unreported_all_zero',
            'Unreported physical FS inode observation overclaimed')
    shortfall = work['required_bytes'] - work['available_bytes_conservative']
    require(shortfall == 14170760796, 'Actual resource shortfall changed')

    audit = {'status': 'verified_pre_cpu_storage_failure_full_remote_source',
        'verified_utc': datetime.now(timezone.utc).isoformat(),
        'captured_utc': metadata['captured_utc'], 'source_manifest_sha256': MANIFEST,
        'source_archive_sha256': ARCHIVE, 'source_archive_bytes': 39644558,
        'local_source_members_verified': 643, 'remote_source_members_rehashed': 643,
        'remote_source_readonly': True, 'remote_source_member_bytes_downloaded': False,
        'delivered_files_verified': 3, 'collector_sha256': sha(collector.read_bytes()),
        'executed_remote_command_sha256': sha(remote.encode()),
        'delivery_stdout_sha256': sha(stdout), 'delivery_stderr_sha256': sha(stderr),
        'gate_sha256': metadata['files'][ROOT + '/staging-01/storage-preflight/gate.json']['sha256'],
        'old_twelve_failure_pins_unchanged': True, 'failed_gate_unchanged': True,
        'jobs_file_present': False, 'jobs': {}, 'cpu_submitted': False,
        'gpu_startup_passed': False, 'native_passed': False, 'optimization_started': False,
        'required_bytes': work['required_bytes'], 'available_bytes_conservative': work['available_bytes_conservative'],
        'shortfall_bytes': shortfall, 'reserve_bytes': 4 * GIB, 'file_reserve': 100000,
        'peer_full_projections_retained': True, 'new_pro_validation_allowance_not_full_projection': True,
        'original_failures_and_saved_results_preserved': True, 'no_local_scientific_execution': True,
        'capacity_limitation': 'Actual mandatory failed admission; no safe verified inactive/duplicate cleanup subset sufficient to resolve it has been identified. Aggregate du is not a proof that all possible deletions are insufficient.',
        'scope': 'Read-only remote full-seal execution authenticated from exact command/RC0; independent local byte/JSON/schema and integer-budget audit only. No scientific readiness or execution pass.'}
    with output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(audit, handle, indent=2, sort_keys=True); handle.write('\n')
    campaign = read(HERE / 'campaign.json')
    require(campaign['status'] == 'pre_cpu_storage_blocked' and campaign['gate'] == gate
            and campaign['jobs'] == {} and campaign['jobs_file_present'] is False,
            'Canonical actual outcome changed unexpectedly')
    campaign['full_remote_source_failure_verification'] = {
        'path': str(output), 'sha256': sha(output.read_bytes())}
    campaign['remote_source_members_rehashed'] = 643
    campaign['remote_source_readonly'] = True
    campaign['current_evidence_captured_utc'] = metadata['captured_utc']
    with (HERE / 'campaign.json').open('w', encoding='utf8', newline='\n') as handle:
        json.dump(campaign, handle, indent=2, sort_keys=True); handle.write('\n')
    with (HERE / 'run-record.md').open('a', encoding='utf8', newline='\n') as handle:
        handle.write('\nFresh read-only evidence at ' + metadata['captured_utc']
            + ' rehashed all643 remote source members and verified the complete seal read-only; the downloaded manifest matches the full local sealed archive. '
            'The same original failed gate is unchanged, and jobs.json remains absent: no CPU, GPU startup, new native pass or optimization was submitted. '
            'The independent second audit authenticates the exact executed collector, RC0 and all3 compact proof files. '
            'No safe verified inactive/duplicate cleanup subset sufficient to resolve the actual admission shortfall has been identified; aggregate du is not a proof about all possible deletions. '
            'No files were deleted by this audit.\n')
    print(json.dumps({'status': audit['status'], 'receipt': str(output),
                      'sha256': sha(output.read_bytes()), 'shortfall_bytes': shortfall}))


if __name__ == '__main__':
    main()
