"""Extract dependency seeds from HPC metadata already saved inside Mosaic."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
INPUT = REPO / 'results/hpc/20261024-storage-offload-01'
PREFIX = '/work3/s234818/latentmol'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    output = HERE / 'latentmol-closure-seeds.json'
    require(not output.exists(), 'Preserve earlier closure seed inventory')
    raw_plan = (INPUT / 'active-vae-plan.json').read_bytes()
    raw_shared = (INPUT / 'shared-account-metadata/stdout.bin').read_bytes()
    plan = json.loads(raw_plan)
    shared = json.loads(raw_shared)
    require(shared['read_only'] is True and len(shared['active_plans']) == 1,
            'Unexpected active plan metadata scope')
    active_path, active = next(iter(shared['active_plans'].items()))
    require(active_path == PREFIX + '/submissions/source138-vae-cli-startup-fix-attempt001/source138-vae-cli-startup-fix-attempt001.json',
            'Actual active VAE plan path differs')
    delivered = active['text'].encode()
    require(len(delivered) == active['bytes'] and sha(delivered) == active['sha256']
            and json.loads(delivered) == plan, 'Separately saved active plan differs from collected raw metadata')
    refs = {}

    def add(path, field, digest=None):
        require(path == PREFIX or path.startswith(PREFIX + '/'), 'Foreign path in seed extraction')
        p = PurePosixPath(path)
        require('..' not in p.parts, 'Unsafe seed path')
        row = refs.setdefault(path, {'path': path, 'declared_by': []})
        row['declared_by'].append(field)
        if digest:
            require(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Malformed declared hash')
            require(row.get('sha256', digest) == digest, 'Conflicting seed hashes')
            row['sha256'] = digest

    def walk(value, field='$'):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.startswith(PREFIX + '/'):
                    add(key, field + '[absolute-key]', child if isinstance(child, str) else None)
                if isinstance(child, str) and child.startswith(PREFIX):
                    digest = value.get(key + '_sha256')
                    if key in {'path', 'remote', 'module'}:
                        digest = value.get('sha256', digest)
                    walk(child, field + '.' + key)
                    if digest:
                        add(child, field + '.' + key, digest)
                else:
                    walk(child, field + '.' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, field + '[' + str(index) + ']')
        elif isinstance(value, str) and (value == PREFIX or value.startswith(PREFIX + '/')):
            add(value, field)

    walk(plan)
    add(active_path, '$actual_active_plan', active['sha256'])
    # Directories are protected scopes, not single files for download.
    directory_keys = {'$.campaign', '$.data', '$.release', '$.parent_release',
                      '$.speed_suite_root', '$.work_root'}
    fetch = []
    for path, row in sorted(refs.items()):
        row['declared_by'] = sorted(set(row['declared_by']))
        row['must_remain_remote_until_closure_complete'] = True
        row['classification'] = ('directory_scope' if directory_keys & set(row['declared_by'])
                                 else 'large_checkpoint_protect_metadata_only' if path.endswith('.pt')
                                 else 'exact_declared_file')
        if row['classification'] == 'exact_declared_file':
            fetch.append({**row, 'mode': 'lstat_then_download_regular_file_if_bounded',
                          'maximum_bytes_per_file': 8 * 1024 ** 2,
                          'on_oversize': 'Preserve as unresolved protected dependency; do not truncate or infer no references'})
    # du timed out. Exact names are useful for the next metadata inspection,
    # but neither folder age nor these partial sizes demonstrate inactivity.
    du = shared['du']
    candidates = []
    for size, path in re.findall(r'(?m)^(\d+)\t(' + re.escape(PREFIX) + r'/runs/202609[^\r\n]+)$', du.get('stdout', '')):
        candidates.append({'path': path, 'observed_du_value': int(size),
                           'inactive_established': False, 'eligible_for_deletion': False,
                           'metadata_first': ['run.json'],
                           'metadata_rule': 'Fetch existing root JSON/manifests/catalogs/input or output descriptors and a no-follow basename/size inventory; do not assume run.json exists or import checkpoint bytes.'})
    release_discovery = [
        {'directory': plan['release'], 'mode': 'list_top_level_regular_metadata_only',
         'expected_manifest_sha256': plan['manifest_sha256'],
         'reason': 'Actual sealed source138 manifest filename is not stated in the delivered plan; discover the real manifest rather than guessing it.'},
        {'directory': plan['parent_release'], 'mode': 'list_top_level_regular_metadata_only',
         'expected_manifest_sha256': plan['parent_manifest_sha256'],
         'reason': 'Original parent35 source/provenance is protected; inspect its declared manifest and code consumers.'},
    ]
    result = {
        'version': 'latentmol-active-closure-seeds-v1',
        'status': 'incomplete_closure_no_september_inactivity_or_delete_pass',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': [{'path': 'results/hpc/20261024-storage-offload-01/active-vae-plan.json',
                    'bytes': len(raw_plan), 'sha256': sha(raw_plan)},
                   {'path': 'results/hpc/20261024-storage-offload-01/shared-account-metadata/stdout.bin',
                    'bytes': len(raw_shared), 'sha256': sha(raw_shared)}],
        'actual_active_vae_plan': {'path': active_path, 'sha256': active['sha256']},
        'active_plan_absolute_reference_seeds': sorted(refs.values(), key=lambda row: row['path']),
        'small_metadata_fetch_requests': fetch,
        'release_manifest_discovery_requests': release_discovery,
        'active_embedding_closure': {'available_in_inputs': False,
            'required_next': 'From authenticated current embedding submission/argv, fetch its exact plan, binding/adoption manifest, current run control and sealed source manifest. Preserve source/env/data/checkpoint/context/catalog scopes until then.',
            'do_not_guess_plan_path_from_historical_names': True},
        'strong_protection_roots': [PREFIX + '/env', PREFIX + '/data', PREFIX + '/cache',
            PREFIX + '/raw', PREFIX + '/boltz-preparation', plan['campaign'], plan['release'],
            plan['speed_suite_root'], str(PurePosixPath(plan['parent_checkpoint']).parent)],
        'special_protection': [
            {'path': plan['starting_checkpoint']['path'], 'sha256': plan['starting_checkpoint']['sha256'],
             'reason': 'Authentic original36 shared by VAE continuation and frozen embedding; no checkpoint deserialization in audit.'},
            {'path': plan['parent_checkpoint'], 'sha256': plan['parent_checkpoint_sha256'],
             'reason': 'Original35 occupancy/reference anchor; lineage inputs remain protected.'},
            {'scope': 'Every source056 prepared context and original source050/50-molecule catalog dependency',
             'exact_paths_available': False, 'preserve_all_until_positive_closure': True,
             'reason': 'Parent explicitly requires protection. Names/paths must come from fetched embedding/data/preparation manifests; no guessed directory may serve as a substitute.'}],
        'september_candidate_metadata_only': sorted(candidates, key=lambda row: row['path']),
        'du_returncode': du['returncode'], 'du_complete': du['returncode'] == 0,
        'du_unit_or_physical_savings_inferred': False,
        'direct_vae_plan_september_references': [path for path in refs if '/runs/202609' in path],
        'closure_algorithm': [
            'Keep blanket latentmol protection while this closure is incomplete. Seed with all delivered VAE absolute references AND the missing actual embedding plan/source/control.',
            'Read only bounded regular JSON/JSONL/YAML/TOML manifests, catalogs, run records, bindings and receipt descriptors; verify declared hashes before following each absolute reference. If a metadata file is too large, malformed or absent, retain its scope unresolved instead of treating it unused.',
            'For paths in dataset manifests, original35/36 run records/completions, source135 adoption/failure_readiness and embedding preparation/catalog manifests, recursively follow dependency/proof/input/output paths and relative paths resolved against their real documented owning root.',
            'Inspect the active sealed controller/loader/readiness modules named by actual source manifests to capture computed/default/glob paths that a simple JSON string search misses; fetching a manifest alone does not prove closure.',
            'For each September candidate, inspect existing run.json plus root metadata/catalog/manifest inventory. Reject deletion if it intersects any resolved protected path or symlink/hardlink target, active command/open file, future continuation/resume input, prepared context or original catalog.',
            'Only parent may certify a candidate inactive after fresh actual all-job/scope checks, full laptop download, every-file hashes and final race-safe identical-byte comparison. Preserve exact unresolved references and never infer inactivity from age, a missing live bjobs record or partial duRC124.'
        ],
        'no_dbmol_checkout_accessed': True, 'no_ssh': True, 'no_numerical_execution': True,
        'no_remote_mutation': True, 'delete_allowlist': []}
    with output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(result, handle, indent=2, sort_keys=True); handle.write('\n')
    print(json.dumps({'path': str(output), 'sha256': sha(output.read_bytes()),
                      'absolute_seeds': len(refs), 'bounded_metadata_requests': len(fetch),
                      'partial_du_september_names': len(candidates), 'september_inactive': False}))


if __name__ == '__main__':
    main()
