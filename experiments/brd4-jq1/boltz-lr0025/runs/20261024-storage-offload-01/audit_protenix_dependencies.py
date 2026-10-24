"""Read local sealed metadata only; emit protection inputs, never a delete plan."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01'
PRO = REPO / 'experiments/brd4-jq1/protenix-extended-50000/runs/20261024-protenix-extended-03'
OLD = BASE + '/restart-lrreset-04'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def main():
    output = HERE / 'protenix-protection.json'
    require(not output.exists(), 'Preserve previous protection inventory')
    archive_path = REPO / 'results/hpc/20261024-protenix-extended-03/source01/source.tar.gz'
    require(archive_path.stat().st_size == 39644558
            and sha(archive_path.read_bytes()) == 'bde714a20a49308365744ab5018c06a6d68dff82c3485dc075848827a5f7a8ef',
            'Exact sealed Pro03 archive differs')
    with tarfile.open(archive_path) as tar:
        names = tar.getnames()
        require(len(names) == len(set(names)) == 644
                and all(m.isfile() and m.name.startswith('source/') and '..' not in Path(m.name).parts
                        for m in tar.getmembers()), 'Unsafe source archive')
        sealed = {name.removeprefix('source/'): tar.extractfile(name).read() for name in names}
    manifest = json.loads(sealed.pop('source_manifest.json'))
    require(len(sealed) == len(manifest['files']) == 643
            and set(sealed) == set(manifest['files'])
            and all(sha(data) == manifest['files'][name] for name, data in sealed.items()),
            'Sealed source byte map differs')
    for alias, canonical in [('scripts/extended_ops.py', 'extended_ops.py'),
                             ('scripts/extended_protocol.py', 'extended_protocol.py'),
                             ('examples/brd4_25_atoms.py', 'extended_driver.py'),
                             ('docs/extended_parent.json', 'parent.json')]:
        require(sealed[alias] == (PRO / canonical).read_bytes(), 'Reviewed canonical alias differs: ' + alias)
    parent = json.loads(sealed['docs/extended_parent.json'])
    native_path = REPO / 'results/hpc/20261022-brd4-jq1-25-fixed-lrreset-04/pro-native-evidence01/payload/work/validate-protenix/payload/study/assessment.json'
    require(sha(native_path.read_bytes()) == '9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd',
            'Pinned625 native readiness evidence differs')
    native = read(native_path)
    collector_path = REPO / 'results/hpc/20261023-brd4-jq1-lrreset-analysis-03/delivery-interim-0001/analysis/complete-protenix/payload/assessment.json'
    require(sha(collector_path.read_bytes()) == parent['proofs']['collector']['sha256'],
            'Parent collector proof differs')
    collector = read(collector_path)
    old_failure_path = REPO / 'results/hpc/20261024-boltz-startup-recovery-02/peer-projection-inspection/stdout.bin'
    failure = read(old_failure_path)
    pinned = {}

    def protect(item, reason, consumer, *, direct=True):
        path = item['path']
        require(path.startswith('/work3/s234818/mosaic/') or path.startswith('/zhome/95/e/206121/mosaic-runs/'),
                'Foreign dependency path')
        row = pinned.setdefault(path, {'path': path, 'sha256': item['sha256'], 'reasons': [],
                                       'consumers': [], 'must_remain_remote': True})
        require(row['sha256'] == item['sha256'], 'Conflicting known dependency hash')
        if 'bytes' in item:
            require(type(item['bytes']) is int and item['bytes'] >= 0, 'Invalid metadata length')
            require(row.get('bytes', item['bytes']) == item['bytes'], 'Conflicting known length')
            row['bytes'] = item['bytes']
        if reason not in row['reasons']:
            row['reasons'].append(reason)
        if consumer not in row['consumers']:
            row['consumers'].append(consumer)
        row['direct_runtime_or_guard_read'] = row.get('direct_runtime_or_guard_read', False) or direct

    protect(parent['snapshot'], 'Exact FP32 logits/probabilities/key continuation boundary',
            'Pro03 extended_ops.input_snapshot; extended_protocol.restore_logits')
    for label, value in parent['proofs'].items():
        protect(value, 'Pinned original completed Pro29792670 ' + label,
                'Pro03 extended_protocol.verify_parent, called by certify_cpu')
    protect({'path': OLD + '/validate-protenix/payload/study/assessment.json',
             'sha256': sha(native_path.read_bytes()), 'bytes': native_path.stat().st_size},
            'Original frozen context assessment', 'Pro03 startup_inputs and unchanged driver validation')
    for name, digest in native['context_files'].items():
        protect({'path': OLD + '/validate-protenix/payload/study/context/' + name, 'sha256': digest},
                'Exact original frozen context; no fresh preparation', 'Pro03 startup_inputs/driver.load_context')
    for proof in native['proofs']:
        protect(proof, 'Original readiness proof closure retained',
                'Original625/report readiness contract verifies every assessment proof')
    for proof in collector['proofs']:
        protect(proof, 'Original complete-frame collector proof closure',
                'Current saved-only report collector/terminal evidence contracts')
    for path, value in failure['files'].items():
        protect({'path': path, 'bytes': value['bytes'], 'sha256': value['sha256']},
                'Original641 pre-model failure evidence; exact staging history pins',
                'Pro03 operator authenticate_history; historical failure preservation', direct=False)
    protect({'path': '/zhome/95/e/206121/mosaic-runs/20261019-brd4-chemistry-ramp-01-readiness/payload/composition_reference.json',
             'sha256': '37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb'},
            'Frozen empirical chemistry composition reference', 'Pro03 extended_cpu.lsf copies into own CPU payload')
    # The legacy baseline-match branch remains in the immutable driver and is
    # also needed by the currently pending original-study report contract.
    protect({'path': '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-protenix-02/run-protenix/payload/study/initialization.npz',
             'sha256': '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'},
            'Original baseline initialization identity', 'Current625 saved-only report identity and declared legacy driver branch')

    # Parse only literal AST expressions from the exact protected features.py;
    # never import the numerical module to inspect its cache contract.
    constants = {}
    def literal(node):
        if isinstance(node, ast.Name):
            return constants[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return literal(node.left) + literal(node.right)
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)
    for node in ast.parse(sealed['src/mosaic/small_molecule/features.py']).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'MODEL_ALIAS', 'ASSET_REVISION', 'ASSET_SHA256'}:
                constants[name] = literal(node.value)
    require(constants['MODEL_ALIAS'] == 'protenix_tiny_default_v0.5.0'
            and len(constants['ASSET_SHA256']) == 5, 'Native Tiny asset scope differs')
    for name, digest in constants['ASSET_SHA256'].items():
        protect({'path': '/work3/s234818/mosaic/cache/mosaic/protenix/' + name, 'sha256': digest},
                'Pinned offline Tiny native asset, checked before native loader import',
                'FrozenProtenixTiny.load -> bootstrap_native_assets(include_model=True)')

    directories = [
        {'path': BASE + '/protenix-extended-03', 'recursive': True,
         'reason': 'Current staged Pro03 control/output namespace: no deletion of present gates/unknown markers or future CPU/startup/native/segment proofs.'},
        {'path': BASE + '/protenix-extended-release-03/source', 'recursive': True,
         'manifest_sha256': 'da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562',
         'files': 643, 'reason': 'Every wrapper and driver reauthenticates full exact sealed execution source.'},
        {'path': BASE + '/protenix-extended-release-01/source', 'recursive': True,
         'manifest_sha256': 'a30f980fd8fe95001fdfc6780d720bdde6486d2421d7e4942b119ace61777f00',
         'files': 641, 'reason': 'Pro03 authenticated ancestor/source comparison and failure provenance; not a redundant upload archive.'},
        {'path': BASE + '/restart-lrreset-release-04/source', 'recursive': True,
         'manifest_sha256': 'c2628d08f7224f3b572ab456f4113d8a66b587a5d69f85a072877052fc09a057',
         'files': 625, 'reason': 'Current original-study report/diagnostics and ancestor provenance require full immutable source.'},
        {'path': OLD, 'recursive': True,
         'reason': 'Parent reports current combined-report29796558 still PEND/invalid dependencies. All3 original saved studies, native readiness/CPU/launcher proof closure and all probability frames remain protected; no job cancellation was authorized by this audit.'},
        {'path': BASE + '/lrreset-analysis-03', 'recursive': True,
         'reason': 'Current report/diagnostic control, source-input pins, collector/diagnostic evidence and pending report outcome must not disappear during offload.'},
        {'path': BASE + '/lrreset-analysis-release-03/source', 'recursive': True,
         'reason': 'Currently pending report/diagnostic immutable execution source; inspect actual submitted command path before any equivalent alias claim.'},
        {'path': '/work3/s234818/mosaic/envs/protenix-tiny', 'recursive': True,
         'reason': 'Actual Pro03 Python/native Protenix/JAX/CUDA dependencies; native source hashes are authenticated inside the environment.'},
        {'path': '/work3/s234818/latentmol', 'recursive': True,
         'reason': 'Parent explicitly reports current VAE/embedding work; entire other project is protected. No dependency/status inference made here.'},
        {'path': '/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-01/prepare/payload/prepared', 'recursive': True,
         'reason': 'Shared unchanged Boltz prepared context used by current peers; sibling review supplies exact asset proof closure.'},
    ]
    inspected = ['extended_ops.py', 'extended_protocol.py', 'extended_driver.py',
                 'extended_contract.py', 'extended_cpu.lsf', 'extended_stage.lsf',
                 'parent.json', 'submit_recovery02.py', 'package.json']
    local_sources = [{'path': str((PRO / name).relative_to(REPO)).replace('\\', '/'),
                      'sha256': sha((PRO / name).read_bytes())} for name in inspected]
    report_dir = REPO / 'experiments/brd4-jq1/fixed-lrreset/runs/20261023-brd4-jq1-lrreset-analysis-03'
    for name in ['core-scripts__jq1_lrboost_report_contract.py', 'core-scripts__lrboost_endpoint_diagnostics.py']:
        p = report_dir / name
        local_sources.append({'path': p.relative_to(REPO).as_posix(), 'sha256': sha(p.read_bytes())})
    inventory = {
        'version': 'mosaic-current-protenix-offload-protection-v1',
        'status': 'local_dependency_protection_only_no_remote_delete_plan',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Pro03/641/625 continuation, frozen inputs and current original-study reporting/diagnostics; read local sealed bytes/JSON/AST only.',
        'source_archive': {'path': archive_path.relative_to(REPO).as_posix(), 'bytes': 39644558,
                           'sha256': sha(archive_path.read_bytes()), 'source_members': 643},
        'inspected_local_sources': local_sources,
        'required_exact_files': sorted(pinned.values(), key=lambda row: row['path']),
        'required_remote_directories': directories,
        'runtime_proof_closure_rules': [
            'For every current cpu/native/segment/startup certificate or readiness assessment, retain all path/sha256 proof targets recursively, the actual submission, raw same-job LSF/scheduler proof, and input asset files.',
            'Do not remove any new segment snapshots: certify_segment hashes every saved state NPZ and places every snapshot into certificate.proofs; successors reverify that entire certificate closure.',
            'Keep all original625 saved frames while current combined-report29796558 remains pending; direct continuation requirements alone do not override the separate reporting dependency.',
            'All new owned control and scientific output namespaces are protected, even if a local snapshot previously said no jobs. Only an actual fresh parent scheduler/artifact query can establish current outcomes.',
            'Do not use an unreferenced-file search as delete authorization: source manifests and environment/native code inventories can authenticate files without literal path strings in this review.'
        ],
        'older_original_snapshots': {
            'direct_continuation_reads': [parent['snapshot']['path']],
            'direct_continuation_other_saved_npz_read': False,
            'current_report_requires_all_frames': True,
            'offload_now': False,
            'reason': 'Current pending saved-only report/collector rereads exact every-frame probability snapshots. Diagnostics additionally select original0/10000 plus nearest declared nearby state (9500,9900,9999) if applicable.'},
        'original_large_metadata': {
            'must_remain_remote_under_current_contract': True,
            'fields': ['summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl'],
            'reason': 'Pro03 CPU verify_parent rehashes and parses these original bytes and compares complete JSONL rows; a laptop-only copy cannot satisfy original remote paths.'},
        'possible_offload_review_only': [
            {'scope': 'A redundant upload .tar.gz beside a fully authenticated source tree',
             'condition': 'Parent proves no currently executing/pending operator needs this exact path, complete laptop archive and every member match, source tree/proofs retained, fresh per-file identity/race check before removal.'},
            {'scope': 'Past pytest cache/owned temporary fixture directories not in any active proof/source map',
             'condition': 'Parent inventories exact paths and verifies no current subprocess, certificate proof or symlink target depends on them; full backup and race-safe identity check still mandatory.'},
            {'scope': 'Other Mosaic model-cache families',
             'candidate_names_only': ['alphafold2', 'boltzgen', 'promera', 'openfold3', 'huggingface', 'opendde', 'msa',
                                      'protenix_mini_default_v0.5.0', 'protenix_base_default_v1.0.0',
                                      'protenix_base_20250630_v1.0.0', 'protenix-v2'],
             'condition': 'These are source-declared loader names, not proof of remote existence or non-use. Parent must check all current experiments and other projects before selecting an exact asset; never delete a whole cache directory containing the five protected Tiny files.'}
        ],
        'space_accounting': {
            'exact_required_file_entries': len(pinned),
            'sum_known_logical_bytes_with_duplicate_paths_deduplicated': sum(v.get('bytes', 0) for v in pinned.values()),
            'complete_remote_inventory_available': False,
            'hardlinks_or_sparse_allocation_observed_locally': False,
            'physical_or_quota_savings_established': False,
            'requirements': 'Remote lstat must record device/inode/link count/mode/size/allocated blocks. Logical sizes of hardlinked paths cannot be added as physical savings; removing one link may free no data. Current conservative quota gate must be rerun after any completed deletion/quota refresh.'},
        'current_status_inference': False,
        'no_ssh': True, 'no_local_numerical_execution': True, 'no_remote_mutation': True,
        'no_saved_result_change': True, 'delete_allowlist': []}
    with output.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(inventory, handle, indent=2, sort_keys=True); handle.write('\n')
    print(json.dumps({'path': str(output), 'sha256': sha(output.read_bytes()),
                      'exact_protected_files': len(pinned),
                      'sum_known_logical_bytes': inventory['space_accounting']['sum_known_logical_bytes_with_duplicate_paths_deduplicated']}))


if __name__ == '__main__':
    main()
