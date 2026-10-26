"""Local stdlib source/plan packaging only; all numerical checks are allocated."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
LOCAL = REPO / 'results/hpc/20261026-chemistry-full-model-05'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'


def raw_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def main():
    import sys
    sys.path.insert(0, str(HERE / 'scripts'))
    from full_model_contract import VERSION, PROFILES, POLICY, validate_config, validate_plan
    inspected = json.loads((REPO / 'results/hpc/20261025-chemistry-full-model-01/inputs-01/raw.json').read_bytes())
    members = {}
    plans = {}
    for arm, profile in [('protenix', 'protenix-random50k'), ('boltz', 'boltz-chem30k')]:
        row = inspected['profiles'][arm]
        validate_config(row['config'], profile)
        plan = {'version': VERSION, 'profile': profile, 'source': row['source'],
                'driver': row['driver'], 'inputs': row['inputs'], 'policy': POLICY,
                'expected': {key: PROFILES[profile][source] for key, source in
                             [('model_digest', 'model'), ('context_digest', 'context')]},
                'timeout_seconds': 6600, 'python': '/work3/s234818/mosaic/envs/protenix-tiny/bin/python'}
        plan['expected']['pocket_sha256'] = row['inputs']['pocket']['sha256']
        if arm == 'boltz':
            plan['worker_python'] = row['worker_python']
        validate_plan(plan)
        plans[arm] = plan
        members['plans/' + arm + '.json'] = raw_json(plan)
    scripts = ['exact_reuse.py', 'parallel_search.py', 'hybrid_search.py', 'gpu_matrix.py',
               'round2_data.py', 'profile_oracle.py', 'full_model_contract.py', 'full_model_pilot.py',
               'gpu_local_bridge.py', 'gpu_local_worker.py', 'gpu_process_admission.py', 'full_model_ops.py', 'full_model_stage.lsf',
               'full_model_context.py', 'replay_context_qualification.py']
    tests = ['test_speed.py', 'test_hybrid.py', 'test_gpu_matrix.py', 'test_oracle_profile.py',
             'test_full_model_pilot.py', 'test_gpu_local_bridge.py', 'test_full_model_ops.py',
             'test_full_model_replay.py', 'test_gpu_process_admission.py', 'test_full_model_retry.py',
             'test_full_model_context.py', 'test_replay_context_qualification.py']
    for name in ['README.md', 'scripts/README.md', 'tests/README.md',
                 *['scripts/' + name for name in scripts], *['tests/' + name for name in tests]]:
        raw = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
        members[name] = raw
    members['plans/README.md'] = b'# Immutable matched full-model plans\n\nOriginal settings and byte-bound inputs; no production adoption.\n'
    members['protocol.md'] = (HERE / 'runs/20261026-chemistry-full-model-05/README.md').read_bytes().replace(b'\r\n', b'\n')
    # Expected counts are declared by the fixture author; only allocated raw XML
    # can establish that the tests actually ran and passed.
    counts = json.loads((LOCAL / 'cpu-counts.json').read_bytes())
    parent_sources = [row['source'] for row in inspected['profiles'].values()] + [
        {'path': BASE + 'chemistry-search-speed-release-02/source', 'files': 19,
         'manifest_sha256': '09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222'},
        {'path': BASE + 'nonchem-stride20-startup-release-03/source', 'files': 12,
         'manifest_sha256': '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'},
        {'path': BASE + 'chemistry-full-model-release-01/source', 'files':26,
         'manifest_sha256':'347e753d481ec3d584f3b0b05cc2801ee745036ee17638d7c3672fc2ca2cba16'},
        {'path': BASE + 'protenix-endpoint-fix-release-02/source', 'files':14,
         'manifest_sha256':'09f759a7befb8a7b8f0e1d203e15acf939126fba493b30e6a2bf9cd2eea2216b'},
        {'path':BASE+'chemistry-full-model-release-02/source','files':30,'manifest_sha256':'05e316a7b1bcceaf286b03231200a2d1f66e3f6a7ce927d4c1e73f08a24f86f9'}]
    prior_observation=REPO/'results/hpc/20261025-chemistry-full-model-01/observation-chemistry-launch-check-20261025-01/files'
    prior_paths=['protenix/payload/assessment.json','protenix/stdout.29811883.log','protenix/payload/comparison/combined/study/summary.json','protenix/payload/comparison/combined/paired-checks/assessment.json','boltz/payload/assessment.json','boltz/stdout.29811884.log','boltz/payload/comparison/combined/study/summary.json','boltz/payload/comparison/combined/study/worker.log','boltz/payload/comparison/combined/gpu-chemistry/receipt.json']
    race_root=REPO/'results/hpc/20261025-chemistry-full-model-02/observation-cpu-terminal-02/files'
    race_names=['protenix/payload/assessment.json','protenix/stdout.29816183.log','protenix/stderr.29816183.log','protenix/payload/comparison/assessment.json']
    race_proofs=[{'path':BASE+'chemistry-full-model-02/'+name,'bytes':(race_root/name).stat().st_size,'sha256':hashlib.sha256((race_root/name).read_bytes()).hexdigest()} for name in race_names]
    prior_proofs=[{'path':BASE+'chemistry-full-model-01/'+name,'bytes':(prior_observation/name).stat().st_size,'sha256':hashlib.sha256((prior_observation/name).read_bytes()).hexdigest()} for name in prior_paths]
    replay_manifest = json.loads((REPO/'results/hpc/20261026-chemistry-replay-02/source01/source_manifest.json').read_bytes())
    with tarfile.open(REPO/'results/hpc/20261026-chemistry-replay-02/source01/source.tar.gz') as saved:
        replay_admission = json.loads(saved.extractfile('admission.json').read())
    for row in replay_admission['parent_sources']:
        if row['path'] not in {p['path'] for p in parent_sources}: parent_sources.append(row)
    parent_sources.append({'path':BASE+'chemistry-replay-release-02/source','files':11,
                          'manifest_sha256':'89b290831e8ebf5633dba07736189a18848cd835d1d6cdf8a3650a5f056a928b'})
    merged = {p['path']:p for p in inspected['preserved_proofs']+prior_proofs+race_proofs}
    for row in replay_admission['preserved_proofs']:
        assert row['path'] not in merged or merged[row['path']]==row
        merged[row['path']]=row
    failed04=REPO/'results/hpc/20261026-chemistry-full-model-04/observation-context-cpu-01/files'
    for name in ('cpu/submission.json','cpu/stdout.29818517.log','cpu/payload/assessment.json','cpu/payload/tests.xml',
                 'cpu/payload/tests.log','after-cpu/submission.json','after-cpu/stdout.29818518.log'):
        data=(failed04/name).read_bytes();path=BASE+'chemistry-full-model-04/'+name
        merged[path]={'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    parent_sources.append({'path':BASE+'chemistry-full-model-release-04/source','files':34,
        'manifest_sha256':'a7c048cd7bfb9ab0fcfa714793fbd69fef50a9cbca9576867ef8ebad6342bcc1'})
    from full_model_context import PRODUCER
    manifest = {'version': 'chemistry-full-model-source-v5', 'cpu_counts': counts,
                'parent_sources': parent_sources, 'preserved_proofs': list(merged.values()), 'replay_context_producer': PRODUCER,
                'failed_predecessors': {'protenix':'29811883','boltz':'29811884'},
                'active_gpu_arms':['protenix'],
                'persistent_cap_bytes': 2 * 1024**3, 'worker_count': 4,
                'scientific_production_changed': False, 'monitor_remains_paused': True,
                'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(members.items())}}
    manifest_raw = raw_json(manifest)
    members['source_manifest.json'] = manifest_raw
    out = LOCAL / 'source01'
    out.mkdir(exist_ok=False)
    (out / 'source_manifest.json').write_bytes(manifest_raw)
    archive = out / 'source.tar.gz'
    with tarfile.open(archive, 'x:gz') as target:
        for name, raw in sorted(members.items()):
            item = tarfile.TarInfo(name); item.size = len(raw); item.mode = 0o444; item.mtime = 0
            target.addfile(item, io.BytesIO(raw))
    with tarfile.open(archive) as target:
        rows = target.getmembers()
        assert len(rows) == len(members) and all(row.isfile() for row in rows)
        assert len({row.name for row in rows}) == len(rows)
        for row in rows:
            assert target.extractfile(row).read() == members[row.name]
    result = {'status': 'local_static_AST_byte_schema_only', 'files': len(members)-1,
              'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'archive_bytes': archive.stat().st_size, 'cpu_counts': counts,
              'cpu_cases': sum(counts.values()), 'allocated_CPU_pass': False,
              'full_model_pass': False, 'jobs_submitted': False}
    (out / 'preparation.json').write_bytes(raw_json(result))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
