"""Local stdlib source/plan packaging only; all numerical checks are allocated."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
LOCAL = REPO / 'results/hpc/20261025-chemistry-full-model-01'
BASE = '/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/'


def raw_json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def main():
    import sys
    sys.path.insert(0, str(HERE / 'scripts'))
    from full_model_contract import VERSION, PROFILES, POLICY, validate_config, validate_plan
    inspected = json.loads((LOCAL / 'inputs-01/raw.json').read_bytes())
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
               'gpu_local_bridge.py', 'gpu_local_worker.py', 'full_model_ops.py', 'full_model_stage.lsf']
    tests = ['test_speed.py', 'test_hybrid.py', 'test_gpu_matrix.py', 'test_oracle_profile.py',
             'test_full_model_pilot.py', 'test_gpu_local_bridge.py', 'test_full_model_ops.py']
    for name in ['README.md', 'scripts/README.md', 'tests/README.md',
                 *['scripts/' + name for name in scripts], *['tests/' + name for name in tests]]:
        raw = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.py'):
            ast.parse(raw, filename=name)
        members[name] = raw
    members['plans/README.md'] = b'# Immutable matched full-model plans\n\nOriginal settings and byte-bound inputs; no production adoption.\n'
    members['protocol.md'] = (HERE / 'runs/20261025-chemistry-full-model-01/README.md').read_bytes().replace(b'\r\n', b'\n')
    # Expected counts are declared by the fixture author; only allocated raw XML
    # can establish that the tests actually ran and passed.
    counts = json.loads((LOCAL / 'cpu-counts.json').read_bytes())
    parent_sources = [row['source'] for row in inspected['profiles'].values()] + [
        {'path': BASE + 'chemistry-search-speed-release-02/source', 'files': 19,
         'manifest_sha256': '09c0ab837bede0c5779eac0da06885cfbef6a98744f67af19e7b5a8b33ffa222'},
        {'path': BASE + 'nonchem-stride20-startup-release-03/source', 'files': 12,
         'manifest_sha256': '704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab'}]
    manifest = {'version': 'chemistry-full-model-source-v1', 'cpu_counts': counts,
                'parent_sources': parent_sources, 'preserved_proofs': inspected['preserved_proofs'],
                'persistent_cap_bytes': 2 * 1024**3, 'worker_count': 4,
                'scientific_production_changed': False, 'monitor_remains_paused': True,
                'files': {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(members.items())}}
    manifest_raw = raw_json(manifest)
    members['source_manifest.json'] = manifest_raw
    out = LOCAL / 'source02'
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
