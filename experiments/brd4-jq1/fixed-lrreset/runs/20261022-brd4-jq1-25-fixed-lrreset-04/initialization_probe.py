"""Allocated exact initialization replay diagnosis; no model or update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE = Path('/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/restart-lrreset-release-03/source')
BASELINE = Path('/zhome/95/e/206121/mosaic-runs/20261021-brd4-jq1-25-protenix-02/run-protenix/payload/study/initialization.npz')
BASELINE_SHA = '351789aa62d3ff111a4d2050a062cf844a1f29a87a7e352af465ff4299fd1bbf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['cpu', 'gpu'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    assert os.environ.get('LSB_JOBID'), 'Numerical replay requires allocated HPC'
    assert hashlib.sha256(BASELINE.read_bytes()).hexdigest() == BASELINE_SHA
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts')]
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry.state import GraphLogits
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    from mosaic.small_molecule.state import graph_from_hard

    jax.config.update('jax_default_matmul_precision', 'highest')
    assert not jax.config.jax_enable_x64
    with np.load(BASELINE) as f:
        original = {k: f[k].copy() for k in f.files}
    assert set(original) == {'X', 'E_upper', 'atom_logits', 'bond_logits', 'key_data'}
    assert original['X'].shape == original['atom_logits'].shape == (25, 6)
    assert original['E_upper'].shape == original['bond_logits'].shape == (300, 5)
    assert original['key_data'].shape == (2,)
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for k, v in original.items() if k != 'key_data')

    def compare(values):
        assert set(values) == set(original)
        return {k: {'equal': bool(np.array_equal(original[k], values[k])),
                    'dtype_equal': original[k].dtype == values[k].dtype,
                    'shape_equal': original[k].shape == values[k].shape,
                    'different_entries': int(np.count_nonzero(original[k] != values[k])),
                    'maximum_absolute_difference': float(np.max(np.abs(original[k].astype(np.float64) - values[k].astype(np.float64))))}
                for k in original}

    # Exact comparison fixture: detect a one-ULP difference, without tolerances.
    assert all(x['equal'] for x in compare(original).values())
    damaged = {k: v.copy() for k, v in original.items()}
    damaged['X'][0, 0] = np.nextafter(damaged['X'][0, 0], np.float32(1.0))
    assert compare(damaged)['X']['different_entries'] == 1 and not compare(damaged)['X']['equal']
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    variants = [('cpu', 'regenerated'), ('cpu', 'saved_logits')]
    if args.phase == 'gpu':
        assert jax.devices()[0].platform == 'gpu'
        variants += [('gpu', 'regenerated'), ('gpu', 'saved_logits'), ('gpu', 'saved_logits_jit')]
    for platform, mode in variants:
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            key = jax.random.key(17)
            if mode == 'regenerated':
                hard = graph_from_hard([0] * 25, [int(j == i + 1) for i in range(25) for j in range(i + 1, 25)])
                logits = GraphLogits.from_graph(initialize_graph(hard, {'kind': 'chain_uniform', 'mixing': 0.5}))
            else:
                logits = GraphLogits(jax.device_put(original['atom_logits'], device), jax.device_put(original['bond_logits'], device))
            graph = jax.jit(lambda x: x.graph())(logits) if mode.endswith('_jit') else logits.graph()
            arrays = {'X': np.asarray(graph.X), 'E_upper': np.asarray(graph.E_upper),
                      'atom_logits': np.asarray(logits.atoms), 'bond_logits': np.asarray(logits.bonds),
                      'key_data': np.asarray(jax.random.key_data(key))}
            name = platform + '-' + mode
            path = args.output / (name + '.npz')
            np.savez_compressed(path, **arrays)
            equality = compare(arrays)
            results[name] = {'device': str(device), 'fields': equality,
                             'all_five_arrays_exact': all(v['equal'] and v['shape_equal'] and v['dtype_equal'] for v in equality.values()),
                             'proof': {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
    report = {'status': 'diagnostic_completed', 'phase': args.phase, 'job_id': os.environ['LSB_JOBID'],
              'probe_manifest_sha256': args.manifest, 'baseline_sha256': BASELINE_SHA,
              'jax_version': jax.__version__, 'devices': [str(d) for d in jax.devices()],
              'variants': results, 'can_certify_original_or_start_optimizer': False,
              'saved_results_changed': False, 'fixture_one_ulp_detection': 'passed'}
    (args.output / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n')
    assert sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) <= 1024**2
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
