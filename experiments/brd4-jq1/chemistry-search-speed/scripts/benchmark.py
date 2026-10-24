"""Allocated-only saved-input preparation and frozen-loss speed comparison."""
from contextlib import nullcontext
import cProfile
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import time


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import (
        GraphLogits, mosaic_spec, prepare_failure_guidance, failure_aware_repair_loss,
    )
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from exact_reuse import install_reuse
    from parallel_search import install_parallel

    plan = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(exist_ok=False)
    records = []
    # Materialize authenticated saved logits, never reconstruct an unsaved frame.
    inputs = []
    for row in plan['inputs']:
        path = Path(row['path'])
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and digest(path.read_bytes()) == row['sha256']
        with np.load(path, allow_pickle=False) as f:
            logits = GraphLogits(jnp.asarray(f['atom_logits']), jnp.asarray(f['bond_logits']))
        assert logits.atoms.shape == (25, 6) and logits.bonds.shape == (300, 5)
        assert logits.atoms.dtype == logits.bonds.dtype == jnp.float32
        assert all(np.isfinite(np.asarray(a)).all() for a in (logits.atoms, logits.bonds))
        inputs.append((row, logits))
    spec = mosaic_spec(25)
    search = ProbabilityRepairConfig(**plan['search'])
    progress = ProgressSearchConfig(**plan['progress_search'])
    # Warm only the common JAX decoding/softmax path, not any chemistry search.
    for _, logits in inputs:
        decoded = logits.decoded()
        jax.block_until_ready((decoded, jax.nn.softmax(decoded.atoms, -1),
                              jax.nn.softmax(decoded.charges, -1),
                              jax.nn.softmax(decoded.symmetric_bond_logits, -1)))
    baseline = {}
    modes = ['baseline', 'cost', 'cache', 'combined', 'parallel2', 'parallel4']
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'plan_sha256': digest(Path(sys.argv[1]).read_bytes()), 'modes': modes,
              'scope': 'Detached saved graph preparation only; no native model, optimizer updates or production installation',
              'original_results_changed': False, 'rows': records}
    try:
        for mode in modes:
            context = (nullcontext({}) if mode == 'baseline' else
                       install_parallel(int(mode[-1])) if mode.startswith('parallel') else
                       install_reuse(mode=mode, max_entries=8192))
            with context as stats:
                for index, (row, logits) in enumerate(inputs):
                    times = []
                    cold = None
                    for repeat in range(plan['repeats']):
                        start = time.perf_counter()
                        prepared = prepare_failure_guidance(logits.decoded(), spec, update=index,
                            search=search, progress_search=progress)
                        jax.block_until_ready(prepared.arrays)
                        seconds = time.perf_counter() - start
                        if repeat == 0:
                            cold = seconds
                        times.append(seconds)
                        # All scientific receipt fields and all target/cut/cap arrays.
                        payload = {'receipt': prepared.receipt, 'arrays':
                                   {k: np.asarray(v).tolist() for k, v in prepared.arrays.items()},
                                   'identity': prepared.identity}
                        canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
                        if mode == 'baseline' and repeat == 0:
                            baseline[row['label']] = canonical
                            (out / (row['label'] + '-baseline.json')).write_bytes(canonical)
                        assert canonical == baseline[row['label']], (mode, row['label'], 'preparation differs')
                    value_grad = eqx.filter_value_and_grad(
                        lambda z: failure_aware_repair_loss(z.decoded(), prepared, spec)[0])
                    value, grad = value_grad(logits)
                    jax.block_until_ready((value, grad))
                    scalar, terms = failure_aware_repair_loss(logits.decoded(), prepared, spec)
                    jax.block_until_ready((scalar, terms))
                    gradients = [np.asarray(grad.atoms), np.asarray(grad.bonds)]
                    assert all(np.isfinite(v).all() for v in gradients) and np.isfinite(float(value))
                    evidence = {'value': float(value), 'independent_primal': float(scalar),
                                'terms': {k: float(v) for k, v in terms.items()},
                                'gradient_sha256': [digest(v.tobytes()) for v in gradients],
                                'gradient_norm': float(np.sqrt(sum(np.sum(v*v) for v in gradients)))}
                    if mode == 'baseline':
                        baseline[row['label'] + '-gradient'] = evidence
                    assert evidence == baseline[row['label'] + '-gradient'], (mode, row['label'], 'loss/gradient differs')
                    detail = {'mode': mode, 'input': row, 'first_encounter_seconds': cold, 'seconds': times,
                              'cache_history': 'One persistent context per mode across the ordered saved inputs; pool startup affects first searched graph',
                              'warm_median_seconds': statistics.median(times[1:]),
                              'preparation_identity': prepared.identity, 'exact_preparation_and_gradient': True,
                              'original_checks': prepared.receipt['original_checks'],
                              'additional_checks': prepared.receipt['additional_checks'],
                              'routes': [r['route'] for r in prepared.receipt['rows']], 'loss': evidence,
                              'adapter_stats': dict(stats) if isinstance(stats, dict) else repr(stats)}
                    records.append(detail)
                    (out / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
                    print(json.dumps({k: detail[k] for k in ['mode','input','first_encounter_seconds','warm_median_seconds']}), flush=True)
        # Separate cProfile samples: never mix their overhead into timing rows.
        for mode in ['baseline', 'combined']:
            context = nullcontext({}) if mode == 'baseline' else install_reuse(mode='combined', max_entries=8192)
            with context:
                profile = cProfile.Profile()
                row, logits = inputs[0]
                profile.enable()
                prepare_failure_guidance(logits.decoded(), spec, update=0, search=search, progress_search=progress)
                profile.disable()
                profile.dump_stats(str(out / (mode + '.prof')))
                with (out / (mode + '-profile.txt')).open('w') as stream:
                    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(35)
        report.update(status='passed_exact_on_sampled_saved_inputs', completed_utc=datetime.now(timezone.utc).isoformat(),
                      limitation='Repeated-input warm timings can benefit from exact chemistry caches. First encounter of each adjacent saved input is separately reported. Not whole-run convergence or native readiness.')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        (out / 'assessment.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
