"""Allocated CPU attribution and exact old/new preparation parity at epoch36.

The same current frozen decoder outputs and original target batches feed both
preparations. Searches run serially here so cProfile can attribute their native
work; these timings are NOT an estimate of 12-worker GPU training throughput.
Full native synchronized-update gates remain the training promotion criterion.
"""
import argparse
import cProfile
from dataclasses import asdict
import io
import os
from pathlib import Path
import pstats
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not os.environ.get('LSB_JOBID'):
        raise RuntimeError('Preparation numerical profile requires an allocated HPC job')
    import torch
    from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
    from latentmol.chemistry_continuation import authenticated, read_plan
    from latentmol.evaluate import load_model
    from latentmol.experiments import write_json
    from latentmol.probability_repair import ProbabilityRepairConfig
    from latentmol.repair_execution_reference import load_preparation_reference
    from latentmol.repair_failure_guidance import prepare_failure_guidance
    from latentmol.repair_progress import (ProgressSearchConfig, search_progress_repair,
                                          search_progress_repair_reference)

    plan = read_plan(args.plan, args.plan_sha256)
    reference = plan['execution_reference']
    checkpoint = reference['starting_checkpoint']
    authenticated(checkpoint['path'], checkpoint['sha256'])
    module = Path(reference['release']) / 'src/latentmol/repair_failure_guidance.py'
    oracle = load_preparation_reference(module, reference['module_sha256'])
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve every profile outcome; require a new path')
    report = dict(status='failed', job_id=os.environ['LSB_JOBID'],
        identity=dict(plan_sha256=args.plan_sha256, archive_sha256=plan['archive_sha256'],
                      manifest_sha256=plan['manifest_sha256'], parent_sha256=plan['parent_checkpoint_sha256']),
        checkpoint_sha256=checkpoint['sha256'], completed_epoch=checkpoint['epoch'],
        points=256, seed=123904017, rows=[],
        scope='Same CPU decoded points and base targets; serial attribution only, not native epoch throughput')
    try:
        if torch.get_num_threads() != 4 or int(os.environ.get('LSB_DJOB_NUMPROC', '0')) != 16:
            raise ValueError('Require allocated16 CPU slots and four parent Torch threads')
        model, _ = load_model(Path(checkpoint['path']), 'cpu')
        model.eval().requires_grad_(False)
        generator = torch.Generator().manual_seed(report['seed'])
        z = torch.randn(report['points'], model.config.latent, generator=generator)
        policy = ProgressSearchConfig()  # Unchanged production6/32/4 bond-only law.
        originals = ProbabilityRepairConfig(**plan['protocol']['search'])
        with AtomPreservingRepairTeacher(model.spec, originals, backend='optimized', workers=12) as teacher:
            for start in range(0, len(z), 128):
                with torch.no_grad():
                    decoded = model.decode(z[start:start + 128])
                base = teacher.build_targets(decoded, candidate_ids=list(range(start, start + len(decoded.atoms))), update=0)
                observed, records = {}, {}
                for name, prepare, search in [('historical', oracle.prepare_failure_guidance, search_progress_repair_reference),
                                               ('optimized', prepare_failure_guidance, search_progress_repair)]:
                    profile = cProfile.Profile()
                    phases = {}
                    options = dict(progress_search=policy,
                        progress_map=lambda tasks: [search(*task) for task in tasks])
                    if name == 'optimized':
                        options['phase_seconds'] = phases
                    tick = time.perf_counter()
                    profile.enable()
                    prepared = prepare(decoded, base, model.spec, **options)
                    profile.disable()
                    seconds = time.perf_counter() - tick
                    stream = io.StringIO()
                    pstats.Stats(profile, stream=stream).sort_stats('cumulative').print_stats(30)
                    observed[name] = prepared
                    records[name] = dict(profiled_seconds=seconds, phases=phases,
                        cumulative_profile_top30=stream.getvalue())
                old, new = observed['historical'], observed['optimized']
                if old.identity != new.identity or old.receipt != new.receipt:
                    raise ValueError('Optimized preparation changed scientific targets, costs or coverage')
                report['rows'].append(dict(start=start, count=len(decoded.atoms),
                    failed=new.receipt['failed_rows'], preparation_identity=new.identity,
                    full_policy=policy.record(), coverage=new.receipt,
                    observations=records, exact_preparation_parity=True))
        if sum(row['failed'] for row in report['rows']) < 1:
            raise ValueError('The profile did not observe a failed original repair')
        report.update(status='passed', scientific_receipts_exact=True,
            policy=asdict(policy), note='Profiled serial wall includes profiler overhead; use native GPU benchmark for adoption')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(output, report)
    print({'status': report['status'], 'points': report['points'], 'rows': len(report['rows'])})


if __name__ == '__main__':
    main()
