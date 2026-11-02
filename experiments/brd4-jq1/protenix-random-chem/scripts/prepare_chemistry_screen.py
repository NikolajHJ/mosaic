"""Build a fresh immutable comparison capsule; no numerical work or submission."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def build(repo, tag, version, comparison=None):
    repo = Path(repo)
    local = repo / 'results' / 'hpc' / tag / version
    local.mkdir(parents=True, exist_ok=False)
    source = local / 'source'; source.mkdir()
    remote = '/work3/s234818/mosaic/runs/' + tag
    remote_source = remote + '/' + version + '/source'
    remote_root = remote + '/' + version + '/run'
    baseline = json.loads((repo / 'results/hpc/20261030-protenix-chemistry-noise-01/source03/plan.json').read_bytes())
    original = baseline['arms']['protenix_chemistry']
    sparse = dict(mode='sparse_random', extra_edge_probability=.02, present_probability=.8,
                  absent_probability=.98, single_bond_probability=.8)
    delayed = dict(mode='delayed', initial=.001, final=100., hold_updates=5000, total_updates=50000)
    adaptive = dict(mode='adaptive', initial=1., ratio_target=1., minimum=.1, maximum=100., ema=.9)
    phase = dict(mode='validity_first', max_chemistry_steps=5000, ramp_steps=2000)
    severity = dict(version='severity-progress-v1', max_edits=6, max_checks=12, candidates_per_step=32)
    environment = dict(version='common-organic-neighbor-v1', weight=.1)
    variants = {
        'delayed': {},
        'constant1': dict(chemistry=dict(mode='constant', coefficient=1.)),
        'constant10': dict(chemistry=dict(mode='constant', coefficient=10.)),
        'chemfirst': dict(chemistry=dict(mode='constant', coefficient=10.), binding=phase),
        'severity': dict(severity_progress=severity),
        'adaptive': dict(chemistry=adaptive),
        'neighbor': dict(neighbor_environment=environment),
        'temperature2': dict(temperature=2.),
        'earlyseverity': dict(chemistry=dict(mode='constant', coefficient=1.), severity_progress=severity),
        'combined': dict(chemistry={**adaptive, 'minimum': 1.}, binding=phase,
                         severity_progress=severity, neighbor_environment=environment, temperature=2.),
    }
    if comparison == 'structural':
        variants = {
            'beam': dict(chemistry=dict(mode='constant', coefficient=1.)),
            'structural': dict(chemistry=dict(mode='constant', coefficient=1.),
                               structural_repair=dict(version='connected-backbone-repair-v1',
                                                      max_candidates=6, max_bond_operations=1024)),
        }
    if comparison == 'structural-strong-noise':
        structural = dict(version='connected-backbone-repair-v1', max_candidates=6, max_bond_operations=1024)
        variants = {}
        for start in ('sparse', 'chain'):
            for noisy in (False, True):
                label = 'structural10-' + ('noise-' if noisy else '') + start
                variants[label] = dict(chemistry=dict(mode='constant', coefficient=10.), structural_repair=structural)
                if start == 'chain':
                    variants[label]['initialization'] = dict(mode='carbon_chain_uniform', uniform_fraction=.5, logit_noise_std=.05)
                if noisy:
                    variants[label]['logit_noise'] = dict(version='binding-logit-noise-linear-v1', initial_ratio=.4,
                                                        zero_at_update=20000, seed=230017)
    if comparison == 'annealed50k':
        variants = {'annealed-sparse': dict(
            chemistry=dict(mode='ramp_hold', initial=1., final=30., hold_updates=20000, ramp_end=30000),
            temperature=.7,
            temperature_schedule=dict(version='linear-temperature-v1', initial=.7, final=.4, total_updates=50000),
            structural_repair=dict(version='connected-backbone-repair-v1', max_candidates=6, max_bond_operations=1024),
            logit_noise=dict(version='binding-logit-noise-linear-v1', initial_ratio=.4, zero_at_update=30000, seed=230017))}
    arms = {}
    requested = [(label, seed, False) for label in variants for seed in (17, 23)]
    if comparison in ('structural-strong-noise', 'annealed50k'):
        requested = [(label, 17, False) for label in variants]
    if comparison is None:
        requested += [('delayed', 17, True), ('combined', 17, True)]
    updates = 30000 if comparison in ('structural', 'structural-strong-noise') else 20000
    if comparison == 'annealed50k':
        updates = 50000
    for label, seed, dense in requested:
        name = label + ('-dense' if dense else '') + '-s' + str(seed)
        config = dict(version='protenix-chemistry-screen-v1', updates=updates, init_seed=seed,
                      initialization={'mode': 'dense_original'} if dense else deepcopy(sparse),
                      temperature=1., chemistry=deepcopy(delayed), binding=dict(mode='always'),
                      sampling_interval=1000, sampling_draws=128)
        config.update(deepcopy(variants[label]))
        row = {k: deepcopy(original[k]) for k in ('config', 'science_source', 'reference', 'pocket', 'frozen_identity',
               'adapter_source', 'snapshot', 'context_ready', 'gpu_adapter_source')}
        probe = 'combined-dense-s17' if dense and label == 'combined' else (
            'combined-s17' if label in ('combined', 'severity', 'earlyseverity', 'neighbor', 'temperature2') else
            'chemfirst-s17' if label == 'chemfirst' else 'constant10-s17')
        row.update(name=name, screen=config, rate=.3, model_key=17, composition_weight=0, margin=False,
                   matrix_stride=20, loss_stride=1, output_cap_bytes=512 * 1024**2, file_cap=2500,
                   internal_seconds=24*3600-600, gpu_profile='v100-32gb-v1', queue='gpuv100', timing_probe=probe)
        if comparison == 'structural':
            minutes = 600 if seed == 17 else 525
            row.update(gpu_profile='a100-40gb-v1' if seed == 17 else 'a40-48gb-v1',
                       queue='gpua100' if seed == 17 else 'gpua40',
                       output_cap_bytes=1024**3, file_cap=4500, internal_seconds=minutes*60-600,
                       requested_wall_minutes=minutes,
                       budget_basis='same-device completed20k warm p95 *30000 *1.25 +1800s; rounded5min')
        if comparison == 'structural-strong-noise':
            # Independent arms hedge queue delays across the compatible families.
            # Existing measured same-scorer budgets; actual allocations/timings are
            # retained, so cross-device runtime is not attributed to the treatment.
            profile, queue, minutes = {
                'structural10-sparse': ('a100-40gb-v1', 'gpua100', 600),
                'structural10-noise-sparse': ('v100-32gb-v1', 'gpuv100', 665),
                'structural10-chain': ('a40-48gb-v1', 'gpua40', 525),
                'structural10-noise-chain': ('l40s-48gb-v1', 'gpul40s', 600),
            }[label]
            row.update(gpu_profile=profile, queue=queue, output_cap_bytes=1024**3, file_cap=4500,
                       internal_seconds=minutes*60-600, requested_wall_minutes=minutes,
                       budget_basis='Prior compatible scorer/structural timings plus25% and setup/cleanup; device differs between treatments')
        if comparison == 'annealed50k':
            row.update(gpu_profile='v100-32gb-v1', queue='gpuv100', output_cap_bytes=2*1024**3,
                       file_cap=7000, internal_seconds=1095*60-600, requested_wall_minutes=1095,
                       budget_basis='Actual same-scorer V1000.508s/update upper warm sample;2x for fresh native on cooling;50k*2*0.508*1.25+1800s setup/cleanup; rounded18h15')
        arms[name] = row
    plan = dict(version='protenix-chemistry-strategies-plan-v1', research_profile='exploratory',
                root=remote_root, source_root=remote_source, code_source_root=remote_source,
                routing_source=baseline['routing_source'], capacity_module=remote_source+'/chemistry_screen_ops.py',
                output_cap_bytes=(len(arms)*512+1024)*1024**2, file_cap=len(arms)*2500+10000,
                job_name_prefix='mosaic-chemstrategy-'+version, arms=arms,
                probe_arms=['constant10-s17', 'chemfirst-s17', 'combined-s17', 'combined-dense-s17'],
                interpretation=['20k early-regime comparison; original delayed baseline keeps its50k ramp horizon.',
                                'Combined uses early minimum1 and adaptive coefficient; it is not a strictly constant-weight arm.',
                                'Sparse random tree plus extra edges is connected but is not guaranteed chemically valid.',
                                'Paired seeds share initial logits; temperature2 intentionally changes represented probabilities.',
                                'No noise, composition or confidence margin; original results and existing jobs unchanged.'])
    if comparison == 'structural':
        plan.update(job_name_prefix='mosaic-structural-'+tag.split('-')[0]+'-'+version,
                    output_cap_bytes=4*1024**3+256*1024**2, file_cap=18500,
                    control_tests=['test_protenix_chemistry_screen.py', 'test_chemistry_structural.py'],
                    probe_arms=['beam-s17'],
                    interpretation=[
                        'Four fresh30k trajectories; old beam vs structural front end, initialization seeds17/23.',
                        'Within seed, same GPU family and bit-identical initializer; no device-smoke states adopted.',
                        'Only target discovery differs; existing loss, beam/progress fallback, coefficient1, LR.3 and model unchanged.',
                        'No extra qualification/probe chain; production startup/first updates provide finite-path device checks.',
                        'A100/seed17 and A40/seed23; runtime comparisons are within hardware/seed.',
                    ])
    if comparison == 'structural-strong-noise':
        plan.update(job_name_prefix='mosaic-structnoise-'+tag.split('-')[0]+'-'+version,
                    output_cap_bytes=4*1024**3+256*1024**2, file_cap=18500,
                    control_tests=['test_protenix_chemistry_screen.py', 'test_structural_noise_screen.py'],
                    probe_arms=[next(iter(arms))],
                    interpretation=[
                        'Four fresh30k trajectories, one seed17 per initializer; structural repair and constant coefficient10 throughout.',
                        'Sparse recipe unchanged; chain starts mix50% carbon-chain one-hot with50% uniform probabilities, then row-centered Gaussian logit noise std0.05.',
                        'Noisy treatment uses independent PCG64 seed230017; per-block noise norm0.4*LR*binding-only logit-gradient norm, linearly zero by20k then10k settling.',
                        'Same initializer gives identical logits in both treatments; no composition or confidence margin.',
                        'Full source teacher remains detached/fixed within each trial; one finite candidate, LR0.3, key17, T1.',
                        'Different GPU families reduce queue concentration; cross-device runtime differences are not treatment speedups.',
                        'Prior device evidence reused; no extra native/FD qualification chain or disposable production pilot.',
                    ])
    if comparison == 'annealed50k':
        parent = next(iter(arms))
        plan.update(job_name_prefix='mosaic-annealed50k-'+tag.split('-')[0]+'-'+version,
                    output_cap_bytes=3*1024**3+256*1024**2, file_cap=11500,
                    control_tests=['test_protenix_chemistry_screen.py', 'test_structural_noise_screen.py',
                                   'test_annealed_chemistry_screen.py', 'test_conditional_chemistry_tail.py'],
                    probe_arms=[parent],
                    conditional_tail=dict(parent_arm=parent, arm_name=parent+'-repair20k', target_updates=70000,
                                          coefficient=100., wall_minutes=255, output_cap_bytes=1024**3, file_cap=3500),
                    interpretation=[
                        'One fresh50k sparse seed17 structural-repair trajectory, original LR0.3/model/context/key17/composition0/no margin.',
                        'Chemistry1 through global20k, geometric1->30 at30k, then30; temperature linear0.7->0.4 over0..50k.',
                        'Binding-only per-block noise0.4->0 by30k, then20k settling; independent PCG64 seed230017, checkpointed.',
                        'Temperature is dynamic; source/trial teacher, coefficient andT fixed within each update; newT recomputes native values and gradients.',
                        'Allocated CPU callback checks completed actual50k endpoint argmax sanitization/connectivity; valid skips, invalid submits unique20k child atT0.4/noise0/chemistry100.',
                        'Child resumes exact saved50k logits/key/teacher/noise stream in fresh output; no source/result overwrite.',
                        'Changing temperature changes relaxed scorer inputs; cross-state contact and surrogate totals are not one fixed objective.',
                    ])
    scripts = repo / 'experiments/brd4-jq1/protenix-random-chem/scripts'
    for name in ('protenix_chemistry_screen.py', 'chemistry_screen_ops.py', 'chemistry_screen_control.py', 'chemistry_screen_stage.sh'):
        (source / name).write_bytes((scripts / name).read_bytes().replace(b'\r\n', b'\n'))
    if comparison in ('structural-strong-noise', 'annealed50k'):
        (source/'gradient_scaled_noise.py').write_bytes((scripts/'gradient_scaled_noise.py').read_bytes().replace(b'\r\n', b'\n'))
    library = repo / 'src/mosaic'
    if comparison == 'structural':
        library = repo / 'results/hpc/20261031-protenix-chemistry-strategies-01/source02/source/src/mosaic'
        if not library.is_dir():
            raise FileNotFoundError('Matched baseline source snapshot is required')
    if comparison in ('structural-strong-noise', 'annealed50k'):
        library = repo/'results/hpc/20261102-exclusive-gpu-recovery-01/screen01/source/src/mosaic'
        if not library.is_dir():
            raise FileNotFoundError('Completed structural comparison library snapshot is required')
    for path in library.rglob('*.py'):
        destination = source / 'src/mosaic' / path.relative_to(library)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    if comparison == 'structural':
        for name in ('__init__.py', 'preparation.py', 'structural.py'):
            relative = Path('src/mosaic/small_molecule/chemistry')/name
            shutil.copyfile(repo/relative, source/relative)
    for name in ('gpu_policy.py', 'storage_accounting.py'):
        shutil.copyfile(repo / 'scripts/hpc' / name, source / name)
    (source / 'tests').mkdir()
    for path in (repo / 'experiments/brd4-jq1/protenix-random-chem/tests/test_protenix_chemistry_screen.py',
                 repo / ('tests/small_molecule/test_chemistry_structural.py' if comparison == 'structural'
                         else 'tests/small_molecule/test_chemistry_severity.py')):
        shutil.copyfile(path, source / 'tests' / path.name)
    if comparison in ('structural-strong-noise', 'annealed50k'):
        shutil.copyfile(repo/'experiments/brd4-jq1/protenix-random-chem/tests/test_structural_noise_screen.py',
                        source/'tests/test_structural_noise_screen.py')
    if comparison == 'annealed50k':
        for name in ('test_annealed_chemistry_screen.py', 'test_conditional_chemistry_tail.py'):
            shutil.copyfile(repo/'experiments/brd4-jq1/protenix-random-chem/tests'/name, source/'tests'/name)
    (source / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8', newline='\n')
    checksums = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(source.rglob('*')) if p.is_file()}
    (source / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n', encoding='utf-8', newline='\n')
    archive = local / 'source.tar.gz'
    with tarfile.open(archive, 'w:gz') as output:
        for path in sorted(source.rglob('*')):
            if path.is_file(): output.add(path, arcname=path.relative_to(source).as_posix())
    result = dict(tag=tag, version=version, root=remote_root, source=remote_source,
                  archive=str(archive), archive_bytes=archive.stat().st_size,
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=len(checksums),
                  arms=len(arms), updates_per_arm=updates, total_updates=len(arms)*updates,
                  checksums_sha256=hashlib.sha256((source/'checksums.json').read_bytes()).hexdigest())
    (local/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--tag', default='20261031-protenix-chemistry-strategies-01')
    parser.add_argument('--version', default='source01')
    parser.add_argument('--comparison', choices=('structural', 'structural-strong-noise', 'annealed50k'))
    args = parser.parse_args(); build(args.repo, args.tag, args.version, args.comparison)
