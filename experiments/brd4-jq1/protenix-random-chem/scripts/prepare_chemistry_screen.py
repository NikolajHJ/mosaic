"""Build a fresh immutable comparison capsule; no numerical work or submission."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def build(repo, tag, version):
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
    arms = {}
    for label, seed, dense in [(label, seed, False) for label in variants for seed in (17, 23)] + [
            ('delayed', 17, True), ('combined', 17, True)]:
        name = label + ('-dense' if dense else '') + '-s' + str(seed)
        config = dict(version='protenix-chemistry-screen-v1', updates=20000, init_seed=seed,
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
    scripts = repo / 'experiments/brd4-jq1/protenix-random-chem/scripts'
    for name in ('protenix_chemistry_screen.py', 'chemistry_screen_ops.py', 'chemistry_screen_control.py', 'chemistry_screen_stage.sh'):
        (source / name).write_bytes((scripts / name).read_bytes().replace(b'\r\n', b'\n'))
    for path in (repo / 'src/mosaic').rglob('*.py'):
        destination = source / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    for name in ('gpu_policy.py', 'storage_accounting.py'):
        shutil.copyfile(repo / 'scripts/hpc' / name, source / name)
    (source / 'tests').mkdir()
    for path in (repo / 'experiments/brd4-jq1/protenix-random-chem/tests/test_protenix_chemistry_screen.py',
                 repo / 'tests/small_molecule/test_chemistry_severity.py'):
        shutil.copyfile(path, source / 'tests' / path.name)
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
                  arms=len(arms), updates_per_arm=20000, total_updates=len(arms)*20000,
                  checksums_sha256=hashlib.sha256((source/'checksums.json').read_bytes()).hexdigest())
    (local/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--tag', default='20261031-protenix-chemistry-strategies-01')
    parser.add_argument('--version', default='source01')
    args = parser.parse_args(); build(args.repo, args.tag, args.version)
