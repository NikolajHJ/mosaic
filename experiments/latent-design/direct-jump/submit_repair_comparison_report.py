"""Submit a joint CPU report for compatible completed or queued repair-search campaigns."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import runpy
import tarfile

from submit_repair_search import EXP, OPS


def merge_campaigns(folders):
    tasks, jobs, schedules, outputs, cells = {}, [], {}, set(), set()
    original = None
    for folder in folders:
        if (folder / 'superseded.json').exists():
            raise ValueError(f'Superseded comparison: {folder.name}')
        plan_bytes = (folder / 'gpu-plan.json').read_bytes()
        plan = json.loads(plan_bytes)
        commands = json.loads((folder / 'commands.json').read_bytes())
        if not commands or set(commands) != set(plan['tasks']):
            raise ValueError('GPU plan differs from commands')
        archive = folder / 'source.tar.gz'
        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        first = next(iter(commands.values()))
        config_name = first[first.index('--config') + 1]
        with tarfile.open(archive) as tar:
            config_bytes = tar.extractfile(config_name).read()
            # Test and submission files can change without affecting the optimizer.
            sources = {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar
                       if m.isfile() and (m.name.startswith('src/') or m.name.endswith('.sh') or
                           Path(m.name).name in {'study.py', 'protenix_native.py', 'repair_search.py',
                                                'repair_signal.py', 'repair_frontier.py', 'repair_report.py'})}
        if hashlib.sha256(config_bytes).hexdigest() != plan['config_sha256']:
            raise ValueError('Config checksum mismatch')
        config = json.loads(config_bytes)
        if config.pop('repair_weights') != plan['repair_weights']:
            raise ValueError('Plan differs from requested repair weights')
        settings = (config, sources, plan['cuda_ownership_sha256'])
        if original is None:
            original, source_command = settings, first
        elif settings != original:
            raise ValueError('Campaigns use different optimization settings or scientific source')
        arm_updates = config['founder_updates'] + (config['child_updates'] * config['parents'] *
                      config['children_per_parent'] * config['child_generations'])
        for name, task in plan['tasks'].items():
            command = commands[name]
            receipt = json.loads((folder / f'{name}-submission.json').read_bytes())
            expected = {'--archive-sha256': archive_hash, '--plan-sha256': hashlib.sha256(plan_bytes).hexdigest(),
                        '--output': task['output'], '-q': task['queue'], '--config': config_name, '--task': name,
                        '--seed': str(task['seed']), '--policy': task['policy']}
            if (receipt['argv'] != command or receipt['returncode'] or type(receipt['job_id']) is not int or
                    any(command[command.index(k)+1] != v for k, v in expected.items())):
                raise ValueError(f'Invalid confirmed GPU receipt for {name}')
            weights = task['repair_weights']
            panel = (task['policy'], task['seed'])
            task_cells = {(panel, w) for w in weights}
            if (task['updates'] != arm_updates * len(weights) or not weights or
                    len(weights) != len(task_cells) or not set(weights).issubset(plan['repair_weights']) or
                    task_cells & cells or name in tasks or task['output'] in outputs or receipt['job_id'] in jobs or
                    schedules.setdefault(panel, task['queue']) != task['queue']):
                raise ValueError(f'Duplicate or unmatched comparison for {name}')
            tasks[name] = task
            jobs.append(receipt['job_id'])
            outputs.add(task['output'])
            cells.update(task_cells)
    weights = sorted({w for _, w in cells})
    expected = {((p, s), w) for p in config['search_versions'] for s in config['seeds'] for w in weights}
    if cells != expected:
        raise ValueError('Incomplete matched panels')
    return {'repair_weights': weights, 'tasks': tasks, 'campaigns': [p.name for p in folders]}, jobs, source_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaigns', nargs='+')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    if (len(args.campaigns) < 2 or args.name in args.campaigns or
            not all(re.fullmatch(r'protenix-repair-[a-z0-9-]+', n) for n in [args.name, *args.campaigns])):
        parser.error('Use two or more campaign names and a distinct report name')
    ops = runpy.run_path(str(OPS))
    plan, jobs, gpu = merge_campaigns([Path('outputs') / n for n in args.campaigns])
    control = '/zhome/95/e/206121/dbmol-control/' + args.name
    output = ops['BASE'] + '/runs/' + args.name + '-report'
    plan['report_output'] = output
    local = Path('outputs') / args.name
    local.mkdir(exist_ok=True)
    # The existing launcher verifies and unpacks the original immutable GPU archive.
    phase = gpu.index('--archive')
    launcher = gpu[phase-2:gpu.index('--receipt')]
    remote_plan = control + '/gpu-plan.json'
    command = ['bsub', '-J', args.name + '-report', '-q', 'hpc', '-n', '1', '-R', 'rusage[mem=2048]',
               '-W', '00:20', '-w', ' && '.join(f'ended({j})' for j in jobs),
               '-oo', control + '/report/run.%J.out', '-eo', control + '/report/run.%J.err',
               *launcher, '--receipt', control + '/report/launch.json', '--startup-seconds', '1050',
               '--run-seconds', '1100', '--', '/usr/bin/python3', (EXP / 'repair_report.py').as_posix(),
               '--plan', remote_plan, '--output', output]
    for file, value in [('gpu-plan.json', plan), ('commands.json', {'report': command})]:
        path = local / file
        if path.exists() and ops['read'](path) != value:
            raise FileExistsError(path)
        if not path.exists():
            ops['save'](path, value)
    preflight = ops['remote'](f'''
import json
from pathlib import Path
path = Path({remote_plan!r})
plan = {plan!r}
path.parent.mkdir(exist_ok=True)
if path.exists():
    if json.loads(path.read_bytes()) != plan: raise FileExistsError(path)
else:
    with path.open('x') as stream: json.dump(plan, stream)
log = path.parent / 'report'
log.mkdir(exist_ok=True)
if Path({output!r}).exists() and not (log / 'submission.json').exists(): raise FileExistsError({output!r})
print(json.dumps(dict(status='passed')))
''')
    if not (local / 'storage.json').exists():
        ops['save'](local / 'storage.json', preflight)
    ops['submit'](local, control)


if __name__ == '__main__':
    main()
