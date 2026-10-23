"""Duplicate-safe gates and16serial allocations extending source077 to100.

Login work authenticates bytes and submits argv only. Every numerical branch
runs inside an immutable allocation. All four old10epoch jobs remain intact.
"""
import argparse
import json
from pathlib import Path
import re

from latentmol.decoder_density_long_protocol import read_long_protocol, authenticate_base, segments
from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit
from submit_ramped_random import require_completed_stage


def base_last_job(protocol):
    receipts = Path(protocol['base_receipts_root'])
    identity = read(receipts / 'identity.json')
    if (identity['archive_sha256'] != protocol['base_archive_sha256']
            or identity['manifest_sha256'] != protocol['base_manifest_sha256']
            or identity['config_sha256'] != protocol['base_config_sha256']
            or identity['output_root'] != protocol['base_campaign_root']):
        raise ValueError('Base scheduler lineage differs')
    # source077 was serialized with ended(previous), so the final reservation
    # ending proves that all four prefixes have ended, not that they succeeded.
    receipt = read(receipts / 'seed23-a1-receipt.json')
    if receipt['intent']['identity'] != identity or not re.fullmatch('[0-9]+', receipt['job_id']):
        raise ValueError('Base final job receipt is not authenticated')
    return receipt['job_id']


def workflow(release, work, protocol, base, config_sha, archive_sha, manifest_sha,
             stage, cpu_job=None, gate_job=None, base_job=None):
    declarations = segments(base, protocol) if stage == 'segments' else [{'label': stage}]
    rows = []
    for index, item in enumerate(declarations):
        label = release.name + '-' + item['label']
        cpu = stage == 'cpu'
        argv = ['bsub', '-J', label, '-q', 'hpc' if cpu else 'gpul40s', '-n', '2' if cpu else '4',
            '-R', 'rusage[mem=4GB]' if cpu else 'rusage[mem=16GB]', '-R', 'span[hosts=1]',
            '-W', '0:45' if cpu else '2:00' if stage == 'gate' else '23:00',
            '-oo', str(work / 'logs' / (label + '.%J.out')),
            '-eo', str(work / 'logs' / (label + '.%J.err'))]
        if not cpu:
            dependency = f'done({cpu_job})'
            if stage == 'segments':
                dependency += f' && done({gate_job}) && ended({base_job})'
            argv += ['-gpu', 'num=1:mode=exclusive_process:j_exclusive=yes', '-w', dependency]
        root = work / 'runs' / (release.name + '-density100-study')
        tail = ['bash', 'scripts/hpc/decoder_density_long_tests.sh'] if cpu else [
            'bash', 'scripts/hpc/decoder_density_long_run.sh',
            str(release / 'configs/decoder_density_100_study.json'), config_sha, str(root),
            'segment' if stage == 'segments' else 'gate']
        if stage == 'segments':
            tail += ['--segment', str(index)]
        command = ['env', f'LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz', f'LATENTMOL_WORK_ROOT={work}',
            'bash', 'scripts/hpc/boltz_run.sh', 'python', 'scripts/verify_stability_release.py',
            '--archive-sha256', archive_sha, '--manifest-sha256', manifest_sha, '--',
            'python', '-m', 'latentmol.experiments', '--run-root', str(work / 'runs' / label),
            '--name', 'density100-' + item['label'], '--', *tail]
        rows.append({'stage': item['label'], 'cwd': str(release), 'arguments': argv,
            'job_command': command, 'dependency': None,
            'serial_after_stage': declarations[index - 1]['label'] if stage == 'segments' and index else None})
    return rows


def serial_after(row, preceding):
    if not isinstance(preceding, str) or not re.fullmatch('[0-9]+', preceding):
        raise ValueError('Need numeric preceding allocation ID')
    argv = list(row['arguments'])
    index = argv.index('-w') + 1
    argv[index] += f' && ended({preceding})'
    return {**row, 'arguments': argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu', 'gate', 'segments'), required=True)
    parser.add_argument('--release', type=Path, default=Path.cwd())
    parser.add_argument('--work-root', type=Path, default=Path('/work3/s234818/latentmol'))
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if (not str(work).startswith('/work3/') or not release.is_relative_to(work / 'releases')
            or not args.receipts.is_absolute() or not args.receipts.resolve().is_relative_to(work)):
        raise ValueError('Require contained immutable release and receipts')
    authenticated(Path(str(release) + '.tar.gz'), args.archive_sha256)
    manifest = source_manifest(release)['sha256']
    if read(release / 'release_manifest.json')['sha256'] != manifest:
        raise ValueError('Long source changed after sealing')
    path = release / 'configs/decoder_density_100_study.json'
    config_sha = file_sha256(path)
    protocol = read_long_protocol(path, config_sha)
    base = authenticate_base(protocol)
    old_job = base_last_job(protocol)
    authenticated(Path(base['initializer']), base['initializer_sha256'])
    authenticated(Path(base['data']) / 'dataset.npz', base['dataset_sha256'])
    if read(Path(base['data']) / 'manifest.json')['split_sha256'] != base['split_sha256']:
        raise ValueError('Long protected split changed')
    identity = {'release': str(release), 'archive_sha256': args.archive_sha256,
        'manifest_sha256': manifest, 'config_sha256': config_sha, 'protocol': protocol,
        'initializer_sha256': base['initializer_sha256'], 'dataset_sha256': base['dataset_sha256'],
        'split_sha256': base['split_sha256'], 'base_final_job_id': old_job,
        'output_root': str(work / 'runs' / (release.name + '-density100-study'))}
    cpu_job = require_completed_stage(args.receipts, 'cpu', identity) if args.stage != 'cpu' else None
    gate_job = require_completed_stage(args.receipts, 'gate', identity) if args.stage == 'segments' else None
    if gate_job:
        gate = read(Path(identity['output_root']) / 'gate/passed.json')
        if (gate['status'] != 'passed' or gate['protocol'] != protocol
                or gate['source_archive_sha256'] != args.archive_sha256):
            raise ValueError('Long native gate belongs to another campaign')
    rows = workflow(release, work, protocol, base, config_sha, args.archive_sha256, manifest,
        args.stage, cpu_job, gate_job, old_job)
    if not args.submit:
        print(json.dumps({'identity': identity, 'plan': rows}, indent=2))
        return
    (work / 'logs').mkdir(exist_ok=True)
    jobs, preceding = {}, None
    for row in rows:
        if preceding is not None:
            row = serial_after(row, preceding)
        jobs.update(submit([row], args.receipts, identity))
        preceding = jobs[row['stage']]
    print(json.dumps(jobs), flush=True)


if __name__ == '__main__':
    main()
