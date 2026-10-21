"""Dispatch fresh fixed-step JQ1 runs after full CPU/native/shared-storage gates."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys

from campaign25_contract import ARMS, FIXED25_VERSION, REFERENCE_SHA, config25, sha, verify_readiness
from campaign25_pocket_ops import final_job, read, write
from fixed25_storage import RUN_ROOT, check_storage

POCKET = '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43'
CONTEXTS = {'protenix': 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4',
            'boltz2': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4',
            'boltz2_chemistry': 'fd04211da8e22eccd97ecedd96dd59cf4fb2d80aab6a397fa662af8eb92bbad4'}
AUTHORIZATION = {'version': 'jq1-25-fixed-step-user-authorization-v1', 'fresh_updates_per_arm': 10000,
                 'rates': {'protenix': 0.3, 'boltz2': 0.01, 'boltz2_chemistry': 0.01},
                 'acceptance_policy': 'fixed_step', 'diffusion_gradient': False,
                 'historical_jobs_intentionally_stopped': ['29770093', '29770124']}


def config_path(source, arm):
    if arm not in ARMS:
        raise ValueError('Unknown fixed-step arm')
    return source / f'examples/configs/brd4_jq1_25_fixed_{arm}_10000.json'


def submit(source, root, manifest, phase, *, arm, producer=None):
    if phase not in ('validate', 'after-ready', 'run') or arm not in ARMS:
        raise ValueError('Unknown fixed-step arm/phase')
    if producer is not None and (phase != 'after-ready' or not str(producer).isdigit()):
        raise ValueError('Only its own readiness callback may depend on a producer')
    parent = root / (phase + '-' + arm)
    parent.mkdir(exist_ok=False)
    gpu = phase != 'after-ready'
    command = ['bsub', '-q', 'gpul40s' if gpu else 'hpc', '-n', '4' if gpu else '1',
               '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]',
               '-W', '23:00' if phase == 'run' else ('02:00' if gpu else '00:20'),
               '-J', 'mosaic-jq1-fixed-' + phase + '-' + arm,
               '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if gpu:
        command += ['-gpu', 'num=1:mode=exclusive_process']
    if producer is not None:
        command += ['-w', f'ended({producer})']
    command += ['bash', '-l', str(source / 'scripts/hpc/fixed25_stage.lsf'), str(source), str(root),
                manifest, phase, arm, str(parent), str(producer or '0')]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    receipt = {'submitted_at': datetime.now(timezone.utc).isoformat(), 'args': command,
               'stdout': reply.stdout, 'stderr': reply.stderr, 'returncode': reply.returncode,
               'phase': phase, 'arm': arm, 'run': str(parent)}
    match = re.search(r'Job <(\d+)>', reply.stdout)
    if match:
        receipt['job_id'] = match[1]
    write(parent / 'submission.json', receipt)
    if reply.returncode or not match:
        raise RuntimeError('Unknown/failed submission; preserve receipt, no duplicate')
    raw = subprocess.run(['bjobs', '-l', match[1]], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(raw.stdout + raw.stderr)
    return receipt


def dispatch(source, root, manifest):
    paths = [root / ('decision-' + arm + '.json') for arm in ARMS]
    if not all(p.exists() for p in paths):
        return
    decisions = [read(p) for p in paths]
    if not all(d['status'] == 'passed' for d in decisions):
        raise ValueError('A native readiness failed; no fixed-step production dispatch')
    if (root / 'long-dispatch-started.json').exists():
        return
    for arm, d in zip(ARMS, decisions):
        verify_readiness(root / f'validate-{arm}/payload/study/assessment.json', d['assessment_sha256'], config_path(source, arm), manifest)
    projection = sum(d['projected_bytes'] for d in decisions)
    gate = check_storage(root / 'shared-dispatch-quota', projection)
    with (root / 'long-dispatch-started.json').open('x') as stream:
        json.dump({'decisions': decisions, 'shared_storage_gate': gate, 'authorization': AUTHORIZATION}, stream, indent=2)
    # Exclusive marker prevents resubmission after any unknown/partial outcome.
    jobs = []
    for arm in ARMS:
        jobs.append(submit(source, root, manifest, 'run', arm=arm))
        write(root / 'long-submissions.json', {'jobs': jobs, 'all_three_submitted': len(jobs) == 3})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'root', 'parent'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--phase', choices=('after-cpu', 'after-ready'), required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--producer', required=True)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    if root != RUN_ROOT or not source.is_relative_to(Path('/work3/s234818/mosaic/releases')):
        raise ValueError('Wrong fixed-step source/output scope')
    subprocess.run([sys.executable, str(source / 'scripts/verify_stage0_release.py'), str(source), args.manifest], check=True)
    for arm in ARMS:
        if config25(read(config_path(source, arm)))['version'] != FIXED25_VERSION:
            raise ValueError('Fixed-step controller cannot dispatch a historical policy')
    parent = root / ('cpu' if args.phase == 'after-cpu' else 'validate-' + str(args.arm))
    status, queue, proofs = final_job(args.producer, parent)
    if status != 'DONE' or queue != ('hpc' if args.phase == 'after-cpu' else 'gpul40s'):
        if args.phase == 'after-ready' and args.arm:
            write(root / ('decision-' + args.arm + '.json'), {'status': 'failed', 'native_job': args.producer, 'scheduler_proofs': proofs})
        raise ValueError('Actual fixed-step prerequisite scheduler/queue failed')
    write(root / 'user-authorization.json', AUTHORIZATION)
    if args.phase == 'after-cpu':
        from check_boltz_cpu_proof import verify
        assessment = verify(root / 'cpu/payload', args.manifest)
        pocket = root / 'cpu/payload/pocket_definition.json'
        if sha(pocket) != POCKET or sha(root / 'cpu/payload/composition_reference.json') != REFERENCE_SHA:
            raise ValueError('Original pocket/reference identity changed')
        proofs += assessment['proofs'] + [{'path': str(p), 'sha256': sha(p)} for p in
                  (root / 'cpu/payload/assessment.json', pocket, root / 'cpu/payload/composition_reference.json')]
        write(root / 'cpu-certificate.json', {'status': 'passed', 'source_manifest_sha256': args.manifest,
              'producer_job_id': args.producer, 'tests': assessment['passed'], 'proofs': proofs})
        with (root / 'native-dispatch-started.json').open('x') as stream:
            json.dump({'cpu_certificate_sha256': sha(root / 'cpu-certificate.json')}, stream)
        jobs = []
        for arm in ARMS:
            native = submit(source, root, args.manifest, 'validate', arm=arm)
            callback = submit(source, root, args.manifest, 'after-ready', arm=arm, producer=native['job_id'])
            jobs.append({'arm': arm, 'native': native, 'callback': callback})
            write(root / 'native-submissions.json', {'jobs': jobs, 'long_runs_submitted': False})
    else:
        if not args.arm:
            raise ValueError('Readiness callback requires its own arm')
        path = parent / 'payload/study/assessment.json'
        assessment = verify_readiness(path, sha(path), config_path(source, args.arm), args.manifest)
        if assessment['context_digest'] != CONTEXTS[args.arm] or assessment['pocket_definition_sha256'] != POCKET:
            raise ValueError('Original frozen JQ1 context/pocket changed')
        with (root / 'dispatch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            write(root / ('decision-' + args.arm + '.json'), {'status': 'passed', 'native_job': args.producer,
                  'assessment_sha256': sha(path), 'projected_bytes': assessment['storage_projection']['projected_bytes'], 'scheduler_proofs': proofs})
            dispatch(source, root, args.manifest)


if __name__ == '__main__':
    main()
