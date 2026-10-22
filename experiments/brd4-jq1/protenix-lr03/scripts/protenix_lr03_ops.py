"""Independent Protenix LR0.3 CPU -> native -> long dispatch; no Boltz mutations."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from campaign25_contract import PROTENIX_LR03_VERSION, REFERENCE_SHA, config25, sha, verify_readiness
from campaign25_pocket_ops import final_job, read, write
from protenix_lr03_storage import RUN_ROOT, check_storage


CONFIG = 'examples/configs/brd4_jq1_25_protenix_lr03_10000.json'
AUTHORIZATION = {'version': 'jq1-protenix-lr0p3-user-authorization-v1',
                 'scope': 'One fresh 10000-update Protenix run; LR0.3 only scientific change',
                 'preserve_jobs': ['29770093', '29770124'], 'diffusion_gradient': False}


def submit(source, root, manifest, phase, *, producer=None):
    if phase not in ('validate', 'after-ready', 'run'):
        raise ValueError('Only independent Protenix phases are authorized')
    parent = root / (phase + '-protenix')
    parent.mkdir(exist_ok=False)
    gpu = phase != 'after-ready'
    command = ['bsub', '-q', 'gpul40s' if gpu else 'hpc', '-n', '4' if gpu else '1',
               '-R', 'span[hosts=1]', '-R', 'rusage[mem=8GB]' if gpu else 'rusage[mem=2GB]',
               '-W', '23:00' if phase == 'run' else ('02:00' if gpu else '00:20'),
               '-J', 'mosaic-jq1-protenix-lr03-' + phase,
               '-oo', str(parent / 'stdout.%J.log'), '-eo', str(parent / 'stderr.%J.log')]
    if gpu:
        command += ['-gpu', 'num=1:mode=exclusive_process']
    if producer is not None:
        if phase != 'after-ready' or not str(producer).isdigit():
            raise ValueError('Only native completion callback may have a dependency')
        command += ['-w', f'ended({producer})']
    command += ['bash', '-l', str(_layout_path(source / 'scripts/hpc/protenix_lr03_stage.lsf')),
                str(source), str(root), manifest, phase, str(parent), str(producer or '0')]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    receipt = {'submitted_at': datetime.now(timezone.utc).isoformat(), 'args': command,
               'stdout': reply.stdout, 'stderr': reply.stderr, 'returncode': reply.returncode,
               'phase': phase, 'arm': 'protenix', 'run': str(parent)}
    match = re.search(r'Job <(\d+)>', reply.stdout)
    if match:
        receipt['job_id'] = match[1]
    write(parent / 'submission.json', receipt)
    if reply.returncode or not match:
        raise RuntimeError('Unknown/failed submission; preserve receipt and never duplicate')
    raw = subprocess.run(['bjobs', '-l', match[1]], capture_output=True, text=True, timeout=30)
    (parent / 'scheduler-at-submission.txt').write_text(raw.stdout + raw.stderr)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'root', 'parent'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--phase', choices=('after-cpu', 'after-ready'), required=True)
    parser.add_argument('--producer', required=True)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    if root != RUN_ROOT or not source.is_relative_to(Path('/work3/s234818/mosaic/releases')):
        raise ValueError('Wrong independent release/output scope')
    subprocess.run([sys.executable, str(_layout_path(source / 'scripts/verify_stage0_release.py')), str(source), args.manifest], check=True)
    config = config25(read(source / CONFIG))
    if config['version'] != PROTENIX_LR03_VERSION:
        raise ValueError('Controller requires exact newly authorized LR0.3 version')
    producer_parent = root / ('cpu' if args.phase == 'after-cpu' else 'validate-protenix')
    status, queue, proofs = final_job(args.producer, producer_parent)
    if status != 'DONE' or queue != ('hpc' if args.phase == 'after-cpu' else 'gpul40s'):
        raise ValueError('Actual prerequisite scheduler/queue failed')
    write(root / 'user-authorization.json', AUTHORIZATION)
    if args.phase == 'after-cpu':
        from check_boltz_cpu_proof import verify
        assessment = verify(root / 'cpu/payload', args.manifest)
        pocket = root / 'cpu/payload/pocket_definition.json'
        if sha(pocket) != '9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43':
            raise ValueError('CPU-derived original JQ1 pocket changed')
        proofs += assessment['proofs'] + [{'path': str(p), 'sha256': sha(p)} for p in
                  (root / 'cpu/payload/assessment.json', pocket, root / 'cpu/payload/composition_reference.json')]
        if sha(root / 'cpu/payload/composition_reference.json') != REFERENCE_SHA:
            raise ValueError('Frozen reference changed')
        certificate = {'status': 'passed', 'source_manifest_sha256': args.manifest,
                       'producer_job_id': args.producer, 'tests': assessment['passed'], 'proofs': proofs}
        write(root / 'cpu-certificate.json', certificate)
        with (root / 'native-dispatch-started.json').open('x') as stream:
            json.dump({'cpu_certificate_sha256': sha(root / 'cpu-certificate.json')}, stream)
        native = submit(source, root, args.manifest, 'validate')
        callback = submit(source, root, args.manifest, 'after-ready', producer=native['job_id'])
        write(root / 'native-submissions.json', {'native': native, 'callback': callback, 'long_run_submitted': False})
    else:
        path = root / 'validate-protenix/payload/study/assessment.json'
        assessment = verify_readiness(path, sha(path), source / CONFIG, args.manifest)
        if assessment['context_digest'] != 'b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4':
            raise ValueError('Original Protenix pocket context changed')
        gate = check_storage(args.parent, assessment['storage_projection']['projected_bytes'])
        decision = {'status': 'passed', 'native_job': args.producer, 'assessment_sha256': sha(path),
                    'scheduler_proofs': proofs, 'storage_gate': gate, 'authorization': AUTHORIZATION}
        with (root / 'long-dispatch-started.json').open('x') as stream:
            json.dump(decision, stream, indent=2)
        write(root / 'long-submission.json', submit(source, root, args.manifest, 'run'))


if __name__ == '__main__':
    main()
