"""Submit an engineering gate and a bounded serial replay of failed states.

This diagnostic retains the original production release and exact saved states.
It does not optimize z, call DeFoG, change a tolerance, or approve a failed run.
Run from a sealed release on the DTU login node; all model work is inside LSF.
Durable scheduler intents prevent duplicate jobs after an unknown SSH outcome.
"""
import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_stability_study import authenticated, read, submit
from submit_ramped_random import gpu_request


DIAGNOSTICS = {
    'affinity_decoder_causal_diagnostic': {
        'fixtures': ['tests/test_affinity_decoder_causal_diagnostic.py',
                     'tests/test_decoder_causal_diagnostic.py',
                     'tests/test_affinity_causal_diagnostic.py',
                     'tests/test_head_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '1:00',
        'scope': 'Original prior5 regression-only canonical graph cut and decoder-only held-abs diagnostic; no production patch, latent updates, DeFoG sampling or historical gate approval',
    },
    'affinity_causal_diagnostic': {
        'fixtures': ['tests/test_affinity_causal_diagnostic.py',
                     'tests/test_native_head_gradient_diagnostic.py',
                     'tests/test_head_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '2:00',
        'scope': 'Prespecified original prior5 affinity-ensemble/MLP causal cuts and declared cached precision/activation controls; no production patch, latent updates, DeFoG sampling or historical gate approval',
    },
    'failed_gate_attribution': {
        'fixtures': ['tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '4:00',
        'scope': 'Original production math; no optimization or DeFoG sampling',
    },
    'attention_causal_diagnostic': {
        'fixtures': ['tests/test_attention_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '4:00',
        'scope': 'Prespecified native attention arithmetic/backward diagnostic variants; no production patch, optimization or DeFoG sampling',
    },
    'decoder_causal_diagnostic': {
        'fixtures': ['tests/test_decoder_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py', 'tests/test_defog_single_recovery.py'],
        'wall': '1:00',
        'scope': 'Original native graph cotangent followed by declared decoder-only precision and branch controls; no production patch, optimization or DeFoG sampling',
    },
    'head_causal_diagnostic': {
        'fixtures': ['tests/test_head_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '2:00',
        'scope': 'Prespecified native output-head cotangent replay and declared cached reduction controls; no production patch, optimization or DeFoG sampling',
    },
    'native_head_gradient_diagnostic': {
        'fixtures': ['tests/test_native_head_gradient_diagnostic.py',
                     'tests/test_head_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '2:00',
        'scope': 'Original prior5 native-head latent-VJP attribution; no production arithmetic change, optimization or DeFoG sampling',
    },
    'precision_gradient_diagnostic': {
        'fixtures': ['tests/test_dbmol_precision.py', 'tests/test_dbmol_objective.py',
                     'tests/test_dbmol_model.py', 'tests/test_precision_gradient_diagnostic.py',
                     'tests/test_native_head_gradient_diagnostic.py',
                     'tests/test_head_causal_diagnostic.py',
                     'tests/test_failed_gate_attribution.py', 'tests/test_boltz_local_fd.py',
                     'tests/test_saved_state_submission.py'],
        'wall': '2:00',
        'scope': 'Explicit FP64-contact/DBMol-assembly candidate validation for prior0, then original prior5 native-head attribution; no latent updates, DeFoG sampling or historical gate approval',
    },
}


def workflow(release, work, plan, plan_sha, archive_sha, manifest_sha,
             *, diagnostic='failed_gate_attribution'):
    """Build one registered observer's jobs without model/scheduler execution.

    The default preserves source080's workflow. New observers use the same
    source/plan identity and durable submission gates, with an explicit scope.
    This registry is intentionally limited to reviewed saved-state diagnostics.
    """
    if diagnostic not in DIAGNOSTICS:
        raise ValueError('Unregistered saved-state diagnostic')
    spec = DIAGNOSTICS[diagnostic]
    stages = [
        ('cpu', 'hpc', '0:30', 2, '4GB', None,
         ['python', '-m', 'pytest', '-q', *spec['fixtures']]),
        ('native', 'gpua100', spec['wall'], 4, '16GB', 'done:cpu',
         ['python', '-m', 'latentmol.boltz.' + diagnostic,
          '--plan', str(plan), '--sha256', plan_sha]),
    ]
    rows = []
    for stage, queue, wall, cores, memory, dependency, command in stages:
        label = release.name + '-' + stage
        arguments = ['bsub', '-J', label, '-q', queue, '-n', str(cores),
            '-R', 'rusage[mem=' + memory + ']', '-R', 'span[hosts=1]', '-W', wall,
            '-oo', str(work / 'logs' / (label + '.%J.out')),
            '-eo', str(work / 'logs' / (label + '.%J.err'))]
        if stage == 'native':
            # Retain the original GPU class and LSF-managed MPS arithmetic
            # environment. Only the Boltz replay child holds CUDA here; no
            # DeFoG subprocess or manually managed GPU service is launched.
            arguments += ['-R', 'select[gpu80gb]', '-gpu', gpu_request(managed_mps=True)]
        job = ['env', 'LATENTMOL_SOURCE_ARCHIVE=' + str(release) + '.tar.gz',
            'LATENTMOL_WORK_ROOT=' + str(work), 'bash', 'scripts/hpc/boltz_run.sh',
            'python', 'scripts/verify_stability_release.py',
            '--archive-sha256', archive_sha, '--manifest-sha256', manifest_sha, '--',
            'python', '-m', 'latentmol.experiments', '--run-root', str(work / 'runs'),
            '--name', label, '--', *command]
        rows.append({'stage': stage, 'cwd': str(release), 'arguments': arguments,
                     'job_command': job, 'dependency': dependency})
    return rows


def validate_submission_plan(plan, work, release, diagnostic):
    """Retain historical three-state plans and the exact source091 failure pair.

    Two-state support is deliberately restricted to the control/guided step-15
    failures. This is a saved-state observer, not a replacement pilot or a
    permission to alter starts, weights, tolerances or production objectives.
    """
    cases = plan.get('cases', [])
    historical = isinstance(cases, list) and len(cases) == 3
    matched = (diagnostic == 'failed_gate_attribution' and isinstance(cases, list)
        and [case.get('id') for case in cases] == ['boltz_only', 'boltz_defog']
        and all(case.get('arm') == 'fixed_initial_mask' and case.get('step') == 15 for case in cases))
    if (plan.get('version') != 1 or not (historical or matched)
            or Path(plan['output_root']) != work / 'runs' / (release.name + '-replay')):
        raise ValueError('Require the historical three-state or exact matched step15 diagnostic plan and unique output root')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, default=Path.cwd())
    parser.add_argument('--work-root', type=Path, default=Path('/work3/s234818/latentmol'))
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--receipts', type=Path, required=True)
    parser.add_argument('--diagnostic', choices=tuple(DIAGNOSTICS), default='failed_gate_attribution')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith('/work3/') or not release.is_relative_to(work / 'releases'):
        raise ValueError('Require a sealed release under the DTU work root')
    if any(not path.is_absolute() or not path.resolve().is_relative_to(work)
           for path in (args.plan, args.receipts)):
        raise ValueError('Plan and receipt paths must stay under the declared DTU root')
    authenticated(Path(str(release) + '.tar.gz'), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    if (source_manifest(release)['sha256'] != args.manifest_sha256
            or read(release / 'release_manifest.json')['sha256'] != args.manifest_sha256):
        raise ValueError('Scientific source changed after sealing')
    plan_record = read(args.plan)
    validate_submission_plan(plan_record, work, release, args.diagnostic)
    identity = {'release': str(release), 'archive_sha256': args.archive_sha256,
        'manifest_sha256': args.manifest_sha256, 'plan': str(args.plan),
        'plan_sha256': args.plan_sha256,
        'scope': ('Diagnostic only; original production math, no optimization/DeFoG sampling'
                  if args.diagnostic == 'failed_gate_attribution' else DIAGNOSTICS[args.diagnostic]['scope'])}
    if args.diagnostic != 'failed_gate_attribution':
        identity['diagnostic'] = args.diagnostic
    jobs = workflow(release, work, args.plan, args.plan_sha256, args.archive_sha256,
                    args.manifest_sha256, diagnostic=args.diagnostic)
    if args.submit:
        (work / 'logs').mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({'identity': identity, 'plan': jobs, 'status': 'not_submitted'}, indent=2))


if __name__ == '__main__':
    main()
