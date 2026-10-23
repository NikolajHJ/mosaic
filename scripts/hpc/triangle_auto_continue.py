"""External, fail-closed LSF continuation of the unchanged source098 campaign.

CPU callbacks wait for ended(job), record failures, and only submit after DONE
and authenticated completion/health evidence. There is at most one next GPU
allocation and one callback. A scheduler timeout/ambiguous receipt stops the
chain; failed scientific allocations require explicit recovery, never blind
retries. No scientific model, loss, data, checkpoint or RNG is modified here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

from triangle_continuation_health import strict_json, validate_health


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return strict_json(Path(path).read_text())


def fresh(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def utc():
    return datetime.now(timezone.utc).isoformat()


def contained(path, root):
    path, root = Path(path).resolve(strict=True), Path(root).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Evidence path escapes its declared allocation: ' + str(path))
    return path


def required_quota_gib(epoch, segment_number, max_segments=20):
    """Reserve64MiB/remaining epoch,256MiB/possible allocation,2GiB final/headroom.

    Current checkpoints are ~47.6MB including Adam. This bounds retained epoch
    snapshots and per-allocation aliases/inherited selections without deleting
    evidence. Exhausted quota stops submission; the budget is not a reservation.
    """
    remaining = 100 - epoch
    allocations = min(remaining, max_segments - segment_number)
    return Decimal(2) + Decimal(remaining) / 16 + Decimal(allocations) / 4


class Controller:
    def __init__(self, policy_path, policy_sha):
        self.bundle = Path(__file__).resolve().parents[2]
        manifest = read(self.bundle / 'operator-manifest.json')
        for relative, digest in manifest['files'].items():
            if sha(contained(self.bundle / relative, self.bundle)) != digest:
                raise ValueError('Immutable operator bundle differs: ' + relative)
        # The submission command pins the policy; it in turn pins the manifest.
        if sha(policy_path) != policy_sha:
            raise ValueError('Controller policy differs')
        self.policy = read(policy_path)
        if sha(self.bundle / 'operator-manifest.json') != self.policy['operator_manifest_sha256']:
            raise ValueError('Operator manifest is not the pinned manifest')
        self.policy_path, self.policy_sha = Path(policy_path).resolve(), policy_sha
        self.root = Path(self.policy['control_root'])
        self.root.mkdir(parents=True, exist_ok=True)
        original = Path(self.policy['original_operator'])
        if sha(original) != self.policy['original_operator_sha256']:
            raise ValueError('Original source098 metadata operator differs')
        spec = importlib.util.spec_from_file_location('_source098_operator', original)
        self.original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.original)
        self.release, self.campaign = self.original.RELEASE, self.original.CAMPAIGN
        self.original.authenticated(Path(str(self.release) + '.tar.gz'), self.original.ARCHIVE_SHA)
        self.original.authenticated(self.original.CONTROL, self.original.BINDINGS_SHA)
        self.original.authenticated(self.original.LAUNCHER, self.policy['launcher_sha256'])
        sys.path.insert(0, str(self.release / 'src'))
        sys.path.insert(0, str(self.release / 'scripts/hpc'))
        from latentmol.experiments import source_manifest
        from large_triangle_evidence import require_submitted_stage, require_stage_completion
        from submit_large_triangle import require_completed_stage
        from submit_stability_study import submit
        if source_manifest(self.release)['sha256'] != self.original.MANIFEST_SHA:
            raise ValueError('Immutable scientific source098 changed')
        self.submitted, self.completed, self.require_done, self.submit = (
            require_submitted_stage, require_stage_completion, require_completed_stage, submit)
        self.identity = {'release': str(self.release), 'campaign': str(self.campaign),
                         'archive_sha256': self.original.ARCHIVE_SHA,
                         'manifest_sha256': self.original.MANIFEST_SHA,
                         'bindings_sha256': self.original.BINDINGS_SHA}
        if self.identity != self.policy['scientific_identity']:
            raise ValueError('Controller attached to wrong scientific campaign')
        for stage in ('cpu', 'cache', 'benchmark'):
            self.require_done(self.campaign / 'receipts', stage, self.identity)
        if self.policy['max_segments'] != 20 or self.policy['planned_epochs'] != 100:
            raise ValueError('Require bounded declared100epoch campaign')
        self.operator_identity = {'policy_sha256': policy_sha,
                                  'operator_manifest_sha256': self.policy['operator_manifest_sha256'],
                                  'scientific_identity': self.identity}

    def cpu_gate(self):
        path = self.root / 'cpu-passed.json'
        gate = read(path)
        expected_test = [str(self.original.PYTHON), '-m', 'pytest', '-q',
                         str(self.bundle / 'tests/test_triangle_auto_continue.py')]
        if (gate['status'] != 'passed' or gate['identity'] != self.operator_identity
                or not re.fullmatch('[0-9]+', gate.get('job_id', ''))
                or gate.get('returncode') != 0 or gate.get('command') != expected_test):
            raise ValueError('Continuation CPU gate differs')
        receipt = read(self.root / 'receipts/cpu-receipt.json')
        intent = read(self.root / 'receipts/cpu-intent.json')
        if (receipt.get('intent') != intent or intent.get('identity') != self.operator_identity
                or receipt.get('job_id') != gate['job_id']
                or read(self.root / 'receipts/identity.json') != self.operator_identity
                or intent['command'][-1] != self.cpu_job_command()):
            raise ValueError('CPU gate does not match the sealed submission receipt')
        reply = subprocess.run(['bjobs', '-a', '-noheader', '-o', 'stat', gate['job_id']],
                               capture_output=True, text=True, timeout=30)
        if reply.stdout.strip() not in ('DONE', '') or (reply.returncode and 'not found' not in reply.stderr.lower()):
            raise ValueError('Continuation CPU allocation is not completed')

    def cpu_job_command(self):
        import shlex
        return shlex.join([str(self.original.PYTHON),
            str(self.bundle / 'scripts/hpc/triangle_auto_continue.py'),
            '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha, '--mode', 'cpu'])

    def queue_cpu(self):
        import shlex
        arguments = ['bsub', '-J', 'source100-continuation-cpu', '-q', 'hpc', '-n', '1',
                     '-R', 'rusage[mem=4GB]', '-W', '0:15',
                     '-oo', str(self.original.WORK / 'logs/source100-continuation-cpu.%J.out'),
                     '-eo', str(self.original.WORK / 'logs/source100-continuation-cpu.%J.err')]
        return self.submit([{'stage': 'cpu', 'cwd': str(self.bundle), 'arguments': arguments,
                             'job_command': shlex.split(self.cpu_job_command()), 'dependency': None}],
                           self.root / 'receipts', self.operator_identity)

    def queue_callback(self, stage, job):
        number = stage_number(stage)
        if not 1 <= number <= self.policy['max_segments']:
            raise ValueError('Controller allocation bound exceeded')
        actual, _ = self.submitted(self.campaign / 'receipts', stage, self.identity)
        if actual != job:
            raise ValueError('Wrong predecessor scheduler job')
        key = 'after-' + stage
        command = [str(self.original.PYTHON), str(self.bundle / 'scripts/hpc/triangle_auto_continue.py'),
                   '--policy', str(self.policy_path), '--policy-sha256', self.policy_sha,
                   '--mode', 'advance', '--after-stage', stage]
        arguments = ['bsub', '-J', 'source100-' + key, '-q', 'hpc', '-n', '1',
                     '-R', 'rusage[mem=4GB]', '-W', '0:15', '-w', 'ended(' + job + ')',
                     '-oo', str(self.original.WORK / 'logs' / ('source100-' + key + '.%J.out')),
                     '-eo', str(self.original.WORK / 'logs' / ('source100-' + key + '.%J.err'))]
        plan = [{'stage': key, 'cwd': str(self.bundle), 'arguments': arguments,
                 'job_command': command, 'dependency': None}]
        return self.submit(plan, self.root / 'receipts', self.operator_identity)[key]

    def completed_health(self, stage):
        number = stage_number(stage)
        self.require_done(self.campaign / 'receipts', stage, self.identity)
        job, _ = self.submitted(self.campaign / 'receipts', stage, self.identity)
        if any((self.campaign / name).exists() for name in ('active-segment.json', 'active-recovery.json')):
            raise ValueError('Training/recovery lock remains active')
        progress = read(self.campaign / 'progress.json')
        completion = read(self.campaign / 'stage-completions' / (stage + '.json'))
        gate = read(self.campaign / 'benchmark/passed.json')
        selected = self.campaign / 'benchmark/selected-training.yaml'
        protocol_path = self.release / 'configs/large_triangle_protocol.json'
        expected = {'gate_sha256': sha(self.campaign / 'benchmark/passed.json'),
                    'selected_config_sha256': sha(selected), 'protocol_sha256': sha(protocol_path),
                    'dataset_sha256': gate['dataset_sha256'], 'split_sha256': gate['split_sha256'],
                    'source_sha256': self.original.MANIFEST_SHA, 'source_archive_sha256': self.original.ARCHIVE_SHA}
        if (progress['identity'] != expected or progress['protocol_sha256'] != expected['protocol_sha256']
                or progress['selected_config_sha256'] != expected['selected_config_sha256']):
            raise ValueError('Progress belongs to another training lineage')
        segment = self.campaign / 'segments' / f'{number:03d}'
        intent = read(segment / 'intent.json')
        if intent['identity'] != expected or intent['job_id'] != job:
            raise ValueError('Allocation intent differs from completed job')
        parent = contained(progress['run_path'], segment)
        if parent.parent != segment:
            raise ValueError('Unexpected nested training run path')
        checkpoint = contained(progress['checkpoint'], parent)
        if (checkpoint != parent / 'last.pt' or sha(checkpoint) != progress['checkpoint_sha256']
                or completion['artifacts'].get(str(checkpoint)) != progress['checkpoint_sha256']):
            raise ValueError('Current checkpoint is not the completed allocation artifact')
        record = read(parent / 'run.json')
        rows = [strict_json(line) for line in (parent / 'metrics.jsonl').read_text().splitlines()]
        latest = read(parent / 'latest_epoch.json')
        previous_epoch, resume = 0, None
        if number == 1:
            if job != self.policy['first_job_id'] or intent.get('resume_from') is not None:
                raise ValueError('Wrong initial allocation')
        else:
            preceding = self.root / 'decisions' / f'segment-{number-1:03d}' / 'decision.json'
            decision = read(preceding)
            if (decision['status'] != 'continued' or decision['next_stage'] != stage
                    or decision['next_job_id'] != job or decision['identity'] != self.operator_identity):
                raise ValueError('No authenticated automatic predecessor decision')
            previous_epoch = decision['health']['through_epoch']
            resume = {'checkpoint': decision['checkpoint'], 'sha256': decision['checkpoint_sha256'],
                      'completed_epoch': previous_epoch, 'parent_run_id': decision['run_id']}
            previous_stage = f'segment-{number-1:03d}'
            self.require_done(self.campaign / 'receipts', previous_stage, self.identity)
            old_completion = read(self.campaign / 'stage-completions' / (previous_stage + '.json'))
            if (old_completion['artifacts'].get(resume['checkpoint']) != resume['sha256']
                    or intent['resume_from'] != resume['checkpoint']):
                raise ValueError('Resume path differs from the previous completed checkpoint')
        import yaml  # Metadata parser in the pinned HPC environment; no model imports.
        config, protocol = yaml.safe_load(selected.read_text()), read(protocol_path)
        health = validate_health(record, rows, latest, previous_epoch=previous_epoch, expected_job=job,
                                 identity=expected, config=config, protocol=protocol,
                                 runtime=gate['runtime'], expected_resume=resume)
        if progress['epoch'] != health['through_epoch'] or progress['stop_reason'] != health['stop_reason']:
            raise ValueError('Completed progress and training record differ')
        if number >= self.policy['max_segments'] and health['through_epoch'] < 100:
            raise ValueError('Maximum automatic allocation count reached before100epochs')
        self.original.MINIMUM_FREE_GIB = required_quota_gib(health['through_epoch'], number, self.policy['max_segments'])
        quota = self.original.quota_headroom()
        paths = [parent / 'run.json', parent / 'metrics.jsonl', parent / 'latest_epoch.json',
                 segment / 'intent.json', self.campaign / 'progress.json',
                 self.campaign / 'stage-completions' / (stage + '.json')]
        return {'health': health, 'quota': quota, 'checkpoint': str(checkpoint),
                'checkpoint_sha256': progress['checkpoint_sha256'], 'run_id': record['run_id'],
                'evidence_sha256': {str(path): sha(path) for path in paths}}

    def advance(self, stage):
        self.cpu_gate()
        number = stage_number(stage)
        output = self.root / 'decisions' / stage
        output.mkdir(parents=True, exist_ok=True)
        # Reservation occurs before any mutation. Unknown outcomes stay stopped;
        # an operator inspects both these receipts and source098 GPU receipts.
        fresh(output / 'intent.json', {'utc': utc(), 'stage': stage, 'identity': self.operator_identity,
                                      'controller_job_id': os.environ.get('LSB_JOBID')})
        decision = {'utc': utc(), 'identity': self.operator_identity, 'after_stage': stage}
        try:
            decision.update(self.completed_health(stage))
            final = decision['health']['through_epoch'] == 100
            next_id = 'auto100' if final else f'{number+1:03d}'
            next_kind = 'final' if final else 'segment'
            command = [str(self.original.PYTHON), str(self.original.LAUNCHER), '--stage', next_kind,
                       '--archive-sha256', self.original.ARCHIVE_SHA,
                       '--manifest-sha256', self.original.MANIFEST_SHA,
                       '--control-sha256', self.original.BINDINGS_SHA,
                       '--run-id', next_id, '--after-stage', stage]
            if not final:
                command += ['--segment-id', next_id]
            fresh(output / 'submission-command.json', {'command': command, 'identity': self.operator_identity})
            reply = subprocess.run(command, cwd=self.release, capture_output=True, text=True, timeout=180)
            fresh(output / 'submission-reply.json', {'returncode': reply.returncode,
                                                   'stdout': reply.stdout, 'stderr': reply.stderr})
            reply.check_returncode()
            next_stage = next_kind + '-' + next_id
            next_job, _ = self.submitted(self.campaign / 'receipts', next_stage, self.identity)
            decision.update(status='final_evaluation_submitted' if final else 'continued',
                            next_stage=next_stage, next_job_id=next_job)
            # Persist lineage before a child callback can start. The separate
            # arming receipt distinguishes a submitted GPU from a chained GPU.
            fresh(output / 'decision.json', decision)
            if not final:
                callback = self.queue_callback(next_stage, next_job)
                fresh(output / 'callback.json', {'next_callback_job_id': callback,
                      'next_job_id': next_job, 'identity': self.operator_identity})
                decision['next_callback_job_id'] = callback
            print(json.dumps(decision, indent=2))
        except BaseException as exc:
            decision.update(status='stopped_inspect_before_retry', error=str(exc),
                            exception_type=type(exc).__name__, traceback=traceback.format_exc())
            fresh(output / 'failure.json', decision)
            raise


def stage_number(stage):
    match = re.fullmatch(r'segment-([0-9]{3})', stage or '')
    if match is None or not 1 <= int(match[1]) <= 20:
        raise ValueError('Require a bounded serial segment stage')
    return int(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha256', required=True)
    parser.add_argument('--mode', choices=('queue-cpu', 'cpu', 'arm', 'advance'), required=True)
    parser.add_argument('--after-stage')
    args = parser.parse_args()
    control = Controller(args.policy, args.policy_sha256)
    if args.mode == 'queue-cpu':
        print(json.dumps(control.queue_cpu(), indent=2))
    elif args.mode == 'cpu':
        command = [sys.executable, '-m', 'pytest', '-q', str(control.bundle / 'tests/test_triangle_auto_continue.py')]
        reply = subprocess.run(command, cwd=control.bundle, check=True)
        fresh(control.root / 'cpu-passed.json', {'status': 'passed', 'identity': control.operator_identity,
              'job_id': os.environ['LSB_JOBID'], 'utc': utc(), 'command': command, 'returncode': reply.returncode})
    elif args.mode == 'arm':
        control.cpu_gate()
        job, _ = control.submitted(control.campaign / 'receipts', 'segment-001', control.identity)
        if job != control.policy['first_job_id']:
            raise ValueError('Initial scientific job differs')
        print(json.dumps({'callback_job': control.queue_callback('segment-001', job)}, indent=2))
    else:
        control.advance(args.after_stage)


if __name__ == '__main__':
    main()
