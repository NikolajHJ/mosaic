"""Allocated fixed-step native completion certificate and every-snapshot audit.

Capture actual scheduler/LSF promptly.  Original large summary/case/JSONL and NPZ
files stay in their native study and are authenticated by exact path, byte size
and SHA.  Only compact metadata/proofs are copied into this collector payload.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from jq1_fixed_report_contract import (
    ARMS, authenticate_launcher, authenticate_study, config_path, fixed_counts,
    native_parent, read, ready_path, require, scheduler_outcome, sha,
    source_path, status_or_completion, study_path,
)
from render_jq1_fixed_report import VERSION, frame_arrays, verify_transition


def hashed(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def lsf_success(path, job):
    raw = Path(path).read_text(encoding='utf-8')
    require(f'Subject: Job {job}:' in raw and 'Successfully completed.' in raw,
            'Native same-job LSF did not finish successfully')
    return raw


def terminal_scheduler(arm, output, completions):
    """Capture actual DONE, or copy already authenticated original scheduler bytes."""
    pin = ARMS[arm]
    for attempt in range(7):
        result = subprocess.run(['bjobs', '-l', pin['job']], capture_output=True, text=True, timeout=30)
        raw = result.stdout + result.stderr
        observed = output / f'native-scheduler-{attempt}.txt'
        observed.write_text(raw, encoding='utf-8')
        try:
            status = status_or_completion(raw, arm, completions)
        except (ValueError, FileNotFoundError):
            if attempt == 6:
                raise
            time.sleep(10)
            continue
        require(status != 'EXIT', 'Native execution failed; no completion certificate')
        if status != 'DONE':
            require(attempt != 6, 'Native has no successful final outcome')
            time.sleep(10)
            continue
        try:
            require(scheduler_outcome(raw, pin['job']) == 'DONE', 'Native final scheduler is not DONE')
            return observed, []
        except ValueError:
            # The contract has already authenticated actual persisted native
            # scheduler/LSF, all original pins and full native metadata.  Copy
            # those exact saved scheduler bytes; never manufacture a DONE row.
            if arm == 'protenix':
                from jq1_fixed_report_contract import ROOT04
                record_path = ROOT04 / 'pro-terminal-observation.json'
                record = read(record_path)
                source = Path(record['scheduler']['path'])
                extra = [record_path, *(Path(p['path']) for p in record['proofs'])]
            else:
                record_path = Path(completions) / ('complete-' + arm) / 'payload/assessment.json'
                record = read(record_path)
                candidates = [Path(p['path']) for p in record['proofs']
                              if Path(p['path']).name.startswith(('native-scheduler', 'scheduler-final'))]
                source = next((p for p in candidates
                               if _is_done(p.read_text(encoding='utf-8'), pin['job'])), None)
                require(source is not None, 'Persisted collector has no actual native DONE record')
                extra = [record_path, *(Path(p['path']) for p in record['proofs'])]
            require(scheduler_outcome(source.read_text(encoding='utf-8'), pin['job']) == 'DONE',
                    'Persisted original scheduler does not certify DONE')
            target = output / 'native-scheduler-persisted.txt'
            shutil.copyfile(source, target)
            require(sha(target) == sha(source), 'Persisted native scheduler copy differs')
            return target, extra
    raise ValueError('Native final outcome unavailable')


def _is_done(raw, job):
    try:
        return scheduler_outcome(raw, job) == 'DONE'
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completions', type=Path, required=True)
    parser.add_argument('--report-manifest', required=True)
    parser.add_argument('--cpu-certificate-sha256', required=True)
    args = parser.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Completion checks require allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    pin = ARMS[args.arm]
    native = native_parent(args.arm)
    stdout = native / ('stdout.' + pin['job'] + '.log')
    record = {'status': 'failed', 'report_version': VERSION, 'native_job': pin['job'],
              'collector_job': os.environ['LSB_JOBID'], 'arm': args.arm,
              'captured_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Allocated full original metadata/JSONL and every saved NPZ authentication; '
                       'large originals referenced by exact path/bytes/SHA, compact evidence only copied'}
    try:
        report_source = Path(__file__).resolve().parents[1]
        manifest_path = report_source / 'source_manifest.json'
        require(sha(manifest_path) == args.report_manifest, 'Report source manifest changed')
        own_file = Path(__file__).resolve()
        file_pin = read(manifest_path)['files']['scripts/capture_jq1_fixed_completion.py']
        expected_file_sha = file_pin['sha256'] if isinstance(file_pin, dict) else file_pin
        require(sha(own_file) == expected_file_sha, 'Completion collector source differs from report manifest')
        cpu_path = args.completions / 'cpu-certificate.json'
        require(sha(cpu_path) == args.cpu_certificate_sha256, 'Report CPU certificate bytes changed')
        cpu = read(cpu_path)
        require(cpu['status'] == 'passed' and cpu['source_manifest_sha256'] == args.report_manifest
                and cpu['suite'] == 'jq1-fixed-saved-report-contract-v1'
                and cpu['tests'] > 0 and cpu['proofs'], 'Completion report CPU certificate failed/differs')
        for proof in cpu['proofs']:
            require(sha(proof['path']) == proof['sha256'], 'Report CPU proof changed')
        record.update(report_source_manifest_sha256=args.report_manifest,
                      cpu_certificate_sha256=args.cpu_certificate_sha256,
                      collector_source_script_sha256=sha(own_file))
        launcher = authenticate_launcher(args.arm)
        scheduler, persisted = terminal_scheduler(args.arm, args.output, args.completions)
        lsf_success(stdout, pin['job'])
        shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        study, summary, case = authenticate_study(args.arm)
        counts = fixed_counts(case['updates'], pin['rate'])
        snapshots, previous_probabilities = [], None
        for index, state in enumerate(case['states']):
            probabilities = frame_arrays(study / 'case-0', state, pin['metric'])
            if index:
                verify_transition(case['states'][index - 1], state, case['updates'][index - 1],
                                  previous_probabilities, probabilities, pin['metric'])
            previous_probabilities = probabilities
            snapshots.append({'step': state['step'], **hashed((study / 'case-0' / state['snapshot']).resolve(strict=True))})
        require(len(snapshots) == 10001 and counts['applied'] == 10000,
                'Incomplete full native snapshot/update budget')
        inventory = args.output / 'snapshots.json'
        inventory.write_text(json.dumps({'native_job': pin['job'], 'n_atoms': 25,
            'frames': snapshots, 'status': 'passed', 'validation':
            'Original NPZ SHA, FP32 shape/simplex/scalars/argmax/pairs verified for every state'},
            separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
        sources = [study / name for name in (
            'summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl',
            'initialization.npz', 'pocket_definition.json', 'native_gradient_checks/report.json',
            'terminal_gradient_checks/report.json', 'case-0/combined_initial_checks/report.json')]
        sources += [config_path(args.arm), source_path(args.arm) / 'source_manifest.json',
                    ready_path(args.arm), args.output / 'native-stdout.txt', scheduler, inventory,
                    manifest_path, own_file, cpu_path, *persisted,
                    *(Path(p['path']) for p in cpu['proofs'])]
        # Original snapshots remain referenced in inventory; they are all
        # rechecked again by the strict final renderer before report publication.
        unique = list(dict.fromkeys(Path(p) for p in sources))
        record.update(status='passed', states=len(case['states']), updates=len(case['updates']),
                      n_atoms=25, learning_rate=pin['rate'], acceptance_policy='fixed_step',
                      selection_policy='fixed_step', counts=counts,
                      model_digest=pin['model'], context_digest=pin['context'],
                      source_manifest_sha256=pin['manifest'], config_sha256=pin['config'],
                      readiness_sha256=pin['ready'], all_probability_frames_verified=True,
                      launcher=launcher, proofs=[hashed(p) for p in unique])
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if stdout.exists():
            shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        summary = study_path(args.arm) / 'summary.json'
        if summary.exists():
            record['failed_summary_reference'] = hashed(summary)
        raise
    finally:
        (args.output / 'assessment.json').write_text(
            json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
