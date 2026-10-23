"""Allocated saved-frame audit of passed or explicitly failed native studies.

The presentation audit may pass for an authentic failed scientific trajectory.
That distinction is retained in every assessment; no original file is modified.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from jq1_lrboost_report_contract import (ARMS, authenticate_inputs, authenticate_study, config_path, fixed_counts,
    native_parent, read, ready_path, require, saved_outcome, sha, source_path, status_or_completion, study_path)
from render_jq1_lrboost_report import VERSION, frame_arrays, verify_transition


def hashed(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def terminal_scheduler(arm, output, completions):
    """Actual terminal scheduler, or explicit persisted failure without fabrication."""
    pin = ARMS[arm]
    for attempt in range(7):
        r = subprocess.run(['bjobs', '-l', pin['job']], capture_output=True, text=True, timeout=30)
        raw = r.stdout + r.stderr
        observed = output / f'native-scheduler-{attempt}.txt'; observed.write_text(raw, encoding='utf-8')
        try:
            status = status_or_completion(raw, arm, completions)
        except (ValueError, FileNotFoundError):
            if attempt == 6: raise
            time.sleep(10); continue
        if status in ('DONE', 'EXIT', 'FAILED_PERSISTED', 'SUCCESS_PERSISTED'):
            return status, observed
        require(attempt != 6, 'Native execution is not terminal')
        time.sleep(10)
    raise ValueError('No authentic terminal outcome')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--completions', type=Path, required=True); p.add_argument('--report-manifest', required=True)
    p.add_argument('--cpu-certificate-sha256', required=True); args = p.parse_args()
    require(bool(os.environ.get('LSB_JOBID')), 'Saved-frame validation requires allocated DTU CPU')
    args.output.mkdir(parents=True, exist_ok=False)
    pin = ARMS[args.arm]; native = native_parent(args.arm); stdout = native / ('stdout.' + pin['job'] + '.log')
    record = {'status': 'failed', 'presentation_status': 'failed', 'scientific_status': 'unverified',
              'report_version': VERSION, 'native_job': pin['job'], 'collector_job': os.environ['LSB_JOBID'],
              'arm': args.arm, 'captured_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Every saved original NPZ/JSONL audited on allocated CPU; scientific failure preserved; originals read-only'}
    try:
        report_source = Path(__file__).resolve().parents[1]; manifest = report_source / 'source_manifest.json'
        require(sha(manifest) == args.report_manifest, 'Analysis source manifest changed')
        file_pin = read(manifest)['files']['scripts/capture_jq1_lrboost_completion.py']
        require(sha(__file__) == (file_pin['sha256'] if isinstance(file_pin, dict) else file_pin), 'Collector source changed')
        cpu_path = args.completions / 'cpu-certificate.json'
        require(sha(cpu_path) == args.cpu_certificate_sha256, 'Analysis CPU certificate bytes differ')
        cpu = read(cpu_path)
        require(cpu['status'] == 'passed' and cpu['suite'] == 'jq1-lrreset-saved-report-contract-v1'
                and cpu['source_manifest_sha256'] == args.report_manifest and cpu['tests'] > 0, 'Analysis CPU validation failed/differs')
        for proof in cpu['proofs']: require(sha(proof['path']) == proof['sha256'], 'CPU evidence changed')
        record.update(report_source_manifest_sha256=args.report_manifest, cpu_certificate_sha256=args.cpu_certificate_sha256,
                      collector_source_script_sha256=sha(__file__))
        provenance = authenticate_inputs(args.arm)
        status, scheduler = terminal_scheduler(args.arm, args.output, args.completions)
        study, summary, case = authenticate_study(args.arm, allow_failed=True)
        require((status in ('DONE', 'SUCCESS_PERSISTED')) == (summary['status'] == 'passed'),
                'Actual/persisted terminal outcome contradicts original summary and LSF')
        outcome = saved_outcome(args.arm, status)
        require(outcome['all_terminal'] and outcome['overall_status'] in ('passed', 'failed'), 'Native outcome is not terminal')
        shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        counts = fixed_counts(case['updates'], pin['rate']); snapshots = []; previous = None
        for index, state in enumerate(case['states']):
            probabilities = frame_arrays(study / 'case-0', state, pin['metric'])
            if index:
                verify_transition(case['states'][index - 1], state, case['updates'][index - 1], previous, probabilities, pin['metric'])
            previous = probabilities
            snapshots.append({'step': state['step'], **hashed((study / 'case-0' / state['snapshot']).resolve(strict=True))})
        require(len(snapshots) == len(case['updates']) + 1 and counts['applied'] == len(case['updates']), 'Saved case/frame/update correspondence differs')
        if summary['status'] == 'passed': require(len(snapshots) == 10001 and counts['applied'] == 10000, 'Scientific success requires full budget')
        inventory = args.output / 'snapshots.json'
        inventory.write_text(json.dumps({'native_job': pin['job'], 'n_atoms': 25, 'frames': snapshots,
            'status': 'passed', 'validation': 'Original NPZ SHA/FP32/simplex/scalar/argmax/transition checks for every saved frame'}, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')
        names = ('summary.json', 'case-0/report.json', 'case-0/states.jsonl', 'case-0/updates.jsonl', 'initialization.npz',
                 'pocket_definition.json', 'native_gradient_checks/report.json', 'terminal_gradient_checks/report.json', 'case-0/combined_initial_checks/report.json')
        sources = [study / n for n in names if (study / n).exists()]
        sources += [config_path(args.arm), source_path(args.arm) / 'source_manifest.json', ready_path(args.arm),
                    args.output / 'native-stdout.txt', scheduler, inventory, manifest, Path(__file__).resolve(), cpu_path,
                    *(Path(q['path']) for q in cpu['proofs'])]
        record.update(status='passed', presentation_status='passed', presentation_complete=True,
                      scientific_status=summary['status'], scientific_passed=summary['status'] == 'passed',
                      states=len(case['states']), updates=len(case['updates']), n_atoms=25, learning_rate=pin['rate'],
                      acceptance_policy='fixed_step', selection_policy='fixed_step', counts=counts, outcome=outcome,
                      native_status=status, model_digest=pin['model'], context_digest=pin['context'],
                      source_manifest_sha256=pin['manifest'], config_sha256=pin['config'], readiness_sha256=pin['ready'],
                      all_probability_frames_verified=True, launcher=provenance,
                      proofs=[hashed(q) for q in dict.fromkeys(Path(x) for x in sources)])
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        if stdout.exists() and not (args.output / 'native-stdout.txt').exists(): shutil.copyfile(stdout, args.output / 'native-stdout.txt')
        original = study_path(args.arm) / 'summary.json'
        if original.exists(): record['failed_summary_reference'] = hashed(original)
        raise
    finally:
        (args.output / 'assessment.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
