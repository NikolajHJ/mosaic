"""Allocated focused checks, finite native probes and measured dispatch."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from chemistry_screen_ops import capacity, read, require, submit, write


def controls(plan, plan_path):
    root = Path(plan['root']) / 'controls'; root.mkdir(exist_ok=False)
    source = Path(plan['source_root'])
    names = ['test_protenix_chemistry_screen.py', 'test_chemistry_severity.py']
    argv = [sys.executable, '-m', 'pytest', *[str(source / 'tests' / x) for x in names],
            '-q', '-o', 'addopts=', '-p', 'no:cacheprovider', '--junitxml=' + str(root / 'tests.xml')]
    result = subprocess.run(argv, capture_output=True, text=True)
    write(root / 'tests-command.json', dict(argv=argv, returncode=result.returncode))
    (root / 'tests-stdout.log').write_text(result.stdout); (root / 'tests-stderr.log').write_text(result.stderr)
    require(result.returncode == 0, 'Focused allocated controls failed')
    xml = ET.fromstring((root / 'tests.xml').read_bytes())
    require(not any(list(xml.iter(tag)) for tag in ('failure', 'error', 'skipped')), 'Incomplete allocated controls')
    startups = []
    for arm in plan['arms']:
        argv = [sys.executable, str(source / 'protenix_chemistry_screen.py'), '--plan', str(plan_path), '--arm', arm, '--startup-only']
        result = subprocess.run(argv, capture_output=True, text=True)
        startups.append(dict(arm=arm, argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
        write(root / 'startups.json', startups)
        require(result.returncode == 0, 'Actual metadata startup failed: ' + arm)
    write(root / 'assessment.json', dict(status='passed', cases=len(list(xml.iter('testcase'))),
          startups=len(startups), job_id=os.environ['LSB_JOBID'], numerical_scope='small allocated synthetic controls; no native models'))


def probe(plan, plan_path, internal_seconds):
    root = Path(plan['root']); output = root / 'probe'; output.mkdir(exist_ok=False)
    require(read(root / 'controls' / 'assessment.json')['status'] == 'passed', 'Allocated controls incomplete')
    started = time.monotonic(); results = []
    for arm in plan['probe_arms']:
        argv = [sys.executable, str(Path(plan['source_root']) / 'protenix_chemistry_screen.py'),
                '--plan', str(plan_path), '--arm', arm, '--smoke-updates', '25',
                '--internal-seconds', str(int(internal_seconds - (time.monotonic() - started)))]
        with (output / (arm + '.stdout.log')).open('x') as out, (output / (arm + '.stderr.log')).open('x') as err:
            result = subprocess.run(argv, stdout=out, stderr=err, timeout=max(1, internal_seconds - (time.monotonic() - started)))
        require(result.returncode == 0, 'Native probe failed: ' + arm)
        parent = output / arm
        summary = read(parent / 'summary.json')
        require(summary['status'] == 'completed_exploratory_updates' and summary['applied_updates'] == 25
                and summary['probe_not_adopted'] is True and summary['frozen_before'] == summary['frozen_after'], 'Incomplete finite native probe')
        updates = [json.loads(line) for line in (parent / 'study' / 'updates.jsonl').read_text().splitlines()]
        require(len(updates) == 25 and [r['step'] for r in updates] == list(range(25)), 'Probe update prefix differs')
        warm = [r['optimization_seconds'] for r in updates[5:]]
        require(all(math.isfinite(v) and v > 0 for v in warm), 'Invalid measured timing')
        results.append(dict(arm=arm, status='finite_native_probe_completed', warm_seconds_max=max(warm),
                            warm_seconds_median=statistics.median(warm),
                            setup_and_probe_seconds=summary['elapsed_seconds']+summary['setup_elapsed_seconds'],
                            job_id=os.environ['LSB_JOBID'], derivative_qualified=False, states_adopted=False))
        write(output / 'results.json', results)
    write(output / 'assessment.json', dict(status='passed', job_id=os.environ['LSB_JOBID'], results=results,
          scope='finite attached implementation paths and timing; not precise derivative, binding or chemical efficacy certification'))


def dispatch(plan, plan_path):
    root = Path(plan['root'])
    write(root / 'dispatch-started.json', dict(job_id=os.environ['LSB_JOBID']), exclusive=True)
    assessment = read(root / 'probe' / 'assessment.json')
    require(assessment['status'] == 'passed', 'No complete native implementation probe')
    rows = {r['arm']: r for r in assessment['results']}
    budgets = {}; oversized = {}
    for arm, row in plan['arms'].items():
        measurement = rows[row['timing_probe']]
        # A conservative warm upper sample plus25%, setup and cleanup. No old
        # full-reference budgets or arbitrary inflated23-hour request.
        seconds = measurement['warm_seconds_max'] * row['screen']['updates'] * 1.25 + 1800
        minutes = max(45, math.ceil(seconds / 300) * 5)
        if minutes > 1440:
            oversized[arm] = dict(required_wall_minutes=minutes, reason='Measured full arm exceeds V10024h; needs device/runtime review')
            continue
        budgets[arm] = dict(wall_minutes=minutes, measurement=measurement, margin=1.25, setup_cleanup_seconds=1800)
    write(root / 'measured-budgets.json', budgets)
    write(root / 'runtime-blocked-arms.json', oversized)
    for arm in budgets:
        submit(plan_path, 'run', arm, budgets[arm]['wall_minutes'])
    write(root / 'dispatch-complete.json', dict(status='all_requested_arms_submitted' if not oversized else 'fitting_arms_submitted_oversized_arms_need_review',
          jobs=read(root / 'jobs.json'), runtime_blocked=oversized))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--kind', choices=('controls', 'probe', 'dispatch'), required=True)
    parser.add_argument('--internal-seconds', type=int, required=True)
    args = parser.parse_args(); plan = read(args.plan)
    require(os.environ.get('LSB_JOBID', '').isdigit(), 'Allocated job required')
    write(Path(plan['root']) / ('capacity-' + args.kind + '.json'), capacity(plan, args.kind))
    if args.kind == 'controls': controls(plan, args.plan)
    elif args.kind == 'probe': probe(plan, args.plan, args.internal_seconds)
    else: dispatch(plan, args.plan)


if __name__ == '__main__':
    main()
