"""Small owned-job launcher for the chemistry strategy comparison.

Accepted receipts precede release. Unknown outcomes are never retried. The GPU
probe is an implementation/timing check, not a derivative qualification campaign.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + '.partial')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def require(value, message):
    if not value:
        raise ValueError(message)


def command(argv, timeout=60):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    return dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def capacity(plan, phase):
    # Reuse the canonical remaining-output accounting; no old account-wide
    # campaign forecasts or embedding totals are added to this experiment.
    from storage_accounting import assess_output
    for attempt in range(3):
        try:
            result = assess_output(plan['root'], plan['output_cap_bytes'], plan['file_cap'])
            break
        except FileNotFoundError:
            if attempt == 2:
                raise
    result['phase'] = phase
    require(result['status'] == 'passed', 'Owned remaining output exceeds available capacity')
    return result


def submit(plan_path, kind, arm=None, wall_minutes=None, dependency=None, dependency_type='done'):
    plan_path = Path(plan_path); plan = read(plan_path)
    root = Path(plan['root']); source = Path(plan['source_root'])
    label = kind if arm is None else arm
    parent = root / ('jobs' if kind != 'run' else 'production') / label
    parent.mkdir(parents=True, exist_ok=True)
    require(not list(parent.glob('submission*.json')), 'Existing/unknown submission requires reconciliation')
    require(kind in ('controls', 'probe', 'dispatch', 'run', 'migrate', 'conditional'), 'Unknown phase')
    require(kind != 'run' or arm in plan['arms'], 'Unknown strategy arm')
    require(kind != 'run' or arm in plan.get('production_arms', plan['arms']), 'Probe-only arm cannot launch production')
    require(kind != 'run' or wall_minutes is not None, 'Production needs an explicit measured wall budget')
    if kind == 'conditional':
        policy = plan.get('conditional_tail', {})
        require(arm is None and policy.get('parent_arm') in plan['arms'], 'Conditional parent arm is not declared')
        require(dependency_type == 'done' and str(dependency).isdigit()
                and read(root/'jobs.json').get(policy['parent_arm']) == str(dependency),
                'Conditional callback must depend on its accepted parent success')
    if kind == 'run':
        require(not (root / arm / 'payload').exists(), 'Existing scientific output requires a fresh continuation')
    gate = capacity(plan, 'submit-' + label)
    write(parent / 'capacity.json', gate)
    name = plan['job_name_prefix'] + '-' + label
    account = command(['bjobs', '-u', 's234818', '-json', '-o', 'jobid stat job_name command'])
    require(account['returncode'] == 0, 'Cannot inspect writers')
    live = [r for r in json.loads(account['stdout'])['RECORDS'] if r['STAT'] not in ('EXIT', 'DONE')]
    require(not any(r['JOB_NAME'] == name for r in live), 'Duplicate named writer')
    if kind == 'run':
        require(not any(r['JOBID'] in plan['arms'][arm].get('predecessor_job_ids',[]) for r in live),
                'Predecessor still active; reconcile before continuation')
    if kind == 'run':
        target = str(root / arm).rstrip('/')
        require(not any(any(v == target or v.startswith(target + '/')
                            for v in shlex.split(r['COMMAND'])) for r in live), 'Duplicate output writer')
    gpu = kind in ('probe', 'run')
    from gpu_policy import selected, resource
    gpu_row = plan['arms'][arm or plan['probe_arms'][0]] if gpu else None
    queue = selected(gpu_row)['queue'] if gpu else 'hpc'
    wall_minutes = wall_minutes or (90 if kind == 'probe' else 30)
    require(type(wall_minutes) is int and 1 <= wall_minutes <= 1440, 'Unsupported wall budget')
    require(not gpu or wall_minutes > 10, 'GPU budget must include cleanup reserve')
    argv = ['bsub', '-H', '-q', queue, '-J', name, '-n', '8' if gpu else '4',
            '-R', 'span[hosts=1]', '-R', 'rusage[mem=4096]',
            '-W', f'{wall_minutes // 60:02d}:{wall_minutes % 60:02d}',
            '-env', 'all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE',
            '-o', str(parent / 'stdout.%J.log'), '-e', str(parent / 'stderr.%J.log')]
    if gpu:
        argv += ['-gpu', resource(gpu_row)]
    if dependency:
        require(str(dependency).isdigit(), 'Invalid dependency')
        require(dependency_type in ('done', 'ended'), 'Unknown dependency condition')
        argv += ['-w', dependency_type + '(' + str(dependency) + ')']
    argv += ['bash', '-l', str(source / 'chemistry_screen_stage.sh'), str(source), kind,
             arm or '-', str(plan_path), str(wall_minutes * 60 - 600 if gpu else wall_minutes * 60 - 60)]
    from protenix_chemistry_screen import binding
    intent = dict(argv=argv, kind=kind, arm=arm, job_name=name, plan=binding(plan_path),
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='outcome_unknown_until_receipt')
    write(parent / 'submission-intent.json', intent, exclusive=True)
    try:
        result = command(argv, 90)
    except subprocess.TimeoutExpired as error:
        write(parent / 'submission-timeout.json', dict(stdout=str(error.stdout), stderr=str(error.stderr), unknown=True))
        raise
    write(parent / 'submission-response.json', result)
    matches = re.findall(r'Job <([1-9][0-9]*)> is submitted', result['stdout'])
    require(result['returncode'] == 0 and len(matches) == 1, 'Ambiguous submission; reconcile before retry')
    job = matches[0]
    write(parent / 'submission.json', dict(intent, job_id=job, status='submitted', dispatch_state='held'), exclusive=True)
    registry = root / 'jobs.json'; jobs = read(registry) if registry.exists() else {}
    require(label not in jobs, 'Existing registry entry')
    jobs[label] = job; write(registry, jobs)
    observations = []
    for _ in range(12):
        result = command(['bjobs', '-json', '-o', 'jobid user stat job_name', job])
        require(result['returncode'] == 0, 'Cannot verify accepted job')
        rows = json.loads(result['stdout'])['RECORDS']
        require(len(rows) == 1, 'Ambiguous held job')
        row = rows[0]; observations.append(row)
        require(row['JOBID'] == job and row['USER'] == 's234818' and row['JOB_NAME'] == name, 'Held identity mismatch')
        if row['STAT'] == 'PSUSP':
            break
        require(row['STAT'] == 'PEND', 'Unexpected held state')
        time.sleep(5)
    write(parent / 'held-proof.json', observations)
    require(observations[-1]['STAT'] == 'PSUSP', 'No authenticated held state')
    release = command(['bresume', job]); write(parent / 'release.json', dict(release, job_id=job))
    require(release['returncode'] == 0, 'Release outcome unknown; reconcile')
    print(json.dumps(dict(kind=kind, arm=arm, job_id=job, wall_minutes=wall_minutes)), flush=True)
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--kind', choices=('controls', 'probe', 'dispatch', 'run', 'migrate', 'conditional'), required=True)
    parser.add_argument('--arm'); parser.add_argument('--wall-minutes', type=int)
    parser.add_argument('--dependency')
    parser.add_argument('--dependency-type', choices=('done', 'ended'), default='done')
    args = parser.parse_args()
    submit(args.plan, args.kind, args.arm, args.wall_minutes, args.dependency, args.dependency_type)


if __name__ == '__main__':
    main()
