"""Read exact saved hard-state trajectories and archive forks from DTU."""
from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
import shlex
import subprocess

from collect import CONNECTOR, REMOTE, add_selection_args, select_runs, verify_snapshot_jobs


TREE_REMOTE = r"""
import base64
import gzip


def cycle_number(name):
    return int(name.split('-')[-1]) if name.startswith('cycle-') else None


def archive_records(phase, task):
    archive = read_json(phase / 'archive.json')
    lineage = read_json(phase / 'archives.json')
    if archive is not None:
        return {task: archive['elites']}, {task: archive.get('restart_parents', archive['elites'])}
    return lineage or {}, lineage or {}


def archive_source(row, cycle):
    if not row:
        return None
    match = re.fullmatch(r'cycle-(\d+)/(.+)', row.get('branch') or '')
    step = row.get('step')
    if match and isinstance(step, int) and int(match[1]) < cycle:
        return dict(branch=match[2], cycle=int(match[1]), step=step,
                    smiles=row['smiles'], fresh_loss=row.get('fresh_loss'))
    return None


def match_parent(rows, smiles):
    found = [row for row in rows if smiles and row['smiles'] == smiles]
    if len(found) > 1:
        raise ValueError('Archive has multiple entries for a branch parent')
    return found[0] if found else None


def exact_states(branch_path, rows, spec, retained):
    raw_points = [[row['step'], None, None, None, False, None, None]
                  for row in rows if not (row['raw_valid'] and row.get('smiles'))]
    coverage = attach_raw_graphs(branch_path, raw_points, spec)
    raw_by_step = {point[0]: point[7] for point in raw_points}
    if not rows:
        return [], [], [], coverage, 0
    first_step, last_step = rows[0]['step'], rows[-1]['step']
    observed = set()
    states, repeat_anchors, missing = [], [], []
    prior_step, prior_identity = first_step - 1, None
    changes = 0
    for row in rows:
        step = row['step']
        if step <= prior_step:
            raise ValueError(f'Out-of-order saved update {step} in {branch_path}')
        observed.add(step)
        valid = bool(row['raw_valid'] and row.get('smiles'))
        smiles = row['smiles'] if valid else None
        key = raw_by_step.get(step)
        if valid:
            identity = ('smiles', smiles)
        elif key:
            identity = ('raw', key)
        else:
            missing.append(step)
            identity = ('missing', step)
        changed = identity != prior_identity or step != prior_step + 1
        if changed or step in retained:
            states.append([step, valid, smiles, key, round(row['relaxed_loss'], 7)])
            if not changed:
                repeat_anchors.append(step)
        if changed:
            changes += 1
        prior_step, prior_identity = step, identity
    missing.extend(set(range(first_step, last_step + 1)) - observed)
    return states, repeat_anchors, sorted(missing), coverage, changes


def collect_branch(path, cycle, job, detail, run_parents, preceding_archive, retained, graph_spec):
    start = read_json(path / 'start.json') or {}
    summary = read_json(path / 'summary.json')
    complete = summary is not None
    rows = observations(path / 'metrics.jsonl')
    parent_smiles = detail.get('parent_smiles') or start.get('parent')
    parent_fresh_loss = detail.get('parent_fresh_loss')
    parent_provenance = detail.get('parent_provenance')
    source, parent_row = None, None
    arm, parent_index, repeat, planned = branch_plan(path, cycle, job)
    if job['layout'] == 'cycles':
        if cycle == 1:
            root_type = 'bootstrap'
        else:
            parent_row = match_parent(preceding_archive.get(job['task'], []), parent_smiles)
            source = archive_source(parent_row, cycle)
            saved_source = start.get('parent_source')
            if source is None and saved_source and parent_smiles:
                source = archive_source(dict(saved_source, smiles=parent_smiles), cycle)
            root_type = 'archive-fork' if source else 'unresolved'
    else:
        if arm == 'independent':
            root_type = 'independent'
        elif cycle is None or cycle == 1:
            parent_row = match_parent(run_parents, parent_smiles)
            root_type = 'archived-parent' if parent_row else 'unresolved'
        else:
            parent_row = match_parent(preceding_archive.get('elite', []), parent_smiles)
            source = archive_source(parent_row, cycle)
            root_type = 'archive-fork' if source else ('archived-parent' if parent_row else 'unresolved')
    if parent_row:
        if parent_fresh_loss is None:
            parent_fresh_loss = parent_row.get('fresh_loss')
        if parent_provenance is None:
            parent_provenance = parent_row.get('provenance')
        if parent_provenance is None:
            parent_provenance = {k: parent_row.get(k) for k in ('branch', 'step', 'smiles')}
    states, repeat_anchors, missing, coverage, changes = exact_states(path, rows, graph_spec, retained)
    loss_series = []
    for row in rows:
        values = [row['step'], round(row['relaxed_loss'], 7), round(row['chemistry_loss'], 7),
                  round(row['chemistry_weight'], 7)]
        if row.get('valence_excess') is not None:
            values.append(round(row['valence_excess'], 7))
        loss_series.append(values)
    if complete:
        updates = summary['updates']
    elif rows:
        updates = rows[-1]['step'] + int(rows[-1].get('update') is not None)
    else:
        updates = 0
    return dict(name=path.name, cycle=cycle, arm=arm, parent_index=parent_index, repeat=repeat,
                root_type=root_type, source=source, parent_smiles=parent_smiles,
                parent_fresh_loss=parent_fresh_loss, parent_provenance=parent_provenance,
                complete=complete, planned_updates=planned, completed_updates=min(planned, updates),
                first_step=rows[0]['step'] if rows else None, last_step=rows[-1]['step'] if rows else None,
                observations=len(rows), state_count=len(states), change_state_count=changes,
                repeat_anchor_steps=repeat_anchors, missing_steps=missing,
                raw_recovered=coverage['raw_recovered'], raw_requested=coverage['raw_requested'],
                states=states, loss_series=loss_series)


def tree_job(spec, scheduler):
    job_id, task, campaign, queue = (spec[field] for field in ('id', 'task', 'campaign', 'queue'))
    root = Path(spec['run_path'])
    run = read_json(root / 'run.json') or {}
    result = read_json(root / 'result.json')
    launch = read_json(Path(spec['launch_path'])) or {}
    phases = sorted(root.glob('cycle-*')) or sorted(root.glob('paired'))
    archive_pairs = {cycle_number(phase.name): archive_records(phase, task) for phase in phases}
    retained = {}
    for archive, restart in archive_pairs.values():
        for groups in (archive, restart):
            for rows in groups.values():
                for row in rows:
                    if row.get('branch') and isinstance(row.get('step'), int):
                        retained.setdefault(row['branch'], set()).add(row['step'])
    branch_entries = {}
    for phase in phases:
        branch_entries[phase] = [path for path in sorted(phase.iterdir())
                                 if path.is_dir() and path.name.startswith(('branch-', 'parent-'))]
        for path in branch_entries[phase]:
            source = (read_json(path / 'start.json') or {}).get('parent_source')
            if source and source.get('branch') and isinstance(source.get('step'), int):
                retained.setdefault(source['branch'], set()).add(source['step'])
    branches = []
    for phase in phases:
        cycle = cycle_number(phase.name)
        batch = read_json(phase / 'batch.json') or {}
        detail = {row['name']: row for row in batch.get('branches', [])}
        preceding = archive_pairs.get(cycle - 1, ({}, {}))[1] if cycle else {}
        for path in branch_entries[phase]:
            branches.append(collect_branch(path, cycle, spec, detail.get(path.name, {}),
                                           run.get('parents', []), preceding,
                                           retained.get(f'{phase.name}/{path.name}', set()),
                                           run.get('graph_spec')))
    referenced = {}
    for branch in branches:
        if source := branch['source']:
            referenced.setdefault((source['cycle'], source['branch']), {})[source['step']] = source['smiles']
    confirmed = set()
    for branch in branches:
        key = (branch['cycle'], branch['name'])
        expected = referenced.get(key, {})
        for step, valid, smiles, _, _ in branch['states']:
            if valid and step in expected and smiles == expected[step]:
                confirmed.add((*key, step))
    for branch in branches:
        source = branch['source']
        if source and (source['cycle'], source['branch'], source['step']) not in confirmed:
            branch['source_unavailable'] = source
            branch['source'] = None
            branch['root_type'] = 'unresolved'
    status, queue, reason, status_source = scheduler.get(job_id, ('UNKNOWN', queue, None, None))
    if status == 'UNKNOWN':
        if launch.get('status') == 'worker_exited':
            status = 'DONE' if launch.get('returncode') == 0 else 'EXIT'
            status_source = 'worker receipt'
        elif launch.get('status') == 'launch_failed':
            status, status_source = 'EXIT', 'worker receipt'
    archives = [dict(cycle=cycle, elites={arm: elite_entries(rows) for arm, rows in groups.items()})
                for cycle, (groups, _) in archive_pairs.items() if groups]
    return dict(id=job_id, task=task, campaign=campaign, label=spec.get('label'), queue=queue, status=status,
                reason=reason or launch.get('error'), status_source=status_source,
                worker_status=launch.get('status'), artifact_status=result.get('status') if result else None,
                budget_updates=spec['budget_updates'], planned_cycles=spec.get('cycles'),
                planned_branches=spec['planned_branches'],
                completed_branches=sum(branch['complete'] for branch in branches),
                completed_updates=sum(branch['completed_updates'] for branch in branches),
                result=result, archives=archives, branches=branches)


captured = datetime.now().astimezone().isoformat(timespec='seconds')
scheduler = scheduler_snapshot()
jobs = [tree_job(spec, scheduler) for spec in JOBS]
branches = [branch for job in jobs for branch in job['branches']]
coverage = dict(
    branches=len(branches), complete_branches=sum(branch['complete'] for branch in branches),
    observations=sum(branch['observations'] for branch in branches),
    displayed_states=sum(branch['state_count'] for branch in branches),
    repeated_archive_anchors=sum(len(branch['repeat_anchor_steps']) for branch in branches),
    missing_steps=sum(len(branch['missing_steps']) for branch in branches),
    raw_requested=sum(branch['raw_requested'] for branch in branches),
    raw_recovered=sum(branch['raw_recovered'] for branch in branches),
    verified_forks=sum(branch['source'] is not None for branch in branches),
    unresolved_origins=sum(branch['root_type'] == 'unresolved' for branch in branches),
)
snapshot = dict(captured_at=captured, jobs=jobs, raw_graphs=RAW_GRAPHS, coverage=coverage,
                state_policy='Consecutive canonical SMILES or saved invalid raw graph labels. '
                             'Revisits and exact archived source steps are retained. '
                             'Invalid live observations without flushed labels are marked missing.')
payload = json.dumps(snapshot, separators=(',', ':'), allow_nan=False).encode()
print(base64.b64encode(gzip.compress(payload, compresslevel=4)).decode())
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("tree-snapshot.json"))
    add_selection_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = select_runs(args.manifest, job_ids=args.job, all_runs=args.all_runs)
    timeout = args.timeout if args.timeout is not None else 120
    if timeout < 1:
        parser.error("--timeout must be positive")
    marker = "\ncaptured = datetime."
    definitions, separator, _ = REMOTE.rpartition(marker)
    if not separator:
        raise ValueError("Snapshot collector definitions changed")
    command = "bash -lc " + shlex.quote("/usr/bin/python3 -I -")
    result = subprocess.run(
        [CONNECTOR, "exec", "--timeout", str(timeout), command],
        input=f"JOBS_JSON = {json.dumps(jobs)!r}\n" + definitions + "\n" + TREE_REMOTE,
        capture_output=True, text=True, encoding="utf-8", timeout=timeout + 10,
    )
    if result.returncode:
        raise SystemExit(result.stderr or f"DTU connector returned {result.returncode}")
    payload = gzip.decompress(base64.b64decode(result.stdout.strip(), validate=True))
    snapshot = json.loads(payload)
    verify_snapshot_jobs(snapshot, jobs)
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(snapshot["captured_at"], snapshot["coverage"], len(payload), "bytes")
    for job in snapshot["jobs"]:
        print(job["id"], job["task"], job["status"], job["status_source"],
              f'{job["completed_updates"]}/{job["budget_updates"]}',
              f'{len(job["branches"])} branches')


if __name__ == "__main__":
    main()
