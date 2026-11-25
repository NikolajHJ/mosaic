"""Collect a compact, read-only DTU status and saved-artifact snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess


CONNECTOR = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"
DEFAULT_MANIFEST = Path(__file__).with_name("runs.json")


def add_selection_args(parser):
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="JSON run registry with optional default_jobs selection")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--job", type=int, action="append", help="Job ID to include, repeatable")
    selection.add_argument("--all-runs", action="store_true", help="Collect every run in the manifest")
    parser.add_argument("--timeout", type=int, help="Override the DTU connector timeout in seconds")


def select_runs(manifest, *, job_ids=None, all_runs=False):
    data = json.loads(manifest.read_text(encoding="utf-8"))
    runs = data["runs"]
    by_id = {}
    for run in runs:
        job_id = run["id"]
        if type(job_id) is not int or job_id < 1 or job_id in by_id:
            raise ValueError(f"Invalid or duplicate job ID {job_id!r}")
        for field in ("task", "campaign", "queue"):
            if not isinstance(run[field], str) or not run[field]:
                raise ValueError(f"Job {job_id} needs a nonempty {field}")
        for field in ("run_path", "launch_path"):
            path = PurePosixPath(run[field])
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Job {job_id} requires an absolute, normalized {field}")
        if run["layout"] not in ("cycles", "panel"):
            raise ValueError(f"Job {job_id} needs a cycles or panel layout")
        counts = [run["planned_branches"], run["budget_updates"]]
        if "cycles" in run:
            counts.append(run["cycles"])
        if run["layout"] == "cycles":
            counts.append(run["branches_per_parent"])
        updates = run["updates_per_branch"]
        if isinstance(updates, dict):
            if not updates or run["layout"] != "panel":
                raise ValueError(f"Job {job_id} has an invalid arm update map")
            for update in updates.values():
                if isinstance(update, list):
                    if not update:
                        raise ValueError(f"Job {job_id} has an empty arm repeat schedule")
                    counts.extend(update)
                else:
                    counts.append(update)
        else:
            counts.append(updates)
        if any(type(count) is not int or count < 1 for count in counts):
            raise ValueError(f"Job {job_id} requires positive integer budgets and counts")
        by_id[job_id] = run
    if all_runs:
        selected = list(by_id)
    elif job_ids is not None:
        selected = job_ids
    else:
        selected = data.get("default_jobs", list(by_id))
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Choose at least one run and list each job ID once")
    if unknown := set(selected) - by_id.keys():
        raise ValueError(f"Jobs absent from the manifest: {sorted(unknown)}")
    jobs = [by_id[job_id] for job_id in selected]
    if len({run["run_path"] for run in jobs}) != len(jobs):
        raise ValueError("Selected jobs must have distinct output paths")
    return jobs


def verify_snapshot_jobs(snapshot, jobs):
    expected = [run["id"] for run in jobs]
    collected = [job["id"] for job in snapshot["jobs"]]
    if collected != expected:
        raise ValueError(f"Snapshot jobs differ from manifest selection: {collected!r} != {expected!r}")

REMOTE = r"""
from datetime import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
import zipfile

RAW_GRAPHS = {}
JOBS = json.loads(JOBS_JSON)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        # A running worker may be between opening a new JSON file and flushing it.
        return None


def observations(path):
    try:
        with path.open(encoding='utf-8') as handle:
            return [json.loads(line) for line in handle if line.endswith('\n')]
    except FileNotFoundError:
        return []


def cycle_number(name):
    return int(name.split('-')[-1]) if name.startswith('cycle-') else None


def phase_details(phase, run):
    batch = read_json(phase / 'batch.json') or {}
    details = {row['name']: row for row in batch.get('branches', [])}
    if run.get('runner') == 'restart_search':
        plan = read_json(phase / 'wave-plan.json') or {}
        for row in plan.get('branches', []):
            details.setdefault(row['name'], {}).update(parent_fresh_loss=row['source'].get('fresh_loss'),
                                                       policy=row.get('policy'))
    if phase.name == 'paired' and run.get('source_job_id') is not None:
        for index, pair in enumerate(run.get('pairs', [])):
            for arm, key in (('near-valid', 'invalid_source'), ('exact-valid', 'exact_source')):
                source = pair[key]
                provenance = dict(source, source_job_id=run['source_job_id'], source_wave=pair['source_wave'])
                if arm == 'near-valid':
                    provenance.update(raw_invalid=True, repair_smiles=pair['witness_smiles'],
                                      repair_edits=pair['repair_edits'])
                details.setdefault(f'parent-{index:02d}-{arm}-00', {}).update(
                    parent_smiles=source.get('smiles'), parent_fresh_loss=source.get('archived_fresh_loss'),
                    parent_provenance=provenance)
    return details


def fresh_candidates_by_branch(phase):
    summaries = [phase / 'fresh' / 'summary.json', *sorted(phase.glob('fresh-*/fresh/summary.json'))]
    by_branch = {}
    for path in summaries:
        for row in (read_json(path) or {}).get('candidates', []):
            branch = (row.get('arm') or '').rsplit('/', 1)[-1]
            if branch.startswith(('branch-', 'parent-')):
                by_branch.setdefault(branch, []).append(
                    {key: row.get(key) for key in ('smiles', 'step', 'loss', 'status')})
    return by_branch


def branch_plan(path, cycle, spec):
    planned = spec['updates_per_branch']
    if spec['layout'] == 'cycles':
        ordinal = int(path.name.rsplit('-', 1)[-1]) - 1
        count = spec['branches_per_parent']
        return (spec['task'], ordinal // count if cycle > 1 else None,
                ordinal % count if cycle > 1 else None, planned)
    match = re.fullmatch(r'parent-(\d+)-(.+)-(\d+)', path.name)
    if not match:
        raise ValueError(f'Unexpected panel branch path: {path}')
    parent_index, arm, repeat = int(match[1]), match[2], int(match[3])
    if isinstance(planned, dict):
        planned = planned[arm]
    return arm, parent_index, repeat, planned[repeat] if isinstance(planned, list) else planned


def sample(rows, retained_steps):
    if not rows:
        return []
    count = min(48, len(rows))
    indices = {round(i * (len(rows) - 1) / max(1, count - 1)) for i in range(count)}
    for i in range(1, len(rows)):
        if rows[i]['step'] in retained_steps or rows[i]['step'] in (25, 375):
            indices.add(i)
    valid = [i for i, row in enumerate(rows) if row['raw_valid']]
    if valid:
        indices.add(min(valid, key=lambda i: rows[i]['relaxed_loss']))
    indices.add(min(range(len(rows)), key=lambda i: rows[i]['relaxed_loss']))
    result = []
    for i in sorted(indices):
        row = rows[i]
        update = row.get('update') or {}
        result.append([
            row['step'], round(row['relaxed_loss'], 7), round(row['chemistry_loss'], 7),
            round(row['chemistry_weight'], 7), row['raw_valid'], row.get('smiles'),
            update.get('displacement_norm'),
        ])
    return result


def attach_raw_graphs(path, points, spec):
    import numpy as np

    needed = {point[0]: point for point in points if not (point[4] and point[5])}
    for point in points:
        point.append(None)
    recovered = set()
    if needed and spec:
        element_names = spec['atomic_numbers']
        charge_names = spec['formal_charges']
        bond_names = spec['bond_names']
        empty_atom, empty_bond = element_names.index(0), bond_names.index('NONE')
        for block in sorted((path / 'raw-blocks').glob('states-*.npz')):
            first, last = map(int, block.stem.split('-')[1:])
            targets = {step for step in needed if first <= step <= last}
            if not targets:
                continue
            try:
                if not zipfile.is_zipfile(block):
                    continue
                with np.load(block, allow_pickle=False) as data:
                    steps = data['steps']
                    atoms = data['atoms']
                    charges = data['charges']
                    bonds = data['bonds_upper']
                    left, right = data['upper_left'], data['upper_right']
            except (FileNotFoundError, zipfile.BadZipFile, EOFError):
                # A live worker may have opened the block without finishing the ZIP yet.
                continue
            for row, step in enumerate(steps.tolist()):
                if step not in targets:
                    continue
                occupied = np.flatnonzero(atoms[row] != empty_atom).tolist()
                slot_to_atom = {slot: i for i, slot in enumerate(occupied)}
                compact_atoms = [[element_names[int(atoms[row, slot])],
                                  charge_names[int(charges[row, slot])]] for slot in occupied]
                compact_bonds = []
                for edge in np.flatnonzero(bonds[row] != empty_bond).tolist():
                    a, b = int(left[edge]), int(right[edge])
                    if a not in slot_to_atom or b not in slot_to_atom:
                        raise ValueError(f'Saved bond points to an empty atom slot in {block}')
                    compact_bonds.append([slot_to_atom[a], slot_to_atom[b],
                                          bond_names[int(bonds[row, edge])]])
                graph = dict(atoms=compact_atoms, bonds=compact_bonds)
                encoded = json.dumps(graph, separators=(',', ':')).encode()
                key = hashlib.sha256(encoded).hexdigest()[:24]
                if RAW_GRAPHS.setdefault(key, graph) != graph:
                    raise ValueError('Raw graph identity collision')
                needed[step][7] = key
                recovered.add(step)
    return dict(sampled_points=len(points), valid_smiles=len(points) - len(needed),
                raw_requested=len(needed), raw_recovered=len(recovered),
                raw_unavailable=len(needed) - len(recovered))


def branch_record(path, cycle, job, detail, parents, fresh_candidates, elite_steps, graph_spec):
    summary = read_json(path / 'summary.json')
    start = read_json(path / 'start.json') or {}
    rows = observations(path / 'metrics.jsonl')
    complete = summary is not None
    summary = summary or {}
    proposal = start.get('proposal') or {}
    arm, parent_index, repeat, planned = branch_plan(path, cycle, job)
    parent_smiles = detail.get('parent_smiles') or start.get('parent')
    parent_loss = detail.get('parent_fresh_loss')
    if parent_loss is None and parent_smiles in parents:
        parent_loss = parents[parent_smiles]
    if complete:
        updates = summary['updates']
    elif rows:
        updates = rows[-1]['step'] + int(rows[-1].get('update') is not None)
    else:
        updates = 0
    distinct = {row['smiles'] for row in rows if row['raw_valid']}
    candidates = detail.get('candidates', {})
    if fresh_candidates:
        candidates = candidates | {'fresh': fresh_candidates}
    candidate_steps = {row['step'] for group in candidates.values() for row in group}
    points = sample(rows, candidate_steps | elite_steps)
    branch = dict(
        name=path.name, cycle=cycle, arm=arm, parent_index=parent_index, repeat=repeat,
        planned_updates=planned, completed_updates=min(planned, updates), complete=complete,
        valid_states=summary.get('valid_states', sum(row['raw_valid'] for row in rows)),
        states=summary['updates'] + 1 if complete else len(rows),
        distinct_valid=summary.get('distinct_valid', len(distinct)),
        bounded_steps=summary.get('bounded_steps', sum(
            row['update']['bound_factor'] < 1 for row in rows if row.get('update') is not None)),
        points=points, candidates=candidates, parent_smiles=parent_smiles,
        parent_fresh_loss=parent_loss, policy=detail.get('policy') or proposal.get('policy'),
        parent_provenance=detail.get('parent_provenance'), parent_source=start.get('parent_source'),
        proposal=proposal, max_step_norm=detail.get('max_step_norm'),
    )
    if INCLUDE_RAW:
        branch['playback_coverage'] = attach_raw_graphs(path, points, graph_spec)
    return branch


def elite_entries(rows):
    return [{k: row.get(k) for k in ('smiles', 'fresh_loss', 'step', 'branch')} for row in rows]


def job_record(spec, scheduler):
    job_id, task, campaign, queue = (spec[field] for field in ('id', 'task', 'campaign', 'queue'))
    root = Path(spec['run_path'])
    run = read_json(root / 'run.json') or {}
    result = read_json(root / 'result.json')
    launch = read_json(Path(spec['launch_path'])) or {}
    parents = {row['smiles']: row['fresh_loss'] for row in run.get('parents', [])}
    branches, archives = [], []
    phases = sorted(root.glob('cycle-*')) or sorted(root.glob('paired'))
    archive_files = {}
    elite_steps = {}
    for phase in phases:
        archive = read_json(phase / 'archive.json')
        lineage = read_json(phase / 'archives.json')
        archive_files[phase] = (archive, lineage)
        groups = ([archive['elites'], archive.get('restart_parents', [])]
                  if archive is not None else (lineage or {}).values())
        for rows in groups:
            for row in rows:
                branch = row.get('branch') or ''
                step = row.get('step')
                if '/' in branch and step is not None:
                    elite_steps.setdefault(branch, set()).add(step)
    for phase in phases:
        cycle = cycle_number(phase.name)
        detail = phase_details(phase, run)
        fresh_by_branch = fresh_candidates_by_branch(phase)
        for path in sorted(phase.iterdir()):
            if path.is_dir() and (path.name.startswith('branch-') or path.name.startswith('parent-')):
                branches.append(branch_record(path, cycle, spec, detail.get(path.name, {}),
                                              parents, fresh_by_branch.get(path.name),
                                              elite_steps.get(f'{phase.name}/{path.name}', set()),
                                              run.get('graph_spec')))
        archive, lineage = archive_files[phase]
        if archive is not None:
            archives.append(dict(cycle=cycle, elites={task: elite_entries(archive['elites'])}))
            parents.update({row['smiles']: row['fresh_loss'] for row in archive['elites']})
            parents.update({row['smiles']: row['fresh_loss'] for row in archive.get('restart_parents', [])})
        elif lineage is not None:
            archives.append(dict(cycle=cycle, elites={arm: elite_entries(rows) for arm, rows in lineage.items()}))
            parents.update({row['smiles']: row['fresh_loss'] for rows in lineage.values() for row in rows})
    status, queue, reason, source = scheduler.get(job_id, ('UNKNOWN', queue, None, None))
    if status == 'UNKNOWN':
        if launch.get('status') == 'worker_exited':
            status = 'DONE' if launch.get('returncode') == 0 else 'EXIT'
            source = 'worker receipt'
        elif launch.get('status') == 'launch_failed':
            status, source = 'EXIT', 'worker receipt'
    return dict(
        id=job_id, task=task, campaign=campaign, label=spec.get('label'), queue=queue, status=status,
        reason=reason or launch.get('error'), status_source=source,
        worker_status=launch.get('status'), artifact_status=result.get('status') if result else None,
        budget_updates=spec['budget_updates'], completed_updates=sum(b['completed_updates'] for b in branches),
        planned_cycles=spec.get('cycles'), planned_branches=spec['planned_branches'],
        completed_branches=sum(b['complete'] for b in branches),
        result=result, archives=archives, branches=branches,
    )


def scheduler_snapshot():
    result = subprocess.run(
        ['bjobs', '-a', '-noheader', '-o', 'jobid stat queue pend_reason', *[str(j['id']) for j in JOBS]],
        capture_output=True, text=True, timeout=20,
    )
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        identifier, status, queue = fields[:3]
        reason = fields[3] if len(fields) == 4 and fields[3] != '-' else None
        rows[int(identifier)] = (status, queue, reason, 'bjobs')
    missing = {job['id']: job['queue'] for job in JOBS if job['id'] not in rows}
    if missing:
        try:
            history = subprocess.run(['bhist', '-l', *map(str, missing)],
                                     capture_output=True, text=True, timeout=16).stdout
        except subprocess.TimeoutExpired:
            history = ''
        for block in re.split(r'(?m)(?=^Job <\d+>)', history):
            match = re.match(r'Job <(\d+)>', block)
            if not match or (job_id := int(match[1])) not in missing:
                continue
            if 'Done successfully' in block:
                rows[job_id] = ('DONE', missing[job_id], None, 'bhist')
            elif re.search(r'Exited (?:with|by)', block):
                rows[job_id] = ('EXIT', missing[job_id], None, 'bhist')
    return rows


captured = datetime.now().astimezone().isoformat(timespec='seconds')
scheduler = scheduler_snapshot()
snapshot = dict(captured_at=captured, jobs=[job_record(spec, scheduler) for spec in JOBS])
if INCLUDE_RAW:
    coverage_fields = ('sampled_points', 'valid_smiles', 'raw_requested', 'raw_recovered', 'raw_unavailable')
    coverages = [branch['playback_coverage'] for job in snapshot['jobs'] for branch in job['branches']]
    snapshot['raw_graphs'] = RAW_GRAPHS
    snapshot['playback_coverage'] = {field: sum(row[field] for row in coverages) for field in coverage_fields}
print(json.dumps(snapshot, separators=(',', ':'), allow_nan=False))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("snapshot.json"))
    parser.add_argument("--include-raw", action="store_true")
    add_selection_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    jobs = select_runs(args.manifest, job_ids=args.job, all_runs=args.all_runs)
    timeout = args.timeout if args.timeout is not None else 110
    if timeout < 1:
        parser.error("--timeout must be positive")
    command = "bash -lc " + shlex.quote("/usr/bin/python3 -I -")
    result = subprocess.run(
        [CONNECTOR, "exec", "--timeout", str(timeout), command],
        input=f"INCLUDE_RAW = {args.include_raw!r}\nJOBS_JSON = {json.dumps(jobs)!r}\n" + REMOTE,
        capture_output=True, text=True, encoding="utf-8", timeout=timeout + 10,
    )
    if result.returncode:
        raise SystemExit(result.stderr or f"DTU connector returned {result.returncode}")
    snapshot = json.loads(result.stdout)
    verify_snapshot_jobs(snapshot, jobs)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, separators=(",", ":"), allow_nan=False)
    print(snapshot["captured_at"])
    for job in snapshot["jobs"]:
        print(job["id"], job["task"], job["status"], job["status_source"],
              f'{job["completed_updates"]}/{job["budget_updates"]}',
              f'{job["completed_branches"]}/{job["planned_branches"]}',
              f'{len(job["archives"])} archives')
    if args.include_raw:
        print("Playback coverage", snapshot["playback_coverage"],
              f'{len(snapshot["raw_graphs"])} unique raw graphs')


if __name__ == "__main__":
    main()
