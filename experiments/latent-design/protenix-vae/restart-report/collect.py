"""Collect a compact, read-only DTU status and saved-artifact snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess


CONNECTOR = r"C:\Users\nikol\DTU-HPC\helper-v1\dtu-hpc.exe"

REMOTE = r"""
from datetime import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
import zipfile

ROOT = Path('/work3/s234818/latentmol/runs')
CONTROL = Path('/zhome/95/e/206121/dbmol-control')
RAW_GRAPHS = {}
JOBS = (
    (29914616, 'constant1', 'chemistry', 'gpuv100', 37, 18500),
    (29914617, 'constant10', 'chemistry', 'gpuv100', 37, 18500),
    (29914618, 'late-ramp', 'chemistry', 'gpuv100', 37, 18500),
    (29914683, 'mutation-a', 'followup', 'gpul40s', 84, 8064),
    (29914684, 'mutation-b', 'followup', 'gpua40', 84, 8064),
    (29914686, 'cap-a', 'followup', 'gpul40s', 48, 6144),
    (29914687, 'cap-b', 'followup', 'gpuv100', 48, 6144),
    (29914688, 'duration-a', 'followup', 'gpul40s', 40, 7960),
    (29914689, 'duration-b', 'followup', 'gpua40', 40, 7960),
    (29914690, 'lineage-a', 'followup', 'gpul40s', 90, 8640),
    (29914691, 'lineage-b', 'followup', 'gpuv100', 90, 8640),
)


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


def branch_record(path, cycle, task, campaign, detail, parents, fresh_candidates, elite_steps, spec):
    summary = read_json(path / 'summary.json')
    start = read_json(path / 'start.json') or {}
    rows = observations(path / 'metrics.jsonl')
    complete = summary is not None
    summary = summary or {}
    proposal = start.get('proposal') or {}
    if campaign == 'chemistry':
        ordinal = int(path.name.rsplit('-', 1)[-1]) - 1
        parent_index = ordinal // 3 if cycle > 1 else None
        repeat = ordinal % 3 if cycle > 1 else None
        arm = task
        planned = 500
    else:
        match = re.fullmatch(r'parent-(\d+)-(.+)-(\d+)', path.name)
        parent_index, arm, repeat = int(match[1]), match[2], int(match[3])
        planned = {'mutation': 96, 'cap': 128, 'lineage': 96}.get(task.rsplit('-', 1)[0])
        if planned is None:
            planned = 499 if arm == 'one-long' else 124
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
        proposal=proposal, max_step_norm=detail.get('max_step_norm'),
    )
    if INCLUDE_RAW:
        branch['playback_coverage'] = attach_raw_graphs(path, points, spec)
    return branch


def elite_entries(rows):
    return [{k: row.get(k) for k in ('smiles', 'fresh_loss', 'step', 'branch')} for row in rows]


def job_record(spec, scheduler):
    job_id, task, campaign, queue, planned_branches, budget_updates = spec
    run_name = 'protenix-restarts-20261107a' if campaign == 'chemistry' else 'protenix-restart-followup-20261107a'
    root = ROOT / f'{run_name}-{task}'
    control = CONTROL / run_name / task
    run = read_json(root / 'run.json') or {}
    result = read_json(root / 'result.json')
    launch = read_json(control / 'launch.json') or {}
    parents = {row['smiles']: row['fresh_loss'] for row in run.get('parents', [])}
    branches, archives = [], []
    phases = sorted(root.glob('cycle-*')) or sorted(root.glob('paired'))
    archive_files = {}
    elite_steps = {}
    for phase in phases:
        archive = read_json(phase / 'archive.json')
        lineage = read_json(phase / 'archives.json')
        archive_files[phase] = (archive, lineage)
        groups = [archive['elites']] if archive is not None else (lineage or {}).values()
        for rows in groups:
            for row in rows:
                branch = row.get('branch') or ''
                step = row.get('step')
                if '/' in branch and step is not None:
                    elite_steps.setdefault(branch, set()).add(step)
    for phase in phases:
        cycle = int(phase.name.split('-')[-1]) if phase.name.startswith('cycle-') else None
        batch = read_json(phase / 'batch.json') or {}
        detail = {row['name']: row for row in batch.get('branches', [])}
        fresh = read_json(phase / 'fresh' / 'summary.json') if campaign == 'chemistry' else None
        fresh_by_branch = {}
        for row in (fresh or {}).get('candidates', []):
            branch = row.get('arm', '').rsplit('/', 1)[-1]
            fresh_by_branch.setdefault(branch, []).append(
                {k: row.get(k) for k in ('smiles', 'step', 'loss', 'status')})
        for path in sorted(phase.iterdir()):
            if path.is_dir() and (path.name.startswith('branch-') or path.name.startswith('parent-')):
                branches.append(branch_record(path, cycle, task, campaign, detail.get(path.name, {}),
                                              parents, fresh_by_branch.get(path.name),
                                              elite_steps.get(f'{phase.name}/{path.name}', set()),
                                              run.get('graph_spec')))
        archive, lineage = archive_files[phase]
        if archive is not None:
            archives.append(dict(cycle=cycle, elites={task: elite_entries(archive['elites'])}))
            parents.update({row['smiles']: row['fresh_loss'] for row in archive['elites']})
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
        id=job_id, task=task, campaign=campaign, queue=queue, status=status,
        reason=reason or launch.get('error'), status_source=source,
        worker_status=launch.get('status'), artifact_status=result.get('status') if result else None,
        budget_updates=budget_updates, completed_updates=sum(b['completed_updates'] for b in branches),
        planned_branches=planned_branches, completed_branches=sum(b['complete'] for b in branches),
        result=result, archives=archives, branches=branches,
    )


def scheduler_snapshot():
    result = subprocess.run(
        ['bjobs', '-a', '-noheader', '-o', 'jobid stat queue pend_reason', *[str(j[0]) for j in JOBS]],
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
    missing = {job_id: queue for job_id, _, _, queue, _, _ in JOBS if job_id not in rows}
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
    args = parser.parse_args()
    command = "bash -lc " + shlex.quote("/usr/bin/python3 -I -")
    result = subprocess.run(
        [CONNECTOR, "exec", "--timeout", "110", command],
        input=f"INCLUDE_RAW = {args.include_raw!r}\n" + REMOTE,
        capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    if result.returncode:
        raise SystemExit(result.stderr or f"DTU connector returned {result.returncode}")
    snapshot = json.loads(result.stdout)
    if len(snapshot["jobs"]) != 11:
        raise ValueError("Snapshot must account for all 11 declared jobs")
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
