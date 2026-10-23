"""Serial frozen-Boltz carbon-chain elemental-response panel on DTU HPC.

Each target/count is one fixed-occupancy trajectory. Source056's prior study
remains separate. This new orchestration needs HPC fixture and native validation.
Case boundaries are resumable under the SAME sealed source/protocol. Interrupted
attempts are retained; ordinary failed cases are never replaced or redrawn.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.multimodel.study import verified_file
from .active_set import ActiveSetConfig
from .prepare import installed_boltz_provenance
from .contracts import PreparedComplex
from .workflow_process import run_logged, require_child


def read(path):
    return json.loads(Path(path).read_text())


def load_protocol(path):
    """Strict data-only configuration; no executable command from a manifest."""
    value = yaml.safe_load(path.read_text())
    fields = {'version', 'name', 'base_config', 'targets', 'counts', 'output_root', 'cache',
              'confidence_sha256', 'pose_seed', 'allocation_seconds',
              'preparation_timeout_seconds', 'optimization_timeout_seconds'}
    if not isinstance(value, dict) or set(value) != fields or value['version'] != 1:
        raise ValueError('Unknown protein-panel protocol')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', value['name']):
        raise ValueError('Invalid panel name')
    counts = value['counts']
    if (not isinstance(counts, list) or len(counts) != 10 or len(set(counts)) != 10
            or any(type(n) is not int or not 1 <= n <= 50 for n in counts)):
        raise ValueError('Declare ten distinct carbon-chain counts within capacity')
    for key in ('allocation_seconds', 'preparation_timeout_seconds', 'optimization_timeout_seconds'):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('Budgets must be positive integer seconds')
    if type(value['pose_seed']) is not int or not 0 <= value['pose_seed'] < 2**32:
        raise ValueError('Native pose seed outside range')
    resolve = lambda name: str((path.resolve().parent / name).resolve())
    for key in ('base_config', 'targets'):
        asset = value[key]
        if (not isinstance(asset, dict) or set(asset) != {'path', 'sha256'}
                or not re.fullmatch(r'[0-9a-f]{64}', asset['sha256'])):
            raise ValueError('Assets require path and SHA256')
        value[key] = {**asset, 'path': resolve(asset['path'])}
    for key in ('output_root', 'cache'):
        value[key] = resolve(value[key])
    if not re.fullmatch(r'[0-9a-f]{64}', value['confidence_sha256']):
        raise ValueError('Missing pinned confidence checkpoint hash')
    return value


def panel_summary(rows):
    """Retain the original denominator, including unstarted/failed trajectories."""
    completed = [r for r in rows if r['status'] == 'completed']
    observed = [r for r in rows if r.get('elements')]
    return {'planned_trajectories': len(rows), 'completed': len(completed),
            'failed': sum(r['status'].endswith('_failed') for r in rows),
            'unstarted': sum(r['status'] == 'not_attempted' for r in rows),
            'ever_hard_noncarbon_observed': sum(r['elements']['ever_noncarbon'] for r in observed),
            'completed_with_hard_noncarbon': sum(r['elements']['ever_noncarbon'] for r in completed),
            'last_state_contains_noncarbon': sum(r['elements']['last_observed_noncarbon_count'] > 0 for r in observed),
            'cases': rows,
            'claim_limit': 'Elemental response at frozen predicted poses, fixed count; no fresh native confirmation, binding, synthesis or selectivity claim'}


def elemental_trajectory(pilot):
    """Read all raw element states, even when hard chemistry is invalid."""
    folder = pilot / 'fixed_initial_mask'
    rows, unreadable = [], []
    for p in sorted(folder.glob('metrics-*.json')):
        try:
            rows.append(read(p))
        except (OSError, json.JSONDecodeError) as exc:
            unreadable.append({'path': str(p), 'error': str(exc)})
    trace = []
    for row in rows:
        element = row['score']['elements']
        numbers = element['hard_atomic_numbers']
        noncarbon_slots = [s for s, n in zip(element['active_slots'], numbers) if n != 6]
        trace.append({'step': row['step'], 'hard_atomic_numbers': numbers,
                      'noncarbon_slots': noncarbon_slots, 'smiles': row['argmax_smiles'],
                      'binary_proxy': row['score']['raw'],
                      'conditional_expected_heteroatom_count': element['conditional_expected_heteroatom_count'],
                      'raw_expected_heteroatom_count_all_slots': element['raw_expected_heteroatom_count_all_slots'],
                      'carbon_logit_margins': element['carbon_logit_margin_over_best_heteroatom'],
                      'latent_displacement': row['latent_displacement'],
                      'latent_gradient_norm': row['score']['gradient_norm'],
                      'atom_logit_gradient_norm': element.get('atom_logit_gradient_norm'),
                      'natural_count': row['natural_selection']['count']})
    changes = [r for r in trace if r['noncarbon_slots']]
    return {'observed_states': len(trace), 'ever_noncarbon': bool(changes),
            'first_noncarbon_step': changes[0]['step'] if changes else None,
            'last_observed_noncarbon_count': len(trace[-1]['noncarbon_slots']) if trace else 0,
            'last_observed_smiles': trace[-1]['smiles'] if trace else None, 'trace': trace,
            'unreadable_artifacts': unreadable}


def child_config(base_path, prepared, count, output, case_id):
    """Clone the historical budget; replace target/count and request one arm."""
    value = deepcopy(yaml.safe_load(base_path.read_text()))
    value.pop('context_manifest', None)
    value.pop('contexts', None)
    value.update(name=case_id, contexts={count: {'path': str(prepared.resolve()),
                                               'sha256': file_sha256(prepared)}},
                 anchor_count=count, output_root=str(output.resolve()),
                 arms=['fixed_initial_mask'], element_diagnostics=True,
                 start={'kind': 'encoded_seed'})
    # Additional scales preserve the existing tolerances and required passes.
    value['diagnostics']['fd_epsilons'] = [0.001, 0.003, 0.01, 0.03, 0.1]
    return value


def run_case(protocol, base, row, root, persist, source_sha):
    """One independent native preparation then one latent optimization child."""
    root.mkdir(parents=True, exist_ok=False)
    row.update(status='preparing', attempt_dir=str(root), processes=[])
    persist()
    count = row['count']
    input_dir = root / 'input'
    prediction = root / 'prediction'
    prepared = root / 'prepared.pt'
    stem = f'carbon_chain_{count:02d}'
    commands = [
        [sys.executable, '-m', 'latentmol.boltz.prepare', 'target-chain',
         '--output', input_dir, '--count', count, '--target-manifest', protocol['targets']['path'],
         '--target', row['target']],
        [str(Path(sys.executable).parent / 'boltz'), 'predict', input_dir / f'{stem}.yaml',
         '--cache', protocol['cache'], '--out_dir', prediction, '--recycling_steps', '0',
         '--sampling_steps', '20', '--diffusion_samples', '1', '--sampling_steps_affinity', '20',
         '--diffusion_samples_affinity', '1', '--num_workers', '0', '--preprocessing-threads', '1',
         '--no_kernels', '--seed', protocol['pose_seed'], '--subsample_msa', '--num_subsampled_msa', '128'],
        [sys.executable, '-m', 'latentmol.boltz.prepare', 'snapshot',
         '--prediction-dir', prediction / f'boltz_results_{stem}', '--cache', protocol['cache'],
         '--output', prepared, '--ae-checkpoint', base.decoder['path'],
         '--input-manifest', input_dir / 'input_manifest.json']]
    for index, command in enumerate(commands):
        result = run_logged(command, root / f'prepare-{index}.log',
                            timeout_seconds=protocol['preparation_timeout_seconds'])
        row['processes'].append(result)
        persist()
        if result['returncode'] is None:
            raise RuntimeError('Cannot launch preparation executable; stop the allocation')
        if result['returncode'] != 0 or result['timed_out']:
            row['status'] = 'preparation_failed'
            persist()
            return
    row['prepared'] = {'path': str(prepared), 'sha256': file_sha256(prepared)}
    native = PreparedComplex.load(prepared)
    if (native.metadata.get('target_sequence_sha256') != row['sequence_sha256']
            or native.metadata.get('msa_sha256') != row['msa_sha256']
            or native.metadata.get('seed_smiles') != 'C' * count
            or int(native.active.sum()) != count):
        raise ValueError('Prepared context differs from declared target/count/MSA')
    row['reference_pose_sha256'] = native.metadata['reference_pose_sha256']
    del native
    config_path = root / 'optimization.yaml'
    config_path.write_text(yaml.safe_dump(child_config(Path(protocol['base_config']['path']), prepared,
                                                      count, root / 'optimization', row['id']), sort_keys=False))
    ActiveSetConfig.load(config_path)
    row.update(status='optimizing', config_sha256=file_sha256(config_path))
    persist()
    result = run_logged([sys.executable, '-m', 'latentmol.boltz.active_set', '--config', config_path],
                        root / 'optimize.log', timeout_seconds=protocol['optimization_timeout_seconds'])
    row['processes'].append(result)
    if len(result['run_paths']) == 1:
        pilot = Path(result['run_paths'][0])
        if not pilot.resolve().is_relative_to((root / 'optimization').resolve()):
            raise ValueError('Child output escaped its declared case root')
        record = read(pilot / 'run.json')
        if (record['source']['sha256'] != source_sha
                or record['source_archive_sha256'] != os.environ['LATENTMOL_SOURCE_SHA256']
                or record['config']['config_file_sha256'] != row['config_sha256']
                or record.get('target_sequence_sha256') != row['sequence_sha256']
                or record.get('msa_sha256') != row['msa_sha256']):
            raise ValueError('Child source/config identity mismatch')
        if record.get('error') == 'Frozen decoder or Boltz model state changed':
            raise ValueError('Frozen model changed; stop the whole study')
        row.update(pilot=str(pilot), elements=elemental_trajectory(pilot),
                   initial_audit=read(pilot / 'start-audit.json') if (pilot / 'start-audit.json').exists() else None)
    else:
        raise RuntimeError('Optimizer did not create exactly one experiment record')
    if result['returncode'] != 0 or result['timed_out']:
        row['status'] = 'optimization_failed'
    else:
        require_child(result, experiment=True)
        summary = read(pilot / 'summary.json')
        if (not summary['frozen_state_unchanged'] or set(summary['arms']) != {'fixed_initial_mask'}
                or summary['arms']['fixed_initial_mask']['steps'] != 100
                or [s['step'] for s in row['elements']['trace']] != list(range(101))):
            raise ValueError('Completed child does not meet the declared one-arm budget')
        if row['elements']['unreadable_artifacts']:
            raise ValueError('Completed child has unreadable metric evidence')
        row.update(status='completed')
    persist()


def seal_case(row, study_root):
    """Seal all finished case files, including failures, separately from the table."""
    attempt = Path(row['attempt_dir'])
    row['sealed_artifacts'] = {str(p.relative_to(attempt)): file_sha256(p)
                               for p in sorted(attempt.rglob('*')) if p.is_file()}
    if not row['sealed_artifacts']:
        raise ValueError('Terminal case lacks retained evidence')
    terminal = study_root / 'terminal'
    terminal.mkdir(exist_ok=True)
    path = terminal / f"{row['id']}.json"
    if path.exists():
        raise ValueError('Cannot overwrite a terminal case seal')
    write_json(path, row)


def verify_terminal(row, study_root):
    """A modified case table cannot silently skip a trajectory or its failure."""
    if read(study_root / 'terminal' / f"{row['id']}.json") != row or not row.get('sealed_artifacts'):
        raise ValueError('Terminal case table differs from its sealed evidence')
    attempt = Path(row['attempt_dir'])
    if not attempt.resolve().is_relative_to((study_root / 'cases' / row['id']).resolve()):
        raise ValueError('Terminal attempt escaped the declared case root')
    actual = {str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file()}
    if actual != set(row['sealed_artifacts']):
        raise ValueError('Terminal case file set changed')
    for relative, digest in row['sealed_artifacts'].items():
        verified_file(attempt / relative, digest)


def next_attempt(row, study_root):
    """Clear previous scientific observations and retain interrupted/orphan paths."""
    history = row['attempts'][:]
    if row['status'] != 'not_attempted':
        history.append({k: v for k, v in row.items() if k != 'attempts'})
    immutable = {k: row[k] for k in ('id', 'target', 'accession', 'sequence_sha256',
                                    'msa_sha256', 'count', 'pose_seed')}
    row.clear()
    row.update(**immutable, status='not_attempted', attempts=history)
    case_root = study_root / 'cases' / row['id']
    index = len(history) + 1
    while (case_root / f'attempt-{index:03d}').exists():
        history.append({'status': 'orphan_interrupted',
                        'attempt_dir': str(case_root / f'attempt-{index:03d}')})
        index += 1
    return case_root / f'attempt-{index:03d}'


def run(path, expected_sha, *, resume=False):
    """Continue a frozen case list only at boundaries, under one GPU allocation."""
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise ValueError('Require DTU allocation and immutable source archive')
    verified_file(path, expected_sha)
    protocol = load_protocol(path)
    for asset in (protocol['base_config'], protocol['targets']):
        verified_file(asset['path'], asset['sha256'])
    base = ActiveSetConfig.load(Path(protocol['base_config']['path']))
    if base.mode != 'vae' or base.optimizer.steps != 100:
        raise ValueError('Panel requires the frozen VAE and 100-update reference budget')
    for asset in (base.decoder, base.affinity_checkpoint):
        verified_file(asset['path'], asset['sha256'])
    verified_file(Path(protocol['cache']) / 'boltz2_conf.ckpt', protocol['confidence_sha256'])
    verified_file(Path(protocol['cache']) / 'boltz2_aff.ckpt', base.affinity_checkpoint['sha256'])
    targets_path = Path(protocol['targets']['path'])
    targets = read(targets_path)['targets']
    if len(targets) != 10 or len({t['id'] for t in targets}) != 10:
        raise ValueError('Require ten unique protein targets')
    for t in targets:
        if not re.fullmatch(r'[a-z0-9_]+', t['id']):
            raise ValueError('Unsafe target identifier')
        for key in ('msa', 'raw_record'):
            verified_file(targets_path.parent / t[key]['path'], t[key]['sha256'])
    source_sha = source_manifest(Path.cwd())['sha256']
    if source_sha != read(Path.cwd() / 'release_manifest.json')['sha256']:
        raise ValueError('Source differs from immutable release')
    root = Path(protocol['output_root'])
    identity = {'protocol_sha256': expected_sha, 'source_sha256': source_sha,
                'source_archive_sha256': os.environ['LATENTMOL_SOURCE_SHA256']}
    if resume:
        if read(root / 'identity.json') != identity:
            raise ValueError('Resume requires exactly the same source and protocol')
        rows = read(root / 'cases.json')
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'identity.json', identity)
        rows = [{'id': f"{t['id']}-c{count:02d}", 'target': t['id'], 'accession': t['accession'],
                 'sequence_sha256': t['sequence_sha256'], 'msa_sha256': t['msa']['sha256'],
                 'count': count, 'pose_seed': protocol['pose_seed'],
                 'status': 'not_attempted', 'attempts': []}
                for count in protocol['counts'] for t in targets]
    if [(r['target'], r['count']) for r in rows] != [(t['id'], n) for n in protocol['counts'] for t in targets]:
        raise ValueError('Case order differs from declared Cartesian panel')
    by_target = {t['id']: t for t in targets}
    for row in rows:
        target = by_target[row['target']]
        if (row['id'] != f"{row['target']}-c{row['count']:02d}"
                or row['sequence_sha256'] != target['sequence_sha256']
                or row['msa_sha256'] != target['msa']['sha256']
                or row['pose_seed'] != protocol['pose_seed']):
            raise ValueError('Case identity differs from sealed target protocol')
    def persist():
        # A terminal status is durable only with its independently written seal.
        snapshot = deepcopy(rows)
        for r in snapshot:
            if (r['status'] == 'completed' or r['status'].endswith('_failed')) and not r.get('sealed_artifacts'):
                r.update(pending_terminal_status=r['status'], status='sealing')
        write_json(root / 'cases.json', snapshot)
        write_json(root / 'summary.json', panel_summary(snapshot))
    persist()
    started = time.monotonic()
    pinned = [protocol['base_config'], protocol['targets'], base.decoder, base.affinity_checkpoint,
              {'path': str(Path(protocol['cache']) / 'boltz2_conf.ckpt'),
               'sha256': protocol['confidence_sha256']}]
    # Initial full hashes above plus unchanged file metadata avoid repeatedly
    # rereading multi-GB checkpoints. A changed stamp triggers a full check and
    # stops the study even if the replacement retained identical bytes.
    stamps = {a['path']: (Path(a['path']).stat().st_size, Path(a['path']).stat().st_mtime_ns) for a in pinned}
    def verify_shared():
        verified_file(path, expected_sha)
        if source_manifest(Path.cwd())['sha256'] != source_sha:
            raise ValueError('Executable source changed between cases')
        for a in pinned:
            p = Path(a['path'])
            if (p.stat().st_size, p.stat().st_mtime_ns) != stamps[a['path']]:
                verified_file(p, a['sha256'])
                raise ValueError('Pinned asset was modified during allocation')
        for t in targets:
            for key in ('msa', 'raw_record'):
                verified_file(targets_path.parent / t[key]['path'], t[key]['sha256'])
    # Worst-case child timeouts, plus setup/saving margin, stay inside walltime.
    reserve = 3 * protocol['preparation_timeout_seconds'] + protocol['optimization_timeout_seconds'] + 120
    with ExperimentRun(root / 'allocations', protocol['name'], {**protocol, **identity}, Path.cwd()) as allocation:
        allocation.annotate(native_provenance=installed_boltz_provenance())
        for row in rows:
            # If interrupted between sealing a terminal result and publishing
            # the table, reconcile the independently sealed exact case first.
            terminal = root / 'terminal' / f"{row['id']}.json"
            if row['status'] == 'sealing' and terminal.exists():
                sealed = read(terminal)
                if any(sealed[k] != row[k] for k in ('id', 'target', 'count', 'sequence_sha256',
                                                    'msa_sha256', 'pose_seed', 'attempt_dir')):
                    raise ValueError('Case seal differs from interrupted case identity')
                row.clear()
                row.update(sealed)
            if row['status'] == 'completed' or row['status'].endswith('_failed'):
                verify_terminal(row, root)
                continue
            if time.monotonic() - started + reserve >= protocol['allocation_seconds']:
                allocation.annotate(stopped_at_case_boundary=True, all_cases_processed=False)
                break
            verify_shared()
            attempt = next_attempt(row, root)
            allocation.log({'phase': 'case_start', 'id': row['id']})
            run_case(protocol, base, row, attempt, persist, source_sha)
            verify_shared()
            seal_case(row, root)
            persist()
            allocation.log({'phase': 'case_finished', 'id': row['id'], 'status': row['status']})
        persist()
        allocation.annotate(all_cases_processed=all(r['status'] == 'completed' or r['status'].endswith('_failed')
                                                   for r in rows), panel_summary=str(root / 'summary.json'))
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    print(run(args.config, args.sha256, resume=args.resume), flush=True)


if __name__ == '__main__':
    main()
