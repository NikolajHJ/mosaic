"""Serial, content-addressed fresh-native checkpoint evaluation on DTU HPC.

Source058 is a new workflow. Fixtures and the real processed-input audit must
pass on HPC before native requests start. A successful allocation may finish
only part of the sealed request list; all_requests_terminal is authoritative.
No discrete chemistry runs inside latent optimization or a scorer adapter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from .prepare import installed_boltz_provenance
from .trajectory_selection import artifact_tree, digest, freeze_selection, read, verified, verify_tree
from .trajectory_report import write_report
from .workflow_process import run_logged


def load_protocol(path):
    value = yaml.safe_load(Path(path).read_text())
    fields = {'version', 'name', 'output_root', 'cache', 'parents', 'selection', 'native',
              'checkpoints', 'allocation_seconds', 'native_timeout_seconds', 'audit_timeout_seconds'}
    if not isinstance(value, dict) or set(value) != fields or value['version'] != 1:
        raise ValueError('Unknown fresh-trajectory protocol fields')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', value['name']):
        raise ValueError('Unsafe study name')
    for key in ('output_root', 'cache'):
        if not PurePosixPath(value[key]).is_absolute():
            raise ValueError('HPC paths must be absolute')
    for key in ('allocation_seconds', 'native_timeout_seconds', 'audit_timeout_seconds'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError('Timeouts must be positive integer seconds')
    if value['allocation_seconds'] > 79200 or value['allocation_seconds'] <= value['native_timeout_seconds'] + value['audit_timeout_seconds'] + 120:
        raise ValueError('Allocation must fit the23h queue request with child cleanup reserve')
    selection = value['selection']
    if set(selection) != {'checkpoints', 'panel_sample_counts', 'panel_sample_draws', 'c13_sample_draws', 'random_sample_draws'}:
        raise ValueError('Unsupported candidate selection')
    steps = selection['checkpoints']
    if (not isinstance(steps, list) or steps != sorted(set(steps)) or not steps
            or steps[0] != 0 or steps[-1] != 100 or any(type(n) is not int or not 0 <= n <= 100 for n in steps)):
        raise ValueError('Declare ordered checkpoints spanning0–100')
    counts = selection['panel_sample_counts']
    if not isinstance(counts, list) or len(set(counts)) != len(counts) or any(type(n) is not int or not 6 <= n <= 15 for n in counts):
        raise ValueError('Sample subset must be predefined panel counts')
    for key in ('panel_sample_draws', 'c13_sample_draws', 'random_sample_draws'):
        if type(selection[key]) is not int or not 0 <= selection[key] <= 8:
            raise ValueError('Only saved categorical draw indices1–8 are available')
    native = value['native']
    expected_native = {'pose_seeds', 'structure_steps', 'affinity_steps', 'diffusion_samples',
                       'structure_recycles', 'affinity_recycles', 'precision', 'msa_rows',
                       'max_msa_seqs', 'mw_correction'}
    if set(native) != expected_native:
        raise ValueError('Unknown native settings')
    seeds = native['pose_seeds']
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(type(n) is not int or not 0 <= n < 2**32 for n in seeds):
        raise ValueError('Distinct native RNG seeds required')
    # Affinity recycling/precision are pinned native defaults with no exposed
    # CLI override in this commit; declaring unsupported values would be a lie.
    if (native['affinity_recycles'] != 5 or native['precision'] != 'bf16-mixed'
            or native['diffusion_samples'] != 1 or native['mw_correction'] is not False):
        raise ValueError('Unsupported pinned native affinity mode')
    for key in ('structure_steps', 'affinity_steps', 'msa_rows', 'max_msa_seqs'):
        if type(native[key]) is not int or native[key] < 1:
            raise ValueError('Native budgets must be positive integers')
    if type(native['structure_recycles']) is not int or native['structure_recycles'] < 0:
        raise ValueError('Invalid structure recycling budget')
    if set(value['checkpoints']) != {'affinity', 'confidence'} or any(not re.fullmatch(r'[0-9a-f]{64}', s) for s in value['checkpoints'].values()):
        raise ValueError('Both native checkpoints must be pinned')
    if set(value['parents']) != {'c13', 'random', 'panel'}:
        raise ValueError('Declare all three parent cohorts')
    for name, parent in value['parents'].items():
        expected = {'source_sha256', 'pilot', 'config_sha256', 'template'} if name == 'c13' else {'source_sha256', 'root', 'protocol'}
        if name == 'random':
            expected.add('template')
        if set(parent) != expected or not re.fullmatch(r'[0-9a-f]{64}', parent['source_sha256']):
            raise ValueError('Incomplete parent identity')
        if not PurePosixPath(parent.get('root', parent.get('pilot'))).is_absolute():
            raise ValueError('Absolute parent paths required')
        for key in ('protocol', 'template'):
            if key in parent and (set(parent[key]) != {'path', 'sha256'} or not PurePosixPath(parent[key]['path']).is_absolute()
                                  or not re.fullmatch(r'[0-9a-f]{64}', parent[key]['sha256'])):
                raise ValueError('Parent assets require absolute path and SHA256')
        if name == 'c13' and not re.fullmatch(r'[0-9a-f]{64}', parent['config_sha256']):
            raise ValueError('Missing C13 config identity')
    return value


def native_command(input_path, output, cache, request):
    n = request['native_settings']
    return [str(Path(sys.executable).parent / 'boltz'), 'predict', str(input_path),
            '--cache', str(cache), '--out_dir', str(output), '--model', 'boltz2',
            '--recycling_steps', str(n['structure_recycles']),
            '--sampling_steps', str(n['structure_steps']), '--diffusion_samples', '1',
            '--sampling_steps_affinity', str(n['affinity_steps']), '--diffusion_samples_affinity', '1',
            '--num_workers', '0', '--preprocessing-threads', '1', '--no_kernels',
            '--seed', str(request['seed']), '--subsample_msa', '--num_subsampled_msa', str(n['msa_rows']),
            '--max_msa_seqs', str(n['max_msa_seqs'])]


def execute_request(request, target, cache, attempt, protocol):
    """A fresh output directory prevents native 'already predicted' reuse.

    Process-group timeouts also kill preprocessing workers. Ordinary CLI/model
    failures are terminal retained outcomes; corrupt audit evidence stops the
    allocation rather than laundering it as poor molecular quality.
    """
    attempt.mkdir(parents=True, exist_ok=False)
    rid = 'r_' + request['id']
    input_path = attempt / f'{rid}.yaml'
    template = deepcopy(target['template'])
    next(x['ligand'] for x in template['sequences'] if 'ligand' in x)['smiles'] = request['smiles']
    input_path.write_text(yaml.safe_dump(template, sort_keys=False))
    expected = {key: target[key] for key in ('target_sequence_sha256', 'msa_sha256', 'protein_sequence', 'msa_path')}
    expected.update(smiles=request['smiles'], record_id=rid, input_yaml=str(input_path),
                    max_msa_seqs=request['native_settings']['max_msa_seqs'])
    write_json(attempt / 'expected.json', expected)
    command = native_command(input_path, attempt / 'prediction', cache, request)
    result = {'id': request['id'], 'seed': request['seed'], 'smiles': request['smiles'],
              'status': 'native_failed', 'retention_eligible': False, 'affinity': None, 'identity': None,
              'input_sha256': file_sha256(input_path), 'failure': None}
    process = run_logged(command, attempt / 'native.log', timeout_seconds=protocol['native_timeout_seconds'])
    result['native_process'] = process
    if process['returncode'] is None:
        raise RuntimeError('Cannot launch pinned native executable')
    if process['returncode'] != 0 or process['timed_out']:
        result['failure'] = 'Native CLI timeout/failure; see retained log'
        return result
    prediction = attempt / 'prediction' / f'boltz_results_{rid}'
    affinity_path = prediction / 'predictions' / rid / f'affinity_{rid}.json'
    if not affinity_path.exists():
        result['failure'] = 'Native CLI returned zero but skipped affinity output'
        return result
    affinity = read(affinity_path)
    keys = ('affinity_probability_binary', 'affinity_pred_value')
    if any(k not in affinity or not math.isfinite(float(affinity[k])) for k in keys):
        raise ValueError('Native scalar output missing/nonfinite')
    if not 0 <= float(affinity['affinity_probability_binary']) <= 1:
        raise ValueError('Native binary probability outside range')
    result['affinity'] = affinity
    result['affinity_sha256'] = file_sha256(affinity_path)
    process = run_logged([sys.executable, '-m', 'latentmol.boltz.processed_identity',
                          '--prediction-dir', prediction, '--cache', cache,
                          '--expected', attempt / 'expected.json', '--output', attempt / 'identity.json'],
                         attempt / 'audit.log', timeout_seconds=protocol['audit_timeout_seconds'])
    result['audit_process'] = process
    if process['returncode'] != 0 or process['timed_out']:
        raise RuntimeError('Processed-input audit failed; stop and investigate retained evidence')
    identity = read(attempt / 'identity.json')
    if identity.get('record_id') != rid or identity['artifact_sha256']['affinity'] != result['affinity_sha256']:
        raise ValueError('Native audit/affinity association mismatch')
    result.update(identity=identity, retention_eligible=identity['retention_eligible'],
                  status='completed' if identity['retention_eligible'] else 'identity_ineligible', failure=identity['failure'])
    return result


def run(path, expected_sha, resume=False):
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise ValueError('Require DTU allocation and immutable source archive')
    verified(path, expected_sha)
    protocol = load_protocol(path)
    source_sha = source_manifest(Path.cwd())['sha256']
    if read(Path.cwd() / 'release_manifest.json')['sha256'] != source_sha:
        raise ValueError('Source differs from sealed manifest')
    native_provenance = installed_boltz_provenance()
    cache = Path(protocol['cache'])
    pinned = [{'path': str(cache / f'boltz2_{kind}.ckpt'), 'sha256': sha}
              for kind, sha in (('aff', protocol['checkpoints']['affinity']), ('conf', protocol['checkpoints']['confidence']))]
    for parent in protocol['parents'].values():
        pinned.extend(parent[k] for k in ('protocol', 'template') if k in parent)
    for asset in pinned:
        verified(**asset)
    stamps = {a['path']: (Path(a['path']).stat().st_size, Path(a['path']).stat().st_mtime_ns) for a in pinned}
    root = Path(protocol['output_root'])
    identity = {'source_sha256': source_sha, 'source_archive_sha256': os.environ['LATENTMOL_SOURCE_SHA256'],
                'protocol_sha256': expected_sha}
    # auto allows a later bounded allocation to resume after a killed setup, but
    # rejects an orphan/ambiguous initial directory rather than overwriting it.
    if root.exists():
        if not resume or read(root / 'identity.json') != identity:
            raise ValueError('Existing study requires exact-source/config resume')
        selection = read(root / 'selection.json')
        if read(root / 'selection_identity.json') != {'sha256': file_sha256(root / 'selection.json')}:
            raise ValueError('Frozen selection changed')
    else:
        selection = freeze_selection(protocol)
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / 'identity.json', identity)
        write_json(root / 'selection.json', selection)
        write_json(root / 'selection_identity.json', {'sha256': file_sha256(root / 'selection.json')})
    for parent_root, seal in selection['parent_seals'].items():
        verify_tree(Path(parent_root), seal)
    results = {}
    for request in selection['requests']:
        terminal = root / 'terminal' / f"{request['id']}.json"
        if terminal.exists():
            sealed = read(terminal)
            if sealed['request'] != request:
                raise ValueError('Terminal result belongs to another request')
            verify_tree(Path(sealed['attempt']), sealed['artifact_sha256'])
            results[request['id']] = sealed['result']
    started = time.monotonic()
    reserve = protocol['native_timeout_seconds'] + protocol['audit_timeout_seconds'] + 120
    with ExperimentRun(root / 'allocations', protocol['name'], {**protocol, **identity}, Path.cwd()) as allocation:
        allocation.annotate(native_provenance=native_provenance, selection_sha256=file_sha256(root / 'selection.json'))
        write_report(root, selection, results, protocol['native']['pose_seeds'])
        for request in selection['requests']:
            if request['id'] in results:
                continue
            if time.monotonic() - started + reserve >= protocol['allocation_seconds']:
                allocation.annotate(stopped_at_request_boundary=True)
                break
            verified(path, expected_sha)
            if source_manifest(Path.cwd())['sha256'] != source_sha:
                raise ValueError('Executable source changed')
            for asset in pinned:
                p = Path(asset['path'])
                if (p.stat().st_size, p.stat().st_mtime_ns) != stamps[asset['path']]:
                    raise ValueError('Pinned model/protocol asset modified during allocation')
            target = selection['targets'][request['target_key']]
            verified(target['msa_path'], target['msa_sha256'])
            request_root = root / 'requests' / request['id']
            index = 1
            while (request_root / f'attempt-{index:03d}').exists():
                index += 1  # Interrupted attempts remain; terminal failures are never retried.
            attempt = request_root / f'attempt-{index:03d}'
            allocation.log({'phase': 'native_start', 'request': request['id'], 'attempt': str(attempt)})
            result = execute_request(request, target, cache, attempt, protocol)
            verified(target['msa_path'], target['msa_sha256'])
            for asset in pinned:
                p = Path(asset['path'])
                if (p.stat().st_size, p.stat().st_mtime_ns) != stamps[asset['path']]:
                    raise ValueError('Pinned asset changed during native prediction')
            write_json(attempt / 'result.json', result)
            terminal = root / 'terminal'; terminal.mkdir(exist_ok=True)
            write_json(terminal / f"{request['id']}.json", {'request': request, 'attempt': str(attempt),
                       'result': result, 'artifact_sha256': artifact_tree(attempt)})
            results[request['id']] = result
            allocation.log({'phase': 'native_finished', 'request': request['id'], 'status': result['status'],
                            'native_seconds': result['native_process']['seconds'],
                            'audit_seconds': (result.get('audit_process') or {}).get('seconds')})
            if len(results) % 10 == 0:
                write_report(root, selection, results, protocol['native']['pose_seeds'])
        for parent_root, seal in selection['parent_seals'].items():
            verify_tree(Path(parent_root), seal)
        summary = write_report(root, selection, results, protocol['native']['pose_seeds'])
        allocation.annotate(all_requests_terminal=summary['all_requests_terminal'],
                            terminal_requests=len(results), unique_native_requests=len(selection['requests']))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(run(args.config, args.sha256, args.resume), flush=True)


if __name__ == '__main__':
    main()
