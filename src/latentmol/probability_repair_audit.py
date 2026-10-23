"""Audit saved latent trajectories for probability-supported chemical repairs.

Runs on HPC without loading Boltz or optimizing z. The frozen historical graph
decoder supplies precisely the conditional probabilities of a saved state.
Missing/failed states stay in the declared denominator. Witness molecules are
diagnostic artifacts only, never substituted into native affinity evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import torch

from .config_identity import verify_config_identity
from .evaluate import load_model
from .experiments import ExperimentRun, file_sha256, write_json
from .graph_policy import graph_from_decoder
from .multimodel.study import hard_digest, module_digest, tensor_digest
from .probability_repair import ProbabilityRepairConfig, ProbabilityRepairEvaluator
from .representation import HardGraphBatch


def run_audit(protocol_path, *, expected_sha256):
    """Evaluate all prescribed roles/states; never select by score or validity."""
    if not os.environ.get('LSB_JOBID') or not os.environ.get('LATENTMOL_SOURCE_SHA256'):
        raise ValueError('Require an HPC allocation and immutable source identity')
    protocol_path = Path(protocol_path)
    if file_sha256(protocol_path) != expected_sha256:
        raise ValueError('Probability-repair protocol changed')
    protocol = json.loads(protocol_path.read_text())
    if set(protocol) != {'version', 'name', 'output_root', 'parents', 'search'} or protocol['version'] != 1:
        raise ValueError('Unsupported probability-repair protocol')
    config = ProbabilityRepairConfig(**protocol['search'])
    parents = protocol['parents']
    if (not parents or len({p['id'] for p in parents}) != len(parents)
            or any(set(p) != {'id', 'pilot', 'run_sha256', 'steps'} for p in parents)
            or any(not p['steps'] or len(set(p['steps'])) != len(p['steps'])
                   or any(type(s) is not int or s < 0 for s in p['steps']) for p in parents)):
        raise ValueError('Declare unique parents and unique nonnegative saved-state indices')
    decoders, decoder_digests, evaluators, rows = {}, {}, {}, []
    with ExperimentRun(Path(protocol['output_root']), protocol['name'],
                       {**protocol, 'protocol_sha256': expected_sha256}, Path.cwd()) as run:
        for parent in parents:
            folder = Path(parent['pilot'])
            record_path = folder / 'run.json'
            if file_sha256(record_path) != parent['run_sha256']:
                raise ValueError('Sealed parent run record changed')
            record = json.loads(record_path.read_text())
            config_receipt = verify_config_identity(record, allow_legacy_active_set_contexts=True)
            if record['status'] not in ('completed', 'failed'):
                raise ValueError('Audit only terminal parents, not a changing running record')
            settings = record['config']
            asset = settings['decoder']
            if file_sha256(Path(asset['path'])) != asset['sha256']:
                raise ValueError('Parent frozen decoder checkpoint changed')
            checkpoint_key = asset['sha256']
            if checkpoint_key not in decoders:
                decoder, training = load_model(Path(asset['path']), 'cpu')
                decoder.eval().requires_grad_(False)
                if training['train_config']['mode'] != settings['mode']:
                    raise ValueError('Parent decoder mode differs from checkpoint')
                if decoder.spec.fingerprint() != record['graph_spec_sha256']:
                    raise ValueError('Parent graph schema differs from checkpoint')
                decoders[checkpoint_key] = decoder
                decoder_digests[checkpoint_key] = module_digest(decoder)
                evaluators[checkpoint_key] = ProbabilityRepairEvaluator(decoder.spec, config)
            decoder, evaluator = decoders[checkpoint_key], evaluators[checkpoint_key]
            if decoder.spec.fingerprint() != record['graph_spec_sha256']:
                raise ValueError('Parent graph schema differs from cached checkpoint')
            if decoder_digests[checkpoint_key] != record['decoder_state_sha256']:
                raise ValueError('Decoder weights differ from the historical parent frozen state')
            for arm in settings.get('arms', ('active_set', 'fixed_initial_mask')):
                if arm not in ('active_set', 'fixed_initial_mask'):
                    raise ValueError('Unsupported historical arm')
                candidates_path = folder / arm / 'candidates.jsonl'
                candidate_sha = file_sha256(candidates_path) if candidates_path.exists() else None
                candidates = {}
                if candidate_sha:
                    for line in candidates_path.read_text().splitlines():
                        item = json.loads(line)
                        if item['draw'] == 0:
                            if item['step'] in candidates:
                                raise ValueError('Duplicate saved argmax candidate')
                            candidates[item['step']] = item
                for step in parent['steps']:
                    state_path = folder / arm / f'state-{step:04d}.pt'
                    row = {'parent': parent['id'], 'pilot': str(folder), 'arm': arm, 'step': step,
                           'parent_status': record['status'], 'parent_error': record.get('error'),
                           'parent_source_archive_sha256': record['source_archive_sha256'],
                           'decoder_checkpoint_sha256': checkpoint_key, 'config_identity': config_receipt,
                           'saved_candidate_parity': 'unavailable', 'status': 'missing_state'}
                    if state_path.exists():
                        state_sha = file_sha256(state_path)
                        state = torch.load(state_path, map_location='cpu', weights_only=True)
                        gate_path = folder / arm / f'gate-{step:04d}.json'
                        gate_sha = file_sha256(gate_path) if gate_path.exists() else None
                        if gate_sha:
                            gate = json.loads(gate_path.read_text())
                            if (tensor_digest({'z': state['z']}) != gate['point_sha256']
                                    or state['active'].nonzero().flatten().tolist() != gate['active_slots']
                                    or state['context_sha256'] != gate['context_sha256']):
                                raise ValueError('Saved state differs from the recorded derivative-gate point')
                            row.update(saved_gate_sha256=gate_sha, saved_gate_status=gate['status'],
                                       saved_gate_point_parity='passed')
                        active = state['active']
                        if active.dtype != torch.bool or not active.any():
                            row.update(status='unsupported_occupancy', state_sha256=state_sha)
                        else:
                            with torch.no_grad():
                                decoded = decoder.decode(state['z'])
                                graph = graph_from_decoder(decoded, decoder.spec, temperature=settings['temperature'],
                                                           fixed_active=active)
                                hard = HardGraphBatch(*(x.argmax(-1) for x in (graph.atoms, graph.charges, graph.bonds)))
                            if step in candidates:
                                saved = candidates[step]
                                if (hard_digest(hard) != saved['graph_sha256']
                                        or any(getattr(hard, name)[0].tolist() != saved['labels'][name]
                                               for name in ('atoms', 'charges', 'bonds'))):
                                    raise ValueError('Replayed decoder argmax differs from saved candidate')
                                row['saved_candidate_parity'] = 'passed'
                            if not torch.equal(hard.atoms[0] != 0, active):
                                raise ValueError('Hard candidate occupancy differs from saved conditional mask')
                            report = evaluator.diagnose(*(getattr(hard, k)[0] for k in ('atoms', 'charges', 'bonds')),
                                                        probabilities=(graph.atoms[0], graph.charges[0], graph.bonds[0]))
                            row.update(status=report['status'], repair=report, state_sha256=state_sha,
                                       active_count=int(active.sum()), native_context_sha256=state.get('context_sha256'))
                        if file_sha256(state_path) != state_sha:
                            raise ValueError('Saved state changed during audit')
                        if gate_sha and file_sha256(gate_path) != gate_sha:
                            raise ValueError('Derivative gate changed during audit')
                    rows.append(row)
                    with (run.path / 'repairs.jsonl').open('a') as handle:
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                if file_sha256(record_path) != parent['run_sha256'] or (
                        candidate_sha and file_sha256(candidates_path) != candidate_sha):
                    raise ValueError('Parent artifacts changed during audit')
            run.log({'parent': parent['id'], 'states_recorded': sum(r['parent'] == parent['id'] for r in rows)})
        frozen = all(module_digest(decoders[k]) == decoder_digests[k] for k in decoders)
        if not frozen:
            raise ValueError('Frozen decoder changed during repair audit')
        write_json(run.path / 'summary.json', {
            'planned_state_memberships': len(rows), 'status_counts': dict(Counter(r['status'] for r in rows)),
            'frozen_decoders_unchanged': frozen, 'policy': config.policy(),
            'compute': {k: {'candidate_checks': e.checks, 'cache_hits': e.cache_hits} for k, e in evaluators.items()},
            'binding_model_calls': 0, 'optimization_updates': 0,
            'claim_limit': 'Observed fixed-occupancy repair witnesses and categorical probability support; no global nearest molecule or binding claim'})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(run_audit(args.protocol, expected_sha256=args.sha256), flush=True)


if __name__ == '__main__':
    main()
