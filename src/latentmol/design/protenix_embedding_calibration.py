"""Calibrate a Protenix bridge on decoded contact values and finite latent steps."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from latentmol.design.latent_protenix import LatentDesignConfig, _json, _model_identity, run_design, schedules
from latentmol.design.protenix_embedding import (
    HiddenChemicalBridge, LearnedProjectionScorer, load_frozen_decoder, load_start_state,
    projection_labels, projection_loss,
)
from latentmol.design.protenix_embedding_probe import Probe, load_points
from latentmol.experiments import file_sha256
from latentmol.guidance.mosaic_protenix_embedding import MosaicProtenixEmbeddingScorer
from latentmol.representation import HardGraphBatch


@dataclass(frozen=True)
class ContactNeighborhood:
    name: str
    step: int
    temperature: float
    directions: tuple[str, ...]
    points: torch.Tensor
    targets: torch.Tensor


def build_neighborhood(probe, name, step, point, temperature, radius, generator):
    relaxed, direct_gradient = probe.gradient(point, temperature, 'relaxed', details=False)
    _, mapped_gradient = probe.gradient(point, temperature, 'mapped', details=False)
    random = torch.randn(point.shape, generator=generator)
    points, names, targets = [point], ['center'], [relaxed['loss']]
    with torch.no_grad():
        for direction, vector in [('direct', -direct_gradient), ('mapped', -mapped_gradient), ('random', random)]:
            magnitude = vector.norm()
            if magnitude == 0:
                continue
            candidate = point + radius * (vector / magnitude)
            _, _, graph = probe.decode(candidate, temperature)
            target, _ = probe.native.score_graph(graph)
            points.append(candidate)
            names.append(direction)
            targets.append(float(target))
    return ContactNeighborhood(name, step, temperature, tuple(names), torch.cat(points),
                               torch.tensor(targets, dtype=point.dtype))


def contact_errors(scores, targets, radius):
    value_error = F.mse_loss(scores, targets)
    slopes = (scores[1:] - scores[0]) / radius
    reference = (targets[1:] - targets[0]) / radius
    return value_error, F.mse_loss(slopes, reference)


def predict_neighborhood(decoder, native, bridge, indices, neighborhood):
    with torch.no_grad():
        decoded, features = decoder.decode_with_features(neighborhood.points)
        indices = indices.expand(neighborhood.points.shape[0], -1)
        labels = projection_labels(decoded, indices)
    projections = bridge(features, indices)
    scores = torch.stack([
        native({key: value[i:i+1] for key, value in projections.items()})[0]
        for i in range(len(neighborhood.directions))
    ])
    if not bool(scores.isfinite().all()):
        raise FloatingPointError('Nonfinite mapped contact in calibration')
    return scores, projection_loss(projections, labels, native.bank)[0]


def fit_calibration(decoder, native, original, active, design, config, starts, output):
    probe = Probe(decoder, original, native, active)
    radius = config['radius']
    generator = torch.Generator().manual_seed(config['seed'])

    def prior_points(seeds, prefix):
        return [(f'{prefix}-{seed}', 0, torch.randn((1, decoder.config.latent),
                 generator=torch.Generator().manual_seed(seed))) for seed in seeds]

    if set(config['train_seeds']) & set(config['holdout_seeds']):
        raise ValueError('Calibration training and holdout seeds overlap')
    training_points = prior_points(config['train_seeds'], 'prior')
    for source in config['sources']:
        training_points.extend((f"{source['name']}-{step}", step, point)
                               for _, step, point in load_points(dict(source, fresh_seeds=[]), decoder))
    holdout_points = prior_points(config['holdout_seeds'], 'held-out-prior')
    holdout_points.extend((name, state['step'], state['latent']) for name, state in starts.items())

    def collect(points):
        return [build_neighborhood(probe, name, step, point, schedules(design, step)['temperature'],
                                   radius, generator) for name, step, point in points]

    training, holdout = collect(training_points), collect(holdout_points)
    torch.save(dict(training=[vars(row) for row in training], holdout=[vars(row) for row in holdout]),
               output/'contact-targets.pt')
    print(json.dumps(dict(stage='contact_targets_ready', train=len(training), holdout=len(holdout))), flush=True)
    bridge = copy.deepcopy(original).train().requires_grad_(True)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config['learning_rate'])

    def losses(row):
        scores, embedding = predict_neighborhood(decoder, native, bridge, probe.indices, row)
        value, slope = contact_errors(scores, row.targets, radius)
        return scores, torch.stack((embedding, value, slope))

    def evaluate_holdout():
        rows = []
        with torch.no_grad():
            for row in holdout:
                scores, errors = losses(row)
                rows.append(dict(name=row.name, directions=row.directions, target_losses=row.targets.tolist(),
                                 mapped_losses=scores.tolist(), errors=errors.tolist()))
        return rows

    started = time.monotonic()
    before = evaluate_holdout()
    weights = torch.tensor([config['embedding_weight'], config['value_weight'], config['slope_weight']])
    parameters = tuple(bridge.parameters())
    checked_contact_path = False
    for epoch in range(1, config['epochs'] + 1):
        errors = []
        for index in torch.randperm(len(training), generator=generator).tolist():
            optimizer.zero_grad(set_to_none=True)
            scores, terms = losses(training[index])
            if not scores.requires_grad:
                raise ValueError('Required mapped contact calibration path is detached')
            if not checked_contact_path:
                derivatives = torch.autograd.grad(scores.sum(), parameters, retain_graph=True, allow_unused=True)
                if any(g is None or not bool(g.isfinite().all()) for g in derivatives):
                    raise ValueError('Missing or nonfinite contact-only calibration parameter gradient')
                checked_contact_path = True
            objective = (weights * terms).sum()
            if not bool(objective.isfinite()):
                raise FloatingPointError('Nonfinite calibration objective')
            objective.backward()
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in parameters):
                raise ValueError('Missing or nonfinite calibration parameter gradient')
            optimizer.step()
            errors.append(terms.detach())
        print(json.dumps(dict(stage='calibration_epoch', epoch=epoch,
                              mean_errors=torch.stack(errors).mean(0).tolist())), flush=True)
    bridge.eval().requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)
    _json(output/'calibration.json', dict(config=config, error_order=['embedding_mse', 'contact_mse', 'slope_mse'],
        before=before, after=evaluate_holdout(), seconds=time.monotonic()-started))
    torch.save(dict(state_dict=bridge.state_dict(), bank=bridge.bank, dimensions=bridge.dimensions,
                    hidden=bridge.hidden, calibration_config=config), output/'mapping-calibrated.pt')
    return bridge


class DirectScorer:
    def __init__(self, native):
        self.native = native
        self.identity = dict(native.identity, input_boundary='decoded_relaxed_graph')

    def __call__(self, graph):
        return self.native.score_graph(graph)

    def close(self):
        self.native.close()


def summarize_arm(directory, arm, stride):
    files = directory/'arms/latent'
    metrics = [json.loads(line) for line in (files/'metrics.jsonl').read_text().splitlines()]
    sampled = []
    for row in metrics:
        if row['step'] % stride == 0 or row['endpoint']:
            loss = (row['score']['binding_loss'] if arm == 'direct'
                    else row['native_aux']['relaxed_graph_contact_loss'])
            sampled.append(dict(step=row['step'], decoded_contact=-loss, raw_valid=row['raw_valid'],
                                optimization_contact=-row['score']['binding_loss']))
    ranked = sorted((row for row in sampled if row['raw_valid']), key=lambda row: -row['decoded_contact'])
    candidates = {row['step']: row for row in map(json.loads, (files/'candidates.jsonl').read_text().splitlines())
                  if row['valid'] and 'labels' in row}
    finalists = []
    for row in ranked:
        candidate = candidates[row['step']]
        if all(candidate['smiles'] != item['smiles'] for item in finalists):
            finalists.append(dict(row, smiles=candidate['smiles'], labels=candidate['labels']))
        if len(finalists) == 3:
            break
    return dict(start=sampled[0], final=sampled[-1], best_valid=ranked[0] if ranked else None,
                raw_valid_updates=sum(row['raw_valid'] for row in metrics[:-1]),
                mean_decoded_contact=sum(row['decoded_contact'] for row in sampled)/len(sampled),
                sampled_states=len(sampled), finalists=finalists)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mosaic-python', required=True)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    if not os.environ.get('LSB_JOBID') or file_sha256(Path(args.plan)) != args.plan_sha256:
        raise ValueError('Require the allocated comparison plan')
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    base = json.loads(Path(config['base_config']).read_text())
    checkpoint = base['checkpoint']
    decoder = load_frozen_decoder(checkpoint)
    decoder_before = _model_identity(decoder)
    starts = {'initial': load_start_state(dict(start=config['initial_start']), decoder),
              'step180': load_start_state(base, decoder)}
    active = starts['initial']['active']
    if not torch.equal(active, starts['step180']['active']):
        raise ValueError('The two starts require identical fixed ligand slots')
    design = LatentDesignConfig.from_mapping(base['design'])
    fit = config['calibration']
    stride = design.snapshot_interval
    updates = config['updates_per_start']
    if (not math.isfinite(fit['radius']) or fit['radius'] <= 0 or fit['epochs'] <= 0
            or updates <= 0 or any(state['step'] % stride or updates % stride for state in starts.values())):
        raise ValueError('Calibration radius and matched snapshot grid are incompatible')
    mapping = base['mapping_checkpoint']
    if file_sha256(Path(mapping['path'])) != mapping['sha256']:
        raise ValueError('Original mapping checkpoint changed')
    asset = torch.load(mapping['path'], map_location='cpu', weights_only=True)
    original = HiddenChemicalBridge(asset['bank'], asset['dimensions'], asset['hidden'])
    original.load_state_dict(asset['state_dict'])
    original.eval().requires_grad_(False)
    original_before = _model_identity(original)
    native_config = json.loads(Path(base['native_config']).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    _json(args.output/'start.json', dict(config=config, job_id=os.environ['LSB_JOBID'], plan=args.plan_sha256))

    def open_native(log_name):
        native = MosaicProtenixEmbeddingScorer(decoder.spec, active, config=native_config,
            command=[args.mosaic_python, '-m', 'latentmol.guidance.mosaic_protenix_embedding_worker'],
            stderr_path=args.output/log_name, timeout=900, startup_timeout=900)
        if native.bank != original.bank:
            native.close()
            raise ValueError('Native bank differs from the original trained map')
        return native

    with open_native('calibration-native.stderr.log') as native:
        calibrated = fit_calibration(decoder, native, original, active, design, fit, starts, args.output)
    if _model_identity(decoder) != decoder_before or _model_identity(original) != original_before:
        raise ValueError('Frozen VAE or original bridge changed during calibration')
    results = dict(status='running', checkpoint_sha256=checkpoint['sha256'], original_mapping_sha256=mapping['sha256'],
                   calibrated_mapping_sha256=file_sha256(args.output/'mapping-calibrated.pt'), arms={})
    for start_name, state in starts.items():
        arm_config = replace(design, updates=state['step']+updates)
        for arm, bridge in [('original', original), ('calibrated', calibrated), ('direct', None)]:
            name = f'{start_name}-{arm}'
            directory = args.output/name
            print(json.dumps(dict(stage='comparison_arm', arm=name, updates=updates)), flush=True)
            with open_native(name+'-native.stderr.log') as native:
                def factory(spec, expected_active):
                    if not torch.equal(active, expected_active):
                        raise ValueError('Comparison arm changed fixed slots')
                    return DirectScorer(native) if bridge is None else LearnedProjectionScorer(native, bridge, active, stride)

                run = run_design(decoder, factory, arm_config, directory, checkpoint_sha256=checkpoint['sha256'],
                                 start_state=state, wall_seconds=args.wall_seconds-(time.monotonic()-started))
            if run['status'] != 'completed':
                raise RuntimeError(f'{name} stopped before the requested endpoint')
            results['arms'][name] = dict(summarize_arm(directory, arm, stride),
                                         seconds=run['elapsed_seconds'], updates=run['updates_this_segment'])
            _json(args.output/'progress.json', results)

    with open_native('finalists-native.stderr.log') as native, torch.no_grad():
        for record in results['arms'].values():
            for finalist in record['finalists']:
                labels = finalist.pop('labels')
                graph = HardGraphBatch(*(torch.tensor(labels[key])[None] for key in ('atoms', 'charges', 'bonds')))
                loss, _ = native.score_graph(graph.to_relaxed(decoder.spec))
                finalist['hard_fixed_context_contact'] = -float(loss)
    results.update(status='completed', elapsed_seconds=time.monotonic()-started)
    _json(args.output/'result.json', results)


if __name__ == '__main__':
    main()
