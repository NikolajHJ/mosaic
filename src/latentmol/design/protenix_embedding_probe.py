"""Matched same-latent gradient and finite-step observations for two Protenix paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.embedding_diagnostics import decompose_prototype_span, softmax_sensitivity
from latentmol.design.latent_protenix import _model_identity, _json, hard_from_graph, schedules, LatentDesignConfig
from latentmol.design.protenix_embedding import HiddenChemicalBridge, load_frozen_decoder
from latentmol.graph_policy import graph_from_decoder
from latentmol.experiments import file_sha256
from latentmol.guidance.mosaic_protenix_embedding import MosaicProtenixEmbeddingScorer, HEADS


def norm(v):
    return float(v.double().norm())


def cosine(a, b):
    denominator = norm(a)*norm(b)
    return float((a.double()*b.double()).sum())/denominator if denominator else None


def load_points(config, decoder):
    points = []
    raw = Path(config['raw_blocks'])
    for step in config['steps']:
        found = None
        for path in sorted(raw.glob('states-*.npz')):
            low, high = map(int, path.stem.split('-')[1:])
            if low <= step <= high:
                with np.load(path) as block:
                    row = np.flatnonzero(block['steps'] == step)
                    if len(row) != 1:
                        raise ValueError('Saved latent step is ambiguous')
                    found = torch.tensor(block['latent'][row[0]], dtype=torch.float32)[None]
                break
        if found is None:
            raise ValueError('Saved latent unavailable at step ' + str(step))
        points.append((f'mapping-step-{step}', step, found))
    for seed in config['fresh_seeds']:
        z = torch.randn((1, decoder.config.latent), generator=torch.Generator().manual_seed(seed))
        points.append((f'fresh-prior-{seed}', 0, z))
    return points


class Probe:
    def __init__(self, decoder, bridge, native, active):
        self.decoder, self.bridge, self.native, self.active = decoder, bridge, native, active
        self.indices = torch.nonzero(active).flatten()[None]

    def decode(self, z, temperature):
        logits, features = self.decoder.decode_with_features(z)
        graph = graph_from_decoder(logits, self.decoder.spec, temperature=temperature, fixed_active=self.active)
        return logits, features, graph

    def gradient(self, z, temperature, path, *, details=True):
        point = z.detach().clone().requires_grad_(True)
        logits, features, graph = self.decode(point, temperature)
        if path == 'mapped':
            targets = self.bridge(features, self.indices)
            loss, aux = self.native(targets)
        elif path == 'exact_projection':
            ids = self.indices[0]
            element = torch.tensor(self.native.bank['atom_element']['prototypes'])
            charge = torch.tensor(self.native.bank['atom_charge']['prototypes'])
            charge_labels = self.native.bank['atom_charge']['labels']
            reference = charge_labels.index(1)
            direction = charge[reference] / torch.asinh(torch.tensor(1.))
            expected_charge = graph.charges[0, ids] @ torch.tensor(self.decoder.spec.formal_charges, dtype=torch.float32)
            bond = torch.tensor(self.native.bank['pair_presence']['prototypes'])[1]
            targets = dict(atom_element=(graph.atoms[0, ids, 1:] @ element)[None],
                atom_charge=(expected_charge.asinh()[:, None]*direction)[None],
                pair_presence=((1-graph.bonds[0, ids][:, ids, 0])[..., None]*bond)[None])
            loss, aux = self.native(targets)
        else:
            targets = dict(atoms=graph.atoms, charges=graph.charges, bonds=graph.bonds)
            loss, aux = self.native.score_graph(graph)
        gradient, = torch.autograd.grad(loss, point, retain_graph=True)
        if not bool(gradient.isfinite().all()) or not bool(loss.isfinite()):
            raise ValueError('Nonfinite matched objective or required latent gradient')
        record = dict(loss=float(loss.detach()), gradient_norm=norm(gradient), native_aux=aux)
        if details:
            derivatives = torch.autograd.grad(loss, tuple(targets.values()), retain_graph=True)
            record['channels'] = {}
            orthogonal_latent = torch.zeros_like(point)
            for (key, value), derivative in zip(targets.items(), derivatives):
                latent, = torch.autograd.grad(value, point, grad_outputs=derivative, retain_graph=True)
                info = dict(latent_gradient_norm=norm(latent), native_input_gradient_norm=norm(derivative))
                if path == 'mapped':
                    result = decompose_prototype_span(value, torch.tensor(self.native.bank[key]['prototypes']), derivative)
                    info.update(result['metrics'])
                    off_latent, = torch.autograd.grad(value, point,
                        grad_outputs=result['gradient_orthogonal'].to(value), retain_graph=True)
                    orthogonal_latent += off_latent
                    info['off_span_latent_gradient_norm'] = norm(off_latent)
                record['channels'][key] = info
            if path == 'mapped':
                record['off_span_latent_gradient_norm'] = norm(orthogonal_latent)
                record['off_span_latent_gradient_cosine'] = cosine(gradient, orthogonal_latent)
            ids = self.indices[0]
            upper = torch.triu(torch.ones((len(ids), len(ids)), dtype=torch.bool), diagonal=1)
            record['categorical_sensitivity'] = dict(
                atoms=softmax_sensitivity(logits.atoms[0, ids, 1:], temperature),
                charges=softmax_sensitivity(logits.charges[0, ids], temperature),
                bonds=softmax_sensitivity(logits.symmetric_bond_logits[0, ids][:, ids][upper], temperature))
        return record, gradient.detach()

    def values(self, z, temperature):
        with torch.no_grad():
            _, features, graph = self.decode(z, temperature)
            mapped, _ = self.native(self.bridge(features, self.indices))
            relaxed, _ = self.native.score_graph(graph)
            hard = hard_from_graph(graph)
            hard_loss, _ = self.native.score_graph(hard.to_relaxed(self.decoder.spec))
        inspected = inspect_graph(*(getattr(hard, k)[0] for k in ('atoms', 'charges', 'bonds')), self.decoder.spec)
        return dict(mapped_loss=float(mapped), relaxed_loss=float(relaxed), hard_fixed_context_loss=float(hard_loss),
                    valid=inspected.smiles is not None, smiles=inspected.smiles, failure=inspected.failure_reason), hard


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mosaic-python', required=True)
    p.add_argument('--plan'); p.add_argument('--plan-sha256')
    a = p.parse_args(argv)
    torch.set_num_threads(4)
    config = json.loads(a.config.read_text())
    decoder = load_frozen_decoder(config['checkpoint'])
    decoder_before = _model_identity(decoder)
    if file_sha256(Path(config['mapping'])) != config['mapping_sha256']:
        raise ValueError('Trained mapping checkpoint changed')
    fitted = torch.load(config['mapping'], map_location='cpu', weights_only=True)
    bridge = HiddenChemicalBridge(fitted['bank'], fitted['dimensions'], fitted['hidden'])
    bridge.load_state_dict(fitted['state_dict']); bridge.eval().requires_grad_(False)
    bridge_before = _model_identity(bridge)
    active = torch.zeros(decoder.spec.max_nodes, dtype=torch.bool)
    active[config['active_slots']] = True
    a.output.mkdir(parents=True, exist_ok=False)
    native = MosaicProtenixEmbeddingScorer(decoder.spec, active,
        command=[a.mosaic_python, '-m', 'latentmol.guidance.mosaic_protenix_embedding_worker'],
        config=json.loads(Path(config['native_config']).read_text()), stderr_path=a.output/'native.stderr.log',
        timeout=900, startup_timeout=900)
    if native.bank != fitted['bank']:
        native.close(); raise ValueError('Trained mapping and native chemical bank differ')
    probe = Probe(decoder, bridge, native, active)
    design_config = LatentDesignConfig.from_mapping(config['design'])
    points = load_points(config, decoder)
    report = dict(question='Does the learned projection improve useful latent gradient flow?',
        config=config, native_identity=native.identity, fixed_step_radius=config['step_radius'],
        scope='Matched same-device/context/key/latent; binding-only unit-gradient directions; one fixed radius, no step-size search',
        limitations=['Single seed/model/context; local probes do not establish trajectory superiority',
                     'Hard invalid graphs can be scored but are not valid candidate improvements',
                     'Affine span is weaker than convex-hull chemical feasibility',
                     'Native contact is a surrogate, not affinity or binding confirmation'], points=[])
    started = time.monotonic()
    try:
        for name, step, point in points:
            temperature = schedules(design_config, step)['temperature']
            mapped, gm = probe.gradient(point, temperature, 'mapped')
            relaxed, gr = probe.gradient(point, temperature, 'relaxed')
            exact, ge = probe.gradient(point, temperature, 'exact_projection', details=False)
            baseline, baseline_hard = probe.values(point, temperature)
            row = dict(name=name, source_step=step, temperature=temperature, baseline=baseline,
                       mapped=mapped, relaxed=relaxed, latent_gradient_cosine=cosine(gm, gr), directions={})
            row['exact_projection_control'] = dict(exact, relaxed_loss_difference=exact['loss']-relaxed['loss'],
                gradient_cosine=cosine(ge, gr), relative_gradient_difference=norm(ge-gr)/max(norm(gr),1e-30),
                scope='Same probabilities projected algebraically; includes arcsinh of expected charge')
            row['runtime_seconds'] = {}
            for path in ('mapped', 'relaxed'):
                samples = []
                for _ in range(config['runtime_repeats']):
                    tick = time.monotonic(); probe.gradient(point, temperature, path, details=False)
                    samples.append(time.monotonic()-tick)
                row['runtime_seconds'][path] = samples
            random = torch.randn(point.shape, generator=torch.Generator().manual_seed(7017))
            for direction_name, gradient in [('mapped', gm), ('relaxed', gr), ('random_control', -random)]:
                magnitude = norm(gradient)
                if not magnitude:
                    row['directions'][direction_name] = dict(status='flat_direction', displacement=0.)
                    continue
                direction = -gradient/magnitude
                candidate = point + config['step_radius']*direction
                values, hard = probe.values(candidate, temperature)
                left, right = torch.triu_indices(decoder.spec.max_nodes, decoder.spec.max_nodes, offset=1)
                changes = dict(atoms=int((hard.atoms != baseline_hard.atoms).sum()),
                    charges=int((hard.charges != baseline_hard.charges).sum()),
                    unordered_bonds=int((hard.bonds[:, left, right] != baseline_hard.bonds[:, left, right]).sum()))
                row['directions'][direction_name] = dict(values, displacement=norm(candidate-point),
                    predicted_relaxed_loss_delta=float((gr*direction).sum())*config['step_radius'],
                    relaxed_loss_delta=values['relaxed_loss']-baseline['relaxed_loss'],
                    mapped_loss_delta=values['mapped_loss']-baseline['mapped_loss'],
                    hard_loss_delta=values['hard_fixed_context_loss']-baseline['hard_fixed_context_loss'], changes=changes)
            report['points'].append(row)
            _json(a.output/'progress.json', report)
            print(json.dumps(dict(point=name, gradient_cosine=row['latent_gradient_cosine'],
                                  mapped_gradient_norm=mapped['gradient_norm'], relaxed_gradient_norm=relaxed['gradient_norm'])), flush=True)
        if _model_identity(decoder) != decoder_before or _model_identity(bridge) != bridge_before:
            raise ValueError('Frozen VAE or mapping changed during comparison')
        native.close()
        report.update(status='completed', elapsed_seconds=time.monotonic()-started, frozen_models_unchanged=True)
        _json(a.output/'result.json', report)
    finally:
        native.close()


if __name__ == '__main__':
    main()
