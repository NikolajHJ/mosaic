"""Small frozen-decoder hidden-to-Protenix chemical projection prototype."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.design.latent_protenix import (LatentDesignConfig, choose_active, run_design,
    _model_identity, _json)
from latentmol.representation import RelaxedGraph
from latentmol.graph_policy import graph_from_decoder


class HiddenChemicalBridge(nn.Module):
    def __init__(self, bank, dimensions, hidden=128):
        super().__init__()
        self.bank = bank
        self.dimensions = dict(dimensions)
        self.hidden = hidden
        self.trunks = nn.ModuleDict({k: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU())
                                    for k, d in dimensions.items()})
        self.heads = nn.ModuleDict({k: nn.Linear(hidden, int(v['width'])) for k, v in bank.items()})

    def forward(self, features, indices):
        # indices is [batch, occupied], sorted to retain the native slot order.
        batch = torch.arange(indices.shape[0], device=indices.device)[:, None]
        atom = features.atom_hidden[batch, indices]
        charge = features.charge_input[batch, indices]
        pair = features.bond_input[batch[:, :, None], indices[:, :, None], indices[:, None, :]]
        pair = .5 * (pair + pair.transpose(1, 2))
        values = {'atom_element': atom, 'atom_charge': charge, 'pair_presence': pair}
        result = {key: self.heads[key](self.trunks[key](value)) for key, value in values.items()}
        diagonal = torch.eye(indices.shape[1], dtype=torch.bool, device=indices.device)[None, :, :, None]
        result['pair_presence'] = result['pair_presence'].masked_fill(diagonal, 0)
        if any(not bool(v.isfinite().all()) for v in result.values()):
            raise ValueError('Nonfinite learned chemical projection')
        return result


def selected_indices(decoded, count):
    occupancy = 1 - decoded.atoms.softmax(-1)[..., 0]
    return occupancy.topk(count, dim=-1).indices.sort(-1).values


def projection_labels(decoded, indices):
    batch = torch.arange(indices.shape[0], device=indices.device)[:, None]
    atoms = decoded.atoms[batch, indices, 1:].argmax(-1)
    charges = decoded.charges[batch, indices].argmax(-1)
    bonds = decoded.bonds[batch[:, :, None], indices[:, :, None], indices[:, None, :]].argmax(-1)
    return {'atom_element': atoms, 'atom_charge': charges, 'pair_presence': (bonds != 0).long()}


def labels_and_features(decoder, latents, count):
    with torch.no_grad():
        decoded, features = decoder.decode_with_features(latents)
        indices = selected_indices(decoded, count)
        return features, indices, projection_labels(decoded, indices)


def projection_loss(predictions, labels, bank):
    losses = {}
    for key, predicted in predictions.items():
        prototypes = torch.tensor(bank[key]['prototypes'], dtype=predicted.dtype, device=predicted.device)
        target = prototypes[labels[key]]
        error = (predicted - target).square().mean(-1)
        if key == 'pair_presence':
            diagonal = torch.eye(error.shape[-1], dtype=torch.bool, device=error.device)[None]
            parts = [error[(labels[key] == label) & ~diagonal].mean()
                     for label in (0, 1) if bool(((labels[key] == label) & ~diagonal).any())]
            loss = torch.stack(parts).mean()
        else:
            loss = error.mean()
        losses[key] = loss
    return torch.stack(list(losses.values())).mean(), losses


def fit_mapping(decoder, bank, config, output, count):
    torch.manual_seed(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    latents = torch.randn(config['train_samples'] + config['dev_samples'], decoder.config.latent, generator=generator)
    training, development = latents[:config['train_samples']], latents[config['train_samples']:]
    features, _, _ = labels_and_features(decoder, training[:1], count)
    dimensions = dict(atom_element=features.atom_hidden.shape[-1], atom_charge=features.charge_input.shape[-1],
                      pair_presence=features.bond_input.shape[-1])
    bridge = HiddenChemicalBridge(bank, dimensions, config['hidden'])
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config['learning_rate'])

    def validate():
        rows = []
        with torch.no_grad():
            for batch in development.split(config['batch_size']):
                f, indices, labels = labels_and_features(decoder, batch, count)
                _, terms = projection_loss(bridge(f, indices), labels, bank)
                rows.append({k: float(v) for k, v in terms.items()})
        return {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}

    initial = validate()
    best_loss, best_weights, best_step = float('inf'), None, None
    started = time.monotonic()
    with (output / 'fit.jsonl').open('x') as stream:
        for step in range(1, config['updates'] + 1):
            chosen = torch.randint(len(training), (config['batch_size'],), generator=generator)
            f, indices, labels = labels_and_features(decoder, training[chosen], count)
            loss, _ = projection_loss(bridge(f, indices), labels, bank)
            if not bool(loss.isfinite()):
                raise ValueError('Nonfinite mapping loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in bridge.parameters()):
                raise ValueError('Missing/nonfinite mapping parameter gradient')
            optimizer.step()
            if step == 1 or step % 200 == 0 or step == config['updates']:
                dev = validate()
                value = sum(dev.values())/len(dev)
                if value < best_loss:
                    best_loss, best_step = value, step
                    best_weights = {k: v.detach().clone() for k, v in bridge.state_dict().items()}
                row = dict(step=step, train_loss=float(loss), dev=dev, elapsed=time.monotonic()-started)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                print(json.dumps(dict(stage='fit', **row)), flush=True)
    bridge.load_state_dict(best_weights)
    bridge.eval().requires_grad_(False)
    asset = dict(state_dict=bridge.state_dict(), bank=bank, dimensions=dimensions, hidden=config['hidden'],
                 fit_config=config, initial_dev=initial, final_dev=validate(), best_step=best_step)
    torch.save(asset, output / 'mapping.pt')
    _json(output / 'fit-result.json', {k: v for k, v in asset.items() if k not in ('state_dict', 'bank')})
    return bridge


class LearnedProjectionScorer:
    uses_decoder_hidden = True
    gradient_scope = 'frozen epoch45 decoder hidden -> frozen learned chemical projections -> frozen Protenix contact; graph logits -> detached structural target NLL'

    def __init__(self, native, bridge, active, observe_every=10):
        self.native, self.bridge = native, bridge
        self.indices = torch.nonzero(active).flatten()[None]
        self.identity = dict(native.identity, mapping_sha256=_model_identity(bridge), input_boundary='learned_chemical_projection')
        self.observe_every = observe_every
        self.before = _model_identity(bridge)

    def score_hidden(self, features, graph, active, *, step):
        if not torch.equal(torch.nonzero(active).flatten()[None].cpu(), self.indices.cpu()):
            raise ValueError('Selected ligand slots changed')
        predictions = self.bridge(features, self.indices)
        loss, aux = self.native(predictions)
        aux = dict(aux, input_boundary='learned_chemical_projection')
        if step % self.observe_every == 0:
            reference = self.native.evaluate_graph(graph)
            aux['relaxed_graph_contact_loss'] = float(reference['native_loss'])
            aux['learned_minus_relaxed_loss'] = float(loss.detach()) - float(reference['native_loss'])
        return loss, aux

    def close(self):
        if _model_identity(self.bridge) != self.before or any(p.grad is not None for p in self.bridge.parameters()):
            raise ValueError('Frozen mapping changed during latent optimization')
        self.native.close()


def load_frozen_decoder(asset):
    checkpoint_path = Path(asset['path'])
    if file_sha256(checkpoint_path) != asset['sha256']:
        raise ValueError('Frozen epoch45 checkpoint SHA differs')
    decoder, checkpoint = load_model(checkpoint_path, 'cpu')
    if checkpoint['epoch'] != 45:
        raise ValueError('Expected the completed epoch45 VAE')
    return decoder.eval().requires_grad_(False)


def load_start_state(config, decoder):
    """Start a new arm at recorded latent bytes, without rewriting old receipts."""
    source=config.get('start')
    if source is None:
        return None
    path=Path(source['raw_block'])
    if source.get('kind')!='saved_trajectory_latent' or file_sha256(path)!=source['raw_block_sha256']:
        raise ValueError('Saved starting latent block changed')
    with np.load(path,allow_pickle=False) as block:
        matches=np.flatnonzero(block['steps']==source['step'])
        if len(matches)!=1:
            raise ValueError('Starting latent step missing or ambiguous')
        point=torch.as_tensor(block['latent'][int(matches[0])],dtype=torch.float32)[None].clone()
    active=torch.zeros(decoder.spec.max_nodes,dtype=torch.bool)
    slots=source['active_slots']
    if (not slots or len(set(slots))!=len(slots)
            or any(type(v) is not int or not 0<=v<decoder.spec.max_nodes for v in slots)):
        raise ValueError('Saved starting slots incompatible with decoder')
    active[slots]=True
    return dict(latent=point,active=active,step=source['step'],provenance=source)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--wall-seconds', type=float, default=78000)
    parser.add_argument('--mosaic-python', required=True)
    parser.add_argument('--plan'); parser.add_argument('--plan-sha256')
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    source = config['checkpoint']
    decoder = load_frozen_decoder(source)
    decoder_before = _model_identity(decoder)
    args.output.mkdir(parents=True, exist_ok=False)
    native_config = json.loads(Path(config['native_config']).read_text())
    design = LatentDesignConfig.from_mapping(config['design'])
    start_state=load_start_state(config,decoder) if args.resume is None else None
    if args.resume is not None:
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        z,active=saved['latent'],saved['active']
    elif start_state is not None:
        z,active=start_state['latent'],start_state['active']
    else:
        z = torch.randn((1, decoder.config.latent), generator=torch.Generator().manual_seed(design.seed))
        with torch.no_grad():
            active = choose_active(decoder.decode(z), decoder.spec, design.occupied_atoms)
    from latentmol.guidance.mosaic_protenix_embedding import MosaicProtenixEmbeddingScorer
    native = MosaicProtenixEmbeddingScorer(decoder.spec, active,
        command=[args.mosaic_python, '-m', 'latentmol.guidance.mosaic_protenix_embedding_worker'],
        config=native_config, stderr_path=args.output/'native.stderr.log', timeout=900, startup_timeout=900)
    closed = False
    try:
        with torch.no_grad():
            graph = graph_from_decoder(decoder.decode(z), decoder.spec,
                temperature=design.temperature_initial, fixed_active=active)
            hard = RelaxedGraph(*(F.one_hot(getattr(graph, key).argmax(-1), getattr(graph, key).shape[-1]).float()
                                  for key in ('atoms', 'charges', 'bonds')))
        smoke = native.evaluate_graph(hard)
        _json(args.output/'native-projection-smoke.json', smoke)
        if smoke['absolute_difference'] > 1e-5 + 1e-4 * abs(smoke['native_loss']):
            raise ValueError('Native hard graph and chemical projection injection disagree')
        if args.mapping:
            declared=config.get('mapping_checkpoint')
            if declared and file_sha256(args.mapping)!=declared['sha256']:
                raise ValueError('Frozen trained mapping checkpoint differs')
            fitted = torch.load(args.mapping, map_location='cpu', weights_only=True)
            if fitted['bank'] != native.bank:
                raise ValueError('Mapping/native projection bank differs')
            bridge = HiddenChemicalBridge(native.bank, fitted['dimensions'], fitted['hidden'])
            bridge.load_state_dict(fitted['state_dict'])
            bridge.eval().requires_grad_(False)
        else:
            bridge = fit_mapping(decoder, native.bank, config['mapping'], args.output, design.occupied_atoms)
        # Parameter gradients from supervised fitting must not enter the frozen phase.
        for p in bridge.parameters():
            p.grad = None
        if _model_identity(decoder) != decoder_before:
            raise ValueError('Frozen VAE changed during mapping fit')
        holder = LearnedProjectionScorer(native, bridge, active)
        def factory(spec, expected_active):
            if not torch.equal(expected_active, active):
                raise ValueError('Latent design selected different fixed slots')
            return holder
        result = run_design(decoder, factory, design, args.output/'design', checkpoint_sha256=source['sha256'],
            resume=args.resume,start_state=start_state,wall_seconds=args.wall_seconds,
            native_metadata=dict(input_boundary='learned_chemical_projection',
                mapping_file=str(args.mapping or args.output/'mapping.pt'),
                checkpoint_epoch=45, objective='Protenix contact surrogate; no affinity claim'))
        closed = True
        _json(args.output/'result.json', dict(status=result['status'], design=result, checkpoint_sha256=source['sha256'],
            mapping_sha256=file_sha256(args.mapping or args.output/'mapping.pt')))
        return result
    finally:
        if not closed:
            native.close()


if __name__ == '__main__':
    main()
