"""Shared VAE latent loop with the existing frozen DBMol Boltz objective."""
import argparse
import json
from pathlib import Path

import torch

from latentmol.boltz.contracts import PreparedComplex
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, run_design
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import HardGraphBatch, RelaxedGraph


def rebind_slots(prepared, spec, active):
    """Reindex a fixed native context to decoder slots without dropping categories.

    Native atom/token order and every prepared feature remain unchanged. The
    prepared seed is translated by actual element/charge/bond identity, then
    existing seed-feature parity validates the derived view.
    """
    source = prepared.active.nonzero().flatten()
    target = active.nonzero().flatten().to(source.device)
    if len(source) != len(target) or active.shape != (spec.max_nodes,):
        raise ValueError('Prepared and decoder occupancy counts differ')
    atom_map = torch.tensor([spec.atomic_numbers.index(v) for v in prepared.spec.atomic_numbers], device=source.device)
    charge_map = torch.tensor([spec.formal_charges.index(v) for v in prepared.spec.formal_charges], device=source.device)
    bond_map = torch.tensor([spec.bond_names.index(v) for v in prepared.spec.bond_names], device=source.device)
    atoms = torch.zeros((1, spec.max_nodes), dtype=torch.long, device=source.device)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.full((1, spec.max_nodes, spec.max_nodes), spec.bond_names.index('NONE'), dtype=torch.long, device=source.device)
    atoms[0, target] = atom_map[prepared.seed.atoms[0, source]]
    charges[0, target] = charge_map[prepared.seed.charges[0, source]]
    bonds[0, target[:, None], target[None, :]] = bond_map[prepared.seed.bonds[0, source[:, None], source[None, :]]]
    ai = torch.full((spec.max_nodes,), -1, dtype=torch.long, device=source.device)
    ti = ai.clone()
    ai[target], ti[target] = prepared.slot_to_atom[source], prepared.slot_to_token[source]
    view = PreparedComplex(prepared.features, prepared.coordinates, ai, ti,
                           HardGraphBatch(atoms, charges, bonds), spec, dict(prepared.metadata))
    view.validate()
    return view


class BoltzLatentScorer:
    label = 'VAE + DBMol composite latent design'
    claim_limit = 'Exploratory fixed-pose DBMol composite optimization; hard-molecule binding not established'
    gradient_scope = 'shared decoder atom/charge/bond probabilities -> frozen affinity and structure/confidence trunks plus detached repair CE -> latent only'

    def __init__(self, spec, active, native, *, device='cuda'):
        from latentmol.boltz.model import load_affinity_model
        from latentmol.boltz.dbmol_model import FrozenBoltzDBMol, load_confidence_model
        from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
        from latentmol.boltz.dbmol_prepare import initial_pose_pocket
        from latentmol.boltz.provenance import context_digest
        from latentmol.boltz.scoring import BoltzGraphBranch, boltz_branch_manifest
        for key in ('prepared', 'affinity', 'confidence'):
            if file_sha256(Path(native[key]['path'])) != native[key]['sha256']:
                raise ValueError('Native input changed: '+key)
        original = PreparedComplex.load(Path(native['prepared']['path']), device)
        self.prepared = rebind_slots(original, spec, active)
        pocket = initial_pose_pocket(self.prepared)
        profile = DBMolObjectiveConfig.from_mapping(dict(
            confidence_checkpoint=native['confidence'], pocket_residues=pocket['pocket_residues'],
            ipsae_weight=native.get('ipsae_weight', 0.0),
            pocket_origin='initial_pose_heavy_atom_8A', pocket_reference=dict(
                prepared_sha256=native['prepared']['sha256'], context_sha256=context_digest(self.prepared))), Path('/'))
        profile.verify_reference(self.prepared, native['prepared']['sha256'])
        self.scorer = FrozenBoltzDBMol(load_affinity_model(Path(native['affinity']['path'])),
            load_confidence_model(profile.confidence_checkpoint), profile, recycling_steps=0).to(device).eval()
        manifest = boltz_branch_manifest(self.prepared, adapter_source_sha256=native['source_sha256'],
                                         recycling_steps=0, dbmol_profile=profile.record())
        self.branch = BoltzGraphBranch(self.scorer, self.prepared, manifest, allow_lossy_charge=True)
        self.before = _model_identity(self.scorer)
        self.device = device
        self.identity = dict(protocol='vae-dbmol-shared-graph-v1', prepared=native['prepared'],
            context_digest=context_digest(self.prepared), model_digest=self.before,
            dbmol_profile=profile.record(), graph_spec=spec.to_dict(),
            slot_to_atom=self.prepared.slot_to_atom.cpu().tolist(), slot_to_token=self.prepared.slot_to_token.cpu().tolist(),
            limitations=['Fixed prepared pose; no diffusion or coordinate gradient.',
                         'Native charge feature consumes the expected formal charge.',
                         'All decoder element and typed-bond categories are preserved.'])

    def __call__(self, graph):
        # Device copies retain autograd back to the CPU decoder and latent.
        shared = RelaxedGraph(*(getattr(graph, k).to(self.device) for k in ('atoms','charges','bonds')))
        values = self.branch(shared)
        loss = values['dbmol_total_loss']
        if not loss.requires_grad:
            raise ValueError('Detached required DBMol total')
        aux = {k: float(v.detach()) for k,v in values.items()
               if isinstance(v, torch.Tensor) and v.numel()==1}
        return loss.to(graph.atoms.device), aux

    def close(self):
        self.branch.verify_context_unchanged()
        if _model_identity(self.scorer) != self.before or any(p.grad is not None for p in self.scorer.parameters()):
            raise ValueError('Frozen Boltz weights/parameter gradients changed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','native-config','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-updates', type=int)
    p.add_argument('--wall-seconds', type=float)
    p.add_argument('--threads', type=int, default=4)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    if file_sha256(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError('Frozen decoder checkpoint changed')
    config=LatentDesignConfig.from_mapping(json.loads(a.config.read_text()))
    native=json.loads(a.native_config.read_text())
    decoder,_=load_model(a.checkpoint,'cpu')
    return run_design(decoder,lambda spec,active:BoltzLatentScorer(spec,active,native), config,a.output,
        checkpoint_sha256=a.checkpoint_sha256,resume=a.resume,max_updates=a.max_updates,
        wall_seconds=a.wall_seconds,native_metadata=dict(config_path=str(a.native_config),
            config_sha256=file_sha256(a.native_config),objective='dbmol_fixed_pose_v1'))


if __name__=='__main__':
    main()
