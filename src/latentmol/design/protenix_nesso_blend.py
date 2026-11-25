"""Frozen Protenix/contact and Nesso/affinity VJPs on one epoch-48 founder lineage."""
from importlib import import_module
import math
from pathlib import Path
from statistics import median

import torch

from latentmol.design.latent_protenix import prepare_point
from latentmol.design.protenix_encoded_founder import replay_encoded_founder
from latentmol.design.protenix_restarts import write_json
from latentmol.guidance.mosaic_protenix import _NativeVJP
from latentmol.representation import GraphSpec


def schema_founder_record(model, record, design, checkpoint_sha256):
    candidate, active = replay_encoded_founder(model, record, design, checkpoint_sha256)
    hard = candidate.hard
    spec = GraphSpec(max_nodes=25)
    if (active.sum().item() != spec.max_nodes or model.spec.atomic_numbers != spec.atomic_numbers
            or model.spec.formal_charges != spec.formal_charges or model.spec.bond_names != spec.bond_names):
        raise ValueError("Nesso requires a fixed 25-atom graph in the shared vocabulary")
    return dict(spec=spec.to_dict(), smiles=candidate.smiles,
                atoms=hard.atoms[0, active].tolist(), charges=hard.charges[0, active].tolist(),
                bonds=hard.bonds[0, active][:, active].tolist())


def nesso_value(worker, graph, active):
    """Gather occupied rows once; the upper triangle's native VJP includes both orientations."""
    active = active.to(graph.atoms.device)
    atoms = graph.atoms[0, active, 1:]
    charges = graph.charges[0, active]
    matrix = graph.bonds[0, active][:, active]
    i, j = torch.triu_indices(atoms.shape[0], atoms.shape[0], 1, device=matrix.device)
    rows = (atoms, charges, matrix[i, j])
    value, derivatives = worker.schema_gradient(*(r.detach().cpu().numpy() for r in rows))
    grads = tuple(torch.as_tensor(d, dtype=r.dtype, device=r.device) for r, d in zip(rows, derivatives, strict=True))
    return _NativeVJP.apply(*rows, value, grads)


def fit_latent_weights(pairs, affinity_share):
    """Match median latent-gradient contributions and the original contact update scale."""
    measured = [(p.double(), n.double()) for p, n in pairs]
    if not measured or not 0 < affinity_share < 1:
        raise ValueError("Expected latent gradient probes and a share strictly between zero and one")
    for p, n in measured:
        if p.shape != n.shape or not p.isfinite().all() or not n.isfinite().all():
            raise ValueError("Latent gradient probes must be finite and match in shape")
    usable = [(p, n) for p, n in measured if p.norm() > 0 and n.norm() > 0]
    if usable:
        ratio = median(float(p.norm() / n.norm()) for p, n in usable) * affinity_share / (1 - affinity_share)
        length = median(float((p + ratio*n).norm() / p.norm()) for p, n in usable)
        wp = 1 / max(length, .1)
        wn = wp * ratio
        status = "calibrated" if length >= .1 else "cancellation_limited"
    else:
        wp, wn = 1.0, affinity_share / (1 - affinity_share)
        status = "no_nonzero_pairs"
    if not math.isfinite(wp) or not math.isfinite(wn):
        raise ValueError("Nonfinite model coefficients")
    measurements = []
    for p, n in measured:
        pnorm, nnorm = float(p.norm()), float(n.norm())
        measurements.append(dict(contact_norm=pnorm, affinity_norm=nnorm,
            weighted_affinity_to_contact=(wn*nnorm)/(wp*pnorm) if pnorm else None,
            mixed_to_contact=float((wp*p + wn*n).norm())/pnorm if pnorm else None))
    return dict(protenix_weight=wp, nesso_weight=wn, status=status, probes=measurements)


class BlendScorer:
    def __init__(self, protenix, session):
        self.protenix = protenix
        self.session = session
        self.identity = dict(protenix=protenix.identity, nesso=session.worker.identity,
                             calibration=session.weights)

    def score_at(self, graph, active):
        if not torch.equal(active.cpu(), self.session.active):
            raise ValueError("Nesso fixed-founder slot assignment changed")
        contact, aux = self.protenix.score_at(graph, active)
        affinity = nesso_value(self.session.worker, graph, active)
        w = self.session.weights
        total = w["protenix_weight"]*contact + w["nesso_weight"]*affinity
        return total, dict(aux, model_blend=dict(protenix_contact=float(contact.detach()),
                                                  nesso_affinity=float(affinity.detach())))

    def close(self):
        return self.protenix.close()


class NessoBlendSession:
    """One native fixed-crop Nesso process and one coefficient fit for all search waves."""
    def __init__(self, config, runtime, model, design, founder, factory, stack):
        self.config, self.runtime = config, runtime
        self.model, self.design, self.founder = model, design, founder
        self.factory, self.stack = factory, stack
        self.worker = self.weights = None
        self.hard_founder = None
        self.active = torch.zeros(model.spec.max_nodes, dtype=torch.bool)
        self.active[founder["active_slots"]] = True

    def scorer(self, spec, active, root):
        if not torch.equal(self.active, active.cpu()):
            raise ValueError("Blend search requires a shared founder mapping")
        if self.worker is None:
            native_worker = import_module("experiments.latent-design.direct-jump.combined_native").NessoWorker
            record = schema_founder_record(self.model, self.founder, self.design, self.runtime.checkpoint_sha256)
            write_json(self.runtime.output / "nesso-schema-founder.json", record)
            self.worker = self.stack.enter_context(native_worker(self.config["seed"], None,
                self.runtime.output / "nesso-worker", Path(self.config["assets_dir"]), schema_founder=record))
            from latentmol.design.protenix_nesso_readout import _native
            baseline = _native(self.worker.fresh, record["smiles"])
            write_json(self.runtime.output / "nesso-hard-founder.json", dict(
                smiles=record["smiles"], seed=self.config["seed"], native=baseline,
                model_sha256=self.worker.identity["model_sha256"]))
            if baseline["status"] != "scored":
                raise ValueError(f"Native Nesso founder preflight failed: {baseline}")
            self.hard_founder = (record["smiles"], baseline)
        protenix = self.factory(spec, active, root)
        try:
            if self.weights is None:
                z = torch.tensor(self.founder["latent"], dtype=torch.float32)
                rng = torch.Generator().manual_seed(self.config["seed"])
                direction = torch.randn(z.shape, generator=rng)
                direction /= direction.norm()
                pairs = []
                for radius in (0., -.1, .1):
                    point = prepare_point(self.model, z + radius*direction, active, self.design, 0)
                    contact, _ = protenix.score_at(point.graph, active)
                    affinity = nesso_value(self.worker, point.graph, active)
                    pg, = torch.autograd.grad(contact, point.point, retain_graph=True)
                    ng, = torch.autograd.grad(affinity, point.point)
                    pairs.append((pg, ng))
                self.weights = fit_latent_weights(pairs, self.config["affinity_share"])
                write_json(self.runtime.output / "latent-blend-calibration.json",
                           dict(**self.weights, radii=[0., -.1, .1], seed=self.config["seed"],
                                geometry="global L2 of full decoder-to-latent VJPs before repair and clipping",
                                nesso=self.worker.identity, protenix=protenix.identity))
            return BlendScorer(protenix, self)
        except BaseException:
            protenix.close()
            raise

    def fresh(self, smiles):
        if self.hard_founder is not None and smiles == self.hard_founder[0]:
            return self.hard_founder[1].copy()
        return self.worker.fresh(smiles)
