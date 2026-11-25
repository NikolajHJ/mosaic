"""Opt-in atom/charge repair and fixed-reference structure preservation.

Search and RDKit stay outside autograd. Costs are preferences, not chemistry
rules. References never refresh implicitly; occupancy and slot alignment stay
fixed. Bounded beam witnesses are not certified nearest valid molecules.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
from rdkit import Chem

from latentmol.chemistry import _label_molecule, canonical_slot_order, encode_smiles, inspect_graph
from latentmol.evaluation.quality import diagnose_graph
from latentmol.probability_repair import ProbabilityRepairConfig, measure_probability_edits, validate_probabilities
from latentmol.repair_search import round_robin_candidates, single_label_edits
from latentmol.repair_guidance.loss import FrozenRepairTarget, fixed_target_nll
from latentmol.representation import HardGraphBatch


@dataclass(frozen=True)
class PreservationConfig:
    """Continuous reference penalties only; never modifies repair discovery."""
    atom_extra_cost: float = 0.
    charge_extra_cost: float = 0.
    substructure_extra_cost: float = 0.
    anchor_scale: float = .05

    def __post_init__(self):
        for name in ('atom_extra_cost','charge_extra_cost','substructure_extra_cost','anchor_scale'):
            value=getattr(self,name)
            if isinstance(value,bool) or not math.isfinite(value) or value<0:
                raise ValueError('Require finite nonnegative '+name)


@dataclass(frozen=True)
class ExchangeConfig:
    atom_extra_cost: float = 2.
    charge_extra_cost: float = 1.
    substructure_extra_cost: float = 2.
    anchor_scale: float = .05
    max_edits: int = 3
    checks_per_depth: int = 64
    beam_width: int = 4
    probability_weight: float = 1.
    edit_cost: str = 'log_regret'

    def __post_init__(self):
        if self.edit_cost not in ('log_regret', 'one_minus_probability'):
            raise ValueError('Unknown edit_cost')
        if self.edit_cost == 'one_minus_probability' and self.probability_weight != 1:
            raise ValueError('one_minus_probability uses exactly 1-p; probability_weight must be 1')
        for name in ('atom_extra_cost', 'charge_extra_cost', 'substructure_extra_cost', 'anchor_scale', 'probability_weight'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError('Require finite nonnegative ' + name)
        ProbabilityRepairConfig(max_edits=self.max_edits, checks_per_depth=self.checks_per_depth,
                                beam_width=self.beam_width, probability_weight=self.probability_weight)


@dataclass(frozen=True)
class StructureReference:
    target: FrozenRepairTarget
    atom_mask: np.ndarray
    pair_mask: np.ndarray
    motifs: tuple
    spec_fingerprint: str

    @classmethod
    def create(cls, raw, spec):
        raw = tuple(np.asarray(x).copy() for x in raw)
        if len(raw) != 3 or any(x.dtype.kind not in 'iu' for x in raw):
            raise ValueError('Reference labels must be integers')
        raw = tuple(x.astype(np.int64) for x in raw)
        # Reject graphs requiring aromatic normalization: the penalty compares
        # canonical aromatic labels, not alternative Kekule spellings.
        canonical_slot_order(*raw, spec)
        active = np.flatnonzero(raw[0])
        mol = _label_molecule(raw[0], raw[1], raw[2][np.ix_(active, active)], spec, active)
        Chem.SanitizeMol(mol)
        groups = [('ring',group) for group in mol.GetRingInfo().AtomRings()]
        for name,smarts in (('amide','[NX3][CX3](=[OX1])'),
                            ('carboxyl','[CX3](=[OX1])[OX2]'), ('sulfonyl','[SX4](=[OX1])(=[OX1])')):
            groups.extend((name,group) for group in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)))
        atom_mask = np.zeros(spec.max_nodes, dtype=bool)
        pair_mask = np.zeros((spec.max_nodes, spec.max_nodes), dtype=bool)
        for _,group in groups:
            slots = active[list(group)]
            atom_mask[slots] = True
            # Protect absence of chords as well as existing internal bonds.
            pair_mask[np.ix_(slots, slots)] = True
        np.fill_diagonal(pair_mask, False)
        atom_mask.setflags(write=False); pair_mask.setflags(write=False)
        target = FrozenRepairTarget.create(HardGraphBatch(*(torch.tensor(x)[None] for x in raw)),
            torch.tensor(raw[0] != 0), refresh_step=0, receipt_path='fixed_valid_reference')
        motifs = tuple((name,tuple(int(active[i]) for i in group)) for name,group in groups)
        return cls(target, atom_mask, pair_mask, motifs, spec.fingerprint())

    def verify_spec(self, spec):
        if self.spec_fingerprint != spec.fingerprint():
            raise ValueError('Reference graph schema differs')

    def labels(self):
        self.target.verify(self.target.active)
        return tuple(getattr(self.target.graph, k)[0].numpy().copy() for k in ('atoms', 'charges', 'bonds'))


def structure_distance(candidate, reference):
    labels = reference.labels()
    return int(np.count_nonzero((candidate[0] != labels[0]) & reference.atom_mask)
        + np.count_nonzero((candidate[1] != labels[1]) & reference.atom_mask)
        + np.count_nonzero(np.triu((candidate[2] != labels[2]) & reference.pair_mask, 1)))


def _edit_base_cost(old_probability, new_probability, config):
    if config.edit_cost == 'one_minus_probability':
        return 1 - new_probability
    regret = max(0., math.log(max(old_probability, 1e-12)) - math.log(max(new_probability, 1e-12)))
    return 1 + config.probability_weight * regret


def exchange_cost(raw, candidate, probabilities, spec, config, reference):
    reference.verify_spec(spec)
    base = measure_probability_edits(raw, candidate, probabilities, spec,
                                    ProbabilityRepairConfig(probability_weight=config.probability_weight))
    motif = structure_distance(candidate, reference)
    changes = []
    for change in base['changes']:
        cost = _edit_base_cost(change['old_probability'],change['new_probability'],config)
        extra = config.atom_extra_cost if change['field']=='atom' else config.charge_extra_cost if change['field']=='charge' else 0.
        changes.append(dict(change, base_cost=cost, extra_cost=extra, weighted_cost=cost+extra))
    base_cost = sum(change['base_cost'] for change in changes)
    total = (base_cost + config.atom_extra_cost * base['edit_counts']['atom']
             + config.charge_extra_cost * base['edit_counts']['charge']
             + config.substructure_extra_cost * motif)
    return dict(base, changes=changes, edit_cost=config.edit_cost, legacy_log_regret_cost=base['weighted_cost'],
                base_cost=base_cost, weighted_cost=total, substructure_changes=motif)


def _mixed_neighbors(graph, original, spec, diagnostic, rank):
    # Reserve opportunities for each edit family even under a tiny budget.
    # Otherwise the legacy bond-first order can exhaust all checks before an
    # atom/charge proposal is ever visited, especially at high exchange costs.
    def family(field):
        for candidate in single_label_edits(graph, original, spec, diagnostic,
                operation_rank=lambda op: (op[1] != field, *rank(op))):
            if np.array_equal(candidate[field], graph[field]):
                break
            yield candidate
    return round_robin_candidates([family(2), family(0), family(1)])


def search_exchange(raw, probabilities, spec, config, reference):
    """Cost-ranked mixed edits, with exact whole-graph acceptance.

    Unlike the legacy unit-depth search, valid candidates remain expandable:
    a second edit can restore a protected motif and reduce total cost.
    """
    reference.verify_spec(spec)
    p = validate_probabilities(probabilities, spec)
    raw = tuple(np.array(x, copy=True) for x in raw)
    measure_probability_edits(raw, raw, p, spec)
    if not np.array_equal(raw[0] != 0, reference.target.active.numpy()):
        raise ValueError('Reference occupancy differs')
    diag = diagnose_graph(*raw, spec)
    report = dict(policy=asdict(config), checks=0, checked_operations=dict(atom=0,charge=0,bond=0),
                  weighted_witness=None, distance_is_exact=False)
    if diag['valid']:
        return dict(report, status='already_valid')
    if not diag.get('categorical_support_labels_ok'):
        raise ValueError('Unsupported shared graph labels')
    seen = {tuple(x.tobytes() for x in raw)}
    beam = [(raw, diag)]
    best = None
    for depth in range(1, config.max_edits + 1):
        frontier = []
        def rank(op):
            priority, field, i, j, new = op
            index = (i, j) if field == 2 else (i,)
            old = int(raw[field][index])
            cost = _edit_base_cost(float(p[field][index+(old,)]),float(p[field][index+(new,)]),config)
            extra = config.atom_extra_cost if field == 0 else config.charge_extra_cost if field == 1 else 0.
            # Locality remains first; costs decide which alternatives get checked.
            return priority, cost + extra, field, i, j, new
        streams = [_mixed_neighbors(g, raw, spec, d, rank) for g, d in beam]
        checked = 0
        for candidate in round_robin_candidates(streams):
            key = tuple(x.tobytes() for x in candidate)
            if key in seen:
                continue
            if checked >= config.checks_per_depth:
                break
            seen.add(key); checked += 1; report['checks'] += 1
            d = diagnose_graph(*candidate, spec)
            cost = exchange_cost(raw, candidate, p, spec, config, reference)
            for name,count in cost['edit_counts'].items():
                report['checked_operations'][name] += count
            valid = d['valid']
            if valid:
                try:
                    valid = encode_smiles(d['smiles'], spec)['smiles'] == d['smiles']
                except ValueError:
                    valid = False
            if valid and (best is None or cost['weighted_cost'] < best['weighted_cost']):
                best = dict(cost, labels={k:x.tolist() for k,x in zip(('atoms','charges','bonds'), candidate)}, smiles=d['smiles'])
            burden = d.get('problem_burden')
            burden = 1000000 if burden is None else burden
            priority = (burden + max(0, d.get('component_count', 1)-1), cost['weighted_cost'], checked)
            frontier.append((priority, candidate, d))
        frontier.sort(key=lambda x:x[0])
        beam = [(g,d) for _,g,d in frontier[:config.beam_width]]
        if not beam:
            break
    return dict(report, status='witness_found' if best else 'censored', weighted_witness=best)


def preservation_loss(graph, reference, config):
    """Expected reference mismatch; independent of current repair success.

    Atom/charge means use occupied slots. Motif mean uses its atoms, charges
    and unordered pairs (including NONE). Scale is explicit and separate from
    search costs. This is not a learned chemistry or population prior.
    """
    reference.labels()
    target = reference.target.graph.to(graph.atoms.device)
    active = reference.target.active.to(graph.atoms.device)
    if graph.atoms.shape[:2] != target.atoms.shape:
        raise ValueError('Preservation requires one aligned graph')
    errors = [1-getattr(graph,k).gather(-1,getattr(target,k)[...,None]).squeeze(-1)
              for k in ('atoms','charges','bonds')]
    atom = errors[0][:,active].mean(); charge = errors[1][:,active].mean()
    am = torch.tensor(reference.atom_mask.copy(), device=graph.atoms.device)
    pm = torch.tensor(np.triu(reference.pair_mask,1).copy(), device=graph.atoms.device)
    count = 2*int(am.sum())+int(pm.sum())
    motif = (errors[0][:,am].sum()+errors[1][:,am].sum()+errors[2][:,pm].sum())/max(1,count)
    loss = config.anchor_scale*(config.atom_extra_cost*atom + config.charge_extra_cost*charge
                                + config.substructure_extra_cost*motif)
    if not bool(loss.isfinite()):
        raise ValueError('Nonfinite preservation loss')
    return loss, dict(atom_mismatch=float(atom.detach()), charge_mismatch=float(charge.detach()),
                      motif_mismatch=float(motif.detach()), preservation_loss=float(loss.detach()))


def exchange_guidance(decoded, graph, spec, reference, config, *, temperature=1., step=0):
    raw = tuple(getattr(graph,k)[0].detach().argmax(-1).cpu().numpy() for k in ('atoms','charges','bonds'))
    p = tuple(getattr(graph,k)[0].detach().cpu().numpy() for k in ('atoms','charges','bonds'))
    report = search_exchange(raw,p,spec,config,reference)
    target = None
    repair_loss = sum(getattr(decoded,k).sum()*0 for k in ('atoms','charges','bonds'))
    detail = dict(target_available=False, normalized_nll=0.)
    if report['weighted_witness'] is not None:
        labels = report['weighted_witness']['labels']
        target = FrozenRepairTarget.create(HardGraphBatch(*(torch.tensor(labels[k])[None] for k in ('atoms','charges','bonds'))),
            reference.target.active, refresh_step=step, receipt_path='exchange_search')
        repair_loss, detail = fixed_target_nll(decoded,graph,target,spec,temperature=temperature,normalization='mean_per_decision')
        detail['target_available'] = True
    anchor, terms = preservation_loss(graph,reference,config)
    return repair_loss+anchor, dict(detail, **terms), target, report
