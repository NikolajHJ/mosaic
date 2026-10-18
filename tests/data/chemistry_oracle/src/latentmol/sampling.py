"""Seeded data subsets and explicit categorical draws from decoder logits.

Graph draws use conditional bond distributions, never softly occupancy-gated marginals.
Sampling does not repair, reject or resample chemically invalid molecules.
"""

import math

import numpy as np
import torch

from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph


def subset_indices(size: int, limit: int | None, seed: int, *, shuffle_full: bool = False) -> list[int]:
    """Select without replacement; avoid prefixes of canonically sorted SMILES.

    The same size/limit/seed reproduces the same indices. Use this also when
    evaluating the memorization subset of a smoke-training run. ``shuffle_full``
    makes a complete evaluation split's prefixes representative too; leave it
    false when reproducing the original pilot's ordering.
    """
    if size < 1:
        raise ValueError("Cannot sample an empty dataset")
    if limit is not None and limit < 1:
        raise ValueError("Subset limit must be positive")
    if limit is None or limit >= size:
        if shuffle_full:
            return np.random.default_rng(seed).permutation(size).tolist()
        return list(range(size))
    # Keep the random order so a smaller chemistry-evaluation prefix is also
    # representative; sorting would reintroduce a canonical-SMILES prefix bias.
    return np.random.default_rng(seed).choice(size, limit, replace=False).tolist()


def _validate_graph_fields(graph, spec: GraphSpec) -> tuple[int, int]:
    """Validate common tensor dimensions without interpreting logits as probabilities."""
    if graph.atoms.ndim != 3 or graph.atoms.shape[0] < 1:
        raise ValueError("Atom values must have shape [batch, node, category]")
    batch, nodes, _ = graph.atoms.shape
    expected = ((batch, spec.max_nodes, len(spec.atomic_numbers)),
                (batch, spec.max_nodes, len(spec.formal_charges)),
                (batch, spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
    for values, shape in zip((graph.atoms, graph.charges, graph.bonds), expected):
        if tuple(values.shape) != shape or not values.is_floating_point():
            raise ValueError("Graph tensors disagree with the graph schema")
        if values.device != graph.atoms.device or values.dtype != graph.atoms.dtype:
            raise ValueError("All graph tensors must share device and dtype")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Graph tensors must be finite")
    return batch, nodes


def _fixed_mask(fixed_active: torch.Tensor, batch: int, nodes: int, device):
    if fixed_active.dtype != torch.bool or tuple(fixed_active.shape) not in (
            (nodes,), (batch, nodes)):
        raise ValueError("Fixed occupancy must be boolean [node] or [batch, node]")
    active = fixed_active.to(device)
    return active[None].expand(batch, -1) if active.ndim == 1 else active


def _graph_log_probabilities(decoded: DecoderOutput, spec: GraphSpec,
                             temperature: float, fixed_active: torch.Tensor | None):
    """Construct the shared sampling/scoring law without changing input tensors."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling temperature must be finite and positive")
    batch, nodes = _validate_graph_fields(decoded, spec)
    atom_log = (decoded.atoms / temperature).log_softmax(-1)
    active = None
    if fixed_active is not None:
        active = _fixed_mask(fixed_active, batch, nodes, decoded.atoms.device)
        absent = torch.full_like(atom_log, -torch.inf)
        absent[..., 0] = 0.0
        if len(spec.atomic_numbers) == 1:
            if bool(active.any()):
                raise ValueError("Fixed occupied slots require a non-NO_ATOM category")
            atom_log = absent
        else:
            real_log = (decoded.atoms[..., 1:] / temperature).log_softmax(-1)
            present = torch.cat((torch.full_like(real_log[..., :1], -torch.inf), real_log), -1)
            atom_log = torch.where(active[..., None], present, absent)
    charge_log = (decoded.charges / temperature).log_softmax(-1)
    bond_log = (decoded.symmetric_bond_logits / temperature).log_softmax(-1)
    return atom_log, charge_log, bond_log, active


def _sample_probabilities(atom_prob, charge_prob, bond_prob, spec, generator):
    """Shared independent categorical draws; bond probabilities are conditional."""
    def draw(probabilities):
        return torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]),
                                 1, generator=generator).reshape(probabilities.shape[:-1])

    atoms = draw(atom_prob)
    active = atoms != 0
    charges = draw(charge_prob).masked_fill(~active, spec.formal_charges.index(0))
    batch, nodes = atoms.shape
    bonds = torch.zeros((batch, nodes, nodes), dtype=torch.long, device=atoms.device)
    left, right = torch.triu_indices(nodes, nodes, offset=1, device=atoms.device)
    if left.numel():
        edges = draw(bond_prob[:, left, right]).masked_fill(~(active[:, left] & active[:, right]), 0)
        bonds[:, left, right] = edges
        bonds[:, right, left] = edges
    return HardGraphBatch(atoms, charges, bonds)


@torch.no_grad()
def sample_hard_graph(decoded: DecoderOutput, spec: GraphSpec, *,
                      generator: torch.Generator, temperature: float = 1.0,
                      fixed_active: torch.Tensor | None = None) -> HardGraphBatch:
    """Draw one categorical graph per batch row; retain all attempted draws.

    The graph law is p(A) times the product of p(Q_i) over occupied atoms times
    the product of p(E_ij) over occupied unordered pairs i<j. Atom/charge
    probabilities are temperature-scaled softmaxes; bond probabilities use the
    symmetric raw logits. Absent charges and bonds are marginalized away and
    represented by neutral charge and NONE labels. This is a factorized decoder
    distribution, not a calibrated chemistry likelihood or a valid-only law.

    With fixed_active, present atoms are conditioned on not being NO_ATOM and
    other slots are deterministically absent. The mask may be [node] shared by
    the batch or [batch, node]. The supplied RNG must match the logits' device.
    Repeating this function or expanding decoder rows produces further draws.

    Drawing charges/pairs before masking consumes RNG for absent slots too,
    but those draws are not part of the returned graph's probability. Sampling
    is deliberately outside autograd; hard_graph_log_prob implements its score
    function without differentiating the sampled labels.
    """
    atom_log, charge_log, bond_log, _ = _graph_log_probabilities(
        decoded, spec, temperature, fixed_active)
    return _sample_probabilities(atom_log.exp(), charge_log.exp(), bond_log.exp(), spec, generator)


@torch.no_grad()
def sample_fixed_relaxed_graph(graph: RelaxedGraph, spec: GraphSpec, *,
                               generator: torch.Generator,
                               fixed_active: torch.Tensor) -> HardGraphBatch:
    """Sample saved fixed-occupancy probabilities without reconstructing logits.

    Occupancy must be exactly zero or one and agree with fixed_active. In this
    special case the stored occupied-pair bond marginals equal conditional bond
    probabilities, so the same factorized law as sample_hard_graph applies.
    General soft occupancy is rejected: those marginals cannot be sampled and
    masked again without changing the graph law.

    All fields must be normalized, nonnegative categorical probabilities.
    Absent charges must be neutral; absent/diagonal bonds must be NONE; bonds
    must be symmetric. Small floating-point normalization/symmetry tolerances
    allow saved FP32 tensors. Deterministic masks are checked exactly. The
    upper-triangle probabilities are passed directly to the shared sampler;
    neither CPU re-decoding nor temperature adjustment occurs here.
    """
    batch, nodes = _validate_graph_fields(graph, spec)
    active = _fixed_mask(fixed_active, batch, nodes, graph.atoms.device)
    for probabilities in (graph.atoms, graph.charges, graph.bonds):
        if bool((probabilities < 0).any()) or not torch.allclose(
                probabilities.sum(-1), torch.ones_like(probabilities[..., 0]), atol=1e-5, rtol=0):
            raise ValueError("Saved graph probabilities must be nonnegative and normalized")
    if not torch.equal(graph.atoms[..., 0], (~active).to(graph.atoms)):
        raise ValueError("Saved graph must have exact fixed occupancy")
    absent_atoms = graph.atoms[~active]
    if bool((absent_atoms[..., 1:] != 0).any()):
        raise ValueError("Absent slots must have deterministic NO_ATOM probabilities")
    neutral = torch.zeros_like(graph.charges[~active])
    neutral[..., spec.formal_charges.index(0)] = 1
    if not torch.equal(graph.charges[~active], neutral):
        raise ValueError("Absent charges must have deterministic neutral probabilities")
    if not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=1e-6, rtol=0):
        raise ValueError("Saved graph bond probabilities must be symmetric")
    diagonal = torch.eye(nodes, dtype=torch.bool, device=graph.atoms.device)[None]
    forced_none = ~(active[:, :, None] & active[:, None, :]) | diagonal
    no_bonds = torch.zeros_like(graph.bonds[forced_none])
    no_bonds[..., 0] = 1
    if not torch.equal(graph.bonds[forced_none], no_bonds):
        raise ValueError("Absent-pair and diagonal bonds must have deterministic NONE probabilities")
    return _sample_probabilities(graph.atoms, graph.charges, graph.bonds, spec, generator)


def hard_graph_log_prob_terms(decoded: DecoderOutput, graph: HardGraphBatch, spec: GraphSpec, *,
                        temperature: float = 1.0,
                        fixed_active: torch.Tensor | None = None) -> dict:
    """Expose log-probability components and structural support under the draw law.

    Only occupied charges and unordered occupied bonds contribute. Deterministic
    padding/diagonal constraints and fixed occupancy have probability one when
    satisfied and zero otherwise. Structural support violations receive -inf;
    wrong shapes, dtypes and out-of-vocabulary labels raise ValueError.
    Chemically invalid but structurally well-formed graphs keep their actual
    probability. Do not reject those draws in later score-function estimators.
    """
    atom_log, charge_log, bond_log, fixed = _graph_log_probabilities(
        decoded, spec, temperature, fixed_active)
    batch, nodes = decoded.atoms.shape[:2]
    for labels, shape, categories in (
            (graph.atoms, (batch, nodes), len(spec.atomic_numbers)),
            (graph.charges, (batch, nodes), len(spec.formal_charges)),
            (graph.bonds, (batch, nodes, nodes), len(spec.bond_names))):
        if labels.shape != shape or labels.dtype != torch.long or labels.device != decoded.atoms.device:
            raise ValueError("Graph labels must be long tensors matching decoder batch/device/schema")
        if bool(((labels < 0) | (labels >= categories)).any()):
            raise ValueError("Graph labels contain categories outside the schema")
    active = graph.atoms != 0
    pair = active[:, :, None] & active[:, None, :]
    supported = ((graph.bonds == graph.bonds.transpose(1, 2)).all(dim=(-1, -2))
                 & (graph.bonds.diagonal(dim1=1, dim2=2) == 0).all(-1)
                 & ((graph.bonds == 0) | pair).all(dim=(-1, -2))
                 & ((graph.charges == spec.formal_charges.index(0)) | active).all(-1))
    if fixed is not None:
        supported = supported & (active == fixed).all(-1)
    atom_term = atom_log.gather(-1, graph.atoms[..., None]).squeeze(-1).sum(-1)
    charge_term = charge_log.gather(-1, graph.charges[..., None]).squeeze(-1)
    charge_term = torch.where(active, charge_term, torch.zeros_like(charge_term)).sum(-1)
    left, right = torch.triu_indices(nodes, nodes, offset=1, device=graph.atoms.device)
    edge_term = bond_log[:, left, right].gather(
        -1, graph.bonds[:, left, right, None]).squeeze(-1)
    edge_term = torch.where(active[:, left] & active[:, right], edge_term,
                            torch.zeros_like(edge_term)).sum(-1)
    return {"atoms": atom_term, "charges": charge_term, "bonds": edge_term,
            "supported": supported}


def hard_graph_log_prob(decoded: DecoderOutput, graph: HardGraphBatch, spec: GraphSpec, *,
                        temperature: float = 1.0,
                        fixed_active: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable log q, with the legacy addition order and support behavior."""
    terms = hard_graph_log_prob_terms(decoded, graph, spec, temperature=temperature,
                                     fixed_active=fixed_active)
    return (terms["atoms"] + terms["charges"] + terms["bonds"]).masked_fill(
        ~terms["supported"], -torch.inf)
