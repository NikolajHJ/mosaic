"""Discrete, lossless input ordering for an explicitly selected encoder mode.

The molecular decoder and all relaxed scorer paths are independent of this
module. No molecule is repaired, sampled, normalized or selected by a score.
"""
import torch
from rdkit import rdBase

from latentmol.chemistry import canonical_slot_order
from latentmol.datasets.data import collate_graphs
from latentmol.representation import HardGraphBatch, RelaxedGraph


def canonical_encoder_input(graph: RelaxedGraph, spec) -> RelaxedGraph:
    """Canonicalize verified one-hot input; never silently harden a soft graph.

    Encoder *parameter* gradients remain live. Required input gradients are
    explicitly rejected: the discrete ordering is not differentiable. Latent
    optimization uses the decoder only and never enters this function.
    """
    batch, nodes = graph.atoms.shape[:2]
    shapes = ((batch, spec.max_nodes, len(spec.atomic_numbers)),
              (batch, spec.max_nodes, len(spec.formal_charges)),
              (batch, spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
    values = (graph.atoms, graph.charges, graph.bonds)
    if batch < 1 or nodes != spec.max_nodes or len({v.device for v in values}) != 1:
        raise ValueError("Canonical encoder input has incompatible shape/devices")
    for value, shape in zip(values, shapes):
        if value.requires_grad:
            raise ValueError("Canonical input ordering cannot supply input gradients; encoder parameter gradients remain supported")
        if value.shape != shape or not bool(torch.isfinite(value).all()):
            raise ValueError("Canonical encoder input has incompatible shape or nonfinite values")
        if not bool((((value == 0) | (value == 1)).all()) and (value.sum(-1) == 1).all()):
            raise ValueError("Canonical input ordering requires discrete one-hot graphs; soft inputs are unsupported")
    hard = HardGraphBatch(*(v.argmax(-1).cpu() for v in values))
    orders = torch.tensor([canonical_slot_order(hard.atoms[i], hard.charges[i], hard.bonds[i], spec)
                           for i in range(batch)], dtype=torch.long, device=graph.atoms.device)
    atoms = graph.atoms.gather(1, orders[..., None].expand(-1, -1, shapes[0][-1]))
    charges = graph.charges.gather(1, orders[..., None].expand(-1, -1, shapes[1][-1]))
    bonds = graph.bonds.gather(1, orders[:, :, None, None].expand(-1, -1, nodes, shapes[2][-1]))
    bonds = bonds.gather(2, orders[:, None, :, None].expand(-1, nodes, -1, shapes[2][-1]))
    return RelaxedGraph(atoms, charges, bonds)


def verify_canonical_training_cache(dataset, indices) -> int:
    """Verify cache provenance and a bounded sample before bypassing reordering."""
    convention = dataset.manifest.get("provenance", {}).get("canonicalization")
    if convention not in {
        "RDKit canonical SMILES; reparsed for stable atom ordering",
        "RDKit canonical stereo-free SMILES; reparsed for ordered atom slots",
    }:
        raise ValueError("Canonical training requires a prepared cache with declared canonical atom ordering")
    if dataset.manifest.get("rdkit_version") != rdBase.rdkitVersion:
        raise ValueError("Canonical training requires the RDKit version used to prepare the cache")
    selected = indices[:16]
    graph = collate_graphs([dataset[index] for index in selected]).to_relaxed(dataset.spec)
    canonical = canonical_encoder_input(graph, dataset.spec)
    if any(not torch.equal(getattr(graph, key), getattr(canonical, key))
           for key in ("atoms", "charges", "bonds")):
        raise ValueError("Cached training graphs differ from the encoder's canonical ordering")
    return len(selected)
