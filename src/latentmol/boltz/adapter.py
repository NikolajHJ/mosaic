"""Continuous chemistry channels on explicit, fixed native Boltz atom/token rows.

No SMILES, RDKit, argmax, rematching or conformer generation occurs here. Element,
expected formal charge, bond connectivity and typed-bond embeddings vary. All
other native features, including atom names, geometry and masks, remain fixed.
"""

import torch
from torch.nn import functional as F

from latentmol.graph_policy import fixed_occupancy_graph  # compatibility re-export
from latentmol.representation import RelaxedGraph
from .contracts import NATIVE_BOND_IDS, PreparedComplex


def adapt_graph(graph: RelaxedGraph, prepared: PreparedComplex):
    """Return native continuous features and native seven-category bond marginals.

    ``token_bonds`` is P(any bond), never expected bond order. Native typed IDs
    are NONE=0, OTHER=1, SINGLE=2, DOUBLE=3, TRIPLE=4, AROMATIC=5, COVALENT=6.
    The NONE embedding participates in the expectation, including nonbonded
    ligand pairs. Protein and protein-ligand fields are retained exactly.
    """
    if graph.atoms.shape[0] != 1:
        raise ValueError("Only batch size one is supported by fixed-pose affinity")
    active = prepared.active
    if not torch.allclose(graph.occupancy[0], active.to(graph.atoms), atol=1e-6, rtol=0):
        raise ValueError("Boltz adapter requires exact prepared occupancy; condition raw logits first")
    spec, f = prepared.spec, dict(prepared.features)
    slots = active.nonzero().flatten()
    ai, ti = prepared.slot_to_atom[active], prepared.slot_to_token[active]
    element_map = F.one_hot(torch.tensor(spec.atomic_numbers, device=graph.atoms.device), 128).to(graph.atoms)
    elements = graph.atoms[:, slots] @ element_map
    f["ref_element"] = f["ref_element"].to(elements).index_copy(1, ai, elements)
    charge = graph.expected_charge(spec)[:, slots]
    f["ref_charge"] = f["ref_charge"].to(charge).index_copy(1, ai, charge)
    ligand_bonds = graph.bonds[:, slots[:, None], slots[None, :]]
    t = f["token_pad_mask"].shape[1]
    flat_pairs = (ti[:, None] * t + ti[None, :]).flatten()
    connectivity = (1 - ligand_bonds[..., 0]).reshape(1, -1, 1)
    f["token_bonds"] = f["token_bonds"].to(connectivity).reshape(1, t*t, 1).index_copy(
        1, flat_pairs, connectivity).reshape(1, t, t, 1)
    native_map = F.one_hot(torch.tensor([NATIVE_BOND_IDS[name] for name in spec.bond_names],
                                        device=graph.bonds.device), 7).to(graph.bonds)
    native_ligand = (ligand_bonds @ native_map).reshape(1, -1, 7)
    native = F.one_hot(f["type_bonds"].long(), 7).to(graph.bonds)
    native = native.reshape(1, t*t, 7).index_copy(1, flat_pairs, native_ligand).reshape(1, t, t, 7)
    return f, native
