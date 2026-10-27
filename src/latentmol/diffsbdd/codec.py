"""Explicit native-atom/canonical-VAE slot maps for detached graph handoffs.

Ordinary VAE posterior-mean decoding must recover the complete admitted graph
before a fixed-occupancy policy is introduced. Canonical SMILES alone is never
used as an atom map, including for symmetric molecules.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph
from .transport import BOND_NAMES, HandoffRejected, MolecularGraph


@dataclass(frozen=True)
class SlotMap:
    active_slots: tuple[int, ...]
    native_to_slot: tuple[int, ...]
    native_atom_ids: tuple[int, ...]

    def __post_init__(self):
        if (len(self.active_slots) != len(self.native_to_slot)
                or len(set(self.active_slots)) != len(self.active_slots)
                or set(self.active_slots) != set(self.native_to_slot)
                or len(self.native_atom_ids) != len(self.active_slots)
                or len(set(self.native_atom_ids)) != len(self.active_slots)):
            raise ValueError("Invalid full bijection between native atoms and VAE slots")

    @property
    def slot_to_native(self) -> tuple[int, ...]:
        return tuple(self.native_to_slot.index(slot) for slot in self.active_slots)

    def record(self) -> dict:
        return {"active_slots": list(self.active_slots),
                "native_to_slot": list(self.native_to_slot),
                "slot_to_native": list(self.slot_to_native),
                "native_atom_ids": list(self.native_atom_ids),
                "symmetry_policy": "explicit_verified_full_graph_isomorphism; no_unique_pose_claim"}


@dataclass(frozen=True)
class EncodedGraph:
    hard: HardGraphBatch
    canonical_smiles: str
    slots: SlotMap


def _match_slots(native_graph: MolecularGraph, slot_graph: MolecularGraph,
                 active_slots: tuple[int, ...]) -> SlotMap:
    """Find a full chemical isomorphism, then verify its complete induced graph."""
    native = native_graph.to_rdkit(torch.zeros(native_graph.size, 3))
    query = slot_graph.to_rdkit(torch.zeros(slot_graph.size, 3))
    match = tuple(native.GetSubstructMatch(query, useChirality=False))
    if len(match) != native_graph.size or set(match) != set(range(native_graph.size)):
        raise HandoffRejected("VAE_native_graph_has_no_full_atom_permutation")
    native_to_slot = [-1] * native_graph.size
    for query_index, native_index in enumerate(match):
        native_to_slot[native_index] = active_slots[query_index]
    # Substructure matching alone is not an equality certificate. Explicitly
    # compare all atoms, charges and every absent/present bond after permutation.
    for q, n in enumerate(match):
        if (slot_graph.elements[q] != native_graph.elements[n]
                or slot_graph.charges[q] != native_graph.charges[n]):
            raise HandoffRejected("VAE_native_atom_permutation_changes_chemistry")
        for q2, n2 in enumerate(match):
            if slot_graph.bonds[q][q2] != native_graph.bonds[n][n2]:
                raise HandoffRejected("VAE_native_atom_permutation_changes_bonds")
    return SlotMap(active_slots, tuple(native_to_slot), native_graph.atom_ids)


def _slot_graph(hard: HardGraphBatch, spec: GraphSpec) -> tuple[MolecularGraph, tuple[int, ...]]:
    from rdkit import Chem

    if hard.atoms.shape != (1, spec.max_nodes):
        raise ValueError("A handoff requires exactly one complete VAE graph")
    active = tuple((hard.atoms[0] != 0).nonzero().flatten().cpu().tolist())
    elements = tuple(Chem.GetPeriodicTable().GetElementSymbol(
        spec.atomic_numbers[int(hard.atoms[0, slot])]) for slot in active)
    charges = tuple(spec.formal_charges[int(hard.charges[0, slot])] for slot in active)
    bonds = tuple(tuple(BOND_NAMES.index(spec.bond_names[int(hard.bonds[0, i, j])])
                        for j in active) for i in active)
    if not active:
        raise HandoffRejected("VAE_empty_natural_decode")
    return MolecularGraph(elements, charges, bonds, tuple(range(len(active)))), active


def graph_in_native_order(hard: HardGraphBatch, spec: GraphSpec,
                          slots: SlotMap) -> MolecularGraph:
    slot_graph, active = _slot_graph(hard, spec)
    if active != slots.active_slots:
        raise HandoffRejected("VAE_occupancy_changed_at_fixed_handoff")
    lookup = {slot: i for i, slot in enumerate(active)}
    order = [lookup[slot] for slot in slots.native_to_slot]
    return MolecularGraph(tuple(slot_graph.elements[i] for i in order),
                          tuple(slot_graph.charges[i] for i in order),
                          tuple(tuple(slot_graph.bonds[i][j] for j in order) for i in order),
                          slots.native_atom_ids)


def encode_graph(graph: MolecularGraph, spec: GraphSpec,
                 device="cpu") -> EncodedGraph:
    """Use the exact DBMol dataset canonical codec, retaining native atom order."""
    from rdkit import Chem

    if graph.size > spec.max_nodes:
        raise HandoffRejected("VAE_graph_exceeds_slot_capacity")
    mol = graph.to_rdkit(torch.zeros(graph.size, 3))
    Chem.RemoveStereochemistry(mol)
    source = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    try:
        encoded = encode_smiles(source, spec)
    except ValueError as exc:
        raise HandoffRejected(f"VAE_schema:{exc}") from exc
    hard = HardGraphBatch(*(torch.as_tensor(encoded[name], device=device,
                                            dtype=torch.long).unsqueeze(0)
                            for name in ("atom_types", "charges", "bond_types")))
    canonical_graph, active_slots = _slot_graph(hard, spec)
    slots = _match_slots(graph, canonical_graph, active_slots)
    if graph_in_native_order(hard, spec, slots) != graph:
        raise RuntimeError("Canonical codec did not preserve the complete native graph")
    return EncodedGraph(hard, encoded["smiles"], slots)


def _hard_record(hard: HardGraphBatch, spec: GraphSpec) -> dict:
    result = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    return {"smiles": result.smiles, "failure_reason": result.failure_reason,
            "detail": result.detail, "occupied_slots": int((hard.atoms != 0).sum()),
            "atoms": hard.atoms.detach().cpu().tolist(),
            "charges": hard.charges.detach().cpu().tolist(),
            "bonds": hard.bonds.detach().cpu().tolist()}


@dataclass(frozen=True)
class CodecAdmission:
    latent: Tensor
    encoded: EncodedGraph
    active: Tensor
    slots: SlotMap
    natural_hard: HardGraphBatch
    native_graph: MolecularGraph
    receipt: dict


@dataclass(frozen=True)
class CodecDecode:
    graph: MolecularGraph
    conditioned_hard: HardGraphBatch
    natural_hard: HardGraphBatch
    receipt: dict


class FrozenVAECodec:
    def __init__(self, model, spec: GraphSpec | None = None, *,
                 atom_count: int = 25, device=None):
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("The VAE codec requires a frozen eval model")
        self.model = model
        self.spec = model.spec if spec is None else spec
        if hasattr(model, "spec") and model.spec != self.spec:
            raise ValueError("VAE checkpoint and codec GraphSpec differ")
        self.atom_count = atom_count
        if device is None:
            first_parameter = next(model.parameters(), None)
            device = "cpu" if first_parameter is None else first_parameter.device
        self.device = torch.device(device)

    @torch.no_grad()
    def admit(self, graph: MolecularGraph) -> CodecAdmission:
        graph.require_pilot(self.atom_count)
        encoded = encode_graph(graph, self.spec, self.device)
        mu, logvar = self.model.encode(encoded.hard.to_relaxed(self.spec))
        if not torch.isfinite(mu).all() or not torch.isfinite(logvar).all():
            raise RuntimeError("VAE encoder returned nonfinite posterior parameters")
        decoded = self.model.decode(mu)
        natural = decoded.harden(self.spec)
        natural_record = _hard_record(natural, self.spec)
        if natural_record["occupied_slots"] != graph.size:
            raise HandoffRejected("VAE_natural_roundtrip_changed_atom_count",
                                  {"natural": natural_record})
        if natural_record["smiles"] != encoded.canonical_smiles:
            raise HandoffRejected("VAE_natural_roundtrip_changed_identity",
                                  {"natural": natural_record,
                                   "input_smiles": encoded.canonical_smiles})
        decoded_graph, active_slots = _slot_graph(natural, self.spec)
        slots = _match_slots(graph, decoded_graph, active_slots)
        roundtrip = graph_in_native_order(natural, self.spec, slots)
        if roundtrip != graph:
            raise HandoffRejected("VAE_natural_roundtrip_complete_graph_mismatch")
        active = (natural.atoms[0] != 0).detach()
        # Only after the ordinary roundtrip passed may conditioning be used.
        fixed = fixed_occupancy_graph(decoded, active, self.spec, 1.0)
        conditioned = HardGraphBatch(fixed.atoms.argmax(-1), fixed.charges.argmax(-1),
                                    fixed.bonds.argmax(-1))
        if graph_in_native_order(conditioned, self.spec, slots) != graph:
            raise HandoffRejected("VAE_fixed_policy_changed_admitted_graph")
        receipt = {"schema": "diffsbdd-vae-native-roundtrip-v1", "passed": True,
                   "posterior": "mean", "input_smiles": encoded.canonical_smiles,
                   "canonical_input_slots": encoded.slots.record(),
                   "decoded_slots": slots.record(), "natural": natural_record,
                   "conditioned": _hard_record(conditioned, self.spec),
                   "complete_ordered_graph_identity": True,
                   "fixed_occupancy_introduced_after_natural_identity": True,
                   "stereochemistry_scope": "stripped_graph_only"}
        return CodecAdmission(mu.detach(), encoded, active, slots, natural,
                              graph, receipt)

    def relaxed(self, latent: Tensor, admitted: CodecAdmission) -> RelaxedGraph:
        """The differentiable path: ordinary decoder, one fixed occupancy policy."""
        return fixed_occupancy_graph(self.model.decode(latent), admitted.active,
                                     self.spec, 1.0)

    @torch.no_grad()
    def decode(self, latent: Tensor, admitted: CodecAdmission) -> CodecDecode:
        decoded = self.model.decode(latent)
        natural = decoded.harden(self.spec)
        soft = fixed_occupancy_graph(decoded, admitted.active, self.spec, 1.0)
        conditioned = HardGraphBatch(soft.atoms.argmax(-1), soft.charges.argmax(-1),
                                    soft.bonds.argmax(-1))
        candidate = graph_in_native_order(conditioned, self.spec, admitted.slots)
        candidate.require_pilot(self.atom_count)
        report = {"natural": _hard_record(natural, self.spec),
                  "conditioned": _hard_record(conditioned, self.spec),
                  "slots": admitted.slots.record()}
        if report["conditioned"]["smiles"] is None:
            raise HandoffRejected("VAE_proposed_graph_invalid", report)
        return CodecDecode(candidate, conditioned, natural, report)
