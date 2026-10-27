"""Detached graph/pose handoffs, including the native builder's limitations.

No gradient is claimed through RDKit, OpenBabel, bond perception or UFF. The
native OpenBabel builder reconstructs neutral atoms to remove radicals. We save
that policy explicitly and require complete *ordered*, stereo-stripped graph
recovery, rather than accepting an attractive canonical SMILES alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor


BOND_NAMES = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
PILOT_ELEMENTS = frozenset(("C", "N", "O", "S", "F", "Cl"))
NATIVE_BUILDER_POLICY = {
    "bond_perception": "pinned native OpenBabel xyz->sdf",
    "native_atom_reconstruction": "reconstruct atoms by symbol to remove radicals",
    "formal_charge_preservation": False,
    "stereochemistry_optimized": False,
    "largest_fragment": False,
    "atom_removal": False,
    "relaxation_inside_builder": False,
    "native_xyz_decimal_places": 3,
    "pose_quantization_policy": "verify native .3f serialization; retain original supplied pose separately",
}


class HandoffRejected(ValueError):
    """A recorded ineligible handoff, rather than a failed native contract."""

    def __init__(self, reason: str, receipt: dict | None = None):
        super().__init__(reason)
        self.receipt = {} if receipt is None else receipt


@dataclass(frozen=True)
class MolecularGraph:
    """All atom arrays and bond rows use the native point-cloud atom order."""

    elements: tuple[str, ...]
    charges: tuple[int, ...]
    bonds: tuple[tuple[int, ...], ...]
    atom_ids: tuple[int, ...]

    def __post_init__(self):
        n = len(self.elements)
        if not n or len(self.charges) != n or len(self.atom_ids) != n:
            raise ValueError("Graph atom fields disagree")
        if len(set(self.atom_ids)) != n:
            raise ValueError("Stable native atom IDs must be unique")
        if len(self.bonds) != n or any(len(row) != n for row in self.bonds):
            raise ValueError("Bond matrix shape mismatch")
        for i, row in enumerate(self.bonds):
            if row[i] != 0:
                raise ValueError("Bond diagonal must be NONE")
            for j, bond in enumerate(row):
                if bond not in range(len(BOND_NAMES)) or bond != self.bonds[j][i]:
                    raise ValueError("Invalid/asymmetric ordered bond matrix")

    @property
    def size(self) -> int:
        return len(self.elements)

    def is_connected(self) -> bool:
        seen, frontier = {0}, [0]
        while frontier:
            for j, bond in enumerate(self.bonds[frontier.pop()]):
                if bond and j not in seen:
                    seen.add(j)
                    frontier.append(j)
        return len(seen) == self.size

    def require_pilot(self, atom_count: int = 25) -> None:
        if self.size != atom_count:
            raise HandoffRejected("changed_or_unsupported_atom_count")
        if not set(self.elements) <= PILOT_ELEMENTS:
            raise HandoffRejected("unsupported_element")
        if any(self.charges):
            raise HandoffRejected("charged_graph")
        if not self.is_connected():
            raise HandoffRejected("disconnected_graph")

    @classmethod
    def from_rdkit(cls, mol, atom_ids: tuple[int, ...] | None = None):
        from rdkit import Chem

        copied = Chem.Mol(mol)
        # No fragment selection, hydrogen removal, atom reordering or redocking.
        Chem.SanitizeMol(copied)
        n = copied.GetNumAtoms()
        bonds = [[0] * n for _ in range(n)]
        for bond in copied.GetBonds():
            name = str(bond.GetBondType())
            if name not in BOND_NAMES:
                raise HandoffRejected(f"unsupported_bond:{name}")
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds[i][j] = bonds[j][i] = BOND_NAMES.index(name)
        return cls(tuple(a.GetSymbol() for a in copied.GetAtoms()),
                   tuple(a.GetFormalCharge() for a in copied.GetAtoms()),
                   tuple(tuple(row) for row in bonds),
                   tuple(range(n)) if atom_ids is None else atom_ids)

    def to_rdkit(self, positions: Tensor):
        from rdkit import Chem

        if positions.shape != (self.size, 3) or not torch.isfinite(positions).all():
            raise HandoffRejected("invalid_pose")
        molecule = Chem.RWMol()
        for symbol, charge in zip(self.elements, self.charges):
            atom = Chem.Atom(symbol)
            atom.SetFormalCharge(charge)
            molecule.AddAtom(atom)
        kinds = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
                 Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
        for i in range(self.size):
            for j in range(i):
                if self.bonds[i][j]:
                    molecule.AddBond(i, j, kinds[self.bonds[i][j]])
                    if self.bonds[i][j] == BOND_NAMES.index("AROMATIC"):
                        molecule.GetAtomWithIdx(i).SetIsAromatic(True)
                        molecule.GetAtomWithIdx(j).SetIsAromatic(True)
        mol = molecule.GetMol()
        Chem.SanitizeMol(mol)
        conf = Chem.Conformer(self.size)
        for i, xyz in enumerate(positions.detach().cpu().double().tolist()):
            conf.SetAtomPosition(i, tuple(xyz))
        mol.AddConformer(conf, assignId=True)
        return mol


@dataclass(frozen=True)
class PoseCandidate:
    graph: MolecularGraph
    positions: Tensor


@dataclass(frozen=True)
class TransportResult:
    candidate: PoseCandidate
    changed: bool
    receipt: dict = field(default_factory=dict)


class NativeBuilder:
    """Adapter around pinned ``analysis.molecule_builder.build_molecule``.

    ``build_molecule`` must be the authenticated upstream callable, passed in
    by the asset loader. Native output is retained before sanitization; callers
    can serialize ``raw_molecule`` and raw point clouds in their own receipts.
    """

    def __init__(self, build_molecule: Callable, dataset_info: dict,
                 coordinate_tolerance: float = 2e-5):
        self.build_molecule = build_molecule
        self.dataset_info = dataset_info
        self.coordinate_tolerance = coordinate_tolerance

    def __call__(self, candidate: PoseCandidate) -> tuple[MolecularGraph, dict]:
        from rdkit import Chem

        graph, xyz = candidate.graph, candidate.positions
        decoder = self.dataset_info["atom_decoder"]
        encoder = {symbol: i for i, symbol in enumerate(decoder)}
        if not set(graph.elements) <= set(encoder):
            raise HandoffRejected("native_builder_unsupported_element")
        types = torch.tensor([encoder[s] for s in graph.elements], dtype=torch.long,
                             device=xyz.device)
        try:
            raw = self.build_molecule(xyz.detach().cpu(), types.detach().cpu(),
                                      self.dataset_info, add_coords=True,
                                      use_openbabel=True)
        except (ValueError, RuntimeError, IndexError) as exc:
            raise HandoffRejected("native_builder_failure", {
                "policy": dict(NATIVE_BUILDER_POLICY), "error": str(exc),
                "input_elements": list(graph.elements),
                "input_positions": xyz.detach().cpu().tolist()}) from exc
        if raw is None:
            raise HandoffRejected("native_builder_none")
        if raw.GetNumAtoms() != graph.size or raw.GetNumConformers() != 1:
            raise HandoffRejected("native_builder_atom_or_conformer_change")
        built_xyz = xyz.new_tensor(raw.GetConformer().GetPositions())
        # Pinned utils.write_xyz_file uses .3f, so comparing the built conformer
        # directly to full-precision input at 2e-5 would reject valid native
        # behavior. Verify its exact documented serialization transformation.
        quantized_xyz = xyz.new_tensor([[float(f"{value:.3f}") for value in row]
                                       for row in xyz.detach().cpu().tolist()])
        if not torch.allclose(built_xyz, quantized_xyz,
                              atol=self.coordinate_tolerance, rtol=0):
            raise HandoffRejected("native_builder_atom_order_or_pose_change")
        if tuple(a.GetSymbol() for a in raw.GetAtoms()) != graph.elements:
            raise HandoffRejected("native_builder_element_order_change")
        raw_receipt = {
            "policy": dict(NATIVE_BUILDER_POLICY),
            "raw_atom_symbols": [a.GetSymbol() for a in raw.GetAtoms()],
            "raw_atom_formal_charges": [a.GetFormalCharge() for a in raw.GetAtoms()],
            "raw_atom_radicals": [a.GetNumRadicalElectrons() for a in raw.GetAtoms()],
            "raw_bonds": [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
                          for b in raw.GetBonds()],
            "raw_positions": built_xyz.detach().cpu().tolist(),
            "input_full_precision_positions": xyz.detach().cpu().tolist(),
            "expected_native_serialized_positions": quantized_xyz.detach().cpu().tolist(),
            "maximum_native_coordinate_quantization": float((built_xyz - xyz).abs().max()),
        }
        try:
            raw_smiles = Chem.MolToSmiles(raw, canonical=False)
            built = MolecularGraph.from_rdkit(raw, graph.atom_ids)
        except (ValueError, RuntimeError) as exc:
            raw_receipt["error"] = str(exc)
            raise HandoffRejected(f"native_builder_sanitize:{type(exc).__name__}", raw_receipt) from exc
        sanitized = Chem.Mol(raw)
        Chem.SanitizeMol(sanitized)
        return built, {
            **raw_receipt,
            "raw_ordered_smiles": raw_smiles,
            "sanitized_ordered_smiles": Chem.MolToSmiles(sanitized, canonical=False),
            "sanitized_canonical_smiles": Chem.MolToSmiles(sanitized),
            "complete_ordered_graph_matches": built == graph,
        }


@dataclass(frozen=True)
class TransportPolicy:
    stage: str = "T1"
    atom_count: int = 25
    max_iterations: int = 50
    max_displacement: float = 0.5
    force_constant: float = 10.0

    def __post_init__(self):
        if self.stage not in {"T1", "T2"}:
            raise ValueError("Unknown transport stage")
        if self.max_iterations != 50 or self.max_displacement != 0.5 or self.force_constant != 10:
            raise ValueError("Changing sealed UFF limits requires a new protocol")

    def lift(self, baseline: PoseCandidate, proposed: MolecularGraph,
             native_builder: Callable[[PoseCandidate], tuple[MolecularGraph, dict]]) -> TransportResult:
        """Return exact no-op, T1 pose, or a bounded T2 restrained UFF pose."""
        baseline.graph.require_pilot(self.atom_count)
        proposed.require_pilot(self.atom_count)
        if baseline.graph.atom_ids != proposed.atom_ids:
            raise HandoffRejected("native_atom_map_changed")
        if baseline.positions.shape != (self.atom_count, 3):
            raise HandoffRejected("baseline_pose_shape")
        if not torch.isfinite(baseline.positions).all():
            raise HandoffRejected("baseline_pose_nonfinite")
        if proposed == baseline.graph:
            # Keep the original tensor, including its exact bits. H0 must not
            # invoke a force field or introduce builder float serialization drift.
            return TransportResult(baseline, False,
                                   {"stage": self.stage, "unchanged_graph": True,
                                    "uff_bypassed": True, "max_displacement": 0.0})
        if self.stage == "T1" and proposed.bonds != baseline.graph.bonds:
            raise HandoffRejected("T1_bond_change")
        positions = baseline.positions
        receipt = {"stage": self.stage, "unchanged_graph": False,
                   "uff_bypassed": self.stage == "T1"}
        if self.stage == "T2":
            positions, uff_receipt = self._restrained_uff(proposed, positions)
            receipt.update(uff_receipt)
        candidate = PoseCandidate(proposed, positions)
        built, builder_receipt = native_builder(candidate)
        receipt["native_builder"] = builder_receipt
        if built != proposed:
            raise HandoffRejected("proposed_graph_not_recovered_by_native_builder", receipt)
        return TransportResult(candidate, True, receipt)

    def _restrained_uff(self, graph: MolecularGraph, positions: Tensor):
        from rdkit.Chem import rdForceFieldHelpers

        # No EmbedMolecule/random conformer: start at exactly the supplied pose.
        molecule = graph.to_rdkit(positions)
        if not rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule):
            raise HandoffRejected("UFF_missing_parameters")
        forcefield = rdForceFieldHelpers.UFFGetMoleculeForceField(molecule, confId=0)
        if forcefield is None:
            raise HandoffRejected("UFF_forcefield_unavailable")
        for i in range(graph.size):
            forcefield.UFFAddPositionConstraint(i, self.max_displacement,
                                                self.force_constant)
        forcefield.Initialize()
        energy_before = float(forcefield.CalcEnergy())
        converged = forcefield.Minimize(maxIts=self.max_iterations)
        current_xyz = positions.new_tensor(molecule.GetConformer().GetPositions())
        if converged != 0:
            raise HandoffRejected("UFF_not_converged_within_50", {
                "uff_convergence_code": int(converged),
                "uff_max_iterations": self.max_iterations,
                "pose_before": positions.detach().cpu().tolist(),
                "pose_after": current_xyz.detach().cpu().tolist(),
                "uff_energy_before": energy_before,
                "uff_energy_after": float(forcefield.CalcEnergy())})
        lifted = current_xyz
        displacement = (lifted - positions).norm(dim=-1)
        if not torch.isfinite(lifted).all() or (displacement > self.max_displacement).any():
            raise HandoffRejected("UFF_displacement_exceeded", {
                "pose_before": positions.detach().cpu().tolist(),
                "pose_after": lifted.detach().cpu().tolist(),
                "per_atom_displacement": displacement.detach().cpu().tolist(),
                "max_displacement": float(displacement.max())})
        if MolecularGraph.from_rdkit(molecule, graph.atom_ids) != graph:
            raise HandoffRejected("UFF_graph_or_atom_map_changed")
        return lifted, {
            "uff_convergence_code": int(converged),
            "uff_max_iterations": self.max_iterations,
            "uff_position_constraint_max_displacement": self.max_displacement,
            "uff_position_constraint_force_constant": self.force_constant,
            "uff_energy_before": energy_before,
            "uff_energy_after": float(forcefield.CalcEnergy()),
            "per_atom_displacement": displacement.detach().cpu().tolist(),
            "max_displacement": float(displacement.max()),
            "pose_before": positions.detach().cpu().tolist(),
            "pose_after": lifted.detach().cpu().tolist(),
        }
