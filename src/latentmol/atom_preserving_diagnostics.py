"""Exact search diagnostics for fixed atoms/charges and changing bonds only.

This is an execution optimization of ``quality.diagnose_graph`` for detached
bond-only search. It does not change chemical validity or the repair objective.
Fixed labels are checked once and an atom-only RDKit template is reused. Each
candidate constructs one raw bonded molecule, detects the same chemistry
problems and sanitizes one clone for validity/radicals. No supplementary valence
proxy or graph hash is calculated unless the caller requests an identity.

The returned dictionary intentionally contains only fields used for search and
its audit, not every graded-quality metric. Unknown schemas, malformed matrices
and construction failures fall back to the unchanged authoritative diagnostic.
Use the full reference diagnostic again before accepting a final witness.
Numerical/RDKit validation belongs on allocated DTU HPC.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, rdBase

from .quality import VERSION, _labels, diagnose_graph as reference_diagnose, graph_identity


SEARCH_FIELDS = ("valid", "smiles", "schema_supported", "heavy_atoms",
                 "categorical_support_labels_ok", "problems", "problem_burden", "component_count")
BOND_TYPES = {"SINGLE": Chem.BondType.SINGLE, "DOUBLE": Chem.BondType.DOUBLE,
              "TRIPLE": Chem.BondType.TRIPLE, "AROMATIC": Chem.BondType.AROMATIC}
_CONSTRUCTION_ERRORS = (ValueError, TypeError, IndexError, KeyError, RuntimeError, OverflowError)


class BondOnlyDiagnostic:
    """Reuse exact fixed slot chemistry across a bounded bond-only search.

    ``raw`` is one unbatched hard graph. Arrays are privately copied, so caller
    mutations cannot change the atom/charge template during a search. The small
    candidate guard prevents this specialized interface silently interpreting a
    noncanonical matrix as a trusted search neighbor. Its fast path retains the
    original RDKit sanitization, disconnectedness and radical rejection rules.
    """

    def __init__(self, raw, spec):
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            raise ValueError("Bond-only diagnostics require one three-channel hard graph")
        self.spec = spec
        self.raw = tuple(np.array(value.detach().cpu().numpy() if hasattr(value, "detach") else value,
                                  copy=True) for value in raw)
        self._template = None
        self._slots = []
        self._supported = False
        try:
            atoms, charges, bonds = _labels(*self.raw, spec)
            active = atoms != 0
            neutral = spec.formal_charges.index(0)
            canonical = (np.array_equal(self.raw[1][~active], np.full(int((~active).sum()), neutral))
                         and not np.any(self.raw[2][~(active[:, None] & active[None, :])])
                         and np.array_equal(bonds, bonds.T) and not np.any(np.diag(bonds)))
            if not canonical or set(spec.bond_names) - {"NONE", *BOND_TYPES}:
                return
            self._atoms, self._charges = atoms, charges
            self._slots = np.flatnonzero(active).tolist()
            self._padding_pairs = ~(active[:, None] & active[None, :])
            self._size = len(atoms)
            self._native_bond_types = tuple(None if name == "NONE" else BOND_TYPES[name]
                                             for name in spec.bond_names)
            rw = Chem.RWMol()
            for slot in self._slots:
                atom = Chem.Atom(int(spec.atomic_numbers[int(atoms[slot])]))
                atom.SetFormalCharge(int(spec.formal_charges[int(charges[slot])]))
                rw.AddAtom(atom)
            self._template = rw.GetMol()
            self._supported = True
        except _CONSTRUCTION_ERRORS:
            # The authoritative path records unsupported labels/elements and
            # unexpected construction failures rather than inventing a burden.
            self._supported = False

    def _fallback(self, bonds, *, include_graph_id):
        result = reference_diagnose(self.raw[0], self.raw[1], bonds, self.spec)
        if not include_graph_id:
            result.pop("graph_id", None)
        return result

    def _matrix_supported(self, bonds):
        """One bounded vectorized guard, with no copies or repeated hashing."""
        return (bonds.shape == (self._size, self._size) and bonds.dtype.kind in "iu"
                and not np.any(bonds < 0) and not np.any(bonds >= len(self.spec.bond_names))
                and np.array_equal(bonds, bonds.T) and not np.any(np.diag(bonds))
                and not np.any(bonds[self._padding_pairs]))

    def diagnose(self, bonds, *, include_graph_id=False):
        """Return reference-equivalent search fields; identity is opt-in.

        Unlike the full quality report, this intentionally omits unused valence
        severity, charge totals and grading fields. Problem order and slot IDs
        stay identical to the reference, because they affect beam proposals and
        rank. A caller can request ``graph_id`` only for a retained witness.
        """
        try:
            matrix = np.asarray(bonds.detach().cpu().numpy() if hasattr(bonds, "detach") else bonds)
            if not self._supported or not self._matrix_supported(matrix):
                return self._fallback(matrix, include_graph_id=include_graph_id)
            if not self._slots:
                result = {"version": VERSION, "valid": False, "smiles": None,
                          "schema_supported": True, "heavy_atoms": 0,
                          "categorical_support_labels_ok": True, "component_count": 0,
                          "problems": [{"type": "empty", "slots": []}], "problem_burden": 1}
            else:
                result = self._diagnose_supported(matrix)
            if include_graph_id:
                result["graph_id"] = graph_identity(self.raw[0], self.raw[1], matrix, self.spec)
            return result
        except _CONSTRUCTION_ERRORS:
            return self._fallback(bonds, include_graph_id=include_graph_id)

    def _diagnose_supported(self, bonds):
        slots = self._slots
        rw = Chem.RWMol(self._template)
        active_bonds = bonds[np.ix_(slots, slots)]
        # Candidate matrices are symmetric: each undirected bond is constructed
        # exactly once, in the same sorted slot order as both reference paths.
        lefts, rights = np.nonzero(np.triu(active_bonds, 1))
        for left, right in zip(lefts.tolist(), rights.tolist()):
            category = int(active_bonds[left, right])
            name = self.spec.bond_names[category]
            rw.AddBond(left, right, self._native_bond_types[category])
            if name == "AROMATIC":
                rw.GetAtomWithIdx(left).SetIsAromatic(True)
                rw.GetAtomWithIdx(right).SetIsAromatic(True)
                # inspect_graph explicitly sets this flag. AddBond(AROMATIC)
                # also supplies it to the quality raw molecule. Preserve both.
                rw.GetBondBetweenAtoms(left, right).SetIsAromatic(True)
        mol = rw.GetMol()
        component_count = len(Chem.GetMolFrags(mol))
        with rdBase.BlockLogs():
            detected = Chem.DetectChemistryProblems(mol)
        problems, implicated, global_count = [], set(), 0
        for problem in detected:
            ids = ([problem.GetAtomIdx()] if hasattr(problem, "GetAtomIdx") else
                   list(problem.GetAtomIndices()) if hasattr(problem, "GetAtomIndices") else [])
            problem_slots = [slots[index] for index in ids]
            implicated.update(problem_slots)
            global_count += int(not problem_slots)
            problems.append({"type": problem.GetType(), "slots": problem_slots})
        with rdBase.BlockLogs():
            sanitized = Chem.Mol(mol)
            flag = Chem.SanitizeMol(sanitized, catchErrors=True)
        radical_slots = []
        if int(flag) == 0:
            radical_slots = [slots[atom.GetIdx()] for atom in sanitized.GetAtoms()
                             if atom.GetNumRadicalElectrons()]
            implicated.update(radical_slots)
            if radical_slots:
                problems.append({"type": "radical", "slots": radical_slots})
        elif not detected:
            global_count += 1
            problems.append({"type": "sanitization_other", "slots": []})
        valid = int(flag) == 0 and component_count == 1 and not radical_slots
        smiles = Chem.MolToSmiles(sanitized, canonical=True, isomericSmiles=False) if valid else None
        return {"version": VERSION, "valid": valid, "smiles": smiles,
                "schema_supported": True, "heavy_atoms": len(slots),
                "categorical_support_labels_ok": True, "component_count": component_count,
                "problems": problems, "problem_burden": len(implicated) + global_count}
