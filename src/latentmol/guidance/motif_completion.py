"""Local bond-order supervision on prior decodes with a chemically feasible near miss."""

import numpy as np
import math
import torch
from torch.nn import functional as F
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from latentmol.chemistry import _label_molecule, inspect_graph


AMIDE = Chem.MolFromSmarts("[CX3](=[OX1])-[NX3,NX4+]")
_ALCOHOL_NEXT_TO_N = Chem.MolFromSmarts("[O;X2;H1;+0]-[C;X4;+0]-[N;X3,X4+]")
_AROMATIC = Chem.BondType.AROMATIC
_DOUBLE = Chem.BondType.DOUBLE
_SINGLE = Chem.BondType.SINGLE


def _valid(mol):
    if Chem.SanitizeMol(mol, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
        return False
    return len(Chem.GetMolFrags(mol)) == 1 and all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())


def _roundtrips(mol):
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)) is not None


def _completion_targets(atoms, charges, bonds, spec, need_ring, need_amide, ring_encoding="aromatic"):
    """Detached proposals on occupied slots. Accept at most one verified edit per group per row."""
    slots = np.flatnonzero(atoms)
    if not len(slots):
        return [], []
    mol = _label_molecule(atoms, charges, bonds[np.ix_(slots, slots)], spec, slots)
    if not _valid(mol):
        return [], []
    ring_edges, amide_edges = [], []
    occupied_ring = set()
    if need_ring and not rdMolDescriptors.CalcNumAromaticRings(mol):
        candidates = []
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) != 6:
                continue
            centers = [mol.GetAtomWithIdx(i) for i in ring]
            if any(a.GetAtomicNum() not in (6, 7) or a.GetFormalCharge() or
                   a.GetDegree() > (3 if a.GetAtomicNum() == 6 else 2) for a in centers):
                continue
            edges = list(zip(ring, ring[1:] + ring[:1]))
            types = [mol.GetBondBetweenAtoms(i, j).GetBondType() for i, j in edges]
            if all(t in (_SINGLE, _DOUBLE) for t in types):
                candidates.append((types.count(_DOUBLE), edges))
        # Prefer a partially unsaturated ring. Limit work even for unusual polycycles.
        for _, edges in sorted(candidates, key=lambda item: -item[0])[:3]:
            proposed = Chem.Mol(mol)
            for i, j in edges:
                bond = proposed.GetBondBetweenAtoms(i, j)
                bond.SetBondType(_AROMATIC)
                bond.SetIsAromatic(True)
                proposed.GetAtomWithIdx(i).SetIsAromatic(True)
                proposed.GetAtomWithIdx(j).SetIsAromatic(True)
            if (_valid(proposed) and all(proposed.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in edges)
                    and _roundtrips(proposed)):
                if ring_encoding == "kekule":
                    # Teach an alternating assignment which ends at the same canonical
                    # molecule, even when the proposed ring belongs to a fused system.
                    endpoint = Chem.MolToSmiles(proposed, canonical=True, isomericSmiles=False)
                    Chem.Kekulize(proposed, clearAromaticFlags=True)
                    encoded = bonds.copy()
                    for i, j in edges:
                        label = str(proposed.GetBondBetweenAtoms(i, j).GetBondType())
                        encoded[slots[i], slots[j]] = encoded[slots[j], slots[i]] = spec.bond_names.index(label)
                    if inspect_graph(atoms, charges, encoded, spec).smiles != endpoint:
                        continue
                ring_edges = [(int(slots[i]), int(slots[j]),
                               spec.bond_names.index(str(proposed.GetBondBetweenAtoms(i, j).GetBondType())))
                              for i, j in edges]
                occupied_ring = {i for i, _ in edges}
                break
    if need_amide and not mol.HasSubstructMatch(AMIDE):
        for oxygen, carbon, nitrogen in mol.GetSubstructMatches(_ALCOHOL_NEXT_TO_N, maxMatches=3):
            if occupied_ring.intersection((oxygen, carbon, nitrogen)):
                continue
            proposed = Chem.Mol(mol)
            proposed.GetBondBetweenAtoms(carbon, oxygen).SetBondType(_DOUBLE)
            if _valid(proposed) and proposed.HasSubstructMatch(AMIDE) and _roundtrips(proposed):
                amide_edges = [(int(slots[carbon]), int(slots[oxygen]), spec.bond_names.index("DOUBLE"))]
                break
    return ring_edges, amide_edges


def completion_loss(decoded, spec, *, need_ring, need_amide, ring_encoding="aromatic", amide_relative_weight=1.):
    """Train conditional bond types on fixed candidate edges of compatible hard skeletons.

    Site selection and full-graph sanitization have no derivative. Each selected ring
    contributes the mean over its six edges. Each motif contributes half the objective,
    including zero if no suitable site exists. The NO_BOND logit receives no direct
    gradient here. Updates to the shared decoder can still change edge probabilities.
    """
    if ring_encoding not in ("aromatic", "kekule"):
        raise ValueError(f"Unknown aromatic ring target encoding: {ring_encoding}")
    if not math.isfinite(amide_relative_weight) or amide_relative_weight < 0:
        raise ValueError("Amide completion weight must be finite and nonnegative")
    zero = decoded.bonds[0, 0, 0, 0] * 0
    metrics = dict(completion_aromatic_rows=0, completion_amide_rows=0)
    if not (need_ring or need_amide):
        return zero, metrics
    with torch.no_grad(), rdBase.BlockLogs():
        hard = decoded.harden(spec)
        arrays = [t.cpu().numpy() for t in (hard.atoms, hard.charges, hard.bonds)]
        targets = [[], []]
        for row, graph in enumerate(zip(*arrays)):
            choices = _completion_targets(*graph, spec, need_ring, need_amide, ring_encoding)
            for group, edges in enumerate(choices):
                targets[group].extend((row, i, j, label) for i, j, label in edges)
    total = zero
    terms = ((targets[0], "completion_aromatic_rows", 6, 1.),
             (targets[1], "completion_amide_rows", 1, amide_relative_weight))
    for edges, key, length, scale in terms:
        metrics[key] = len(edges) // length
        if edges:
            row, left, right, labels = torch.as_tensor(edges, device=decoded.bonds.device).T
            logits = (decoded.bonds[row, left, right, 1:] + decoded.bonds[row, right, left, 1:]) * .5
            total = total + (.5 * scale) * F.cross_entropy(logits, labels - 1)
    return total, metrics
