"""Hard-molecule SA used by the audited synthesis experiments."""

import math

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


def audited_sa_score(mol: Chem.Mol) -> float:
    """Return the RDKit SA scaled value before the discontinuous upper-tail transform.

    Uses the fragment weights and structural terms of RDKit Contrib SA_Score
    (Release_2024_09_6). The value is unrounded and unclipped, matching the
    archived threshold audit. Pass a sanitized molecule with implicit hydrogens.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Cannot evaluate synthetic accessibility on an empty molecule")
    if sascorer._fscores is None:
        sascorer.readFragmentScores()
    counts = _MORGAN.GetSparseCountFingerprint(mol).GetNonzeroElements()
    fragment_mean = sum(count * sascorer._fscores.get(key, -4.0)
                        for key, count in counts.items()) / sum(counts.values())
    atoms = mol.GetNumAtoms()
    centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridges = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    macrocycle = any(len(ring) > 8 for ring in mol.GetRingInfo().AtomRings())
    penalty = (atoms ** 1.005 - atoms + math.log10(1 + centers) + math.log10(1 + spiro)
               + math.log10(1 + bridges) + math.log10(1 + int(macrocycle)))
    symmetry = 0.5 * math.log(max(1.0, atoms / len(counts)))
    scaled = 11.0 - (fragment_mean - penalty + symmetry + 5.0) * 9.0 / 6.5
    if not math.isfinite(scaled):
        raise ValueError("Nonfinite synthetic accessibility score")
    return scaled
