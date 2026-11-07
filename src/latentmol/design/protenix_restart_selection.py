"""Candidate selection and graph diagnostics for Protenix restart campaigns."""
from __future__ import annotations

import math

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def select_candidates(ranked, limit, compare):
    """Fill half of the fresh shortlist by relaxed rank and half by source step."""
    relaxed = ranked[:limit]
    count = (limit + 1) // 2
    top = ranked[:count]
    remaining = sorted(ranked[count:], key=lambda row: (row.step, row.smiles))
    slots = min(limit - len(top), len(remaining))
    timed = [remaining[(2 * i + 1) * len(remaining) // (2 * slots)] for i in range(slots)]
    selected = {"mixed": top + timed}
    if compare:
        selected["relaxed"] = relaxed
    return selected


def _morgan_fingerprints(smiles_list):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ValueError(f"Diversity selection requires a connected, sanitized molecule: {smiles}")
        yield smiles, generator.GetFingerprint(molecule)


def select_diverse_output(archive, limit, *, max_similarity):
    """Greedily keep the best fresh scores within a strict pairwise similarity cap."""
    if type(limit) is not int or limit < 1 or isinstance(max_similarity, bool) or not 0 <= max_similarity <= 1:
        raise ValueError("Invalid output diversity selection parameters")
    ranked = sorted(archive, key=lambda smiles: (archive[smiles][1], smiles))
    selected = {}
    fingerprints = []
    for smiles, fingerprint in _morgan_fingerprints(ranked):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
        if any(similarity > max_similarity for similarity in similarities):
            continue
        selected[smiles] = archive[smiles]
        fingerprints.append(fingerprint)
        if len(selected) == limit:
            break
    return selected


def max_pairwise_similarity(smiles_list):
    """Return the maximum Morgan Tanimoto similarity, or None without a pair."""
    fingerprints = []
    maximum = None
    for _, fingerprint in _morgan_fingerprints(smiles_list):
        if fingerprints:
            similarity = max(DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints))
            maximum = similarity if maximum is None else max(maximum, similarity)
        fingerprints.append(fingerprint)
    return maximum


def select_restart_parents(archive, limit, *, score_margin=0.1, max_similarity=0.85):
    """Keep the fresh winner and spread the remaining slots over strong, distinct graphs."""
    if limit < 1 or not math.isfinite(score_margin) or score_margin < 0 or not 0 <= max_similarity <= 1:
        raise ValueError("Invalid restart diversity selection parameters")
    ranked = sorted(archive, key=lambda smiles: (archive[smiles][1], smiles))
    if not ranked:
        return {}
    best_loss = archive[ranked[0]][1]
    eligible = [smiles for smiles in ranked if archive[smiles][1] <= best_loss + score_margin]
    fingerprints = dict(_morgan_fingerprints(eligible))

    chosen = eligible[:1]
    remaining = eligible[1:]
    while remaining and len(chosen) < limit:
        branches = {archive[smiles][0].branch for smiles in chosen}
        similarity = {
            smiles: max(DataStructs.BulkTanimotoSimilarity(fingerprints[smiles],
                [fingerprints[other] for other in chosen])) for smiles in remaining
        }
        distinct = [smiles for smiles in remaining if similarity[smiles] <= max_similarity]
        smiles = min(distinct or remaining, key=lambda candidate: (
            archive[candidate][0].branch in branches, similarity[candidate], archive[candidate][1], candidate))
        chosen.append(smiles)
        remaining.remove(smiles)

    if len(chosen) < limit:
        fallback = [smiles for smiles in ranked if smiles not in chosen]
        chosen.extend(fallback[:limit - len(chosen)])
    return {smiles: archive[smiles] for smiles in chosen}


def audit_molecule(smiles):
    """Record chemistry diagnostics without filtering or changing a fresh score."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return dict(sanitized=False, connected=False, three_membered_rings=None, small_ring_flag=False)
    ring_count = sum(len(ring) == 3 for ring in mol.GetRingInfo().AtomRings())
    return dict(sanitized=True, connected=len(Chem.GetMolFrags(mol)) == 1,
                three_membered_rings=ring_count, small_ring_flag=ring_count >= 4)
