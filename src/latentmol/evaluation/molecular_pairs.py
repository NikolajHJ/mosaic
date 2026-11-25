"""Bounded, model-independent selection and atom matching of chemical neighbors.

The map is B's original atom index for each A-aligned slot. An element/bond
preserving MCS anchors the shared core; a small exact assignment fills the
remaining slots using local environment costs. This is a bounded matching
heuristic, not a proof of a unique or globally optimal chemical correspondence.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import itertools
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

from latentmol.representation import HardGraphBatch
from latentmol.sampling import subset_indices


def _bond_labels(molecule):
    labels = np.zeros((molecule.GetNumAtoms(), molecule.GetNumAtoms()), dtype=np.int8)
    vocabulary = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        labels[i, j] = labels[j, i] = vocabulary[str(bond.GetBondType())]
    return labels


def _assignment(costs):
    """Exact minimum additive assignment for at most six unmatched atoms."""
    count = len(costs)
    if count > 6 or any(len(row) != count for row in costs):
        raise ValueError("Residual atom matching requires a square matrix of size <=6")
    @lru_cache(None)
    def visit(i, used):
        if i == count:
            return 0, ()
        options = []
        for j in range(count):
            if not used & (1 << j):
                value, tail = visit(i+1, used | (1 << j))
                options.append((int(costs[i][j]) + value, (j,) + tail))
        return min(options)
    return visit(0, 0)[1]


def match_atoms(a, b, *, timeout=1, minimum_fraction=.8, match_limit=16, alignment_limit=32):
    """Anchor a common core and return a bijection for equal-sized molecules."""
    n = a.GetNumAtoms()
    if n != b.GetNumAtoms() or n < 1:
        return None, "different_atom_count"
    result = rdFMCS.FindMCS([a, b], timeout=timeout, atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact, ringMatchesRingOnly=True, completeRingsOnly=True)
    if result.canceled:
        return None, "mcs_timeout"
    if result.numAtoms / n < minimum_fraction or n - result.numAtoms > 6:
        return None, "insufficient_common_core"
    query = Chem.MolFromSmarts(result.smartsString)
    if query is None or query.GetNumAtoms() != result.numAtoms:
        raise ValueError("MCS query is incompatible with its declared atom count")
    matches_a = a.GetSubstructMatches(query, uniquify=False, maxMatches=match_limit)
    matches_b = b.GetSubstructMatches(query, uniquify=False, maxMatches=match_limit)
    alternatives = sorted(itertools.product(matches_a, matches_b),
        key=lambda pair: (-sum(i == j for i, j in zip(*pair)), pair))[:alignment_limit]
    ea = np.array([atom.GetAtomicNum() for atom in a.GetAtoms()])
    eb = np.array([atom.GetAtomicNum() for atom in b.GetAtoms()])
    qa = np.array([atom.GetFormalCharge() for atom in a.GetAtoms()])
    qb = np.array([atom.GetFormalCharge() for atom in b.GetAtoms()])
    ba, bb = _bond_labels(a), _bond_labels(b)
    best = None
    for core_a, core_b in alternatives:
        remaining_a, remaining_b = sorted(set(range(n))-set(core_a)), sorted(set(range(n))-set(core_b))
        costs = [[5*int(ea[i] != eb[j]) + 2*int(qa[i] != qb[j]) +
            2*int(np.sum(ba[i, list(core_a)] != bb[j, list(core_b)])) +
            abs(a.GetAtomWithIdx(i).GetDegree()-b.GetAtomWithIdx(j).GetDegree()) +
            int(a.GetAtomWithIdx(i).IsInRing() != b.GetAtomWithIdx(j).IsInRing())
            for j in remaining_b] for i in remaining_a]
        permutation = [-1] * n
        for i, j in zip(core_a, core_b):
            permutation[i] = j
        for i, j in zip(remaining_a, _assignment(costs)):
            permutation[i] = remaining_b[j]
        if sorted(permutation) != list(range(n)):
            raise ValueError("Atom correspondence is not a bijection")
        atom_changes = int(np.sum(ea != eb[permutation]))
        charge_changes = int(np.sum(qa != qb[permutation]))
        bond_changes = int(np.sum(np.triu(ba != bb[np.ix_(permutation, permutation)], 1)))
        score = (4*atom_changes + 2*charge_changes + bond_changes,
                 -sum(i == j for i, j in enumerate(permutation)), tuple(permutation))
        if best is None or score < best[0]:
            best = (score, dict(b_original_atom_for_aligned_slot=permutation,
                common_core_atoms=result.numAtoms, common_core_fraction=result.numAtoms/n,
                core_a=list(core_a), core_b=list(core_b), mcs_smarts=result.smartsString,
                atom_changes=atom_changes, charge_changes=charge_changes, bond_changes=bond_changes,
                identity_permutation=permutation == list(range(n)),
                embedding_counts=[len(matches_a), len(matches_b)], alignments_examined=len(alternatives)))
    return (best[1], None) if best else (None, "no_core_embedding")


def match_optional_atom(smaller, larger, *, match_limit=64):
    """Map an exact shared core and one terminal atom into the smaller frame.

    Both input molecules must be sanitized. The returned permutation lists
    original larger indices in smaller atom order, followed by the optional
    atom. Core element, formal charge and every core bond must match.
    """
    count = smaller.GetNumAtoms()
    if not count or larger.GetNumAtoms() != count + 1:
        return None
    terminals = {atom.GetIdx() for atom in larger.GetAtoms() if atom.GetDegree() == 1}
    if not terminals:
        return None
    smaller_bonds, larger_bonds = _bond_labels(smaller), _bond_labels(larger)
    smaller_labels = [(atom.GetAtomicNum(), atom.GetFormalCharge()) for atom in smaller.GetAtoms()]
    larger_labels = [(atom.GetAtomicNum(), atom.GetFormalCharge()) for atom in larger.GetAtoms()]
    for core in larger.GetSubstructMatches(smaller, uniquify=False, maxMatches=match_limit):
        missing, = set(range(count + 1)).difference(core)
        if (missing in terminals and [larger_labels[index] for index in core] == smaller_labels
                and np.array_equal(smaller_bonds, larger_bonds[np.ix_(core, core)])):
            return (*core, missing)
    return None


def permute_graph(graph, permutation):
    """Relabel a single input graph; preserve padding and chemical identity."""
    n = len(permutation)
    if graph.atoms.shape[0] != 1 or sorted(permutation) != list(range(n)):
        raise ValueError("A single graph and bijective occupied-slot map are required")
    if int((graph.atoms[0] != 0).sum()) != n or not bool((graph.atoms[0, :n] != 0).all()):
        raise ValueError("Expected a canonical input with contiguous occupied slots")
    import torch
    order = torch.tensor(permutation + list(range(n, graph.atoms.shape[1])), device=graph.atoms.device)
    return HardGraphBatch(graph.atoms[:, order], graph.charges[:, order], graph.bonds[:, order][:, :, order])


def relabel_categories(atoms, charges, bonds, orders):
    """Gather node categories and both bond axes into a shared pair frame."""
    nodes = atoms.shape[1]
    atoms = atoms.gather(1, orders[..., None].expand_as(atoms))
    charges = charges.gather(1, orders[..., None].expand_as(charges))
    bonds = bonds.gather(1, orders[:, :, None, None].expand(-1, -1, nodes, bonds.shape[-1]))
    bonds = bonds.gather(2, orders[:, None, :, None].expand_as(bonds))
    return atoms, charges, bonds


def select_pairs(smiles, config, *, progress=None):
    """Choose disjoint similar pairs before seeing any decoder output."""
    started = time.monotonic()
    chosen = subset_indices(len(smiles), config["pool_count"], config["pair_seed"], shuffle_full=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    groups, molecules = defaultdict(list), {}
    excluded = Counter()
    for index in chosen:
        molecule = Chem.MolFromSmiles(smiles[index])
        if molecule is None:
            raise ValueError("Authenticated source molecule cannot be parsed")
        count = molecule.GetNumAtoms()
        if not config["min_atoms"] <= count <= config["max_atoms"]:
            excluded["atom_count"] += 1
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        fingerprint = generator.GetFingerprint(molecule)
        groups[(count, scaffold)].append((index, fingerprint))
        molecules[index] = molecule
    candidates = []
    for (count, scaffold), values in groups.items():
        # Query the earlier rows only: each undirected candidate appears once.
        for position, (i, fingerprint) in enumerate(values):
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, [fp for _, fp in values[:position]])
            nearest = sorted(((score, values[j][0]) for j, score in enumerate(similarities)
                              if score >= config["min_similarity"]), key=lambda row: (-row[0], row[1]))
            candidates.extend((score, min(i, j), max(i, j), count, scaffold)
                              for score, j in nearest[:config["neighbors_per_molecule"]])
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    pairs, used, scaffold_counts, rejected = [], set(), Counter(), Counter()
    examined = 0
    for similarity, i, j, count, scaffold in candidates:
        if len(pairs) >= config["pair_count"] or examined >= config["max_candidate_pairs"]:
            break
        if i in used or j in used:
            rejected["reused_endpoint"] += 1
            continue
        if scaffold_counts[scaffold] >= config["max_pairs_per_scaffold"]:
            rejected["scaffold_cap"] += 1
            continue
        examined += 1
        mapping, reason = match_atoms(molecules[i], molecules[j],
            timeout=config["mcs_timeout_seconds"], minimum_fraction=config["min_core_fraction"],
            match_limit=config["match_limit"], alignment_limit=config["alignment_limit"])
        if reason:
            rejected[reason] += 1
        elif (mapping["atom_changes"] > config["max_atom_changes"] or
              mapping["charge_changes"] > config["max_charge_changes"] or
              mapping["bond_changes"] > config["max_bond_changes"]):
            rejected["edit_budget"] += 1
        else:
            pairs.append(dict(pair_id=len(pairs), a_index=i, b_index=j,
                a_smiles=smiles[i], b_smiles=smiles[j], atoms=count, scaffold=scaffold,
                tanimoto=similarity, mapping=mapping))
            used.update((i, j)); scaffold_counts[scaffold] += 1
        if progress and (examined % 25 == 0 or len(pairs) == config["pair_count"]):
            progress(dict(stage="pair_selection", candidates_examined=examined, pairs=len(pairs),
                          elapsed_seconds=time.monotonic()-started))
    receipt = dict(pool_rows=len(chosen), eligible_rows=len(molecules), candidate_edges=len(candidates),
        candidates_examined=examined, selected_pairs=len(pairs), requested_pairs=config["pair_count"],
        rejected=dict(rejected), excluded=dict(excluded), elapsed_seconds=time.monotonic()-started,
        identity_permutation_pairs=sum(p["mapping"]["identity_permutation"] for p in pairs),
        scaffold_counts=dict(scaffold_counts),
        warning="Pair budget not reached; no thresholds were relaxed" if len(pairs) < config["pair_count"] else None,
        policy="Disjoint endpoints; same size/scaffold; fingerprint and bounded MCS; no decoder-based selection")
    return pairs, receipt
