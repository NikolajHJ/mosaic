"""Mine real A–B–C molecules linked by disjoint, exact single graph edits."""

from collections import Counter
import time

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from latentmol.chemistry import MoleculeRejected, canonical_slot_order, decode_graph, encode_smiles
from latentmol.sampling import subset_indices


def _edits(atoms, bonds, spec):
    """Enumerate conservative element or bond-order edits in canonical B slots."""
    size = int(np.count_nonzero(atoms))
    aromatic = spec.bond_names.index("AROMATIC")
    active = bonds[:size, :size]
    for i, row in enumerate(active):
        degree = np.count_nonzero(row)
        if degree > 2 or aromatic in row:
            continue
        for element in (6, 7, 8, 9, 16, 17, 35):
            if element in spec.atomic_numbers:
                replacement = spec.atomic_numbers.index(element)
                if replacement != atoms[i]:
                    yield "atom", (i,), replacement
    orders = tuple(spec.bond_names.index(name) for name in ("SINGLE", "DOUBLE", "TRIPLE"))
    for i, j in zip(*np.nonzero(np.triu(active, 1))):
        current = bonds[i, j]
        if current in orders:
            position = orders.index(current)
            for alternate in (position - 1, position + 1):
                if 0 <= alternate < len(orders):
                    yield "bond", (int(i), int(j)), orders[alternate]


def _check_cache(dataset, index, encoded):
    row = dataset[index]
    if any(not np.array_equal(np.asarray(row[name]), encoded[key])
           for name, key in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))):
        raise ValueError("Cached labels differ from the verified canonical triplet molecule")


def _edited_smiles(molecule, kind, slots, replacement, spec):
    """Fast RDKit membership prefilter, before exact graph-label round-trip."""
    edited = Chem.RWMol(molecule)
    if kind == "atom":
        i, = slots
        atom = Chem.Atom(spec.atomic_numbers[replacement])
        atom.SetFormalCharge(molecule.GetAtomWithIdx(i).GetFormalCharge())
        edited.ReplaceAtom(i, atom)
    else:
        order = getattr(Chem.BondType, spec.bond_names[replacement])
        edited.GetBondBetweenAtoms(*slots).SetBondType(order)
    try:
        Chem.SanitizeMol(edited)
    except (ValueError, RuntimeError):
        return None
    return Chem.MolToSmiles(edited, canonical=True, isomericSmiles=False)


def _neighbors(center, molecule, spec, identity, used, excluded, allow_derived, counters):
    atoms, charges, bonds = (center[key] for key in ("atom_types", "charges", "bond_types"))
    size = int(np.count_nonzero(atoms))
    atom_work, bond_work = atoms.copy(), bonds.copy()
    for kind, slots, replacement in _edits(atoms, bonds, spec):
        counters["proposals"] += 1
        smiles = _edited_smiles(molecule, kind, slots, replacement, spec)
        if (smiles is None or smiles == center["smiles"] or smiles in used or smiles in excluded
                or (not allow_derived and smiles not in identity)):
            continue
        if kind == "atom":
            i, = slots
            previous = int(atoms[i])
            atom_work[i] = replacement
        else:
            i, j = slots
            previous = int(bonds[i, j])
            bond_work[i, j] = bond_work[j, i] = replacement
        if decode_graph(atom_work, charges, bond_work, spec) == smiles:
            try:
                order = canonical_slot_order(atom_work, charges, bond_work, spec)
            except MoleculeRejected:
                counters["unrepresentable_maps"] += 1
            else:
                canonical_for_b_slot = np.argsort(order)[:size].tolist()
                index = identity.get(smiles)
                provenance = "observed" if index is not None else "derived"
                counters[provenance + "_mapped_proposals"] += 1
                yield dict(index=index, smiles=smiles, order=canonical_for_b_slot, provenance=provenance,
                           edit=dict(kind=kind, slots=list(slots), from_label=previous, to_label=replacement))
        else:
            counters["representation_mismatch"] += 1
        if kind == "atom":
            atom_work[i] = previous
        else:
            bond_work[i, j] = bond_work[j, i] = previous


def select_triplets(dataset, config, *, progress=None, excluded_identities=()):
    """Select triplets within one source split without reading model outputs.

    Each endpoint differs from B in one label in the same graph frame. The
    changed slots of the two edits are disjoint. By default every identity is
    observed in this split. Held-out panels may use verified derived endpoints
    outside the training split by passing all blocked identities explicitly.
    ``require_matching_slots`` keeps both endpoint maps in canonical B order.
    """
    started = time.monotonic()
    count, limit = config["count"], config["center_limit"]
    lower, upper = config["min_atoms"], config["max_atoms"]
    cap = config.get("max_per_scaffold", 8)
    allow_derived = config.get("allow_derived_endpoints", False)
    require_matching_slots = config.get("require_matching_slots", False)
    excluded = excluded_identities if isinstance(excluded_identities, (set, frozenset)) else set(excluded_identities)
    if (any(type(v) is not int or v < 1 for v in (count, limit, lower, upper, cap))
            or lower > upper or upper > dataset.spec.max_nodes or type(allow_derived) is not bool
            or type(require_matching_slots) is not bool
            or (allow_derived and not excluded)):
        raise ValueError("Invalid graph triplet mining limits")
    identity = {smiles: index for index, smiles in enumerate(dataset.smiles)}
    if len(identity) != len(dataset):
        raise ValueError("Triplet source split requires unique canonical molecules")
    centers = subset_indices(len(dataset), limit, config["seed"], shuffle_full=True)
    spec = dataset.spec
    used, scaffold_counts, counters, provenance, triplets = set(), Counter(), Counter(), Counter(), []
    for examined, row in enumerate(centers, start=1):
        b_smiles = dataset.smiles[row]
        if b_smiles in used or b_smiles in excluded:
            continue
        center = encode_smiles(b_smiles, spec)
        size = int(np.count_nonzero(center["atom_types"]))
        if not lower <= size <= upper:
            continue
        molecule = Chem.MolFromSmiles(b_smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
        if scaffold_counts[scaffold] >= cap:
            continue
        earlier = []
        b_order = list(range(size))
        with rdBase.BlockLogs():
            for candidate in _neighbors(center, molecule, spec, identity, used, excluded, allow_derived, counters):
                if require_matching_slots and candidate["order"] != b_order:
                    counters["nonmatching_slot_proposals"] += 1
                    continue
                candidate_slots = set(candidate["edit"]["slots"])
                first = next((a for a in earlier if a["smiles"] != candidate["smiles"]
                              and candidate_slots.isdisjoint(a["edit"]["slots"])), None)
                if first is None:
                    earlier.append(candidate)
                    continue
                a, c = first, candidate
                for endpoint in (a, c):
                    provenance[endpoint["provenance"] + "_endpoints"] += 1
                    if endpoint["index"] is not None:
                        _check_cache(dataset, endpoint["index"], encode_smiles(endpoint["smiles"], spec))
                derived = a["index"] is None or c["index"] is None
                provenance["triplets_with_derived_endpoints" if derived else "all_observed_triplets"] += 1
                _check_cache(dataset, row, center)
                triplets.append(dict(
                    triplet_id=len(triplets), atoms=size, scaffold=scaffold,
                    a_index=a["index"], b_index=row, c_index=c["index"],
                    a_smiles=a["smiles"], b_smiles=b_smiles, c_smiles=c["smiles"],
                    a_provenance=a["provenance"], b_provenance="observed", c_provenance=c["provenance"],
                    mapping=dict(a_original_atom_for_b_slot=a["order"],
                                 c_original_atom_for_b_slot=c["order"]),
                    a_from_b_edit=a["edit"], c_from_b_edit=c["edit"],
                ))
                used.update((a["smiles"], b_smiles, c["smiles"]))
                scaffold_counts[scaffold] += 1
                break
        if progress and (examined % 250 == 0 or len(triplets) == count):
            progress(dict(stage="triplet_mining", centers_examined=examined,
                          selected_triplets=len(triplets), seconds=time.monotonic() - started))
        if len(triplets) == count:
            break
    receipt = dict(
        source_rows=len(identity), centers_examined=examined, requested_triplets=count,
        selected_triplets=len(triplets), unique_molecules=len(used), scaffold_counts=dict(scaffold_counts),
        proposal_counts=dict(counters), provenance=dict(provenance), allow_derived_endpoints=allow_derived,
        require_matching_slots=require_matching_slots,
        excluded_identities=len(excluded), elapsed_seconds=time.monotonic() - started,
        policy="Observed canonical center; two disjoint single-label edits; verified B-frame atom maps",
    )
    return triplets, receipt
