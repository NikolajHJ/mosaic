"""Detached archives of all hard graphs and the best distinct valid molecules.

No native model is called. RDKit descriptors characterize the sanitized 2D
graph; they are not affinity, 3D quality or efficacy measurements. Canonical
SMILES deduplicates chemical identity, while saved tensors retain the selected
ordered graph and its exact stored-probability state at discovery.
"""

import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


SCHEMA = "defog-stage3-candidates-v1"
ELEMENTS = ("C", "N", "O", "S", "F", "Cl")
DESCRIPTION_SCOPE = (
    "RDKit descriptors of the raw sanitized connected 2D graph; no native rescoring, "
    "repair, coordinates, affinity or efficacy inference. Hard graph atom order is preserved."
)
DESCRIPTORS = (
    "molecular_weight", "logp", "tpsa", "qed", "hbd", "hba", "rotatable_bonds",
    "rings", "aromatic_rings", "heavy_atoms", "formal_charge", "rdkit_version",
)


def _score(value):
    return float(value) if isinstance(value, (int, float, np.number)) and math.isfinite(value) else None


def _identity(row):
    return row.get("hard_id", row.get("hard_graph_id"))


def _valid(row):
    return bool(row.get("valid") and row.get("sanitized") and row.get("connected"))


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _temporary(path):
    descriptor, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent)
    os.close(descriptor)
    return Path(name)


def _write_json(path, value):
    temporary = _temporary(path)
    try:
        content = json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n"
        if path.suffix == ".gz":
            temporary.write_bytes(gzip.compress(content.encode("utf-8"), mtime=0))
        else:
            temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _molecule(elements, bonds):
    from rdkit import Chem, rdBase

    molecule = Chem.RWMol()
    for category in elements:
        molecule.AddAtom(Chem.Atom(ELEMENTS[int(category)]))
    types = (None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
    i, j = np.triu_indices(len(elements), 1)
    for a, b, category in zip(i.tolist(), j.tolist(), bonds.tolist(), strict=True):
        if category:
            molecule.AddBond(a, b, types[category])
            if category == 4:
                molecule.GetAtomWithIdx(a).SetIsAromatic(True)
                molecule.GetAtomWithIdx(b).SetIsAromatic(True)
    molecule = molecule.GetMol()
    with rdBase.BlockLogs():
        Chem.SanitizeMol(molecule)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError("A candidate marked valid is disconnected")
    return molecule


def _descriptors(elements, bonds, expected_smiles):
    from rdkit import Chem, rdBase
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors

    molecule = _molecule(elements, bonds)
    actual_smiles = Chem.MolToSmiles(molecule, canonical=True)
    if expected_smiles != actual_smiles:
        raise ValueError("Valid candidate canonical SMILES differs from its ordered graph labels")
    calculators = {
        "molecular_weight": lambda: float(Descriptors.MolWt(molecule)),
        "logp": lambda: float(Crippen.MolLogP(molecule)),
        "tpsa": lambda: float(rdMolDescriptors.CalcTPSA(molecule)),
        "qed": lambda: float(QED.qed(molecule)),
        "hbd": lambda: int(Lipinski.NumHDonors(molecule)),
        "hba": lambda: int(Lipinski.NumHAcceptors(molecule)),
        "rotatable_bonds": lambda: int(Lipinski.NumRotatableBonds(molecule)),
        "rings": lambda: int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": lambda: int(rdMolDescriptors.CalcNumAromaticRings(molecule)),
        "heavy_atoms": lambda: int(molecule.GetNumHeavyAtoms()),
        "formal_charge": lambda: int(Chem.GetFormalCharge(molecule)),
    }
    values, errors = {"rdkit_version": rdBase.rdkitVersion}, {}
    for name, calculator in calculators.items():
        try:
            value = calculator()
            if not math.isfinite(value):
                raise ValueError("Descriptor calculation produced a nonfinite value")
            values[name] = value
        except Exception as error:
            # Optional descriptive metadata must not invalidate an observed
            # candidate or a completed optimization. Ordered-graph integrity
            # and sanitization checks above remain required and still raise.
            values[name] = None
            errors[name] = f"{type(error).__name__}: {error}"
    values.update(descriptor_status="partial" if errors else "computed",
                  descriptor_error="; ".join(f"{name}: {error}" for name, error in errors.items()) or None,
                  descriptor_errors=errors)
    return values


def write_candidate_archive(output, config, states, candidates, arrays_x, arrays_e, limit=20,
                            *, details_by_hard_id=None, compress=False):
    """Write top_valid.{json,csv,npz} and all_candidates.json for one run.

    ``states`` and probability arrays align in contiguous update order beginning
    at zero. Candidate rows contain the original first-seen ordered hard graph
    labels. The primary ranking is native hard contact score, descending; an
    exact tie keeps the earlier discovery, then the ordered hard ID.

    Top rows' first_seen/last_seen/visits refer to their selected ordered graph;
    chemical_* fields pool every observed mapping of that canonical molecule.
    Source tensors always come from the selected mapping's first occurrence,
    never the final state or a regenerated graph. Missing model_key is recorded
    explicitly rather than guessed. Existing archive files are atomically
    replaced; the run owner's writer lease provides interprocess ownership.
    Optional discovery-time gradients and teacher targets use separate boolean
    availability masks and zero-filled missing entries in top_valid.npz.
    ``compress=True`` losslessly writes all_candidates.json.gz instead of the
    uncompressed all-candidate JSON. Top-molecule tensors and ranking do not
    change; metadata identifies the chosen all-candidate archive filename.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not isinstance(compress, bool):
        raise TypeError("compress must be boolean")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    states = list(states)
    source_candidates = list(candidates.values()) if isinstance(candidates, dict) else list(candidates)
    if len(states) != len(arrays_x) or len(states) != len(arrays_e):
        raise ValueError("State records and probability arrays must align")
    if [s.get("update") for s in states] != list(range(len(states))):
        raise ValueError("Archive states must have contiguous updates beginning at zero")
    n = int(config.get("n_atoms", np.asarray(arrays_x[0]).shape[0] if len(arrays_x) else 25))
    pairs = n * (n - 1) // 2
    x = np.asarray(arrays_x) if len(arrays_x) else np.empty((0, n, 6), np.float32)
    e = np.asarray(arrays_e) if len(arrays_e) else np.empty((0, pairs, 5), np.float32)
    if x.shape != (len(states), n, 6) or e.shape != (len(states), pairs, 5):
        raise ValueError("Archived source probabilities must use the shared graph schema")
    for values in (x, e):
        if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError("Archived probabilities must be finite floating arrays")
        if (values < -1e-6).any() or not np.allclose(values.sum(-1), 1., atol=1e-6, rtol=0):
            raise ValueError("Archived source probabilities must have simplex rows")
    observed = {}
    for state in states:
        hard_id = _identity(state)
        if not isinstance(hard_id, str) or not hard_id:
            raise ValueError("Each archived state needs its ordered hard ID")
        observed.setdefault(hard_id, []).append(state)
    initial = states[0] if states else {}
    initial_smiles, initial_hard_id = initial.get("smiles"), _identity(initial)
    rows, seen, descriptor_cache = [], set(), {}
    for candidate in source_candidates:
        hard_id = _identity(candidate)
        if hard_id in seen or hard_id not in observed:
            raise ValueError("Candidate IDs must be unique and observed in the trajectory")
        seen.add(hard_id)
        visits = observed[hard_id]
        discovery = visits[0]
        first = discovery["update"]
        elements = np.asarray(candidate["element_indices"], np.int32)
        bonds = np.asarray(candidate["bond_upper_indices"], np.int32)
        if elements.shape != (n,) or bonds.shape != (pairs,):
            raise ValueError("Candidate ordered labels must match the source graph shape")
        if (elements < 0).any() or (elements >= 6).any() or (bonds < 0).any() or (bonds >= 5).any():
            raise ValueError("Candidate contains an unsupported graph category")
        if not (np.array_equal(elements, x[first].argmax(-1)) and np.array_equal(bonds, e[first].argmax(-1))):
            raise ValueError("Candidate labels differ from their discovery probabilities")
        valid = _valid(candidate)
        row = {**candidate,
            "hard_id": hard_id, "valid": valid, "hard_score": _score(candidate.get("hard_score")),
            "first_seen": first, "last_seen": visits[-1]["update"], "visits": len(visits),
            "update": first, "source_update": first,
            "is_initial_state": first == 0,
            "is_initial_identity": candidate.get("smiles") == initial_smiles if initial_smiles
                else hard_id == initial_hard_id,
            "discovery_soft_score": _score(discovery.get("soft_score")),
            "discovery_stored_soft_score": _score(discovery.get("stored_soft_score")),
            "discovery_soft_hard_gap": _score(discovery.get("soft_hard_gap")),
            "descriptor_status": "not_valid_connected",
        }
        if valid:
            smiles = candidate.get("smiles")
            if not isinstance(smiles, str) or not smiles:
                raise ValueError("A valid candidate must have canonical SMILES")
            if smiles not in descriptor_cache:
                descriptor_cache[smiles] = _descriptors(elements, bonds, smiles)
            row.update(descriptor_cache[smiles])
        rows.append(row)
    if seen != set(observed):
        raise ValueError("Every observed ordered hard graph must have a candidate record")
    rows.sort(key=lambda row: (row["first_seen"], row["hard_id"]))
    chemical = {}
    for row in rows:
        if not row["valid"] or row["hard_score"] is None:
            continue
        chemical.setdefault(row["smiles"], []).append(row)
    ranked = []
    for smiles, mappings in chemical.items():
        chosen = min(mappings, key=lambda row: (-row["hard_score"], row["first_seen"], row["hard_id"]))
        selected = {k: v for k, v in chosen.items() if k not in ("element_indices", "bond_upper_indices")}
        selected.update({
            "chemical_first_seen": min(row["first_seen"] for row in mappings),
            "chemical_last_seen": max(row["last_seen"] for row in mappings),
            "chemical_visits": sum(row["visits"] for row in mappings),
            "chemical_mapping_count": len(mappings),
        })
        ranked.append(selected)
    ranked.sort(key=lambda row: (-row["hard_score"], row["first_seen"], row["hard_id"]))
    top = ranked[:limit]
    for rank, row in enumerate(top, 1):
        row.update(rank=rank, artifact_index=rank - 1, hard_graph_artifact="top_valid.npz",
                   source_soft_artifact="top_valid.npz", metadata_artifact="top_valid.json")
    details_by_hard_id = details_by_hard_id or {}
    evidence = {}
    evidence_rows = []
    for row in top:
        details = details_by_hard_id.get(row["hard_id"], {})
        if details and details.get("update") != row["source_update"]:
            raise ValueError("Candidate detail evidence must come from the selected discovery update")
        evidence_rows.append(details)
    array_keys = [f"gradient_{component}_{leaf}" for component in ("native", "teacher", "chemistry", "combined")
                  for leaf in ("X", "E_upper")]
    array_keys += ["teacher_target_X", "teacher_target_E_upper"]
    scalar_keys = ["teacher_weight", "applied_teacher_weight", "temperature", "teacher_time"]
    for key in array_keys:
        shape = (n, 6) if key.endswith("_X") else (pairs, 5)
        values, available = [], []
        for details in evidence_rows:
            value = details.get(key)
            present = value is not None
            current = np.asarray(value) if present else np.zeros(shape, np.float32)
            if current.shape != shape or not np.issubdtype(current.dtype, np.floating) or not np.isfinite(current).all():
                raise ValueError(f"Candidate evidence {key} must be finite and match its graph head")
            values.append(current.astype(np.float32, copy=False))
            available.append(present)
        evidence[key] = np.stack(values) if values else np.empty((0, *shape), np.float32)
        evidence[f"{key}_available"] = np.asarray(available, dtype=bool)
    for key in scalar_keys:
        values = []
        available = []
        for details in evidence_rows:
            value = _score(details.get(key))
            values.append(value if value is not None else 0.)
            available.append(value is not None)
        evidence[key] = np.asarray(values, np.float64)
        evidence[f"{key}_available"] = np.asarray(available, dtype=bool)
    for index, row in enumerate(top):
        row["discovery_gradient_available"] = any(evidence[f"{key}_available"][index].item()
                                                  for key in array_keys if key.startswith("gradient_"))
        row["discovery_teacher_target_available"] = bool(
            evidence["teacher_target_X_available"][index] and evidence["teacher_target_E_upper_available"][index])
    metadata = {key: config.get(key) for key in ("campaign_id", "run_id", "condition_id", "seed", "initialization", "mock")}
    shared = {"schema": SCHEMA, **metadata, "description_scope": DESCRIPTION_SCOPE,
              "graph_schema": {"elements": list(ELEMENTS), "bond_categories": ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
                               "pairs": "lexicographic i<j", "n_atoms": n},
              "model_key_available": config.get("model_key") is not None}
    all_payload = {**shared, "candidates": rows, "ordered_graph_count": len(rows),
                   "observed_state_count": len(states)}
    top_payload = {**shared, "molecules": top, "limit": limit,
                   "unique_scored_valid_molecules": len(ranked),
                   "score_direction": "maximize native hard contact score",
                   "selection": "distinct canonical SMILES; highest-scoring observed ordered mapping; ties earliest discovery",
                   "source": "exact stored probabilities at selected ordered mapping's first occurrence",
                   "visit_scope": "first_seen/last_seen/visits are ordered mapping statistics; chemical_* pool canonical identity",
                   "discovery_evidence": {
                       "artifact": "top_valid.npz", "keys": array_keys + scalar_keys,
                       "gradient_scope": "Derivatives with respect to stored graph probabilities P, including the chosen cooled-view derivative when applicable.",
                       "teacher_gradient_scope": "Raw teacher-discrepancy gradient before teacher_weight/applied_teacher_weight; combined includes the applied coefficient.",
                       "missing": "Each evidence key has a boolean <key>_available mask; missing entries are zero-filled, never measured zero gradients.",
                       "endpoint": "A final-state discovery may have no optimization gradient or teacher target; availability remains false.",
                   }}
    all_name = "all_candidates.json.gz" if compress else "all_candidates.json"
    top_payload["all_candidates_artifact"] = all_name
    _write_json(output / all_name, all_payload)
    _write_json(output / "top_valid.json", top_payload)
    columns = ["rank", "smiles", "hard_score", "hard_id", "update", "source_update", "first_seen", "last_seen", "visits",
               "chemical_first_seen", "chemical_last_seen", "chemical_visits", "chemical_mapping_count",
               "is_initial_state", "is_initial_identity", "discovery_soft_score", "discovery_stored_soft_score",
               "discovery_soft_hard_gap", "discovery_gradient_available", "discovery_teacher_target_available",
               *DESCRIPTORS, "descriptor_status", "descriptor_error",
               "artifact_index", "hard_graph_artifact", "source_soft_artifact", "metadata_artifact"]
    csv_path = output / "top_valid.csv"
    temporary = _temporary(csv_path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(_json_safe(top))
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    by_id = {row["hard_id"]: row for row in rows}
    hard_x = np.stack([np.eye(6, dtype=np.float32)[by_id[row["hard_id"]]["element_indices"]] for row in top]) if top else np.empty((0, n, 6), np.float32)
    hard_e = np.stack([np.eye(5, dtype=np.float32)[by_id[row["hard_id"]]["bond_upper_indices"]] for row in top]) if top else np.empty((0, pairs, 5), np.float32)
    indices = np.asarray([row["source_update"] for row in top], np.int64)
    npz_path = output / "top_valid.npz"
    temporary = _temporary(npz_path)
    try:
        np.savez_compressed(temporary, hard_X=hard_x, hard_E_upper=hard_e,
                            source_soft_X=x[indices], source_soft_E_upper=e[indices],
                            source_updates=indices, hard_scores=np.asarray([row["hard_score"] for row in top], np.float64),
                            ranks=np.arange(1, len(top) + 1, dtype=np.int32),
                            smiles=np.asarray([row["smiles"] for row in top], dtype=str),
                            hard_ids=np.asarray([row["hard_id"] for row in top], dtype=str),
                            model_key=np.asarray(config.get("model_key") if config.get("model_key") is not None else [], dtype=np.uint32),
                            **evidence)
        os.replace(temporary, npz_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"top_valid": str(output / "top_valid.json"), "all_candidates": str(output / all_name),
            "unique_scored_valid_molecules": len(ranked), "saved_top_valid": len(top)}
