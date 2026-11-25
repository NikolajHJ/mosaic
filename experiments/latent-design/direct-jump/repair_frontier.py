"""Parent selection on ordered 25-slot graphs for the BRD4 direct-probability search."""

import hashlib
import math

import numpy as np
from rdkit import Chem, rdBase

from study import hard_smiles


def _labels(graph):
    atom, bond = graph
    for value, shape in ((atom, (25, 6)), (bond, (300, 5))):
        if (value.shape != shape or not np.isfinite(value).all()
                or (value < -1e-6).any() or not np.allclose(value.sum(1), 1, atol=2e-5, rtol=0)):
            raise ValueError("Expected finite normalized 25-slot atom and unordered bond rows")
    return atom.argmax(1).astype(np.uint8), bond.argmax(1).astype(np.uint8)


def hard_graph_key(graph):
    """Slot order changes the Protenix input even for the same canonical molecule."""
    atoms, bonds = _labels(graph)
    return hashlib.sha256(atoms.tobytes() + bonds.tobytes()).hexdigest()


def _canonical(smiles):
    if not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() != 25 or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _relaxed(point):
    value = float(point["relaxed_loss"])
    if not math.isfinite(value):
        raise ValueError("Nonfinite relaxed predicted pocket contact")
    return value


def near_valid_key(point):
    """Prefer verified bond-repair witnesses, then fewer chemical defects and lower contact loss."""
    edits = point.get("repair_edits")
    witness = bool(point.get("repair_smiles")) and edits is not None
    if witness and (int(edits) != edits or edits < 1):
        raise ValueError("An invalid graph needs at least one edit for a valid repair")
    components = point.get("component_count", point.get("components"))
    if components is None or int(components) != components or components < 1:
        raise ValueError("Near-valid selection requires the number of connected components")
    problems = point.get("problem_burden")
    valence = point.get("overvalence_excess", point.get("valence_excess"))
    if any(x is not None and (not math.isfinite(x) or x < 0) for x in (problems, valence)):
        raise ValueError("Invalid near-valid chemistry diagnostics")
    extra = int(components) - 1
    return (int(not witness), int(edits) if witness else 0,
            int(problems is None), extra + (problems or 0), extra,
            int(valence is None), valence or 0., _relaxed(point))


def _record(point, key):
    return dict(graph_key=key, smiles=point.get("smiles"), branch=point["branch"],
                step=int(point["step"]), relaxed_loss=_relaxed(point))


def _percentiles(values):
    """Midrank equal scores; zero is best. A singleton has rank zero."""
    ordered = sorted(values, key=values.get)
    result = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        percentile = (start + end - 1) / (2 * max(1, len(ordered) - 1))
        for name in ordered[start:end]:
            result[name] = percentile
        start = end
    return result


class ParentFrontier:
    """One bound native scorer (including its seed) and one fixed-receptor Vina panel."""

    def __init__(self, backend, docking=None):
        self.backend = backend
        self.docking = docking
        self.contact_cache = {}

    def hard_contact(self, graph):
        key = hard_graph_key(graph)
        if key not in self.contact_cache:
            atoms, bonds = _labels(graph)
            hard = (np.eye(6, dtype=np.float32)[atoms], np.eye(5, dtype=np.float32)[bonds])
            if not _canonical(hard_smiles(hard)):
                raise ValueError("Hard predicted pocket contact requested for an invalid molecule")
            # Keep the ordered atom-to-model mapping. Re-encoding SMILES can permute slots.
            loss = float(self.backend.hard_value(*hard))
            if not math.isfinite(loss):
                raise ValueError("Nonfinite hard predicted pocket contact")
            self.contact_cache[key] = loss
        return self.contact_cache[key]

    def select(self, valid_points, near_points, *, count, vina_weight, expanded_keys=()):
        """Score every eligible ordered graph and dock each distinct identity before ranking."""
        if (type(count) is not int or not 1 <= count <= 20 or not 0 <= vina_weight <= 1
                or (vina_weight and self.docking is None)):
            raise ValueError("Invalid parent count or Vina rank weight")
        expanded = set(expanded_keys)
        receipt = dict(mode="valid", vina_weight=vina_weight, candidates=[], rejected=[],
                       near_candidates=[], selected=[])
        points = {}
        for point in valid_points:
            if point["step"] <= 0:
                continue
            key = hard_graph_key(point["graph"])
            former = points.get(key)
            if former is None or (_relaxed(point), point["branch"], point["step"]) < (
                    _relaxed(former), former["branch"], former["step"]):
                points[key] = point

        representatives = {}
        for key, point in sorted(points.items()):
            graph = point["graph"]
            smi = _canonical(hard_smiles(graph))
            row = _record(point, key)
            if smi is None or _canonical(point.get("smiles")) != smi:
                receipt["rejected"].append(dict(row, reason="invalid_reparse_or_identity_mismatch"))
                continue
            row.update(smiles=smi, hard_contact_loss=self.hard_contact(graph))
            receipt["candidates"].append(row)
            former = representatives.get(smi)
            if former is None or (row["hard_contact_loss"], row["relaxed_loss"], key) < (
                    former["hard_contact_loss"], former["relaxed_loss"], former["graph_key"]):
                representatives[smi] = row

        if not representatives:
            receipt["mode"] = "near_valid"
            pool = {}
            for point in near_points:
                if point["step"] <= 0 or point.get("smiles"):
                    continue
                key = hard_graph_key(point["graph"])
                criteria = near_valid_key(point)
                rank = (criteria, point["branch"], point["step"])
                if key not in pool or rank < pool[key][0]:
                    pool[key] = (rank, point)
            best = sorted(pool, key=lambda key: (pool[key][0], key))[:20]
            receipt["near_pool_size"] = len(pool)
            for key in best:
                rank, point = pool[key]
                receipt["near_candidates"].append(dict(_record(point, key), near_valid_key=list(rank[0]),
                    repair_edits=point.get("repair_edits"), repair_smiles=point.get("repair_smiles"),
                    component_count=point.get("component_count", point.get("components")),
                    problem_burden=point.get("problem_burden"),
                    overvalence_excess=point.get("overvalence_excess", point.get("valence_excess"))))
            ranked = sorted(receipt["near_candidates"], key=lambda r: r["graph_key"] in expanded)
            receipt["selected"] = ranked[:count]
            return [pool[r["graph_key"]][1] for r in ranked[:count]], receipt

        names = sorted(representatives)
        docked = {}
        if self.docking is not None:
            for i in range(0, len(names), self.docking.panel_size):
                docked.update(self.docking.score_many(names[i:i + self.docking.panel_size]))
        contact = _percentiles({s: r["hard_contact_loss"] for s, r in representatives.items()})
        vina = _percentiles({s: r["affinity_kcal_mol"] for s, r in docked.items() if r["status"] == "ok"})
        weight = vina_weight if vina else 0.
        receipt["effective_vina_weight"] = weight
        for row in receipt["candidates"]:
            name = row["smiles"]
            dock = docked.get(name, {})
            row.update(vina_status=dock.get("status", "disabled"), vina_reason=dock.get("reason"),
                       vina_kcal_mol=dock.get("affinity_kcal_mol"),
                       is_representative=representatives[name] is row, contact_rank=contact[name],
                       vina_rank=vina.get(name),
                       combined_rank=(1 - weight) * contact[name] + weight * vina.get(name, 1.))
        # Protect successful docking when that objective participates. A total docking failure
        # falls back to hard contact over valid parents; it never converts them into invalids.
        ranked = sorted(representatives.values(), key=lambda r: (
            bool(weight) and r["vina_status"] != "ok", r["graph_key"] in expanded,
            r["combined_rank"], r["hard_contact_loss"], r["smiles"]))
        receipt["selected"] = ranked[:count]
        return [points[r["graph_key"]] for r in ranked[:count]], receipt
