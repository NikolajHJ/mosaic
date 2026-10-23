"""Small shared configuration helpers for ligand graph experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path


@dataclass(frozen=True)
class LossSchedule:
    """Interpolate normalized percentage weights between update keyframes."""

    names: tuple[str, ...]
    keyframes: tuple[dict, ...]
    updates: int

    def __post_init__(self):
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Loss names must be nonempty and unique")
        if type(self.updates) is not int or self.updates < 1:
            raise ValueError("updates must be a positive integer")
        if not self.keyframes:
            raise ValueError("At least one loss-weight keyframe is required")
        previous = 0
        for frame in self.keyframes:
            step, weights = frame.get("update"), frame.get("weights_pct")
            if type(step) is not int or not 1 <= step <= self.updates or step <= previous:
                raise ValueError("Keyframe updates must increase within the run")
            if set(weights or {}) != set(self.names):
                raise ValueError("Every keyframe must specify every configured loss")
            values = list(weights.values())
            if any(not math.isfinite(x) or x < 0 for x in values) or not math.isclose(
                sum(values), 100.0, abs_tol=1e-6
            ):
                raise ValueError("Loss percentages must be nonnegative and sum to 100")
            previous = step

    def at(self, update: int) -> dict[str, float]:
        if type(update) is not int or not 1 <= update <= self.updates:
            raise ValueError("Update is outside the configured run")
        frames = self.keyframes
        if update <= frames[0]["update"]:
            row = frames[0]["weights_pct"]
        elif update >= frames[-1]["update"]:
            row = frames[-1]["weights_pct"]
        else:
            left, right = next(
                (a, b) for a, b in zip(frames, frames[1:])
                if a["update"] <= update <= b["update"]
            )
            t = (update - left["update"]) / (right["update"] - left["update"])
            row = {
                name: (1 - t) * left["weights_pct"][name]
                + t * right["weights_pct"][name]
                for name in self.names
            }
        return {name: float(value) / 100 for name, value in row.items()}


_AA = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_target(path: str | Path, chain_id: str, pocket: dict) -> dict:
    """Extract an observed protein chain and select pocket residues from a complex.

    The reference ligand is used only for residue selection; it is not included
    in the returned protein sequence or any model context.
    """
    import gemmi
    import numpy as np
    from scipy.spatial.distance import cdist

    path = Path(path).resolve()
    structure = gemmi.read_structure(str(path))
    chain = next((c for c in structure[0] if c.name == chain_id), None) if len(structure) else None
    if chain is None:
        raise ValueError(f"Protein chain {chain_id!r} is missing from {path}")
    residues, sequence = [], []
    for residue in chain:
        name = residue.name.upper()
        if name not in _AA:
            continue
        atoms = [
            np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
            for atom in residue
            if atom.element.name not in ("H", "D")
        ]
        if atoms:
            sequence.append(_AA[name])
            residues.append({
                "sequence_position": len(sequence),
                "author_residue": int(residue.seqid.num),
                "insertion": "" if residue.seqid.icode in (" ", "\x00") else str(residue.seqid.icode).strip(),
                "residue_name": name,
                "coordinates": np.stack(atoms),
            })
    if not sequence:
        raise ValueError("Selected chain has no observed standard amino-acid residues")

    if pocket.get("kind") == "reference_ligand":
        selection = pocket.get("ligand", {})
        ligand_chain = selection.get("chain", chain_id)
        ligand_name = selection.get("residue_name")
        if not ligand_name:
            raise ValueError("Reference ligand needs a residue_name")
        candidates = []
        for candidate_chain in structure[0]:
            if candidate_chain.name != ligand_chain:
                continue
            for residue in candidate_chain:
                if residue.name.upper() != ligand_name.upper():
                    continue
                if selection.get("number") is not None and int(residue.seqid.num) != int(selection["number"]):
                    continue
                insertion = "" if residue.seqid.icode in (" ", "\x00") else str(residue.seqid.icode).strip()
                if selection.get("insertion", "") != insertion:
                    continue
                xyz = [
                    [atom.pos.x, atom.pos.y, atom.pos.z]
                    for atom in residue if atom.element.name not in ("H", "D")
                ]
                if xyz:
                    candidates.append(np.asarray(xyz, dtype=float))
        if len(candidates) != 1:
            raise ValueError(f"Ligand selector matched {len(candidates)} residues; make it unique")
        cutoff = float(pocket.get("cutoff_angstrom", 6.0))
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("Pocket cutoff must be positive and finite")
        ligand_xyz = candidates[0]
        selected = [r for r in residues if cdist(r["coordinates"], ligand_xyz).min() <= cutoff]
        selector = {"kind": "reference_ligand", **selection, "cutoff_angstrom": cutoff}
    elif pocket.get("kind") == "residues":
        lookup = {(r["author_residue"], r["insertion"]): r for r in residues}
        selected = []
        for item in pocket.get("residues", []):
            key = (int(item["number"]), item.get("insertion", ""))
            if key not in lookup:
                raise ValueError(f"Selected protein residue {key} is absent from chain {chain_id}")
            selected.append(lookup[key])
        if not selected or len({r["sequence_position"] for r in selected}) != len(selected):
            raise ValueError("Explicit pocket residues must be nonempty and unique")
        selector = {"kind": "residues", "residues": pocket["residues"]}
    else:
        raise ValueError("Pocket kind must be reference_ligand or residues")
    if not selected:
        raise ValueError("Pocket selection contains no protein residues")
    rows = [
        {k: r[k] for k in ("sequence_position", "author_residue", "insertion", "residue_name")}
        for r in selected
    ]
    return {
        "structure": str(path),
        "structure_sha256": file_sha256(path),
        "chain": chain_id,
        "protein_sequence": "".join(sequence),
        "sequence_policy": "observed standard residues in input-chain order",
        "pocket": selector,
        "residues": rows,
    }
