"""Versioned crystallographic pocket metadata, outside differentiable scoring.

Extraction/geometry audits run only on allocated HPC. Author residue numbers,
one-based input sequence positions and native token IDs are different spaces.
No ligand repair, learned bridge or diffusion derivative is introduced here.
"""

import hashlib
import json
import math
from pathlib import Path

PDB_SHA256 = "534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f"
VERSION = "brd4-jq1-heavy-neighborhood-v1"
THREE_TO_ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_definition(pocket, fixture=None):
    """Require the declared experimental source, selection and anchor identities."""
    if (pocket.get("version") != VERSION or pocket.get("status") != "defined"
        or pocket.get("pdb_id") != "3MXF" or pocket.get("pdb_sha256") != PDB_SHA256
        or pocket.get("protein_chain") != "A" or pocket.get("reference_ligand") != "JQ1"
        or pocket.get("radius_angstrom") != 6.0
        or pocket.get("anchor_author_residues") != [81, 140]
        or pocket.get("anchor_sequence_positions") != [40, 99]
        or pocket.get("pose_requirements") != {"anchor_max_distance_angstrom": 6.0,
            "pocket_max_distance_angstrom": 6.0, "minimum_ligand_fraction": 0.5}):
        raise ValueError("Changed JQ1 pocket definition/protocol")
    rows = pocket["residues"]
    positions = [r["sequence_position"] for r in rows]
    if not rows or positions != sorted(set(positions)) or not {40, 56, 99}.issubset(positions):
        raise ValueError("Incomplete/duplicate JQ1 pocket residue mapping")
    if len(pocket["protein_sequence"]) != 127:
        raise ValueError("Wrong BRD4 construct")
    for r in rows:
        i = r["sequence_position"] - 1
        if (not 0 <= i < 127 or r["author_residue"] != i + 42
            or THREE_TO_ONE.get(r["residue_name"]) != pocket["protein_sequence"][i]
            or not math.isfinite(r["minimum_jq1_distance_angstrom"])
            or not 0 <= r["minimum_jq1_distance_angstrom"] <= 6.0):
            raise ValueError("Author/sequence/amino-acid pocket identity mismatch")
    if fixture is not None and (fixture["protein_sequence"] != pocket["protein_sequence"]
        or fixture["protein_author_residues"] != list(range(42, 169))
        or fixture["pdb_sha256"] != PDB_SHA256):
        raise ValueError("Pocket and fixture differ")
    return pocket


def derive_definition(pdb_path, fixture_path):
    """Select protein heavy atoms <=6A from JQ1; never run in an optimizer."""
    if file_sha(pdb_path) != PDB_SHA256:
        raise ValueError("Experimental 3MXF bytes differ")
    fixture = json.loads(Path(fixture_path).read_text())
    atoms = {}
    for line in Path(pdb_path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in ("ATOM", "HETATM") or line[21:22] != "A":
            continue
        if line[76:78].strip() in ("H", "D"):
            continue
        record, name, residue = line[:6].strip(), line[12:16].strip(), line[17:20].strip()
        number, insertion = int(line[22:26]), line[26:27]
        if insertion.strip():
            raise ValueError("Unexpected insertion code in pinned construct")
        if not (record == "ATOM" and number in fixture["protein_author_residues"] or
                record == "HETATM" and residue == "JQ1"):
            continue
        key = (record, number, residue, name)
        alt = line[16:17]
        rank = (float(line[54:60]), alt == " ", alt == "A")
        coords = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        if not all(math.isfinite(x) for x in coords):
            raise ValueError("Nonfinite crystallographic coordinate")
        if key not in atoms or rank > atoms[key][0]:
            atoms[key] = (rank, coords)
    ligands = {key[1] for key in atoms if key[0] == "HETATM"}
    ligand = [v[1] for k, v in atoms.items() if k[0] == "HETATM"]
    if len(ligands) != 1 or len(ligand) != 31:
        raise ValueError("Require exactly the crystallographic 31-heavy-atom JQ1")
    rows = []
    for position, author in enumerate(fixture["protein_author_residues"], 1):
        selected = [(k, v[1]) for k, v in atoms.items() if k[0] == "ATOM" and k[1] == author]
        names = {k[2] for k, _ in selected}
        if len(names) != 1 or THREE_TO_ONE.get(next(iter(names))) != fixture["protein_sequence"][position - 1]:
            raise ValueError("Crystallographic protein and supplied sequence differ")
        distance = min(math.dist(a, b) for _, a in selected for b in ligand)
        if distance <= 6.0:
            rows.append({"author_residue": author, "sequence_position": position,
                         "residue_name": next(iter(names)), "minimum_jq1_distance_angstrom": distance})
    return validate_definition({"version": VERSION, "status": "defined", "pdb_id": "3MXF",
        "pdb_sha256": PDB_SHA256, "fixture_sha256": file_sha(fixture_path),
        "protein_chain": "A", "protein_sequence": fixture["protein_sequence"],
        "reference_ligand": "JQ1", "reference_ligand_author_residue": next(iter(ligands)),
        "reference_ligand_heavy_atoms": len(ligand), "radius_angstrom": 6.0,
        "selection": "protein heavy atoms within 6A of JQ1 heavy atoms; no solvent/ions",
        "anchor_author_residues": [81, 140], "anchor_sequence_positions": [40, 99],
        "pose_requirements": {"anchor_max_distance_angstrom": 6.0,
            "pocket_max_distance_angstrom": 6.0, "minimum_ligand_fraction": 0.5},
        "residues": rows}, fixture)


def boltz_constraints(pocket):
    validate_definition(pocket)
    return [{"pocket": {"binder": "L", "contacts": [["A", i] for i in pocket["anchor_sequence_positions"]],
                        "max_distance": 6.0, "force": True}}]


def map_protenix_pocket(context, pocket):
    """Authenticate full one-chain native residue metadata before token selection."""
    import numpy as np
    validate_definition(pocket)
    f = context.base_features
    if context.identities.get("protein_sequence") != pocket["protein_sequence"]:
        raise ValueError("Protenix context lacks the authenticated pocket sequence")
    protein = np.asarray(context.protein_token_indices)
    residue = np.asarray(f["residue_index"])[protein]
    asym = np.asarray(f["asym_id"])[protein]
    if len(protein) != 127 or len(set(asym.tolist())) != 1 or not np.array_equal(np.sort(residue), np.arange(1, 128)):
        raise ValueError("Native Protenix protein token/residue mapping differs")
    mapping = dict(zip(residue.tolist(), protein.tolist()))
    return tuple(int(mapping[r["sequence_position"]]) for r in pocket["residues"])


def audit_boltz_pose(tokenized, batch, pocket, ligand_atom_indices):
    """Gate the actual prepared pose, including post-affinity-crop residue IDs.

    Native tokens retain zero-based sequence residue indices. Atom-row offsets
    are rebuilt from token atom counts, rather than treating token IDs as atoms.
    """
    import numpy as np
    validate_definition(pocket)
    coords = batch["coords"].detach().cpu().numpy().reshape(-1, 3)
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite prepared pose")
    mapping, offset, protein_chains = {}, 0, set()
    for ti, token in enumerate(tokenized.tokens):
        count = int(token["atom_num"])
        if int(token["mol_type"]) == 0:
            position = int(token["res_idx"]) + 1
            protein_chains.add(int(token["asym_id"]))
            if position in mapping or not 1 <= position <= 127 or THREE_TO_ONE.get(str(token["res_name"])) != pocket["protein_sequence"][position - 1]:
                raise ValueError("Native Boltz protein residue/sequence mapping differs")
            mapping[position] = {"token": ti, "atoms": list(range(offset, offset + count))}
        offset += count
    pad = batch["atom_pad_mask"].detach().cpu().numpy().reshape(-1)
    if (offset > len(coords) or len(pad) != len(coords) or not np.all(pad[:offset] == 1)
        or np.any(pad[offset:] != 0) or len(protein_chains) != 1):
        raise ValueError("Unexpected native coordinate/crop/chain scope")
    positions = [r["sequence_position"] for r in pocket["residues"]]
    if not set(positions).issubset(mapping):
        raise ValueError("Affinity crop lost a selected pocket residue")
    ligand = coords[np.asarray(ligand_atom_indices, dtype=int)]
    distances = {}
    for position in positions:
        protein = coords[mapping[position]["atoms"]]
        distances[position] = np.linalg.norm(ligand[:, None] - protein[None, :], axis=-1).min(axis=1)
    anchors = {str(i): float(distances[i].min()) for i in pocket["anchor_sequence_positions"]}
    minimum = np.stack(list(distances.values())).min(axis=0)
    fraction = float(np.mean(minimum <= 6.0))
    result = {"version": "brd4-jq1-native-pose-audit-v1", "status": "passed",
        "anchor_minimum_distances_angstrom": anchors, "ligand_fraction_within_6A_of_pocket": fraction,
        "ligand_minimum_pocket_distances_angstrom": minimum.tolist(),
        "native_pocket_token_indices": [mapping[i]["token"] for i in positions],
        "protein_sequence_positions": positions, "coordinates_detached": not batch["coords"].requires_grad,
        "scope": "residue-identified predicted pocket; no ligand RMSD or binding claim"}
    if any(x > 6.0 for x in anchors.values()) or fraction < 0.5 or batch["coords"].requires_grad:
        result["status"] = "failed"
    return result
