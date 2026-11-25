"""Download PDB/mmCIF and resolve author-numbered pockets for a search config.

Requires gemmi for mmCIF parsing. Preparation uses deposited chain A sequence,
including unresolved residues. No structure cropping, mutation or repair occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen

from latentmol.guidance.protenix_pocket import THREE_TO_ONE, VERSION, validate_pocket


def fetch(url, path):
    if not path.exists():
        with urlopen(url, timeout=60) as response, path.open("xb") as output:
            output.write(response.read())
    return path.read_bytes()


def resolve_target(target_id, target, downloads):
    import gemmi

    root = downloads / target["pdb"].lower()
    root.mkdir(parents=True, exist_ok=True)
    pdb = target["pdb"]
    raw = fetch(f"https://files.rcsb.org/download/{pdb}.pdb", root / f"{pdb}.pdb")
    cif = fetch(f"https://files.rcsb.org/download/{pdb}.cif", root / f"{pdb}.cif")
    block = gemmi.cif.read_string(cif.decode()).sole_block()
    scheme = block.find("_pdbx_poly_seq_scheme.", ["seq_id", "mon_id", "pdb_seq_num", "pdb_strand_id", "pdb_ins_code", "entity_id"])
    rows = [list(row) for row in scheme if row[3] == "A"]
    if [int(row[0]) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"Nonunique chain A sequence mapping: {pdb}")
    if any(row[4] not in (".", "?") for row in rows):
        raise ValueError(f"Handle author insertion codes explicitly before searching {pdb}")
    entities = {str(row[0]): gemmi.cif.as_string(row[1]).replace("\n", "").replace(" ", "")
                for row in block.find('_entity_poly.', ['entity_id', 'pdbx_seq_one_letter_code_can'])}
    sequence = entities[rows[0][5]]
    if len(sequence) != len(rows) or any(THREE_TO_ONE.get(row[1], sequence[i]) != sequence[i]
                                       for i, row in enumerate(rows)):
        raise ValueError(f"Canonical and standard residue sequences disagree: {pdb}")
    substitutions = [dict(author_residue=int(row[2]), deposited_component=row[1],
                          canonical_amino_acid=sequence[i]) for i, row in enumerate(rows)
                     if row[1] not in THREE_TO_ONE]
    numbering = [int(row[2]) if row[2] not in (".", "?") else None for row in rows]
    protein, ligand = {}, {}
    for line in raw.decode().splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in ("ATOM", "HETATM") or line[21] != "A" or line[76:78].strip() in ("H", "D"):
            continue
        number, name = int(line[22:26]), line[17:20].strip()
        point = tuple(float(line[i:i+8]) for i in (30, 38, 46))
        if not all(map(math.isfinite, point)):
            raise ValueError("Nonfinite crystallographic coordinate")
        if line[:6].strip() == "ATOM":
            protein.setdefault((number, name), []).append(point)
        elif number == target["ligand_residue"] and name == target["ligand"]:
            ligand.setdefault(line[12:16].strip(), point)
    if not ligand:
        raise ValueError(f"Reference ligand missing at exact locator: {pdb}")
    pocket_rows = []
    for label in target["pocket"].split():
        name, number = label[:3], int(label[3:])
        coords = protein.get((number, name))
        matches = [i + 1 for i, row in enumerate(rows) if row[1] == name and numbering[i] == number]
        if not coords or len(matches) != 1:
            raise ValueError(f"Pocket residue mapping missing/ambiguous: {pdb} {label}")
        pocket_rows.append(dict(author_residue=number, residue_name=name, sequence_position=matches[0],
            minimum_reference_distance_angstrom=min(math.dist(a,b) for a in coords for b in ligand.values())))
    pocket = dict(version=VERSION, status="defined", pdb_id=pdb,
        pdb_sha256=hashlib.sha256(raw).hexdigest(), mmcif_sha256=hashlib.sha256(cif).hexdigest(),
        protein_chain="A", protein_sequence=sequence, protein_author_residues=numbering,
        reference_ligand_locator=dict(component=target["ligand"], chain="A", residue=target["ligand_residue"]),
        reference_ligand_heavy_atoms=len(ligand), selection="Exact user-supplied residue list",
        residues=sorted(pocket_rows, key=lambda row: row["sequence_position"]))
    validate_pocket(pocket)
    result = dict(id=target_id, name=target["name"], initialized_atoms=target["atoms"], pocket=pocket,
                  source_url=f"https://www.rcsb.org/structure/{pdb}",
                  context_scope="Native Boltz2 crystallographic chain A plus reference ligand; fixed pose, native affinity crop, no diffusion",
                  structure_path=f"inputs/restart-targets/{pdb}.cif", sequence_only_canonicalization_applied=False,
                  modified_residues=substitutions)
    print(json.dumps(dict(id=target_id, sequence_length=len(sequence), reference_heavy_atoms=len(ligand),
                          initialized_atoms=target["atoms"], pocket_positions=[r["sequence_position"] for r in pocket_rows])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    template = json.loads(args.catalog.with_name("restart-search-lr60-short.json").read_text())
    template.pop("parent_cohorts", None)
    template["output_cap_mib"] = 512
    template["scorer"] = dict(kind="boltz2_dbmol", assets_config="restart-boltz-dtu.json")
    template["budget_status"] = "user_selected_200_updates_resume_required"
    template["design"].update(updates=200, schedule_horizon=200, chemistry_hold=150, chemistry_ramp_end=200,
                              noise_initial=0.0, noise_zero_at=100, seed=271101, learning_rate=1.0)
    template["search"].update(initialization="bootstrap", bootstrap_branches=12, parents=3, children=4,
        waves=6, parent_policy="top", max_similarity=1.0, max_ancestor_fraction=None,
        fresh_per_branch=6, diversity_limit=20, diversity_max_similarity=1.0, audit_chemistry=True,
        time_budget_seconds=23*3600, hard_scoring="fixed_pose_dbmol")
    # Weak diversity: exact molecule deduplication only; keep ranking by score.
    template["perturbation_seed"] = 272101
    template["prior_seed"] = 273101
    template["arms"] = {key: dict(target=resolve_target(key, value, args.downloads),
        queue=value["queue"], design=dict(occupied_atoms=value["atoms"], seed=271101 + 100*index))
        for index, (key, value) in enumerate(catalog["targets"].items())}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(template, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
