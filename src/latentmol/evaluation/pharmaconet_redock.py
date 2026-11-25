"""Measure eligible hard ligand features in the same crystallographic frame before and after docking."""

import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem

from latentmol.chemistry import _label_molecule
from latentmol.experiments import file_sha256
from latentmol.guidance.pharmaconet import hard_feature_groups
from latentmol.representation import GraphSpec


def checked_frame(map_file, source, cfg, founder):
    digest = file_sha256(map_file)
    if digest != cfg["pharmaconet_map"]["sha256"] or digest != source["pharmaconet_map_sha256"]:
        raise ValueError("Optimization and readout PharmacoNet maps differ")
    field = json.loads(map_file.read_text())
    box = json.loads(Path(cfg["vina_parent_panel"]["box"]["path"]).read_text())
    if (field["pdb_id"] != cfg["scorer"]["pocket_pdb_id"] or field["chain"] != cfg["chain"] or
            field["pdb_sha256"] != box["pdb_sha256"]):
        raise ValueError("The docking receptor box and the PharmacoNet map use different source structures")
    names = founder["reference_atom_names"]
    if len(set(names)) != len(names) or set(names) != set(field["reference_atoms"]):
        raise ValueError("PharmacoNet crystal coordinates do not cover the design slots")
    xyz = np.asarray([field["reference_atoms"][name] for name in names], dtype=float)
    if (xyz.shape != (len(founder["active_slots"]), 3) or not np.isfinite(xyz).all() or
            not np.allclose(xyz, founder["pharmaconet_reference_coordinates"], atol=.002, rtol=0) or
            not np.allclose((xyz.max(0) + xyz.min(0)) / 2, box["center"], atol=.002, rtol=0)):
        raise ValueError("Fixed optimization slots and independent redocking are in different frames")
    return field, xyz


def fixed_molecule(record, spec, active, expected_smiles):
    atoms, charges, bonds = (np.asarray(record[key], dtype=int) for key in ("atoms", "charges", "bonds"))
    if list(np.flatnonzero(atoms)) != active:
        raise ValueError("Hard scoring record changed reference atom occupancy")
    molecule = _label_molecule(atoms, charges, bonds[np.ix_(active, active)], spec, active)
    Chem.SanitizeMol(molecule)
    if Chem.MolToSmiles(molecule, isomericSmiles=False) != expected_smiles:
        raise ValueError("The archived slot graph and scored molecule differ")
    return molecule


def receptor_hydrogens(hotspots, receptor_pdbqt):
    """Recover actual receptor donor hydrogens from the same rigid Vina receptor."""
    lines = Path(receptor_pdbqt).read_text().splitlines()
    hydrogens = np.asarray([[float(line[i:i+8]) for i in (30, 38, 46)] for line in lines
                            if line.startswith("ATOM") and line.split()[-1] in {"HD", "H"}]).reshape(-1, 3)
    result = []
    for site in hotspots:
        site = dict(site)
        if site["feature"] == "acceptor":
            attached = np.linalg.norm(hydrogens - site["protein_position"], axis=1) <= 1.25
            site["donor_hydrogens"] = hydrogens[attached].tolist()
            if not site["donor_hydrogens"]:
                raise ValueError(f"Receptor donor for hotspot {site['id']} has no identifiable hydrogen")
        result.append(site)
    return result


def _angle(u, v):
    length = np.linalg.norm(u) * np.linalg.norm(v)
    return math.degrees(math.acos(np.clip(np.dot(u, v) / length, -1., 1.))) if length > 1e-8 else None


def _geometry(site, points):
    center = points.mean(0)
    delta = center - site["protein_position"]
    distance = float(np.linalg.norm(delta))
    detail = dict(distance_to_protein_A=round(distance, 4))
    role = site["feature"]
    if role == "hydrophobic":
        contact = 2.5 <= distance <= 4.
    elif role in {"acceptor", "donor"}:
        contact = 2.4 <= distance <= 3.5
        if "donor_hydrogens" in site:
            angles = [_angle(np.subtract(site["protein_position"], h), center - h)
                      for h in site["donor_hydrogens"]]
            angle = max((a for a in angles if a is not None), default=None)
            detail["best_donor_H_acceptor_angle_deg"] = round(angle, 3) if angle is not None else None
            contact = contact and angle is not None and angle >= 120.
    elif role == "aromatic":
        # A ring center alone cannot establish the indicated type of pi stacking.
        protein = np.asarray(site["protein_ring_coordinates"], dtype=float)
        if protein.ndim != 2 or protein.shape[1] != 3 or len(protein) < 5 or not np.isfinite(protein).all():
            raise ValueError(f"Invalid protein ring coordinates for {site['id']}")
        normals = []
        for ring in (points, protein):
            _, values, axes = np.linalg.svd(ring - ring.mean(0), full_matrices=False)
            if values[1] < 1e-6:
                return dict(**detail, contact=False, reason="degenerate ring geometry")
            normals.append(axes[-1])
        angle = _angle(*normals)
        angle = min(angle, 180.-angle)
        offset = min(np.linalg.norm(delta - np.dot(delta, n)*n) for n in normals)
        detail.update(plane_angle_deg=round(angle, 3), min_lateral_offset_A=round(float(offset), 4))
        t = site["interaction_type"]
        if t not in {"PiStacking_P", "PiStacking_T"}:
            raise ValueError(f"Unsupported ring contact {t}")
        contact = 2.8 <= distance <= 5.5 and offset <= 2.0 and (angle <= 30. if t == "PiStacking_P" else angle >= 60.)
    else:
        raise ValueError(f"Unsupported PharmacoNet role {role}")
    return dict(**detail, contact=bool(contact))


def measure(mol, xyz, hotspots):
    """All complementary feature atoms at each pocket hotspot; coordinates are never aligned."""
    xyz = np.asarray(xyz, dtype=float)
    if xyz.shape != (mol.GetNumAtoms(), 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected finite heavy-atom coordinates for every ligand atom")
    eligible = hard_feature_groups(mol)
    # Canonical equivalence classes let the readout distinguish a different
    # donor/acceptor serving the same hotspot from a symmetry-equivalent atom.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    measurements = {}
    for site in hotspots:
        hits, partner_hits = [], []
        for group in sorted(eligible[site["feature"]]):
            points = xyz[list(group)]
            to_center = float(np.linalg.norm(points.mean(0) - site["center"]))
            row = dict(atom_indices=list(group), symmetry_classes=sorted(ranks[i] for i in group),
                       distance_to_center_A=round(to_center, 4), **_geometry(site, points))
            if to_center <= site["radius"]:
                hits.append(row)
            if row["contact"]:
                partner_hits.append(row)
        measurements[site["id"]] = dict(feature=site["feature"], protein_atom=site.get("protein_atom"),
                                        in_hotspot=bool(hits), contact=any(h["contact"] for h in hits), hits=hits,
                                        partner_contact=bool(partner_hits), partner_hits=partner_hits)
    return measurements


def compare(before, after):
    def ids(measurement, key):
        return {site for site, row in measurement.items() if row[key]}

    a, b = ids(before, "contact"), ids(after, "contact")
    prior, post = ids(before, "partner_contact"), ids(after, "partner_contact")
    shared = a & b
    def classes(row):
        return {tuple(h["symmetry_classes"]) for h in row["hits"] if h["contact"]}
    same = [site for site in shared if classes(before[site]) & classes(after[site])]
    return dict(designed_contacts=sorted(a), redocked_contacts=sorted(b), retained_contacts=sorted(shared),
                retained_with_equivalent_ligand_atom=sorted(same), lost_contacts=sorted(a-b),
                designed_partner_contacts=sorted(prior), redocked_partner_contacts=sorted(post),
                retained_partner_contacts=sorted(prior & post), lost_partner_contacts=sorted(prior - post),
                gained_partner_contacts=sorted(post - prior),
                retained_protein_partners=[dict(hotspot=i, protein_atom=before[i]["protein_atom"],
                                                role=before[i]["feature"]) for i in sorted(prior & post)],
                gained_contacts=sorted(b-a), retained_hotspot_overlaps=sorted(ids(before, "in_hotspot") & ids(after, "in_hotspot")))


def summarize(rows, updates, roles):
    observed = [r for r in rows if r["status"] == "evaluated"]
    hits = sum(bool(r["comparison"]["retained_contacts"]) for r in observed)
    before = sum(sum(h["contact"] for h in r["before"].values()) for r in observed)
    retained = sum(len(r["comparison"]["retained_contacts"]) for r in observed)
    by_role = {}
    for role in sorted(set(roles.values())):
        ids = {key for key in roles if roles[key] == role}
        keys = ["designed_contacts", "redocked_contacts", "retained_contacts", "retained_hotspot_overlaps",
                "designed_partner_contacts", "redocked_partner_contacts", "retained_partner_contacts"]
        by_role[role] = {k: sum(len(ids.intersection(r["comparison"][k])) for r in observed) for k in keys}
        by_role[role]["distinct_molecules_with_retained_contact"] = sum(
            bool(ids.intersection(r["comparison"]["retained_contacts"])) for r in observed)
    return dict(selected=len(rows), evaluated=len(observed), missing_pose=len(rows)-len(observed), by_family=by_role,
                with_designed_contact=sum(any(h["contact"] for h in r["before"].values()) for r in rows),
                with_redocked_contact=sum(bool(r["comparison"]["redocked_contacts"]) for r in observed),
                with_retained_partner_contact=sum(bool(r["comparison"]["retained_partner_contacts"]) for r in observed),
                with_retained_contact=hits, with_same_site_and_equivalent_atom=sum(
                    bool(r["comparison"]["retained_with_equivalent_ligand_atom"]) for r in observed),
                retained_molecules_per_1000_updates=1000*hits/updates,
                retained_contacts=retained, designed_contacts_among_evaluated=before,
                contact_retention_fraction=retained/before if before else None)


def add_hotspot_readout(result, source, cfg, run, output):
    founder = json.loads((run / "founder.json").read_text())
    field, xyz = checked_frame(run / "pharmaconet-map.json", source, cfg, founder)
    hotspots = receptor_hydrogens(field["hotspots"], cfg["vina_parent_panel"]["receptor"]["path"])
    spec = GraphSpec.from_dict(cfg["graph_spec"])
    post_cache = {}

    def interaction(row, record):
        smi = row["chemistry"]["smiles"]
        before_mol = fixed_molecule(record, spec, founder["active_slots"], smi)
        before = measure(before_mol, xyz, hotspots)
        vina = row["vina"]
        if "pose_sdf" not in vina:
            return dict(status="missing_pose", before=before,
                        reason=vina.get("pose_export_error") or vina.get("reason") or "Unavailable Vina coordinates")
        if smi not in post_cache:
            path = output / "poses" / vina["pose_sdf"]
            try:
                molecule = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)), None)
                if molecule is None or Chem.MolToSmiles(molecule, isomericSmiles=False) != smi:
                    raise ValueError("Exported SDF does not reproduce the archive graph")
                post_cache[smi] = measure(molecule, molecule.GetConformer().GetPositions(), hotspots)
            except (OSError, ValueError, RuntimeError, IndexError) as exc:
                return dict(status="missing_pose", before=before, reason=str(exc))
        after = post_cache[smi]
        return dict(status="evaluated", before=before, after=after, comparison=compare(before, after))

    result["founder"]["interactions"] = interaction(result["founder"], founder["hard_graph"])
    for name, variant in result["variants"].items():
        by_key = {row["scoring_key"]: row for row in source["variants"][name]["top20"]}
        for row in variant["selected"]:
            row["interactions"] = interaction(row, by_key[row["scoring_key"]]["hard_graph"])
        variant["interaction_summary"] = summarize(
            [row["interactions"] for row in variant["selected"]], source["variants"][name]["updates"],
            {site["id"]: site["feature"] for site in hotspots})
    result["interaction_definition"] = dict(
        map_sha256=source["pharmaconet_map_sha256"], hotspots=hotspots,
        scope="Same PharmacoNet hotspot ID and complementary hard ligand feature in the original PDB frame.",
        overlap="Ligand feature atom or aromatic ring centroid within the native ligand-side hotspot radius.",
        contact="Overlap plus family-specific geometry in angstroms. H-bond range 2.4–3.5; for receptor donors also D-H-A angle >=120 degrees. Hydrophobic atom-pair range 2.5–4.0. Ring centroids 2.8–5.5, smaller lateral offset <=2 and normals within 30 degrees of the predicted parallel or perpendicular orientation.",
        partner_contact="The same complementary ligand feature class and geometric thresholds applied to the named receptor atom or ring anywhere around that partner, independently of the predicted ligand-side hotspot radius.",
        same_atom="Secondary measure requires an overlapping ligand atom equivalence class (stereo-free symmetry ranks).",
        missing="Docking or export failures have unknown post-dock contacts and contribute zero recovered molecules per update.")
