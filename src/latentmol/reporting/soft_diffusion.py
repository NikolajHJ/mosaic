"""Render saved soft-diffusion coordinates as JSON or a standalone HTML report."""

import argparse
import json
from pathlib import Path

import numpy as np


def align_to_protein(coordinates, reference, protein_mask):
    """Apply one proper rigid fit using receptor atoms; never fit the ligand."""
    if protein_mask.sum() < 3:
        raise ValueError("Need at least three protein atoms for a common frame")
    mobile = coordinates[protein_mask]
    target = reference[protein_mask]
    source_center, target_center = mobile.mean(0), target.mean(0)
    left, _, right = np.linalg.svd((mobile-source_center).T @ (target-target_center))
    rotation = left @ np.diag([1, 1, np.linalg.det(left @ right)]) @ right
    return (coordinates-source_center) @ rotation + target_center


def build_payload(root, *, all_estimates=False):
    result = json.loads((root / "result.json").read_text())
    graph = json.loads((root / "soft_graph.json").read_text())
    if result["status"] != "completed":
        raise ValueError("Native prediction did not complete")
    with np.load(root / "coordinates.npz", allow_pickle=False) as saved:
        predicted = saved["predicted"][0].astype(np.float64)
        reference = saved["reference"][0].astype(np.float64)
        denoised = saved["denoised_estimates"].astype(np.float64)
        protein = saved["protein_atom_mask"].astype(bool)
        slot_map = saved["slot_to_atom"]
    if (predicted.shape != reference.shape or predicted.ndim != 2
            or predicted.shape[1] != 3 or denoised.shape[1:] != predicted.shape
            or protein.shape != predicted.shape[:1]
            or not all(np.isfinite(x).all() for x in (predicted, reference, denoised))):
        raise ValueError("Invalid saved native coordinates")
    slots = np.flatnonzero(slot_map >= 0)
    atom_rows = slot_map[slots]
    if (graph["active_slots"] != slots.tolist() or not len(slots)
            or (atom_rows >= len(predicted)).any() or len(set(atom_rows)) != len(slots)):
        raise ValueError("Coordinate and probability slot correspondence differs")
    fitted = align_to_protein(predicted, reference, protein)
    ligand_reference, ligand_final = reference[atom_rows], fitted[atom_rows]
    together = np.concatenate((ligand_reference, ligand_final))
    center = together.mean(0)
    _, _, basis = np.linalg.svd(together-center, full_matrices=False)
    atoms = np.asarray(graph["atoms"])[slots].tolist()
    charges = np.asarray(graph["charges"])[slots].tolist()
    bonds = np.asarray(graph["bonds"])[np.ix_(slots, slots)].tolist()
    snapshots = []

    def frame(label, coordinates, kind):
        ligand = coordinates[atom_rows]
        snapshots.append(dict(label=label, kind=kind, step=graph["step"], atoms=atoms,
                              charges=charges, bonds=bonds,
                              coordinates_angstrom=ligand.tolist(),
                              camera_coordinates_angstrom=((ligand-center) @ basis.T).tolist()))

    frame("Fixed reference", reference, "conditioning_reference")
    steps = range(1, len(denoised) + 1) if all_estimates else (10, 20, 40)
    for step in steps:
        if step <= len(denoised):
            frame(f"Estimate {step}", align_to_protein(denoised[step-1], reference, protein), "native_denoised_estimate")
    frame("Final prediction", fitted, "native_final_sample")
    rmsd = lambda left, right: float(np.sqrt(np.mean(np.sum((left-right)**2, axis=-1))))
    strong = []
    for i in range(len(slots)):
        for j in range(i+1, len(slots)):
            probability = 1-bonds[i][j][0]
            if probability > .5:
                strong.append(dict(slots=[int(slots[i]), int(slots[j])], probability=probability,
                    reference_distance_angstrom=float(np.linalg.norm(ligand_reference[i]-ligand_reference[j])),
                    predicted_distance_angstrom=float(np.linalg.norm(ligand_final[i]-ligand_final[j]))))
    return dict(run_id="soft-diffusion-20261103-attempt001", model="Boltz-2", slots=slots.tolist(),
                   graph_spec=graph["graph_spec"], snapshots=snapshots, default_state=len(snapshots)-1,
                   geometry_scope="New native diffusion prediction from continuous soft features, rigidly aligned using protein atoms for display",
                   diagnostics=dict(ligand_rmsd_after_protein_fit_angstrom=rmsd(ligand_final, ligand_reference),
                                    receptor_rmsd_angstrom=rmsd(fitted[protein], reference[protein]),
                                    strong_bond_distances=strong),
                   native_result=result)


def assemble(root, output):
    payload = build_payload(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps(dict(output=str(output), snapshots=len(payload["snapshots"]), diagnostics=payload["diagnostics"])))


def report_case(root, *, label="Saved Boltz ligand", source=None, input_policy=None):
    payload = build_payload(root, all_estimates=True)
    result = payload["native_result"]
    graph = payload["snapshots"][0]
    frames = [{"label": frame["label"], "kind": frame["kind"],
               "points": frame["camera_coordinates_angstrom"]}
              for frame in payload["snapshots"]]
    spec = payload["graph_spec"]
    slots = payload["slots"]
    if len(slots) < 2 or len(frames) < 3:
        raise ValueError("The report needs at least two atom slots and one denoising estimate")
    atoms = np.asarray(graph["atoms"])
    charges = np.asarray(graph["charges"])
    bonds = np.asarray(graph["bonds"])
    for values, shape in ((atoms, (len(slots), len(spec["atomic_numbers"]))),
                          (charges, (len(slots), len(spec["formal_charges"]))),
                          (bonds, (len(slots), len(slots), len(spec["bond_names"])))):
        if (values.shape != shape or not np.isfinite(values).all()
                or (values < 0).any() or (values > 1).any()
                or not np.allclose(values.sum(axis=-1), 1, atol=2e-6, rtol=0)):
            raise ValueError("Malformed saved probabilities for the 3D report")
    return dict(label=label, source=source or f"Saved VAE soft graph · step {graph['step']}",
                input_policy=input_policy or "Saved fixed-occupancy graph.", slots=slots, frames=frames,
                atoms=graph["atoms"], charges=graph["charges"], bonds=graph["bonds"], graph_spec=spec,
                graph_step=graph["step"],
                run={key: result[key] for key in ("job_id", "seed", "sampling_steps", "device", "boltz_commit")},
                ligand_rmsd=payload["diagnostics"]["ligand_rmsd_after_protein_fit_angstrom"])


def _write_report(cases, default, output):
    template = Path(__file__).with_name("assets") / "soft_diffusion_report.html"
    data = dict(molecules=cases, default=default)
    embedded = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    document = template.read_text(encoding="utf-8").replace("__SOFT_DIFFUSION_DATA__", embedded)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return output


def render_report(root, output):
    return _write_report([report_case(root)], 0, output)


def render_gallery(manifest, output):
    record = json.loads(manifest.read_text(encoding="utf-8"))
    entries = record["molecules"]
    default = record.get("default", 0)
    if not isinstance(default, int) or not 0 <= default < len(entries):
        raise ValueError("Gallery needs a valid default molecule index")
    cases = []
    seen = set()
    for entry in entries:
        root = (manifest.parent / entry["prediction"]).resolve()
        if root in seen:
            raise ValueError("Duplicate prediction directory in gallery")
        seen.add(root)
        case = report_case(root, label=entry["label"], source=entry["source"],
                           input_policy=entry.get("input_policy"))
        if entry.get("count", len(case["slots"])) != len(case["slots"]):
            raise ValueError(f"Gallery slot count differs from the saved prediction: {entry['label']}")
        cases.append(case)
    return _write_report(cases, default, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", type=Path)
    source.add_argument("--gallery", type=Path, help="Manifest of saved prediction directories and source labels")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Write the sampled-frame JSON view")
    destination.add_argument("--html", type=Path, help="Write a standalone interactive report with every saved frame")
    args = parser.parse_args()
    if args.gallery:
        if not args.html:
            parser.error("--gallery requires --html")
        print(render_gallery(args.gallery, args.html))
    elif args.html:
        print(render_report(args.prediction, args.html))
    else:
        assemble(args.prediction, args.output)
