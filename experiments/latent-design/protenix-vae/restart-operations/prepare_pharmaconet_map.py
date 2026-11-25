"""Export ligand-side density nodes from the pretrained BRD4 PharmacoNet model."""

import argparse
import hashlib
import importlib
import json
import math
import os
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import subprocess
import types
from unittest.mock import patch


SOURCE_REVISION = "0f2feec4e398b618fa04ce6eca8f86d2855160f4"
WEIGHT_SHA = "6e7633d767b3c07ec3d00dc6b782358cde48b273d0f8021e2c6e8f3553be9f1a"
FEATURES = {"HBond_acceptor": "acceptor", "HBond_donor": "donor",
            "Hydrophobic": "hydrophobic", "Aromatic": "aromatic"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def position(line):
    return [float(line[i:i + 8]) for i in (30, 38, 46)]


def reference_geometry(pdb, receptor):
    native = {}
    ligand = {}
    for line in pdb.read_text().splitlines():
        if line[21:22] != "A" or line[16:17] not in (" ", "A") or line[76:78].strip() == "H":
            continue
        if line.startswith("ATOM  "):
            native[line[12:16].strip(), line[17:20], line[22:27]] = position(line)
        elif line.startswith("HETATM") and line[17:20] == "JQ1" and line[22:27].strip() == "1":
            ligand[line[12:16].strip()] = position(line)
    if len(ligand) != 31:
        raise ValueError("Expected 31 native JQ1 chain A residue 1 heavy atoms")
    count = 0
    max_delta = 0.
    total = 0
    receptor_atoms = []
    for line in receptor.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[21:22] != "A" or not line.startswith("ATOM  "):
            raise ValueError("Model receptor must contain protein chain A only")
        total += 1
        key = line[12:16].strip(), line[17:20], line[22:27]
        xyz = position(line)
        receptor_atoms.append((xyz, f"{line[17:20]} {line[21]}{line[22:27].strip()} {key[0]}"))
        if key in native:
            count += 1
            distances = [math.dist(native[key], xyz)]
            # Receptor preparation exchanges equivalent C-terminal oxygen names.
            if key[0] in ("O", "OXT") and ("OXT", *key[1:]) in native:
                distances.extend(math.dist(native[(name, *key[1:])], xyz) for name in ("O", "OXT"))
            max_delta = max(max_delta, min(distances))
    if count < .95 * total or total == 0 or max_delta > .01:
        raise ValueError(f"Receptor frame check failed: {count}/{total}, max difference {max_delta} Å")
    return ligand, receptor_atoms, dict(receptor_atoms=total, native_matched_atoms=count,
                                        max_difference_angstrom=max_delta, terminal_oxygens_exchangeable=True)


def checked_openbabel(executable):
    # The pinned upstream parser embeds Unix stderr redirection in os.system.
    # Run the same conversion as an argument list on Windows (also works on Unix).
    def convert(command):
        prefix, suffix = "obabel ", " -d 2>/dev/null"
        if not command.startswith(prefix) or not command.endswith(suffix):
            raise ValueError("Unexpected PharmacoNet pocket extraction command")
        source, destination = command[len(prefix):-len(suffix)].split(" -O ")
        if source != destination:
            raise ValueError("Unexpected PharmacoNet conversion path")
        result = subprocess.run([str(executable), source, "-O", destination, "-d"], capture_output=True,
                                text=True, check=True)
        if "1 molecule converted" not in result.stderr or not Path(destination).stat().st_size:
            raise ValueError(f"Open Babel did not convert the receptor: {result.stderr}")
        return result.returncode
    upstream = importlib.import_module("pmnet.data.extract_pocket")
    return patch.object(upstream, "os", types.SimpleNamespace(system=convert))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("pdb", "receptor", "box", "weights", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--obabel", type=Path, help="Explicit upstream Open Babel executable")
    parser.add_argument("--hbond-only", action="store_true", help="Restrict the exported native nodes to H-bonds")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.weights) != WEIGHT_SHA:
        raise ValueError("Expected the pinned official PharmacoNet checkpoint")
    box = json.loads(args.box.read_text())
    if digest(args.pdb) != box["pdb_sha256"]:
        raise ValueError("Reference PDB differs from the docking box source")
    ligand, receptor_atoms, frame_check = reference_geometry(args.pdb, args.receptor)
    center = tuple(box["center"])

    import torch
    from pmnet.module import PharmacoNet
    from pmnet.data.objects import Protein
    torch.set_num_threads(4)
    model = PharmacoNet(device="cpu", verbose=False, molvoxel_library="numpy", weight_path=args.weights)
    conversion = checked_openbabel(args.obabel) if args.obabel else nullcontext()
    with torch.inference_mode(), conversion:
        cloud = model.run(args.receptor, center=center)
    protein = Protein.from_pdbfile(str(args.receptor))
    nodes = []
    for node in cloud.nodes:
        if args.hbond_only and node.type not in ("HBond_acceptor", "HBond_donor"):
            continue
        xyz = [float(v) for v in node.center]
        partner = [float(v) for v in node.hotspot_position]
        if not all(math.isfinite(v) for v in [*xyz, *partner, node.radius, node.score]) or node.radius <= 0:
            raise ValueError("Invalid predicted interaction node geometry")
        row = dict(id=f"pmnet-{node.index:03d}", feature=FEATURES[node.type], center=xyz,
                   radius=float(node.radius), score=float(node.score), protein_position=partner,
                   interaction_type=node.interaction_type)
        if node.interaction_type in ("PiStacking_P", "PiStacking_T"):
            ring = min(protein.rings_all, key=lambda r: math.dist(tuple(r.center), partner))
            distance = math.dist(tuple(ring.center), partner)
            res = ring.obatoms[0].GetResidue()
            row.update(protein_atom=f"{res.GetName()} {res.GetChain()}{res.GetNum()} ring",
                       protein_normal=[float(v) for v in ring.normal],
                       protein_ring_atoms=[res.GetAtomID(a).strip() for a in ring.obatoms],
                       protein_ring_coordinates=[[a.GetX(), a.GetY(), a.GetZ()] for a in ring.obatoms])
        else:
            distance, row["protein_atom"] = min((math.dist(partner, point), label) for point, label in receptor_atoms)
        if distance > .01:
            raise ValueError(f"Predicted protein-side interaction has no match in receptor: {distance}")
        nodes.append(row)
    if not nodes or len({r["id"] for r in nodes}) != len(nodes):
        raise ValueError("Missing or duplicate density nodes")
    result = dict(model="PharmacoNet", pdb_id="3MXF", chain="A", frame="deposited 3MXF coordinates in angstrom",
                  reference_ligand="JQ1 A 1", reference_atoms=ligand, hotspots=nodes,
                  source_revision=SOURCE_REVISION, source_url="https://github.com/SeonghwanSeo/PharmacoNet",
                  checkpoint_sha256=WEIGHT_SHA,
                  checkpoint_url="https://drive.google.com/file/d/1gzjdM7bD3jPm23LBcDXtkSk18nETL04p/view",
                  pdb_sha256=box["pdb_sha256"], receptor_sha256=digest(args.receptor), box_sha256=digest(args.box),
                  box_center=center, frame_check=frame_check, model_score_thresholds=model.score_threshold,
                  all_density_node_counts=dict(Counter(n.type for n in cloud.nodes)),
                  feature_scope=("Hydrogen bond nodes only" if args.hbond_only else "All native ligand feature nodes"),
                  radius_definition="Sphere-equivalent radius derived from each density component's voxel count.",
                  score_definition="Native relative score percentile within interaction type.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(path=os.fspath(args.output), nodes=len(nodes), frame=frame_check,
                          features=dict(Counter(row["feature"] for row in nodes)), sha256=digest(args.output))))


if __name__ == "__main__":
    main()
