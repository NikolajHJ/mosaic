"""Inspect saved soft chemistry on its authenticated, fixed Boltz coordinates.

This command reads saved artifacts only; it never runs a decoder or a scorer.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(run, context, arm, steps, output):
    record = json.loads((run / "run.json").read_text())
    raw = torch.load(context, map_location="cpu", weights_only=True)
    slot_map = raw["slot_to_atom"].numpy()
    slots = np.flatnonzero(slot_map >= 0)
    asset = record["config"]["contexts"][str(len(slots))]
    if sha256(context) != asset["sha256"]:
        raise ValueError("Coordinate snapshot does not match the recorded run context")
    if len(set(slot_map[slots])) != len(slots):
        raise ValueError("Duplicate native atoms in slot map")
    all_xyz = raw["coordinates"].numpy().reshape(-1, 3)
    if (slot_map[slots] >= len(all_xyz)).any() or not np.isfinite(all_xyz).all():
        raise ValueError("Invalid coordinates or slot map")
    xyz = all_xyz[slot_map[slots]].astype(np.float64)
    centered = xyz - xyz.mean(axis=0)
    # A rigid camera basis only: no distance or geometry changes.
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    view_xyz = centered @ basis.T
    snapshots, sources = [], []
    for step in steps:
        path = run / arm / f"probabilities-{step:04d}.json"
        saved = json.loads(path.read_text())
        if saved["snapshot_version"] != "shared-relaxed-graph-v1":
            raise ValueError("Unsupported saved probability snapshot")
        if saved["graph_spec"] != raw["graph_spec"] or saved["active_slots"] != slots.tolist():
            raise ValueError("Snapshot schema or active-slot mapping differs")
        rows = saved["row_slots"]
        if len(set(rows)) != len(rows) or not set(slots).issubset(rows):
            raise ValueError("Missing/duplicate saved probability row slots")
        indices = [rows.index(int(slot)) for slot in slots]
        n = len(rows)
        arrays = {key: np.asarray(saved[key], dtype=np.float64) for key in ("atoms", "charges", "bonds")}
        spec = raw["graph_spec"]
        shapes = {"atoms": (n, len(spec["atomic_numbers"])),
                  "charges": (n, len(spec["formal_charges"])),
                  "bonds": (n, n, len(spec["bond_names"]))}
        for key, values in arrays.items():
            if values.shape != shapes[key] or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} shape or nonfinite probabilities")
            if (values < -1e-6).any() or (values > 1 + 1e-6).any() or not np.allclose(values.sum(-1), 1, atol=2e-6):
                raise ValueError(f"Invalid {key} probability simplex")
        if not np.allclose(arrays["bonds"], arrays["bonds"].transpose(1, 0, 2), atol=2e-6):
            raise ValueError("Asymmetric bond probabilities")
        snapshots.append({"step": step, "temperature": saved["temperature"],
                          "atoms": arrays["atoms"][indices].round(7).tolist(),
                          "charges": arrays["charges"][indices].round(7).tolist(),
                          "bonds": arrays["bonds"][np.ix_(indices, indices)].round(7).tolist()})
        sources.append({"path": str(path.resolve()), "sha256": sha256(path)})
    summary = json.loads((run / "summary.json").read_text())["arms"][arm]
    if summary["context_switches"] != 0:
        raise ValueError("This inspection requires one fixed context throughout")
    result = {"run_id": record["run_id"], "arm": arm, "graph_spec": raw["graph_spec"],
              "slots": slots.tolist(), "coordinates_angstrom": xyz.tolist(),
              "camera_coordinates_angstrom": view_xyz.tolist(), "snapshots": snapshots,
              "geometry_scope": "Fixed detached carbon-chain reference pose with saved soft chemistry overlaid; no diffusion or energy minimization",
              "original_run_status": summary["optimization_status"],
              "fresh_native_pose_confirmation": summary["fresh_native_pose_confirmation"],
              "sources": [{"path": str(context.resolve()), "sha256": sha256(context)},
                          {"path": str((run / "run.json").resolve()), "sha256": sha256(run / "run.json")},
                          {"path": str((run / "summary.json").resolve()), "sha256": sha256(run / "summary.json")}, *sources]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
    print(json.dumps({"output": str(output.resolve()), "slots": len(slots), "steps": steps,
                      "context_sha256_verified": True, "geometry_recomputed": False,
                      "original_run_status": summary["optimization_status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--arm", default="fixed_initial_mask")
    parser.add_argument("--steps", type=int, nargs="+", default=[0, 100, 225])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extract(args.run, args.context, args.arm, args.steps, args.output)
