#!/usr/bin/env python3
"""Export saved Mosaic trajectories to a model-free, browsable run library."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import fields

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src/mosaic/small_molecule/Stage2"))
from operators import (Stage2Config, ELEMENTS, BOND_STATES, NEUTRAL_VALENCE,
                       project_simplex_rows, project_valence, soft_valence,
                       valence_ceiling, components, apply_fluidity_floor)


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")))
    tmp.replace(path)


def discover(root):
    """Stop at run boundaries; never walk large evaluations/report trees."""
    for base, dirs, _ in os.walk(root, followlinks=False):
        if "trajectory" in dirs and (Path(base) / "trajectory/trace.jsonl").is_file():
            yield Path(base)
            dirs[:] = []
        else:
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in
                             {"report", "reports", "evaluations", "readouts", "visualizations"})


def entropy(rows):
    if rows is None or not len(rows):
        return None
    return float(-(rows * np.log(np.clip(rows, 1e-30, 1))).sum(-1).mean())


def array(values):
    return np.round(np.asarray(values, dtype=np.float64), 6).tolist()


def read_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("X", "E_upper", "gradient_X", "gradient_E_upper",
                                  "charge_probabilities") if k in z}


def safe_artifact(trajectory, name):
    path = (trajectory / name).resolve()
    if not path.is_relative_to(trajectory.resolve()):
        raise ValueError(f"artifact outside trajectory: {name}")
    return path


def covalent_stretch(atoms, edges, distances, radii, threshold=1.35):
    """A generous element-aware diagnostic, distinct from the study gate."""
    i, j = np.triu_indices(len(atoms), 1)
    present = edges.argmax(-1) != 0
    types = atoms.argmax(-1)
    ratios = distances[i, j][present] / (radii[types[i[present]]] + radii[types[j[present]]])
    return {"covalent_stretched_bonds": int((ratios > threshold).sum()),
            "maximum_covalent_stretch_ratio": float(ratios.max()) if ratios.size else None}


class Depictions:
    def __init__(self):
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdDepictor
        RDLogger.DisableLog("rdApp.*")
        self.Chem, self.depict = Chem, rdDepictor
        self.cache = {}
        self.previous = None

    def coordinates(self, atoms, edges):
        key = (tuple(atoms.argmax(-1)), tuple(edges.argmax(-1)))
        n = len(atoms)
        if key not in self.cache:
            mol = self.Chem.RWMol()
            for element in key[0]:
                atom = self.Chem.Atom(ELEMENTS[element])
                atom.SetNoImplicit(True)
                mol.AddAtom(atom)
            types = [None, self.Chem.BondType.SINGLE, self.Chem.BondType.DOUBLE,
                     self.Chem.BondType.TRIPLE, self.Chem.BondType.AROMATIC]
            for (i, j), bond in zip(zip(*np.triu_indices(n, 1)), key[1]):
                if bond:
                    mol.AddBond(int(i), int(j), types[bond])
            try:
                self.depict.Compute2DCoords(mol)
                conf = mol.GetConformer()
                xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                               for i in range(n)])
                kind = "RDKit 2D layout of the raw argmax graph"
            except Exception:
                angle = np.arange(n) * 2 * np.pi / max(1, n)
                xy = np.stack([np.cos(angle), np.sin(angle)], axis=1)
                kind = "Circular graph layout (RDKit layout unavailable)"
            xy -= xy.mean(0)
            self.cache[key] = (xy, kind)
        xy, kind = self.cache[key]
        xy = xy.copy()
        if self.previous is not None:
            u, _, vt = np.linalg.svd(xy.T @ self.previous)
            xy = xy @ (u @ vt)
        self.previous = xy
        return array(xy), kind


def describe(config, name):
    """Describe settings only. Never invent a conclusion from a run name."""
    op = config.get("stage2", {})
    mode = op.get("mode", "none")
    goals = {
        "none": "Stage 1 control: see what the optimizer does without valence projection (5.2), connectivity (5.3), or sparsity (5.4).",
        "valence": "Experiment A: test valence projection (5.2). Check whether keeping each atom below its valence ceiling helps without stopping useful graph changes.",
        "connected": "Experiment B: test connectivity (5.3) after valence projection (5.2). Check whether connecting separate components reduces fragmentation.",
        "sparse": "Experiment C: test sparsity (5.4) after valence projection (5.2) and connectivity (5.3). Check whether reducing weak bond probabilities removes diffuse edge mass.",
        "sparse_only": "Additional control: test sparsity (5.4) with valence projection (5.2), but without connectivity (5.3). This isolates sparsity from connectivity and is separate from the protocol's A/B/C ladder.",
    }
    text = goals.get(mode, f"Inspect the saved {name} experiment.")
    if config.get("hardening"):
        text += " Experiment D: test hardening (5.5). Gradually make the model's input discrete and check whether updates become smaller."
    text += f" Atom learning rate: {config.get('eta_X', 'not recorded')}. Bond learning rate: {config.get('eta_E', 'not recorded')}."
    extras = [f"{key.replace('_', ' ')}: {op[key]}" for key in
              ("aromatic_order", "bridge_probability", "sparsity_amount", "sparsity_threshold",
               "readout_caps", "valence_damping", "fluidity_threshold", "fluidity_strength", "bridge_retry")
              if key in op and op[key] != getattr(Stage2Config(), key)]
    if extras:
        text += " Settings changed from the default: " + "; ".join(extras) + "."
    if config.get("hardening"):
        text += f" Backward gradient: {config.get('hardening_gradient', 'not recorded')}."
    text += f" Updates: {config.get('steps', 'not recorded')}. Seed: {config.get('seed', 'not recorded')}."
    return text + " This describes the setup; it does not claim that the experiment worked."


def native_evaluation(directory, frames):
    """Attach related native results while keeping their scores off the trajectory."""
    if read_json(directory.parent / "trial_spec.json", {}).get("task") != "native_proposal":
        return None
    native = directory.parent / "native_validation"
    predictions = read_json(native / "native_results.json", {})
    if not predictions:
        return None
    audits = read_json(native / "native_pose_audit.json", {})
    execution = read_json(native / "execution.json", {})
    candidates = []
    for name, result in predictions.items():
        identity, values = result.get("identity", {}), result["affinity"]
        audit = audits.get(name, {})
        step = identity.get("step")
        smiles = identity.get("canonical_smiles") or identity.get("smiles")
        # A source step is clickable only when its actual saved identity agrees.
        frame = next((i for i, f in enumerate(frames) if f["step"] == step
                      and f["readout"].get("canonical_smiles") == smiles), None)
        probabilities = [values.get(key) for key in ("affinity_probability_binary", "affinity_probability_binary1", "affinity_probability_binary2")]
        if any(p is not None and (not np.isfinite(p) or not 0 <= p <= 1) for p in probabilities):
            raise ValueError("Native binding-head probability is outside [0,1] or nonfinite")
        candidates.append({"name": name, "smiles": smiles, "role": identity.get("role", "selected_candidate"),
                           "proposal_frame": frame, "proposal_step": step, "affinity": values,
                           "exact_scored_pose_checked": audit.get("exact_affinity_scored_pose") is True,
                           "pose": {key:audit.get(key) for key in ("cavity_placement_pass", "maximum_bond_length_A", "bonds_outside_0_8_to_2_5A", "protein_ligand_pairs_below_1_2A", "ligand_fraction_within_6A_of_pocket")}})
    return {"source": str(native.resolve()), "protocol": execution.get("native_protocol"),
            "complete": read_json(directory.parent / "run_summary.json", {}).get("complete") is True,
            "candidates": candidates,
            "scope": "Rebuilt native features and poses, evaluated after candidate selection. These values may be tuning feedback and are not held out or experimental binding evidence. First-structure confidence and pre-affinity geometry do not validate the affinity-scored pose."}


def export_run(directory, label, geometry=None):
    trajectory = directory / "trajectory"
    raw_trace = (trajectory / "trace.jsonl").read_bytes()
    trace = [json.loads(line) for line in raw_trace.splitlines() if line.strip()]
    if not trace:
        raise ValueError("empty trace")
    config = read_json(directory / "stage2_config.json", {})
    geometry = geometry or read_json(directory / "frozen_geometry.json", {})
    distances = np.asarray(geometry.get("ligand_pair_distances_angstrom", []))
    if distances.size and geometry.get("context_sha256") != config.get("context_sha256"):
        raise ValueError("Geometry audit and scored context differ")
    op_config = config.get("stage2") or {}
    config_fields = {f.name for f in fields(Stage2Config)}
    op = Stage2Config(**{k: v for k, v in op_config.items() if k in config_fields})
    depiction = Depictions()
    from rdkit import rdBase
    radii = np.asarray([depiction.Chem.GetPeriodicTable().GetRcovalent(element) for element in ELEMENTS])
    frames = []
    previous = None
    for index, row in enumerate(trace):
        name = row.get("evaluated_arrays")
        if not name or not safe_artifact(trajectory, name).is_file():
            raise ValueError(f"missing evaluated arrays at step {row.get('evaluated_step')}")
        z = read_arrays(safe_artifact(trajectory, name))
        x, e = z["X"], z["E_upper"]
        ro = read_json(safe_artifact(trajectory, row["readout"]), {}) if row.get("readout") else {}
        n = len(x)
        caps, _ = valence_ceiling(x, readout_caps=op.readout_caps)
        v = soft_valence(e, n, aromatic_order=op.aromatic_order)
        hard_v = soft_valence(np.eye(5)[e.argmax(-1)], n, aromatic_order=op.aromatic_order)
        hard_caps = NEUTRAL_VALENCE[x.argmax(-1)]
        count = int(len(np.unique(components(n, e))))
        sanitized = ro.get("sanitized")  # absent readout is unknown, never false or carried forward
        scalars = {
            "loss": row.get("loss"), "atom_entropy": entropy(x), "bond_entropy": entropy(e),
            "charge_entropy": entropy(z.get("charge_probabilities")),
            "sanitized": None if sanitized is None else int(sanitized),
            "connected": int(count == 1),
            "valid_connected": None if sanitized is None else int(bool(sanitized) and count == 1),
            "components": count, "over_valence_atoms": int((hard_v > hard_caps + 2e-6).sum()),
            "bond_mass": float((1 - e[:, 0]).sum()),
            "gradient_X": row.get("raw_gradient_norm_X"),
            "gradient_E": row.get("raw_gradient_norm_E_upper"),
        }
        if distances.size:
            if distances.shape != (n, n) or not np.isfinite(distances).all():
                raise ValueError("Frozen geometry has the wrong atom count or nonfinite distances")
            lengths = distances[np.triu_indices(n, 1)][e.argmax(-1) != 0]
            scalars["frozen_long_bonds"] = int((lengths > 2.5).sum())
            scalars["frozen_short_bonds"] = int((lengths < .8).sum())
            scalars["maximum_frozen_bond_length_A"] = float(lengths.max()) if lengths.size else None
            scalars["frozen_bond_length_pass"] = int(not (scalars["frozen_long_bonds"] or scalars["frozen_short_bonds"]))
            scalars.update(covalent_stretch(x, e, distances, radii))
        for key, val in (row.get("aux_metrics") or {}).items():
            if isinstance(val, (float, int)) and not isinstance(val, bool) and np.isfinite(val):
                scalars[f"aux:{key}"] = val
        operators = row.get("post_step_operators")
        projection = None
        if operators and "gradient_X" in z and config.get("eta_X") is not None and config.get("eta_E") is not None:
            px = project_simplex_rows(x - config["eta_X"] * z["gradient_X"])
            pe = project_simplex_rows(e - config["eta_E"] * z["gradient_E_upper"])
            scalars["gradient_step_X"] = float(np.linalg.norm(config["eta_X"] * z["gradient_X"]))
            scalars["gradient_step_E"] = float(np.linalg.norm(config["eta_E"] * z["gradient_E_upper"]))
            scalars["simplex_step_X"] = float(np.linalg.norm(px - x))
            scalars["simplex_step_E"] = float(np.linalg.norm(pe - e))
            if op.mode != "none":
                atom_support = (operators.get("experimental") or {}).get("atom_valence_support")
                atom_support_verified = True
                projection_scope = "Reconstructed first valence projection after update and optional fluidity correction."
                if atom_support:
                    guarded = np.asarray(atom_support.get("projected_atoms", []))
                    if not guarded.size and op.fluidity_strength == 0 and index+1 < len(trace):
                        # Legacy guard receipts omitted their atom array. Here
                        # the guard was the only atom-changing correction, so
                        # its output is the recorded following-state X.
                        guarded = read_arrays(safe_artifact(trajectory, trace[index+1]["evaluated_arrays"]))["X"]
                    atom_support_verified = guarded.shape == px.shape and abs(float(np.linalg.norm(guarded-px))-atom_support["displacement_l2"]) < 1e-4
                    if atom_support_verified:
                        px = guarded
                        projection_scope = "First bond-valence projection after the recorded atom-support correction and optional fluidity correction."
                if op.fluidity_strength > 0 and op.fluidity_threshold < 1:
                    px, _ = apply_fluidity_floor(px, threshold=op.fluidity_threshold, strength=op.fluidity_strength)
                    pe, _ = apply_fluidity_floor(pe, threshold=op.fluidity_threshold, strength=op.fluidity_strength)
                pcaps, _ = valence_ceiling(px, readout_caps=op.readout_caps)
                before = soft_valence(pe, n, aromatic_order=op.aromatic_order)
                after_e, receipt = project_valence(px, pe, config=op)
                after = soft_valence(after_e, n, aromatic_order=op.aromatic_order)
                recorded = (operators.get("valence") or {}).get("displacement_l2")
                verified = recorded is not None and abs(receipt["displacement_l2"] - recorded) < 1e-4
                verified = verified and atom_support_verified
                projection = {"caps": array(pcaps), "before": array(before), "after": array(after),
                              "element_indices": px.argmax(-1).tolist(),
                              "replay_matches_receipt": verified,
                              "scope": projection_scope} if verified else None
                scalars["projection_over_atoms"] = int((before > pcaps + 2e-6).sum()) if verified else None
                scalars["projection_changed_atoms"] = int((np.abs(after - before) > 2e-6).sum()) if verified else None
            valence = operators.get("valence") or {}
            connectivity = operators.get("connectivity") or {}
            sparse = operators.get("sparsity") or {}
            extra = operators.get("experimental") or {}
            for key, value in {
                "projection_l2": valence.get("displacement_l2"),
                "valence_excess_before": valence.get("max_excess_before"),
                "valence_excess_after": valence.get("max_excess_after"),
                "bridges_attempted": len(connectivity["bridges"]) if "bridges" in connectivity else None,
                "bridges_surviving": connectivity.get("bridges_surviving"),
                "bridges_lost": connectivity.get("bridges_lost"),
                "components_after": operators.get("components_after"),
                "retry_bridges": len(connectivity["retry"]["bridges"]) if connectivity.get("retry") else None,
                "sparsity_pairs": sparse.get("changed_pairs"),
                "sparsity_removed": sparse.get("removed_bond_mass"),
                "fluidity_atoms": ((operators.get("fluidity_floor") or {}).get("atoms") or {}).get("rows_relaxed"),
                "readout_pruned_edges": sum((extra.get(k) or {}).get("count", 0) for k in
                    ("readout_valence", "final_readout_valence", "after_sparsity_readout_valence")) if extra.get("readout_safe") else None,
                "aromatic_mass_merged": extra.get("aromatic_mass_moved_to_single"),
                "experimental_bridges": len(extra["reconnect"].get("bridges", [])) if "reconnect" in extra else None,
                "experimental_rewires": len(extra["saturated_component_rewire"].get("rewires", [])) if "saturated_component_rewire" in extra else None,
                "atom_support_atoms": (extra.get("atom_valence_support") or {}).get("count"),
                "atom_support_changed_argmax": (extra.get("atom_valence_support") or {}).get("changed_argmax_atoms"),
                "atom_support_l2": (extra.get("atom_valence_support") or {}).get("displacement_l2"),
            }.items():
                scalars[key] = value
        if previous is not None:
            # This observed displacement belongs to the preceding update, not the current loss.
            frames[-1]["metrics"]["observed_step_X"] = float(np.linalg.norm(x - previous[0]))
            frames[-1]["metrics"]["observed_step_E"] = float(np.linalg.norm(e - previous[1]))
            for suffix, current, prior, gradient in (("X", x, previous[0], previous[2]), ("E", e, previous[1], previous[3])):
                if gradient is not None:
                    delta = current-prior
                    tangent = gradient-gradient.mean(-1, keepdims=True)
                    norm = np.linalg.norm(tangent)*np.linalg.norm(delta)
                    frames[-1]["metrics"][f"descent_alignment_{suffix}"] = float(np.sum(-tangent*delta)/norm) if norm > 1e-12 else None
        previous = (x, e, z.get("gradient_X"), z.get("gradient_E_upper"))
        if config.get("hardening"):
            # Store only the actual recorded schedule where available.
            scalars["hardness"] = row.get("aux_metrics", {}).get("forward_hardness")
        xy, layout = depiction.coordinates(x, e)
        frames.append({"step": row["evaluated_step"], "final": bool(row.get("final")),
                       "metrics": scalars, "X": array(x), "E": array(e),
                       "argmax_atoms": x.argmax(-1).tolist(), "argmax_bonds": e.argmax(-1).tolist(),
                       "Q": array(z["charge_probabilities"]) if "charge_probabilities" in z else None,
                       "xy": xy, "layout": layout,
                       "readout": ro, "valence": {"caps": array(caps), "expected": array(v),
                           "discrete": array(hard_v), "discrete_caps": array(hard_caps)},
                       "projection": projection, "operators": operators})
    gaps = read_json(directory / "soft_hard_gap.json", [])
    if config.get("hardening"):
        for point in gaps:
            if not point.get("soft_score_source"):
                point["optimization_forward_score"] = point.get("soft_score")
                point["soft_score"] = None
                point["soft_score_source"] = "unqualified_legacy_hardening_auxiliary"
                point["soft_score_note"] = "Legacy writer copied the hardened forward auxiliary score. Separate soft_loss and hard_loss remain valid; the raw source file is unchanged."
    return {"schema_version": 1, "name": label, "source": str(directory.resolve()),
            "trace_sha256": hashlib.sha256(raw_trace).hexdigest(), "config": config,
            "description": read_json(directory / "visualization.json", {}).get("description") or describe(config, label),
            "elements": list(ELEMENTS), "bonds": list(BOND_STATES),
            "frozen_distances_angstrom": distances.tolist() if distances.size else None,
            "covalent_radii_angstrom": radii.tolist(), "covalent_stretch_threshold": 1.35,
            "covalent_radii_source": {"library": "RDKit", "version": rdBase.rdkitVersion, "function": "GetRcovalent"},
            "charge_labels": config.get("charge_labels"), "frames": frames, "soft_hard_gap": gaps,
            "native_evaluation": native_evaluation(directory, frames)}


def build(targets, output, geometry_paths=()):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # The library is generated output, and must never enclose a source experiment.
    sources = sorted({p.resolve() for root in targets for p in discover(root.resolve())})
    if any(p.is_relative_to(output) or output == p for p in sources):
        raise ValueError("library output must not contain source runs")
    curation = read_json(output / "curation.json", {})
    index = read_json(output / "index.json", {"schema_version": 1, "runs": []})
    entries = {r["id"]: r for r in index["runs"]}
    failed = []
    geometries = {}
    for path in geometry_paths:
        geometry = read_json(path)
        geometries[geometry["context_sha256"]] = geometry
    for directory in sources:
        rid = hashlib.sha256(str(directory).encode()).hexdigest()[:20]
        if curation.get(rid, {}).get("deleted"):
            print(f"skip trashed: {directory.name}", flush=True)
            continue
        label = f"{directory.parent.name} / {directory.name}"
        try:
            context = read_json(directory / "stage2_config.json", {}).get("context_sha256")
            payload = export_run(directory, label, geometries.get(context))
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
            destination = output / "runs" / rid
            destination.mkdir(parents=True, exist_ok=True)
            tmp = destination / "data.json.gz.tmp"
            tmp.write_bytes(gzip.compress(encoded))
            tmp.replace(destination / "data.json.gz")
            entry = {"id": rid, "name": label, "source": payload["source"],
                     "frames": len(payload["frames"]), "description": payload["description"],
                     "backend": payload["config"].get("backend", "not recorded"),
                     "mode": (payload["config"].get("stage2") or {}).get("mode", "not recorded"),
                     "complete": read_json(directory / "run_summary.json", {}).get("complete") is True,
                     "trace_sha256": payload["trace_sha256"]}
            entries[rid] = entry
            write_json(destination / "meta.json", entry)
            print(f"exported {label}: {entry['frames']} frames", flush=True)
        except Exception as error:
            failed.append({"source": str(directory), "error": f"{type(error).__name__}: {error}"})
            print(f"FAILED {directory}: {error}", file=sys.stderr, flush=True)
    write_json(output / "index.json", {"schema_version": 1, "runs": list(entries.values()), "errors": failed})
    shutil.copy2(HERE / "template.html", output / "index.html")
    for name in ("viewer.js", "style.css"):
        shutil.copy2(HERE / "assets" / name, output / name)
    if not sources:
        raise ValueError("no trajectory/trace.jsonl runs found")
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed; details are in index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "results/visualizations")
    parser.add_argument("--geometry", action="append", type=Path, default=[], help="Optional frozen-context audit; matched by context SHA. Repeat for multiple contexts.")
    args = parser.parse_args()
    build(args.targets, args.output, args.geometry)
