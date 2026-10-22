"""Bounded native-protein inference on an allocated HPC GPU.

This exercises Mosaic's existing Protenix interface, not the Stage 0 ligand
graph adapter. Assets must be staged beforehand; inference runs offline.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "started",
        "scope": "native protein installation smoke; not Stage 0 graph validation",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "seed": 0,
        "use_msa": False,
        "template": None,
        "model": "protenix_tiny_default_v0.5.0",
        "recycling_steps": 1,
        "sampling_steps": 2,
        "sampler_overrides": {"gamma0": 0.0, "step_scale_eta": 1.0,
                              "noise_scale_lambda": 1.0, "N_steps": 20},
        "hostname": socket.gethostname(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "job_id": os.environ.get("LSB_JOBID"),
        "gradient_scope": "none; inference only; no parameter updates",
        "plddt_units": "normalized [0,1]; structure B-factors contain 100*pLDDT",
        "interpretation": "Arbitrary short sequence and two sampling steps; not a validated fold.",
    }
    started = time.monotonic()
    try:
        cache = Path(os.environ["MOSAIC_CACHE_DIR"]).resolve() / "protenix"
        assets = ["components.v20240608.cif", "components.v20240608.cif.rdkit_mol.pkl",
                  "clusters-by-entity-40.txt", summary["model"] + ".eqx",
                  summary["model"] + ".skeleton.pkl"]
        for name in assets:
            if not (cache / name).is_file():
                raise FileNotFoundError(f"Pre-stage required asset: {cache / name}")

        # Upstream's data config hardcodes ~/.protenix. Set the three native
        # asset paths BEFORE ccd.py imports/captures them; no home symlink or
        # upstream patch is needed. Mosaic redirects the model backend itself.
        from protenix.configs.configs_data import data_configs
        native_paths = dict(zip(("ccd_components_file", "ccd_components_rdkit_mol_file",
                                "pdb_cluster_file"), (str(cache / name) for name in assets[:3])))
        for name, path in native_paths.items():
            if name not in data_configs:
                raise KeyError(f"Upstream data config changed: {name}")
            data_configs[name] = path
        from protenix.data import ccd
        if Path(ccd.COMPONENTS_FILE) != cache / assets[0] or ccd.RKDIT_MOL_PKL != cache / assets[1]:
            raise RuntimeError("Upstream CCD paths do not point to the staged scratch assets")
        summary["native_asset_paths"] = native_paths

        import gemmi
        import jax
        import numpy as np
        from mosaic.losses.protenix import biotite_array_to_gemmi_struct
        from mosaic.models.protenix import ProtenixTiny
        from mosaic.structure_prediction import TargetChain

        devices = jax.devices()
        summary["devices"] = [str(device) for device in devices]
        if not devices or any(device.platform != "gpu" for device in devices):
            raise RuntimeError(f"An allocated GPU is required, found {devices}")
        summary["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "jax-cuda12-plugin", "equinox", "numpy",
                         "protenix", "biotite", "rdkit", "gemmi")
        }
        random.seed(summary["seed"])
        np.random.seed(summary["seed"])
        print("Loading one frozen Protenix-Tiny model", flush=True)
        stage = time.monotonic()
        model = ProtenixTiny()
        jax.block_until_ready(model)
        summary["load_seconds"] = time.monotonic() - stage
        print("Featurizing 20 residues without MSA or templates", flush=True)
        stage = time.monotonic()
        features, writer = model.target_only_features([
            TargetChain(sequence=summary["sequence"], use_msa=False)
        ])
        jax.block_until_ready(features)
        summary["featurization_seconds"] = time.monotonic() - stage
        summary["feature_shapes"] = {name: list(value.shape) for name, value in features.items()
                                     if hasattr(value, "shape")}
        print("Running one recycle and two sampling steps", flush=True)
        stage = time.monotonic()
        with jax.default_matmul_precision("float32"):
            output = model.model_output(features=features, recycling_steps=1,
                                        sampling_steps=2, key=jax.random.key(0))
        jax.block_until_ready(output)
        summary["forward_seconds_including_compilation"] = time.monotonic() - stage
        arrays = {name: np.asarray(getattr(output, name)) for name in
                  ("structure_coordinates", "distogram_logits", "distogram_bins",
                   "plddt", "pae", "pae_logits", "pae_bins", "atom37_coords", "atom37_mask")}
        summary["output_shapes"] = {name: list(value.shape) for name, value in arrays.items()}
        summary["field_finiteness"] = {name: bool(np.isfinite(value).all())
                                      for name, value in arrays.items()}
        np.savez_compressed(args.output / "model_outputs.npz", **arrays)
        if not all(summary["field_finiteness"].values()):
            raise RuntimeError("Non-finite model output")
        atom_count = len(writer)
        if arrays["structure_coordinates"].shape != (1, atom_count, 3):
            raise RuntimeError("Coordinate shape does not match the native atom writer")
        if arrays["plddt"].shape != (20,) or arrays["pae"].shape != (20, 20):
            raise RuntimeError("Unexpected token confidence shape")
        atom_to_token = np.asarray(features["atom_to_token_idx"])
        structure = biotite_array_to_gemmi_struct(
            writer, arrays["structure_coordinates"][0],
            100 * arrays["plddt"][atom_to_token],
        )
        structure.name = "protenix_tiny_native_smoke"
        structure.setup_entities()
        structure.write_pdb(str(args.output / "prediction.pdb"))
        structure.make_mmcif_document().write_file(str(args.output / "prediction.cif"))
        for filename in ("prediction.pdb", "prediction.cif"):
            restored = gemmi.read_structure(str(args.output / filename))
            restored_atoms = sum(len(residue) for chain in restored[0] for residue in chain)
            if restored_atoms != atom_count:
                raise RuntimeError(f"Structure round trip changed atom count: {filename}")
        (args.output / "input.fasta").write_text(">native_smoke\n" + summary["sequence"] + "\n")
        summary.update(status="passed", atom_count=atom_count, all_outputs_finite=True,
                       output_shapes={name: list(value.shape) for name, value in arrays.items()},
                       mean_plddt=float(arrays["plddt"].mean()),
                       mean_pae_angstrom=float(arrays["pae"].mean()),
                       mosaic_cache=str(cache.parent))
        print("Native forward and structure round trips passed", flush=True)
    except Exception:
        summary["status"] = "failed"
        summary["traceback"] = traceback.format_exc()
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest = {path.name: {"bytes": path.stat().st_size,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in sorted(args.output.iterdir()) if path.is_file()}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
