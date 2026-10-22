"""Allocated CPU: empirical training-only composition, without a model load."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Empirical preprocessing requires an allocated DTU job")
    import numpy as np
    from mosaic.small_molecule.chemistry.state import mosaic_spec

    config = json.loads(args.config.read_text())
    binding = config["composition_data"]
    data = Path(binding["path"])
    manifest = json.loads((data / "manifest.json").read_text())
    if (
        sha(data / "manifest.json") != binding["manifest_sha256"]
        or manifest["dataset_sha256"] != binding["dataset_sha256"]
        or manifest["split_sha256"] != binding["split_sha256"]
    ):
        raise ValueError("Empirical reference data/split/manifest identity changed")
    arrays, receipts = {}, {}
    for name in ("atom_types.npy", "charges.npy", "split.npy"):
        entry = manifest["files"][name]
        before = (data / name).stat()
        actual = sha(data / name)
        if before.st_size != entry["bytes"] or actual != entry["sha256"]:
            raise ValueError("Reference source array differs: " + name)
        arrays[name] = np.load(data / name, mmap_mode="r", allow_pickle=False)
        receipts[name] = {
            "sha256": actual,
            "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns,
        }
    spec = manifest["spec"]
    a, q, labels = (arrays[k] for k in ("atom_types.npy", "charges.npy", "split.npy"))
    if (
        a.shape != q.shape
        or a.shape[0] != len(labels)
        or a.shape[1] != spec["max_nodes"]
        or spec["atomic_numbers"][0] != 0
    ):
        raise ValueError("Reference label shapes/schema differ")
    counts = np.zeros(len(spec["atomic_numbers"]), dtype=np.int64)
    charges = np.zeros(len(spec["formal_charges"]), dtype=np.int64)
    rows = 0
    for start in range(0, len(labels), 8192):
        take = (
            np.asarray(labels[start : start + 8192]) == 0
        )  # Authenticated DBMol TRAIN label.
        atoms, charge = (
            np.asarray(a[start : start + 8192])[take],
            np.asarray(q[start : start + 8192])[take],
        )
        occupied = atoms != 0
        counts += np.bincount(atoms[occupied], minlength=len(counts))
        charges += np.bincount(charge[occupied], minlength=len(charges))
        rows += int(take.sum())
    if rows != manifest["counts"]["train"] or rows != binding["training_molecules"]:
        raise ValueError("Training population differs")
    for name, receipt in receipts.items():
        after = (data / name).stat()
        if (after.st_size, after.st_mtime_ns) != (
            receipt["bytes"],
            receipt["mtime_ns"],
        ):
            raise ValueError("Reference data changed during preprocessing")
    target = mosaic_spec(config["n_atoms"])
    supported = list(target.atomic_numbers[1:])
    selected = [int(counts[spec["atomic_numbers"].index(z)]) for z in supported]
    if not sum(selected):
        raise ValueError("No supported empirical element mass")
    result = {
        "status": "passed",
        "version": "mosaic-training-composition-v1",
        "job_id": os.environ["LSB_JOBID"],
        "split": "train",
        "molecules": rows,
        "data": binding,
        "source_files": receipts,
        "source_schema": spec,
        "source_element_counts": counts.tolist(),
        "source_charge_counts": charges.tolist(),
        "atomic_numbers": supported,
        "element_counts": selected,
        "excluded_element_counts": {
            str(z): int(counts[k])
            for k, z in enumerate(spec["atomic_numbers"])
            if z and z not in supported
        },
        "conditioning": "source occupied TRAIN atom counts conditioned on Mosaic supported elements; all source charges retained in counts; occupancy/charge fixed in design",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                "reference": str(args.output),
                "sha256": sha(args.output),
                "status": "passed",
            }
        )
    )


if __name__ == "__main__":
    main()
