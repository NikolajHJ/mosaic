"""Prepare a target context inside an allocation, before optimization starts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from latentmol.guidance.protenix_pocket import validate_pocket


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare_runtime(runtime, target, occupied_atoms):
    """Launch preprocessing in the native environment; leave old assets alone."""
    validate_pocket(target["pocket"])
    if target["initialized_atoms"] != occupied_atoms:
        raise ValueError("Target atom count and design occupancy differ")
    if runtime.output.exists():
        raise FileExistsError(runtime.output)
    root = runtime.output.with_name(runtime.output.name + "-inputs")
    root.mkdir()
    write(root / "target.json", target)
    with (root / "prepare.log").open("x", encoding="utf-8") as log:
        subprocess.run([runtime.mosaic_python, "-m", __name__, "--target", str(root / "target.json"),
                        "--native-template", str(runtime.native_config.resolve()), "--output", str(root)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=2400)
    fresh = json.loads(runtime.fresh_config.read_text(encoding="utf-8"))
    fresh["assets_config"] = str((runtime.fresh_config.parent / fresh["assets_config"]).resolve())
    fresh["pocket"] = dict(path=str(root / "pocket.json"), sha256=digest(root / "pocket.json"))
    write(root / "fresh.json", fresh)
    runtime.native_config, runtime.fresh_config = root / "native.json", root / "fresh.json"
    return root


def prepare(target_path, template_path, output):
    if not (os.environ.get("LSB_JOBID", "").isdigit() or os.environ.get("PBS_JOBID")):
        raise RuntimeError("Native context preparation requires an allocated job")
    target = json.loads(target_path.read_text())
    pocket = validate_pocket(target["pocket"])
    native = json.loads(template_path.read_text())
    sys.path.insert(0, native["mosaic_source"])
    from mosaic.small_molecule.features import prepare_context
    from mosaic.small_molecule.io import save_context
    from mosaic.small_molecule.gradient_checks import array_digest
    from latentmol.guidance.protenix_pocket import map_explicit_pocket

    # The placeholder only supplies fixed ligand slots. Native scoring replaces
    # atom/charge/bond features and masks all ligand reference geometry.
    context, _ = prepare_context("C" * target["initialized_atoms"], [pocket["protein_sequence"]],
        cache_dir=native["cache_dir"], input_name=target["id"],
        identities=dict(protein_sequence=pocket["protein_sequence"], target_id=target["id"],
                        pdb_id=pocket["pdb_id"], pdb_sha256=pocket["pdb_sha256"]))
    tokens = map_explicit_pocket(context, pocket)
    if context.n_atoms != target["initialized_atoms"]:
        raise ValueError("Prepared native slot count differs from requested occupancy")
    context_files = save_context(output / "context", context)
    write(output / "assessment.json", dict(status="prepared", context_files=context_files,
        target_id=target["id"], pocket_tokens=tokens, target_sha256=digest(target_path),
        scope="Sequence-only, query-only protein context; ligand geometry off; no model evaluation."))
    write(output / "pocket.json", pocket)
    native.update(context_ready=str(output / "assessment.json"),
        context_ready_sha256=digest(output / "assessment.json"), expected_context_digest=array_digest(context),
        pocket_path=str(output / "pocket.json"), pocket_sha256=digest(output / "pocket.json"),
        startup_timeout_seconds=2400)
    write(output / "native.json", native)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--native-template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.target, args.native_template, args.output)


if __name__ == "__main__":
    main()
