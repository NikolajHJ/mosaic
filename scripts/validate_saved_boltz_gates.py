"""Validate local_v1 at saved failures using authenticated original science.

The protocol entry executes children serially. Each child imports scientific
modules from the immutable original release; only the new validation/helper and
measurement modules come from this release. No original files are changed.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def _load_new_module(name: str, filename: str):
    path = Path(__file__).resolve().parents[1] / "src/latentmol/boltz" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load immutable local validation helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap(production_root: Path, parent: Path):
    """Verify old science before importing it, then load only new audit code."""
    from diagnose_saved_boltz_gate import bootstrap as original_bootstrap
    original_bootstrap(production_root, parent)
    _load_new_module("latentmol.boltz.local_fd", "local_fd.py")
    return _load_new_module("latentmol.boltz._local_fd_validation", "local_fd_validation.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", type=Path)
    mode.add_argument("--parent-run", type=Path)
    parser.add_argument("--production-release", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--arm", choices=("active_set", "fixed_initial_mask"))
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require HPC allocation and immutable validation archive identity")
    if args.protocol is not None:
        if any(value is not None for value in (args.production_release, args.run_root, args.arm, args.step)):
            parser.error("Protocol mode cannot mix saved-child arguments")
        from latentmol.boltz.local_fd_validation import run_study
        print(run_study(args.protocol), flush=True)
    else:
        if any(value is None for value in (args.production_release, args.run_root, args.arm, args.step)):
            parser.error("A saved child needs release, run-root, arm and step")
        validation = bootstrap(args.production_release, args.parent_run)
        print(validation.run_saved(args.parent_run, args.run_root,
                                   production_root=args.production_release,
                                   arm=args.arm, step=args.step), flush=True)


if __name__ == "__main__":
    main()
