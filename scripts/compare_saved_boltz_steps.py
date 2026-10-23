"""Run authenticated one-step update comparisons on DTU HPC only.

Original scientific modules come from each parent's immutable production release.
Only metadata verification, bounded step control and comparison instrumentation
come from this release. No chemistry conversion or new learned scorer is added.
Native-model tests are unvalidated until an HPC outcome appears in the ledger.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def load_new(name: str, relative: str):
    path = Path(__file__).resolve().parents[1] / "src/latentmol" / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load immutable step-comparison module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--protocol", type=Path)
    mode.add_argument("--parent-run", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--validation-archive-sha256", required=True)
    parser.add_argument("--production-release", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--arm", choices=("active_set", "fixed_initial_mask"))
    parser.add_argument("--step", type=int)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require DTU allocation and immutable comparison source")
    if args.protocol is not None:
        if args.validation_root is None or any(value is not None for value in (
                args.validation_report, args.production_release, args.run_root, args.arm, args.step)):
            parser.error("Protocol mode requires validation-root and no saved-child arguments")
        from latentmol.boltz.step_control_comparison import run_study
        print(run_study(args.protocol, args.validation_root, args.validation_archive_sha256), flush=True)
    else:
        if args.validation_root is not None or any(value is None for value in (
                args.validation_report, args.production_release, args.run_root, args.arm, args.step)):
            parser.error("Saved mode requires parent/release/arm/step/run-root/validation-report")
        from diagnose_saved_boltz_gate import bootstrap
        bootstrap(args.production_release, args.parent_run)
        load_new("latentmol.config_identity", "config_identity.py")
        load_new("latentmol.optimization", "optimization.py")
        comparison = load_new("latentmol.boltz._step_control_comparison", "boltz/step_control_comparison.py")
        print(comparison.run_saved(args.parent_run, args.run_root,
            production_root=args.production_release, arm=args.arm, step=args.step,
            validation_path=args.validation_report, validation_archive_sha256=args.validation_archive_sha256), flush=True)


if __name__ == "__main__":
    main()
