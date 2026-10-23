"""Load new diagnostic probes against a parent's unchanged scientific code.

Run only inside the new immutable diagnostic release's DTU GPU allocation.
Every production source file is verified before importing its package. The
diagnostic itself comes from this release, while scorer, adapter, decoder,
objective, graph policy and context loaders come from the original release.
This separation permits bookkeeping/chemistry work in a newer release without
silently replaying a failed derivative using changed scientific code.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def bootstrap(production_root: Path, parent: Path):
    """Authenticate original source before the first latentmol import."""
    source = json.loads((parent / "run.json").read_text())["source"]
    root = production_root.resolve()
    for name, expected in source["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Missing original production file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original production file changed: {name}")
    if any(name == "latentmol" or name.startswith("latentmol.") for name in sys.modules):
        raise ValueError("Production bootstrap must run before importing latentmol")
    sys.path.insert(0, str(root / "src"))
    diagnostic_path = Path(__file__).resolve().parents[1] / "src/latentmol/boltz/active_set_diagnostic.py"
    spec = importlib.util.spec_from_file_location("latentmol.boltz._saved_gate_diagnostic", diagnostic_path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the immutable diagnostic module")
    diagnostic = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = diagnostic
    spec.loader.exec_module(diagnostic)
    # Imports may otherwise resolve to an editable installation on sys.path.
    import latentmol.boltz.active_set as production
    if Path(production.__file__).resolve() != root / "src/latentmol/boltz/active_set.py":
        raise ValueError("Scientific modules did not resolve to the original release")
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--production-release", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("active_set", "fixed_initial_mask"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--device-override", choices=("cpu",))
    parser.add_argument("--probe-profile", choices=("extended", "focused_cpu"), default="extended")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an HPC allocation and immutable diagnostic archive identity")
    diagnostic = bootstrap(args.production_release, args.parent_run)
    print(diagnostic.run(args.parent_run, args.run_root, arm=args.arm, step=args.step,
                         production_root=args.production_release, device_override=args.device_override,
                         probe_profile=args.probe_profile), flush=True)


if __name__ == "__main__":
    main()
