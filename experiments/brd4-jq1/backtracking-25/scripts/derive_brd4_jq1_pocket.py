"""Extract the immutable JQ1 pocket on an allocated CPU node."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import json
import os
from pathlib import Path

from mosaic.small_molecule.pocket import derive_definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Pocket extraction requires allocated DTU HPC")
    pocket = derive_definition(args.pdb, args.fixture)
    with args.output.open("x") as stream:
        json.dump(pocket, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": pocket["status"], "residues": len(pocket["residues"])}))


if __name__ == "__main__":
    main()
