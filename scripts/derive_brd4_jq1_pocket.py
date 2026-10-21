"""Extract the immutable JQ1 pocket on an allocated CPU node."""

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
