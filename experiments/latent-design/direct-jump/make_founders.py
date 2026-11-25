"""Generate disjoint 25-atom sparse-tree starts using Mosaic Stage 3's initializer."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


PANELS = (
    (0, "train", 2611000, 24), (0, "test", 2612000, 32),
    (1, "train", 2613000, 24), (1, "test", 2614000, 32),
)


def generate(mosaic: Path):
    experiment = mosaic / "experiments/learned-prior/protenix-defog-composition-30000"
    sys.path[:0] = [str(mosaic / "src"), str(experiment / "scripts")]
    from mosaic.small_molecule.Stage1.diagnostics import raw_readout
    from stage3_math import make_initial_graph

    rows = []
    seen = set()
    for cohort, split, base, count in PANELS:
        selected = 0
        for seed in range(base, base + 1000):
            graph = make_initial_graph(seed, "sparse", n_atoms=25)
            _, readout = raw_readout(graph)
            smiles = readout["canonical_smiles"]
            if not readout["rdkit_sanitized"] or readout["fragment_count"] != 1:
                raise ValueError(f"Stage 3 sparse founder failed chemistry validation at seed {seed}")
            if smiles in seen:
                continue
            seen.add(smiles)
            rows.append(dict(cohort=cohort, split=split, seed=seed, atoms=readout["element_indices"],
                             bonds=readout["bond_upper_indices"], smiles=smiles))
            selected += 1
            if selected == count:
                break
        if selected != count:
            raise ValueError(f"Insufficient distinct founders in cohort {cohort} {split}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaic", type=Path, required=True, help="Mosaic Stage 3 source root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    encoded = (json.dumps(generate(args.mosaic.resolve()), separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(f"{hashlib.sha256(encoded).hexdigest()}  {args.output}")


if __name__ == "__main__":
    main()
