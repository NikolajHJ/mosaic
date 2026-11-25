"""Run one of the epoch-48 encoder arms from the same operator-pinned saved update."""

import argparse
import json
from pathlib import Path

from experiments.vae.encoder_upgrade_pilot import PILOT_RECIPE, run


# The short LSF allocation is eight hours. Leave ten minutes above the trainer
# budget for environment setup and exit. The trainer reserves another two minutes
# for its final save and panel inside these budgets.
WALL_SECONDS = {"control": 28_200, "candidate": 81_000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=WALL_SECONDS, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--resume-receipt", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True, help="Digest of the common saved checkpoint")
    parser.add_argument("--production-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.updates = None
    args.wall_seconds = WALL_SECONDS[args.arm]
    run(args, json.loads(PILOT_RECIPE.read_bytes()), early_arm=args.arm, source_sha256=args.source_sha256)


if __name__ == "__main__":
    main()
