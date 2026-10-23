"""Seal, transfer and dispatch the independent matched comparison, using shared operators."""
import argparse
import json
import re

from embedding_bridge_campaign import ROOT, seal, stage, remote


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "stage", "submit", "collect"))
    parser.add_argument("--attempt", default="attempt001")
    parser.add_argument("--stage", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if not re.fullmatch(r"attempt[0-9]{3}", args.attempt):
        raise ValueError("Need a versioned attemptNNN")
    folder = ROOT / "outputs/hpc-submissions" / ("source138-epoch36-embedding-comparison-" + args.attempt)
    if args.action == "seal":
        seal(folder, args.attempt, number=138, campaign_name="epoch36-embedding-comparison",
             config_relative="configs/epoch36_embedding_comparison.json",
             operator_relative="scripts/hpc/embedding_comparison_operator.py")
    else:
        identity = json.loads((folder / "seal.json").read_text())
        if args.action == "stage":
            stage(folder, identity)
        else:
            remote(folder, identity, args.action, args.stage)


if __name__ == "__main__":
    main()
