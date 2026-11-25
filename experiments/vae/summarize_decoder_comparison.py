"""Print bond and molecule quality from a saved compact decoder comparison."""

import argparse
import json
from pathlib import Path


def percent(value):
    return "n/a" if value is None else f"{value:.1%}"


def measurement(variant, step, observation, topology):
    r, prior = observation["reconstruction"], observation["prior"]
    confusion = r["bond_confusion_target_present_pairs"]
    matrix = confusion["rows_target_columns_prediction"]
    if confusion["labels"][0] != "NONE":
        raise ValueError("Expected NONE as the first bond label")
    # The confusion table only counts pairs of occupied target atoms after
    # graph alignment. The global counts include every unordered slot pair.
    fp_occupied = sum(matrix[0][1:])
    true_positives = r["correct_bonded_pairs"]
    fp_padding = r["predicted_bonded_pairs"] - true_positives - fp_occupied
    if fp_padding < 0 or sum(sum(row[1:]) for row in matrix[1:]) != true_positives:
        raise ValueError(f"Conflicting global and within-target bond counts for {variant} at {step}")
    connected = "n/a"
    if topology:
        group = topology["all"]
        connected = f"{group['connected_predictions']}/{group['targets']}"
    return (variant, step, percent(r["bonded_pair_precision"]), percent(r["bonded_pair_recall"]),
            fp_occupied, fp_padding, connected, f"{r['valid_chemistry_count']}/{r['chemistry_samples']}",
            f"{prior['prior_valid']}/{prior['prior_points']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training", type=Path, help="training directory containing comparison.json")
    parser.add_argument("--all-updates", action="store_true", help="include every saved quality measurement")
    args = parser.parse_args()
    report = json.loads((args.training / "comparison.json").read_bytes())
    endpoint = report["updates_per_arm"]
    rows = []
    for variant, arm in report["arms"].items():
        if args.all_updates:
            path = args.training / variant / "quality.json"
            if not path.exists() and arm["status"] in ("initializing", "training"):
                continue
            observations = json.loads(path.read_bytes())
        elif "endpoint" in arm:
            observations = {endpoint: arm["endpoint"]}
        else:
            continue
        for key, observation in sorted(observations.items(), key=lambda item: int(item[0])):
            step = int(key)
            topology = observation.get("topology") or (arm.get("topology") if step == endpoint else None)
            rows.append(measurement(variant, step, observation, topology))
    if not rows:
        print(f"No {'measurements' if args.all_updates else 'completed endpoints'} saved yet.")
        return
    headers = ("Decoder", "Update", "Bond precision", "Bond recall", "False bonds on real slots",
               "False bonds involving padding", "Connected", "Valid reconstruction", "Valid prior argmax")
    for row in (headers, ["---"] * len(headers), *rows):
        print("| " + " | ".join(map(str, row)) + " |")
    print("\nSlot categories use aligned target atoms. n/a indicates an unavailable measurement.")


if __name__ == "__main__":
    main()
