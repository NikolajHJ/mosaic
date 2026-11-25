"""Compare completed BRD4 method pilots per seed at equal elapsed GPU time."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


GROUPS = {"allocation": ("fixed400", "fixed150", "extend"),
          "screening": ("prior", "screen"), "reuse": ("fixed150", "reuse")}


def fixed75_sources(bundle):
    with tarfile.open(bundle) as source:
        plan = json.load(source.extractfile("experiments/latent-design/protenix-vae/restart-campaign-gpu-plan.json"))
    if set(plan["arms"]) != {"fixed75-a", "fixed75-b", "fixed75-c"}:
        raise ValueError("Expected the matched fixed-75 pilot bundle")
    return hashlib.sha256(bundle.read_bytes()).hexdigest(), plan["matched_source_sha256"]


def compare(results, generation_policies=None, fixed75_bundle=None):
    groups = GROUPS
    fixed75_pair = fixed75_sources(fixed75_bundle) if fixed75_bundle else None
    if any(row["case_name"] == "fixed75" for row in results):
        if fixed75_pair is None or generation_policies is not None:
            raise ValueError("Supply --fixed75-bundle for the matched fixed-75 pilot")
        groups = {**groups, "fixed_lengths": ("fixed400", "fixed150", "fixed75")}
    if generation_policies is not None:
        if len(set(generation_policies)) != 2 or set(generation_policies) - set(GROUPS["allocation"]):
            raise ValueError("Two distinct allocation policies are required")
        groups = {"three_generations": tuple(generation_policies)}
    panels = {}
    for row in results:
        if ("waves" in row) != (generation_policies is not None):
            raise ValueError("Supply the selected pair when comparing generation runs")
        if generation_policies is not None and (
            row["case_name"] not in generation_policies or len(row["waves"]) != 3
            or row["updates"] != 10_800 or row["fresh_budget"] != 144
        ):
            raise ValueError("Expected a completed selected three-generation arm")
        panel = panels.setdefault(row["case"]["seed"], {})
        case = row["case_name"]
        if case in panel:
            raise ValueError(f"Duplicate result for seed {row['case']['seed']} and case {case}")
        panel[case] = row
    comparisons = []
    for seed, panel in sorted(panels.items()):
        for name, cases in groups.items():
            missing = [key for key in cases if key not in panel]
            if missing:
                comparisons.append(dict(seed=seed, comparison=name, missing=missing))
                continue
            rows = [panel[key] for key in cases]
            starts = [row["start_hashes"] for row in rows]
            longest = max(starts, key=len)
            for sequence in starts:
                length = 1 if name == "screening" else len(sequence)
                if not sequence or sequence[:length] != longest[:length]:
                    raise ValueError(f"Paired starts differ for seed {seed} in {name}")
            same_code = len({row["source_sha256"] for row in rows}) == 1
            if name == "fixed_lengths":
                new_source, control_source = fixed75_pair
                same_code = all(row["source_sha256"] == (new_source if key == "fixed75" else control_source)
                                for key, row in zip(cases, rows, strict=True))
            if not same_code or len({(row["model_device"], row["updates"], row["fresh_budget"])
                                      for row in rows}) != 1:
                raise ValueError("Paired jobs have different devices, budgets or code")
            budget = min(row["total_seconds"] for row in rows)
            arms = {}
            for key, row in zip(cases, rows, strict=True):
                # A score is available only once its segment has actually been assessed.
                eligible = [m for m in row["milestones"] if row["setup_seconds"] + m["elapsed_seconds"] <= budget]
                last = eligible[-1] if eligible else {}
                fields = ("best_loss", "tenth_loss", "distinct_scored", "fresh_attempts", "updates")
                arms[key] = dict(common_time={f: last.get(f, 0 if f in fields[2:] else None) for f in fields},
                                 final={f: row[f] for f in fields}, call_counts=row["call_counts"],
                                 total_seconds=row["total_seconds"],
                                 seconds_to_ten=(row["setup_seconds"] + row["time_to_ten_seconds"]
                                                 if row["time_to_ten_seconds"] is not None else None))
                if generation_policies is not None:
                    arms[key]["generations"] = [{f: wave[f] for f in (
                        "generation", "cumulative_updates", "cumulative_fresh_attempts",
                        "cumulative_distinct", "best_novel", "tenth_novel")} for wave in row["waves"]]
            comparisons.append(dict(seed=seed, comparison=name, common_seconds=budget, arms=arms))
    return comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="One or more downloaded result.json files")
    parser.add_argument("--generation-policies", choices=GROUPS["allocation"], nargs=2,
                        help="Selected pair for the three-generation follow-up")
    parser.add_argument("--fixed75-bundle", type=Path, help="Sealed source.tar.gz of the matched 75-update arm")
    args = parser.parse_args()
    print(json.dumps(compare([json.loads(p.read_text()) for p in args.results], args.generation_policies,
                             args.fixed75_bundle),
                     indent=2, allow_nan=False))
