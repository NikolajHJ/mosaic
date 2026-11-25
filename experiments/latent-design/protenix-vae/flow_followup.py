"""Analyze held-out BRD4 comparisons at the level of independent starts."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from flow_lr_selection import load_config


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def paired(blocks, margin, *, draws=10000, seed=281061):
    """Bootstrap paired outcomes within each fixed cohort."""
    rng = np.random.default_rng(seed)
    simulations = np.zeros(draws)
    wins = losses = count = 0
    for values in blocks:
        difference = np.asarray(values, dtype=np.int8)
        if not difference.size:
            continue
        wins += int((difference > 0).sum())
        losses += int((difference < 0).sum())
        count += len(difference)
        indices = rng.integers(len(difference), size=(draws, len(difference)))
        simulations += difference[indices].sum(axis=1)
    if not count:
        return dict(n=0, decision="no_comparable_pairs")
    low, high = map(float, np.quantile(simulations / count, [0.025, 0.975]))
    if low > margin:
        decision = "candidate_practically_superior"
    elif high < -margin:
        decision = "control_practically_superior"
    elif low >= -margin and high <= margin:
        decision = "within_practical_margin"
    else:
        decision = "inconclusive"
    return dict(n=count, wins=wins, losses=losses, ties=count-wins-losses,
                delta=(wins-losses)/count, interval_95=[low, high], decision=decision,
                practical_gain_ruled_out=high < margin)


def analyze_latent(roots, config):
    comparisons = {
        "portfolio": (("sgd_short", "residual"), ("sgd_short", "sgd_restart")),
        "residual_vs_restart": (("residual",), ("sgd_restart",)),
        "residual_vs_random_jump": (("residual",), ("random_jump",)),
        "flow_vs_residual": (("flow",), ("residual",)),
        "24_vs_7_updates": (("sgd_short", "teacher"), ("sgd_short",)),
    }
    student_arms = config.get("student_arms", ["residual", "flow"])
    if "flow" not in student_arms:
        comparisons.pop("flow_vs_residual")
    cohort_data = []
    all_starts = set()
    all_smiles = set()
    expected = set(config["cohort_seeds"])
    observed = set()
    selection_proof = None
    for root in roots:
        panel = read(root / "panel.json")
        fit = read(root / "students" / "fit.json")
        baseline = read(root / "fresh" / "summary.json")
        students = read(root / "students" / "fresh" / "summary.json")
        seed = panel["config"]["seed"]
        if (seed not in expected - observed or not panel["config"].get("portfolio_test")
                or panel["checkpoint_sha256"] != config["checkpoint_sha256"]
                or baseline["status"] != "completed" or students["status"] != "completed"
                or baseline["job_id"] != students["job_id"]):
            raise ValueError(f"Foreign, incomplete, or repeated cohort: {root}")
        if "tuning_seeds" in config:
            proof = (panel["learning_rate"], panel["lr_selection_sha256"])
            if proof[0] not in config["learning_rates"] or not proof[1] or (
                selection_proof is not None and proof != selection_proof
            ):
                raise ValueError("All validation cohorts must use the same selected rate")
            selection_proof = proof
        identities = set(panel["start_smiles"].values())
        start_ids = set(panel["starts"])
        if (start_ids & all_starts or identities & all_smiles or len(identities) != len(start_ids)
                or set(panel["start_smiles"]) != start_ids):
            raise ValueError("Cohorts overlap in start seeds or initial molecular identities")
        all_starts.update(start_ids)
        all_smiles.update(identities)
        observed.add(seed)
        holdout = start_ids - set(panel["train"])
        if len(holdout) != config["test_starts"] or len(panel["train"]) != config["train_starts"]:
            raise ValueError("Unexpected train/holdout sizes")
        if any(row["start"] not in panel["train"] for row in fit["pairs"]):
            raise ValueError("A held-out start leaked into the student fit")
        if fit["status"] == "trained" and len(fit["pairs"]) < config["min_pairs"]:
            raise ValueError("Trained student has too few verified endpoint pairs")
        data = dict(seed=seed, pairs=len(fit["pairs"]), fit=fit["status"], holdout=[])
        for start in sorted(holdout):
            rows = [row for row in baseline["candidates"] + students["candidates"] if row["start"] == start]
            originals = [row for row in rows if row["role"] == "initial"]
            if not originals or {r["smiles"] for r in originals} != {panel["start_smiles"][start]}:
                raise ValueError("Arms must score the same original molecule")
            original = originals[0]["loss"]
            for row in originals:
                loss = row["loss"]
                if (loss is None) != (original is None) or (
                    loss is not None and not math.isclose(loss, original, abs_tol=1e-5, rel_tol=1e-6)
                ):
                    raise ValueError("Repeated starting-molecule scores differ")
            arms = ["sgd_short", "teacher", "sgd_restart"]
            if fit["status"] == "trained":
                arms += [*student_arms, "random_jump"]
            methods = {}
            for arm in arms:
                edits = [row for row in rows if row["arm"] == arm and row["role"] != "initial"]
                if sum(r["arm"] == arm for r in originals) != 1:
                    raise ValueError(f"Missing or duplicate arm {arm} for {start}")
                deltas = [row["loss_delta_from_valid_start"] for row in edits
                          if row["loss_delta_from_valid_start"] is not None]
                methods[arm] = dict(best_delta=min(deltas, default=None), candidates=len(edits),
                    scored=sum(row["status"] == "scored" for row in edits),
                    improved=[row["smiles"] for row in edits if row["loss_delta_from_valid_start"] is not None
                              and row["loss_delta_from_valid_start"] < -config["min_score_gain"]])
            data["holdout"].append(dict(start=start, original_loss=original, arms=methods))
        cohort_data.append(data)
    result = {}
    for name, (test, reference) in comparisons.items():
        blocks, complete, rates = [], [], []
        for cohort in cohort_data:
            differences, scored = [], []
            test_hits = reference_hits = 0
            for row in cohort["holdout"]:
                arms = row["arms"]
                if any(arm not in arms for arm in (*test, *reference)):
                    continue
                test_hit = any(arms[arm]["improved"] for arm in test)
                reference_hit = any(arms[arm]["improved"] for arm in reference)
                test_hits += test_hit
                reference_hits += reference_hit
                differences.append(int(test_hit) - int(reference_hit))
                if row["original_loss"] is not None:
                    scored.append(differences[-1])
            blocks.append(differences)
            complete.append(scored)
            rates.append(dict(seed=cohort["seed"], n=len(differences), usable_baseline=len(scored),
                              candidate_hits=test_hits, control_hits=reference_hits))
        margin = config["practical_rate_margin"]
        analysis = paired(blocks, margin)
        if (observed != expected or any(len(block) != config["test_starts"] for block in blocks)):
            analysis["decision"] = "incomplete_cohorts"
            analysis.pop("practical_gain_ruled_out", None)
        result[name] = dict(primary=analysis, scored_baseline_sensitivity=paired(complete, margin), cohorts=rates)
    analysis = dict(cohorts=cohort_data, comparisons=result)
    if selection_proof is not None:
        analysis.update(learning_rate=selection_proof[0], lr_selection_sha256=selection_proof[1])
    return analysis


def analyze_repair(root, config):
    result = read(root / "repair" / "repair.json")
    fresh = read(root / "repair" / "fresh" / "summary.json")
    if fresh["status"] != "completed":
        raise ValueError("Repair fresh scoring incomplete")
    scores = {(row["start"], row["arm"], row["role"]): row["loss"] for row in fresh["candidates"]}
    cases = [case for case in result["cases"] if case["eligibility"] == "eligible"]
    if len({case["start"] for case in cases}) != len(cases):
        raise ValueError("Repair inputs must come from distinct starts")
    rows = []
    for case in cases:
        name = f"{case['start']}-{case['step']}"
        methods = {}
        for arm in ("structural", *config["repair_schedules"]):
            report = case[arm]
            smiles = report["connected_smiles"]
            initial = scores.get((name, arm, "initial"))
            same = bool(smiles and smiles == case["input_smiles"])
            loss = initial if same else scores.get((name, arm, "repair"))
            edits = ((0 if same else report["edit_count"]) if arm == "structural" else
                     sum(report["edits"].values()) if report["edits"] is not None else None)
            local = bool(smiles and edits is not None and edits <= config["repair_edit_budget"])
            retained = (local and loss is not None and initial is not None and
                        loss <= initial + config["repair_score_margin"])
            methods[arm] = dict(connected=bool(smiles), unchanged=same, edits=edits, local=local,
                                input_loss=initial, output_loss=loss, retained=retained)
        rows.append(dict(start=case["start"], step=case["step"], valid_input=case["valid_input"], arms=methods))
    comparisons = {}
    for arm in config["repair_schedules"]:
        comparisons[arm] = {}
        for label, valid, key in (("invalid_local_recovery", False, "local"),
                                  ("valid_local_retention", True, "retained")):
            panel = [row["arms"] for row in rows if row["valid_input"] == valid]
            differences = [int(row[arm][key]) - int(row["structural"][key]) for row in panel]
            comparisons[arm][label] = dict(candidate=sum(row[arm][key] for row in panel),
                structural=sum(row["structural"][key] for row in panel),
                paired=paired([differences], config["practical_rate_margin"]))
    return dict(cases=rows, comparisons=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--latent", type=Path, nargs="*", default=[])
    parser.add_argument("--repair", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = {}
    if args.latent:
        result["latent"] = analyze_latent(args.latent, config)
    if args.repair:
        result["repair"] = analyze_repair(args.repair, config)
    if not result:
        parser.error("Provide at least one completed result root")
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
