"""Summarize saved matched ipSAE runs; no model execution or mutation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    steps = [row["step"] for row in rows]
    if len(steps) != len(set(steps)):
        raise ValueError(f"Duplicate steps in {path}")
    return rows


def summarize_arm(path):
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    arm = path / "arms" / "latent"
    metrics, candidates, updates = (read_rows(arm / name) for name in
                                    ("metrics.jsonl", "candidates.jsonl", "updates.jsonl"))
    metric_by_step = {row["step"]: row for row in metrics}
    candidate_by_step = {row["step"]: row for row in candidates}
    if (not metrics or set(metric_by_step) != set(candidate_by_step)
            or not set(row["step"] for row in updates).issubset(metric_by_step)):
        raise ValueError(f"Incompatible trajectory rows in {path}")
    profile = meta["native_identity"]["dbmol_profile"]
    weight = profile.get("ipsae_weight", 0.0)
    outer = profile["confidence_weight"]

    def comparable(row):
        # The saved binding loss includes -outer*weight*ipSAE. Undo only that
        # term, so both arms are compared on their original shared DBMol loss.
        score = row["score"]["binding_loss"]
        if weight:
            score += outer * weight * row["native_aux"]["ligand_ipsae"]
        return score

    valid = [candidate_by_step[row["step"]] for row in metrics
             if candidate_by_step[row["step"]]["valid"]]
    sources = [candidate_by_step[row["step"]] for row in updates]
    valid_sources = [row for row in sources if row["valid"]]
    valid_metrics = [row for row in metrics if candidate_by_step[row["step"]]["valid"]]
    best = min(valid_metrics, key=comparable) if valid_metrics else None
    endpoint = metrics[-1]
    final_candidate = candidate_by_step[endpoint["step"]]
    displacements = [row["displacement_norm"] for row in updates]
    return dict(status=result["status"], completed_step=result["completed_step"],
                updates=len(updates), sampled_states=len(metrics),
                ipsae_inner_weight=weight, confidence_outer_weight=outer,
                valid_update_sources=len(valid_sources),
                distinct_valid_update_smiles=len({row["smiles"] for row in valid_sources}),
                valid_states_including_endpoint=len(valid),
                distinct_valid_smiles_including_endpoint=len({row["smiles"] for row in valid}),
                capped_updates=sum(row["bound_factor"] < 1 for row in updates),
                median_displacement=statistics.median(displacements) if displacements else None,
                start_common_dbmol_loss=comparable(metrics[0]),
                final_common_dbmol_loss=comparable(endpoint),
                best_valid_common_dbmol_loss=(comparable(best) if best else None),
                best_valid_common_dbmol_step=(best["step"] if best else None),
                final_valid=final_candidate["valid"], final_smiles=final_candidate.get("smiles"),
                final_ipsae=endpoint["native_aux"].get("ligand_ipsae"),
                scope="Same fixed-pose DBMol objective on soft graphs; no fresh hard-graph rescore")


def summarize_probe(path):
    probe = json.loads(path.read_text(encoding="utf-8"))
    keys = ("run", "step", "ipsae", "original_confidence_loss", "original_confidence_gradient_norm",
            "ipsae_loss_gradient_norm", "ipsae_to_confidence_norm_ratio", "cosine", "angle_degrees",
            "effective_added_to_confidence_norm_ratio", "zero_gradient")
    return [{key: row.get(key) for key in keys} for row in probe["rows"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = dict(baseline=summarize_arm(args.root / "baseline"),
                  ipsae=summarize_arm(args.root / "ipsae"),
                  gradient_comparison=summarize_probe(args.root / "gradient-comparison.json"),
                  limitations=["Single matched start and fixed pose", "Common score is on relaxed graphs",
                               "ipSAE and confidence gradients are model surrogates, not experimental binding"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
