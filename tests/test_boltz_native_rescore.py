"""Selection checks for the later native, fresh-pose comparison; run on HPC."""

from latentmol.boltz.native_rescore import ARMS, _paired_native_results, select_candidates


def _row(arm, index, smiles, score):
    return {"arm": arm, "repeat": index // 8, "start_index": index % 8,
            "hard": {"smiles": smiles}, "start_hard": {"smiles": "C"},
            "hard_projection_score": {
                "fixed_context_discrete_affinity_probability_binary": score},
            "start_hard_projection_score": {
                "fixed_context_discrete_affinity_probability_binary": 0.5}}


def test_selection_freezes_unique_changed_candidates_per_arm():
    baseline = {"hard": {"smiles": "C"}}
    rows = [_row(arm, index, "CC" if index < 2 else "C", 0.7 if index else 0.9)
            for arm in ARMS for index in range(16)]
    rows[2]["hard"]["smiles"] = "CCC"
    rows[2]["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"] = 0.8
    rows[0]["start_hard"]["smiles"] = "CO"
    rows[2]["start_hard"]["smiles"] = "CN"
    selection = select_candidates(baseline, rows)
    assert selection["by_arm"]["target"]["eligible_attempts"] == 3
    assert [row["smiles"] for row in selection["by_arm"]["target"]["chosen"]] == ["CC", "CCC"]
    assert [row["start_smiles"] for row in selection["by_arm"]["target"]["chosen"]] == ["CO", "CN"]
    assert selection["unique_smiles"] == ["C", "CC", "CCC", "CN", "CO"]


def test_native_comparison_keeps_failed_pose_in_denominator():
    selection = {"baseline_smiles": "C", "unique_smiles": ["C", "CC"],
                 "by_arm": {arm: {"chosen": []} for arm in ARMS}}
    selection["by_arm"]["target"]["chosen"] = [{
        "smiles": "CC", "start_smiles": "C", "repeat": 0, "start_index": 0}]
    results = [
        {"smiles": "C", "seed": 17, "affinity": {
            "affinity_probability_binary": 0.4, "affinity_pred_value": 2.0}},
        {"smiles": "CC", "seed": 17, "affinity": {
            "affinity_probability_binary": 0.5, "affinity_pred_value": 1.0}},
        {"smiles": "C", "seed": 23, "affinity": None},
        {"smiles": "CC", "seed": 23, "affinity": {
            "affinity_probability_binary": 0.6, "affinity_pred_value": 1.0}},
    ]
    report = _paired_native_results(selection, results)["target"][0]
    assert report["successful_triples"] == 1
    assert report["positive_vs_both_on_both_seeds"] is False
