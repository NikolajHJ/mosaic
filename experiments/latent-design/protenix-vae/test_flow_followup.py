import json

import pytest

from flow_followup import analyze_latent, paired
from flow_lr_selection import config_digest, select_rate
from flow_pilot import short_candidates


def test_short_panel_includes_every_connected_edit_once():
    observations = [dict(step=i, smiles=s, valid=valid) for i, (s, valid) in
                    enumerate((("A", True), ("B", True), ("B", True), (None, False), ("C", True)))]
    panel = short_candidates(observations, "1", "flow", "D")
    assert [(row["step"], row["role"], row["smiles"]) for row in panel] == [
        (-1, "initial", "D"), (0, "proposal", "A"), (1, "edit", "B"), (4, "edit", "C")]


def test_paired_analysis_uses_discordant_starts_and_practical_margin():
    result = paired([[1, 0], [-1, 0]], .15)
    assert (result["n"], result["wins"], result["losses"], result["ties"], result["delta"]) == (4, 1, 1, 2, 0)
    assert result["decision"] == "inconclusive"


def test_portfolio_comparison_counts_same_start_once(tmp_path):
    def save(relative, data):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    config = dict(cohort_seeds=[10], checkpoint_sha256="test", test_starts=2, train_starts=1,
                  min_pairs=1, min_score_gain=.005, practical_rate_margin=.15)
    panel = dict(config=dict(seed=10, portfolio_test=True), starts=["1", "2", "3"], train=["1"],
                 start_smiles={"1": "N", "2": "C", "3": "O"}, checkpoint_sha256="test")
    save("panel.json", panel)
    save("students/fit.json", dict(status="trained", pairs=[dict(start="1")]))

    def fresh(arms):
        rows = []
        for start, loss in (("2", 1.), ("3", None)):
            for arm in arms:
                rows.append(dict(start=start, arm=arm, role="initial", loss=loss,
                                 smiles=panel["start_smiles"][start]))
                for smiles in ("CO", "CCO"):
                    delta = -.1 if loss is not None and arm == "residual" else None
                    rows.append(dict(start=start, arm=arm, role="edit", status="scored", smiles=smiles,
                                     loss_delta_from_valid_start=delta))
        return dict(status="completed", job_id="10", candidates=rows)

    save("fresh/summary.json", fresh(("sgd_short", "teacher")))
    save("students/fresh/summary.json", fresh(("sgd_restart", "residual", "flow", "random_jump")))
    result = analyze_latent([tmp_path], config)["comparisons"]["portfolio"]
    assert result["primary"]["n"] == 2
    assert result["primary"]["wins"] == 1
    assert result["scored_baseline_sensitivity"]["n"] == 1


def test_lr_selection_uses_gain_and_requires_all_tuning_cohorts(tmp_path):
    config = dict(learning_rates=[.3, 3, 150], tuning_seeds=[10, 20], tuning_starts_per_cohort=1,
                  checkpoint_sha256="ckpt", refine_updates=7, min_score_gain=.005)
    roots = []
    for seed, smiles in ((10, "C"), (20, "O")):
        root = tmp_path / str(seed)
        roots.append(root)
        (root / "fresh").mkdir(parents=True)
        trajectories = {}
        rows = []
        for arm, gain in (("lr_0.3", .03), ("lr_3", .05), ("lr_150", .05)):
            trajectories[arm] = dict(scorer_calls=8, bounded_steps=7 if arm == "lr_150" else 0)
            rows.extend([dict(start=str(seed), arm=arm, role="initial", smiles=smiles, loss=1.),
                         dict(start=str(seed), arm=arm, role="edit", status="scored",
                              loss_delta_from_valid_start=-gain)])
        (root / "tuning.json").write_text(json.dumps(dict(seed=seed, config_digest=config_digest(config),
            checkpoint_sha256="ckpt", job_id=str(seed), starts={str(seed): smiles},
            trajectories={str(seed): trajectories})), encoding="utf-8")
        (root / "fresh" / "summary.json").write_text(json.dumps(dict(status="completed", job_id=str(seed),
            candidates=rows)), encoding="utf-8")

    with pytest.raises(ValueError, match="complete prespecified"):
        select_rate(roots[:1], config)
    result = select_rate(roots, config)
    assert result["learning_rate"] == 3
    assert result["tuning_starts"] == {"10": "C", "20": "O"}
    assert result["table"][-1]["bounded_steps"] == 14
