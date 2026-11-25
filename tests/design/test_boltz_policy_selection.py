from copy import deepcopy

import pytest

from latentmol.design.boltz_policy_selection import COUNTS, JOBS, SOURCE, select_policies


def fixtures():
    rows = []
    for seed, jobs in JOBS.items():
        for key, job in jobs.items():
            policy = "promote" if key == "extend" else key
            milestone = dict(updates=2400, elapsed_seconds=10, distinct_scored=2, fresh_attempts=2,
                             best_loss=-1, tenth_loss=None)
            row = dict(job_id=str(job), source_sha256=SOURCE, status="completed", full_objective="fixed_pose_dbmol",
                       case_name=key, case=dict(seed=seed, count=COUNTS[key], start="archive", fresh_budget=48,
                                               policy=policy, updates=2400),
                       updates=2400, fresh_budget=48, policy=policy, execution=dict(autocast="none", tf32=False),
                       start_hashes=[f"{seed}-{i}" for i in range(COUNTS[key])],
                       setup_seconds=5, total_seconds=20, milestones=[milestone],
                       fresh_attempts=2, distinct_scored=2, model_device=f"gpu-{seed}")
            rows.append(row)
    return rows


def test_sxm2_timing_repeat_preserves_controls_and_rejects_mixed_devices():
    rows = fixtures()
    first_seed = [row for row in rows if row["case"]["seed"] == 811211]
    for row in first_seed:
        row["model_device"] = "Tesla V100-SXM2-32GB"
    repeat = next(row for row in first_seed if row["case_name"] == "fixed150")
    repeat["job_id"] = "99"
    with pytest.raises(ValueError, match="Wrong job"):
        select_policies(rows)
    assert select_policies(rows, fixed150_sxm2_job=99)["panels"][0]["gpu"] == "Tesla V100-SXM2-32GB"
    repeat["model_device"] = "Tesla V100-PCIE-32GB"
    with pytest.raises(ValueError, match="SXM2"):
        select_policies(rows, fixed150_sxm2_job=99)


def test_selects_fixed_control_and_challenger_using_only_matching_time_prefixes():
    rows = fixtures()
    for row in rows:
        key = row["case_name"]
        if key == "extend":
            row.update(total_seconds=30, fresh_attempts=15, distinct_scored=15)
            # An appealing final panel is available only after the common horizon.
            final = dict(updates=2400, elapsed_seconds=22, distinct_scored=15, fresh_attempts=15,
                         best_loss=-50, tenth_loss=-40)
            row["milestones"][0].update(updates=1200, distinct_scored=1, best_loss=-5)
            row["milestones"].append(final)
        elif key == "fixed400":
            row.update(fresh_attempts=10, distinct_scored=10)
            row["milestones"][0].update(fresh_attempts=10, distinct_scored=10, tenth_loss=-2, best_loss=-4)
    result = select_policies(reversed(rows))
    assert result["status"] == "selected"
    assert result["selected_policies"] == ["fixed400", "fixed150"]
    assert all(panel["milestones"]["extend"]["distinct_scored"] == 1 for panel in result["panels"])
    assert all(panel["milestones"]["extend"]["seconds_to_ten"] is None for panel in result["panels"])


def test_splits_tied_ranks_and_prefers_extension_in_challenger_tie():
    rows = fixtures()
    for row in rows:
        if row["case_name"] == "fixed150":
            row["milestones"][0]["best_loss"] = -2.0
    result = select_policies(rows)
    assert result["selected_policies"] == ["fixed150", "extend"]
    assert result["rank_sums"] == {"fixed150": 3, "fixed400": 7.5, "extend": 7.5}
    assert select_policies(fixtures())["status"] == "inconclusive"
    assert select_policies(fixtures())["selected_policies"] == []


def test_sparse_count_cannot_beat_a_full_panel_and_tenth_loss_precedes_speed():
    rows = fixtures()
    for row in rows:
        key = row["case_name"]
        if key == "fixed400":
            row["milestones"][0]["best_loss"] = -100
        else:
            row.update(fresh_attempts=10, distinct_scored=10)
            row["milestones"][0].update(distinct_scored=10, fresh_attempts=10, best_loss=-5,
                tenth_loss=-3 if key == "extend" else -2, elapsed_seconds=14 if key == "extend" else 5)
    result = select_policies(rows)
    assert result["selected_policies"] == ["fixed150", "extend"]
    assert result["rank_sums"]["extend"] == 3


def test_rejects_partial_foreign_or_mispaired_results():
    rows = fixtures()
    with pytest.raises(ValueError, match="Missing completed"):
        select_policies(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        select_policies(rows + [rows[0]])
    for change in ({"job_id": 99}, {"status": "partial"}, {"source_sha256": "wrong"}, {"updates": 100}):
        with pytest.raises(ValueError, match="Wrong job"):
            select_policies([rows[0] | change, *rows[1:]])
    bad_prefix = deepcopy(rows)
    # Compare all twelve starts shared by extend and fixed150, not just six shared by all arms.
    event = next(r for r in bad_prefix if r["case_name"] == "extend")
    event["start_hashes"][-1] = "foreign"
    with pytest.raises(ValueError, match="Starting children differ"):
        select_policies(bad_prefix)
