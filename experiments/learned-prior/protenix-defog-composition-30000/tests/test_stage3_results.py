"""Endpoint and missing-data semantics for plot-ready Stage 3 outputs."""

import csv
import gzip
import importlib.util
import json
from pathlib import Path
import tracemalloc
import zlib

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts" / "stage3_results.py"
SPEC = importlib.util.spec_from_file_location("stage3_results", PATH)
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


def state(update, score, smiles="CC", *, connected=True, sanitized=True, graph=None):
    return {"update": update, "hard_score": score, "smiles": smiles,
            "sanitized": sanitized, "connected": connected,
            "hard_graph_id": graph or f"graph-{update}", "soft_hard_gap": 0.03}


def test_primary_is_best_valid_connected_not_final_or_best_soft():
    rows = [state(0, 0.1), state(1, 0.9, "C.C", connected=False),
            state(2, 0.4, "CO"), state(3, 0.2, "CN")]
    summary = results.summarize_run({"updates": 3}, rows)
    assert summary["best_valid_hard_score"] == 0.4
    assert summary["best_valid_update"] == 2
    assert summary["best_valid_smiles"] == "CO"
    assert summary["final_hard_score"] == 0.2
    assert summary["valid_state_count"] == 3
    assert summary["best_new_valid_hard_score"] == 0.4


def test_initial_only_is_explicit_and_chemical_identity_ignores_atom_order():
    rows = [state(0, 0.4, graph="ordered-a"), state(1, 0.3, graph="ordered-b")]
    summary = results.summarize_run({"updates": 1}, rows)
    assert summary["best_valid_is_initial"] is True
    assert summary["any_new_valid"] is False
    assert summary["best_new_valid_hard_score"] is None
    assert summary["unique_valid_molecules"] == 1
    assert summary["unique_valid_hard_graphs"] == 2


def test_missing_terminal_does_not_relabel_last_state_as_final():
    summary = results.summarize_run({"updates": 500}, [state(0, 0.2), state(1, 0.4, "CO")])
    assert summary["final_valid"] is None
    assert summary["final_observed"] is False
    assert summary["status"] == "partial"
    assert summary["best_valid_hard_score"] == 0.4
    assert summary["valid_planned_state_fraction"] == 2 / 501


def test_nonfinite_score_is_missing_not_zero_or_success():
    summary = results.summarize_run({"updates": 1}, [state(0, float("nan"))])
    assert summary["any_valid"] is True
    assert summary["any_scored_valid"] is False
    assert summary["best_valid_hard_score"] is None
    assert summary["valid_without_score_count"] == 1
    json.dumps(summary, allow_nan=False)


def test_duplicate_updates_are_rejected():
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        results.summarize_run({"updates": 1}, [state(0, 0.1), state(0, 0.2)])


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_collector_keeps_failed_and_missing_planned_runs_and_pairs(tmp_path):
    runs = [{"run_id": f"r-{condition}-{seed}", "condition_id": condition,
             "family": "weight", "seed": seed, "initialization": "sparse", "updates": 1,
             "reference_condition_id": "control" if condition == "guided" else None,
             "condition": {"teacher": {"weight": 0 if condition == "control" else 0.0625}}}
            for condition in ("control", "guided") for seed in (17, 23)]
    write(tmp_path / "plan.json", {"campaign_id": "pilot", "runs": runs})
    for config in runs[:3]:
        folder = tmp_path / "runs" / config["run_id"]
        write(folder / "config.json", config)
        write(folder / "status.json", {"status": "failed" if config["seed"] == 23 else "complete"})
        rows = [state(0, 0.1), state(1, 0.3 if config["condition_id"] == "guided" else 0.2, "CO")]
        (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = (tmp_path / "runs" / runs[0]["run_id"] / "states.jsonl").read_bytes()
    report = results.write_campaign_tables(tmp_path)
    assert report["rows"]["runs"] == 4
    summaries = csv_rows(tmp_path / "tables" / "condition_summary.csv")
    guided = next(row for row in summaries if row["condition_id"] == "guided")
    assert guided["runs_planned"] == "2"
    assert guided["runs_missing"] == "1"
    assert guided["valid_run_yield"] == "0.5"
    paired = csv_rows(tmp_path / "tables" / "paired.csv")
    paired17 = next(row for row in paired if row["seed"] == "17")
    assert float(paired17["delta_best_valid_hard_score"]) == pytest.approx(0.1)
    paired23 = next(row for row in paired if row["seed"] == "23")
    assert paired23["delta_best_valid_hard_score"] == ""
    assert paired23["status"] == "missing"
    flattened = csv_rows(tmp_path / "tables" / "runs.csv")
    assert "config.condition.teacher.weight" in flattened[0]
    assert before == (tmp_path / "runs" / runs[0]["run_id"] / "states.jsonl").read_bytes()
    assert results.write_campaign_tables(tmp_path) == report


def test_collector_does_not_mix_unplanned_runs(tmp_path):
    write(tmp_path / "plan.json", {"campaign_id": "x", "runs": []})
    (tmp_path / "runs" / "someone-else").mkdir(parents=True)
    with pytest.raises(ValueError, match="Unplanned"):
        results.write_campaign_tables(tmp_path)


def test_active_writer_partial_last_record_is_ignored(tmp_path):
    path = tmp_path / "states.jsonl"
    path.write_text(json.dumps(state(0, 0.1)) + "\n{\"update\":", encoding="utf-8")
    assert len(results._read_jsonl(path)) == 1


def test_collector_lease_blocks_duplicate_writer(tmp_path):
    write(tmp_path / "plan.json", {"runs": []})
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / ".collector.lock").write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        results.write_campaign_tables(tmp_path)


def test_duplicate_replicate_cannot_inflate_seed_denominator(tmp_path):
    write(tmp_path / "plan.json", {"runs": [
        {"run_id": "first", "condition_id": "a", "seed": 17, "initialization": "sparse"},
        {"run_id": "duplicate", "condition_id": "a", "seed": 17, "initialization": "sparse"},
    ]})
    with pytest.raises(ValueError, match="Duplicate planned condition"):
        results.write_campaign_tables(tmp_path)


def test_saved_scientific_settings_cannot_silently_change(tmp_path):
    config = {"run_id": "first", "condition_id": "a", "seed": 17,
              "initialization": "sparse", "condition": {"teacher": {"weight": 0.0625}}}
    write(tmp_path / "plan.json", {"runs": [config]})
    changed = {**config, "condition": {"teacher": {"weight": 0.5}}}
    write(tmp_path / "runs" / "first" / "config.json", changed)
    with pytest.raises(ValueError, match="Saved scientific config"):
        results.write_campaign_tables(tmp_path)


def test_hard_id_and_component_count_match_runner_schema():
    rows = [{**state(0, 0.1), "hard_id": "ordered-one", "component_count": 1},
            {**state(1, 0.2, "CO"), "hard_id": "ordered-two", "component_count": 1}]
    summary = results.summarize_run({"updates": 1}, rows)
    assert summary["best_valid_hard_id"] == "ordered-two"
    assert summary["unique_valid_hard_graphs"] == 2
    assert summary["final_components"] == 1


def test_mock_effective_budget_requires_explicit_label_and_retains_plan():
    rows = [state(0, 0.1), state(1, 0.2)]
    summary = results.summarize_run({"updates": 1, "planned_updates": 500, "mock": True}, rows)
    assert summary["mock"] is True
    assert summary["planned_updates"] == 500
    assert summary["effective_updates"] == 1
    assert summary["complete_trajectory"] is True
    assert summary["complete_planned_trajectory"] is False
    assert summary["valid_planned_state_fraction"] == 2 / 501
    with pytest.raises(ValueError, match="explicitly labelled mock"):
        results.summarize_run({"updates": 1, "planned_updates": 500}, rows)


def test_collect_mock_teacher_status_and_assessment_metadata(tmp_path):
    config = {"run_id": "smoke", "condition_id": "a", "seed": 17,
              "initialization": "sparse", "updates": 500}
    write(tmp_path / "plan.json", {"mock": True, "runs": [config]})
    folder = tmp_path / "runs" / "smoke"
    write(folder / "config.json", {**config, "updates": 1, "planned_updates": 500,
                                   "mock": True, "hardware": {"devices": ["CpuDevice(id=0)"]}})
    write(folder / "status.json", {"status": "complete", "elapsed_seconds": 1.2, "teacher_queries": 1})
    write(folder / "final_assessment.json", {"repair": {"witness": None}, "budget": 20})
    (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [state(0, 0.1), state(1, 0.2)]), encoding="utf-8")
    (folder / "teacher.jsonl").write_text(json.dumps({"update": 0, "metrics": {"excluded_mass": 0.12}}) + "\n", encoding="utf-8")
    results.write_campaign_tables(tmp_path)
    row = csv_rows(tmp_path / "tables" / "runs.csv")[0]
    assert row["mock"] == "True"
    assert row["planned_updates"] == "500"
    assert row["effective_updates"] == "1"
    assert row["complete_planned_trajectory"] == "False"
    assert row["elapsed_seconds"] == "1.2"
    assert row["teacher_queries"] == "1"
    assert row["final_assessment.budget"] == "20"
    teacher = csv_rows(tmp_path / "tables" / "teacher.csv")[0]
    assert teacher["metrics.excluded_mass"] == "0.12"
    assert teacher["mock"] == "True"


def test_mock_override_without_retained_plan_is_rejected(tmp_path):
    config = {"run_id": "smoke", "condition_id": "a", "seed": 17,
              "initialization": "sparse", "updates": 500}
    write(tmp_path / "plan.json", {"mock": True, "runs": [config]})
    write(tmp_path / "runs" / "smoke" / "config.json", {**config, "updates": 1, "mock": True})
    with pytest.raises(ValueError, match="Saved scientific config updates"):
        results.write_campaign_tables(tmp_path)


def test_paired_with_no_valid_score_keeps_failure_and_null_delta(tmp_path):
    runs = [{"run_id": "control", "condition_id": "control", "seed": 17,
             "initialization": "sparse", "updates": 1},
            {"run_id": "arm", "condition_id": "arm", "seed": 17,
             "initialization": "sparse", "updates": 1,
             "condition": {"paired_with": "control"}}]
    write(tmp_path / "plan.json", {"runs": runs})
    for config in runs:
        folder = tmp_path / "runs" / config["run_id"]
        write(folder / "config.json", config)
        write(folder / "status.json", {"status": "complete"})
        rows = [state(0, 0.1, connected=config["run_id"] == "arm"),
                state(1, 0.2, connected=config["run_id"] == "arm")]
        (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    results.write_campaign_tables(tmp_path)
    row = csv_rows(tmp_path / "tables" / "paired.csv")[0]
    assert row["validity_outcome"] == "arm_only_valid"
    assert row["scored_valid_presence_delta"] == "1"
    assert row["delta_best_valid_hard_score"] == ""
    assert row["both_scored_valid"] == "False"


def test_operator_steps_are_numeric_columns_in_order_and_original_is_retained(tmp_path):
    config = {"run_id": "arm", "condition_id": "a", "seed": 17,
              "initialization": "sparse", "updates": 1}
    write(tmp_path / "plan.json", {"runs": [config]})
    folder = tmp_path / "runs" / "arm"
    write(folder / "config.json", config)
    record = {"update": 1, "mode": "connected", "applied": True, "steps": [
        {"operator": "bridge", "bond_displacement_l2": 0.2, "components_after": 1,
         "bridges": [[1, 2]]},
        {"operator": "post_bridge_valence", "bond_displacement_l2": 0.1,
         "components_after": 2, "diagnostics": {"residual": 0.01}},
    ]}
    (folder / "operators.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = results.write_campaign_tables(tmp_path)
    assert report["rows"]["operators"] == 1
    assert report["rows"]["operator_steps"] == 2
    rows = csv_rows(tmp_path / "tables" / "operator_steps.csv")
    assert [row["step_index"] for row in rows] == ["0", "1"]
    assert [row["operator_name"] for row in rows] == ["bridge", "post_bridge_valence"]
    assert float(rows[0]["bond_displacement_l2"]) == 0.2
    assert rows[1]["diagnostics.residual"] == "0.01"
    original = csv_rows(tmp_path / "tables" / "operators.csv")[0]
    assert json.loads(original["steps"])[1]["operator"] == "post_bridge_valence"


def test_common_valid_gaps_avoid_different_valid_subset_selection():
    arm = {0: (True, 0.1), 1: (True, 0.8), 2: (False, 0.4), 3: (True, 0.3), 4: (True, 0.9)}
    reference = {0: (True, 0.05), 1: (False, 0.2), 2: (True, 0.1), 3: (True, 0.15)}
    paired = results._paired_gap_metrics(arm, reference)
    assert paired["common_state_count"] == 4
    assert paired["common_valid_state_count"] == 2
    assert paired["common_valid_gap_state_count"] == 2
    assert paired["common_valid_soft_hard_gap_mean"] == pytest.approx(0.2)
    assert paired["reference_common_valid_soft_hard_gap_mean"] == pytest.approx(0.1)
    assert paired["delta_common_valid_soft_hard_gap_mean"] == pytest.approx(0.1)
    assert paired["common_state_soft_hard_gap_mean"] == pytest.approx(0.4)
    assert paired["reference_common_state_soft_hard_gap_mean"] == pytest.approx(0.125)
    assert paired["delta_common_state_soft_hard_gap_mean"] == pytest.approx(0.275)


def test_common_gap_no_overlap_or_nonfinite_remains_unknown():
    none = results._paired_gap_metrics({0: (True, 0.1)}, {1: (True, 0.2)})
    assert none["common_state_count"] == 0
    assert none["delta_common_valid_soft_hard_gap_mean"] is None
    missing = results._paired_gap_metrics({0: (True, float("nan"))}, {0: (True, 0.2)})
    assert missing["common_valid_state_count"] == 1
    assert missing["common_valid_gap_state_count"] == 0
    assert missing["delta_common_state_soft_hard_gap_mean"] is None


def test_run_summary_all_state_gap_includes_invalid_states():
    rows = [state(0, 0.1), {**state(1, 0.2, connected=False), "soft_hard_gap": 0.2}]
    summary = results.summarize_run({"updates": 1}, rows)
    assert summary["all_state_soft_hard_gap_mean"] == pytest.approx(0.115)
    assert summary["valid_soft_hard_gap_mean"] == pytest.approx(0.03)


def test_scientific_exclusion_keeps_raw_scores_but_excludes_yield_and_deltas(tmp_path):
    runs = [{"run_id": "control", "condition_id": "control", "seed": 17,
             "initialization": "sparse", "updates": 1},
            {"run_id": "arm", "condition_id": "arm", "seed": 17,
             "initialization": "sparse", "updates": 1,
             "condition": {"paired_with": "control"}}]
    write(tmp_path / "plan.json", {"runs": runs})
    for config in runs:
        folder = tmp_path / "runs" / config["run_id"]
        write(folder / "config.json", config)
        excluded = config["run_id"] == "arm"
        write(folder / "status.json", {"status": "failed" if excluded else "complete",
              "scientific_usable": not excluded,
              "exclusion_reason": "frozen_model_or_context_check_failed" if excluded else None})
        rows = [state(0, 0.1), state(1, 0.9 if excluded else 0.2, "CO")]
        (folder / "states.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    results.write_campaign_tables(tmp_path)
    arm = next(row for row in csv_rows(tmp_path / "tables" / "runs.csv") if row["run_id"] == "arm")
    assert arm["best_valid_hard_score"] == "0.9"
    assert arm["scientific_usable"] == "False"
    aggregate = next(row for row in csv_rows(tmp_path / "tables" / "condition_summary.csv") if row["condition_id"] == "arm")
    assert aggregate["runs_planned"] == "1"
    assert aggregate["runs_with_valid_raw"] == "1"
    assert aggregate["runs_with_valid"] == "0"
    assert aggregate["runs_excluded_scientifically"] == "1"
    assert aggregate["valid_run_yield"] == "0.0"
    assert aggregate["best_valid_hard_score_mean_conditional"] == ""
    pair = csv_rows(tmp_path / "tables" / "paired.csv")[0]
    assert pair["comparison_eligible"] == "False"
    assert pair["best_valid_hard_score"] == "0.9"
    assert pair["delta_best_valid_hard_score"] == ""
    assert pair["delta_common_valid_soft_hard_gap_mean"] == ""
    observed = [row for row in csv_rows(tmp_path / "tables" / "states.csv") if row["run_id"] == "arm"]
    assert all(row["scientific_usable"] == "False" for row in observed)
    assert observed[0]["exclusion_reason"] == "frozen_model_or_context_check_failed"


def test_compact_gap_trace_preserves_pair_results():
    rows = [state(update, 0.1, connected=update % 2 == 0) for update in range(501)]
    trace = results._compact_gaps(rows)
    assert not isinstance(trace, dict)
    assert len(trace[0]) * trace[0].itemsize + len(trace[1]) + len(trace[2]) * trace[2].itemsize == 501 * 13
    pair = results._paired_gap_metrics(trace, trace)
    assert pair["common_state_count"] == 501
    assert pair["common_valid_state_count"] == 251
    assert pair["delta_common_valid_soft_hard_gap_mean"] == pytest.approx(0)


def test_collection_streams_many_runs_and_retains_top_molecules(tmp_path):
    # Large text per state makes campaign-wide row retention observable without
    # creating a native/model workload. Twenty-four runs exceed a seed block.
    n_runs, updates, payload_size = 24, 40, 8000
    configs = [{"run_id": f"r-{seed:02d}", "condition_id": "control", "family": "reference",
                "seed": seed, "initialization": "sparse", "updates": updates,
                "condition": {"id": "control", "teacher": {"weight": {"kind": "constant", "value": 0}}}}
               for seed in range(n_runs)]
    write(tmp_path / "plan.json", {"campaign_id": "streaming-fixture", "runs": configs})
    for config in configs:
        folder = tmp_path / "runs" / config["run_id"]
        write(folder / "config.json", config)
        write(folder / "status.json", {"status": "complete"})
        with (folder / "states.jsonl").open("w", encoding="utf-8") as stream:
            for update in range(updates + 1):
                stream.write(json.dumps({**state(update, 0.1 + update / 1000),
                                         "large_metadata": "x" * payload_size}) + "\n")
        write(folder / "top_valid.json", {"molecules": [{"rank": 1, "hard_score": 0.14,
              "hard_id": "ordered-best", "smiles": "CC", "update": 40,
              "artifacts": {"probabilities": "top_valid/01.npz"}}]})
    tracemalloc.start()
    try:
        report = results.write_campaign_tables(tmp_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Retaining all state strings alone would consume >7 MB. Collection should
    # keep one run's records plus small summaries and compact gap arrays.
    assert peak < n_runs * (updates + 1) * payload_size / 2
    assert report["rows"]["runs"] == n_runs
    assert report["rows"]["states"] == n_runs * (updates + 1)
    assert report["rows"]["top_valid"] == n_runs
    assert report["rows"]["conditions"] == 1
    top = csv_rows(tmp_path / "tables" / "top_valid.csv")
    assert top[0]["rank"] == "1"
    assert top[0]["artifacts.probabilities"] == "top_valid/01.npz"
    assert not list((tmp_path / "tables").glob(".collect-*"))
    assert not (tmp_path / "tables" / ".collector.lock").exists()


def test_jsonl_gzip_fallback_matches_plain_stream(tmp_path):
    plain = tmp_path / "states.jsonl"
    rows = [state(0, 0.1), state(1, 0.2)]
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    plain.write_text(payload, encoding="utf-8")
    assert results._read_jsonl(plain) == rows
    plain.unlink()
    with gzip.open(str(plain) + ".gz", "wt", encoding="utf-8") as stream:
        stream.write(payload)
    assert results._read_jsonl(plain) == rows
    assert results._read_jsonl(Path(str(plain) + ".gz")) == rows


def test_active_gzip_flush_without_footer_retains_complete_rows(tmp_path):
    plain = tmp_path / "states.jsonl"
    compressed = Path(str(plain) + ".gz")
    rows = [state(0, 0.1), state(1, 0.2)]
    with compressed.open("wb") as raw:
        writer = gzip.GzipFile(fileobj=raw, mode="wb")
        writer.write("".join(json.dumps(row) + "\n" for row in rows).encode())
        writer.flush(zlib.Z_SYNC_FLUSH)
        raw.flush()
        assert results._read_jsonl(plain) == rows
        writer.close()
    assert results._read_jsonl(plain) == rows


def test_ambiguous_plain_and_gzip_stream_is_rejected(tmp_path):
    plain = tmp_path / "states.jsonl"
    plain.write_text("{}\n", encoding="utf-8")
    with gzip.open(str(plain) + ".gz", "wt", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="Ambiguous plain and compressed"):
        results._read_jsonl(plain)
