"""Read-only campaign collection into tidy, deterministic CSV tables.

The primary endpoint is the highest native hard score among raw sanitized,
connected argmax molecules visited by a run. State zero is included, with a
separate endpoint for a new molecular identity. Missing/failed runs remain in
the planned denominator. Trajectory states are observations, not replicates.
This module requires only the Python standard library and loads no model.
"""

from __future__ import annotations

from array import array
import csv
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Iterable


SCHEMA_VERSION = "stage3-tidy-v1"
KEYS = ("campaign_id", "run_id", "condition_id", "family", "seed", "initialization")
LABELS = (*KEYS, "mock")
STATE_COLUMNS = (*KEYS, "update", "valid", "sanitized", "connected", "smiles",
                 "hard_score", "soft_score", "soft_hard_gap")
SUMMARY_COLUMNS = (*LABELS, "status", "planned_updates", "effective_updates", "updates_requested", "states_observed",
    "updates_observed", "complete_trajectory", "scientific_usable", "exclusion_reason", "any_valid",
                   "any_scored_valid", "valid_state_count", "valid_state_fraction",
                   "best_valid_hard_score", "best_valid_update", "best_valid_smiles",
                   "best_valid_is_initial", "best_new_valid_hard_score",
                   "best_new_valid_update", "best_new_valid_smiles", "any_new_valid",
                   "final_observed", "final_valid", "final_hard_score",
                   "final_smiles", "unique_valid_molecules", "unique_valid_hard_graphs")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _clean(value):
    """Keep machine-readable output finite; absent numerical evidence is null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def _flatten(value, prefix=""):
    output = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            output.update(_flatten(item, name))
        elif isinstance(item, (list, tuple)):
            output[name] = json.dumps(_clean(item), sort_keys=True, separators=(",", ":"))
        else:
            output[name] = _clean(item)
    return output


def _update(row):
    value = row.get("update", row.get("state"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Each state must have a nonnegative integer update")
    return value


def _valid(row):
    """An explicit false validity flag cannot be overridden by component flags."""
    sanitized = row.get("sanitized", row.get("valid", False))
    connected = row.get("connected")
    if connected is None:
        components = row.get("component_count", row.get("components", row.get("n_components", row.get("fragment_count"))))
        connected = components == 1 if components is not None else row.get("valid", False)
    return bool(sanitized and connected and row.get("valid", True))


def _score(row):
    value = row.get("hard_score")
    return float(value) if _finite(value) else None


def _identity(row):
    # Native scoring is cached by ordered graph, chemical diversity by SMILES.
    return row.get("hard_id", row.get("hard_graph_id", row.get("hard_identity", row.get("graph_hash"))))


def _best(rows):
    scored = [row for row in rows if _valid(row) and _score(row) is not None]
    # A tie selects its first occurrence, making the endpoint deterministic.
    return max(scored, key=lambda row: (_score(row), -_update(row)), default=None)


def _endpoint(prefix, row):
    return {f"{prefix}_hard_score": _score(row) if row else None,
            f"{prefix}_update": _update(row) if row else None,
            f"{prefix}_smiles": row.get("smiles") if row else None,
            f"{prefix}_hard_id": _identity(row) if row else None}


def summarize_run(config, states: Iterable[dict], candidates: Iterable[dict] = ()):
    """Summarize one seed/condition without repairing or rescoring molecules.

    ``config`` contains run keys plus ``updates`` (default 500). ``states`` uses
    update, sanitized, connected, valid, hard_score and canonical smiles. An
    ordered graph hash may be supplied as hard_graph_id. Candidate records are
    retained separately; they cannot substitute for missing trajectory states.
    """
    states = sorted(list(states), key=_update)
    candidates = list(candidates)
    updates = [_update(row) for row in states]
    if len(updates) != len(set(updates)):
        raise ValueError("Duplicate trajectory update in one run")
    requested = config.get("updates", config.get("epochs", 500))
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("updates must be a positive integer")
    planned = config.get("planned_updates", requested)
    if isinstance(planned, bool) or not isinstance(planned, int) or planned < requested:
        raise ValueError("planned_updates must be an integer at least as large as effective updates")
    mock = config.get("mock", False)
    if not isinstance(mock, bool):
        raise ValueError("mock must be an explicit boolean")
    if planned != requested and not mock:
        raise ValueError("Only explicitly labelled mock runs may reduce their effective budget")
    if updates and updates[-1] > requested:
        raise ValueError("Trajectory extends beyond the requested budget")
    initial = next((row for row in states if _update(row) == 0), None)
    initial_smiles = initial.get("smiles") if initial else None
    initial_graph = _identity(initial) if initial else None

    def is_new(row):
        if initial is None:
            return False  # Unknown initial identity is not evidence of novelty.
        if row.get("smiles") and initial_smiles:
            return row["smiles"] != initial_smiles
        if not _valid(initial):
            return True
        return bool(initial_graph and _identity(row) and _identity(row) != initial_graph)

    valid = [row for row in states if _valid(row)]
    new_valid = [row for row in valid if is_new(row)]
    best = _best(valid)
    best_new = _best(new_valid)
    final = next((row for row in states if _update(row) == requested), None)
    complete = updates == list(range(requested + 1))
    window = [row for row in states if max(1, requested - 99) <= _update(row) <= requested]
    window_valid = [row for row in window if _valid(row)]
    initial_score = _score(initial) if initial and _valid(initial) else None
    valid_smiles = {row["smiles"] for row in valid if row.get("smiles")}
    valid_graphs = {_identity(row) for row in valid if _identity(row)}
    row = {key: config.get(key) for key in KEYS}
    row.update({
        "status": config.get("status", "complete" if complete else "partial" if states else "missing"),
        "mock": mock,
        "planned_updates": planned,
        "effective_updates": requested,
        "updates_requested": requested,
        "states_observed": len(states),
        "updates_observed": max(updates, default=None),
        "complete_trajectory": complete,
        "complete_planned_trajectory": complete and planned == requested,
        "scientific_usable": config.get("scientific_usable", True),
        "exclusion_reason": config.get("exclusion_reason"),
        "any_valid": bool(valid),
        "any_scored_valid": best is not None,
        "valid_state_count": len(valid),
        "valid_state_fraction": len(valid) / len(states) if states else None,
        "valid_planned_state_fraction": len(valid) / (planned + 1),
        "valid_effective_state_fraction": len(valid) / (requested + 1),
        "scored_valid_state_count": sum(_score(item) is not None for item in valid),
        "valid_without_score_count": sum(_score(item) is None for item in valid),
        **_endpoint("best_valid", best),
        "best_valid_is_initial": _update(best) == 0 if best else None,
        "best_valid_is_initial_identity": not is_new(best) if best and initial else None,
        **_endpoint("best_new_valid", best_new),
        "any_new_valid": bool(new_valid),
        "initial_valid": _valid(initial) if initial else None,
        "initial_hard_score": initial_score,
        "best_valid_improvement_over_initial": _score(best) - initial_score
            if best and initial_score is not None else None,
        "best_new_valid_improvement_over_initial": _score(best_new) - initial_score
            if best_new and initial_score is not None else None,
        "final_observed": final is not None,
        "final_update": requested if final is not None else None,
        "final_valid": _valid(final) if final else None,
        "final_hard_score": _score(final) if final else None,
        "final_smiles": final.get("smiles") if final else None,
        "final_components": final.get("component_count", final.get("components", final.get("n_components", final.get("fragment_count"))))
            if final else None,
        "last100_states_observed": len(window),
        "last100_valid_state_count": len(window_valid),
        "last100_valid_state_fraction": len(window_valid) / len(window) if window else None,
        "unique_valid_molecules": len(valid_smiles),
        "unique_valid_hard_graphs": len(valid_graphs),
        "candidate_records": len(candidates),
        "valid_soft_hard_gap_mean": _mean(item.get("soft_hard_gap") for item in valid),
        "all_state_soft_hard_gap_mean": _mean(item.get("soft_hard_gap") for item in states),
        "all_state_gap_count": sum(_finite(item.get("soft_hard_gap")) for item in states),
    })
    for threshold in (0.02, 0.05, 0.1):
        gaps = [item.get("soft_hard_gap") for item in valid if _finite(item.get("soft_hard_gap"))]
        row[f"valid_gap_above_{threshold:g}_count"] = sum(item > threshold for item in gaps)
    return _clean(row)


def _mean(values):
    values = [float(value) for value in values if _finite(value)]
    return statistics.fmean(values) if values else None


def _read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _iter_jsonl(path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A collector may race the writer's last unflushed record.
                if not line.endswith("\n"):
                    break
                raise
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path}:{index + 1}")
            yield value


def _read_jsonl(path):
    return list(_iter_jsonl(path))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(_clean(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class _RowSpool:
    """One temporary JSONL snapshot; keep only row count and columns in RAM."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.columns = set()
        self.count = 0

    def append(self, row):
        self.columns.update(row)
        self.stream.write(json.dumps(_clean(row), separators=(",", ":"), allow_nan=False) + "\n")
        self.count += 1

    def close(self):
        if not self.stream.closed:
            self.stream.close()

    def __len__(self):
        return self.count

    def __iter__(self):
        self.close()
        with self.path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def _write_csv(path, rows, preferred):
    all_columns = rows.columns if isinstance(rows, _RowSpool) else set().union(*(row.keys() for row in rows)) if rows else set()
    columns = list(dict.fromkeys((*preferred, *sorted(all_columns - set(preferred)))))
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(_clean(row))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return columns


def _condition_summary(rows):
    groups = {}
    for row in rows:
        key = (row.get("condition_id"), row.get("initialization"), row.get("mock", False))
        groups.setdefault(key, []).append(row)
    output = []
    for key, runs in sorted(groups.items()):
        first = runs[0]
        usable = [row for row in runs if row.get("scientific_usable", True)]
        scores = [row["best_valid_hard_score"] for row in usable if _finite(row.get("best_valid_hard_score"))]
        output.append({
            "campaign_id": first.get("campaign_id"), "condition_id": key[0],
            "family": first.get("family"), "initialization": key[1],
            "mock": key[2],
            "runs_planned": len(runs),
            "runs_complete": sum(row.get("status") == "complete" for row in runs),
            "runs_complete_planned_trajectory": sum(row.get("status") == "complete" and row.get("complete_planned_trajectory") for row in runs),
            "runs_failed": sum(row.get("status") == "failed" for row in runs),
            "runs_missing": sum(row.get("status") == "missing" for row in runs),
            "runs_running_or_partial": sum(row.get("status") in ("running", "partial") for row in runs),
            "runs_excluded_scientifically": len(runs) - len(usable),
            "runs_with_valid_raw": sum(bool(row.get("any_valid")) for row in runs),
            "runs_with_valid": sum(bool(row.get("any_valid")) for row in usable),
            "runs_with_scored_valid": len(scores),
            "runs_with_new_valid": sum(bool(row.get("any_new_valid")) for row in usable),
            "runs_with_valid_final": sum(row.get("final_valid") is True for row in usable),
            "valid_run_yield": sum(bool(row.get("any_valid")) for row in usable) / len(runs),
            "new_valid_run_yield": sum(bool(row.get("any_new_valid")) for row in usable) / len(runs),
            "valid_final_yield": sum(row.get("final_valid") is True for row in usable) / len(runs),
            "best_valid_hard_score_mean_conditional": _mean(scores),
            "best_valid_hard_score_sd_conditional": statistics.stdev(scores) if len(scores) >= 2 else None,
            "best_valid_hard_score_mean_complete_conditional": _mean(row.get("best_valid_hard_score")
                for row in usable if row.get("status") == "complete" and row.get("complete_trajectory")),
            "best_new_valid_hard_score_mean_conditional": _mean(row.get("best_new_valid_hard_score") for row in usable),
            "mean_valid_planned_state_fraction": _mean(row.get("valid_planned_state_fraction") if row in usable else 0 for row in runs),
        })
    return output


def _compact_gaps(states):
    """About 13 bytes per state, rather than a campaign of Python row dicts."""
    return (array("I", (_update(row) for row in states)),
            bytearray(_valid(row) for row in states),
            array("d", (row["soft_hard_gap"] if _finite(row.get("soft_hard_gap")) else math.nan for row in states)))


def _gap_mapping(trace):
    if isinstance(trace, dict):
        return trace
    updates, valid, gaps = trace
    return {int(update): (bool(is_valid), float(gap)) for update, is_valid, gap in zip(updates, valid, gaps)}


def _paired_gap_metrics(arm, reference):
    """Descriptive means on identical updates, without treating states as seeds."""
    arm, reference = _gap_mapping(arm), _gap_mapping(reference)
    common = sorted(set(arm).intersection(reference))
    valid = [update for update in common if arm[update][0] and reference[update][0]]
    result = {"common_state_count": len(common), "common_valid_state_count": len(valid)}
    for prefix, selected in (("common_state", common), ("common_valid", valid)):
        selected = [update for update in selected
                    if _finite(arm[update][1]) and _finite(reference[update][1])]
        left = _mean(arm[update][1] for update in selected)
        right = _mean(reference[update][1] for update in selected)
        result[f"{prefix}_gap_state_count"] = len(selected)
        result[f"{prefix}_soft_hard_gap_mean"] = left
        result[f"reference_{prefix}_soft_hard_gap_mean"] = right
        result[f"delta_{prefix}_soft_hard_gap_mean"] = left - right if left is not None and right is not None else None
    return result


def _paired(rows, configs, gaps_by_run):
    by_key = {(row["condition_id"], row["seed"], row["initialization"]): row for row in rows}
    output = []
    for row in rows:
        config = configs[row["run_id"]]
        condition = config.get("condition", {})
        reference = config.get("reference_condition_id", condition.get("reference_condition_id", condition.get("paired_with")))
        if not reference or reference == row["condition_id"]:
            continue
        control = by_key.get((reference, row["seed"], row["initialization"]))
        pair = {key: row.get(key) for key in LABELS}
        arm_valid = bool(row.get("any_scored_valid"))
        reference_valid = bool(control and control.get("any_scored_valid"))
        outcome = ("both_valid" if arm_valid and reference_valid else "arm_only_valid" if arm_valid
                   else "reference_only_valid" if reference_valid else "neither_valid")
        arm_hardware = config.get("hardware", {})
        reference_hardware = configs[control["run_id"]].get("hardware", {}) if control else {}
        eligible = bool(control and row.get("scientific_usable", True) and control.get("scientific_usable", True))
        pair.update({"reference_condition_id": reference,
                     "reference_run_id": control.get("run_id") if control else None,
                     "status": row.get("status"),
                     "reference_status": control.get("status") if control else "unplanned",
                     "both_complete": bool(control and row.get("status") == control.get("status") == "complete"),
                     "both_complete_planned": bool(control and row.get("status") == control.get("status") == "complete"
                         and row.get("complete_planned_trajectory") and control.get("complete_planned_trajectory")),
                     "both_scored_valid": arm_valid and reference_valid,
                     "arm_has_scored_valid": arm_valid,
                     "reference_has_scored_valid": reference_valid,
                     "scored_valid_presence_delta": int(arm_valid) - int(reference_valid) if eligible else None,
                     "comparison_eligible": eligible,
                     "scientific_usable": row.get("scientific_usable", True),
                     "reference_scientific_usable": control.get("scientific_usable", True) if control else None,
                     "exclusion_reason": row.get("exclusion_reason"),
                     "reference_exclusion_reason": control.get("exclusion_reason") if control else None,
                     "validity_outcome": outcome if control else "reference_unplanned",
                     "arm_final_valid": row.get("final_valid"),
                     "reference_final_valid": control.get("final_valid") if control else None,
                     "same_effective_budget": row.get("effective_updates") == control.get("effective_updates") if control else None,
                     "arm_hardware": json.dumps(arm_hardware, sort_keys=True),
                     "reference_hardware": json.dumps(reference_hardware, sort_keys=True) if control else None,
                     "same_device_description": arm_hardware.get("devices") == reference_hardware.get("devices")
                         if arm_hardware.get("devices") and reference_hardware.get("devices") else None})
        for metric in ("best_valid_hard_score", "best_new_valid_hard_score", "valid_state_fraction",
                       "valid_planned_state_fraction", "unique_valid_molecules", "valid_soft_hard_gap_mean",
                       "all_state_soft_hard_gap_mean"):
            left, right = row.get(metric), control.get(metric) if control else None
            pair[metric] = left
            pair[f"reference_{metric}"] = right
            pair[f"delta_{metric}"] = left - right if eligible and _finite(left) and _finite(right) else None
        pair.update(_paired_gap_metrics(gaps_by_run.get(row["run_id"], {}),
            gaps_by_run.get(control["run_id"], {}) if control else {}))
        if not eligible:
            for metric in pair:
                if metric.startswith("delta_"):
                    pair[metric] = None
        output.append(pair)
    return output


def _collect_runs(actual, configs, saved_configs, campaign_mock, tables):
    """Keep one run in memory; spool campaign observations immediately."""
    gaps_by_run = {}
    for run_id, planned in sorted(configs.items()):
        folder = actual / run_id
        saved = saved_configs[run_id]
        for key in KEYS:
            if key in saved and key in planned and saved[key] != planned[key]:
                raise ValueError(f"Saved {key} differs from plan for {run_id}")
        for key in ("condition", "teacher", "projection", "cooling", "updates", "learning_rate"):
            if key in saved and key in planned and saved[key] != planned[key]:
                reduced_mock = (key == "updates" and campaign_mock and saved.get("mock") is True
                    and saved.get("planned_updates") == planned[key]
                    and isinstance(saved[key], int) and not isinstance(saved[key], bool)
                    and 1 <= saved[key] <= planned[key])
                if not reduced_mock:
                    raise ValueError(f"Saved scientific config {key} differs from plan for {run_id}")
        planned_updates = planned.get("updates", planned.get("epochs", 500))
        if "planned_updates" in saved and saved["planned_updates"] != planned_updates:
            raise ValueError(f"Saved planned_updates differs from plan for {run_id}")
        config = {**planned, "planned_updates": planned_updates, **saved, "mock": campaign_mock}
        status = _read_json(folder / "status.json", {})
        if isinstance(status, str):
            status = {"status": status}
        for key in ("status", "scientific_usable", "exclusion_reason"):
            if key in status:
                config[key] = status[key]
        configs[run_id] = config
        states = _read_jsonl(folder / "states.jsonl")
        candidates = _read_jsonl(folder / "candidates.jsonl")
        summary = summarize_run(config, states, candidates)
        gaps_by_run[run_id] = _compact_gaps(states)
        summary.update({f"config.{key}": value for key, value in _flatten(config).items()
                        if key not in KEYS and key != "status"})
        summary.update({f"status.{key}": value for key, value in _flatten(status).items()})
        for key in ("elapsed_seconds", "teacher_queries"):
            summary[key] = status.get(key)
        assessment = _read_json(folder / "final_assessment.json", {})
        summary.update(_flatten(assessment, "final_assessment"))
        summary["error"] = status.get("error")
        tables["runs"].append(summary)
        keys = {key: config.get(key) for key in LABELS}
        keys.update(scientific_usable=summary["scientific_usable"],
                    exclusion_reason=summary.get("exclusion_reason"))
        top_valid = _read_json(folder / "top_valid.json", {"molecules": []})
        if not isinstance(top_valid, dict) or not isinstance(top_valid.get("molecules"), list):
            raise ValueError(f"Expected top_valid.json with a molecules list for {run_id}")
        # Other streams are read one at a time as generators. State/candidate
        # lists above are bounded by this run's budget, never the whole campaign.
        streams = {"states": states, "candidates": candidates, "top_valid": top_valid["molecules"]}
        for name in ("states", "candidates", "gradients", "operators", "teacher", "top_valid"):
            observations = streams[name] if name in streams else _iter_jsonl(folder / f"{name}.jsonl")
            for observation in observations:
                for key in KEYS:
                    if key in observation and observation[key] != keys[key]:
                        raise ValueError(f"Conflicting {key} in {run_id}/{name}")
                tables[name].append({**keys, **_flatten(observation)})
                if name == "operators":
                    for index, step in enumerate(observation.get("steps", [])):
                        if not isinstance(step, dict) or not isinstance(step.get("operator"), str):
                            raise ValueError(f"Invalid operator step in {run_id} at update {_update(observation)}")
                        details = {key: value for key, value in step.items() if key != "operator"}
                        tables["operator_steps"].append({**keys, "update": _update(observation),
                            "step_index": index, "operator_name": step["operator"],
                            "mode": observation.get("mode"), "applied": observation.get("applied"),
                            **_flatten(details)})
    return gaps_by_run


def _condition_rows(plan, configs, campaign_mock):
    """Small planned-setting table for direct joins without parsing filenames."""
    conditions = {}
    for config in configs.values():
        condition_id = config.get("condition_id")
        conditions.setdefault(condition_id, {"id": condition_id, "family": config.get("family"),
                                            **config.get("condition", {})})
    for condition in plan.get("conditions", []):
        if condition.get("id") in conditions:
            conditions[condition["id"]] = condition
    return [{"campaign_id": plan.get("campaign_id"), "condition_id": condition_id,
             "mock": campaign_mock, **_flatten({key: value for key, value in condition.items() if key != "id"})}
            for condition_id, condition in sorted(conditions.items())]


def write_campaign_tables(root):
    """Collect planned runs under ``root/runs``; write only ``root/tables``.

    plan.json must contain a runs list of full run configs. Missing planned runs
    are emitted with status=missing. Unexpected run directories are rejected to
    avoid silently mixing another campaign. Running rows are valid snapshots;
    the collector is safely repeatable while workers append new observations.
    """
    root = Path(root).resolve()
    plan = _read_json(root / "plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list):
        raise ValueError("plan.json must contain a runs list")
    configs = {}
    replicate_keys = set()
    for item in plan["runs"]:
        config = {"campaign_id": plan.get("campaign_id"), **item}
        run_id = config.get("run_id")
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            raise ValueError("Each planned run_id must be a unique directory name")
        if run_id in configs:
            raise ValueError(f"Duplicate planned run_id: {run_id}")
        replicate = (config.get("condition_id"), config.get("seed"), config.get("initialization"))
        if replicate in replicate_keys:
            raise ValueError(f"Duplicate planned condition/seed/initialization: {replicate}")
        replicate_keys.add(replicate)
        configs[run_id] = config
    actual = root / "runs"
    if actual.exists():
        unplanned = {path.name for path in actual.iterdir() if path.is_dir()} - set(configs)
        if unplanned:
            raise ValueError(f"Unplanned run directories: {sorted(unplanned)}")
    saved_configs = {run_id: _read_json(actual / run_id / "config.json", {}) for run_id in configs}
    observed_modes = {saved.get("mock", False) for saved in saved_configs.values() if saved}
    if any(not isinstance(mode, bool) for mode in observed_modes) or len(observed_modes) > 1:
        raise ValueError("A campaign cannot mix mock and native run configurations")
    campaign_mock = plan.get("mock", next(iter(observed_modes), False))
    if not isinstance(campaign_mock, bool) or any(mode != campaign_mock for mode in observed_modes):
        raise ValueError("Saved mock/native mode differs from the campaign plan")
    destination = root / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    # A collector lease prevents two simultaneous refreshes from interleaving.
    lock = destination / ".collector.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    temporary = None
    tables = {"runs": []}
    try:
        temporary = tempfile.TemporaryDirectory(prefix=".collect-", dir=destination)
        spool_root = Path(temporary.name).resolve()
        if spool_root.parent != destination.resolve():
            raise ValueError("Collector temporary files must remain within tables")
        for name in ("states", "candidates", "gradients", "operators", "operator_steps", "teacher", "top_valid"):
            tables[name] = _RowSpool(spool_root / f"{name}.jsonl")
        gaps_by_run = _collect_runs(actual, configs, saved_configs, campaign_mock, tables)
        tables["paired"] = _paired(tables["runs"], configs, gaps_by_run)
        tables["condition_summary"] = _condition_summary(tables["runs"])
        tables["conditions"] = _condition_rows(plan, configs, campaign_mock)
        columns = {}
        for name, rows in tables.items():
            preferred = (SUMMARY_COLUMNS if name == "runs" else (*STATE_COLUMNS, "mock") if name == "states"
                else (*LABELS, "update", "step_index", "operator_name") if name == "operator_steps"
                else ("campaign_id", "condition_id", "family", "mock", "question") if name == "conditions"
                else (*LABELS, "rank", "hard_score", "smiles", "hard_id", "update") if name == "top_valid" else LABELS)
            columns[name] = _write_csv(destination / f"{name}.csv", rows, preferred)
        schema = {
            "schema_version": SCHEMA_VERSION,
            "primary_endpoint": "best_valid_hard_score: maximum native hard contact score among raw sanitized and connected argmax states, including update zero",
            "score_direction": "higher is better; contact proxy, not binding affinity",
            "experimental_unit": "one seed × initialization × condition run; states are correlated observations",
            "mock": campaign_mock,
            "mode": "All tables carry mock. Mock runs are software fixtures, excluded from native scientific conclusions.",
            "budget": "planned_updates retains the design budget; effective_updates/updates_requested describe this execution. Only explicitly labelled mock runs may reduce the effective budget. final_* uses its effective endpoint; complete_planned_trajectory remains false for shortened mocks.",
            "missing_values": "empty CSV cells/null JSON mean unavailable evidence, never a zero score",
            "planned_denominator": "condition_summary yields retain failed, partial and missing planned runs",
            "scientific_exclusions": "Raw run scores/data are retained when scientific_usable=false (for example a frozen-model/context failure). Such runs remain in planned denominators, contribute no valid yield or conditional score, and have comparison_eligible=false with null paired deltas. Other failed/partial runs retain their observed evidence.",
            "conditional_means": "score means use scored valid runs only; interpret with runs_planned and yield",
            "identity": "hard_id (hard_graph_id also accepted) distinguishes atom ordering for native cache; canonical smiles defines chemical diversity",
            "novelty": "best_new_valid excludes initial canonical molecular identity; with no initial state novelty is unknown",
            "best_selection": "first occurrence wins exact hard-score ties",
            "gap": "soft_hard_gap is soft_score minus hard_score; positive means relaxed-score optimism",
            "paired_gaps": "All-state means include invalid states to avoid conditioning on an arm's success. common_state_* uses identical observed updates in both arms; common_valid_* uses only identical updates valid in both. *_gap_state_count reports pairs with finite gaps. These are descriptive per-run summaries; states are correlated, not independent replicates.",
            "pairing": "paired.csv uses explicit reference_condition_id or condition.paired_with, same seed and initialization. No-valid score deltas are null; validity_outcome retains which arm succeeded. Hardware descriptions and budget/completion flags are covariates, not equality assertions.",
            "teacher": "teacher.csv has one row per recorded native teacher query. Nested response metadata are flattened under metrics.*, preserving excluded vocabulary mass and one-forward diagnostics.",
            "operator_steps": "operator_steps.csv explodes operators.jsonl.steps in recorded order using zero-based step_index. Numeric movement, component and projection diagnostics are columns; nested details use dotted names. operators.csv retains the original full sequence.",
            "top_valid": "top_valid.csv collects each run's top_valid.json molecules (up to 20 distinct valid connected canonical identities), with rank, score, ordered graph identity and artifact paths. The complete candidate stream remains in candidates.csv.",
            "conditions": "conditions.csv contains one planned condition with flattened settings/requirements for joins; it does not expand schedules into extra rows.",
            "collection_memory": "Observation rows stream through temporary JSONL snapshots and into CSV; only one run's state/candidate records, small run summaries, and compact 13-byte/state gap arrays are retained. Temporary snapshots are deleted after collection. Inputs remain unchanged.",
            "final_assessment": "Optional final_assessment.json fields are flattened under final_assessment.* in runs.csv; absent assessments remain unavailable.",
            "last100": "updates max(1, requested-99) through requested, inclusive",
            "columns": columns,
            "rows": {name: len(rows) for name, rows in tables.items()},
        }
        _atomic_json(destination / "schema.json", schema)
    finally:
        try:
            for rows in tables.values():
                if isinstance(rows, _RowSpool):
                    rows.close()
            if temporary is not None:
                temporary.cleanup()
        finally:
            lock.unlink()
    return {"tables": str(destination), "rows": {name: len(rows) for name, rows in tables.items()}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Campaign output root containing plan.json")
    args = parser.parse_args()
    print(json.dumps(write_campaign_tables(args.root), indent=2, allow_nan=False))
