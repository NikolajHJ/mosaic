"""Summarize pinned scalar forensic evidence on an allocated DTU HPC CPU.

This is an analysis addendum, not a gradient checker. It decodes no scientific
arrays, evaluates no model and never changes an admission, study or setting.
Relative agreement counts below are descriptive diagnostics, not gate passes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct


VERSION = "protenix-fixed-point-forensic-summary-v1"
REPORT_SHA256 = "acbb5273e0c794d5719067210e27a8031717fb8a51a21dc0eb19988881ee8359"
REPORT_BYTES = 7849579
ADMISSION_SHA256 = "3b4e6c14aa21aed083594ab040a7faf4a50f93b4adce5a4fa679108bc531aaf1"
SCALES = (0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003)
CONTACT_PATHS = ("primal_contacts", "ad_contacts", "primal_contacts64", "ad_contacts64")
COMPARISONS = ("direct_primal_vs_baseline_AD", "AD_scalar_vs_baseline_AD",
               "direct_primal_vs_endpoint_trapezoid", "AD_scalar_vs_endpoint_trapezoid")


def require(value, message):
    if not value:
        raise ValueError(message)


def allocated_cpu():
    require(os.environ.get("LSB_JOBID", "").isdigit() and os.environ.get("LSB_QUEUE") == "hpc",
            "Scalar scientific analysis requires an allocated DTU HPC hpc CPU job")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Expected a finite scalar")
    return float(value)


def mean(values):
    require(bool(values), "Empty scalar repeat cohort")
    return math.fsum(finite(value) for value in values) / len(values)


def read_pinned(path, supplied_sha, expected_sha, expected_bytes=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected a regular input file")
    raw = path.read_bytes()
    require(supplied_sha == expected_sha and digest(raw) == expected_sha,
            "Input bytes do not match the exact predeclared evidence SHA256")
    require(expected_bytes is None or len(raw) == expected_bytes, "Input byte count changed")
    return json.loads(raw), {"path": str(path.resolve()), "bytes": len(raw), "sha256": digest(raw)}


def comparison(observed, predicted, floor):
    """Mirror the historical relative-error denominator without assigning passes."""
    observed, predicted, floor = finite(observed), finite(predicted), finite(floor)
    residual = observed - predicted
    denominator = max(abs(observed), abs(predicted))
    relative = abs(residual) / denominator if denominator else None
    same_sign = observed * predicted > 0
    return {"observed_difference": observed, "prediction": predicted,
            "residual": residual, "absolute_residual": abs(residual),
            "relative_error": relative, "same_nonzero_sign": same_sign,
            "same_sign_and_relative_error_at_most_15_percent": bool(
                same_sign and relative is not None and relative <= .15),
            "absolute_residual_at_most_old_reported_floor": abs(residual) <= floor,
            "largest_difference_above_old_reported_floor": denominator > floor}


def point_scalar(points, identity, field):
    return mean([item[field] for item in points[identity]["evaluations"]])


def validate(report, admission):
    require(report["version"] == "protenix-fixed-point-forensics-v1"
            and report["status"] == "diagnostics_completed_original_failure_preserved",
            "Forensics must have completed, without replacing the original failure")
    require(report["original_failed_admission_status"] == "failed"
            and report["original_run_failed_preserved"] is True
            and report["can_admit_new_child"] is False
            and report["can_certify_or_restart_original_study"] is False
            and report["original_optimization_settings_changed"] is False
            and report["original_scientific_gate_replaced"] is False
            and report["original_payload_unchanged"] is True
            and report["frozen_unchanged"] is True
            and report["frozen_before"] == report["frozen_after"]
            and report["failures"] == [], "Original failure, frozen/input or no-admission flags changed")
    require(admission["version"] == "prospective-endpoint-admission-v1"
            and admission["status"] == "failed"
            and admission["can_admit_new_child"] is False
            and admission["can_certify_original_study"] is False
            and admission["original_run_failed_preserved"] is True,
            "The original admission must remain failed")
    require(report["original_admission_sha256"] == digest(canonical(admission))
            and report["original_admission_receipt_sha256"] == admission["receipt_sha256"]
            and admission["receipt_sha256"] == digest(canonical({
                key: value for key, value in admission.items() if key != "receipt_sha256"}))
            and report["original_plan"] == admission["plan"], "Original admission or fixed plan binding changed")
    require(report["unique_graph_count"] == len(report["points"]) == 108
            and len(report["directions"]) == len(admission["plan"]["directions"]) == 9,
            "Exact 108-point/nine-direction cohort changed")
    points = {item["graph_id"]: item for item in report["points"]}
    require(len(points) == 108 and report["baseline_graph_id"] in points, "Duplicate or missing fixed point")
    for identity, point in points.items():
        require(point["can_certify_or_restart_original_study"] is False, "Point scope changed")
        expected_repeats = 3 if identity == report["baseline_graph_id"] else 1
        require(len(point["evaluations"]) == expected_repeats, "Fixed point repeats changed")
        for evaluation in point["evaluations"]:
            require(evaluation["can_certify_or_restart_original_study"] is False,
                    "Evaluation scope changed")
            finite(evaluation["primal"])
            finite(evaluation["ad_value"])
            require(set(evaluation["contacts"]) == set(CONTACT_PATHS), "Reduction control cohort changed")
    for index, direction in enumerate(report["directions"]):
        original = admission["plan"]["directions"][index]
        require(direction["direction_index"] == index
                and all(direction[key] == original[key] for key in ("field", "row", "source", "target", "roles"))
                and tuple(probe["epsilon"] for probe in direction["probes"]) == SCALES,
                "Direction identity or predeclared six scales changed")
        for probe, old in zip(direction["probes"], original["probes"]):
            require(probe["original_fixed_probe"] == old and probe["scheme"] == old["scheme"],
                    "Fixed probe changed")
            require(probe["analysis"]["evaluated"] is (probe["scheme"] != "unavailable"),
                    "Feasibility/evaluation mismatch")
    return points


def summarize_row(direction, probe, original, points, floor):
    row = {key: direction[key] for key in ("direction_index", "field", "row", "source", "target", "roles")}
    row.update(epsilon=probe["epsilon"], scheme=probe["scheme"],
               feasible=probe["scheme"] != "unavailable", evaluated=probe["analysis"]["evaluated"],
               original_probe_scheme=probe["original_fixed_probe"]["scheme"],
               old_baseline_resolution_floor=floor,
               comparison_floor_source="original_failed_admission.baseline_checks.resolution_floor",
               old_probe_reported_resolution_floor=original.get("resolution_floor"),
               old_probe_comparison_resolved=original.get("comparison_resolved"),
               old_probe_agrees=original.get("agrees"),
               diagnostic_only=True, can_admit_new_child=False)
    if not row["evaluated"]:
        row["reason"] = probe["analysis"]["reason"]
        return row
    analysis = probe["analysis"]
    left, right = probe["left"], probe["right"]
    direct = point_scalar(points, right, "primal") - point_scalar(points, left, "primal")
    matched = point_scalar(points, right, "ad_value") - point_scalar(points, left, "ad_value")
    require(direct == analysis["independent_primal_difference"]
            and matched == analysis["AD_returned_scalar_difference"], "Scalar differences do not reproduce")
    predicted = finite(analysis["baseline_AD_dot_actual_displacement"])
    transport = finite(analysis["endpoint_gradient_trapezoid_dot_actual_displacement"])
    row.update(left_graph_id=left, right_graph_id=right, direct_primal_difference=direct,
               AD_returned_scalar_difference=matched, baseline_AD_prediction=predicted,
               endpoint_gradient_trapezoid_prediction=transport,
               scalar_path_difference=direct - matched,
               absolute_scalar_path_difference=abs(direct - matched),
               prediction_at_most_old_reported_floor=abs(predicted) <= floor,
               prediction_within_four_old_reported_floors=abs(predicted) <= 4 * floor,
               scalar_path_difference_exceeds_old_reported_floor=abs(direct - matched) > floor,
               endpoint_curvature_prediction_difference=transport - predicted,
               endpoint_curvature_difference_exceeds_old_reported_floor=abs(transport - predicted) > floor,
               comparisons=dict(zip(COMPARISONS, (
                   comparison(direct, predicted, floor), comparison(matched, predicted, floor),
                   comparison(direct, transport, floor), comparison(matched, transport, floor)))))
    direct_error = abs(direct - predicted)
    matched_error = abs(matched - predicted)
    row["matched_scalar_path_residual_change"] = matched_error - direct_error
    row["matched_scalar_path_relation"] = ("improves" if matched_error < direct_error else
                                             "worsens" if matched_error > direct_error else "equal")
    row["reduction_controls"] = {}
    for path in CONTACT_PATHS:
        control = analysis[path + "_reduction_control"]
        left_changes, right_changes = control["left_changed_winner_rows"], control["right_changed_winner_rows"]
        row["reduction_controls"][path] = {
            "best_difference": finite(control["best_difference"]),
            "fixed_baseline_winner_difference": finite(control["fixed_baseline_winner_difference"]),
            "best_minus_fixed_winner_difference": control["best_difference"] - control["fixed_baseline_winner_difference"],
            "left_changed_winner_rows": left_changes, "right_changed_winner_rows": right_changes,
            "any_endpoint_winner_change": bool(left_changes or right_changes)}
    row["reduction64_minus_32_best_difference"] = {
        path: row["reduction_controls"][path + "64"]["best_difference"]
              - row["reduction_controls"][path]["best_difference"]
        for path in ("primal_contacts", "ad_contacts")}
    return row


def aggregate(rows):
    evaluated = [row for row in rows if row["evaluated"]]
    counts = {"probe_rows": len(rows), "feasible_rows": sum(row["feasible"] for row in rows),
              "evaluated_rows": len(evaluated), "unavailable_rows": len(rows) - len(evaluated),
              "matched_scalar_path_relation": dict(Counter(row["matched_scalar_path_relation"] for row in evaluated)),
              "prediction_at_most_old_reported_floor": sum(row["prediction_at_most_old_reported_floor"] for row in evaluated),
              "prediction_within_four_old_reported_floors": sum(row["prediction_within_four_old_reported_floors"] for row in evaluated),
              "scalar_path_difference_exceeds_old_reported_floor": sum(row["scalar_path_difference_exceeds_old_reported_floor"] for row in evaluated),
              "endpoint_curvature_difference_exceeds_old_reported_floor": sum(row["endpoint_curvature_difference_exceeds_old_reported_floor"] for row in evaluated),
              "comparisons": {}, "contact_winner_change_probe_counts": {}}
    for name in COMPARISONS:
        comparisons = [row["comparisons"][name] for row in evaluated]
        counts["comparisons"][name] = {
            "same_sign_and_relative_error_at_most_15_percent": sum(item["same_sign_and_relative_error_at_most_15_percent"] for item in comparisons),
            "same_nonzero_sign": sum(item["same_nonzero_sign"] for item in comparisons),
            "absolute_residual_at_most_old_reported_floor": sum(item["absolute_residual_at_most_old_reported_floor"] for item in comparisons),
            "largest_difference_above_old_reported_floor": sum(item["largest_difference_above_old_reported_floor"] for item in comparisons)}
    for path in CONTACT_PATHS:
        counts["contact_winner_change_probe_counts"][path] = sum(
            row["reduction_controls"][path]["any_endpoint_winner_change"] for row in evaluated)
    return counts


def summarize(report, admission, report_binding, admission_binding):
    points = validate(report, admission)
    floor = finite(admission["baseline_checks"]["resolution_floor"])
    require(floor > 0, "Original resolution floor must be positive")
    rows = []
    for direction in report["directions"]:
        old = admission["directions"][direction["direction_index"]]
        old_probes = {probe["epsilon"]: probe for probe in old["probes"]}
        rows.extend(summarize_row(direction, probe, old_probes[probe["epsilon"]], points, floor)
                    for probe in direction["probes"])
    require(len(rows) == 54, "Exactly 54 fixed probe rows must remain")
    primal, ad = report["baseline_repeats"]["primal"], report["baseline_repeats"]["ad_value"]
    require(len(primal) == len(ad) == 3, "Exactly three baseline repeats per execution path are required")
    point_winners = {path: {"point_evaluations": 0, "evaluations_with_changed_winner_rows": 0,
                           "changed_row_occurrences": 0} for path in CONTACT_PATHS}
    for point in points.values():
        for evaluation in point["evaluations"]:
            for path in CONTACT_PATHS:
                changed = evaluation["contacts"][path]["changed_winner_rows_from_baseline"]
                item = point_winners[path]
                item["point_evaluations"] += 1
                item["evaluations_with_changed_winner_rows"] += bool(changed)
                item["changed_row_occurrences"] += len(changed)
    result = {"version": VERSION, "status": "diagnostic_scalar_summary_completed_original_failure_preserved",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["LSB_JOBID"],
              "queue": os.environ["LSB_QUEUE"], "report": report_binding, "original_failed_admission": admission_binding,
              "unique_graph_count": 108, "direction_count": 9, "predeclared_scales": list(SCALES),
              "old_baseline_resolution_floor": floor, "baseline": {
                  "independent_primal_repeats": primal, "AD_returned_scalar_repeats": ad,
                  "independent_primal_range": [min(primal), max(primal)],
                  "AD_returned_scalar_range": [min(ad), max(ad)],
                  "independent_primal_mean_minus_AD_scalar_mean": mean(primal) - mean(ad)},
              "all_rows": rows, "overall": aggregate(rows),
              "by_direction": {str(index): aggregate([row for row in rows if row["direction_index"] == index])
                               for index in range(9)},
              "by_declared_role": {role: aggregate([row for row in rows if role in row["roles"]])
                                   for role in sorted({role for row in rows for role in row["roles"]})},
              "by_field": {field: aggregate([row for row in rows if row["field"] == field])
                           for field in ("X", "E_upper")},
              "by_scale": {str(scale): aggregate([row for row in rows if row["epsilon"] == scale]) for scale in SCALES},
              "by_field_and_scale": {field: {str(scale): aggregate([
                  row for row in rows if row["field"] == field and row["epsilon"] == scale])
                  for scale in SCALES} for field in ("X", "E_upper")},
              "all_point_contact_winner_changes": point_winners,
              "original_run_failed_preserved": True, "can_admit_new_child": False,
              "can_certify_or_restart_original_study": False, "original_optimization_settings_changed": False,
              "original_scientific_gate_replaced": False, "scientific_arrays_decoded": False,
              "interpretation_limits": [
                  "Descriptive same-sign/15-percent counts are diagnostic observations, not acceptance or certification.",
                  "The old reported scalar floor is a reference, not a validated bound on cross-input network rounding.",
                  "Matched scalar paths and endpoint-gradient trapezoids reuse the same native model and are not independent derivative oracles.",
                  "Unchanged contact winners do not rule out internal ReLU switches or other curvature.",
                  "Float64 controls change only reductions of existing FP32 model outputs; there is no independent FP64 trunk evaluation.",
                  "These observations cannot establish a unique cause, alter an old failure, change a gate or authorize restart."]}
    result["summary_sha256"] = digest(canonical(result))
    return result


def self_test():
    allocated_cpu()
    floor = 8 * 2 ** -26
    quantized = struct.unpack("f", struct.pack("f", 1 + 2 ** -27))[0] - 1
    rounding = comparison(quantized, 2 ** -27, floor)
    require(quantized == 0 and rounding["absolute_residual_at_most_old_reported_floor"]
            and not rounding["same_sign_and_relative_error_at_most_15_percent"], "FP32 rounding toy failed")
    # f(x)=x^2 on [0,1]: baseline tangent predicts zero; endpoint trapezoid is exact.
    require(comparison(1, 0, floor)["relative_error"] == 1
            and comparison(1, .5 * (0 + 2), floor)["absolute_residual"] == 0, "Curvature toy failed")
    # max(x,0) crosses its winner between -1 and 1; a fixed left winner is zero.
    require(max(1, 0) - max(-1, 0) == 1 and 0 - 0 == 0, "Max-branch control failed")
    require(comparison(.5, .5, floor)["same_sign_and_relative_error_at_most_15_percent"]
            and not comparison(-.5, .5, floor)["same_nonzero_sign"], "Sign/relative-error control failed")
    return {"status": "passed", "cases": 4, "scope": "Allocated hand-calculable scalar controls only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    allocated_cpu()
    if args.self_test:
        require(all(value is None for value in (args.input, args.input_sha256, args.admission,
                                               args.admission_sha256, args.output)),
                "Self-test must not replace an evidence summary")
        print(json.dumps(self_test(), sort_keys=True))
        return
    require(all(value is not None for value in (args.input, args.input_sha256, args.admission,
                                                args.admission_sha256, args.output)), "All pinned input/output flags are required")
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing an existing summary output")
    report, report_binding = read_pinned(args.input, args.input_sha256, REPORT_SHA256, REPORT_BYTES)
    admission, admission_binding = read_pinned(args.admission, args.admission_sha256, ADMISSION_SHA256)
    result = summarize(report, admission, report_binding, admission_binding)
    with args.output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "summary_sha256": result["summary_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
