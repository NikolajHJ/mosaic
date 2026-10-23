"""Read saved paired-experiment JSON and produce a standalone reviewable report.

No models, tensor files, network resources or numerical training runtime are
loaded. Objective panels use separate axes; only the common hard-context
diagnostic overlays arms. Invalid chemical scores retain their diagnostic scope.
"""

from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path


def _text(value):
    return escape("—" if value is None else str(value), quote=True)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        return "—"
    return f"{value:.6g}"


def _boolean(value):
    return "yes" if value is True else "no" if value is False else "not evaluated"


def _table(headers, rows):
    head = "".join("<th>" + _text(value) + "</th>" for value in headers)
    body = "".join("<tr>" + "".join("<td>" + _text(value) + "</td>" for value in row) + "</tr>" for row in rows)
    return '<div class="table-scroll"><table><thead><tr>' + head + "</tr></thead><tbody>" + body + "</tbody></table></div>"


def _chart(title, series):
    """SVG coordinates are generated from finite saved scalar observations."""
    series = {name: [(x, y) for x, y in points if type(x) in (int, float)
        and type(y) in (int, float) and math.isfinite(x) and math.isfinite(y)]
        for name, points in series.items()}
    points = [point for values in series.values() for point in values]
    if not points:
        return '<div class="chart"><h4>' + _text(title) + "</h4><p>No finite saved observations.</p></div>"
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    padding = max((ymax - ymin) * .08, abs(ymin) * .005, 1e-8)
    ymin, ymax = ymin - padding, ymax + padding
    left, top, width, height = 65, 12, 365, 175
    sx = lambda x: left + (x - xmin) / max(xmax - xmin, 1) * width
    sy = lambda y: top + (ymax - y) / (ymax - ymin) * height
    svg = [f'<svg viewBox="0 0 450 235" role="img" aria-label="{_text(title)}">',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="#8893a4"/>',
        f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="#8893a4"/>']
    for fraction in (0, .5, 1):
        y = top + height * fraction
        value = ymax - (ymax - ymin) * fraction
        svg.append(f'<text x="{left - 7}" y="{y + 4}" text-anchor="end">{_number(value)}</text>')
        svg.append(f'<line x1="{left}" y1="{y}" x2="{left + width}" y2="{y}" stroke="#e5e9f0"/>')
    for x in sorted({xmin, xmax}):
        svg.append(f'<text x="{sx(x)}" y="{top + height + 18}" text-anchor="middle">{_number(x)}</text>')
    svg.append('<text x="247" y="228" text-anchor="middle">Applied update</text>')
    colors = ("#265fae", "#ac5b15", "#276d50")
    legends = []
    for index, (name, values) in enumerate(series.items()):
        color = colors[index % len(colors)]
        if values:
            coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in values)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in values:
                svg.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2" fill="{color}"><title>update {_number(x)}, loss {_number(y)}</title></circle>')
        legends.append(f'<span style="color:{color}">● {_text(name)}</span>')
    svg.append("</svg>")
    return '<div class="chart"><h4>' + _text(title) + "</h4>" + "".join(svg) + '<div class="legend">' + " · ".join(legends) + "</div></div>"


def _case_section(summary, comparison):
    sections = ["<section><h2>" + _text(summary["case_id"]) + "</h2>"]
    start = summary.get("start", {})
    sections.append("<p>Start: <code>" + _text(start.get("identity")) + "</code>. Occupied slots: " +
        _text(start.get("occupied_slots")) + ". Shared context: <code>" +
        _text(start.get("bound_context_sha256")) + "</code>.</p>")
    runs = comparison["runs"]
    first = runs[0]["checkpoints"][0] if runs else {}
    initial_identity = first.get("canonical_identities", {}).get("conditioned")
    states, counts, charts, hard_series = [], [], [], {}
    for run in runs:
        path = run["path"]
        completed = [row for row in run["checkpoints"] if row.get("status") == "completed_audit"]
        if not completed:
            continue
        selected = [("initial", completed[0]), ("terminal", completed[-1])]
        best = run.get("best_observed")
        if best:
            matching = next((row for row in completed if row["update"] == best["update"]), None)
            if matching:
                selected.append(("best own objective", matching))
        for role, row in selected:
            assessment = row.get("classifications", {}).get("conditioned", {})
            identity = row.get("canonical_identities", {}).get("conditioned")
            changed = identity != initial_identity if identity and initial_identity else None
            scores = row.get("scores", {})
            states.append([path, role, row["update"], _number(scores.get(path, {}).get("dbmol_total_loss")),
                _number(scores.get("hard_fixed_context", {}).get("dbmol_total_loss")),
                _boolean(assessment.get("raw_chemical_valid")), _boolean(assessment.get("strict_teacher_valid")),
                _boolean(changed), identity or assessment.get("reason"),
                _number(row.get("learned_minus_hard_total_loss"))])
        calls = run.get("callback_calls", {})
        counts.append([path, run["status"], run["applied"], run["improving"], run["forced_fallback"],
            run["full_step_nonincrease"], run["no_op"], calls.get("attempted"), calls.get("completed"), calls.get("failed")])
        charts.append(_chart(path.capitalize() + " objective (own scalar)", {path: [
            (row["update"], row["scores"][path]["dbmol_total_loss"]) for row in completed]}))
        hard_series[path] = [(row["update"], row["scores"]["hard_fixed_context"]["dbmol_total_loss"])
                             for row in completed]
    charts.append(_chart("Common hard fixed-context diagnostic", hard_series))
    sections.append('<div class="charts">' + "".join(charts) + "</div>")
    sections.append("<p>Each objective panel has its own scale and scalar definition. The hard diagnostic shares the frozen pose and atom names; chemically invalid hard graphs remain tensor diagnostics.</p>")
    sections.append(_table(["Arm", "State", "Update", "Own objective", "Hard diagnostic", "Raw valid", "Strict valid", "Identity changed", "Identity / failure", "Learned − hard"], states))
    sections.append(_table(["Arm", "Status", "Applied", "Improving", "Fallback", "Full nonincrease", "No-op", "Callback attempted", "Completed", "Failed"], counts))
    actual = comparison.get("actual_model_calls", {})
    sections.append("<p>Actual DBMol scorer invocations, including checkpoint audits: " +
        _text(actual.get("boltz_forward_attempted")) + " attempted / " + _text(actual.get("boltz_forward_completed")) +
        " completed. Backward: " + _text(actual.get("backward_attempted")) + " attempted / " +
        _text(actual.get("backward_completed")) + " completed. Each scorer invocation executes both frozen checkpoint trunks. Callback counts above cover baseline-gradient and trial evaluations.</p>")
    sections.append('<details><summary>Saved counter and identity evidence</summary><pre>' +
        _text(json.dumps({"actual_model_calls": actual, "frozen_after": comparison.get("frozen_after"),
            "source_record": summary["full_record"]}, indent=2, sort_keys=True)) + "</pre></details></section>")
    return "".join(sections)


def _native_section(native, results):
    rows = []
    for case in native.get("cases", []):
        for pair in case.get("paired_seeds", []):
            affinity = pair.get("affinity_pred_value", {})
            probability = pair.get("affinity_probability_binary", {})
            rows.append([case.get("case_id"), case.get("arm"), case.get("state"), pair.get("seed"),
                _boolean(pair.get("paired_identity_eligible")), _number(affinity.get("start")),
                _number(affinity.get("candidate")), _number(affinity.get("candidate_minus_start")),
                _number(probability.get("candidate_minus_start")), case.get("smiles") or case.get("failure")])
    failures = [[row.get("id"), row.get("seed"), row.get("smiles"), row.get("status"), row.get("failure")]
                for row in results if row.get("retention_eligible") is not True]
    return ("<section><h2>Fresh native pose follow-up</h2><p>Status: " + _text(native.get("status")) +
        ". Eligible requests: " + _text(native.get("successful_native_requests")) + " / " +
        _text(native.get("planned_native_requests")) + ". Native affinities at fresh poses use ordinary inference; they are a separate measurement from the fixed-pose full DBMol objective.</p>" +
        _table(["Case", "Arm", "State", "Pose seed", "Eligible pair", "Start affinity", "Candidate affinity", "Affinity delta", "Probability delta", "Identity / failure"], rows) +
        "<p>Delta means candidate minus its shared start at the same pose seed. Missing or invalid pairs remain absent from gain estimates. Positive native deltas are descriptive predictions, not measured binding.</p>" +
        (_table(["Failed request", "Pose seed", "Identity", "Status", "Failure"], failures) if failures else
         "<p>No terminal native request failures are present in the saved result list. Unlaunched requests, if any, are excluded from this statement.</p>") + "</section>")


def generate_report(output_root, cases_summary, native_summary):
    """Write a portable HTML report from already persisted experiment evidence."""
    output = Path(output_root)
    if not output.is_dir():
        raise ValueError("Comparison report requires an existing experiment directory")
    sections, records = [], []
    for case in cases_summary:
        comparison = json.loads(Path(case["full_record"]).read_text(encoding="utf-8"))
        sections.append(_case_section(case, comparison))
        records.append(dict(case_id=case["case_id"], source_record=case["full_record"],
                            status=comparison.get("status")))
    native_path = output / "fresh-native" / "native-results.json"
    native_results = json.loads(native_path.read_text(encoding="utf-8")) if native_path.exists() else []
    metadata = dict(case_count=len(records), cases=records, native_status=native_summary.get("status"),
        purpose="paired exploratory optimization review", model_execution=False,
        statistical_superiority_established=False, charged_chemistry_coverage_established=False)
    (output / "report-data.json").write_text(json.dumps(metadata, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n", encoding="utf-8")
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Frozen epoch-36 mapping comparison</title><style>
body{font:15px/1.5 system-ui,sans-serif;color:#172334;background:#f5f7fb;max-width:1500px;margin:28px auto;padding:0 22px}h1{font-size:27px}h2{font-size:22px}h4{margin:0 0 8px;font-size:15px}section{background:white;border:1px solid #dfe4ed;border-radius:8px;padding:20px;margin:22px 0}code,pre{font:12px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #e5e9f0;padding:8px;text-align:left;vertical-align:top}th{background:#eef2f8}td{max-width:360px;overflow-wrap:anywhere}.table-scroll{overflow:auto;margin:16px 0}.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:18px}.chart{border:1px solid #e2e7f0;border-radius:6px;padding:12px}.chart svg{width:100%;height:auto}.chart svg text{font:11px system-ui,sans-serif;fill:#465468}.legend{font-size:12px}.scope{background:#eaf0f9;padding:14px;border-left:4px solid #265fae}
</style><body><h1>Frozen epoch-36 mapping versus relaxed graph</h1><p class="scope">Three paired starting latents, fixed occupancy and pose, frozen VAE/Boltz/mapping weights. Both arms use the same update policy and requested budget; actual forward costs may differ. Raw learned and relaxed objective totals are boundary-specific and are not ranked against each other. This exploratory pilot does not establish statistical superiority or charged-category coverage.</p>""" + "".join(sections) + _native_section(native_summary, native_results) + \
        "<footer><p>All tables and curves read saved records. No model ran while generating this report. Existing VAE continuation remains unchanged.</p></footer></body></html>"
    path = output / "comparison-report.html"
    path.write_text(document, encoding="utf-8")
    return path
