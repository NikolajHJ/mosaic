"""Add authenticated hard-control results to an existing allocated-HPC report.

Copies the completed report and its figures without recomputing trajectories.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--controls-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Report augmentation requires allocated DTU HPC")
    data = (args.controls / "summary.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != args.controls_sha256:
        raise ValueError("Hard-controls summary differs")
    controls = json.loads(data)
    analysis = json.loads((args.report / "analysis_summary.json").read_text())
    if (
        controls["status"] != "passed"
        or controls["native_summary_sha256"] != analysis["summary_sha256"]
    ):
        raise ValueError("Controls must pass and bind this report native study")
    for name in (
        "model_digest_before",
        "model_digest_after",
        "context_digest_before",
        "context_digest_after",
    ):
        if controls[name] != analysis["study_identity"][name]:
            raise ValueError("Control/report identity mismatch: " + name)
    shutil.copytree(args.report, args.output)
    rows = []
    for item in controls["controls"]:
        readout = item["readout"]
        identity = (
            readout["canonical_smiles"]
            if readout["sanitized"]
            else "INVALID: " + readout["conversion_error"]
        )
        rows.append(
            "<tr><th>"
            + html.escape(item["name"])
            + "</th><td>"
            + format(item["primal_loss"], ".7f")
            + "</td><td>"
            + html.escape(identity)
            + "</td></tr>"
        )
    table = (
        '<section class="panel"><h2>Hard molecules compared with hard benzene</h2><p>Every row uses one-hot atom/bond categories and the same frozen context, RNG and contact objective. The curves below instead score relaxed distributions. Identical endpoint graphs reuse exactly the same hard-chain score. Invalid graphs are retained without repair; their scores are not ligand-quality measurements.</p><table><thead><tr><th>Control</th><th>Hard-graph contact loss</th><th>Raw readout</th></tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table><p class="caption">Sequence-only protein context; no crystal-pose docking, affinity or aromaticity-recovery claim.</p><details><summary>Hard-control source and identity receipt</summary><pre>'
        + html.escape(json.dumps(controls, indent=2))
        + "</pre></details></section>"
    )
    path = args.output / "report.html"
    document = path.read_text(encoding="utf-8")
    marker = '<section class="panel"><div class="section-head"><div><h2>Loss across the complete trajectory</h2>'
    if document.count(marker) != 1:
        raise ValueError("Unrecognized report layout")
    path.write_text(document.replace(marker, table + marker), encoding="utf-8")
    analysis["hard_controls"] = {"summary_sha256": args.controls_sha256, **controls}
    (args.output / "analysis_summary.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    (args.output / "hard_controls_summary.json").write_bytes(data)
    print(
        json.dumps(
            {
                "job_id": os.environ["LSB_JOBID"],
                "status": "completed",
                "input_html_sha256": hashlib.sha256(
                    (args.report / "report.html").read_bytes()
                ).hexdigest(),
                "controls_sha256": args.controls_sha256,
                "report_html_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
