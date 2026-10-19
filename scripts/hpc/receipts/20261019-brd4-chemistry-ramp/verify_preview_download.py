"""Stdlib checks for the new standalone report after safe archive extraction.

This reads saved JSON/HTML and bytes only. It does not import numerical or
chemistry libraries, render plots, repair graphs or execute a native model.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve(strict=True)
    tag = "20261019-brd4-chemistry-preview-01"
    payload = root / tag / "payload"
    report = payload / "report"
    receipt = json.loads((report / "report_receipt.json").read_text())
    require(receipt["render_job_id"] == "29755226", "Unexpected renderer")
    require(receipt["input_jobs"] == ["29753667"], "Unexpected native inputs")
    require(receipt["states"] == [10001], "Missing native states")
    require(
        receipt["included_studies_complete"] and not receipt["complete"],
        "Completed arm versus pending comparison status differs",
    )
    require(
        [p["job_id"] for p in receipt["pending_inputs"]] == ["29753668"],
        "Pending random identity differs",
    )
    for name, entry in receipt["files"].items():
        require(Path(name).name == name, "Unsafe report artifact path")
        path = report / name
        require(
            path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
            "Report artifact bytes differ: " + name,
        )
    summary_path = payload / "native-proof/summary.json"
    require(
        sha(summary_path)
        == "0236e16f8da2aa333bf469ec5cf07b2fbe962b3a5e4d09a631d90ae60e0dfca8",
        "Pinned native summary differs",
    )
    summary = json.loads(summary_path.read_text())
    require(
        summary["status"] == "passed" and summary["job_id"] == "29753667",
        "Native outcome differs",
    )
    html = (report / "report.html").read_text(encoding="utf-8")
    data = json.loads(
        html.split('<script id="saved-data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )
    require(
        len(data["studies"]) == 1 and data["historical_contact"] is None,
        "Historical or fabricated studies included",
    )
    require(
        data["included_studies_complete"] and not data["complete"],
        "Incorrect preview banner semantics",
    )
    require(
        [p["job_id"] for p in data["pending_studies"]] == ["29753668"]
        and all("frames" not in p for p in data["pending_studies"]),
        "Pending input has invented data",
    )
    frames = data["studies"][0]["frames"]
    require(
        [f["step"] for f in frames] == list(range(10001)), "Noncontiguous saved frames"
    )
    require(
        all(
            len(f["X"]) == 6
            and all(len(row) == 6 for row in f["X"])
            and len(f["E"]) == 15
            and all(len(row) == 5 for row in f["E"])
            for f in frames
        ),
        "Saved graph dimensions differ",
    )
    for field, digest in (
        ("model", "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2"),
        ("context", "6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6"),
    ):
        require(
            data[field + "_digest"]
            == summary[field + "_digest_before"]
            == summary[field + "_digest_after"]
            == digest,
            "Frozen identity differs",
        )
    suites = list(ET.parse(payload / "report-tests.xml").getroot().iter("testsuite"))
    require(
        suites
        and all(
            sum(int(s.get(k, 0)) for k in ("failures", "errors", "skipped")) == 0
            for s in suites
        ),
        "Allocated report tests failed/skipped",
    )
    tests = sum(int(s.get("tests", 0)) for s in suites)
    require(tests == 19, "Unexpected test selection")
    result = {
        "status": "passed",
        "scope": "saved bytes/schema/status only; no local numerical validation",
        "report": str(report / "report.html"),
        "report_sha256": sha(report / "report.html"),
        "native_job": "29753667",
        "pending_job": "29753668",
        "states": len(frames),
        "allocated_report_tests": tests,
        "counts": data["studies"][0]["counts"],
        "contact_initial": frames[0]["contact"],
        "contact_final": frames[-1]["contact"],
        "initial_coefficient": frames[0]["repair_weight"],
        "final_coefficient": frames[-1]["repair_weight"],
        "all_saved_chemistry_losses_zero": all(
            f["components"]["chemistry_loss"] == 0 for f in frames
        ),
        "saved_smiles": sorted(
            {f["readout"]["canonical_smiles"] or "<none>" for f in frames}
        ),
    }
    (root.parent / "preview-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
