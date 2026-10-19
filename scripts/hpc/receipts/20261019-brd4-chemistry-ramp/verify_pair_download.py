"""Authenticate the saved two-arm report using stdlib byte/record checks only.

Run after scripts/verify_hpc_evidence.py. --finish-byte-checks is a bounded
continuation only if that tool completed safe extraction and its slow Windows
serial hash loop was interrupted. Every original file gate remains required.
No numerical libraries, chemistry, model execution or plotting are used here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import xml.etree.ElementTree as ET


TAG = "20261019-brd4-chemistry-ramp-01-report"
MODEL = "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2"
CONTEXT = "6747ba18833013da02dbf407d9faefb1adbc8314353b4882a34f99557b96d4f6"
CONFIGS = {
    "mix50": "1b9ea19c4405a2289f111db7dd8fc41f9df17f4ddffdfe1033de2554197c56ed",
    "random": "3f3bb1fd0b087417a462f7b4b5743ce1fc72ddbd8d099af0348380912ac0685f",
}
SUMMARIES = {
    "mix50": "0236e16f8da2aa333bf469ec5cf07b2fbe962b3a5e4d09a631d90ae60e0dfca8",
    "random": "9a2ebae13049babdeaca670bb13e707927058b8f2cd273a706d12fbf2177bdb2",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finish_bytes(base, receipt, expected_sha):
    archive = base / (TAG + ".tar.gz")
    root = (base / TAG).resolve(strict=True)
    require(
        receipt["path"] == f"/zhome/95/e/206121/mosaic-runs/{TAG}.tar.gz",
        "Unexpected archive path",
    )
    require(archive.stat().st_size == receipt["bytes"], "Archive length differs")
    require(sha(archive) == receipt["sha256"] == expected_sha, "Archive hash differs")
    raw = (root / "evidence_manifest.json").read_bytes()
    manifest = json.loads(raw)
    require(len(manifest["files"]) == receipt["files"], "Manifest count differs")
    with tarfile.open(archive) as handle:
        names, regular = set(), set()
        for member in handle:
            path = PurePosixPath(member.name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and ":" not in member.name
                and "\\" not in member.name
                and (member.isdir() or member.isfile())
                and member.name not in names,
                "Unsafe/duplicate archive member",
            )
            names.add(member.name)
            if member.isfile():
                regular.add(member.name)
            if member.name == "evidence_manifest.json":
                require(
                    handle.extractfile(member).read() == raw, "Manifest bytes differ"
                )
        require("evidence_manifest.json" in names, "Missing manifest")
        require(
            regular - {"evidence_manifest.json"} == set(manifest["files"]),
            "Archive/manifest member sets differ",
        )

    def check(item):
        name, entry = item
        path = (root / name).resolve(strict=True)
        require(
            path.is_relative_to(root)
            and path.is_file()
            and path.stat().st_size == entry["bytes"],
            "Evidence path/length differs: " + name,
        )
        require(sha(path) == entry["sha256"], "Evidence hash differs: " + name)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        items, count = iter(manifest["files"].items()), 0
        while batch := list(itertools.islice(items, 5000)):
            list(pool.map(check, batch))
            count += len(batch)
            print(
                f"{count}/{len(manifest['files'])} byte-verified ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    result = {
        "archive": str(archive),
        "sha256": expected_sha,
        "verified_files": len(manifest["files"]),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "scope": "stdlib byte integrity only; bounded continuation after safe extraction",
    }
    (base / (TAG + "-download.json")).write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("--finish-byte-checks", action="store_true")
    args = parser.parse_args()
    base = args.directory.resolve(strict=True)
    receipt = json.loads((base / (TAG + ".tar.receipt.json")).read_text())
    if args.finish_byte_checks:
        finish_bytes(base, receipt, args.sha256)
    proof = json.loads((base / (TAG + "-download.json")).read_text())
    require(
        proof["sha256"] == receipt["sha256"] == args.sha256
        and proof["verified_files"] == receipt["files"],
        "Missing every-file byte proof",
    )
    root = base / TAG
    file_manifest = json.loads((root / "evidence_manifest.json").read_text())["files"]
    source_manifest = (
        Path(__file__).resolve().parents[4]
        / "results/releases/20261019-brd4-chemistry-ramp-01/source_manifest.json"
    )
    require(
        sha(source_manifest)
        == "dc54acf72b906da9f5e2ceb0f6458314064e48408414496d1717a81117a91c80",
        "Native immutable source manifest differs",
    )
    source_files = json.loads(source_manifest.read_text())["files"]
    execution_files = {
        k: v
        for k, v in source_files.items()
        if k == "pyproject.toml"
        or (
            k.startswith(("src/", "tests/", "examples/", "scripts/"))
            and Path(k).suffix in (".py", ".json", ".lsf", ".sh", ".toml", ".txt")
        )
    }
    readiness = root / "20261019-brd4-chemistry-ramp-01-readiness/payload"
    require(
        sha(readiness / "assessment.json")
        == "35dac42a903ee397e77ff06f2b7a35cde5a191a475ebde007f844cc44912a895",
        "Readiness assessment differs",
    )
    assessment = json.loads((readiness / "assessment.json").read_text())
    require(
        assessment["status"] == "passed" and assessment["job_id"] == "29753654",
        "Readiness did not pass",
    )
    test_count = 0
    for name, entry in assessment["tests"].items():
        require(sha(readiness / name) == entry["sha256"], "Readiness XML bytes differ")
        suites = list(ET.parse(readiness / name).getroot().iter("testsuite"))
        require(
            suites
            and all(
                sum(int(s.get(k, 0)) for k in ("failures", "errors", "skipped")) == 0
                for s in suites
            ),
            "Readiness tests failed/skipped",
        )
        count = sum(int(s.get("tests", 0)) for s in suites)
        require(count == entry["tests"], "Readiness count differs")
        test_count += count
    require(test_count == 274, "Readiness selection differs")
    report = root / TAG / "payload/report"
    rendered = json.loads((report / "report_receipt.json").read_text())
    require(
        rendered["render_job_id"] == "29753669" and rendered["complete"],
        "Incomplete/wrong render",
    )
    require(
        rendered["input_jobs"] == ["29753667", "29753668"]
        and rendered["states"] == [10001, 10001],
        "Unexpected input IDs/counts",
    )
    for name, entry in rendered["files"].items():
        require(Path(name).name == name, "Unsafe report filename")
        require(
            (report / name).stat().st_size == entry["bytes"]
            and sha(report / name) == entry["sha256"],
            "Rendered file differs",
        )
    text = (report / "report.html").read_text(encoding="utf-8")
    data = json.loads(
        text.split('<script id="saved-data" type="application/json">', 1)[1].split(
            "</script>", 1
        )[0]
    )
    require(
        data["complete"]
        and data["historical_contact"] is None
        and not data.get("pending_studies"),
        "Incomplete/historical report",
    )
    require(
        [s["job_id"] for s in data["studies"]] == ["29753667", "29753668"],
        "Viewer input IDs differ",
    )
    require(
        data["model_digest"] == MODEL and data["context_digest"] == CONTEXT,
        "Viewer identities differ",
    )
    findings = []
    for kind, job, view, entry in zip(
        ("mix50", "random"),
        ("29753667", "29753668"),
        data["studies"],
        rendered["inputs"],
    ):
        study = root / ("20261019-brd4-chemistry-ramp-01-" + kind) / "payload/study"
        summary = json.loads((study / "summary.json").read_text())
        case = json.loads((study / "case-0/report.json").read_text())
        require(
            summary["job_id"] == job
            and summary["status"] == case["status"] == "passed",
            "Native status differs",
        )
        require(
            summary["source"]["files"] == execution_files,
            "Native execution source differs",
        )
        require(
            summary["initialization"]
            == entry["initialization"]
            == view["initialization"],
            "Initialization records differ",
        )
        init = summary["initialization"]
        require(
            init["recipe"] == summary["config"]["initialization"]
            and init["artifact"] == "initialization.npz"
            and init["native_key_seed"] == 17
            and sha(study / "initialization.npz") == init["sha256"],
            "Saved initialization bytes/recipe/key differ",
        )
        require(
            sha(study / "summary.json") == entry["summary_sha256"] == SUMMARIES[kind],
            "Native/render summary differs",
        )
        require(
            summary["config_sha256"]
            == sha(study / "configuration.json")
            == entry["config_sha256"]
            == CONFIGS[kind],
            "Configuration pin differs",
        )
        require(
            summary["config"]["steps"] == 10000 and summary["config"]["n_atoms"] == 6,
            "Scientific budget/schema differs",
        )
        require(
            summary["config"] == json.loads((study / "configuration.json").read_text()),
            "Copied config differs",
        )
        require(
            summary["composition_reference_sha256"]
            == sha(study / "composition_reference.json")
            == "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb",
            "Reference pin differs",
        )
        for stem, digest in (("model", MODEL), ("context", CONTEXT)):
            require(
                summary[stem + "_digest_before"]
                == summary[stem + "_digest_after"]
                == digest,
                "Frozen native identity differs",
            )
        require(
            all(summary["cases"][0].get(k) == v for k, v in case.items()),
            "Terminal case differs",
        )
        for gate in (
            "native_contact_checks",
            "coupled_chemistry_probe",
            "high_weight_chemistry_probe",
            "pilot",
            "high_weight_pilot",
        ):
            require(summary[gate]["status"] == "passed", "Native gate failed: " + gate)
        require(
            [s["step"] for s in case["states"]] == list(range(10001))
            and [u["step"] for u in case["updates"]] == list(range(10000))
            and case["counts"]["applied"] == 10000,
            "Incomplete native budget",
        )
        frames = view["frames"]
        require(
            [f["step"] for f in frames] == list(range(10001)), "Missing viewer frames"
        )
        require(
            all(
                len(f["X"]) == 6
                and all(len(r) == 6 for r in f["X"])
                and len(f["E"]) == 15
                and all(len(r) == 5 for r in f["E"])
                for f in frames
            ),
            "Viewer graph dimensions differ",
        )
        require(
            frames[0]["repair_weight"] == 0.001 and frames[-1]["repair_weight"] == 50.0,
            "Endpoint coefficients differ",
        )
        require(view["counts"] == case["counts"], "Viewer optimizer counts differ")
        for frame, state in zip(frames, case["states"]):
            snapshot_name = PurePosixPath(state["snapshot"])
            require(
                not snapshot_name.is_absolute()
                and ".." not in snapshot_name.parts
                and ":" not in state["snapshot"]
                and "\\" not in state["snapshot"],
                "Unsafe native snapshot path",
            )
            snapshot = study / "case-0" / Path(*snapshot_name.parts)
            relative_snapshot = snapshot.relative_to(root).as_posix()
            require(
                file_manifest[relative_snapshot]["sha256"] == state["snapshot_sha256"],
                "Snapshot is not bound to verified evidence bytes",
            )
            require(
                frame["snapshot_sha256"] == state["snapshot_sha256"]
                and frame["readout"] == state["readout"]
                and frame["components"] == state["components"]
                and frame["repair_weight"] == state["repair_weight"],
                "Viewer/native saved record differs",
            )
        findings.append(
            {
                "kind": kind,
                "job": job,
                "counts": case["counts"],
                "contact_initial": frames[0]["contact"],
                "contact_final": frames[-1]["contact"],
                "chemistry_initial": frames[0]["components"]["chemistry_loss"],
                "chemistry_final": frames[-1]["components"]["chemistry_loss"],
                "saved_zero_chemistry_states": sum(
                    f["components"]["chemistry_loss"] == 0 for f in frames
                ),
                "saved_sanitized_readouts": sum(
                    bool(f["readout"]["sanitized"]) for f in frames
                ),
                "saved_native_schema_valid_states": sum(
                    bool(f["chemistry_preparation"][0]["raw_native_valid"])
                    for f in frames
                ),
                "first_saved_native_schema_valid_step": next(
                    (
                        f["step"]
                        for f in frames
                        if f["chemistry_preparation"][0]["raw_native_valid"]
                    ),
                    None,
                ),
                "final_readout": frames[-1]["readout"],
                "summary_sha256": sha(study / "summary.json"),
            }
        )
    result = {
        "status": "passed",
        "report": str(report / "report.html"),
        "report_sha256": sha(report / "report.html"),
        "report_receipt_sha256": sha(report / "report_receipt.json"),
        "report_bytes": (report / "report.html").stat().st_size,
        "input_jobs": rendered["input_jobs"],
        "states": rendered["states"],
        "findings": findings,
        "scope": "saved byte/schema/native-record verification only; no local numerical evaluation",
    }
    (base / "pair-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
