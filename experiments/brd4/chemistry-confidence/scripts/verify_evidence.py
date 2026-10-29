"""Independent stdlib byte/XML/JSON/scheduler audit; no scientific arrays."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile
import xml.etree.ElementTree as ET


def sha(data): return hashlib.sha256(data).hexdigest()


def members(data):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for item in archive.getmembers():
            p = PurePosixPath(item.name)
            assert item.isfile() and not p.is_absolute() and ".." not in p.parts and "\\" not in item.name and item.name not in result
            value = archive.extractfile(item).read(); assert len(value) == item.size
            result[item.name] = value
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(); repo = Path(__file__).resolve().parents[4]
    preparation = json.loads((args.run/"preparation.json").read_bytes())
    verification = json.loads((args.run/"verification.json").read_bytes())
    source_bytes = (args.artifacts/"source01/source.tar.gz").read_bytes()
    assert len(source_bytes) == preparation["archive_bytes"] and sha(source_bytes) == preparation["archive_sha256"]
    source = members(source_bytes)
    assert sha(source["source-manifest.json"]) == preparation["manifest_sha256"]
    manifest = json.loads(source["source-manifest.json"])
    assert set(source) == set(manifest["files"]) | {"source-manifest.json"}
    for name, pin in manifest["files"].items():
        assert len(source[name]) == pin["bytes"] and sha(source[name]) == pin["sha256"]
    for path in ("src/mosaic/small_molecule/chemistry/confidence.py",
                 "src/mosaic/small_molecule/chemistry/trajectory.py",
                 "tests/small_molecule/test_chemistry_confidence.py"):
        assert (repo/path).read_bytes() == source[path], "Working scientific/test bytes changed after qualification"
    archive_bytes = (args.artifacts/"allocated01/evidence.tar.gz").read_bytes()
    assert len(archive_bytes) == verification["archive_bytes"] and sha(archive_bytes) == verification["archive_sha256"]
    evidence = members(archive_bytes); pins = json.loads(evidence["evidence-manifest.json"])
    assert set(evidence) == set(pins["files"]) | {"evidence-manifest.json"}
    for name, pin in pins["files"].items():
        assert len(evidence[name]) == pin["bytes"] and sha(evidence[name]) == pin["sha256"]
        assert (args.artifacts/"allocated01/verified"/name).read_bytes() == evidence[name]
    assessment = json.loads(evidence["assessment.json"]); runtime = json.loads(evidence["runtime.json"])
    assert assessment["status"] == "passed" and assessment["source_manifest_sha256"] == preparation["manifest_sha256"]
    assert evidence["source-manifest.json"] == source["source-manifest.json"]
    cases = ET.fromstring(evidence["tests.xml"]).findall(".//testcase")
    assert len(cases) == manifest["expected_tests"] == assessment["tests_passed"] == 35
    assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
    scheduler = json.loads((args.run/"scheduler.json").read_bytes())
    jid = assessment["job_id"]
    assert runtime["job_id"] == jid and scheduler["actual_job_ids"] == [jid] and scheduler["bsub_returncode"] == 0
    assert runtime["queue"] == "hpcint" and runtime["slots"] == "4" and runtime["persistent_shared_writes"] is False
    assert runtime["node_local_root"].startswith("/tmp/") and runtime["host"].startswith("n-")
    assert runtime['child_temp_directory'] == runtime['node_local_root']+'/tmp'
    temporary = json.loads(evidence['tempdir-proof.json'])
    assert temporary['returncode'] == 0 and temporary['stdout'].strip() == temporary['expected'] == runtime['child_temp_directory']
    raw_jobs = [p for p in scheduler["proofs"] if p["argv"][1] == "-json"]
    assert len(raw_jobs) == 1 and raw_jobs[0]["returncode"] == 0
    records = json.loads(raw_jobs[0]["stdout"])["RECORDS"]
    assert len(records) == 1 and records[0]["JOBID"] == jid and records[0]["STAT"] == "DONE" and records[0]["QUEUE"] == "hpcint"
    logs = [p for p in scheduler["proofs"] if p["argv"][1] == "-l"]
    assert len(logs) == 1 and logs[0]["returncode"] == 0
    text = " ".join(logs[0]["stdout"].split())
    assert "Job <"+jid+">" in text and "Status <DONE>" in text and "Done successfully" in text
    comparison = json.loads(evidence["comparison/comparison.json"])
    assert comparison["status"] == "completed_measurements_not_native_validation" and comparison["native_binding_performance"] == "not_evaluated"
    assert comparison["element_js_weight"] == 0 and comparison["repair_weight"] == 1 and comparison["rate"] == .3
    assert comparison["margin"] == {"version": "valid-confidence-huber-v1", "weight": .01, "margin": .2, "slope_cap": 1.}
    expected = {(case, seed, arm) for case in ("uncertain_valid_ccc", "confident_valid_ccc", "valid_15_atoms_3_sulfur", "invalid_two_bond_fluorine")
                for seed in (17, 23, 29) for arm in ("current_repair", "repair_plus_margin")}
    rows = comparison["rows"]
    assert len(rows) == 24 and {(r["case"], r["seed"], r["arm"]) for r in rows} == expected
    sample_files = []
    for row in rows:
        assert [u["step"] for u in row["updates"]] == list(range(128))
        assert [c["step"] for c in row["checkpoints"]] == [0, 64, 128]
        for update in row["updates"]:
            assert re.fullmatch("[a-f0-9]{64}", update["preparation_identity"])
            assert all(math.isfinite(update[k]) for k in ("fixed_before", "fixed_after", "atom_gradient_norm", "bond_gradient_norm"))
        for point in row["checkpoints"]:
            assert point["sample_count"] == 256 and 0 <= point["strict_valid"] <= point["raw_valid"] <= 256
            name = "comparison/"+point["sample_archive"]
            assert name in evidence and sha(evidence[name]) == point["sample_archive_sha256"]
            sample_files.append(name)
    assert len(set(sample_files)) == 72
    result = {"status": "passed", "scope": "independent stdlib byte/XML/JSON/scheduler/schema audit; no local scientific array calculations",
        "job_id": jid, "source_manifest_sha256": preparation["manifest_sha256"],
        "evidence_archive_sha256": sha(archive_bytes), "source_files": len(manifest["files"]),
        "verified_evidence_members": len(evidence), "actual_test_cases": len(cases),
        "matched_pairs": 12, "trajectories": 24, "sample_archives": 72,
        "saved_results_changed": False, "native_binding_validated": False}
    destination = args.run/"independent-verification.json"
    with destination.open("x", encoding="utf8", newline="\n") as stream: json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
