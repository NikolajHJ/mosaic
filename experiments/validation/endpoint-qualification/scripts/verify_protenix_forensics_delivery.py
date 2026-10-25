"""Authenticate a fixed diagnostic delivery using standard-library bytes only.

No scientific arrays are evaluated: ZIP/NPY headers, payload byte identities and
already allocated numerical assessments are checked without importing NumPy.
The archive is read in memory and never extracted into the workspace.
"""
import argparse
import ast
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import xml.etree.ElementTree as ET
import zipfile


BASE = "/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01"
ROOT = BASE + "/protenix-endpoint-forensics-01"
SOURCE = BASE + "/protenix-endpoint-forensics-release-01/source"
MANIFEST = "d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8"
PLAN = "14d0cf5b03af73ee5d87ece55b8c48616897521c50bbb40e8f76b6a9bb8f5861"
ADMISSION = "3b4e6c14aa21aed083594ab040a7faf4a50f93b4adce5a4fa679108bc531aaf1"
ENDPOINT = "6be2b3868d58162d4aa3cc9d960cfb85a84d4078748f0b24d50b138ae3e47dbd"
FROZEN = {
    "model_digest": "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2",
    "context_digest": "b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4",
}
SCALES = [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]
MAX_ARCHIVE = 16 * 1024**2
MAX_UNPACKED = 32 * 1024**2
SHAPES = {"X": [25, 6], "E_upper": [300, 5]}
COUNTS = {
    "tests.test_protenix_endpoint_forensics": 24,
    "tests.test_protenix_forensics_native": 39,
    "tests.test_protenix_forensics_ops": 59,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(raw):
    return json.loads(raw, object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def seal(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name
            and not path.is_absolute() and all(part not in ("", ".", "..") for part in name.split("/"))
            and str(path) == name, "Unsafe/noncanonical archive path")
    return name


def record(raw, proof, label):
    require(type(proof.get("bytes")) is int and proof["bytes"] == len(raw)
            and re.fullmatch(r"[a-f0-9]{64}", proof.get("sha256", "")) is not None
            and sha(raw) == proof["sha256"], "Byte binding mismatch: " + label)


def archive_files(archive, receipt):
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size <= MAX_ARCHIVE,
            "Fresh regular bounded archive required")
    raw = archive.read_bytes()
    proof = receipt["archive"]
    require(proof.get("path") == ROOT + "/delivery-01/forensics-evidence.tar.gz", "Exact remote archive path required")
    record(raw, proof, "archive")
    listing = receipt["files"]
    require(isinstance(listing, dict) and listing, "Full archive file receipt required")
    require(all(relative(name) == name for name in listing), "Bad receipt path")
    result, total = {}, 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as stream:
        for member in stream:
            name = relative(member.name)
            require(member.isfile() and not member.issparse() and name not in result,
                    "Archive contains nonregular/duplicate/sparse member")
            require(0 <= member.size <= MAX_UNPACKED, "Oversized archive member")
            total += member.size
            require(total <= MAX_UNPACKED and name in listing, "Unlisted/oversized archive")
            file = stream.extractfile(member)
            require(file is not None, "Unreadable archive member")
            data = file.read(member.size + 1)
            require(len(data) == member.size, "Truncated archive member")
            record(data, listing[name], name)
            result[name] = data
    require(set(result) == set(listing), "Archive and receipt member sets differ")
    return result, total


def npy_payload(raw):
    require(raw.startswith(b"\x93NUMPY") and len(raw) >= 10, "Invalid NPY magic/header")
    version = tuple(raw[6:8])
    require(version in ((1, 0), (2, 0), (3, 0)), "Unsupported NPY version")
    size_bytes = 2 if version == (1, 0) else 4
    offset = 8 + size_bytes
    length = int.from_bytes(raw[8:offset], "little")
    require(0 < length <= 16384 and offset + length <= len(raw), "Bounded complete NPY header required")
    header = ast.literal_eval(raw[offset:offset + length].decode("utf8" if version == (3, 0) else "latin1"))
    require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}
            and header["fortran_order"] is False, "Plain C-order NPY required")
    dtype, shape = header["descr"], header["shape"]
    require(dtype in ("<f4", "<f8", "<i4", "<i8", "<u4", "<u8", "|u1", "|i1", "|b1")
            and isinstance(shape, tuple) and len(shape) <= 4
            and all(type(dim) is int and 0 <= dim <= 1000000 for dim in shape),
            "Unsupported NPY dtype/shape")
    payload = raw[offset + length:]
    width = int(dtype[2:])
    require(len(payload) == math.prod(shape) * width, "NPY payload length differs from header")
    return dtype, list(shape), payload


def npz_files(raw):
    result = {}
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        require(0 < len(infos) <= 32, "Invalid bounded NPZ member count")
        for item in infos:
            name = relative(item.filename)
            require("/" not in name and name.endswith(".npy") and name not in result
                    and not item.is_dir() and not item.flag_bits & 1
                    and item.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    "Unsafe/duplicate/encrypted NPZ member")
            require(item.file_size <= 1024**2, "Oversized NPZ member")
            total += item.file_size
            require(total <= 4 * 1024**2, "Oversized NPZ payload")
            result[name] = archive.read(item)  # ZipFile validates each member CRC.
        require(archive.testzip() is None, "NPZ CRC failure")
    return result


def payload_matches(raw, description, expected_dtype=None):
    dtype, shape, payload = npy_payload(raw)
    declared = {"float32": "<f4", "float64": "<f8"}.get(description.get("dtype"))
    require(declared is not None and dtype == declared
            and (expected_dtype is None or dtype == expected_dtype)
            and shape == description.get("shape") and sha(payload) == description.get("sha256"),
            "NPY payload does not match allocated array record")
    if "base64" in description:
        require(base64.b64decode(description["base64"], validate=True) == payload,
                "Exact base64 payload differs")
    return payload


def same_job_lsf(raw, job, queue, name):
    value = "".join(raw.decode("utf8").split())
    require(re.findall(r"Subject:Job(\d+):", value) == [job]
            and re.findall(r"inqueue<([^>]+)>", value) == [queue]
            and re.findall(r"Job<([^>]+)>", value)[:1] == [name]
            and value.count("Successfullycompleted.") == 1 and "Exitedwithexit" not in value,
            "Exact same-job successful LSF proof required")


def verify(files):
    def raw(name):
        require(name in files, "Missing proof: " + name)
        return files[name]

    def read(name):
        return read_json(raw(name))

    def bound(proof):
        require(proof["path"].startswith(ROOT + "/"), "Proof points outside owned output")
        name = relative(proof["path"][len(ROOT) + 1:])
        record(raw(name), proof, name)
        return raw(name)

    require(sha(raw("source_manifest.json")) == MANIFEST and sha(raw("plan.json")) == PLAN,
            "Pinned source/plan bytes differ")
    manifest, plan = read("source_manifest.json"), read("plan.json")
    require(len(manifest["files"]) == 13 and manifest["files"]["plan.json"] == PLAN
            and manifest["cpu_counts"] == COUNTS, "Pinned manifest schema differs")
    require(plan["optimizer_updates"] == 0 and plan["optimization_settings_changed"] is False
            and plan["original_results_changed"] is False and plan["no_automatic_continuation"] is True
            and plan["can_certify_or_restart_original_study"] is False
            and plan["arm"]["rate"] == 0.3 and plan["arm"]["global_start"] == 20000
            and plan["arm"]["global_target"] == 50000, "Diagnostic settings/scope differ")
    require(sha(raw("original_admission.json")) == ADMISSION
            and len(raw("original_admission.json")) == 2531142
            and sha(raw("original_endpoint.npz")) == ENDPOINT
            and len(raw("original_endpoint.npz")) == 14284, "Original raw input bytes differ")
    original = read("original_admission.json")
    require(original["status"] == "failed" and original["can_admit_new_child"] is False
            and original["original_run_failed_preserved"] is True and original["can_certify_original_study"] is False
            and original["receipt_sha256"] == seal({k: v for k, v in original.items() if k != "receipt_sha256"}),
            "Original admission failure/receipt not preserved")
    cert = read("cpu-certificate.json")
    require(cert["status"] == "passed" and cert["cases"] == 122
            and cert["source_manifest_sha256"] == MANIFEST and cert["plan_sha256"] == PLAN
            and cert["original_scientific_cpu_replaced"] is False, "CPU certificate differs")
    for proof in cert["proofs"]:
        bound(proof)
    cases = list(ET.fromstring(raw("cpu/payload/tests.xml")).iter("testcase"))
    require(len(cases) == 122 and all(not list(case) for case in cases)
            and len({(case.get("classname"), case.get("name")) for case in cases}) == 122
            and dict(Counter(case.get("classname") for case in cases)) == COUNTS,
            "Actual raw CPU fixture cases differ")
    cpu = read("cpu/submission.json")
    require(cpu["job_id"] == "29813983", "CPU job identity differs")
    same_job_lsf(raw("cpu/stdout.29813983.log"), "29813983", "hpc", "mosaic-pro-forensics-cpu")
    cpu_queries = [read(name) for name in files if name.startswith("cpu/terminal-query-") and name.endswith(".json")]
    require(any(q["returncode"] == 0 and "Job<29813983>" in "".join(q["stdout"].split())
                and "Status<DONE>" in "".join(q["stdout"].split()) for q in cpu_queries),
            "Original persisted CPU DONE query missing")
    gpu = read("gpu/submission.json")
    require(gpu["job_id"] == "29813996" and gpu["status"] == "submitted"
            and gpu["source_manifest_sha256"] == MANIFEST and gpu["dependency"] is None,
            "Actual GPU submission identity differs")
    argv = gpu["argv"]
    require(argv[argv.index("-q") + 1] == "gpul40s" and argv[argv.index("-n") + 1] == "4"
            and argv[argv.index("-W") + 1] == "02:00"
            and argv[argv.index("-gpu") + 1] == "num=1:mode=exclusive_process"
            and "rusage[mem=8GB]" in argv, "Declared GPU resources differ")
    same_job_lsf(raw("gpu/stdout.29813996.log"), "29813996", "gpul40s", "mosaic-pro-forensics-gpu")
    native = read("gpu/payload/assessment.json")
    require(native["status"] == "diagnostics_completed_original_failure_preserved" and native["job_id"] == "29813996"
            and native["allocation"]["LSB_JOBID"] == "29813996"
            and native["allocation"]["LSB_QUEUE"] == "gpul40s" and native["allocation"]["LSB_DJOB_NUMPROC"] == "4"
            and native["updates"] == 0 and native["optimization_settings_changed"] is False
            and native["can_certify_or_restart_original_study"] is False and native["rate_metadata_only"] == 0.3
            and native["original_failed_outcomes_preserved"] is True
            and native["protected_inputs_unchanged"] is True and native["frozen_unchanged"] is True
            and native["source_trees_unchanged"] is True and native["preservation_errors"] == [],
            "Native diagnostic completion/preservation differs")
    record(raw("plan.json"), native["plan"], "native plan")
    report = read_json(bound(native["report"]))
    invocation, preservation = read("gpu/native-invocation.json"), read("gpu/post-execution-preservation.json")
    require(preservation["execution_status"] == "returned" and preservation["returncode"] == 0
            and preservation["preservation_status"] == "passed" and preservation["original_results_changed"] is False
            and preservation["source_manifest_sha256"] == MANIFEST and preservation["argv"] == invocation["argv"]
            and invocation["optimization_updates"] == 0 and invocation["original_results_changed"] is False
            and invocation["can_certify_or_restart_original_study"] is False,
            "Native execution/preservation proof differs")
    before, after = read("gpu/payload/protected-before.json"), read("gpu/payload/protected-after.json")
    require(len(before) == 21 and before == after, "Protected original input set changed")
    endpoint_proof = plan["arm"]["parent_snapshot"]
    require(before[endpoint_proof["path"]] == endpoint_proof and endpoint_proof["sha256"] == ENDPOINT,
            "Preserved original endpoint identity differs")
    require(read("gpu/payload/frozen-before.json") == FROZEN == read("gpu/payload/frozen-after.json"),
            "Frozen model/context identity differs")
    identity = read("gpu/payload/native-identity.json")
    require(all(identity[key] == value for key, value in FROZEN.items())
            and identity["key_data"] == [0, 17] and identity["rate_metadata_only"] == 0.3
            and identity["updates"] == 0 and identity["snapshot"] == endpoint_proof,
            "Exact native checkpoint/RNG identity differs")
    require(report["status"] == "diagnostics_completed_original_failure_preserved"
            and report["scope"] == "separate_fixed_point_diagnostic_only"
            and report["can_certify_or_restart_original_study"] is False and report["can_admit_new_child"] is False
            and report["original_run_failed_preserved"] is True and report["original_optimization_settings_changed"] is False
            and report["original_scientific_gate_replaced"] is False and report["failures"] == []
            and report["frozen_before"] == FROZEN == report["frozen_after"] and report["frozen_unchanged"] is True
            and report["original_payload_unchanged"] is True and report["original_plan"] == original["plan"]
            and report["original_admission_sha256"] == seal(original)
            and report["original_admission_receipt_sha256"] == original["receipt_sha256"],
            "Full diagnostic report scope/provenance differs")
    points = report["points"]
    require(len(points) == report["unique_graph_count"] == 108
            and len({point["graph_id"] for point in points}) == 108, "Complete unique point cohort required")
    point_map, proofs, evaluations = {}, set(), 0
    for point in points:
        inputs = point["input"]
        require(set(inputs) == set(SHAPES) and all(inputs[key]["shape"] == shape for key, shape in SHAPES.items()),
                "Native point shapes differ")
        graph_id = seal({name: {key: value[key] for key in ("dtype", "shape", "sha256")} for name, value in inputs.items()})
        require(graph_id == point["graph_id"] and point["can_certify_or_restart_original_study"] is False,
                "Native point identity/scope differs")
        expected_repeats = 3 if graph_id == report["baseline_graph_id"] else 1
        require(len(point["evaluations"]) == expected_repeats, "Incomplete native point repeats")
        for number, evaluation in enumerate(point["evaluations"]):
            require(evaluation["repeat"] == number and evaluation["can_certify_or_restart_original_study"] is False,
                    "Evaluation repeat/scope differs")
            proof = evaluation["proof"]
            require(proof["path"].startswith(ROOT + "/gpu/payload/forensics/point-")
                    and proof["path"].endswith("-repeat-" + str(number) + ".npz") and proof["path"] not in proofs,
                    "Unexpected/repeated native raw proof")
            proofs.add(proof["path"])
            arrays = npz_files(bound(proof))
            expected = {"X.npy", "E_upper.npy", "primal.npy", "ad_value.npy", "gradient_X.npy", "gradient_E_upper.npy"}
            expected |= {name + ".npy" for name in evaluation["contacts"]}
            require(set(arrays) == expected and set(evaluation["gradients"]) == set(SHAPES),
                    "Raw evaluation NPZ member set differs")
            for field in SHAPES:
                payload_matches(arrays[field + ".npy"], inputs[field], "<f4")
                payload_matches(arrays["gradient_" + field + ".npy"], evaluation["gradients"][field], "<f4")
            for scalar in ("primal", "ad_value"):
                dtype, shape, payload = npy_payload(arrays[scalar + ".npy"])
                require(dtype == "<f8" and shape == []
                        and payload == struct.pack("<d", evaluation[scalar]), "Raw scalar bytes differ")
            require(set(evaluation["contacts"]) == {"primal_contacts", "ad_contacts", "primal_contacts64", "ad_contacts64"},
                    "Both FP32/reduction-control contact arrays required")
            for name, contact in evaluation["contacts"].items():
                require(contact["array"]["shape"] == [25, 19], "Native contact geometry differs")
                payload_matches(arrays[name + ".npy"], contact["array"], "<f8" if name.endswith("64") else "<f4")
            evaluations += 1
        point_map[graph_id] = point
    require(evaluations == 110, "Incomplete raw evaluation proof count")
    baseline = point_map[report["baseline_graph_id"]]
    endpoint_arrays = npz_files(raw("original_endpoint.npz"))
    for field in SHAPES:
        payload_matches(endpoint_arrays[field + ".npy"], baseline["input"][field], "<f4")
    directions = report["directions"]
    require(len(directions) == 9, "Complete direction cohort required")
    for number, direction in enumerate(directions):
        original_direction = original["plan"]["directions"][number]
        require(direction["direction_index"] == number
                and all(direction[key] == original_direction[key] for key in ("field", "row", "source", "target", "roles"))
                and [probe["epsilon"] for probe in direction["probes"]] == SCALES, "Direction/scales changed")
        for probe, original_probe in zip(direction["probes"], original_direction["probes"]):
            require(probe["original_fixed_probe"] == original_probe and probe["scheme"] == original_probe["scheme"]
                    and probe["analysis"]["evaluated"] is True
                    and probe["can_certify_or_restart_original_study"] is False
                    and probe["analysis"]["can_certify_or_restart_original_study"] is False,
                    "Incomplete/changed fixed probe or expanded scope")
            for side in ("left", "right"):
                require(probe[side] in point_map, "Fixed probe graph missing")
                expected_input = dict(original["plan"]["baseline"])
                expected_input[direction["field"]] = original_probe[side]
                require(point_map[probe[side]]["input"] == expected_input, "Raw point differs from original fixed probe")
    return {
        "status": "passed_independent_byte_zip_npy_schema_lsf_audit",
        "gpu_job_id": "29813996", "gpu_completion_evidence": "authentic same-job LSF success plus allocated assessment",
        "live_scheduler_state": "not_established_by_this_offline_audit; aged-out records are not manufactured",
        "cpu_cases": 122, "unique_graphs": 108, "raw_npz_evaluations": 110, "directions": 9, "fixed_scale_probes": 54,
        "protected_input_bindings_unchanged": 21, "frozen_identity_unchanged": True,
        "original_endpoint_sha256": ENDPOINT, "original_admission_file_sha256": ADMISSION,
        "diagnostic_report_sha256": native["report"]["sha256"], "source_manifest_sha256": MANIFEST, "plan_sha256": PLAN,
        "optimizer_updates": 0, "original_results_changed": False, "optimization_settings_changed": False,
        "can_certify_or_restart_original_study": False, "can_admit_new_child": False,
        "scope": "Standard-library byte/header/schema/provenance authentication only; allocated numerical findings not recomputed locally",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and args.output.parent.is_dir(), "Fresh output path with existing parent required")
    receipt_bytes = args.receipt.read_bytes()
    files, total = archive_files(args.archive, read_json(receipt_bytes))
    result = verify(files)
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(), archive_sha256=sha(args.archive.read_bytes()),
                  receipt_sha256=sha(receipt_bytes), archive_files=len(files), unpacked_bytes=total)
    with args.output.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
