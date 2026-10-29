"""Read-only shared-quota forecast for the exact concurrent embedding study.

No external code/model is imported or changed. Hold twice the measured full
three-case 10k byte forecast plus 1GiB controls above the complete baseline.
This is a Mosaic admission forecast, not a new external experiment hard cap.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex

VERSION = "mosaic-authenticated-embedding-shared-quota-v1"
WORK = PurePosixPath("/work3/s234818/latentmol")
MAIN = "source142-epoch36-guarded-embedding-long-attempt001"
REPAIR = "source143-embedding-handoff-recovery-attempt002"
MANIFESTS = ("89dd0f43752a417cb4a4ab0658a4fba294eba72eb65fd5899036a42d8207c4f2",
             "57469430d6e13a8708fb593d5433b454a867cd0109c62a8355dfde3d7107077f")
ROOTS = tuple(str(WORK / kind / tag) for kind in ("runs", "submissions") for tag in (MAIN, REPAIR))


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and all(not p.is_symlink() for p in path.parents), "Embedding metadata is linked or special")
    with path.open("rb") as f: digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def forecast(pilot):
    rows = pilot["cases"]
    require([r["case"] for r in rows] == ["case-001", "case-002", "case-003"], "Embedding three pilot cases changed")
    amount = 0
    for row in rows:
        require(row["gates_passed"] is True and row["completed_updates"] == 20, "Embedding actual pilot did not pass")
        profile = row["profile"]
        for key in ("bytes_per_update", "fixed_bytes"):
            value = profile[key]
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    "Embedding storage profile is not finite nonnegative metadata")
        amount += 2 * 10000 * profile["bytes_per_update"] + profile["fixed_bytes"]
    return math.ceil(amount) + 1024**3


def validate_profile(profile):
    require(set(profile) == {"version", "proofs", "source_manifests", "baseline", "growth_bytes", "growth_files", "job_templates"}
            and profile["version"] == VERSION, "Embedding profile schema differs")
    require([r["sha256"] for r in profile["source_manifests"]] == list(MANIFESTS), "Embedding sealed source pins differ")
    require(profile["growth_files"] == 150000 and type(profile["growth_bytes"]) is int and profile["growth_bytes"] > 1024**3,
            "Embedding empirical/file allowances differ")
    require(set(profile["baseline"]) == set(ROOTS), "Embedding output/control root selection changed")
    for row in profile["baseline"].values():
        require(set(row) == {"bytes", "files"} and all(type(v) is int and v >= 0 for v in row.values()),
                "Embedding baseline must be finite nonnegative bytes/files")
    require(set(profile["job_templates"]) == {"gpu", "callback"}, "Embedding command family missing")
    for kind, command in profile["job_templates"].items():
        tokens = shlex.split(command)
        require(tokens[-2] == ("--node" if kind == "gpu" else "--after")
                and tokens[-1] == "case-001-segment-0001" and all(pin in command for pin in MANIFESTS),
                "Embedding actual pinned command template differs")
    return profile


def matches(actual, template):
    a, b = shlex.split(actual), shlex.split(template)
    return len(a) == len(b) and a[:-1] == b[:-1] and bool(re.fullmatch(r"case-00[123]-segment-\d{4}|followup", a[-1]))


def observe(profile, scheduler):
    validate_profile(profile)
    for proof in profile["proofs"] + profile["source_manifests"]:
        require(binding(proof["path"]) == proof, "Embedding immutable proof changed: " + proof["path"])
    source_files = 0
    for proof in profile["source_manifests"]:
        root = Path(proof["path"]).parent
        for name, digest in read(proof["path"])["members"].items():
            rel = PurePosixPath(name)
            require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == name, "Unsafe embedding member path")
            require(binding(root / name)["sha256"] == digest, "Embedding sealed member changed: " + name)
            source_files += 1
    pilot = read(WORK / "runs" / MAIN / "pilot/result.json")
    require(forecast(pilot) == profile["growth_bytes"], "Embedding measured whole-horizon forecast changed")
    spec = importlib.util.spec_from_file_location("_embedding_readonly_inventory", Path(__file__).with_name("vae_shared_capacity.py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    inventories = []
    for root in ROOTS:
        path = Path(root)
        require(not path.is_symlink() and all(not p.is_symlink() for p in path.parents), "Embedding selected root linked")
        inventories.append(module.inventory(root) if path.exists() else {"root": root, "complete": True, "nofollow": True,
                                                                       "actual_bytes": 0, "actual_files": 0, "rows": []})
    current_bytes = sum(i["actual_bytes"] for i in inventories)
    current_files = sum(i["actual_files"] for i in inventories)
    cap = sum(i["bytes"] for i in profile["baseline"].values()) + profile["growth_bytes"]
    cap_files = sum(i["files"] for i in profile["baseline"].values()) + profile["growth_files"]
    require(current_bytes <= cap and current_files <= cap_files, "Embedding exceeded held forecast; stop own admission")
    jobs = []
    for row in scheduler:
        for kind, template in profile["job_templates"].items():
            if matches(row["COMMAND"], template):
                require(row["QUEUE"] == ("gpua100" if kind == "gpu" else "hpc"), "Embedding queue changed")
                jobs.append(row["JOBID"])
    return {"version": VERSION, "accounted_job_ids": jobs, "source_files_authenticated": source_files,
            "inventories": inventories, "actual_bytes": current_bytes, "actual_files": current_files,
            "remaining_bytes": cap-current_bytes, "remaining_files": cap_files-current_files,
            "forecast_held_until_Mosaic_work_ends": True, "external_files_jobs_settings_changed": False}


def augment(result, observation):
    value = dict(result)
    value["available_bytes"] -= observation["actual_bytes"]
    value["available_files"] -= observation["actual_files"]
    value["required_bytes"] += observation["remaining_bytes"]
    value["required_files"] += observation["remaining_files"]
    value["byte_margin"] = value["available_bytes"] - value["required_bytes"]
    value["file_margin"] = value["available_files"] - value["required_files"]
    value["status"] = "passed" if value["byte_margin"] >= 0 and value["file_margin"] >= 0 else "failed"
    value["shared_embedding_accounting"] = observation
    return value
