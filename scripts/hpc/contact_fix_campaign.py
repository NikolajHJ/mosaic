"""Queue contact-fix readiness, then promote a fresh pilot only on genuine passes.

All numerical checks run in DTU allocations. This metadata controller uses
immutable source/config hashes and durable bsub receipts. Readiness failure
stops promotion; no retry, tolerance change or historical gate waiver exists.
"""

import argparse
from decimal import Decimal
import importlib
import json
from pathlib import Path
import re
import shlex
import subprocess

from latentmol.experiments import file_sha256, source_manifest, write_json
from submit_stability_study import authenticated, read, submit
from submit_ramped_random import gpu_request


def protocol_settings(version):
    """Dispatch only declared contracts; old source105 plans retain their policy."""
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unsupported contact-fix campaign protocol")
    return {
        1: {"policy": "branch_local_v2", "pilot_version": 2,
            "readiness_module": "latentmol.boltz.contact_precision_validation", "readiness_wall": "6:00"},
        2: {"policy": "additive_branch_v3", "pilot_version": 3,
            "readiness_module": "latentmol.boltz.additive_readiness", "readiness_wall": "3:15"},
        3: {"policy": "pointwise_native_ad_v4", "pilot_version": 4,
            "readiness_module": "latentmol.boltz.pointwise_readiness", "readiness_wall": "0:45"},
    }[version]


def diagnostic_predecessor(protocol, validation=None):
    """Authenticate the external observer receipt before building a dependency.

    The hash-bound receipt includes its exact intent. Recheck the adjacent
    intent, observer source and declared plan/CLI so an unrelated successful
    job cannot release new readiness. This authenticates submission only; LSF
    must still observe done(job), and native readiness must authenticate the
    completed observer's actual outputs before issuing its own certificate.
    """
    reference = protocol.get("diagnostic_job_receipt")
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Additive readiness requires an exact diagnostic job receipt")
    work = Path(protocol["work_root"]).resolve(strict=True)
    path = Path(reference["path"])
    if not path.is_absolute() or not path.resolve().is_relative_to(work / "submissions"):
        raise ValueError("Diagnostic receipt must stay inside DTU submissions")
    authenticated(path, reference["sha256"])
    receipt = read(path)
    job, intent = receipt.get("job_id"), receipt.get("intent")
    if (type(job) is not str or not re.fullmatch(r"[1-9][0-9]*", job)
            or receipt.get("status") != "submitted_not_passed" or not isinstance(intent, dict)
            or path.name != "replay-receipt.json" or read(path.with_name("replay-intent.json")) != intent):
        raise ValueError("Diagnostic job receipt/intent is invalid")
    if validation is None:
        bound = protocol["validation_plan"]
        authenticated(Path(bound["path"]), bound["sha256"])
        validation = read(Path(bound["path"]))
    diagnostic = validation.get("diagnostic_reference")
    if (not isinstance(diagnostic, dict) or set(diagnostic) != {"plan", "output_root"}
            or not isinstance(diagnostic["plan"], dict) or set(diagnostic["plan"]) != {"path", "sha256"}):
        raise ValueError("Additive readiness lacks its declared observer plan")
    observer_path = Path(diagnostic["plan"]["path"])
    if not observer_path.is_absolute() or not observer_path.resolve().is_relative_to(work / "submissions"):
        raise ValueError("Diagnostic plan must stay inside DTU submissions")
    authenticated(observer_path, diagnostic["plan"]["sha256"])
    observer = read(observer_path)
    identity = intent.get("identity", {})
    if (not isinstance(identity, dict) or set(identity) != {"archive_sha256", "manifest_sha256", "plan_sha256"}
            or any(type(value) is not str or not re.fullmatch(r"[0-9a-f]{64}", value) for value in identity.values())
            or identity["plan_sha256"] != diagnostic["plan"]["sha256"]
            or observer.get("output_root") != diagnostic["output_root"]):
        raise ValueError("Diagnostic job belongs to a different observer plan/source")
    release = Path(intent.get("cwd", "")).resolve(strict=True)
    if not release.is_relative_to(work / "releases"):
        raise ValueError("Diagnostic execution did not use an immutable DTU release")
    authenticated(Path(str(release) + ".tar.gz"), identity["archive_sha256"])
    if source_manifest(release)["sha256"] != identity["manifest_sha256"]:
        raise ValueError("Diagnostic executable source differs from its job receipt")
    command = intent.get("command")
    if (not isinstance(command, list) or not command or command[0] != "bsub"
            or any(type(value) is not str for value in command)):
        raise ValueError("Diagnostic receipt lacks an exact scheduler command")
    tokens = shlex.split(command[-1])
    module = "latentmol.boltz.trajectory_gradient_diagnostic"
    if tokens.count(module) != 1 or tokens.index(module) == 0 or tokens[tokens.index(module) - 1] != "-m":
        raise ValueError("Diagnostic receipt did not submit the native observer")
    for flag, expected in (("--plan", str(observer_path)), ("--sha256", diagnostic["plan"]["sha256"]),
                           ("--archive-sha256", identity["archive_sha256"]),
                           ("--manifest-sha256", identity["manifest_sha256"])):
        if tokens.count(flag) != 1 or tokens.index(flag) + 1 >= len(tokens) or tokens[tokens.index(flag) + 1] != expected:
            raise ValueError("Diagnostic command differs from its plan/source receipt: " + flag)
    return job


def validate_plan_policies(protocol, validation, pilot):
    """Keep arithmetic, derivative policy, pilot schema and readiness source bound."""
    mode = protocol_settings(protocol["version"])
    if (validation.get("objective_arithmetic") != "contact_head_and_assembly_float64_v1"
            or validation.get("fd_policy") != mode["policy"]
            or type(pilot.get("version")) is not int or pilot.get("version") != mode["pilot_version"]
            or pilot.get("derivative_contract") != mode["policy"]
            or pilot.get("saved_state_readiness") != {"plan": protocol["validation_plan"], "output_root": validation.get("output_root")}
            or any(pilot.get(key) != protocol[key] for key in ("source_archive_sha256", "source_manifest_sha256"))):
        raise ValueError("Pilot is not bound to this readiness/source/policy")


def available_quota_gib(text, user):
    """Parse official per-user pool quotas, rejecting unknown/absent limits."""
    units = {"Byte": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}
    remaining = []
    def amount(value):
        number, unit = value.strip().split()
        result = Decimal(number) * units[unit]
        if not result.is_finite() or result < 0:
            raise ValueError("Invalid quota amount")
        return result
    for line in text.splitlines():
        if line.strip().startswith(user + "|"):
            fields = line.split("||")[1].split("|")
            if len(fields) != 2:
                raise ValueError("Unsupported official quota row")
            used, hard = (value.strip() for value in fields)
            if hard != "unlimited":
                remaining.append((amount(hard) - amount(used)) / Decimal(1024**3))
    if not remaining:
        raise ValueError("Finite user quota missing")
    return min(remaining)


def load_protocol(path, expected):
    authenticated(path, expected)
    protocol = read(path)
    fields = {"version", "release", "work_root", "source_archive_sha256", "source_manifest_sha256",
              "validation_plan", "pilot_plan", "receipts", "allocation_root"}
    protocol_settings(protocol.get("version"))
    if protocol["version"] == 2:
        fields.add("diagnostic_job_receipt")
    if set(protocol) != fields:
        raise ValueError("Unsupported contact-fix campaign protocol")
    work, release = Path(protocol["work_root"]).resolve(strict=True), Path(protocol["release"]).resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require immutable release inside the declared DTU root")
    for label in ("receipts", "allocation_root"):
        target = Path(protocol[label])
        if not target.is_absolute() or not target.resolve().is_relative_to(work):
            raise ValueError("Campaign controls must stay inside the DTU work root")
    authenticated(Path(str(release) + ".tar.gz"), protocol["source_archive_sha256"])
    if source_manifest(release)["sha256"] != protocol["source_manifest_sha256"]:
        raise ValueError("Sealed source changed")
    for key in ("validation_plan", "pilot_plan"):
        reference = protocol[key]
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("Plans require an exact path/hash binding")
        target = Path(reference["path"])
        if not target.is_absolute() or not target.resolve().is_relative_to(work):
            raise ValueError("Plan must stay inside DTU work root")
        authenticated(target, reference["sha256"])
    validation, pilot = (read(Path(protocol[key]["path"])) for key in ("validation_plan", "pilot_plan"))
    validate_plan_policies(protocol, validation, pilot)
    if protocol["version"] == 2:
        diagnostic_predecessor(protocol, validation)
    elif protocol["version"] == 3:
        # The focused pointwise contract is justified by an immutable saved-data
        # review. It does not wait for, or masquerade as, source106's observer.
        from latentmol.boltz.pointwise_readiness import load_plan
        load_plan(Path(protocol["validation_plan"]["path"]), protocol["validation_plan"]["sha256"])
    for root in (validation.get("output_root"), pilot.get("output_root")):
        if not isinstance(root, str) or not Path(root).is_absolute() or not Path(root).resolve().is_relative_to(work / "runs"):
            raise ValueError("Require unique output roots inside DTU runs")
    if validation["output_root"] == pilot["output_root"]:
        raise ValueError("Readiness and pilot must have separate output roots")
    return protocol


def workflow(protocol, protocol_path, protocol_sha):
    release, work = Path(protocol["release"]), Path(protocol["work_root"])
    validation = protocol["validation_plan"]
    mode = protocol_settings(protocol["version"])
    stages = [
        ("cpu", "hpc", "0:45", 2, "4GB", None,
         ["bash", "scripts/hpc/contact_fix_preflight.sh", str(protocol_path), protocol_sha]),
        ("readiness", "gpua100", mode["readiness_wall"], 4, "16GB", "done:cpu",
         ["python", "-m", mode["readiness_module"], "--plan", validation["path"], "--sha256", validation["sha256"]]),
        ("promotion", "hpc", "0:15", 2, "4GB", "ended:readiness",
         ["python", "scripts/hpc/contact_fix_campaign.py", "--protocol", str(protocol_path), "--sha256", protocol_sha, "--mode", "advance"]),
    ]
    result = []
    for stage, queue, wall, cores, memory, dependency, command in stages:
        label = release.name + "-" + stage
        args = ["bsub", "-J", label, "-q", queue, "-n", str(cores), "-W", wall,
            "-R", "rusage[mem=" + memory + "]", "-R", "span[hosts=1]",
            "-oo", str(work / "logs" / (label + ".%J.out")), "-eo", str(work / "logs" / (label + ".%J.err"))]
        if stage == "readiness":
            args += ["-R", "select[gpu80gb]", "-gpu", gpu_request(managed_mps=True)]
        job = ["env", "LATENTMOL_SOURCE_ARCHIVE=" + str(release) + ".tar.gz", "LATENTMOL_WORK_ROOT=" + str(work),
            "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
            "--archive-sha256", protocol["source_archive_sha256"], "--manifest-sha256", protocol["source_manifest_sha256"],
            "--", "python", "-m", "latentmol.experiments", "--run-root", protocol["allocation_root"],
            "--name", label, "--", *command]
        row = {"stage": stage, "cwd": str(release), "arguments": args,
               "job_command": job, "dependency": dependency}
        if stage == "readiness" and protocol["version"] == 2:
            row["external_dependencies"] = [{"kind": "done", "job_id": diagnostic_predecessor(protocol)}]
        result.append(row)
    return result


def advance(protocol, identity):
    """Read actual completed readiness and scheduler state before one pilot bsub."""
    mode = protocol_settings(protocol["version"])
    require_readiness = importlib.import_module(mode["readiness_module"]).require_readiness
    receipt_root = Path(protocol["receipts"])
    native_receipt = read(receipt_root / "readiness-receipt.json")
    job = native_receipt["job_id"]
    intent = read(receipt_root / "readiness-intent.json")
    if (not isinstance(job, str) or not re.fullmatch("[0-9]+", job)
            or native_receipt.get("intent") != intent or intent.get("identity") != identity
            or intent.get("cwd") != protocol["release"]):
        raise ValueError("Readiness scheduler receipt belongs to another immutable campaign")
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                           capture_output=True, text=True, timeout=30)
    decision = {"status": "checking", "predecessor_job": job,
        "scheduler": {"returncode": reply.returncode, "stdout": reply.stdout, "stderr": reply.stderr},
        "pilot_submitted": False, "identity": identity}
    destination = receipt_root / "promotion-decision.json"
    write_json(destination, decision)
    try:
        state = reply.stdout.strip()
        # Readiness can outlive LSF's short retained history. Missing history is
        # accepted only with full persisted completion and source/plan proof.
        if state != "DONE" and not (not state and "is not found" in reply.stderr):
            raise ValueError("Readiness predecessor did not finish successfully: " + state)
        pilot = read(Path(protocol["pilot_plan"]["path"]))
        source_identity = {key: protocol[key] for key in ("source_archive_sha256", "source_manifest_sha256")}
        proof = require_readiness(pilot["saved_state_readiness"], source_identity)
        if (proof.get("passed") is not True or proof.get("allocation_job_id") != job
                or proof.get("fd_policy") != mode["policy"]
                or proof.get("source_identity") != source_identity):
            raise ValueError("Readiness proof came from a different GPU allocation")
        quota = subprocess.run(["getquota_work3.sh"], capture_output=True, text=True, timeout=30)
        if quota.returncode != 0:
            raise ValueError("Cannot authenticate available work3 quota")
        remaining_gib = available_quota_gib(quota.stdout, Path(protocol["work_root"]).parts[2])
        measured_quota = {"remaining_gib": str(remaining_gib), "minimum_gib": 5,
                          "stdout": quota.stdout, "stderr": quota.stderr}
        if remaining_gib < 5:
            raise ValueError("Insufficient work3 margin for the matched pilot")
        from submit_defog_matched_pilot import workflow as pilot_workflow
        stages = pilot_workflow(Path(protocol["release"]), Path(protocol["work_root"]),
            Path(protocol["pilot_plan"]["path"]), protocol["pilot_plan"]["sha256"],
            protocol["source_archive_sha256"], protocol["source_manifest_sha256"])
        native = stages[1]
        native["stage"], native["dependency"] = "pilot", None
        native["arguments"][native["arguments"].index("-J") + 1] += "-pilot"
        decision.update(readiness=proof, quota=measured_quota,
            jobs=submit([native], receipt_root / "pilot", {**identity, "readiness": proof}),
            pilot_submitted=True, status="submitted_not_passed")
    except Exception as exc:
        decision.update(status="stopped", error=str(exc), error_type=type(exc).__name__)
        write_json(destination, decision)
        raise
    write_json(destination, decision)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--mode", choices=("prepare", "submit", "advance"), default="prepare")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol, args.sha256)
    identity = {key: protocol[key] for key in ("source_archive_sha256", "source_manifest_sha256")}
    identity.update(protocol=str(args.protocol), protocol_sha256=args.sha256)
    if args.mode == "advance":
        print(json.dumps(advance(protocol, identity), indent=2))
    elif args.mode == "submit":
        print(json.dumps(submit(workflow(protocol, args.protocol, args.sha256), Path(protocol["receipts"]), identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "stages": workflow(protocol, args.protocol, args.sha256), "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
