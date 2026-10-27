"""Allocated, immutable stage runner for the independent DiffSBDD study.

This module uses only the standard library. The local campaign controller owns
environment creation, storage admission, staging and explicit serial submission.
An operator invocation launches exactly one numerical Python process, never a
successor job. Failed, timed-out and ambiguous attempts keep their evidence and
cannot be retried in place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time


VERSION = "diffsbdd-allocated-operator-v1"
SOURCE_VERSION = "diffsbdd-source-manifest-v1"
STAGE_VERSION = "diffsbdd-stage-v1"
OUTPUT_VERSION = "diffsbdd-stage-outputs-v1"
STAGES = ("cpu", "native", "scorer", "fit", "audit", "smoke", "pilot", "followup")
LIMITS_SECONDS = dict(cpu=2400, native=3300, scorer=6600, fit=13800,
                      audit=6600, smoke=13800, pilot=82800, followup=82800)
RUNTIME_PYTHON = Path("/work3/s234818/latentmol/env/diffsbdd-v1-attempt001/bin/python")
WORK_ROOT = Path("/work3/s234818/latentmol")
CAMPAIGN_NAME = re.compile(r"diffsbdd-vae-alternation-v1-attempt[0-9]{3,}\Z")
IGNORED_CACHE_PARTS = frozenset(("__pycache__", ".pytest_cache"))
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + str(key))
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_pairs,
                      parse_constant=reject_constant)


def write_new(path, value):
    """Exclusive receipt writes preserve every earlier attempt unchanged."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def check_sha(value):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError("Require a lowercase SHA256 binding")
    return value


def member_path(root, relative):
    """Resolve a safe POSIX manifest member, refusing links outside the root."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or ":" in relative or "\x00" in relative):
        raise ValueError("Unsafe manifest member: " + repr(relative))
    pure = PurePosixPath(relative)
    if (pure.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/"))
            or any(part in IGNORED_CACHE_PARTS or part == ".git" for part in pure.parts)):
        raise ValueError("Unsafe manifest member: " + relative)
    unresolved = root / relative
    if any((root.joinpath(*pure.parts[:index])).is_symlink()
           for index in range(1, len(pure.parts) + 1)):
        raise ValueError("Symlinked manifest member: " + relative)
    path = unresolved.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Manifest member escapes its root or is not a file: " + relative)
    return path


def _runtime_cache(path, root):
    relative = path.relative_to(root)
    # Bytecode/pytest metadata are runtime data, but importable Python hidden
    # under a cache directory must never escape the complete source inventory.
    return (("__pycache__" in relative.parts and path.suffix == ".pyc")
            or (".pytest_cache" in relative.parts and path.suffix != ".py"))


def authenticate_source(source_root, source_manifest, source_manifest_sha256,
                        config, config_sha256):
    root = Path(source_root).resolve(strict=True)
    manifest_path = Path(source_manifest).resolve(strict=True)
    config_path = Path(config).resolve(strict=True)
    if (not root.is_dir() or not manifest_path.is_relative_to(root)
            or not config_path.is_relative_to(root)):
        raise ValueError("Source manifest and config must belong to this release")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Immutable source release cannot contain symlinks")
    check_sha(source_manifest_sha256)
    check_sha(config_sha256)
    if digest(manifest_path) != source_manifest_sha256:
        raise ValueError("Source manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("version") != SOURCE_VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported versioned source manifest")
    members = manifest["files"]
    if not members:
        raise ValueError("Source manifest has no members")
    manifest_relative = manifest_path.relative_to(root).as_posix()
    if manifest_relative in members:
        raise ValueError("Source manifest cannot hash itself")
    for name, expected in members.items():
        path = member_path(root, name)
        if digest(path) != check_sha(expected):
            raise ValueError("Source member SHA256 mismatch: " + name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not _runtime_cache(path, root)}
    if actual != set(members) | {manifest_relative}:
        raise ValueError("Source release file set differs from its authenticated manifest")
    config_relative = config_path.relative_to(root).as_posix()
    if members.get(config_relative) != config_sha256 or digest(config_path) != config_sha256:
        raise ValueError("Config SHA256 differs from the authenticated source/config binding")
    configuration = read_json(config_path)
    if (not isinstance(configuration, dict) or not isinstance(configuration.get("version"), str)
            or not configuration["version"]):
        raise ValueError("Config must be a versioned JSON object")
    return dict(source_root=str(root), source_manifest=str(manifest_path),
                source_manifest_sha256=source_manifest_sha256,
                config=str(config_path), config_sha256=config_sha256,
                config_version=configuration["version"], source_members=len(members))


def allocation(stage, environ=None):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    env = os.environ if environ is None else environ
    job, queue = env.get("LSB_JOBID", ""), env.get("LSB_QUEUE", "")
    if not job.isdigit() or not queue:
        raise RuntimeError("Numerical execution requires an identifiable LSF allocation")
    if (stage == "cpu" and queue != "hpc") or (stage != "cpu" and not queue.startswith("gpu")):
        raise RuntimeError("Stage requires its declared CPU/GPU LSF queue class")
    return {key: env.get(key) for key in ("LSB_JOBID", "LSB_QUEUE", "LSB_HOSTS",
                "LSB_DJOB_NUMPROC", "LSB_BIND_CPU_LIST", "CUDA_VISIBLE_DEVICES")}


def validate_stage_output(output, stage):
    """Bind the pass decision to its complete, versioned artifact inventory."""
    if Path(output).is_symlink():
        raise ValueError("Stage output directory cannot be a symlink")
    root = Path(output).resolve(strict=True)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Stage output cannot contain symlinked artifacts")
    stage_path = root / "stage.json"
    result = read_json(stage_path)
    checks = result.get("checks")
    if (result.get("version") != STAGE_VERSION or result.get("stage") != stage
            or result.get("passed") is not True or not isinstance(checks, dict)
            or not checks or any(value is not True for value in checks.values())):
        raise ValueError("Stage lacks an exact versioned all-checks pass: " + stage)
    binding = result.get("output_manifest")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Passed stage must bind its versioned output manifest")
    manifest_path = member_path(root, binding["path"])
    if manifest_path == stage_path or digest(manifest_path) != check_sha(binding["sha256"]):
        raise ValueError("Stage output manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    if manifest.get("version") != OUTPUT_VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported stage output manifest")
    members = manifest["files"]
    excluded = {"stage.json", manifest_path.relative_to(root).as_posix()}
    if set(members) & excluded:
        raise ValueError("Stage artifact inventory cannot contain its own decision/manifest")
    for name, expected in members.items():
        if digest(member_path(root, name)) != check_sha(expected):
            raise ValueError("Stage output artifact SHA256 mismatch: " + name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and not _runtime_cache(path, root)}
    if actual != set(members) | excluded:
        raise ValueError("Stage output file set differs from its authenticated artifact inventory")
    return dict(stage=stage, stage_json_sha256=digest(stage_path),
                output_manifest_sha256=binding["sha256"], artifacts=len(members))


def predecessors(run_root, stage, identity):
    """Verify every earlier stage and its recursively bound receipt chain."""
    root = Path(run_root).resolve(strict=True)
    chain = []
    for previous in STAGES[:STAGES.index(stage)]:
        receipt_path = root / "operator" / previous / "receipt.json"
        receipt = read_json(receipt_path)
        if (receipt.get("version") != VERSION or receipt.get("stage") != previous
                or receipt.get("passed") is not True or receipt.get("identity") != identity
                or receipt.get("predecessors") != chain):
            raise ValueError("Required predecessor operator receipt is not an exact release pass: " + previous)
        bound = validate_stage_output(root / previous, previous)
        if receipt.get("output") != bound:
            raise ValueError("Required predecessor output differs from its passed operator receipt: " + previous)
        chain.append({**bound, "operator_receipt_sha256": digest(receipt_path)})
    return chain


def _terminate(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def launch(command, source_root, environment, log_path, timeout):
    """One child process; the child process group also stops on timeout."""
    with Path(log_path).open("xb") as stream:
        process = subprocess.Popen(command, cwd=source_root, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == "posix")
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            _terminate(process)
            raise


def run_stage(*, stage, source_root, source_manifest, source_manifest_sha256,
              config, config_sha256, run_root, python, internal_seconds):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    if (not isinstance(internal_seconds, (int, float)) or isinstance(internal_seconds, bool)
            or not math.isfinite(internal_seconds) or not 0 < internal_seconds <= LIMITS_SECONDS[stage]):
        raise ValueError("Stage internal wall time exceeds its sealed protocol cap")
    started, started_utc = time.monotonic(), utc_now()
    root = Path(run_root).resolve()
    source_root = Path(source_root).resolve(strict=True)
    work = WORK_ROOT.resolve()
    if (root.parent != work / "runs" or not CAMPAIGN_NAME.fullmatch(root.name)
            or source_root != work / "releases" / root.name / "source"):
        raise ValueError("Only this independent DiffSBDD release/run namespace may be written")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise ValueError("Runtime outputs must be independent of the immutable release")
    output = root / stage
    control = root / "operator" / stage
    if (root / "operator").is_symlink():
        raise ValueError("Operator controls cannot follow a symlink into another campaign")
    if output.exists() or output.is_symlink() or control.exists() or control.is_symlink():
        raise FileExistsError("Stage was already attempted; preserve it and use a new immutable attempt")
    control.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, stage=stage, passed=False,
                  status="initializing_not_passed", started_utc=started_utc,
                  internal_seconds=internal_seconds, numerical_processes_launched=0)
    try:
        identity = authenticate_source(source_root, source_manifest, source_manifest_sha256,
                                       config, config_sha256)
        runtime = Path(python).absolute()
        # Keep the venv path (rather than resolving its interpreter symlink) so
        # Python observes the dedicated runtime's pyvenv.cfg and package layer.
        if runtime != RUNTIME_PYTHON.absolute() or not runtime.is_file():
            raise ValueError("Only the independent experiment runtime may execute this campaign")
        identity.update(run_root=str(root), python=str(runtime))
        report["identity"] = identity
        report["allocation"] = allocation(stage)
        chain = predecessors(root, stage, identity)
        report["predecessors"] = chain
        command = [str(runtime), "-m", "latentmol.diffsbdd.experiment", "--config", identity["config"],
                   "--stage", stage, "--output", str(output)]
        report.update(status="running_not_passed", command=command)
        write_new(control / "intent.json", report)
        environment = dict(os.environ)
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
            PYTHONPATH=str(source_root / "src"),
            DIFFSBDD_SOURCE_MANIFEST_SHA256=source_manifest_sha256,
            DIFFSBDD_CONFIG_SHA256=config_sha256,
            LATENTMOL_SOURCE_SHA256=source_manifest_sha256,
            LATENTMOL_DIFFSBDD_SOURCE_MANIFEST_SHA256=source_manifest_sha256,
            LATENTMOL_DIFFSBDD_CONFIG_SHA256=config_sha256,
            DIFFSBDD_DEADLINE_UNIX=str(time.time() + internal_seconds - (time.monotonic() - started)))
        remaining = internal_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Internal stage deadline expired during authentication")
        report["numerical_processes_launched"] = 1
        returncode = launch(command, str(source_root), environment,
                            control / "stage.log", remaining)
        report["returncode"] = returncode
        if returncode != 0:
            raise RuntimeError("Numerical stage process exited nonzero: " + str(returncode))
        final_identity = authenticate_source(source_root, source_manifest, source_manifest_sha256,
                                             config, config_sha256)
        final_identity.update(run_root=str(root), python=str(runtime))
        if final_identity != identity or predecessors(root, stage, identity) != chain:
            raise ValueError("Release/config/predecessor identities changed during this stage")
        report["output"] = validate_stage_output(output, stage)
        if time.monotonic() - started > internal_seconds:
            raise TimeoutError("Internal stage deadline expired before authenticated completion")
        report.update(passed=True, status="passed")
    except BaseException as error:
        report.update(status="failed_not_passed", error_type=type(error).__name__, error=str(error))
        if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            report["status"] = "timed_out_not_passed"
        raise
    finally:
        report.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - started)
        log = control / "stage.log"
        if log.is_file():
            report["log_sha256"] = digest(log)
        write_new(control / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run",))
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=RUNTIME_PYTHON)
    parser.add_argument("--internal-seconds", type=float, required=True)
    args = vars(parser.parse_args())
    args.pop("action")
    result = run_stage(**args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
