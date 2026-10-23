"""Allocated real generated-command startup, stopped before any native model."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from unittest.mock import patch


VERSION = "mosaic-actual-generated-command-premodel-v1"
MESSAGE = "Intentional startup-only boundary; no native model loaded"


class PremodelBoundaryReached(RuntimeError):
    """The only successful endpoint of a startup-only execution."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def authenticate_source(source, manifest):
    """Authenticate complete sealed execution bytes and explicit revision schema."""
    source = Path(source).resolve()
    path = source / "source_manifest.json"
    require(isinstance(manifest, str) and re.fullmatch("[0-9a-f]{64}", manifest), "Malformed source identity")
    require(sha(path) == manifest, "Source manifest changed")
    value = json.loads(path.read_bytes())
    require("git_revision_reference" in value, "Source manifest lacks explicit git_revision_reference")
    revision = value["git_revision_reference"]
    require(revision is None or (isinstance(revision, str) and re.fullmatch("[0-9a-f]{40}", revision)), "Malformed revision reference")
    files = value.get("files")
    require(isinstance(files, dict) and "pyproject.toml" in files, "Incomplete source file map")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(files) | {"source_manifest.json"}, "Source member scope differs")
    for item in [source, *source.rglob("*")]:
        require(not item.is_symlink() and (item.is_file() or item.is_dir()) and not item.stat().st_mode & 0o222, "Execution source must be sealed read-only regular content")
    for name, digest in files.items():
        member = source / name
        require(member.resolve().is_relative_to(source) and sha(member) == digest, "Source member identity changed")
    return value


def _owned_tmp(directory):
    require(os.environ.get("LSB_JOBID") and os.environ.get("LSB_QUEUE") == "hpc", "Allocated DTU HPC CPU callback required")
    require(os.environ.get("TMPDIR"), "Owned node TMPDIR required")
    root = Path(os.environ["TMPDIR"]).resolve()
    directory = Path(directory)
    require(not directory.is_symlink() and not directory.exists(), "Startup directory must be new")
    require(directory.resolve().is_relative_to(root) and directory.resolve() != root, "Startup outputs must stay below node TMPDIR")
    return directory.resolve()


@contextmanager
def _execution_context(source, argv):
    old_argv, old_path, old_cwd = sys.argv[:], sys.path[:], Path.cwd()
    try:
        sys.argv = list(argv)
        sys.path[:0] = [str(source / "scripts"), str(source / "src"), str(source / "tests")]
        os.chdir(source)
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        sys.argv[:] = old_argv


def _argument(args, name):
    require(args.count(name) == 1 and args.index(name) + 1 < len(args), "Missing or duplicated driver argument: " + name)
    return args[args.index(name) + 1]


def capture_generated_command(ops, manifest, directory, *, arm=None, index=0, phase="native"):
    """Run real production admission; intercept only its final driver subprocess."""
    require(phase in ("native", "segment") and type(index) is int and index >= 0, "Undeclared startup phase/index")
    directory = Path(directory)
    directory.mkdir()
    parent = directory / "command"
    parent.mkdir()
    source = Path(ops.SOURCE).resolve()
    driver = source / ("scripts/dbmol_driver.py" if arm is None else "examples/brd4_25_atoms.py")
    calls = []
    original_run = subprocess.run

    def capture(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and len(args) > 1 and Path(str(args[1])).resolve() == driver:
            require(not calls and not positional and kwargs.get("cwd") == ops.SOURCE and kwargs.get("check") is True, "Unexpected generated driver call")
            require(not kwargs.get("shell", False), "Driver shell execution is forbidden")
            calls.append([str(value) for value in args])
            return subprocess.CompletedProcess(args, 0)
        return original_run(args, *positional, **kwargs)

    with patch.object(subprocess, "run", capture):
        if arm is None:
            ops.produce(parent, phase, index, manifest)
        else:
            require(arm in ("boltz2", "boltz2_chemistry"), "Undeclared startup arm")
            ops.produce(parent, phase, arm, index, manifest)
    require(len(calls) == 1, "Actual controller did not generate one driver command")
    args = calls[0]
    require(Path(args[0]).resolve() == Path(sys.executable).resolve(), "Startup must use the generated production interpreter")
    require(_argument(args, "--manifest") == manifest and _argument(args, "--mode") == ("validate" if phase == "native" else "run"), "Generated phase identity differs")
    require(("--ready" in args) is (phase == "segment"), "Actual readiness applicability differs")
    require(("--resume-snapshot" in args) is (index > 0), "Actual predecessor applicability differs")
    require(int(_argument(args, "--offset")) == index * int(ops.SEGMENT), "Generated update offset differs")
    output = Path(_argument(args, "--output")).resolve()
    require(output.is_relative_to(directory.resolve()) and not output.exists(), "Generated output escapes disposable startup directory")
    return args


def execute_generated_command(args, source, manifest, *, arm=None):
    """Run the actual driver, patching only its native worker constructor."""
    source = Path(source).resolve()
    manifest_value = authenticate_source(source, manifest)
    expected_driver = source / ("scripts/dbmol_driver.py" if arm is None else "examples/brd4_25_atoms.py")
    require(Path(args[1]).resolve() == expected_driver, "Driver outside authenticated generated entry point")
    output = Path(_argument(args, "--output")).resolve()
    binding = ("dbmol_bridge", "DBMolWorker") if arm is None else ("mosaic.small_molecule.boltz2.bridge", "AffinityWorker")
    reached = []

    def boundary(command, *, stderr, **kwargs):
        reached.append({"command": [str(value) for value in command], "stderr": str(stderr)})
        raise PremodelBoundaryReached(MESSAGE)

    caught = None
    with _execution_context(source, args[1:]):
        module = importlib.import_module(binding[0])
        require(Path(module.__file__).resolve().is_relative_to(source), "Worker adapter imported outside authenticated execution source")
        with patch.object(module, binding[1], boundary):
            try:
                runpy.run_path(args[1], run_name="__main__")
            except PremodelBoundaryReached as error:
                caught = error
    require(caught is not None and len(reached) == 1, "Actual driver did not reach the intentional pre-model boundary")
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    require(summary.get("status") == "failed" and summary.get("error") == "PremodelBoundaryReached: " + MESSAGE, "Startup sentinel must remain an intentional failed driver summary")
    require(summary.get("source_manifest_sha256") == manifest and summary["source"]["git_revision_reference"] == manifest_value["git_revision_reference"], "Actual driver provenance differs")
    execution_files = {name: digest for name, digest in manifest_value["files"].items()
                       if name == "pyproject.toml" or (Path(name).parts[0] in ("src", "tests", "examples", "scripts", "experiments")
                           and Path(name).suffix in (".py", ".json", ".lsf", ".sh", ".toml", ".txt")
                           and "__pycache__" not in Path(name).parts)}
    require(summary["source"]["files"] == execution_files, "Actual driver execution file map differs from manifest")
    require(summary["source"]["execution_content_sha256"] == hashlib.sha256(json.dumps(execution_files, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "Actual execution content digest differs")
    require(not (output / "assessment.json").exists() and not summary.get("native_gradient_checks") and not summary.get("cases"), "Startup check must not produce scientific readiness or optimization")
    return {"summary": summary, "summary_sha256": sha(summary_path), "boundary": reached[0]}


def run_startup_smoke(ops, manifest, directory, *, arm=None, index=0, phase="native"):
    """Return a startup-only receipt; caller persists it before native dispatch."""
    directory = _owned_tmp(directory)
    source = Path(ops.SOURCE).resolve()
    before = authenticate_source(source, manifest)
    args = capture_generated_command(ops, manifest, directory, arm=arm, index=index, phase=phase)
    result = execute_generated_command(args, source, manifest, arm=arm)
    require(authenticate_source(source, manifest) == before, "Execution source changed during startup")
    inputs = {}
    for option in ("--config", "--cpu-certificate", "--reference", "--pocket", "--ready", "--resume-snapshot"):
        if option not in args:
            continue
        path = Path(_argument(args, option))
        inputs[option.removeprefix("--")] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
    require(inputs["cpu-certificate"]["sha256"] == _argument(args, "--cpu-certificate-sha256"), "Actual CPU certificate changed")
    context = Path(_argument(args, "--prepared")) / "context.pt"
    inputs["prepared_context"] = {"path": str(context), "bytes": context.stat().st_size, "sha256": sha(context)}
    require(inputs["prepared_context"]["sha256"] == _argument(args, "--context-sha256"), "Actual prepared context changed")
    for option in ("ready", "resume-snapshot"):
        if option in inputs:
            require(inputs[option]["sha256"] == _argument(args, "--" + ("resume-sha256" if option == "resume-snapshot" else "ready-sha256")), "Actual saved input changed")
    return {"version": VERSION, "status": "startup_only_passed", "job_id": os.environ["LSB_JOBID"],
            "arm": "boltz2_dbmol" if arm is None else arm, "phase": phase, "index": index,
            "execution_owner": "current allocated CPU callback; no GPU job impersonated", "source": str(source),
            "source_manifest_sha256": manifest, "git_revision_reference": before["git_revision_reference"],
            "args": args, "inputs": inputs, **result,
            "model_loaded": False, "scientific_readiness_passed": False, "gradient_passed": False,
            "optimizer_updates": 0, "saved_results_changed": False,
            "disposable_outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
                                   for path in sorted(directory.rglob("*")) if path.is_file()]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--arm", choices=("boltz2", "boltz2_chemistry"))
    parser.add_argument("--phase", choices=("native", "segment"), default="native")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--launcher-manifest")
    args = parser.parse_args()
    require(not os.path.lexists(args.receipt), "Startup receipt already exists; inspect the earlier outcome")
    require(args.receipt.resolve().is_relative_to(args.root.resolve()), "Startup receipt must belong to this campaign root")
    authenticate_source(args.source, args.manifest)
    expected_ops_root = args.launcher or args.source
    require(args.ops.resolve().is_relative_to(expected_ops_root.resolve()), "Controller outside authenticated source")
    if args.launcher:
        # Launcher manifests are full content seals but need not be Git sources.
        require(args.launcher_manifest and sha(args.launcher / "source_manifest.json") == args.launcher_manifest, "Launcher manifest changed")
        launcher_files = json.loads((args.launcher / "source_manifest.json").read_bytes())["files"]
        for name, digest in launcher_files.items():
            require(sha(args.launcher / name) == digest, "Launcher member changed")
    with _execution_context(args.source, [str(args.ops)]):
        spec = importlib.util.spec_from_file_location("_startup_actual_ops", args.ops)
        ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ops
        spec.loader.exec_module(ops)
        require(Path(ops.SOURCE).resolve() == args.source.resolve() and Path(ops.ROOT).resolve() == args.root.resolve(), "Controller root/source differs")
        if args.launcher:
            require(Path(ops.LAUNCHER).resolve() == args.launcher.resolve(), "Wrong controller launcher")
            ops.verify_launcher(args.manifest, args.launcher_manifest)
        receipt = run_startup_smoke(ops, args.manifest, args.directory, arm=args.arm, phase=args.phase, index=args.index)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
