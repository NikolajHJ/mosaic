"""Allocated startup control contracts; no native worker is ever constructed."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

try:
    import premodel_startup as startup
except ModuleNotFoundError:
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "scripts/hpc/startup/premodel_startup.py"
    spec = importlib.util.spec_from_file_location("premodel_startup", path)
    startup = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = startup
    spec.loader.exec_module(startup)


STARTUP_CASES = 28
REVISION = "c74c659344123937608077e3e4078f75f09cd743"


DRIVER = '''import argparse, json
from pathlib import Path
from mosaic.small_molecule.provenance import source_identity, environment_identity
from dbmol_bridge import DBMolWorker
p=argparse.ArgumentParser()
for flag in ('config','output','manifest','cpu-certificate','cpu-certificate-sha256','reference','pocket','prepared','context-sha256','mode','offset','segment-steps'):
 p.add_argument('--'+flag,required=True)
p.add_argument('--ready');p.add_argument('--ready-sha256')
p.add_argument('--resume-snapshot');p.add_argument('--resume-sha256')
a=p.parse_args()
out=Path(a.output);out.mkdir()
r={'status':'running','source_manifest_sha256':a.manifest,
   'source':source_identity(Path(__file__).resolve().parents[1]),
   'environment':environment_identity()}
try:
 DBMolWorker(['fixed-allocated-worker','--context',str(Path(a.prepared)/'context.pt')],stderr=out/'worker.log')
except Exception as e:
 r.update(status='failed',error=type(e).__name__+': '+str(e));raise
finally:
 (out/'summary.json').write_text(json.dumps(r,sort_keys=True)+'\\n')
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def writable(root):
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def seal(root, *, revision=REVISION):
    files = {p.relative_to(root).as_posix(): startup.sha(p)
             for p in root.rglob("*") if p.is_file() and p.name != "source_manifest.json"}
    manifest = {"files": files, "git_revision_reference": revision}
    write(root / "source_manifest.json", manifest)
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return startup.sha(root / "source_manifest.json")


@pytest.fixture
def source(tmp_path, monkeypatch):
    assert os.environ.get("LSB_JOBID"), "Run startup fixtures only in an allocated HPC job"
    root = tmp_path / "source"
    (root / "scripts").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="synthetic-startup-contract"\nversion="0"\n')
    (root / "scripts/dbmol_driver.py").write_text(DRIVER)
    (root / "scripts/dbmol_bridge.py").write_text('class DBMolWorker:\n def __init__(self,*a,**k):\n  raise AssertionError("Unpatched fixture worker constructor")\n')
    digest = seal(root)
    saved = sys.modules.pop("dbmol_bridge", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    try:
        yield root, digest
    finally:
        sys.modules.pop("dbmol_bridge", None)
        if saved is not None:
            sys.modules["dbmol_bridge"] = saved
        writable(root)


@pytest.fixture
def controller(source, tmp_path):
    root, manifest = source
    assets = tmp_path / "assets"
    for name in ("config.json", "cpu.json", "reference.json", "pocket.json", "ready.json", "snapshot.npz"):
        write(assets / name, {"metadata_fixture_only": name})
    write(assets / "prepared/context.pt", {"fixture": "context"})
    actions = []
    ops = SimpleNamespace(SOURCE=root, ROOT=tmp_path / "actual-campaign", SEGMENT=2500)

    def produce(parent, phase, index, supplied_manifest):
        # This is an intentionally small controller fixture. Actual callback
        # smoke calls each immutable production controller's real produce.
        actions.extend(("actual_factory", "cpu-proof", "input-snapshot", "storage", "context"))
        assert supplied_manifest == manifest and startup.sha(assets / "cpu.json")
        payload = parent / "payload"
        payload.mkdir()
        args = [sys.executable, str(root / "scripts/dbmol_driver.py"),
                "--mode", "validate" if phase == "native" else "run",
                "--config", str(assets / "config.json"), "--output", str(payload / "study"),
                "--manifest", manifest, "--cpu-certificate", str(assets / "cpu.json"),
                "--cpu-certificate-sha256", startup.sha(assets / "cpu.json"),
                "--reference", str(assets / "reference.json"), "--pocket", str(assets / "pocket.json"),
                "--prepared", str(assets / "prepared"), "--context-sha256", startup.sha(assets / "prepared/context.pt"),
                "--offset", str(index * ops.SEGMENT), "--segment-steps", str(ops.SEGMENT)]
        if phase == "segment":
            args += ["--ready", str(assets / "ready.json"), "--ready-sha256", startup.sha(assets / "ready.json")]
        if index:
            args += ["--resume-snapshot", str(assets / "snapshot.npz"), "--resume-sha256", startup.sha(assets / "snapshot.npz")]
        startup.subprocess.run(args, cwd=ops.SOURCE, check=True, timeout=60)
    ops.produce = produce
    return ops, manifest, actions


def test_real_source_identity_reads_explicit_authenticated_revision(source):
    from mosaic.small_molecule.provenance import source_identity
    root, manifest = source
    sealed = startup.authenticate_source(root, manifest)
    observed = source_identity(root)
    assert observed["git_revision_reference"] == REVISION
    assert observed["files"] == sealed["files"]


@pytest.mark.parametrize("fault", ("missing_revision", "malformed_revision", "manifest_hash", "member_hash", "extra", "missing_member", "writable", "symlink", "bad_relative"))
def test_missing_metadata_or_changed_sealed_execution_source_is_rejected(source, fault):
    root, manifest = source
    writable(root)
    path = root / "source_manifest.json"
    value = json.loads(path.read_bytes())
    if fault == "missing_revision": del value["git_revision_reference"]
    elif fault == "malformed_revision": value["git_revision_reference"] = "invented"
    elif fault == "manifest_hash": manifest = "a" * 64
    elif fault == "member_hash": (root / "scripts/dbmol_driver.py").write_text("# modified\n")
    elif fault == "extra": (root / "undeclared.py").write_text("# undeclared\n")
    elif fault == "missing_member": (root / "scripts/dbmol_driver.py").unlink()
    elif fault == "symlink": (root / "alias.py").symlink_to(root / "scripts/dbmol_driver.py")
    elif fault == "bad_relative": value["files"]["../escape.py"] = "a" * 64
    if fault in ("missing_revision", "malformed_revision", "bad_relative"):
        write(path, value); manifest = startup.sha(path)
    for item in root.rglob("*"):
        if not item.is_symlink(): item.chmod(0o555 if item.is_dir() else 0o444)
    root.chmod(0o755 if fault == "writable" else 0o555)
    if fault == "missing_revision":
        from mosaic.small_molecule.provenance import source_identity
        with pytest.raises(KeyError, match="git_revision_reference"):
            source_identity(root)
    with pytest.raises((ValueError, OSError)):
        startup.authenticate_source(root, manifest)


def test_command_capture_uses_actual_factory_without_replacing_its_guards(controller, tmp_path):
    ops, manifest, actions = controller
    args = startup.capture_generated_command(ops, manifest, tmp_path / "capture")
    assert actions == ["actual_factory", "cpu-proof", "input-snapshot", "storage", "context"]
    assert startup._argument(args, "--manifest") == manifest
    assert startup._argument(args, "--output").startswith(str(tmp_path / "capture"))
    assert not ops.ROOT.exists()


@pytest.mark.parametrize("fault", ("phase", "index", "missing_driver", "two_drivers", "escape_output", "wrong_manifest"))
def test_nonexact_generated_commands_cannot_be_startup_proofs(controller, tmp_path, fault):
    ops, manifest, _ = controller
    original = ops.produce
    if fault in ("phase", "index"):
        with pytest.raises(ValueError):
            startup.capture_generated_command(ops, manifest, tmp_path / "reject", phase="unknown" if fault == "phase" else "native", index=-1 if fault == "index" else 0)
        return
    def altered(parent, phase, index, supplied_manifest):
        if fault == "missing_driver": return
        if fault == "two_drivers":
            original(parent, phase, index, supplied_manifest)
            args = [sys.executable, str(ops.SOURCE / "scripts/dbmol_driver.py")]
            startup.subprocess.run(args, cwd=ops.SOURCE, check=True)
            return
        args = [sys.executable, str(ops.SOURCE / "scripts/dbmol_driver.py"),
                "--mode", "validate", "--manifest", "a" * 64 if fault == "wrong_manifest" else manifest,
                "--offset", "0", "--output", str(tmp_path / "outside"), "--segment-steps", "2500"]
        startup.subprocess.run(args, cwd=ops.SOURCE, check=True)
    ops.produce = altered
    with pytest.raises(ValueError): startup.capture_generated_command(ops, manifest, tmp_path / "reject")


def test_real_driver_parser_provenance_and_boundary_summary_are_executed(controller, tmp_path):
    ops, manifest, _ = controller
    receipt = startup.run_startup_smoke(ops, manifest, tmp_path / "smoke")
    assert receipt["status"] == "startup_only_passed"
    assert receipt["job_id"] == os.environ["LSB_JOBID"]
    assert receipt["model_loaded"] is receipt["gradient_passed"] is receipt["scientific_readiness_passed"] is False
    assert receipt["optimizer_updates"] == 0 and receipt["saved_results_changed"] is False
    assert receipt["summary"]["status"] == "failed"
    assert receipt["summary"]["source"]["git_revision_reference"] == REVISION
    assert not ops.ROOT.exists()


@pytest.mark.parametrize("flag", ("--mode", "--manifest", "--config", "--output", "--cpu-certificate"))
def test_actual_driver_rejects_missing_required_generated_arguments_before_worker(controller, tmp_path, flag):
    ops, manifest, _ = controller
    args = startup.capture_generated_command(ops, manifest, tmp_path / "bad-args")
    position = args.index(flag)
    del args[position:position + 2]
    with pytest.raises((ValueError, SystemExit)):
        startup.execute_generated_command(args, ops.SOURCE, manifest)
    assert not list((tmp_path / "bad-args").rglob("assessment.json"))


@pytest.mark.parametrize("fault", ("outside", "root", "existing", "no_allocation"))
def test_smoke_cannot_write_outside_new_owned_allocated_node_temporary_directory(source, tmp_path, monkeypatch, fault):
    root, _ = source
    directory = tmp_path / "candidate"
    if fault == "outside": directory = tmp_path.parent / "outside"
    elif fault == "root": directory = tmp_path
    elif fault == "existing": directory.mkdir()
    else: monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(ValueError): startup._owned_tmp(directory)


def test_segment_command_retains_actual_readiness_and_saved_predecessor_arguments(controller, tmp_path):
    ops, manifest, _ = controller
    args = startup.capture_generated_command(ops, manifest, tmp_path / "segment", phase="segment", index=1)
    assert startup._argument(args, "--mode") == "run"
    assert startup._argument(args, "--offset") == "2500"
    for name, digest_name in (("--ready", "--ready-sha256"), ("--resume-snapshot", "--resume-sha256")):
        assert startup.sha(startup._argument(args, name)) == startup._argument(args, digest_name)
