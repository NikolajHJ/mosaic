"""Allocated metadata/toy boundary contracts, never real Protenix execution.

The CPU callback's genuine production command capture is separate. Genuine GPU
entry later executes the unchanged real driver/backend/provenance to its model
constructor sentinel. The toy GPU metadata below cannot establish that pass.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import subprocess

import pytest

try:
    import extended_startup as startup
except ModuleNotFoundError:
    path = Path(__file__).with_name("extended_startup.py")
    spec = importlib.util.spec_from_file_location("extended_startup", path)
    startup = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = startup
    spec.loader.exec_module(startup)


STARTUP_CASES = 34

TOY_DRIVER = '''import argparse,hashlib,importlib,json,os
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('mode','config','output','manifest','cpu-certificate','cpu-certificate-sha256','reference','pocket','pocket-sha256','offset','segment-steps'):
 p.add_argument('--'+name,required=True)
for name in ('ready','ready-sha256','resume-snapshot','resume-sha256'):p.add_argument('--'+name)
p.add_argument('--cache',default='/toy/cache')
a=p.parse_args()
# These are explicitly toy backend metadata, not a substitute GPU assertion.
if os.environ.get('PRO_STARTUP_TOY_BACKEND','gpu')!='gpu' or os.environ.get('PRO_STARTUP_TOY_X64','0')!='0':
 raise RuntimeError('Toy unchanged GPU/FP32 guard rejected')
source=Path(__file__).resolve().parents[1]
manifest=json.loads((source/'source_manifest.json').read_bytes())
files={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in manifest['files']
 if name=='pyproject.toml' or (Path(name).parts[0] in ('src','tests','examples','scripts','experiments')
 and Path(name).suffix in ('.py','.json','.lsf','.sh','.toml','.txt') and '__pycache__' not in Path(name).parts)}
out=Path(a.output);out.mkdir()
r={'status':'running','source_manifest_sha256':a.manifest,'source':{
 'git_revision_reference':manifest['git_revision_reference'],'files':files,
 'execution_content_sha256':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()}}
try:
 model=importlib.import_module('mosaic.small_molecule.protenix').FrozenProtenixTiny.load(cache_dir=Path(a.cache))
except Exception as error:
 r.update(status='failed',error=type(error).__name__+': '+str(error));raise
finally:
 (out/'summary.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\\n')
'''


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def writable(root):
    root.chmod(0o755)
    for path in root.rglob("*"):
        if not path.is_symlink():
            path.chmod(0o755 if path.is_dir() else 0o644)


def seal(root):
    files = {path.relative_to(root).as_posix(): startup.sha(path) for path in root.rglob("*")
             if path.is_file() and path.name != "source_manifest.json"}
    write(root / "source_manifest.json", {"files": files, "git_revision_reference": startup.REVISION})
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return startup.sha(root / "source_manifest.json")


@pytest.fixture
def source(tmp_path, monkeypatch):
    # Never synthesize an allocation ID. All these fixtures themselves run on
    # allocated CPU; only the explicitly toy entry tests change queue metadata.
    assert os.environ.get("LSB_JOBID") and os.environ.get("LSB_QUEUE") == "hpc"
    root = tmp_path / "source"
    (root / "examples").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="toy-pro-startup"\nversion="0"\n')
    (root / "examples/brd4_25_atoms.py").write_text(TOY_DRIVER)
    model_path = root / "src/mosaic/small_molecule/protenix.py"
    model_path.parent.mkdir(parents=True)
    (root / "src/mosaic/__init__.py").write_text("# Toy package; no model dependency.\n")
    (root / "src/mosaic/small_molecule/__init__.py").write_text("# Toy package.\n")
    model_path.write_text('class FrozenProtenixTiny:\n @staticmethod\n def load(*a,**k):\n  raise AssertionError("Toy constructor must be intercepted")\n')
    helper = root / "scripts/extended_startup.py"
    helper.parent.mkdir()
    helper.write_bytes(Path(startup.__file__).read_bytes())
    manifest = seal(root)
    # A child executes the exact helper bytes from its own sealed toy source.
    # No global fake JAX module and no in-memory-only model substitution.
    monkeypatch.setattr(startup, "__file__", str(helper))
    for name in ("mosaic.small_molecule.protenix", "mosaic.small_molecule", "mosaic"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    try:
        yield root, manifest
    finally:
        for name in ("mosaic.small_molecule.protenix", "mosaic.small_molecule", "mosaic"):
            sys.modules.pop(name, None)
        writable(root)


@pytest.fixture
def controller(source, tmp_path):
    root, manifest = source
    assets = tmp_path / "assets"
    for name in ("random.json", "continuation.json", "cpu.json", "reference.json", "pocket.json", "ready.json", "snapshot.npz"):
        write(assets / name, {"synthetic_metadata_only": name})
    ops = SimpleNamespace(SOURCE=root, ROOT=tmp_path / "actual-campaign")
    actions = []

    def offset(arm, index):
        limit = 5 if arm == "random" else 4
        if arm not in ("random", "continuation") or type(index) is not int or not 0 <= index < limit:
            raise ValueError("Undeclared Protenix segment")
        return (0 if arm == "random" else 10000) + index * 10000

    def inputs(arm, phase, index, supplied_manifest):
        assert supplied_manifest == manifest
        paths = {"config": assets / (arm + ".json"), "cpu-certificate": assets / "cpu.json",
                 "reference": assets / "reference.json", "pocket": assets / "pocket.json"}
        if phase == "segment": paths["ready"] = assets / "ready.json"
        if offset(arm, index): paths["resume-snapshot"] = assets / "snapshot.npz"
        return {key: {"path": str(path), "bytes": path.stat().st_size, "sha256": startup.sha(path)}
                for key, path in paths.items()}

    def produce(parent, phase, arm, index, supplied_manifest):
        values = inputs(arm, phase, index, supplied_manifest)
        actions.extend(("actual_factory", "cpu-proof", "snapshot-proof", "storage"))
        payload = parent / "payload"; payload.mkdir()
        args = [sys.executable, str(root / "examples/brd4_25_atoms.py"), "--mode", "validate" if phase == "native" else "run",
                "--config", values["config"]["path"], "--output", str(payload / "study"), "--manifest", manifest,
                "--cpu-certificate", values["cpu-certificate"]["path"], "--cpu-certificate-sha256", values["cpu-certificate"]["sha256"],
                "--reference", values["reference"]["path"], "--pocket", values["pocket"]["path"], "--pocket-sha256", values["pocket"]["sha256"],
                "--offset", str(offset(arm, index)), "--segment-steps", "10000"]
        if "resume-snapshot" in values:
            args += ["--resume-snapshot", values["resume-snapshot"]["path"], "--resume-sha256", values["resume-snapshot"]["sha256"]]
        if "ready" in values:
            args += ["--ready", values["ready"]["path"], "--ready-sha256", values["ready"]["sha256"]]
        startup.subprocess.run(args, cwd=ops.SOURCE, check=True, timeout=60)
    ops.offset, ops.startup_inputs, ops.produce = offset, inputs, produce
    return ops, manifest, actions


@pytest.mark.parametrize("arm,index,phase", (
    ("random", 0, "native"), ("random", 0, "segment"), ("random", 4, "native"), ("random", 4, "segment"),
    ("continuation", 0, "native"), ("continuation", 0, "segment"), ("continuation", 3, "native"), ("continuation", 3, "segment"),
))
def test_capture_uses_actual_factory_and_preserves_global_ready_predecessor_scope(controller, tmp_path, arm, index, phase):
    ops, manifest, actions = controller
    args = startup.capture_command(ops, manifest, tmp_path / "capture", arm=arm, index=index, phase=phase)
    assert actions == ["actual_factory", "cpu-proof", "snapshot-proof", "storage"]
    assert startup.argument(args, "--offset") == str(ops.offset(arm, index))
    assert ("--ready" in args) is (phase == "segment")
    assert ("--resume-snapshot" in args) is (ops.offset(arm, index) > 0)
    for path_flag, sha_flag in (("--ready", "--ready-sha256"), ("--resume-snapshot", "--resume-sha256")):
        if path_flag in args: assert startup.sha(startup.argument(args, path_flag)) == startup.argument(args, sha_flag)
    assert not ops.ROOT.exists()


@pytest.mark.parametrize("fault", ("arm", "phase", "index", "manifest", "escape", "duplicate"))
def test_undeclared_or_nonexact_generated_commands_cannot_certify_startup(controller, tmp_path, fault):
    ops, manifest, _ = controller
    original = ops.produce
    if fault in ("arm", "phase", "index"):
        with pytest.raises(ValueError):
            startup.capture_command(ops, manifest, tmp_path / "bad", arm="other" if fault == "arm" else "random",
                                    phase="other" if fault == "phase" else "native", index=-1 if fault == "index" else 0)
        return
    def bad(parent, phase, arm, index, supplied_manifest):
        if fault == "duplicate":
            original(parent, phase, arm, index, supplied_manifest)
            startup.subprocess.run([sys.executable, str(ops.SOURCE / "examples/brd4_25_atoms.py")], cwd=ops.SOURCE, check=True)
            return
        args = [sys.executable, str(ops.SOURCE / "examples/brd4_25_atoms.py"), "--manifest", "a" * 64 if fault == "manifest" else manifest,
                "--mode", "validate", "--offset", "0", "--segment-steps", "10000", "--output", str(tmp_path / "escape")]
        startup.subprocess.run(args, cwd=ops.SOURCE, check=True)
    ops.produce = bad
    with pytest.raises(ValueError): startup.capture_command(ops, manifest, tmp_path / "bad", arm="random")


@pytest.mark.parametrize("fault", ("missing_revision", "changed_member", "writable", "manifest_hash"))
def test_missing_provenance_or_modified_execution_seal_cannot_enter_startup(source, fault):
    root, manifest = source
    writable(root)
    path = root / "source_manifest.json"
    if fault == "missing_revision":
        value = json.loads(path.read_bytes()); del value["git_revision_reference"]; write(path, value); manifest = startup.sha(path)
    elif fault == "changed_member": (root / "examples/brd4_25_atoms.py").write_text("# changed\n")
    elif fault == "manifest_hash": manifest = "a" * 64
    for item in root.rglob("*"): item.chmod(0o555 if item.is_dir() else 0o444)
    root.chmod(0o755 if fault == "writable" else 0o555)
    with pytest.raises((ValueError, KeyError)):
        startup.authenticate_source(root, manifest)


def test_cpu_capture_explicitly_does_not_execute_gpu_driver_or_claim_a_startup_pass(controller, tmp_path):
    ops, manifest, _ = controller
    result = startup.capture_cpu(ops, manifest, tmp_path / "cpu-only", arm="continuation")
    assert result["kind"] == "cpu_generated_command_only" and result["status"] == "command_only_authenticated"
    assert result["actual_driver_executed"] is result["gpu_startup_passed"] is False
    assert result["model_loaded"] is result["gradient_passed"] is result["scientific_readiness_passed"] is False
    assert result["optimizer_updates"] == 0 and result["job_id"] == os.environ["LSB_JOBID"]
    assert not list((tmp_path / "cpu-only").rglob("summary.json")) and not ops.ROOT.exists()


def test_toy_gpu_boundary_remains_failed_summary_without_model_or_scientific_pass(controller, tmp_path, monkeypatch):
    ops, manifest, _ = controller
    # Toy allocation/backend metadata only; actual GPU proof runs independently.
    monkeypatch.setenv("LSB_QUEUE", "gpul40s")
    result = startup.gpu_startup(ops, manifest, tmp_path / "toy-gpu", arm="random")
    assert result["kind"] == "gpu_startup_only" and result["status"] == "startup_only_passed"
    assert result["actual_driver_executed"] is result["gpu_backend_guard_executed_unchanged"] is True
    assert result["summary"]["status"] == "failed"
    assert result["summary"]["error"] == "ProtenixPremodelBoundaryReached: " + startup.MESSAGE
    assert result["boundary"]["constructor"] == "FrozenProtenixTiny.load"
    assert result["model_loaded"] is result["gradient_passed"] is result["scientific_readiness_passed"] is False
    assert result["optimizer_updates"] == 0 and not ops.ROOT.exists()
    isolation = result["execution_isolation"]
    assert isolation["mode"] == "short_lived_same_allocation_subprocess"
    assert isolation["returncode"] == 0 and isolation["child_fully_reaped"] is True
    assert isolation["pid"] != os.getpid() and isolation["parent_pid"] == os.getpid()
    assert isolation["parent_native_backend_imported"] is False
    assert isolation["child_receipt_value"]["pid"] == isolation["pid"]


@pytest.mark.parametrize("fault", ("queue", "backend", "precision"))
def test_gpu_entry_and_unchanged_backend_precision_guards_reject_before_model(controller, tmp_path, monkeypatch, fault):
    ops, manifest, _ = controller
    args = startup.capture_command(ops, manifest, tmp_path / "guard", arm="random")
    if fault != "queue": monkeypatch.setenv("LSB_QUEUE", "gpul40s")
    if fault == "backend": monkeypatch.setenv("PRO_STARTUP_TOY_BACKEND", "cpu")
    if fault == "precision": monkeypatch.setenv("PRO_STARTUP_TOY_X64", "1")
    with pytest.raises((ValueError, RuntimeError)):
        startup.execute_command(args, ops.SOURCE, manifest)
    assert not list((tmp_path / "guard").rglob("assessment.json"))


@pytest.mark.parametrize("flag", ("--config", "--mode", "--manifest"))
def test_actual_toy_argument_parser_rejects_missing_required_command_fields(controller, tmp_path, monkeypatch, flag):
    ops, manifest, _ = controller
    args = startup.capture_command(ops, manifest, tmp_path / "missing", arm="random")
    position = args.index(flag); del args[position:position + 2]
    monkeypatch.setenv("LSB_QUEUE", "gpul40s")
    with pytest.raises((ValueError, SystemExit)):
        startup.execute_command(args, ops.SOURCE, manifest)
    assert not list((tmp_path / "missing").rglob("assessment.json"))


@pytest.mark.parametrize("fault", ("existing", "outside"))
def test_all_capture_outputs_must_be_new_owned_node_temporary_paths(source, tmp_path, fault):
    directory = tmp_path / "owned"
    if fault == "existing": directory.mkdir()
    else: directory = tmp_path.parent / "outside"
    with pytest.raises(ValueError): startup.owned_tmp(directory, "hpc")


@pytest.mark.parametrize("fault", ("nonzero", "missing", "tampered", "timeout", "parent_backend", "foreign_pid"))
def test_isolated_child_failure_or_unreaped_or_unbound_receipt_cannot_admit_production(controller, tmp_path, monkeypatch, fault):
    ops, manifest, _ = controller
    directory = tmp_path / "child-fault"
    args = startup.capture_command(ops, manifest, directory, arm="random")
    monkeypatch.setenv("LSB_QUEUE", "gpul40s")
    events = []

    class Child:
        pid = 65432
        def __init__(self, command, **kwargs):
            events.append("spawn")
            assert command[:2] == [sys.executable, str(Path(startup.__file__).resolve())]
            assert "--execute-child" in command and kwargs["cwd"] == ops.SOURCE
            assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["stdout"] == kwargs["stderr"] == subprocess.PIPE
            assert "env" not in kwargs  # inherited real LSF owner/queue, never a replacement allocation
            self.command, self.calls = command, 0
            self.returncode = 1 if fault == "nonzero" else 0
            command_path = Path(command[command.index("--argv-json") + 1])
            receipt = Path(command[command.index("--receipt") + 1])
            if fault != "missing":
                write(receipt, {"version": startup.VERSION, "status": "startup_only_passed",
                    "job_id": os.environ["LSB_JOBID"], "queue": "gpul40s", "source": str(ops.SOURCE),
                    "source_manifest_sha256": "a" * 64 if fault == "tampered" else manifest,
                    "command_sha256": startup.sha(command_path), "pid": self.pid + (1 if fault == "foreign_pid" else 0),
                    "parent_pid": os.getpid(), "summary": {}, "summary_sha256": "b" * 64, "boundary": {}})
        def communicate(self, timeout=None):
            self.calls += 1; events.append("communicate")
            if fault == "timeout" and self.calls == 1:
                raise subprocess.TimeoutExpired(self.command, timeout)
            return b"synthetic child stdout\n", b"synthetic child stderr\n"
        def kill(self):
            events.append("kill_owned_child"); self.returncode = -9
        def poll(self):
            return self.returncode

    monkeypatch.setattr(startup.subprocess, "Popen", Child)
    if fault == "parent_backend":
        # Presence-only toy metadata tests the prohibition without importing
        # JAX or initializing any numerical/backend state in this CPU process.
        monkeypatch.setitem(sys.modules, "jax", ModuleType("jax"))
    with pytest.raises(ValueError):
        startup.run_startup_child(args, ops.SOURCE, manifest, directory)
    assert not ops.ROOT.exists()
    if fault == "parent_backend":
        assert events == [] and not (directory / "child-command.json").exists()
    else:
        failure = json.loads((directory / "child-exit.json").read_bytes())
        assert failure["pid"] == Child.pid and failure["parent_pid"] == os.getpid()
        assert failure["child_fully_reaped"] is True
        assert (directory / "child-stdout.bin").read_bytes() == b"synthetic child stdout\n"
        assert (directory / "child-stderr.bin").read_bytes() == b"synthetic child stderr\n"
        if fault == "timeout":
            assert events == ["spawn", "communicate", "kill_owned_child", "communicate"]
            assert failure["timed_out"] is True and failure["returncode"] == -9
