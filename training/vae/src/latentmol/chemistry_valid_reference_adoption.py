"""Authenticate, then explicitly adopt the completed source135 scientific proof.

This is evidence reuse for the user's epoch36->40 training authorization. No
historical receipt is rewritten and no disposable validation state is a resume
checkpoint. Derived launch receipts name both identities and are recomputed at
every launch/resume. The complete old native and 2,500-update verifiers still
run, using the original plan, source identities, observations and thresholds.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET


VERSION = "validated-frozen-reference-adoption-v1"
SOURCE_PLAN_SHA = "ad7533a6ce9011c17ba8c962fe939f1e3391b77b73c4ae7574bdbebaf203c609"
SOURCE_ARCHIVE_SHA = "c9e2c267fd22feb3ed59d3e5b08a76245a58989ffa85c804288cc5a4afafa658"
SOURCE_MANIFEST_SHA = "20908934fb7ced28122fcf0b2a5a5a1ea7db09d7d10733c38ffbf6aa28399108"
SOURCE_STABILITY_SHA = "218a3c9e0318286f9122017cf9c4507885202ed7981bea61852d0c16886a39b0"
SHARED_METADATA_FUNCTIONS = {
    "src/latentmol/chemistry_short_continuation.py": {"validate_short_continuation", "require_training_branch",
        "continuation_end_epoch", "_validate_origin_bindings"},
    "src/latentmol/chemistry_valid_reference_policy.py": {"validate_valid_reference_test"},
    "src/latentmol/chemistry_continuation.py": {"require_failure_readiness", "authenticate_chemistry_handoff",
        "setup", "phase_checkpoint", "train_phase"},
    "src/latentmol/chemistry_early_health.py": {"_baseline_context"},
    "src/latentmol/chemistry_restart_policy.py": {"authenticate_stability"},
    "src/latentmol/failure_guidance_selection.py": {"authenticate_readiness"},
    "scripts/hpc/chemistry_campaign.py": {"advance"},
}
NEW_EXECUTABLES = {
    "src/latentmol/chemistry_valid_reference_training_policy.py",
    "src/latentmol/chemistry_valid_reference_adoption.py",
    "scripts/hpc/chemistry_valid_reference_campaign.py",
    "tests/test_chemistry_valid_reference_training_policy.py",
    "tests/test_chemistry_valid_reference_adoption.py",
    "scripts/hpc/check_chemistry_launch.py",
    "tests/test_chemistry_launch.py",
}


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read(path):
    from .chemistry_restart_policy import _read as strict_read
    return strict_read(path)


def _manifest(release, archive_sha, manifest_sha):
    # Reuse the exact existing archive/coverage/member verifier. It reads bytes
    # only; it cannot start an allocation, import Torch or perform model work.
    from .chemistry_runtime_adoption import _manifest as authenticate_manifest
    return authenticate_manifest(release, archive_sha, manifest_sha)


def compare_source(old_release, new_release, old_files, new_files):
    """The molecular implementation/configuration must remain source135 bytes.

    A closed list permits launch/receipt metadata changes. For each shared
    module, every other existing function and every module constant is also
    AST-identical; in particular all loss, search, gradient and update bodies
    outside the five explicit continuation integration boundaries are unchanged.
    """
    changed = {p for p in old_files.keys() | new_files.keys() if old_files.get(p) != new_files.get(p)}
    _check(not old_files.keys() - new_files.keys(), "Adoption cannot remove source135 members")
    _check(changed <= set(SHARED_METADATA_FUNCTIONS) | NEW_EXECUTABLES,
           "Adoption changed undeclared scientific code/configuration: " + repr(sorted(changed)))
    for name in changed & old_files.keys():
        allowed = SHARED_METADATA_FUNCTIONS.get(name)
        _check(name in SHARED_METADATA_FUNCTIONS, "Unexpected preexisting adoption overlay")
        old = ast.parse((Path(old_release) / name).read_bytes())
        new = ast.parse((Path(new_release) / name).read_bytes())
        def parts(tree):
            functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            constants = [ast.dump(node, include_attributes=False) for node in tree.body
                         if isinstance(node, (ast.Assign, ast.AnnAssign))]
            return functions, constants
        before, old_constants = parts(old)
        after, new_constants = parts(new)
        _check(old_constants == new_constants, "Adoption changed module constants: " + name)
        for function, body in before.items():
            if (name == "src/latentmol/chemistry_continuation.py" and function == "main"
                    and after.get(function) != body):
                # Permit exactly the CLI applicability correction, not arbitrary
                # changes to main or the numerical training implementation.
                original = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                expected = deepcopy(original)
                predicate = ast.parse('plan["protocol"].get("failure_guidance") is not None', mode="eval").body
                matches = [n for n in ast.walk(expected) if isinstance(n, ast.If)
                           and ast.dump(n.test) == ast.dump(predicate)]
                _check(len(matches) == 1, "CLI readiness correction has no unique original boundary")
                matches[0].test = ast.parse('plan["protocol"].get("failure_guidance") is not None '
                    'or plan.get("short_continuation") is not None '
                    'or plan.get("valid_reference_continuation") is not None', mode="eval").body
                _check(after.get(function) == ast.dump(expected, include_attributes=False),
                       "CLI correction changed more than readiness applicability")
                continue
            _check(function in allowed or after.get(function) == body,
                   "Adoption changed a nonmetadata function: " + name + ":" + function)
        _check(set(after) - set(before) <= allowed, "Adoption introduced an undeclared function: " + name)
    return sorted(changed)


def _junit(path, expected_passes, expected_skips):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    _check(suites and not any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures")),
           "Historical allocated fixture suite failed")
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    passed = sum(int(s.get("tests", 0)) - int(s.get("skipped", 0)) for s in suites)
    _check(passed == expected_passes and skipped == expected_skips,
           "Historical allocated fixture count/mandatory CUDA coverage changed")


def _terminal(source_root, job, archive_sha, manifest_sha):
    matches = [_read(p) for p in (source_root / "allocations").glob("*/run.json")
               if _read(p).get("environment", {}).get("job_id") == job]
    _check(len(matches) == 1 and matches[0].get("status") == "completed"
           and matches[0].get("returncode") == 0
           and matches[0].get("source_archive_sha256") == archive_sha
           and matches[0].get("source", {}).get("sha256") == manifest_sha,
           "Adopted allocation lacks its unique completed source-bound wrapper")
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                           capture_output=True, text=True, timeout=15)
    _check((reply.returncode == 0 and reply.stdout.strip() == "DONE" and not reply.stderr.strip())
           or (not reply.stdout.strip() and reply.stderr.strip() == f"Job <{job}> is not found"),
           "Source135 allocation is live, failed or scheduler state is unknown")


def _build(plan):
    """Derive additive receipts only after full authentication of the old proof."""
    if "epoch45_extension" in plan:
        from .chemistry_epoch45_extension import build_adoption
        return build_adoption(plan)
    from .chemistry_valid_reference_training_policy import validate_valid_reference_continuation
    marker = validate_valid_reference_continuation(plan)
    _check(marker is not None, "Only explicit validated-reference training can adopt this proof")
    declaration = plan["valid_reference_adoption"]
    work = Path(plan["work_root"]).resolve(strict=True)
    source_path = Path(declaration["source_plan"]).resolve(strict=True)
    _check(source_path.is_relative_to(work / "submissions") and _sha(source_path) == SOURCE_PLAN_SHA,
           "Original source135 plan bytes changed")
    for name, sha in declaration["artifacts"].items():
        path = Path(name).resolve(strict=True)
        _check(path.is_relative_to(work) and _sha(path) == sha, "Adopted evidence changed: " + name)
    _check(declaration["artifacts"].get(str(source_path)) == SOURCE_PLAN_SHA,
           "Original source135 plan must be explicitly bound")
    source = _read(source_path)
    source["_plan_sha256"] = SOURCE_PLAN_SHA
    _check(source["archive_sha256"] == SOURCE_ARCHIVE_SHA and source["manifest_sha256"] == SOURCE_MANIFEST_SHA,
           "Adoption is restricted to the completed source135 proof")
    excluded = {"name", "release", "campaign", "archive_sha256", "manifest_sha256", "short_continuation",
                "valid_reference_test", "valid_reference_probe", "valid_reference_continuation", "valid_reference_adoption"}
    scientific = lambda value: {k: v for k, v in value.items() if k not in excluded and not k.startswith("_")}
    _check(scientific(source) == scientific(plan), "Adoption changed scientific protocol/lineage/checks")
    root = Path(plan["campaign"]).resolve()
    source_root = Path(source["campaign"]).resolve(strict=True)
    _check(root != source_root and root.is_relative_to(work / "runs"), "Require a fresh training campaign")
    for name in ("cpu_gate.json", "guarded-cpu-pytest.xml", "independent-self-cuda-pytest.xml",
                 "benchmark_gate.json", "failure_readiness.json", "early-health-baseline.json",
                 "stability_gate.json", "valid_reference_probe.json", "training_composition.json", "repair_profile.json"):
        _check(str(source_root / name) in declaration["artifacts"], "Unbound mandatory historical evidence: " + name)
    _check(_sha(source_root / "stability_gate.json") == SOURCE_STABILITY_SHA,
           "Only the passed complete 2,500-update proof may be adopted")
    old_files = _manifest(source["release"], SOURCE_ARCHIVE_SHA, SOURCE_MANIFEST_SHA)
    new_files = _manifest(plan["release"], plan["archive_sha256"], plan["manifest_sha256"])
    changed = compare_source(source["release"], plan["release"], old_files, new_files)
    cpu = _read(source_root / "cpu_gate.json")
    _check(cpu.get("status") == "passed" and cpu.get("returncode") == 0
           and cpu.get("job_id") == "29783433" and cpu.get("profile_returncode") == 0
           and cpu.get("junit_sha256") == _sha(source_root / "guarded-cpu-pytest.xml")
           and cpu.get("profile_sha256") == _sha(source_root / "repair_profile.json"), "Original allocated CPU gate changed")
    _junit(source_root / "guarded-cpu-pytest.xml", 1917, 113)
    _junit(source_root / "independent-self-cuda-pytest.xml", 518, 0)
    for job in ("29783433", "29783863"):
        _terminal(source_root, job, SOURCE_ARCHIVE_SHA, SOURCE_MANIFEST_SHA)
    from .failure_guidance_selection import authenticate_readiness
    from .chemistry_restart_policy import authenticate_stability
    from .chemistry_valid_reference_validation import require_probe_completion
    from .chemistry_early_health import require_early_health_baseline, _baseline_context
    readiness = authenticate_readiness(source, _sha(source_root / "failure_readiness.json"))
    stability = authenticate_stability(source)
    require_probe_completion(source, _read(source_root / "valid_reference_probe.json"))
    original_baseline = require_early_health_baseline(source, readiness["early_health_baseline_sha256"])
    _check(stability["completed_updates"] == 2500 and stability["validation_weights_discarded"] is True,
           "Partial validation or disposable weights cannot become training")
    # Check actual original files at launch: model/AdamW/native/fresh RNG are
    # restored by the existing trainer, never from the probe/stability states.
    for key, digest in (("path", "sha256"), ("source_plan", "source_plan_sha256"), ("completion", "completion_sha256")):
        _check(_sha(plan["starting_checkpoint"][key]) == plan["starting_checkpoint"][digest], "Epoch36 ancestry changed")
    _check(_sha(plan["parent_checkpoint"]) == plan["parent_checkpoint_sha256"], "Reference35 changed")
    provenance = dict(version=VERSION, source_plan_sha256=SOURCE_PLAN_SHA,
        source_archive_sha256=SOURCE_ARCHIVE_SHA, source_manifest_sha256=SOURCE_MANIFEST_SHA,
        source_stability_sha256=SOURCE_STABILITY_SHA, source_campaign=str(source_root),
        source_observations_reused=True, new_native_observation=False,
        validation_weights_discarded=True, original36_restored_before_training=True)
    baseline = _baseline_context(plan)
    baseline.update({key: deepcopy(original_baseline[key]) for key in
        ("status", "completed_epoch", "updates", "model_sha256", "job_id", "created_at", "observation")})
    baseline["evidence_adoption"] = provenance
    baseline_sha = hashlib.sha256(_encode(baseline)).hexdigest()
    benchmark = deepcopy(_read(source_root / "benchmark_gate.json"))
    benchmark.update(plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], evidence_adoption=provenance)
    benchmark_sha = hashlib.sha256(_encode(benchmark)).hexdigest()
    adopted_stability = dict(status="passed", version=VERSION, completed_updates=2500,
        plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], gates=deepcopy(stability["gates"]),
        validation_weights_discarded=True, conservative_epoch_seconds=stability["conservative_epoch_seconds"],
        valid_reference_continuation=marker, evidence_adoption=provenance)
    stability_sha = hashlib.sha256(_encode(adopted_stability)).hexdigest()
    adopted_readiness = deepcopy(readiness)
    adopted_readiness.update(plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], native_benchmark_sha256=benchmark_sha,
        early_health_baseline_sha256=baseline_sha, stability_gate_sha256=stability_sha,
        valid_reference_continuation=marker, evidence_adoption=provenance)
    adopted_readiness.pop("valid_reference_test", None)
    adoption = dict(status="passed_evidence_adoption_not_new_native_test", version=VERSION,
        plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], evidence_adoption=provenance,
        metadata_source_overlays=changed, valid_reference_continuation=marker,
        passed_source_native_gates=18, passed_source_supplementary_gates=15,
        passed_source_cuda_fixtures=518, completed_source_stability_updates=2500,
        unchanged_scientific_protocol=True)
    return {"adoption.json": adoption, "early-health-baseline.json": baseline,
        "benchmark_gate.json": benchmark, "stability_gate.json": adopted_stability,
        "failure_readiness.json": adopted_readiness,
        "training_composition.json": _read(source_root / "training_composition.json")}, stability


def materialize_adoption(plan):
    """Allocated contracts create only additive deterministic launch receipts."""
    import os
    _check(os.environ.get("LSB_JOBID", "").isdigit(), "Adoption requires allocated CPU execution")
    artifacts, _ = _build(plan)
    root = Path(plan["campaign"])
    for name, value in artifacts.items():
        path = root / name
        data = _encode(value)
        if path.exists():
            _check(path.read_bytes() == data, "Existing adoption evidence differs: " + name)
        else:
            with path.open("xb") as stream:
                stream.write(data)
    return artifacts["adoption.json"]


def authenticate_adoption(plan, expected_sha=None, *, stability=False):
    artifacts, original_stability = _build(plan)
    root = Path(plan["campaign"])
    for name, value in artifacts.items():
        _check((root / name).read_bytes() == _encode(value), "Derived adoption evidence changed: " + name)
    if expected_sha is not None:
        _check(_sha(root / "failure_readiness.json") == expected_sha, "Adopted readiness SHA changed")
    if plan.get("_benchmark_gate_sha256") is not None:
        _check(_sha(root / "benchmark_gate.json") == plan["_benchmark_gate_sha256"], "Adopted benchmark SHA changed")
    return original_stability if stability else artifacts["failure_readiness.json"]
