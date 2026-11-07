"""Explicit epoch40->45 handoff; molecular objective and old evidence stay frozen.

Metadata authentication is safe on a login node. Tensor loading/restoration is
performed only by the allocated trainer, through its existing exact-resume
checks. The original35 occupancy anchor and original36 valid-target anchor are
never replaced by the epoch40 training checkpoint.
"""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path


VERSION = "epoch40-to45-valid-reference-extension-v1"
SOURCE_ID = "source138-vae-cli-startup-fix-attempt001"
WORK = "/work3/s234818/latentmol"
SOURCE_ROOT = WORK + "/runs/" + SOURCE_ID
SOURCE_PLAN = WORK + "/submissions/" + SOURCE_ID + "/" + SOURCE_ID + ".json"
SOURCE_PLAN_SHA = "183abba4bcd8e185b407ccdb5596a8fd6c37c0f9a82aeaf7c21e28a9cdb219c4"
SOURCE_ARCHIVE_SHA = "3dd998a9953aa4d61dddc38af92b7779221308edd0a5a75140a4475d1273259a"
SOURCE_MANIFEST_SHA = "c702a81e665ad8b77b19b37ba5d3c19ee1595f5877e3cc926a6811024a279a84"
CHECKPOINT = SOURCE_ROOT + "/segments/004/20261026T230230Z_chemistry-train_e9f28b40/epoch_0040.pt"
CHECKPOINT_SHA = "31af05597dbd176b02a8108651db0d6b33bb3a8883f69ecf354a4465ed18787e"
COMPLETION_SHA = "3957b9bcd99e686ee878e2c686184c49bddbeff2167c5858ebb880e215d1f4fa"
BRANCH_SHA = "5dd518c3c8d1c21125521886ed43397771cc8a78b53a0c249a54f4b2dd4a6866"
POLICY = dict(version=VERSION, start_epoch=40, end_epoch=45,
    checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA, updates=263840,
    source_plan=SOURCE_PLAN, source_plan_sha256=SOURCE_PLAN_SHA,
    completion=SOURCE_ROOT + "/segment-004-completion.json", completion_sha256=COMPLETION_SHA,
    branch=SOURCE_ROOT + "/bounded-branch-completion.json", branch_sha256=BRANCH_SHA,
    save_every_epoch=True,
    generation=dict(version="epoch-checkpoint-generation-v1", seeds=[140041, 140042],
        prior_count=1024, categorical_samples_per_latent=4, evaluation_batch_size=32))
PLAN_OPERATIONS = {"name", "release", "campaign", "archive_sha256", "manifest_sha256", "epoch45_extension"}
METADATA_FUNCTIONS = {
    "src/latentmol/chemistry_short_continuation.py": {"validate_short_continuation"},
    "src/latentmol/chemistry_valid_reference_adoption.py": {"_build"},
    "src/latentmol/chemistry_continuation.py": {"setup", "phase_checkpoint", "train_phase"},
    "scripts/hpc/chemistry_campaign.py": {"advance"},
    "scripts/hpc/chemistry_valid_reference_campaign.py": {"check_launch"},
}
NEW_FILES = {"src/latentmol/chemistry_epoch45_extension.py", "src/latentmol/chemistry_epoch45_generation.py",
    "src/latentmol/cuda_process.py", "tests/test_cuda_process.py",
    "scripts/hpc/chemistry_epoch45_campaign.py", "scripts/hpc/chemistry_epoch45_worker.py",
    "tests/test_chemistry_epoch45_extension.py", "tests/test_chemistry_epoch45_generation.py"}


def _exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def validate_extension(plan):
    """A new explicit authorization; never reinterpret the old through40 marker."""
    if "epoch45_extension" not in plan:
        return None
    if not _exact(plan["epoch45_extension"], POLICY):
        raise ValueError("Require the exact completed epoch40 through45 extension and observation policy")
    from .chemistry_valid_reference_training_policy import POLICY as ORIGINAL_POLICY
    if not _exact(plan.get("valid_reference_continuation"), ORIGINAL_POLICY):
        raise ValueError("The original36->40 ancestry marker must remain intact")
    return dict(version=VERSION, start_epoch=40, end_epoch=45, purpose="validated_reference_training")


def compare_sources(old_release, new_release, old_files, new_files):
    """Keep every loss/update/model/configuration byte; audit metadata boundaries."""
    changed = {p for p in old_files.keys() | new_files.keys() if old_files.get(p) != new_files.get(p)}
    if old_files.keys() - new_files.keys() or not changed <= METADATA_FUNCTIONS.keys() | NEW_FILES:
        raise ValueError("Extension changed an undeclared executable or removed historical code")
    for name in changed & old_files.keys():
        allowed = METADATA_FUNCTIONS.get(name)
        if allowed is None:
            raise ValueError("New extension file collides with a historical member")
        old = ast.parse((Path(old_release) / name).read_bytes())
        new = ast.parse((Path(new_release) / name).read_bytes())
        def parts(tree):
            functions = {n.name: ast.dump(n) for n in ast.walk(tree)
                         if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            constants = [ast.dump(n) for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))]
            return functions, constants
        before, old_constants = parts(old)
        after, new_constants = parts(new)
        if old_constants != new_constants or set(before) != set(after):
            raise ValueError("Extension changed module constants or function coverage: " + name)
        if any(after[key] != value for key, value in before.items() if key not in allowed):
            raise ValueError("Extension changed a numerical/nonmetadata function: " + name)
    return sorted(changed)


def authenticate_source(plan):
    """Authenticate completed source138 and all earlier scientific gates unchanged."""
    if validate_extension(plan) is None:
        raise ValueError("Missing explicit epoch45 extension")
    from .chemistry_valid_reference_adoption import _read, _sha, _manifest, _build, _terminal
    from .chemistry_early_health import require_early_health_baseline
    if _sha(SOURCE_PLAN) != SOURCE_PLAN_SHA or _sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Authenticated source138 plan/checkpoint changed")
    old = _read(SOURCE_PLAN)
    if old["archive_sha256"] != SOURCE_ARCHIVE_SHA or old["manifest_sha256"] != SOURCE_MANIFEST_SHA:
        raise ValueError("Source138 release identity differs")
    scientific = lambda p: {k: v for k, v in p.items() if k not in PLAN_OPERATIONS and not k.startswith("_")}
    if not _exact(scientific(old), scientific(plan)) or old["campaign"] == plan["campaign"]:
        raise ValueError("Extension changed loss, sampling, reference, optimizer, data or safeguards")
    old["_plan_sha256"] = SOURCE_PLAN_SHA
    expected, stability = _build(old)  # No extension marker: exact historical adoption verifier.
    root = Path(SOURCE_ROOT)
    for name, value in expected.items():
        from .chemistry_valid_reference_adoption import _encode
        if (root / name).read_bytes() != _encode(value):
            raise ValueError("Source138 scientific adoption receipt changed: " + name)
    old["_benchmark_gate_sha256"] = _sha(root / "benchmark_gate.json")
    old["_failure_readiness_sha256"] = _sha(root / "failure_readiness.json")
    old["_early_health_baseline_sha256"] = expected["failure_readiness.json"]["early_health_baseline_sha256"]
    require_early_health_baseline(old, old["_early_health_baseline_sha256"])
    for path, sha in ((POLICY["completion"], COMPLETION_SHA), (POLICY["branch"], BRANCH_SHA)):
        if _sha(path) != sha:
            raise ValueError("Completed epoch40 terminal receipt changed")
    completion, branch = _read(POLICY["completion"]), _read(POLICY["branch"])
    if (completion.get("status") != "passed" or completion.get("completed_epoch") != 40
            or completion.get("epoch_complete") is not True or completion.get("updates") != 263840
            or completion.get("checkpoint_sha256") != CHECKPOINT_SHA or completion.get("checkpoint") != CHECKPOINT
            or completion.get("plan_sha256") != SOURCE_PLAN_SHA or completion.get("stop_reason") != "bounded_horizon"
            or branch.get("status") != "completed_bounded_branch" or branch.get("completed_epoch") != 40
            or branch.get("checkpoint_sha256") != CHECKPOINT_SHA or branch.get("completion_sha256") != COMPLETION_SHA):
        raise ValueError("Epoch40 source is not an authenticated completed training branch")
    row = _read(Path(CHECKPOINT).parent / "latest_epoch.json")
    if (row.get("checkpoint_sha256") != CHECKPOINT_SHA or row.get("epoch") != 40
            or not row.get("health") or not all(type(v) is bool and v for v in row["health"].values())):
        raise ValueError("Epoch40 end-epoch health failed")
    health = sorted(root.glob("segments/*/*/early_health/update_*.json"))
    if len(health) != 268:
        raise ValueError("Source138 early-health history is incomplete or changed")
    for path in health:
        panel = _read(path)
        if panel.get("status") != "passed" or not panel.get("checks") or not all(
                type(v) is bool and v for v in panel["checks"].values()):
            raise ValueError("Source138 has a failed early-health panel")
    _terminal(root, "29822674", SOURCE_ARCHIVE_SHA, SOURCE_MANIFEST_SHA)
    cuda = _read(root / "training-launch-cuda-gate.json")
    from .chemistry_valid_reference_adoption import _junit
    _junit(root / "training-launch-cuda-pytest.xml", 599, 0)
    if cuda.get("status") != "passed" or cuda.get("junit_sha256") != _sha(root / "training-launch-cuda-pytest.xml"):
        raise ValueError("Source138 mandatory CUDA launch evidence changed")
    old_files = _manifest(old["release"], SOURCE_ARCHIVE_SHA, SOURCE_MANIFEST_SHA)
    new_files = _manifest(plan["release"], plan["archive_sha256"], plan["manifest_sha256"])
    changed = compare_sources(old["release"], plan["release"], old_files, new_files)
    return old, expected, stability, changed


def build_adoption(plan):
    """Derive new receipts; old source135/source138 observations stay immutable."""
    from .chemistry_valid_reference_adoption import _read, _encode
    from .chemistry_early_health import _baseline_context
    old, source, stability, changed = authenticate_source(plan)
    provenance = dict(version=VERSION, source_plan_sha256=SOURCE_PLAN_SHA,
        epoch40_checkpoint_sha256=CHECKPOINT_SHA, source_branch_sha256=BRANCH_SHA,
        source_observations_reused=True, new_native_observation=False,
        original36_reference_preserved=True, original35_occupancy_preserved=True)
    result = deepcopy(source)
    baseline = _baseline_context(plan)
    baseline.update({k: deepcopy(source["early-health-baseline.json"][k]) for k in
        ("status", "completed_epoch", "updates", "model_sha256", "job_id", "created_at", "observation")})
    baseline["extension_evidence"] = provenance
    result["early-health-baseline.json"] = baseline
    sha = lambda value: hashlib.sha256(_encode(value)).hexdigest()
    for name in ("benchmark_gate.json", "stability_gate.json", "failure_readiness.json", "adoption.json"):
        result[name].update(plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
            manifest_sha256=plan["manifest_sha256"], epoch45_extension=deepcopy(POLICY), extension_evidence=provenance)
    result["failure_readiness.json"].update(early_health_baseline_sha256=sha(baseline),
        native_benchmark_sha256=sha(result["benchmark_gate.json"]),
        stability_gate_sha256=sha(result["stability_gate.json"]))
    result["adoption.json"]["extension_metadata_overlays"] = changed
    return result, stability


def initial_checkpoint(plan):
    """Operational training start; frozen references remain original36/original35."""
    if validate_extension(plan) is None:
        return plan.get("starting_checkpoint")
    return dict(epoch=40, updates=263840, path=CHECKPOINT, sha256=CHECKPOINT_SHA)


def resume_identity(plan, path, sha, algorithm):
    """First handoff uses old identities; subsequent resumes use exact new ones."""
    if validate_extension(plan) is None or str(path) != CHECKPOINT:
        return plan, algorithm
    if sha != CHECKPOINT_SHA:
        raise ValueError("Epoch40 handoff checksum differs")
    old, _, _, _ = authenticate_source(plan)
    # The algorithm hashes every package module; new metadata files legitimately
    # change the new identity. Check the old payload against old source bytes.
    files = {p.relative_to(Path(old["release"]) / "src/latentmol").as_posix():
             hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((Path(old["release"]) / "src/latentmol").rglob("*.py"))}
    return old, dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
