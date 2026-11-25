"""Metadata-only adoption of source126's runtime-only failed bond6 benchmark.

This is an explicit scheduling-policy amendment, never a new GPU benchmark or
a rewrite of the failed source evidence. All molecular code/configuration is
unchanged. Both the scheduler and trainer authenticate the same additive proof.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from .chemistry_runtime import epoch_fit_policy, epoch_fit_seconds
from .chemistry_speed_selection import REQUIRED_GATES, REQUIRED_PARITY_CHECKS


VERSION = "chemistry-runtime-adoption-v1"
KIND = "runtime_policy_adoption_of_saved_native"
SOURCE_PLAN_SHA256 = "81329b89cad5c6c40a7542c1d116e5e0d38fee8411cc39a70f3e8326bea48da6"
SOURCE_ARCHIVE_SHA256 = "53d5fb13f4ad615afb6f0e1601be52325ef8379596951cf1dee694d2f5114967"
SOURCE_MANIFEST_SHA256 = "7b7006dc1465f3167a8a30c828f69225011fb994ca1fee3d5ad8f7c0daedd0ec"
SOURCE_SUITE_SHA256 = "3a0de8659ff38f4f9f3e9bb4512bd74268fd4883709cbdb9fe9a92c8cdc8120c"
RUNTIME_ERROR = "RuntimeError: Chemistry benchmark failed explicit promotion gates"
RUNTIME_SOURCE_FILES = frozenset((
    "src/latentmol/chemistry_runtime.py", "src/latentmol/chemistry_continuation.py",
    "src/latentmol/failure_guidance_benchmark.py", "src/latentmol/failure_guidance_selection.py",
    "src/latentmol/chemistry_runtime_adoption.py", "scripts/hpc/chemistry_campaign.py",
    "scripts/hpc/chemistry_runtime_continuation.py", "tests/test_chemistry_epoch_fit_amendment.py",
    "tests/test_chemistry_runtime_adoption.py", "tests/test_chemistry_runtime_continuation.py"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate adoption evidence key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    def finite(part):
        if isinstance(part, float) and not math.isfinite(part):
            raise ValueError("Nonfinite adoption evidence")
        if isinstance(part, dict):
            for item in part.values(): finite(item)
        elif isinstance(part, list):
            for item in part: finite(item)
    finite(value)
    return value


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _manifest(release, archive_sha, manifest_sha):
    release = Path(release)
    _check(_sha(str(release) + ".tar.gz") == archive_sha, "Adoption source archive changed")
    value = _read(release / "release_manifest.json")
    files = value["files"]
    actual = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    _check(actual == manifest_sha == value["sha256"] and value["root"] == str(release),
           "Adoption source manifest changed")
    observed = {p.relative_to(release).as_posix() for folder in ("src", "tests", "configs", "scripts")
                for p in (release / folder).rglob("*") if p.is_file()
                and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    observed.update(name for name in ("pyproject.toml", "requirements-hpc.txt") if (release / name).is_file())
    _check(observed == set(files), "Adoption source manifest coverage changed")
    for name, sha in files.items():
        path = release / name
        _check(path.resolve().is_relative_to(release.resolve()) and _sha(path) == sha,
               "Adoption executable bytes changed: " + name)
    return files


def _no_successor(root):
    root = Path(root)
    patterns = ("advance-after-*.json", "receipts/segment-*.json", "receipts/after-segment-*.json",
                "receipts/final-*.json", "segment-*-completion.json", "segments/*")
    _check(not (root / "active-phase.json").exists()
           and not any(any(root.glob(pattern)) for pattern in patterns),
           "Source adoption is blocked by an active phase or successor")


def _terminal(job, log):
    _check(re.search(r"^Subject: Job " + re.escape(job) + r": .* Done$", log, re.M) is not None
           and "\nSuccessfully completed.\n" in log and "Exited with exit code" not in log,
           "Adoption predecessor lacks exact persisted LSF DONE proof")
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                           capture_output=True, text=True, timeout=15)
    _check((reply.returncode == 0 and reply.stdout.strip() == "DONE" and not reply.stderr.strip())
           or (not reply.stdout.strip() and reply.stderr.strip() == f"Job <{job}> is not found"),
           "Adoption predecessor is live or its scheduler state is unknown")


def _retention(before, after, old_rec, new_rec):
    """The unchanged nine non-runtime starting-checkpoint retention checks."""
    return dict(finite=all(math.isfinite(after[k]) for k in
                    ("expected_size", "element_js", "charge_js", "sulfur_frequency")),
        size_retention=.9 <= after["expected_size"] / max(before["expected_size"], 1e-8) <= 1.1,
        nonempty_retention=after["nonempty_fraction"] >= before["nonempty_fraction"] - .02,
        sulfur_no_inflation=after["sulfur_frequency"] <= before["sulfur_frequency"] * 1.5 + .005,
        composition_no_deterioration=after["element_js"] <= before["element_js"] + .02,
        charge_no_deterioration=after["charge_js"] <= before["charge_js"] + .02,
        reconstruction_retention=new_rec["exact_reconstruction"] >= old_rec["exact_reconstruction"] - .02,
        valid_reconstruction_retention=new_rec["validity"] >= old_rec["validity"] - .02,
        raw_prior_validity_retention=after["raw_argmax_validity"] >= before["raw_argmax_validity"] - .02)


def _build(plan):
    """Reauthenticate every original byte and derive deterministic new receipts."""
    value = plan.get("runtime_adoption")
    _check(type(value) is dict and set(value) == {"version", "source_plan", "source_plan_sha256", "artifacts"}
           and value["version"] == VERSION and value["source_plan_sha256"] == SOURCE_PLAN_SHA256,
           "Require the exact source126 runtime-adoption declaration")
    _check(re.fullmatch(r"[0-9a-f]{64}", plan.get("_plan_sha256", "")) is not None,
           "Adoption requires the new immutable plan SHA")
    policy = epoch_fit_policy(plan)
    _check(policy["factor"] == 1.25 and policy["reserve_seconds"] == 1800
           and plan.get("runtime_policy_amendment") is not None, "Require the authorized runtime amendment")
    work = Path(plan["work_root"]).resolve(strict=True)
    artifacts = value["artifacts"]
    _check(type(artifacts) is dict and artifacts, "Require complete bound adoption artifacts")
    for name, sha in artifacts.items():
        _check(Path(name).is_absolute() and Path(name).resolve().is_relative_to(work)
               and type(sha) is str and re.fullmatch(r"[0-9a-f]{64}", sha) is not None and _sha(name) == sha,
               "Runtime-adoption artifact changed: " + name)
    def bound(path, *, text=False):
        path = Path(path)
        _check(str(path) in artifacts, "Missing required adoption proof: " + str(path))
        return path.read_text() if text else _read(path)
    source_path = Path(value["source_plan"])
    _check(source_path.is_absolute() and source_path.resolve().is_relative_to(work)
           and _sha(source_path) == SOURCE_PLAN_SHA256, "Source126 plan changed")
    source = bound(source_path)
    _check(source["archive_sha256"] == SOURCE_ARCHIVE_SHA256
           and source["manifest_sha256"] == SOURCE_MANIFEST_SHA256
           and source.get("failure_suite_sha256") == SOURCE_SUITE_SHA256
           and source.get("failure_variant_id") == "bond6", "Adoption is restricted to source126 bond6")
    excluded = {"release", "campaign", "archive_sha256", "manifest_sha256", "runtime_policy_amendment", "runtime_adoption"}
    scientific = lambda p: {k: v for k, v in p.items() if k not in excluded and not k.startswith("_")}
    _check(scientific(source) == scientific(plan), "Runtime adoption changed scientific protocol or lineage")
    root, old_root = Path(plan["campaign"]).resolve(), Path(source["campaign"]).resolve()
    _check(root != old_root and root.is_relative_to(work / "runs") and old_root.is_relative_to(work / "runs"),
           "Adoption requires a new campaign inside the declared work root")
    _check(all(Path(p["release"]).resolve().is_relative_to(work / "releases") for p in (source, plan)),
           "Adoption source release escaped the declared work root")
    old_files = _manifest(source["release"], SOURCE_ARCHIVE_SHA256, SOURCE_MANIFEST_SHA256)
    new_files = _manifest(plan["release"], plan["archive_sha256"], plan["manifest_sha256"])
    changed = {p for p in old_files.keys() | new_files.keys() if old_files.get(p) != new_files.get(p)}
    _check(not old_files.keys() - new_files.keys() and changed and changed <= RUNTIME_SOURCE_FILES,
           "Runtime adoption changed molecular code/configuration outside its declared whitelist")
    suite_path = Path(source["failure_suite_path"])
    _check(_sha(suite_path) == SOURCE_SUITE_SHA256, "Source126 suite identity changed")
    suite = _read(suite_path)
    suite_root = Path(suite["campaign"]).resolve()
    _check(suite_root.is_relative_to(work / "runs") and old_root == suite_root / "variants/bond6"
           and suite["archive_sha256"] == SOURCE_ARCHIVE_SHA256
           and suite["manifest_sha256"] == SOURCE_MANIFEST_SHA256, "Source126 suite changed its source/campaign")
    for candidate in (suite_root, *(suite_root / "variants" / name for name in ("legacy3", "fallback", "bond6"))):
        _no_successor(candidate)
    binding = bound(suite_root / "starting-checkpoint.json")
    plans = bound(suite_root / "bound-plans.json")
    _check(binding == source["starting_checkpoint"] == plan["starting_checkpoint"]
           and binding["epoch"] == 36 and _sha(binding["path"]) == binding["sha256"]
           and plans["suite_sha256"] == SOURCE_SUITE_SHA256
           and plans["binding_sha256"] == artifacts[str(suite_root / "starting-checkpoint.json")]
           and tuple(row["id"] for row in plans["variants"]) == ("legacy3", "fallback", "bond6")
           and plans["variants"][2]["path"] == str(source_path)
           and plans["variants"][2]["sha256"] == SOURCE_PLAN_SHA256, "Source126 epoch36/shared-plan binding changed")
    for key, sha in (("source_plan", "source_plan_sha256"), ("completion", "completion_sha256")):
        _check(_sha(binding[key]) == binding[sha], "Original epoch36 ancestry bytes changed")
    ancestor_root = Path(_read(binding["source_plan"])["campaign"])
    _check(ancestor_root.resolve().is_relative_to(work / "runs")
           and not (ancestor_root / "active-phase.json").exists()
           and not any((ancestor_root / p).exists() for p in ("advance-after-segment-001-intent.json",
               "advance-after-segment-001.json", "receipts/segment-002-intent.json", "receipts/segment-002-receipt.json")),
           "Original epoch36 training source already has an active successor")
    n = bound(old_root / "benchmark_gate.json")
    failed = bound(old_root / "failure_readiness.json")
    _check(n["status"] == failed["status"] == "failed" and n.get("error") == failed.get("error") == RUNTIME_ERROR
           and n["plan_sha256"] == failed["plan_sha256"] == SOURCE_PLAN_SHA256
           and set(n["gates"]) == REQUIRED_GATES and n["gates"]["complete_epoch_fits"] is False
           and all(v is True for k, v in n["gates"].items() if k != "complete_epoch_fits")
           and not failed.get("gates") and failed.get("conservative_epoch_seconds") is None,
           "Only the recorded original runtime-only failure may be adopted")
    panel = failed["native_panel"]
    _check(panel["status"] == "observed" and panel["completed_epoch"] == 36
           and panel["plan_sha256"] == SOURCE_PLAN_SHA256
           and panel["archive_sha256"] == failed["archive_sha256"] == SOURCE_ARCHIVE_SHA256
           and panel["manifest_sha256"] == failed["manifest_sha256"] == SOURCE_MANIFEST_SHA256
           and panel["panel_count"] == 1024 and panel["panel_seed"] == 123904017,
           "Saved native panel changed its source/epoch/cohort")
    identity = dict(plan_sha256=SOURCE_SUITE_SHA256, archive_sha256=SOURCE_ARCHIVE_SHA256,
                    manifest_sha256=SOURCE_MANIFEST_SHA256, parent_sha256=source["parent_checkpoint_sha256"])
    def job(stage):
        receipt = bound(suite_root / "receipts" / (stage + "-receipt.json"))
        intent = bound(suite_root / "receipts" / (stage + "-intent.json"))
        _check(receipt["intent"] == intent and intent["identity"] == identity
               and re.fullmatch(r"[0-9]+", receipt["job_id"]) is not None, "Source job receipt identity changed")
        command = intent["command"]
        _check(SOURCE_SUITE_SHA256 in " ".join(command), "Source command changed its suite binding")
        log = Path(command[command.index("-oo") + 1].replace("%J", receipt["job_id"]))
        _check(log.resolve().is_relative_to(work / "logs"), "Source LSF log escaped the work root")
        _terminal(receipt["job_id"], bound(log, text=True))
        return receipt["job_id"]
    native_job, callback_job, cpu_job = job("native"), job("after-native"), job("cpu")
    _check(panel["job_id"] == failed["job_id"] == native_job, "Saved panel/native allocation changed")
    selection, promotion = bound(suite_root / "selection.json"), bound(suite_root / "promotion.json")
    _check(selection["suite_sha256"] == SOURCE_SUITE_SHA256 and selection["job_id"] == native_job
           and selection["winner"] is None and selection["reason"] == "no_passing_new_objective"
           and selection["bound_plans_sha256"] == artifacts[str(suite_root / "bound-plans.json")]
           and promotion["status"] == "stopped_no_passing_new_objective" and promotion["selection"] == selection,
           "Source126 was not the preserved stopped comparison")
    control = selection["rows"][0]
    _check(control["id"] == "legacy3" and control["eligible"] is True
           and all(v is True for v in control["gates"].values())
           and control["readiness"]["native_panel"]["source_snapshot_chain_sha256"] == panel["source_snapshot_chain_sha256"],
           "Matched native control/shared panel did not pass")
    for stage, expected_job in (("cpu", cpu_job), ("native", native_job)):
        matches = [bound(p) for p in (suite_root / "allocations").glob("*/run.json")
                   if str(p) in artifacts and bound(p).get("environment", {}).get("job_id") == expected_job]
        _check(len(matches) == 1 and matches[0]["status"] == "completed" and matches[0]["returncode"] == 0
               and matches[0]["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
               and matches[0]["source"]["sha256"] == SOURCE_MANIFEST_SHA256,
               "Source allocation lacks its unique completed wrapper")
    cpu = bound(suite_root / "cpu_gate.json")
    _check(cpu["status"] == "passed" and cpu["returncode"] == 0 and cpu["identity"] == identity
           and cpu["job_id"] == cpu_job, "Source CPU gate did not pass")
    for name, key in (("repair_profile.json", "profile_sha256"),
                      ("failure_preparation_profile.json", "execution_profile_sha256")):
        p = bound(suite_root / name)
        _check(cpu[key] == artifacts[str(suite_root / name)] and p["status"] == "passed"
               and p["identity"] == identity and p["job_id"] == cpu_job, "Source CPU profile changed")
        if name == "failure_preparation_profile.json":
            _check(p["checkpoint_sha256"] == binding["sha256"] and p["completed_epoch"] == 36
                   and p["scientific_receipts_exact"] is True and p["points"] == 256
                   and [(row["start"], row["count"]) for row in p["rows"]] == [(0, 128), (128, 128)]
                   and all(row["exact_preparation_parity"] is True for row in p["rows"]),
                   "Source preparation profile changed its epoch36/exact-parity binding")
    for gate_name, xml_name, count, skipped in (("cpu_gate.json", "cpu-pytest.xml", 1126, 65),
            ("deterministic-events-gate.json", "deterministic-events-pytest.xml", 104, 0),
            ("loss-validation-gate.json", "loss-validation-pytest.xml", 22, 0)):
        gate = bound(suite_root / gate_name)
        _check(gate["returncode"] == 0 and gate["job_id"] == (cpu_job if gate_name == "cpu_gate.json" else native_job),
               "Source mandatory fixture allocation did not pass")
        if gate_name != "cpu_gate.json":
            _check(gate["suite_sha256"] == SOURCE_SUITE_SHA256, "Source mandatory CUDA fixture changed its suite")
        suites = list(ET.fromstring(bound(suite_root / xml_name, text=True)).iter("testsuite"))
        _check(suites and sum(int(r.get("tests", "0")) for r in suites) == count
               and sum(int(r.get("skipped", "0")) for r in suites) == skipped
               and all(int(r.get(k, "0")) == 0 for r in suites for k in ("errors", "failures")),
               "Source mandatory fixture coverage/failure count changed")
        if gate_name == "loss-validation-gate.json":
            _check(gate["junit_sha256"] == artifacts[str(suite_root / xml_name)]
                   and gate["identity"] == identity and gate["environment"]["LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES"] == "1"
                   and gate["reference_module_sha256"] == "34c1a67cb339233312a1317db39a926b55f378c0eb8ed42c38bde86307fdd184"
                   and gate["command"][-1] == "tests/test_failure_loss_validation_reuse.py",
                   "Source loss parity lacks mandatory CUDA/source binding")
    cuda = bound(suite_root / "cuda-fixtures.json")
    _check(cuda["returncode"] == 0 and cuda["job_id"] == native_job and cuda["suite_sha256"] == SOURCE_SUITE_SHA256,
           "Source native CUDA resume fixtures failed")
    run = Path(n["run_dir"])
    _check(run.resolve().is_relative_to(old_root / "benchmark"), "Source benchmark run escaped its campaign")
    record, run_gate = bound(run / "run.json"), bound(run / "gate.json")
    # The original catch adds the error and refreshes wall time AFTER writing
    # run/gate.json. Scientific fields must still be identical, byte-bound in
    # both files; no broader normalization or status conversion is permitted.
    comparable = lambda value: {k: v for k, v in value.items() if k not in ("error", "elapsed_seconds")}
    _check(comparable(run_gate) == comparable(n) and "error" not in run_gate
           and 0 < run_gate["elapsed_seconds"] <= n["elapsed_seconds"] and record["status"] == "failed"
           and record["error"] == RUNTIME_ERROR.removeprefix("RuntimeError: ")
           and record["gate_status"] == "failed" and record["gates"] == n["gates"]
           and record["environment"]["job_id"] == native_job
           and record["source_archive_sha256"] == SOURCE_ARCHIVE_SHA256
           and record["source"]["sha256"] == SOURCE_MANIFEST_SHA256
           and record["config"]["_plan_sha256"] == SOURCE_PLAN_SHA256,
           "Source runtime failure lacks its exact benchmark run/gate identity")
    proof = n["native_execution_parity"]
    _check(proof["status"] == "passed" and proof["passed"] is True and proof["restoration"]["passed"] is True
           and set(proof["checks"]) == REQUIRED_PARITY_CHECKS and all(v is True for v in proof["checks"].values())
           and n["native_resume_probe"]["status"] == "passed"
           and n["native_resume_probe"]["checks"] == dict(model=True, optimizer=True, rng=True)
           and n["parent_checkpoint_authentication"]["passed"] is True
           and n["parent_sha256"] == source["parent_checkpoint_sha256"]
           and n["parent_checkpoint_authentication"]["parent_checkpoint_sha256"] == source["parent_checkpoint_sha256"]
           and n["parent_checkpoint_authentication"]["objective_change_handoff"]["starting_checkpoint"] == binding,
           "Source native parity/resume/AdamW/RNG lineage did not pass")
    updates = n["updates"]
    _check(len(updates) == source["protocol"]["benchmark_updates"] == 8
           and source["protocol"]["real_batch_size"] == 256 and source["protocol"]["prior_ratio"] == 2,
           "Source timed update/ratio coverage changed")
    for row in updates:
        _check(type(row["update_seconds"]) in (int, float) and row["update_seconds"] > 0
               and row["real_points"] == 256 and row["fresh_prior_points"] == 512,
               "Source full-update timing or sample count changed")
    mean = source["training_molecules"] * sum(r["update_seconds"] for r in updates) / (8 * 256) + 600.
    _check(mean == n["projected_epoch_seconds"] and 1.5 * mean + 1800 >= 82800,
           "Source failed mean runtime projection changed")
    conservative = max(mean, max(r["update_seconds"] for r in updates) * source["training_molecules"] / 256 + 600.)
    _check(epoch_fit_seconds(plan, conservative) < 82800, "Amended conservative epoch still does not fit")
    counts = panel["counts"]
    _check(counts["attempts"] == 1024 and counts["already_valid"] == 643 and counts["original_accepted"] == 359
           and counts["original_censored"] == counts["original_failed"] == 22 and counts["unsupported"] == 0
           and counts["extra_valid"] == 12 and counts["partial"] == 10
           and counts["failed_with_nonzero_gradient"] == 22 and counts["failed_without_nonzero_gradient"] == 0,
           "Source failed-row coverage changed")
    gradients = panel["failed_row_gradient_norms"]
    checks = _retention(panel["starting_population"], n["updated_population"],
                        panel["starting_reconstruction"], n["updated_reconstruction"])
    checks.update(failed_rows_observed=counts["original_failed"] > 0,
        failed_gradient_coverage=counts["failed_with_nonzero_gradient"] >= .9 * counts["original_failed"]
            and len(gradients) == counts["original_failed"]
            and sum(v > 1e-10 for v in gradients) == counts["failed_with_nonzero_gradient"],
        zero_guidance_reduced=counts["failed_with_nonzero_gradient"] > 0,
        native_extra_decoder_vjp=any(v > 1e-10 for v in panel["additional_decoder_gradient_norms"]),
        frozen_target_directional_derivative=panel["native_directional_check"]["passed"] is True
            and panel["native_directional_check"]["dtype"] == "float64"
            and panel["native_directional_check"]["rtol"] == 1e-4
            and panel["native_directional_check"]["atol"] == 1e-8
            and abs(panel["native_directional_check"]["analytic"]) > 1e-12
            and math.isclose(panel["native_directional_check"]["analytic"],
                             panel["native_directional_check"]["finite_difference"], rel_tol=1e-4, abs_tol=1e-8),
        no_atom_edit_shortcut=panel["complete_target_atom_edits"] == 0
            and source["protocol"]["failure_guidance"]["progress_search"]["allow_atom_edits"] is False,
        production_extra_gradient=any((r.get("failure_only_decoder_gradient_norm") or 0) > 1e-10 for r in updates),
        source_native_nonruntime_gates=True, amended_conservative_complete_epoch_fits=True)
    _check(all(v is True for v in checks.values()), "A non-runtime adoption gate failed")
    composition_path = old_root / "training_composition.json"
    composition = bound(composition_path)
    _check(artifacts[str(composition_path)] == n["composition_sha256"]
           and composition["receipt"]["dataset_sha256"] == plan["dataset_sha256"]
           and composition["receipt"]["split_sha256"] == plan["split_sha256"]
           and composition["receipt"]["split"] == "train"
           and composition["receipt"]["molecules"] == plan["training_molecules"], "Source training composition changed")
    source_proof = dict(source_plan=str(source_path), source_plan_sha256=SOURCE_PLAN_SHA256,
        archive_sha256=SOURCE_ARCHIVE_SHA256, manifest_sha256=SOURCE_MANIFEST_SHA256,
        source_suite_sha256=SOURCE_SUITE_SHA256, source_native_job=native_job, source_callback_job=callback_job,
        source_native_gate_sha256=artifacts[str(old_root / "benchmark_gate.json")],
        source_run_gate_sha256=artifacts[str(run / "gate.json")],
        source_readiness_sha256=artifacts[str(old_root / "failure_readiness.json")],
        original_status="failed", original_failed_gates=["complete_epoch_fits"], artifacts=artifacts)
    gate = dict(deepcopy(n), status="passed", evidence_kind=KIND, plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        gates=dict(n["gates"], complete_epoch_fits=True), source_evidence=source_proof,
        projected_epoch_seconds=conservative, original_projected_epoch_seconds=mean,
        runtime_policy_amendment=plan["runtime_policy_amendment"], benchmark_weights_discarded=True)
    gate.pop("error", None)
    receipt = dict(version=VERSION, status="passed", evidence_kind=KIND, plan_sha256=plan["_plan_sha256"],
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"], gates=checks,
        native_panel=panel, source_evidence=source_proof, benchmark_weights_discarded=True,
        native_benchmark_sha256=hashlib.sha256(_encode(gate)).hexdigest(),
        conservative_epoch_seconds=conservative, original_projected_epoch_seconds=mean,
        amended_required_seconds=epoch_fit_seconds(plan, conservative),
        runtime_policy_amendment=plan["runtime_policy_amendment"],
        scientific_gate_schema="runtime-adoption-keeps-all-nonruntime-gates",
        scope="Saved source126 native science; newly authorized runtime policy; no new GPU benchmark")
    return receipt, gate, composition_path


def _exclusive(path, content):
    with Path(path).open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def materialize_adoption(plan):
    """Write additive new-campaign evidence once; uncertain partial writes stay put."""
    receipt, gate, composition = _build(plan)
    root = Path(plan["campaign"])
    root.mkdir(parents=True, exist_ok=True)
    names = ("runtime-adoption-intent.json", "benchmark_gate.json", "failure_readiness.json", "training_composition.json")
    _check(not any((root / name).exists() for name in names) and not (root / "active-phase.json").exists(),
           "Preserve an existing/uncertain adoption; do not materialize twice")
    _exclusive(root / names[0], _encode(dict(version=VERSION, plan_sha256=plan["_plan_sha256"], evidence_kind=KIND)))
    _exclusive(root / "training_composition.json", composition.read_bytes())
    _exclusive(root / "benchmark_gate.json", _encode(gate))
    _exclusive(root / "failure_readiness.json", _encode(receipt))
    return receipt


def authenticate_adoption(plan, expected_sha):
    """Recompute the proof from authenticated originals on every public entry."""
    path = Path(plan["campaign"]) / "failure_readiness.json"
    _check(type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha) is not None
           and _sha(path) == expected_sha, "Runtime-adoption readiness SHA changed")
    expected, gate, composition = _build(plan)
    actual = _read(path)
    _check(actual == expected and _read(path.parent / "benchmark_gate.json") == gate
           and _sha(path.parent / "benchmark_gate.json") == actual["native_benchmark_sha256"]
           and _sha(path.parent / "training_composition.json") == _sha(composition),
           "Runtime-adoption receipt/gate/composition changed")
    return actual
