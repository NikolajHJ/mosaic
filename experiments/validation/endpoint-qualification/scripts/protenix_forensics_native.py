"""Read-only native measurements of the authentic absolute-20k Protenix point.

This executable is diagnostic only. It never invokes a production driver, pilot,
optimizer, diffusion sampler, repair routine or continuation dispatcher. Numerical
imports and evaluations occur only after allocation and immutable-input checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import time


VERSION = "protenix-endpoint-forensics-native-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "protenix-endpoint-forensics-01"
SCIENCE = BASE / "protenix-extended-release-03/source"
SCIENCE_SHA = "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562"
HOOK = BASE / "nonchem-endpoint-release-01/source"
HOOK_SHA = "08e53ba66312b06e5f1c2836a3863479ceb85e11e9364c23026079301d2cd713"
ORIGINAL_PLAN = BASE / "nonchem-endpoint-release-01/source/plan.json"
ORIGINAL_PLAN_SHA = "cf7292ff571e202d1dadd12686a341e45d45865816c95d204dd752f8da89de39"
CONFIG = SCIENCE / "examples/configs/extended_continuation.json"
CONFIG_SHA = "f91c90770fe907c00e081f3ed28b46071ac35cc3f869edae504cf8cc16f6b445"
ENDPOINT = BASE / "protenix-extended-03/continuation-segment-0000/payload/study/case-0/snapshots/step-10000.npz"
ENDPOINT_SHA = "6be2b3868d58162d4aa3cc9d960cfb85a84d4078748f0b24d50b138ae3e47dbd"
FAILED = BASE / "nonchem-endpoint-continuation-01/protenix-validate-20000/payload/study/native_gradient_checks-prospective/admission.json"
FAILED_SHA = "3b4e6c14aa21aed083594ab040a7faf4a50f93b4adce5a4fa679108bc531aaf1"
CONTEXT_READY = BASE / "restart-lrreset-04/validate-protenix/payload/study/assessment.json"
CONTEXT_READY_SHA = "9674aacd7c8601f755fcad31df6efb9877dc39cd5114d9b1fe3ee867c1dc55dd"
POCKET_SHA = "9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43"
REFERENCE_SHA = "37b34a2ba7be47886936df48faaabdeac20481e5a9cebca9bafab831c07868eb"
MODEL_DIGEST = "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2"
CONTEXT_DIGEST = "b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4"
CACHE = Path("/work3/s234818/mosaic/cache/mosaic/protenix")
MAX_BYTES = 268435456
INTERNAL_SECONDS = 5400
MAX_GRAPHS = 109


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(),
            "Nonregular absolute protected input")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
            "Exact protected input binding required")
    require(binding(row["path"]) == row, "Protected input bytes changed: " + str(row.get("path")))
    return Path(row["path"])


def verify_tree(source, digest, count):
    """Authenticate the helper before any import of that helper itself."""
    source = Path(source)
    require(source.is_absolute() and not source.is_symlink()
            and sha(source / "source_manifest.json") == digest, "Helper source identity changed")
    files = read(source / "source_manifest.json")["files"]
    require(len(files) == count, "Helper member count changed")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(files) | {"source_manifest.json"}, "Unexpected helper source member")
    for name, expected in files.items():
        part = PurePosixPath(name)
        path = source.joinpath(*part.parts)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and path.is_file() and not path.is_symlink() and sha(path) == expected,
                "Helper source member changed")
    return files


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Missing authenticated module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_plan(plan, output):
    """Metadata-only fail-closed scope; original-arm equality is checked on disk."""
    require(plan.get("version") == VERSION, "Wrong diagnostic plan version")
    require(plan.get("max_bytes") == MAX_BYTES and plan.get("internal_seconds") == INTERNAL_SECONDS,
            "Diagnostic runtime/output cap changed")
    require(Path(plan.get("output", "")) == ROOT / "gpu/payload"
            and Path(output) == Path(plan["output"]), "Diagnostic output outside declared fresh scope")
    require(Path(plan.get("cache", str(CACHE))) == CACHE, "Native asset cache changed")
    require(plan.get("hook_source") == {"path": str(HOOK), "manifest_sha256": HOOK_SHA, "files": 14},
            "Authenticated original helper source changed")
    for key, path, digest, size in (
        ("original_plan", ORIGINAL_PLAN, ORIGINAL_PLAN_SHA, None),
        ("failed_admission", FAILED, FAILED_SHA, 2531142),
        ("context_ready", CONTEXT_READY, CONTEXT_READY_SHA, 21266),
    ):
        value = plan.get(key, {})
        require(value.get("path") == str(path) and value.get("sha256") == digest
                and type(value.get("bytes")) is int and value["bytes"] > 0
                and (size is None or value["bytes"] == size), "Wrong immutable diagnostic input: " + key)
    arm = plan.get("arm", {})
    require(arm.get("id") == "protenix" and arm.get("rate") == .3
            and arm.get("start_global") == 20000 and arm.get("target_global") == 50000
            and arm.get("segment_updates") == 10000, "Protenix saved boundary/settings changed")
    require(arm.get("source") == {"path": str(SCIENCE), "manifest_sha256": SCIENCE_SHA, "files": 643},
            "Scientific source changed")
    require(arm.get("config") == {"path": str(CONFIG), "bytes": 1733, "sha256": CONFIG_SHA},
            "Scientific configuration changed")
    expected = {"path": str(ENDPOINT), "bytes": 14284, "sha256": ENDPOINT_SHA}
    require(arm.get("parent_snapshot") == expected and arm.get("failed_parent", {}).get("endpoint") == expected
            and arm["failed_parent"].get("job_id") == "29809291", "Saved failed endpoint changed")
    require(arm.get("inputs", {}).get("pocket", {}).get("sha256") == POCKET_SHA
            and arm.get("inputs", {}).get("reference", {}).get("sha256") == REFERENCE_SHA,
            "Original pocket/reference changed")
    require(arm.get("expected", {}).get("saved_parent_scientific_status") == "failed",
            "Original failure must remain failed")
    return arm


def allocated_environment(environ):
    require(str(environ.get("LSB_JOBID", "")).isdigit() and environ.get("LSB_QUEUE") == "gpul40s",
            "Allocated gpul40s diagnostic job required")
    counts = []
    if environ.get("LSB_DJOB_NUMPROC"):
        require(str(environ["LSB_DJOB_NUMPROC"]).isdigit(), "Malformed LSF slot count")
        counts.append(int(environ["LSB_DJOB_NUMPROC"]))
    if environ.get("LSB_MCPU_HOSTS"):
        tokens = environ["LSB_MCPU_HOSTS"].split()
        require(len(tokens) % 2 == 0 and tokens and all(value.isdigit() for value in tokens[1::2]),
                "Malformed LSF host slots")
        counts.append(sum(map(int, tokens[1::2])))
    require(counts and all(count == 4 for count in counts), "Exactly four allocated CPU slots required")
    return {key: environ[key] for key in ("LSB_JOBID", "LSB_QUEUE", "LSB_DJOB_NUMPROC", "LSB_MCPU_HOSTS")
            if key in environ}


def output_bytes(directory):
    total = 0
    for path in Path(directory).rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "Special diagnostic output forbidden")
        if path.is_file():
            total += path.stat().st_size
    return total


@contextmanager
def deadline(seconds):
    """A stuck native evaluation still leaves time for final preservation proofs."""
    require(hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"), "Allocated Linux timer required")
    def expired(_signum, _frame):
        raise TimeoutError("Diagnostic internal evaluation deadline exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def protected_bindings(value):
    found = {}
    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"path", "bytes", "sha256"}:
                bound(item)
                require(item["path"] not in found or found[item["path"]] == item, "Conflicting protected binding")
                found[item["path"]] = item
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return found


def verify_inputs(plan, output):
    arm = validate_plan(plan, output)
    verify_tree(HOOK, HOOK_SHA, 14)
    hooks = load_module(HOOK / "scripts/continuation_driver.py", "forensics_pinned_original_hooks")
    hooks.verify_source(SCIENCE, SCIENCE_SHA, 643)
    original = read(bound(plan["original_plan"]))
    require(hooks.arm_record(original, "protenix") == arm, "Diagnostic arm differs from authenticated original plan")
    failed = read(bound(plan["failed_admission"]))
    require(failed.get("status") == "failed" and failed.get("can_admit_new_child") is False,
            "Expected rejected admission is not preserved")
    config = read(bound(arm["config"]))
    require(config.get("initial_step_size") == .3 and config.get("arm") == "protenix"
            and config.get("chemistry_enabled") is False, "Original native-only configuration changed")
    before = protected_bindings({"plan": plan, "arm": arm})
    ready = read(bound(plan["context_ready"]))
    require(ready.get("status") == "passed" and ready.get("model_digest") == MODEL_DIGEST
            and ready.get("context_digest") == CONTEXT_DIGEST, "Original frozen readiness identity changed")
    for name, digest in ready["context_files"].items():
        path = CONTEXT_READY.parent / "context" / name
        require(path.name == name and sha(path) == digest, "Original prepared context bytes changed")
        before[str(path)] = binding(path)
    return arm, hooks, failed, ready, before


def native_evaluator(plan, arm, hooks, ready, output, lifecycle=None):
    """Return one evaluator with exact original loaders, key and point placement."""
    require(not any(name == "mosaic" or name.startswith("mosaic.") for name in sys.modules),
            "Fresh process required before original scientific imports")
    sys.path[:0] = [str(SCIENCE / "src"), str(SCIENCE / "scripts")]
    import jax
    import jax.numpy as jnp
    import numpy as np
    jax.config.update("jax_default_matmul_precision", "highest")
    require(not jax.config.jax_enable_x64 and len(jax.devices("gpu")) == 1
            and all(device.platform == "gpu" for device in jax.devices()), "Exact FP32 one-GPU execution required")
    require("L40S" in jax.devices("gpu")[0].device_kind, "Declared L40S native allocation required")
    from mosaic.small_molecule.protenix import FrozenProtenixTiny, native_distogram_bins
    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.optimizers import evaluate_graph, _nonfinite_paths
    from mosaic.small_molecule.state import RelaxedGraph
    import extended_protocol
    import protenix_initialization
    require(Path(extended_protocol.__file__).resolve() == SCIENCE / "scripts/extended_protocol.py"
            and Path(protenix_initialization.__file__).resolve() == SCIENCE / "scripts/protenix_initialization.py",
            "Original loader module origins differ")
    model = FrozenProtenixTiny.load(cache_dir=CACHE)
    require(array_digest(model) == MODEL_DIGEST, "Loaded original model bytes changed")
    if lifecycle is not None:
        lifecycle.update(model_frozen=lambda: {"model_digest": array_digest(model)},
                         model_before={"model_digest": MODEL_DIGEST})
    write(output / "model-loaded.json", {"model_digest": MODEL_DIGEST, "context_loaded": False,
                                        "updates": 0})
    context = load_context(CONTEXT_READY.parent / "context", ready["context_files"])
    require(context.n_atoms == 25, "Original occupied slot count changed")
    pocket = read(bound(arm["inputs"]["pocket"]))
    pocket_tokens = map_protenix_pocket(context, pocket)
    loss = model.build_loss(context=context, loss=NativeContactLoss(8.0, pocket_tokens))
    key = jax.random.key(17)
    def frozen_identity():
        result = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        require(result == {"model_digest": MODEL_DIGEST, "context_digest": CONTEXT_DIGEST},
                "Original frozen model/context changed")
        return result
    identity_before = frozen_identity()
    if lifecycle is not None:
        lifecycle.update(frozen=frozen_identity, frozen_before=identity_before)
    write(output / "frozen-before.json", identity_before)
    logits = extended_protocol.restore_logits(ENDPOINT, ENDPOINT_SHA,
                protenix_initialization.ProtenixReplayLogits, key, sha)
    template = logits.graph()
    with np.load(ENDPOINT, allow_pickle=False) as saved:
        for field, value in (("X", template.X), ("E_upper", template.E_upper),
                             ("atom_logits", logits.atoms), ("bond_logits", logits.bonds),
                             ("key_data", jax.random.key_data(key))):
            np.testing.assert_array_equal(saved[field], np.asarray(value))
    write(output / "native-identity.json", {**identity_before, "key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "rate_metadata_only": .3, "updates": 0, "source_manifest_sha256": SCIENCE_SHA,
        "snapshot": binding(ENDPOINT), "config": binding(CONFIG), "pocket": arm["inputs"]["pocket"],
        "ligand_token_indices": list(context.ligand_token_indices), "pocket_token_indices": list(pocket_tokens),
        "raw_point_placement": {field: {"device": str(getattr(template, field).device),
            "dtype": str(getattr(template, field).dtype), "shape": list(getattr(template, field).shape),
            "sha256": hashlib.sha256(np.asarray(getattr(template, field)).tobytes()).hexdigest()}
            for field in ("X", "E_upper")},
        "contact_control": "same FP32 native logits; FP64 host softmax/reduction is supplementary only; trunk remains unchanged FP32"})
    ligand = jnp.asarray(context.ligand_token_indices, dtype=jnp.int32)
    partners = jnp.asarray(pocket_tokens, dtype=jnp.int32)
    bins = native_distogram_bins()
    def contacts(aux):
        selected = aux["distogram_logits"][ligand[:, None], partners[None, :], :]
        probabilities = jax.nn.softmax(selected, axis=-1)
        native = jnp.sum(probabilities * (bins < 8.0), axis=-1)
        raw = np.asarray(selected, dtype=np.float64)
        exponential = np.exp(raw - raw.max(axis=-1, keepdims=True))
        host = np.sum((exponential / exponential.sum(axis=-1, keepdims=True))
                      * (np.asarray(bins, dtype=np.float64) < 8.0), axis=-1)
        require(np.isfinite(native).all() and np.isfinite(host).all(), "Nonfinite contact diagnostic reduction")
        return np.asarray(native), host
    started = time.monotonic()
    def evaluator(values):
        require(time.monotonic() - started < INTERNAL_SECONDS, "Diagnostic evaluation deadline exhausted")
        require(output_bytes(output) < MAX_BYTES - 1048576, "Diagnostic persistent output cap exhausted")
        graph = hooks.restore_point_placement(values, template, RelaxedGraph, jax.device_put)
        primal, aux = loss(graph, key=key)
        require(not _nonfinite_paths((primal, aux)), "Nonfinite independent native forward")
        raw = evaluate_graph(loss, graph, key)
        pcontacts, p64 = contacts(aux)
        acontacts, a64 = contacts(raw.aux)
        return {"primal": np.asarray(primal), "ad_value": np.asarray(raw.value),
            "gradients": {"X": np.asarray(raw.gradients.X), "E_upper": np.asarray(raw.gradients.E_upper)},
            "primal_contacts": pcontacts, "ad_contacts": acontacts,
            "primal_contacts64": p64, "ad_contacts64": a64}
    return evaluator, frozen_identity


def final_proofs(output, before, lifecycle, hooks):
    """Attempt each preservation proof independently even after an earlier error."""
    result, failures = {}, []
    try:
        if before is None:
            result["protected_inputs_unchanged"] = "not_established_before_model"
        else:
            after = {path: binding(path) for path in before}
            write(output / "protected-after.json", after)
            require(after == before, "Protected original scientific/proof bytes changed")
            result["protected_inputs_unchanged"] = True
    except BaseException as error:
        failures.append("protected: " + type(error).__name__ + ": " + str(error))
    try:
        if lifecycle.get("frozen") is not None:
            terminal = lifecycle["frozen"]()
            write(output / "frozen-after.json", terminal)
            require(terminal == lifecycle["frozen_before"], "Frozen native identity changed")
            result["frozen_unchanged"] = True
        elif lifecycle.get("model_frozen") is not None:
            terminal = lifecycle["model_frozen"]()
            write(output / "frozen-model-after.json", terminal)
            require(terminal == lifecycle["model_before"], "Loaded frozen model identity changed")
            result.update(frozen_model_unchanged=True, frozen_unchanged="full_context_not_available")
        else:
            result["frozen_unchanged"] = "model_not_available"
    except BaseException as error:
        failures.append("frozen: " + type(error).__name__ + ": " + str(error))
    try:
        verify_tree(HOOK, HOOK_SHA, 14)
        if hooks is not None:
            hooks.verify_source(SCIENCE, SCIENCE_SHA, 643)
        result["source_trees_unchanged"] = True
    except BaseException as error:
        failures.append("source: " + type(error).__name__ + ": " + str(error))
    try:
        result["persistent_bytes_before_assessment"] = output_bytes(output)
        require(result["persistent_bytes_before_assessment"] < MAX_BYTES - 65536,
                "Diagnostic persistent output cap exceeded")
    except BaseException as error:
        failures.append("budget: " + type(error).__name__ + ": " + str(error))
    result["preservation_errors"] = failures
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", "--plan-sha", dest="plan_sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    allocation = allocated_environment(os.environ)
    require(sha(args.plan) == args.plan_sha256, "New diagnostic plan bytes changed")
    plan = read(args.plan)
    validate_plan(plan, args.output)
    require(not args.output.exists() and not args.output.is_symlink(), "Refusing existing diagnostic output")
    args.output.mkdir(parents=True, exist_ok=False)
    assessment = {"version": VERSION, "status": "failed", "job_id": os.environ["LSB_JOBID"],
        "allocation": allocation, "plan": binding(args.plan), "updates": 0,
        "original_failed_outcomes_preserved": True, "optimization_settings_changed": False,
        "can_certify_or_restart_original_study": False, "rate_metadata_only": .3,
        "scope": "separate fixed saved-endpoint diagnostics only; no original gate or prospective admission pass"}
    before = None
    hooks = None
    lifecycle = {}
    error = None
    started = time.monotonic()
    try:
        with deadline(INTERNAL_SECONDS):
            arm, hooks, failed, ready, before = verify_inputs(plan, args.output)
            before[str(args.plan)] = binding(args.plan)
            write(args.output / "protected-before.json", before)
            evaluator, frozen = native_evaluator(plan, arm, hooks, ready, args.output, lifecycle)
            import protenix_endpoint_forensics
            require(Path(protenix_endpoint_forensics.__file__).resolve()
                    == Path(__file__).resolve().with_name("protenix_endpoint_forensics.py"),
                    "Diagnostic computation module origin changed")
            result = protenix_endpoint_forensics.run_forensics(failed, evaluator, frozen,
                args.output / "forensics", maximum_output_bytes=MAX_BYTES - 4194304,
                maximum_unique_graphs=MAX_GRAPHS)
            require(result.get("can_certify_or_restart_original_study") is False,
                    "Forensics cannot certify or restart original study")
            assessment.update(status="diagnostics_completed_original_failure_preserved",
                report=binding(args.output / "forensics/report.json"))
    except BaseException as caught:
        error = caught
        assessment["error"] = type(caught).__name__ + ": " + str(caught)
    finally:
        assessment.update(final_proofs(args.output, before, lifecycle, hooks))
        if assessment["preservation_errors"]:
            assessment["status"] = "failed"
            if error is None:
                error = ValueError("; ".join(assessment["preservation_errors"]))
        assessment["elapsed_seconds"] = time.monotonic() - started
        write(args.output / "assessment.json", assessment)
    if error is not None:
        raise error
    return assessment


if __name__ == "__main__":
    main()
