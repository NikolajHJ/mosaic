"""Allocated matched full-native validation pilots; no production optimization.

The exact original driver runs in --mode validate, separately in fresh baseline
and combined child processes. Its native gates, both ten-update pilots and
runtime verdict remain intact. Additional read-only comparisons use the same
baseline-saved logits, model key and frozen preparation. Saved studies and
sealed scientific sources are inputs only.
"""

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import signal
import stat
import subprocess
import sys
import time
import uuid


VERSION = "chemistry-full-model-speed-pilot-v1"
RUNTIME_ERROR = "RuntimeError: 10,000-update allocation-fit gate failed"
OUTPUT_CAP = 2 * 1024**3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("baseline", "combined"), help=argparse.SUPPRESS)
    parser.add_argument("--baseline", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def build_driver_argv(plan, output):
    """Build the real original parser command, always validation at offset zero."""
    inputs = plan["inputs"]
    args = [
        plan.get("python", sys.executable), plan["driver"]["path"],
        "--mode", "validate", "--config", inputs["config"]["path"],
        "--output", str(output), "--manifest", plan["source"]["manifest_sha256"],
        "--cpu-certificate", inputs["cpu_certificate"]["path"],
        "--cpu-certificate-sha256", inputs["cpu_certificate"]["sha256"],
        "--reference", inputs["reference"]["path"],
        "--pocket", inputs["pocket"]["path"],
        "--pocket-sha256", inputs["pocket"]["sha256"],
        "--offset", "0", "--segment-steps", "10000",
    ]
    if plan["profile"] == "boltz-chem30k":
        args += [
            "--prepared", str(Path(inputs["prepared_context"]["path"]).parent),
            "--context-sha256", inputs["prepared_context"]["sha256"],
            "--worker-python", plan["worker_python"],
        ]
    return args


def verify_bound_file(binding):
    path = Path(binding["path"])
    if (
        not path.is_file() or path.is_symlink()
        or path.stat().st_size != binding["bytes"]
        or sha(path) != binding["sha256"]
    ):
        raise ValueError("Input file identity differs: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def verify_source(plan):
    source = Path(plan["source"]["path"])
    manifest = source / "source_manifest.json"
    if not source.is_dir() or source.is_symlink() or sha(manifest) != plan["source"]["manifest_sha256"]:
        raise ValueError("Original scientific source manifest differs")
    files = json.loads(manifest.read_text())["files"]
    if len(files) != plan["source"]["files"]:
        raise ValueError("Original scientific source member count differs")
    for name, expected in files.items():
        path = source / name
        if (
            not path.resolve().is_relative_to(source.resolve())
            or not path.is_file() or path.is_symlink() or sha(path) != expected
        ):
            raise ValueError("Original scientific source member differs: " + name)
    verify_bound_file(plan["driver"])
    return {"manifest_sha256": sha(manifest), "files": len(files)}


def allocation_guard():
    if (
        not os.environ.get("LSB_JOBID") or os.environ.get("LSB_QUEUE") != "gpul40s"
        or int(os.environ.get("LSB_DJOB_NUMPROC", "0")) != 8
        or sys.flags.optimize
    ):
        raise RuntimeError("Full native pilots require the declared eight-slot L40S allocation")


def output_bytes(root):
    """Count only this new comparison tree; never traverse links or delete data."""
    # Atomic .partial -> final renames may race a live metadata walk. Retry the
    # whole scan, never silently omit a vanished file from an otherwise valid
    # count. A continuously unstable tree still fails closed after three tries.
    for attempt in range(3):
        total = 0
        try:
            for directory, children, files in os.walk(root, followlinks=False):
                for name in children:
                    info = (Path(directory) / name).lstat()
                    if not stat.S_ISDIR(info.st_mode):
                        raise ValueError("Unexpected non-directory in disposable output tree")
                for name in files:
                    info = (Path(directory) / name).lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("Unexpected nonregular disposable output")
                    total += info.st_size
            return total
        except FileNotFoundError:
            if attempt == 2:
                raise


def _owned_group_members(group, owner):
    """Authenticate surviving descendants even after the leader has exited."""
    members = []
    token = ("MOSAIC_FULL_MODEL_PILOT_OWNER=" + owner).encode()
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if int(fields[2]) != group or fields[0] == "Z":
                continue
            if int(fields[3]) != group or token not in (directory / "environ").read_bytes().split(b"\0"):
                raise RuntimeError("Private disposable process group contains an unauthenticated member")
            members.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def _stop_own_group(process, group, owner):
    """Only the isolated session created for this exact child may be stopped."""
    if group != process.pid or group == os.getpgrp():
        raise RuntimeError("Refusing to stop an unowned process group")
    if _owned_group_members(group, owner):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        # Reauthenticate survivors using their inherited unique ownership token.
        if _owned_group_members(group, owner):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=5)
    stopped = time.monotonic()
    while _owned_group_members(group, owner):
        if time.monotonic() - stopped >= 5:
            raise RuntimeError("Owned disposable descendants survived SIGKILL")
        time.sleep(0.05)


@contextmanager
def _parent_sigterm_cleanup():
    """Convert the controller's TERM into cleanup of this parent's own children."""
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("Allocated full-model pilot parent received SIGTERM")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_isolated_child(command, source, environment, stream, timeout, comparison):
    """Bound the whole child session, including its native and pool workers."""
    started = time.monotonic()
    owner = uuid.uuid4().hex
    environment = {**environment, "MOSAIC_FULL_MODEL_PILOT_OWNER": owner}
    process = subprocess.Popen(
        command, cwd=source, env=environment, stdout=stream, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    group = process.pid
    record = {"pid": process.pid, "process_group": group, "new_session": True,
              "owner": owner, "timeout_seconds": timeout, "output_cap_bytes": OUTPUT_CAP}
    try:
        try:
            private = os.getpgid(process.pid) == group and os.getsid(process.pid) == group
        except ProcessLookupError:
            # Popen's start_new_session completed before exec. A tiny failing
            # command may already have exited; it must still be reaped below.
            private = process.poll() is not None
        if not private:
            process.kill()
            process.wait(timeout=5)
            raise RuntimeError("Disposable child did not obtain its own session")
        while True:
            size = output_bytes(comparison)
            if size > OUTPUT_CAP:
                raise RuntimeError("New disposable comparison exceeded its two-GiB output cap")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                code = process.wait(timeout=min(10.0, remaining))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                grace_started = time.monotonic()
                members = _owned_group_members(group, owner)
                while members and time.monotonic() - grace_started < 5.0:
                    time.sleep(0.05)
                    members = _owned_group_members(group, owner)
                if members:
                    raise RuntimeError("Disposable native/pool/bridge descendants remained after child completion")
                record.update(returncode=code, output_bytes=output_bytes(comparison),
                              descendant_exit_grace_seconds=time.monotonic() - grace_started,
                              elapsed_seconds=time.monotonic() - started)
                return record
            except subprocess.TimeoutExpired:
                if time.monotonic() - started >= timeout:
                    raise
    except BaseException:
        _stop_own_group(process, group, owner)
        raise


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _binding(path):
    path = Path(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha(path)}


@contextmanager
def _original_loss_bindings(original):
    """Read-only paired evaluation with original math, outside pilot timings."""
    from mosaic.small_molecule.chemistry import loss
    selected = loss._local, loss._compiled_terms
    try:
        loss._local, loss._compiled_terms = original
        yield
    finally:
        loss._local, loss._compiled_terms = selected


def _evaluation_record(evaluation):
    import numpy as np
    arrays = [np.asarray(evaluation.gradients.atoms), np.asarray(evaluation.gradients.bonds)]
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Nonfinite paired full-native gradient")
    return {
        "value": evaluation.value, "primal": evaluation.primal,
        "native_primal": evaluation.native.primal, "terms": evaluation.terms,
        "preparation_identity": evaluation.preparation_id,
        "repair_weight": evaluation.repair_weight,
        "gradient_sha256": [hashlib.sha256(value.tobytes()).hexdigest() for value in arrays],
    }


def restore_preparation(record, spec):
    """Reconstitute the original flattened record without preparing a new target."""
    import jax.numpy as jnp
    from mosaic.small_molecule.chemistry.preparation import PreparedFailureGuidance

    boolean = {"active", "changed", "cuts", "cut_active"}
    arrays = {
        name: jnp.asarray(value, dtype=jnp.bool_ if name in boolean else jnp.int32)
        for name, value in record["arrays"].items()
    }
    # PreparedFailureGuidance.record() spreads receipt fields at top level;
    # it does not contain a nested "receipt" member. Retain every receipt
    # field in its original form and let its authenticated identity verify it.
    metadata = {"arrays", "identity", "preparation_seconds", "search_seconds"}
    receipt = {name: value for name, value in record.items() if name not in metadata}
    timings = tuple((row["original"], row["additional"]) for row in record["search_seconds"])
    prepared = PreparedFailureGuidance(
        arrays, receipt, record["identity"], record["preparation_seconds"], timings,
    )
    prepared.verify(spec)
    return prepared


def replay_leaf_metadata(value):
    """Record the actual single-device FP32 source of a saved logit leaf.

    A bare ``jnp.asarray`` chooses the process default device. That is not a
    declaration of the original leaf's execution placement. Device identity is
    required here independently of the five saved-array equality checks below.
    This metadata and all array work are collected only in allocated fixtures
    or the actual native pilot; they are not a new optimization policy.
    """
    import numpy as np

    devices = list(value.devices())
    if len(devices) != 1 or type(value.sharding).__name__ != "SingleDeviceSharding":
        raise ValueError("Saved pilot replay requires the original single-device logit placement")
    device = devices[0]
    host = np.asarray(value)
    if host.dtype != np.float32 or not np.isfinite(host).all() or bool(value.weak_type):
        raise ValueError("Saved pilot replay requires finite, strongly typed FP32 logits")
    return {
        "version": "full-model-exact-logit-placement-v1",
        "platform": device.platform, "device_id": int(device.id),
        "process_index": int(device.process_index),
        "sharding_type": "SingleDeviceSharding",
        "memory_kind": value.sharding.memory_kind,
        "dtype": str(host.dtype), "shape": list(host.shape), "weak_type": False,
        "source_committed": bool(value.committed),
        "host_strides_bytes": list(host.strides),
        "host_c_contiguous": bool(host.flags.c_contiguous),
        "host_f_contiguous": bool(host.flags.f_contiguous),
    }


def restore_saved_logits(snapshot, logit_type, key, placement):
    """Replay original logits on their declared devices and prove all five fields.

    The original logit class still computes its own probability graph. Never
    substitute stored probabilities, round/re-normalize logits, select another
    device after seeing a mismatch, or use an equality tolerance. An explicit
    placement is a candidate correction, not evidence that the replay passed.
    The actual baseline snapshots each exercise this canary before paired FD.
    """
    import jax
    import numpy as np

    required = {"atom_logits": (25, 6), "bond_logits": (300, 5),
                "X": (25, 6), "E_upper": (300, 5)}
    if not isinstance(placement, dict) or set(placement) != {"atom_logits", "bond_logits"}:
        raise ValueError("Original logit leaf placement is mandatory for exact replay")
    metadata_names = {
        "version", "platform", "device_id", "process_index", "sharding_type",
        "memory_kind", "dtype", "shape", "weak_type", "source_committed",
        "host_strides_bytes", "host_c_contiguous", "host_f_contiguous",
    }
    verify_bound_file(snapshot)
    with np.load(snapshot["path"], allow_pickle=False) as saved:
        for name, shape in required.items():
            if saved[name].dtype != np.float32 or saved[name].shape != shape or not np.isfinite(saved[name]).all():
                raise ValueError("Saved replay field changed FP32 shape/finiteness: " + name)
        saved_key = saved["key_data"]
        actual_key = np.asarray(jax.random.key_data(key))
        if saved_key.dtype != np.uint32 or saved_key.shape != actual_key.shape:
            raise ValueError("Saved native key precision/shape differs")
        np.testing.assert_array_equal(saved_key, actual_key)
        leaves = []
        for name in ("atom_logits", "bond_logits"):
            declared = placement[name]
            if (not isinstance(declared, dict) or set(declared) != metadata_names
                    or declared["version"] != "full-model-exact-logit-placement-v1"
                    or declared["platform"] not in ("cpu", "gpu")
                    or type(declared["device_id"]) is not int or declared["device_id"] < 0
                    or type(declared["process_index"]) is not int or declared["process_index"] < 0
                    or declared["sharding_type"] != "SingleDeviceSharding"
                    or declared["dtype"] != "float32" or declared["shape"] != list(required[name])
                    or declared["weak_type"] is not False
                    or type(declared["source_committed"]) is not bool
                    or not isinstance(declared["host_strides_bytes"], list)
                    or len(declared["host_strides_bytes"]) != 2
                    or any(type(value) is not int or value <= 0 for value in declared["host_strides_bytes"])
                    or type(declared["host_c_contiguous"]) is not bool
                    or type(declared["host_f_contiguous"]) is not bool
                    or not (declared["memory_kind"] is None or isinstance(declared["memory_kind"], str))):
                raise ValueError("Original logit leaf placement is malformed: " + name)
            candidates = [device for device in jax.devices(declared["platform"])
                          if device.id == declared["device_id"]
                          and device.process_index == declared["process_index"]]
            if len(candidates) != 1:
                raise ValueError("Original logit leaf device is unavailable: " + name)
            sharding = jax.sharding.SingleDeviceSharding(candidates[0], memory_kind=declared["memory_kind"])
            leaf = jax.device_put(saved[name], sharding)
            observed = replay_leaf_metadata(leaf)
            # Host layout and committedness are reported independently. Moving
            # a leaf explicitly can change those transport properties; exact
            # original class/graph/key bytes remain the mandatory criterion.
            transport = {"source_committed", "host_strides_bytes", "host_c_contiguous", "host_f_contiguous"}
            for field in metadata_names - transport:
                if observed[field] != declared[field]:
                    raise ValueError("Restored logit execution placement differs: " + name + ":" + field)
            np.testing.assert_array_equal(saved[name], np.asarray(leaf))
            leaves.append(leaf)
        logits = logit_type(*leaves)
        graph = logits.graph()
        fields = {"atom_logits": logits.atoms, "bond_logits": logits.bonds,
                  "X": graph.X, "E_upper": graph.E_upper, "key_data": actual_key}
        hashes = {}
        for name, actual in fields.items():
            host = np.asarray(actual)
            if saved[name].dtype != host.dtype or saved[name].shape != host.shape:
                raise ValueError("Exact replay changed a saved field dtype/shape: " + name)
            np.testing.assert_array_equal(saved[name], host)
            if saved[name].tobytes(order="C") != host.tobytes(order="C"):
                raise AssertionError("Exact replay changed saved field bytes: " + name)
            hashes[name] = hashlib.sha256(host.tobytes(order="C")).hexdigest()
    return logits, {
        "status": "passed_exact_five_field_replay", "snapshot": snapshot,
        "placement": placement, "restored_placement": {
            "atom_logits": replay_leaf_metadata(logits.atoms),
            "bond_logits": replay_leaf_metadata(logits.bonds),
        },
        "field_bytes_sha256": hashes, "probabilities_recomputed_by_original_logit_class": True,
        "stored_probability_substitution": False, "equality_tolerance": None,
    }


def paired_saved_checks(points, contact_loss, spec, reference, key, output, original_math):
    """Full native AD, independent primals and original FD ladder at same inputs."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.chemistry import trajectory
    from mosaic.small_molecule.chemistry.loss import FailureGuidanceConfig

    output.mkdir(exist_ok=False)
    report = {"status": "running", "atol": 2e-5, "epsilons": [0.01, 0.003, 0.001], "points": []}
    try:
        if {item["label"] for item in points} != {"early-0", "early-10", "high-0", "high-10"}:
            raise ValueError("Both complete baseline pilot boundaries are required")
        for item in points:
            verify_bound_file(item["snapshot"])
            verify_bound_file(item["reference"])
            directory = output / item["label"]
            directory.mkdir()
            module_name, class_name = item["logit_class"].rsplit(".", 1)
            if (module_name, class_name) not in (
                ("mosaic.small_molecule.chemistry.state", "GraphLogits"),
                ("protenix_initialization", "ProtenixReplayLogits"),
            ):
                raise ValueError("Undeclared saved logit coordinate class")
            logit_type = getattr(importlib.import_module(module_name), class_name)
            logits, replay = restore_saved_logits(
                item["snapshot"], logit_type, key, item.get("replay_placement"),
            )
            write_json(directory / "exact-replay.json", replay)
            frozen = json.loads(Path(item["reference"]["path"]).read_text())
            prepared = restore_preparation(frozen, spec)
            config = {**item["config"], "loss": FailureGuidanceConfig(**item["config"]["loss"])}
            if config.get("epsilons", [0.01, 0.003, 0.001]) != [0.01, 0.003, 0.001]:
                raise ValueError("Predeclared paired derivative ladder differs")
            started = time.perf_counter()
            with _original_loss_bindings(original_math):
                baseline = trajectory.evaluate_combined(
                    contact_loss, logits, prepared, spec, reference, key=key, config=config,
                )
                original_checks = trajectory.check_combined_directions(
                    contact_loss, baseline, prepared, spec, reference, key=key,
                    config=config, directory=directory / "original-fd",
                )
            combined = trajectory.evaluate_combined(
                contact_loss, logits, prepared, spec, reference, key=key, config=config,
            )
            combined_checks = trajectory.check_combined_directions(
                contact_loss, combined, prepared, spec, reference, key=key,
                config=config, directory=directory / "combined-fd",
            )
            left, right = _evaluation_record(baseline), _evaluation_record(combined)
            if left["preparation_identity"] != right["preparation_identity"] or left["repair_weight"] != right["repair_weight"]:
                raise ValueError("Paired inputs, frozen target or coefficient differ")
            for name in ("value", "primal", "native_primal"):
                np.testing.assert_allclose(left[name], right[name], atol=2e-5, rtol=0)
            if set(left["terms"]) != set(right["terms"]):
                raise ValueError("Paired objective components differ")
            for name in left["terms"]:
                np.testing.assert_allclose(left["terms"][name], right["terms"][name], atol=2e-5, rtol=0)
            gradient_errors = {}
            for field in ("atoms", "bonds"):
                before = np.asarray(getattr(baseline.gradients, field))
                after = np.asarray(getattr(combined.gradients, field))
                np.testing.assert_allclose(before, after, atol=2e-5, rtol=0)
                gradient_errors[field] = float(np.max(np.abs(before - after)))
            report["points"].append({
                "label": item["label"], "status": "passed", "preparation_identity": prepared.identity,
                "snapshot": item["snapshot"], "reference": item["reference"],
                "exact_saved_replay": replay,
                "original": left, "combined": right, "max_abs_gradient_error": gradient_errors,
                "original_fd": original_checks, "combined_fd": combined_checks,
                "elapsed_seconds": time.perf_counter() - started,
            })
            write_json(output / "assessment.json", report)
        report["status"] = "passed"
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "assessment.json", report)


class _PilotMeasurements:
    def __init__(self, variant, output, baseline, original_math):
        self.variant, self.output, self.baseline = variant, output, baseline
        self.original_math = original_math
        self.timings, self.points, self.pilots = [], [], {}
        self.paired = {"status": "not_applicable_original_baseline" if variant == "baseline" else "not_started"}
        self.active = None

    @contextmanager
    def installed(self):
        from mosaic.small_molecule import chemistry
        from mosaic.small_molecule import gradient_checks
        from mosaic.small_molecule.boltz2 import checks as boltz_checks
        from mosaic.small_molecule.chemistry import trajectory

        original_prepare = chemistry.prepare_failure_guidance
        original_run = trajectory.run_chemical_updates
        original_evaluate = trajectory.evaluate_combined
        original_snapshot = trajectory._snapshot
        original_checks = trajectory.check_combined_directions
        original_raw = trajectory.evaluate_graph
        original_raw_check = gradient_checks.check_graph_gradients
        original_affinity_check = boltz_checks.check_affinity_gradients

        def timed(name, function):
            def wrapped(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    self.timings.append({
                        "stage": name, "pilot": self.active,
                        "elapsed_seconds": time.perf_counter() - started,
                    })
            return wrapped

        def snapshot(directory, step, evaluation, **kwargs):
            result = original_snapshot(directory, step, evaluation, **kwargs)
            if step in (0, 10) and self.active in ("early", "high"):
                point = {
                    "label": self.active + "-" + str(step),
                    "snapshot": _binding(Path(directory) / result["snapshot"]),
                    "reference": _binding(Path(directory) / "references" / f"step-{step:04d}.json"),
                    "logit_class": type(evaluation.logits).__module__ + "." + type(evaluation.logits).__name__,
                    "config": _plain(self.active_config),
                    "replay_placement": {
                        "atom_logits": replay_leaf_metadata(evaluation.logits.atoms),
                        "bond_logits": replay_leaf_metadata(evaluation.logits.bonds),
                    },
                }
                # Actual hardware canary: do not wait until after both pilots
                # to discover that a saved boundary cannot replay exactly.
                _, canary = restore_saved_logits(
                    point["snapshot"], type(evaluation.logits),
                    evaluation.native.evaluation.key, point["replay_placement"],
                )
                point["exact_replay_canary"] = canary
                write_json(Path(directory) / ("exact-replay-" + str(step).zfill(4) + ".json"), canary)
                self.points.append(point)
            return result

        def run(contact_loss, initial, **kwargs):
            directory = Path(kwargs["directory"])
            label = directory.name.removesuffix("_pilot")
            if label not in ("early", "high") or kwargs["steps"] != 10:
                raise ValueError("Only the original early/high ten-update pilots are permitted")
            previous = self.active
            self.active, self.active_config = label, kwargs["config"]
            started = time.perf_counter()
            try:
                result = original_run(contact_loss, initial, **kwargs)
                self.pilots[label] = result
                self.timings.append({
                    "stage": "full_pilot", "pilot": label,
                    "elapsed_seconds": time.perf_counter() - started,
                    "update_seconds": [item["elapsed_seconds"] for item in result["updates"]],
                })
                if label == "high" and self.variant == "combined":
                    self.active = "paired_read_only"
                    points = json.loads((self.baseline / "measurements.json").read_text())["saved_points"]
                    try:
                        self.paired = paired_saved_checks(
                            points, contact_loss, kwargs["spec"], kwargs["reference"], kwargs["key"],
                            self.output / "paired-checks", self.original_math,
                        )
                    except BaseException:
                        assessment = self.output / "paired-checks/assessment.json"
                        if assessment.is_file():
                            self.paired = json.loads(assessment.read_text())
                        raise
                return result
            finally:
                self.active = previous
                report_path = directory / "report.json"
                if report_path.is_file():
                    self.pilots[label] = json.loads(report_path.read_text())

        chemistry.prepare_failure_guidance = timed("detached_preparation", original_prepare)
        trajectory.run_chemical_updates = run
        trajectory.evaluate_combined = timed("combined_value_and_gradient", original_evaluate)
        trajectory.evaluate_graph = timed("native_value_and_gradient", original_raw)
        trajectory._snapshot = snapshot
        trajectory.check_combined_directions = timed("coupled_derivative_checks", original_checks)
        gradient_checks.check_graph_gradients = timed("native_raw_derivative_checks", original_raw_check)
        boltz_checks.check_affinity_gradients = timed("native_raw_derivative_checks", original_affinity_check)
        try:
            yield self
        finally:
            chemistry.prepare_failure_guidance = original_prepare
            trajectory.run_chemical_updates = original_run
            trajectory.evaluate_combined = original_evaluate
            trajectory.evaluate_graph = original_raw
            trajectory._snapshot = original_snapshot
            trajectory.check_combined_directions = original_checks
            gradient_checks.check_graph_gradients = original_raw_check
            boltz_checks.check_affinity_gradients = original_affinity_check
            write_json(self.output / "measurements.json", {
                "timings": self.timings, "saved_points": self.points,
                "paired_checks": self.paired,
                "timing_scope": "Original pilot update timers are authoritative; additional paired checks are outside those timers",
            })


@contextmanager
def combined_adapters(profile, output):
    """One activation path for validated pilots and future pinned adoption.

    The caller authenticates this capsule and original scientific inputs first.
    This context only installs the declared equivalent engineering adapters;
    it does not confer production readiness or relax a native/gradient gate.
    """
    if profile not in ("protenix-random50k", "boltz-chem30k"):
        raise ValueError("Undeclared combined chemistry profile")
    from hybrid_search import install_hybrid
    with ExitStack() as stack:
        record = {"hybrid_policy": stack.enter_context(install_hybrid(workers=4, max_entries=8192))}
        if profile == "protenix-random50k":
            from gpu_matrix import install_matrix_loss
            record["matrix_policy"] = stack.enter_context(install_matrix_loss("matrix"))
        else:
            from gpu_local_bridge import install_offloaded_matrix_loss
            record["matrix_policy"] = stack.enter_context(
                install_offloaded_matrix_loss(Path(output) / "gpu-chemistry"),
            )
        yield record


def require_driver_progress(summary):
    """Keep a primary startup/scientific error ahead of absent later reports."""
    error=summary.get("error")
    if summary.get("status")=="failed" and error!=RUNTIME_ERROR:
        raise ValueError("Original validation failed before complete admission: "+str(error))


def child_run(args, plan):
    allocation_guard()
    from full_model_contract import assess_result, validate_config, validate_plan

    validate_plan(plan)
    config = json.loads(Path(plan["inputs"]["config"]["path"]).read_text())
    validate_config(config, plan["profile"])
    args.output.mkdir(exist_ok=False)
    result = {
        "version": VERSION, "variant": args.child, "profile": plan["profile"],
        "status": "running", "original_results_changed": False,
        "job_id": os.environ["LSB_JOBID"], "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    started = time.perf_counter()
    measurements = None
    try:
        result["original_source_before"] = verify_source(plan)
        result["input_bindings_before"] = {name: verify_bound_file(value) for name, value in plan["inputs"].items()}
        source = Path(plan["source"]["path"])
        sys.path[:0] = [str(source / "scripts"), str(source / "tests"), str(source / "src")]
        import jax
        from mosaic.small_molecule.chemistry import loss
        original_math = loss._local, loss._compiled_terms
        measurements = _PilotMeasurements(args.child, args.output, args.baseline, original_math)
        argv = build_driver_argv(plan, args.output / "study")
        result["driver_argv"] = argv
        result["jax_devices"] = [{"platform": device.platform, "kind": device.device_kind} for device in jax.devices()]
        expected = "gpu" if plan["profile"] == "protenix-random50k" else "cpu"
        if jax.config.jax_enable_x64 or any(device.platform != expected for device in jax.devices()):
            raise RuntimeError("Original declared JAX backend differs")
        old_argv = sys.argv
        try:
            with ExitStack() as stack:
                if args.child == "combined":
                    result.update(stack.enter_context(combined_adapters(plan["profile"], args.output)))
                stack.enter_context(measurements.installed())
                sys.argv = argv[1:]
                try:
                    runpy.run_path(argv[1], run_name="__main__")
                    result["driver_error"] = None
                except Exception as error:
                    result["driver_error"] = f"{type(error).__name__}: {error}"
        finally:
            sys.argv = old_argv
        summary = args.output / "study/summary.json"
        if not summary.is_file():
            raise ValueError("Original full native driver did not save its summary")
        result["driver_summary"] = json.loads(summary.read_text())
        result["driver_summary_file"] = _binding(summary)
        result["pilot_reports"] = measurements.pilots
        result["paired_checks"] = measurements.paired
        # Preserve the first scientific/startup failure rather than replacing
        # it with an absent later report or missing runtime observation.
        require_driver_progress(result["driver_summary"])
        result["coupled_checks"] = {
            name: json.loads((args.output / "study" / (name + "_coupled_checks") / "report.json").read_text())
            for name in ("low", "high")
        }
        result["timings"] = measurements.timings
        result["original_source_after"] = verify_source(plan)
        result["input_bindings_after"] = {name: verify_bound_file(value) for name, value in plan["inputs"].items()}
        result["classification"] = assess_result(result, plan, config)
        result["status"] = result["classification"]["status"]
        return result
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        if measurements is not None:
            result.update(pilot_reports=measurements.pilots, paired_checks=measurements.paired, timings=measurements.timings)
        raise
    finally:
        result.update(completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        write_json(args.output / "assessment.json", result)


def parent_run(args, plan):
    allocation_guard()
    from full_model_contract import validate_plan
    validate_plan(plan)
    if args.baseline is not None:
        raise ValueError("Parent comparison cannot select an external baseline")
    args.output.mkdir(exist_ok=False)
    report = {
        "version": VERSION, "status": "running", "profile": plan["profile"],
        "plan_sha256": sha(args.plan), "original_results_changed": False, "variants": {},
        "scope": "Independent full native validation pilots only; no production/resume/restart",
    }
    started = time.monotonic()
    try:
        for variant in ("baseline", "combined"):
            output = args.output / variant
            command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(args.plan.resolve()),
                       "--output", str(output.resolve()), "--child", variant]
            if variant == "combined":
                command += ["--baseline", str((args.output / "baseline").resolve())]
            source = Path(plan["source"]["path"])
            env = dict(os.environ)
            env.update(
                PYTHONPATH=os.pathsep.join((str(source / "scripts"), str(source / "tests"), str(source / "src"))),
                PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", JAX_ENABLE_X64="0",
                JAX_PLATFORMS="cuda,cpu" if plan["profile"] == "protenix-random50k" else "cpu",
                XLA_PYTHON_CLIENT_PREALLOCATE="false", HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
            )
            remaining = plan["timeout_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Shared allocated pilot budget exhausted")
            write_json(args.output / (variant + "-command.json"), {"args": command, "driver_args": build_driver_argv(plan, output / "study")})
            with (args.output / (variant + ".log")).open("xb") as stream:
                execution = run_isolated_child(command, source, env, stream, remaining, args.output)
            report["variants"][variant] = json.loads((output / "assessment.json").read_text())
            report["variants"][variant]["isolated_child_execution"] = execution
            write_json(args.output / "assessment.json", report)
        import numpy as np
        with np.load(args.output / "baseline/study/initialization.npz", allow_pickle=False) as left, np.load(
            args.output / "combined/study/initialization.npz", allow_pickle=False,
        ) as right:
            if set(left.files) != set(right.files):
                raise ValueError("Matched full pilots have different initialization fields")
            for field in left.files:
                if left[field].dtype != right[field].dtype:
                    raise ValueError("Matched initialization dtype differs")
                np.testing.assert_array_equal(left[field], right[field])
        for name in ("model_digest", "context_digest", "config_sha256", "source_manifest_sha256"):
            if report["variants"]["baseline"]["driver_summary"][name] != report["variants"]["combined"]["driver_summary"][name]:
                raise ValueError("Matched original full-model identity differs: " + name)
        report.update(status="passed_full_model_comparison", exact_initial_arrays_and_key=True,
                      matrix_trajectory_bitwise_claim=False,
                      limitation="Pilot timing and original runtime forecasts are observations, not a production-run launch or completion")
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for variant in ("baseline", "combined"):
            assessment = args.output / variant / "assessment.json"
            if variant not in report["variants"] and assessment.is_file():
                report["variants"][variant] = json.loads(assessment.read_text())
        report["elapsed_seconds"] = time.monotonic() - started
        report["output_bytes"] = output_bytes(args.output)
        write_json(args.output / "assessment.json", report)


def main(argv=None):
    args = parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.child:
        if (args.child == "combined") != (args.baseline is not None):
            raise ValueError("Only combined child requires the same comparison's baseline")
        if args.output.name != args.child or (
            args.baseline is not None
            and args.baseline.resolve() != (args.output.parent / "baseline").resolve()
        ):
            raise ValueError("Child outputs and baseline must belong to the same new comparison")
        child_run(args, plan)
    else:
        # Only this process handles TERM; the freshly exec'ed child sessions use
        # default dispositions and are stopped only by authenticated ownership.
        allocation_guard()
        with _parent_sigterm_cleanup():
            parent_run(args, plan)


if __name__ == "__main__":
    main()
