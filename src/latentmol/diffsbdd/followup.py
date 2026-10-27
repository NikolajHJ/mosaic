"""Independent ordinary Boltz fresh-pose confirmation after the pilot exits.

This module imports only the standard library at module scope. Its allocated
parent performs CPU graph assessment, then launches one ordinary native CLI
child at a time. The parent never initializes CUDA or loads either checkpoint.
Every valid terminal identity retains all seed/arm memberships and receives the
two prespecified fresh seeds. Predictions are model confirmation, not measured
binding affinity or a successful hybrid-method efficacy claim.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time


BOLTZ_COMMIT = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
MAIN_SHA256 = "b5ded4dac1552af2e079c335d56e64ea580b4d5ca0202a9542385ce3e318c3b3"
AFFINITY_SHA256 = "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"
SEEDS = (17, 23)
VERSION = "diffsbdd-independent-native-fresh-pose-followup-v1"
FAILURE_LOG_PATTERNS = (r"Tokenizer failed", r"Cropper failed", r"Molecule loading failed",
    r"Featurizer failed", r"Failed to process .*Skipping", r"Traceback \(most recent call last\)",
    r"Number of failed examples:\s*[1-9][0-9]*", r"Found existing .*skipping")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _verified(asset):
    if not isinstance(asset, dict) or not {"path", "sha256"} <= set(asset):
        raise ValueError("Follow-up assets require explicit path/SHA256 bindings")
    path = Path(asset["path"]).resolve(strict=True)
    if not path.is_file() or _sha(path) != asset["sha256"]:
        raise ValueError("Independent follow-up asset SHA256 mismatch: " + str(path))
    return path


def _parent_cuda_uninitialized():
    module = sys.modules.get("torch")
    if module is not None and getattr(getattr(module, "cuda", None), "_initialized", False):
        raise RuntimeError("Fresh-pose follow-up must be a separate parent process that has never initialized CUDA")


def verify_native_cli():
    """Verify installed CLI metadata and the exact official source inspected."""
    dist = importlib.metadata.distribution("boltz")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != BOLTZ_COMMIT:
        raise ValueError("Fresh-pose CLI must be installed from the pinned native Boltz commit")
    candidates = [dist.locate_file(name) for name in (dist.files or [])
                  if str(name).replace("\\", "/").endswith("boltz/main.py")]
    if not candidates:
        candidates = [dist.locate_file("boltz/main.py")]
    sources = [Path(path).resolve() for path in candidates if Path(path).is_file()]
    if len(set(sources)) != 1 or _sha(sources[0]) != MAIN_SHA256:
        raise ValueError("Installed Boltz CLI source differs from the audited pinned main.py")
    return dict(boltz_commit=BOLTZ_COMMIT, package_version=dist.version,
                main_source_sha256=MAIN_SHA256, main_source_path=str(sources[0]), direct_url=direct)


def _catalog_verified(asset):
    """Authenticate a complete readonly native molecule tree, without editing it."""
    root = Path(asset["path"]).resolve(strict=True)
    manifest_path = _verified(asset["manifest"])
    catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = catalog.get("members")
    if not root.is_dir() or not isinstance(members, dict) or not members:
        raise ValueError("Native molecule cache needs a complete nonempty authenticated member catalog")
    found = set()
    for relative, expected in members.items():
        path = (root / relative).resolve(strict=True)
        expected_hash = expected["sha256"] if isinstance(expected, dict) else expected
        if not path.is_relative_to(root) or not path.is_file() or _sha(path) != expected_hash:
            raise ValueError("Readonly native molecule cache member mismatch: " + relative)
        found.add(relative)
    actual = {str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()}
    if actual != found:
        raise ValueError("Native molecule cache catalog omits or adds files")
    return root, dict(path=str(root), manifest_sha256=asset["manifest"]["sha256"], members=len(members))


def make_owned_cache(cache, assets, *, expected_ready_sha256=None):
    """Native download helper sees only existing, authenticated readonly assets.

    The ordinary CLI calls download_boltz2 even with --checkpoint; its guards
    require mols.tar, mols/, boltz2_conf.ckpt and boltz2_aff.ckpt. A new directory
    owns the symlinks; original caches/checkpoints receive no writes or chmod.
    """
    cache = Path(cache)
    affinity, confidence = _verified(assets["affinity"]), _verified(assets["confidence"])
    if assets["affinity"]["sha256"] != AFFINITY_SHA256:
        raise ValueError("Follow-up affinity checkpoint is not the pinned original native model")
    archive = _verified(assets["mols_tar"])
    molecules, receipt = _catalog_verified(assets["mols_dir"])
    targets = {"boltz2_aff.ckpt": affinity, "boltz2_conf.ckpt": confidence,
               "mols.tar": archive, "mols": molecules}
    ready = cache / "ready.json"
    if cache.exists():
        if expected_ready_sha256 is None or not ready.is_file() or _sha(ready) != expected_ready_sha256:
            raise ValueError("Existing follow-up cache requires its explicitly authenticated ready receipt")
        saved = json.loads(ready.read_text(encoding="utf-8"))
        if saved.get("asset_bindings") != assets or set(saved.get("links", {})) != set(targets):
            raise ValueError("Existing owned follow-up cache has different sealed asset bindings")
        for name, target in targets.items():
            if not (cache / name).is_symlink() or (cache / name).resolve(strict=True) != target:
                raise ValueError("Existing owned follow-up cache symlink target changed")
        return {**saved, "ready_sha256": expected_ready_sha256, "reused_authenticated_cache": True}
    cache.mkdir(parents=True, exist_ok=False)
    for name, target in targets.items():
        (cache / name).symlink_to(target, target_is_directory=target.is_dir())
    receipt = dict(policy="new owned cache; symlinks to authenticated readonly existing assets",
                cache=str(cache.resolve()), links={name: str(target) for name, target in targets.items()},
                asset_bindings=assets, molecules=receipt, source_stats={name: {"size": target.stat().st_size,
                    "mtime_ns": target.stat().st_mtime_ns} for name, target in targets.items()})
    _write(ready, receipt)
    return {**receipt, "ready_sha256": _sha(ready), "reused_authenticated_cache": False}


def _load_builder(source_asset):
    # These imports are inside the allocated CPU stage. No diffusion checkpoint
    # or scorer is loaded, and CUDA is checked before and after graph assessment.
    from .native import authenticate_native_source
    from .transport import NativeBuilder
    receipt = authenticate_native_source(source_asset["path"], source_asset["manifest_sha256"])
    root = Path(receipt["source_root"])
    for name in ("utils", "constants"):
        old = sys.modules.get(name)
        if old is not None and not Path(old.__file__).resolve().is_relative_to(root):
            raise RuntimeError("Conflicting native source module in follow-up parent: " + name)
    path = root / "analysis" / "molecule_builder.py"
    if _sha(path) != receipt["members"]["analysis/molecule_builder.py"]:
        raise ValueError("Native molecule builder differs from authenticated source")
    spec = importlib.util.spec_from_file_location("_diffsbdd_followup_builder", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
        import constants
        dataset_info = constants.dataset_params["crossdock"]
    finally:
        sys.path.remove(str(root))
    return NativeBuilder(module.build_molecule, dataset_info), tuple(dataset_info["atom_decoder"]), receipt


def collect_terminal_identities(pilot_dir, builder, atom_decoder):
    """All 48 terminal memberships, including valid unmodified D baselines."""
    import torch
    from rdkit import Chem
    from .transport import HandoffRejected, MolecularGraph, PoseCandidate
    pilot_dir = Path(pilot_dir)
    cohort = json.loads((pilot_dir / "cohort.json").read_text(encoding="utf-8"))
    if (cohort.get("stage") != "pilot" or cohort.get("seeds") != list(range(31, 39))
            or cohort.get("arms") != ["D", "H0", "S", "A1", "A5", "A5-exact"]
            or cohort.get("status") != "completed_cohort_not_fresh_pose_validation"):
        raise ValueError("Follow-up requires the complete, unchanged prespecified pilot cohort")
    identities, attempts = {}, []
    for seed in cohort["seeds"]:
        for arm in cohort["arms"]:
            directory = pilot_dir / f"seed-{seed:03d}" / arm
            path = directory / "terminal.pt"
            arm_receipt = json.loads((directory / "arm.json").read_text(encoding="utf-8"))
            if (arm_receipt.get("seed") != seed or arm_receipt.get("arm") != arm
                    or arm_receipt.get("status") != "completed_sampler_arm"
                    or _sha(path) != arm_receipt.get("terminal_sha256")):
                raise ValueError("Pilot terminal membership/hash identity mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("seed") != seed or payload.get("arm") != arm:
                raise ValueError("Saved terminal belongs to a different pilot membership")
            tensors = payload["tensors"]
            xyz, features = tensors["positions"], tensors["one_hot"]
            record = dict(seed=seed, arm=arm, terminal_sha256=_sha(path), valid=False,
                          positions=xyz.tolist(), atom_features=features.tolist(), failure=None)
            try:
                if (xyz.shape != (25, 3) or features.shape != (25, len(atom_decoder))
                        or not bool(xyz.isfinite().all() and features.isfinite().all())
                        or not bool(((features == 0) | (features == 1)).all())
                        or not bool((features.sum(-1) == 1).all())):
                    raise HandoffRejected("malformed_native_terminal_point_cloud")
                elements = tuple(atom_decoder[index] for index in features.argmax(-1).tolist())
                placeholder = MolecularGraph(elements, (0,) * 25,
                    tuple((0,) * 25 for _ in range(25)), tuple(range(25)))
                graph, builder_receipt = builder(PoseCandidate(placeholder, xyz))
                record["builder"] = builder_receipt
                if not graph.is_connected():
                    raise HandoffRejected("disconnected_terminal", builder_receipt)
                molecule = graph.to_rdkit(xyz)
                identity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
                key = hashlib.sha256(identity.encode()).hexdigest()
                record.update(valid=True, identity=identity, identity_sha256=key, builder=builder_receipt)
                bucket = identities.setdefault(identity, dict(identity=identity, identity_sha256=key, memberships=[]))
                bucket["memberships"].append(dict(seed=seed, arm=arm, terminal_sha256=record["terminal_sha256"]))
            except (HandoffRejected, ValueError, RuntimeError) as error:
                record.update(failure=str(error), failure_type=type(error).__name__,
                              failure_receipt=getattr(error, "receipt", {}))
            attempts.append(record)
    return list(identities.values()), attempts


def prediction_command(python, yaml_path, output, cache, seed, assets, prediction):
    """All flags below are present in MAIN_SHA256's official Click command."""
    expected = {"recycling_steps", "sampling_steps", "diffusion_samples", "sampling_steps_affinity",
                "diffusion_samples_affinity", "max_msa_seqs"}
    if set(prediction) != expected or any(type(value) is not int or value < 1 for value in prediction.values()):
        raise ValueError("Fresh-pose native sampling settings must be explicitly sealed positive integers")
    if prediction["diffusion_samples"] != 1:
        raise ValueError("One model per fresh seed avoids outcome-based confidence selection")
    if seed not in SEEDS:
        raise ValueError("Fresh-pose seeds are exactly 17 and 23")
    command = [str(python), "-m", "boltz.main", "predict", str(yaml_path), "--out_dir", str(output),
        "--cache", str(cache), "--checkpoint", str(assets["confidence"]["path"]),
        "--affinity_checkpoint", str(assets["affinity"]["path"]), "--model", "boltz2",
        "--accelerator", "gpu", "--devices", "1", "--num_workers", "0",
        "--preprocessing-threads", "1", "--output_format", "mmcif", "--no_kernels", "--seed", str(seed)]
    for key in sorted(prediction):
        command.extend(["--" + key, str(prediction[key])])
    return command


def validate_prediction_output(output, record_id, identity, stdout):
    """Catch native skip/retry behavior and authenticate actual output chemistry."""
    from rdkit import Chem
    failures = [pattern for pattern in FAILURE_LOG_PATTERNS if re.search(pattern, stdout, flags=re.IGNORECASE)]
    if failures:
        raise RuntimeError("Native CLI logged failed/skipped/retried input: " + ", ".join(failures))
    root = Path(output) / f"boltz_results_{record_id}"
    manifest = json.loads((root / "processed" / "manifest.json").read_text(encoding="utf-8"))
    records = manifest["records"] if isinstance(manifest, dict) else manifest
    if len(records) != 1 or records[0].get("id") != record_id or records[0].get("affinity") is None:
        raise ValueError("Native fresh-pose processed manifest dropped/replaced its one input")
    # This pickle is produced by this exact hash-authenticated child, not an
    # arbitrary external pickle. It verifies the parser did not standardize to
    # a different graph before accepting its predicted affinity.
    with (root / "processed" / "mols" / f"{record_id}.pkl").open("rb") as stream:
        molecules = pickle.load(stream)
    if len(molecules) != 1:
        raise ValueError("Native fresh-pose parser changed the ligand molecule count")
    molecule = Chem.RemoveHs(Chem.Mol(next(iter(molecules.values()))))
    Chem.RemoveStereochemistry(molecule)
    if Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) != identity:
        raise ValueError("Native fresh-pose parser changed the terminal molecular identity")
    directory = root / "predictions" / record_id
    files = [directory / f"{record_id}_model_0.cif", directory / f"confidence_{record_id}_model_0.json",
             directory / f"affinity_{record_id}.json", directory / f"pre_affinity_{record_id}.npz"]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files):
        raise ValueError("Native fresh-pose prediction omitted required pose/confidence/affinity output")
    affinity = json.loads(files[2].read_text(encoding="utf-8"))
    confidence = json.loads(files[1].read_text(encoding="utf-8"))
    if not {"affinity_pred_value", "affinity_probability_binary"} <= set(affinity):
        raise ValueError("Native fresh-pose affinity output keys are incomplete")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in affinity.values()):
        raise ValueError("Native fresh-pose affinity values are nonfinite")
    def finite_json(value):
        if isinstance(value, dict):
            return all(finite_json(item) for item in value.values())
        return type(value) in (int, float) and math.isfinite(value)
    if not isinstance(confidence, dict) or "confidence_score" not in confidence or not finite_json(confidence):
        raise ValueError("Native fresh-pose confidence output is incomplete or nonfinite")
    return dict(affinity=affinity, confidence=confidence,
                files={str(path.relative_to(root)): _sha(path) for path in files},
                processed_manifest_sha256=_sha(root / "processed" / "manifest.json"),
                processed_ligand_sha256=_sha(root / "processed" / "mols" / f"{record_id}.pkl"),
                selected_models=1, identity_preserved=True)


def _own_bytes(root):
    return sum(path.lstat().st_size for path in Path(root).rglob("*") if path.is_file() and not path.is_symlink())


def runtime_forecast(measured_seconds, remaining_cases, startup_margin_seconds):
    if (not measured_seconds or type(remaining_cases) is not int or remaining_cases < 0
            or any(not math.isfinite(value) or value <= 0 for value in measured_seconds)
            or not math.isfinite(startup_margin_seconds) or startup_margin_seconds < 0):
        raise ValueError("Runtime forecasting requires measured positive durations and a fixed remaining case count")
    return 1.5 * max(measured_seconds) * remaining_cases + startup_margin_seconds


def run_followup(config, pilot_dir, output, deadline):
    """Run a new allocated confirmation stage, after the pilot process has exited.

    Config requires: python, assets {affinity, confidence, mols_tar, mols_dir
    (path + manifest path/SHA)}, diffusion_source {path,manifest_sha256}, protein
    {sequence,sequence_sha256,msa path/SHA}, prediction (all six native sampling
    settings), run_root, owned_cache, startup_margin_seconds and
    artifact_cap_bytes and bootstrap_forecast. ``owned_cache`` is an explicit new dedicated namespace
    outside run_root, ending in ``followup-native``; the full run inventory
    therefore contains no symlinks. Reuse requires cache_ready_sha256.
    The caller seals
    the config and allocation (<=24h, <=23h internal); no settings are inferred
    from observed affinities. This function does not schedule or resume jobs.
    """
    _parent_cuda_uninitialized()
    started = time.monotonic()
    if Path(config["python"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("Follow-up worker and native children must use the same authenticated interpreter")
    if not math.isfinite(deadline) or not 0 < deadline - started <= 23 * 3600 + 1:
        raise ValueError("Independent follow-up needs an explicit <=23h internal monotonic deadline")
    margin = config["startup_margin_seconds"]
    if not isinstance(margin, (int, float)) or not math.isfinite(margin) or margin < 0:
        raise ValueError("Follow-up forecast requires an explicit nonnegative startup/checkpoint margin")
    cap = config["artifact_cap_bytes"]
    if type(cap) is not int or not 0 < cap <= 2 * 1024**3:
        raise ValueError("Follow-up own-artifact cap cannot exceed the sealed 2GiB campaign cap")
    bootstrap = config["bootstrap_forecast"]
    if bootstrap != dict(mode="bounded_first_case_then_measured_forecast", initial_case_seconds=3300,
                         factor=1.5, retain_all_prespecified_memberships=True):
        raise ValueError("Native follow-up requires the sealed bounded first-case bootstrap and complete remaining cohort forecast")
    output = Path(output)
    run_root = Path(config["run_root"]).resolve(strict=True)
    cache_path = Path(config["owned_cache"])
    if (not cache_path.is_absolute() or cache_path.name != "followup-native"
            or cache_path.resolve().is_relative_to(run_root)):
        raise ValueError("Owned native follow-up cache requires its new explicit outside-run-root followup-native namespace")
    if (not output.resolve().is_relative_to(run_root) or output.resolve() == run_root
            or cache_path.is_relative_to(output.resolve()) or output.resolve().is_relative_to(cache_path)):
        raise ValueError("Follow-up output must be inside the sealed run root and separate from its explicit owned cache")
    output.mkdir(parents=True, exist_ok=False)
    result = dict(version=VERSION, status="running_not_passed", passed=False, seeds=list(SEEDS),
        parent_pid=os.getpid(), gpu_ownership="only one sequential native CLI child; parent CUDA never initialized",
        checkpoint_weights_updated=False, desired_score_labels=False,
        comparison_scope="fresh-pose model confirmation; neither fixed-pose DBMol scores nor experimental affinity",
        completed=[], failed=[], not_reached=[], forecast_safety_factor=1.5,
        bootstrap_forecast=bootstrap)
    try:
        result["native_cli"] = verify_native_cli()
        assets = config["assets"]
        cache = make_owned_cache(cache_path, assets, expected_ready_sha256=config.get("cache_ready_sha256"))
        result["cache"] = cache
        protein = config["protein"]
        msa = _verified(protein["msa"])
        sequence = protein["sequence"]
        if hashlib.sha256(sequence.encode()).hexdigest() != protein["sequence_sha256"]:
            raise ValueError("Fresh-pose receptor sequence binding mismatch")
        # Query parsing is a CPU text operation; never ask the native MSA server.
        from .scorer_context import msa_query
        if msa_query(msa) != sequence:
            raise ValueError("Fresh-pose supplied MSA query differs from frozen receptor sequence")
        builder, decoder, source = _load_builder(config["diffusion_source"])
        result["native_builder_source_manifest_sha256"] = source["manifest_sha256"]
        identities, attempts = collect_terminal_identities(pilot_dir, builder, decoder)
        _parent_cuda_uninitialized()
        _write(output / "terminal-validity.json", dict(attempts=attempts,
            denominator=len(attempts), valid=sum(row["valid"] for row in attempts),
            valid_unmodified_D=sum(row["valid"] and row["arm"] == "D" for row in attempts),
            deduplication="identical stereo-stripped graph identity only; every seed/arm membership preserved"))
        cases = [dict(identity=row["identity"], identity_sha256=row["identity_sha256"],
            memberships=row["memberships"], pose_seed=seed) for row in identities for seed in SEEDS]
        _write(output / "sealed-cases.json", dict(version=VERSION, cases=cases, case_count=len(cases),
            pilot_cohort_sha256=_sha(Path(pilot_dir) / "cohort.json"), seeds=list(SEEDS), failures_replaced=False))
        result.update(terminal_memberships=len(attempts), distinct_valid_identities=len(identities), required_cases=len(cases))
        if not cases:
            raise ValueError("No valid terminal identities exist; no native-confirmation or efficacy pass")
        measured = []
        for index, case in enumerate(cases):
            remaining = len(cases) - index
            if measured:
                forecast = runtime_forecast(measured, remaining, margin)
                result["last_runtime_forecast"] = dict(remaining_cases=remaining, forecast_seconds=forecast,
                    available_seconds=deadline - time.monotonic(), slowest_measured_seconds=max(measured))
                if forecast > deadline - time.monotonic():
                    result.update(status="requires_new_sealed_partition", passed=False,
                        not_reached=cases[index:],
                        reason="Prespecified remaining cohort does not fit; stop without dropping cases or changing this stage")
                    return result
            if time.monotonic() + margin >= deadline:
                raise TimeoutError("Independent native follow-up internal deadline reached")
            record_id = "ligand_" + case["identity_sha256"][:16] + "_s" + str(case["pose_seed"])
            directory = output / record_id
            directory.mkdir(exist_ok=False)
            yaml_path = directory / (record_id + ".yaml")
            schema = {"version": 1, "sequences": [
                {"protein": {"id": "A", "sequence": sequence, "msa": str(msa)}},
                {"ligand": {"id": "L", "smiles": case["identity"]}}],
                "properties": [{"affinity": {"binder": "L"}}]}
            yaml_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
            command = prediction_command(config["python"], yaml_path, directory / "native-output",
                cache_path, case["pose_seed"], assets, config["prediction"])
            run_receipt = dict(**case, command=command, input_sha256=_sha(yaml_path), status="running_not_passed")
            _write(directory / "prediction-receipt.json", run_receipt)
            child_started = time.monotonic()
            try:
                environment = os.environ.copy()
                environment["BOLTZ_CACHE"] = str(cache_path.resolve())
                with (directory / "native-cli.log").open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                        env=environment, text=True, timeout=max(1, min(
                            bootstrap["initial_case_seconds"] if index == 0 else 23 * 3600,
                            deadline - time.monotonic() - margin)), check=False)
                if completed.returncode != 0:
                    raise RuntimeError("Native fresh-pose child exited " + str(completed.returncode))
                native = validate_prediction_output(directory / "native-output", record_id, case["identity"],
                    (directory / "native-cli.log").read_text(encoding="utf-8", errors="replace"))
                run_receipt.update(status="passed_native_fresh_pose_prediction", native=native)
                result["completed"].append(dict(**case, prediction_receipt=str((directory / "prediction-receipt.json").resolve()),
                                               affinity=native["affinity"]))
            except (RuntimeError, ValueError, FileNotFoundError, subprocess.TimeoutExpired) as error:
                run_receipt.update(status="failed_native_fresh_pose_prediction", error_type=type(error).__name__, error=str(error))
                result["failed"].append(dict(**case, prediction_receipt=str((directory / "prediction-receipt.json").resolve()),
                                            error=str(error)))
            finally:
                duration = time.monotonic() - child_started
                measured.append(duration)
                run_receipt["elapsed_seconds"] = duration
                _write(directory / "prediction-receipt.json", run_receipt)
            _parent_cuda_uninitialized()
            if _own_bytes(output) > cap:
                raise RuntimeError("Independent follow-up artifact cap exceeded; preserve outputs and stop")
            if run_receipt["status"] != "passed_native_fresh_pose_prediction":
                result.update(status="failed_native_fresh_pose_case_remaining_not_reached", passed=False,
                              not_reached=cases[index + 1:])
                return result
        result.update(status="completed_native_fresh_pose_confirmation" if not result["failed"] else "failed_native_fresh_pose_cases",
                      passed=not result["failed"] and len(result["completed"]) == len(cases))
        return result
    except Exception as error:
        result.update(status="failed_not_confirmed", passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _parent_cuda_uninitialized()
        if "cache" in result:
            unchanged = all(Path(path).stat().st_size == result["cache"]["source_stats"][name]["size"]
                and Path(path).stat().st_mtime_ns == result["cache"]["source_stats"][name]["mtime_ns"]
                for name, path in result["cache"]["links"].items())
            result["original_assets_stat_unchanged"] = unchanged
            if not unchanged:
                result.update(status="failed_original_asset_changed", passed=False)
        own_bytes = _own_bytes(output)
        if own_bytes > cap:
            result.update(status="failed_own_artifact_cap", passed=False)
        result.update(elapsed_seconds=time.monotonic() - started, parent_cuda_initialized=False,
                      actual_own_artifact_bytes=own_bytes)
        _write(output / "followup-result.json", result)
