"""One allocation, sequential native contexts and unfiltered VAE prior starts.

This orchestrator never holds a GPU model. Each native preparation, optimization
and fresh-pose evaluation exits before the next child starts. Scientific failures
remain study outcomes; provenance failures stop the job. New real-model workflow:
requires HPC validation, and may need refinement after the first execution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import re
import sys

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.initialization import LatentStart
from latentmol.multimodel.study import verified_file
from .active_set import ActiveSetConfig
from .initialization_report import read_json, write_report
from .prepare import EXAMPLE_ASSET_SHA256, installed_boltz_provenance, seal_count_catalog
from .workflow_process import require_child, run_logged


def load_protocol(path: Path) -> dict:
    """Resolve a strict manifest; configurations cannot introduce shell commands."""
    value = yaml.safe_load(path.read_text())
    required = {"version", "name", "output_root", "base_config", "cache", "example_assets",
                "native_template", "confidence_sha256", "seeds", "counts", "fd_epsilons",
                "reference_pilot", "preparation_timeout_seconds", "optimization_timeout_seconds",
                "rescore_timeout_seconds"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Study protocol must contain exactly the documented fields")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unknown study protocol version")
    if not isinstance(value["name"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value["name"]):
        raise ValueError("Invalid study name")
    for field in ("seeds", "counts"):
        if (not isinstance(value[field], list) or not value[field]
                or any(type(x) is not int for x in value[field])
                or len(set(value[field])) != len(value[field])):
            raise ValueError(f"{field} must be a nonempty list of distinct integers")
    for seed in value["seeds"]:
        LatentStart("standard_normal", seed)
    if any(not 1 <= count <= 50 for count in value["counts"]):
        raise ValueError("Requested count outside the current 50-slot schema")
    for field in ("preparation_timeout_seconds", "optimization_timeout_seconds", "rescore_timeout_seconds"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ValueError("Timeouts must be positive integer seconds")
    resolve = lambda name: str((path.resolve().parent / name).resolve())
    for field in ("base_config", "native_template"):
        asset = value[field]
        if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
                or not isinstance(asset["path"], str)
                or not isinstance(asset["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])):
            raise ValueError(f"{field} requires a path and SHA256")
        value[field] = {**asset, "path": resolve(asset["path"])}
    for field in ("output_root", "cache", "example_assets", "reference_pilot"):
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"{field} requires a nonempty path")
        value[field] = resolve(value[field])
    if not isinstance(value["confidence_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["confidence_sha256"]):
        raise ValueError("Missing confidence checkpoint SHA256")
    # Reuse the production derivative policy's validation, without changing its
    # tolerances or minimum pass count. Extra scales are explicitly declared.
    from latentmol.multimodel.config import Diagnostics
    Diagnostics(fd_epsilons=tuple(value["fd_epsilons"]))
    return value


def verify_assets(protocol: dict, base: ActiveSetConfig, source_sha: str) -> None:
    """Shared integrity errors are fatal, rather than ten misleading seed failures."""
    if source_manifest(Path.cwd())["sha256"] != source_sha:
        raise ValueError("Study executable source changed")
    for asset in (protocol["base_config"], protocol["native_template"], base.decoder,
                  base.affinity_checkpoint, base.context_manifest):
        if asset:
            verified_file(asset["path"], asset["sha256"])
    cache = Path(protocol["cache"])
    verified_file(cache / "boltz2_conf.ckpt", protocol["confidence_sha256"])
    verified_file(cache / "boltz2_aff.ckpt", base.affinity_checkpoint["sha256"])
    for asset in base.contexts.values():
        verified_file(asset["path"], asset["sha256"])
    for filename, digest in EXAMPLE_ASSET_SHA256.items():
        verified_file(Path(protocol["example_assets"]) / filename, digest)


def prepare_contexts(protocol: dict, base: ActiveSetConfig, destination: Path) -> Path:
    """Prepare every missing count serially, then seal once before optimization.

    Reused snapshots remain at their original immutable paths. Individual native
    preparation failures are recorded and omitted, never represented by a carbon
    graph of a different size. A later selection of that count fails explicitly.
    """
    destination.mkdir(parents=True, exist_ok=False)
    states = {str(n): {"status": "not_attempted"} for n in protocol["counts"]}
    prepared = {}
    write_json(destination / "preparation.json", states)
    for count in protocol["counts"]:
        state = states[str(count)]
        if count in base.contexts:
            asset = base.contexts[count]
            prepared[count] = verified_file(asset["path"], asset["sha256"])
            state.update(status="reused", **asset)
        else:
            state.update(status="running", commands=[])
            write_json(destination / "preparation.json", states)
            root = destination / f"count-{count:02d}"
            input_dir, prediction_dir = root / "input", root / "prediction"
            snapshot = root / "prepared.pt"
            stem = f"carbon_chain_{count:02d}"
            commands = [
                [sys.executable, "-m", "latentmol.boltz.prepare", "chain-count-example",
                 "--output", input_dir, "--count", count, "--assets-dir", protocol["example_assets"]],
                [str(Path(sys.executable).parent / "boltz"), "predict", input_dir / f"{stem}.yaml",
                 "--cache", protocol["cache"], "--out_dir", prediction_dir, "--recycling_steps", "0",
                 "--sampling_steps", "20", "--diffusion_samples", "1", "--sampling_steps_affinity", "20",
                 "--diffusion_samples_affinity", "1", "--num_workers", "0", "--preprocessing-threads", "1",
                 "--no_kernels", "--seed", "17", "--subsample_msa", "--num_subsampled_msa", "128"],
                [sys.executable, "-m", "latentmol.boltz.prepare", "snapshot",
                 "--prediction-dir", prediction_dir / f"boltz_results_{stem}", "--cache", protocol["cache"],
                 "--output", snapshot, "--ae-checkpoint", base.decoder["path"],
                 "--input-manifest", input_dir / "input_manifest.json"],
            ]
            for stage, command in enumerate(commands):
                result = run_logged(command, root / f"stage-{stage}.log",
                                    timeout_seconds=protocol["preparation_timeout_seconds"])
                state["commands"].append(result)
                write_json(destination / "preparation.json", states)
                if result["returncode"] is None:
                    raise RuntimeError("Cannot launch preparation executable; see retained command log")
                if result["returncode"] != 0 or result["timed_out"]:
                    state.update(status="failed", failed_stage=stage)
                    break
            else:
                if not snapshot.is_file():
                    raise ValueError("Successful preparation child did not produce a snapshot")
                prepared[count] = snapshot
                state.update(status="prepared", path=str(snapshot), sha256=file_sha256(snapshot))
        write_json(destination / "preparation.json", states)
        print(f"CONTEXT={count} STATUS={state['status']}", flush=True)
    return seal_count_catalog(prepared, destination / "contexts.json",
                              failed_counts={n: row for n, row in states.items() if row["status"] == "failed"})


def trial_config(base_path: Path, catalog: Path, root: Path, seed: int, epsilons: list) -> Path:
    """Only initialization, context coverage, output identity and FD scales vary."""
    config = deepcopy(yaml.safe_load(base_path.read_text()))
    base = ActiveSetConfig.load(base_path)
    config.update(name=f"vae-random-{seed:04d}", output_root=str(root / "pilot"),
                  start={"kind": "standard_normal", "seed": seed},
                  context_manifest={"path": str(catalog), "sha256": file_sha256(catalog)},
                  decoder=base.decoder, affinity_checkpoint=base.affinity_checkpoint)
    config.pop("contexts", None)
    config["diagnostics"]["fd_epsilons"] = epsilons
    root.mkdir(parents=True, exist_ok=False)
    path = root / "pilot.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    ActiveSetConfig.load(path)  # Validate before any child model execution.
    return path


def child_path(result: dict, expected_root: Path, source_sha: str) -> Path | None:
    """Inspect even failed children, but never accept a path from another release."""
    if not result["run_paths"]:
        return None
    if len(result["run_paths"]) != 1:
        raise ValueError("Unexpected nested/duplicate child experiment")
    path = Path(result["run_paths"][0]).resolve()
    if path.parent != expected_root.resolve():
        raise ValueError("Child experiment escaped its assigned output root")
    record = read_json(path / "run.json")
    if (record["source"]["sha256"] != source_sha
            or record.get("source_archive_sha256") != os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Child source identity differs from the study")
    return path


def run_trial(protocol: dict, base_path: Path, catalog: Path, root: Path,
              seed: int, source_sha: str, persist) -> dict:
    """One draw, paired optimizer child, then optional fresh-native child.

    The existing paired runner evaluates the active arm first. If that fails its
    fixed control is not attempted; this is visible in the report, never silently
    replaced by another seed. The callback saves child identity before rescoring.
    """
    config_path = trial_config(base_path, catalog, root, seed, protocol["fd_epsilons"])
    trial = {"seed": seed, "status": "optimizing", "config": str(config_path),
             "config_sha256": file_sha256(config_path)}
    persist(trial)
    result = run_logged([sys.executable, "-m", "latentmol.boltz.active_set", "--config", config_path],
                        root / "optimization.log", timeout_seconds=protocol["optimization_timeout_seconds"])
    trial["optimization_process"] = result
    if result["returncode"] is None:
        trial["status"] = "infrastructure_failed"
        persist(trial)
        raise RuntimeError("Cannot launch optimizer executable; see retained command log")
    pilot = child_path(result, root / "pilot", source_sha)
    if pilot is None and not result["timed_out"]:
        trial["status"] = "infrastructure_failed"
        persist(trial)
        raise RuntimeError("Optimizer failed before creating its experiment record; inspect its log")
    if pilot:
        trial["pilot"] = str(pilot)
        record = read_json(pilot / "run.json")
        if (record["config"]["config_file_sha256"] != trial["config_sha256"]
                or record["config"]["start"] != {"kind": "standard_normal", "seed": seed}):
            raise ValueError("Child initialization/config identity differs from declared seed")
        if record.get("error") == "Frozen decoder or Boltz model state changed":
            trial["status"] = "infrastructure_failed"
            persist(trial)
            raise ValueError("A frozen model changed during optimization; stop the whole study")
    if result["returncode"] != 0 or result["timed_out"]:
        trial["status"] = "optimization_failed"
        persist(trial)
        return trial
    require_child(result, experiment=True)
    trial["status"] = "rescoring"
    persist(trial)
    result = run_logged([sys.executable, "-m", "latentmol.boltz.active_set_rescore", "--pilot", pilot,
                         "--template", protocol["native_template"]["path"], "--cache", protocol["cache"],
                         "--run-root", root / "rescore"], root / "rescore.log",
                        timeout_seconds=protocol["rescore_timeout_seconds"])
    trial["rescore_process"] = result
    if result["returncode"] is None:
        trial["status"] = "infrastructure_failed"
        persist(trial)
        raise RuntimeError("Cannot launch native rescore executable; see retained command log")
    rescore = child_path(result, root / "rescore", source_sha)
    if rescore:
        trial["rescore"] = str(rescore)
        record = read_json(rescore / "run.json")
        if record["config"]["parent_run_id"] != pilot.name:
            raise ValueError("Fresh-native evaluation refers to a different pilot")
    if result["returncode"] != 0 or result["timed_out"]:
        trial["status"] = "rescore_failed"
    else:
        require_child(result, experiment=True)
        trial["status"] = "completed"
    persist(trial)
    return trial


def run(protocol_path: Path, expected_sha256: str) -> Path:
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an HPC allocation and immutable source archive")
    verified_file(protocol_path, expected_sha256)
    protocol = load_protocol(protocol_path)
    verified_file(protocol["base_config"]["path"], protocol["base_config"]["sha256"])
    base = ActiveSetConfig.load(Path(protocol["base_config"]["path"]))
    if base.mode != "vae" or base.start.kind != "encoded_seed":
        raise ValueError("The reference configuration must describe an encoded VAE start")
    source_sha = source_manifest(Path.cwd())["sha256"]
    if read_json(Path.cwd() / "release_manifest.json")["sha256"] != source_sha:
        raise ValueError("Source directory differs from the sealed release manifest")
    with ExperimentRun(Path(protocol["output_root"]), protocol["name"],
                       {**protocol, "protocol_sha256": expected_sha256}, Path.cwd()) as experiment:
        trials = [{"seed": seed, "status": "not_attempted"} for seed in protocol["seeds"]]
        reference = Path(protocol["reference_pilot"])
        # The C13 control is descriptive, not a source of selected prior draws.
        # Its exact original config must match; failed/absent controls stay absent.
        if (reference / "run.json").exists():
            original = read_json(reference / "run.json")
            if original["config"].get("config_file_sha256") != protocol["base_config"]["sha256"]:
                raise ValueError("C13 reference does not use the declared baseline configuration")
        def persist(trial=None):
            if trial is not None:
                index = protocol["seeds"].index(trial["seed"])
                trials[index] = dict(trial)
            write_json(experiment.path / "trials.json", trials)
            write_report(experiment.path, trials, reference=reference)
        persist()
        verify_assets(protocol, base, source_sha)
        from .active_set_rescore import _target_identity
        if base.context_identity is None:
            raise ValueError("The study requires a sealed target/MSA context catalog")
        native_identity = _target_identity(Path(protocol["native_template"]["path"]),
                                            base.context_identity["target_sequence_sha256"],
                                            base.context_identity["msa_sha256"])
        native_identity.pop("template")
        experiment.annotate(native_provenance=installed_boltz_provenance(),
                            template_identity=native_identity)
        catalog = prepare_contexts(protocol, base, experiment.path / "contexts")
        verify_assets(protocol, base, source_sha)
        catalog_sha = file_sha256(catalog)
        sealed = read_json(catalog)
        if set(map(int, sealed["contexts"])) | set(map(int, sealed["failed_counts"])) != set(protocol["counts"]):
            raise ValueError("Sealed preparation does not account for every requested count")
        experiment.annotate(context_catalog=str(catalog), context_catalog_sha256=catalog_sha)
        def verify_trial_assets():
            verify_assets(protocol, base, source_sha)
            verified_file(catalog, catalog_sha)
            for asset in read_json(catalog)["contexts"].values():
                verified_file(catalog.parent / asset["path"], asset["sha256"])
        for seed in protocol["seeds"]:
            verify_trial_assets()
            experiment.log({"phase": "seed_start", "seed": seed})
            trial = run_trial(protocol, Path(protocol["base_config"]["path"]), catalog,
                              experiment.path / "seeds" / f"seed-{seed:04d}", seed, source_sha, persist)
            # Include the last/failed child: its asset-integrity failure must
            # never be mislabeled as an ordinary scientific seed outcome.
            verify_trial_assets()
            experiment.log({"phase": "seed_finished", "seed": seed, "status": trial["status"]})
        persist()
        experiment.annotate(all_planned_seeds_processed=True,
                            successful_seed_workflows=sum(t["status"] == "completed" for t in trials),
                            claim_limit="Workflow completion does not mean ten valid molecules or ten passed gates")
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    print(run(args.config, args.sha256), flush=True)


if __name__ == "__main__":
    main()
