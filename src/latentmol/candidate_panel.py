"""Seal explicit decoder checkpoints and a common cohort before new scoring.

Each checkpoint role was selected in its parent workflow before that workflow's
test evaluation. Historical test results may already be known when a *panel*
is assembled; the plan must disclose its rationale and prior test exposure.
This module never ranks candidates by any metric. It records a shared
validation cohort for later AE/VAE and frozen-model arms.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import GraphSpec


SELECTION_ROLES = frozenset({"best_validation_reconstruction",
                             "best_validation_validity", "common_epoch_60"})
PARENT_NAMES = frozenset({"ae-full-workflow", "vae-full-workflow",
                          "ae-epoch-continuation", "vae-epoch-continuation"})


def _inside(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"Declared HPC artifact does not exist: {path}") from exc
    if not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError(f"Artifact lies outside the declared HPC run root: {resolved}")
    return resolved


def _same_json(left: dict, right: dict) -> bool:
    """Normalize tuple/list differences from saved YAML and JSON metadata."""
    return json.loads(json.dumps(left)) == json.loads(json.dumps(right))


def _verify_lineage(parent: dict, checkpoint: Path, payload: dict,
                    role: str, *, run_root: Path, mode: str, variant: str) -> None:
    """Tie a selected alias to its completed full workflow or exact continuation.

    An inherited continuation winner keeps its older training-run/source IDs.
    The new training run records the copied inherited checkpoint's exact hash,
    so accepting that older identity requires the matching inheritance receipt.
    """
    name = parent["name"]
    if name.endswith("-full-workflow"):
        stages = parent.get("stages")
        if not stages or not _same_json(payload["train_config"], parent["config"]["train"]):
            raise ValueError("Full-workflow training configuration or stages differ")
        if (parent["config"]["train"]["model"]["decoder_variant"] != variant
                or parent["config"]["train"]["mode"] != mode
                or payload.get("source_archive_sha256") != parent.get("source_archive_sha256")):
            raise ValueError("Full-workflow model or source differs")
        training = [_inside(Path(stage["training_run"]), run_root) for stage in stages]
        suffix = {"best_validation_reconstruction": "best.pt",
                  "best_validation_validity": "best_validity.pt"}
        expected = (training[-1] / suffix[role] if role in suffix
                    else _inside(Path(parent["comparison_checkpoint"]), run_root))
        if checkpoint != expected or payload.get("run_id") not in {path.name for path in training}:
            raise ValueError("Full-workflow checkpoint is not its sealed stage output")
        return

    if role == "common_epoch_60":
        raise ValueError("A continuation does not select a new common-epoch checkpoint")
    config = parent["config"]
    training = _inside(Path(parent["training_run"]), run_root)
    training_record = json.loads((training / "run.json").read_text())
    if (training_record.get("status") != "completed"
            or training_record.get("name") != f"{mode}-train"
            or not _same_json(training_record.get("config"), payload["train_config"])
            or training_record.get("graph_spec") != payload["graph_spec"]
            or config.get("resume_epoch", -1) >= config.get("target_epoch", -1)):
        raise ValueError("Continuation training run or epoch horizon differs")
    suffix = ("best.pt" if role == "best_validation_reconstruction"
              else "best_validity.pt")
    if checkpoint != training / suffix:
        raise ValueError("Continuation winner is not in its recorded training run")
    resume = training_record.get("resume", {})
    if (resume.get("checkpoint") != config.get("resume_from")
            or resume.get("sha256") != config.get("resume_sha256")
            or resume.get("completed_epoch") != config.get("resume_epoch")):
        raise ValueError("Continuation ancestry differs from its resume receipt")
    if payload["epoch"] > config["resume_epoch"]:
        if (payload.get("run_id") != training.name
                or payload.get("source_archive_sha256") != parent.get("source_archive_sha256")):
            raise ValueError("New continuation winner has another training identity")
    else:
        key = ("reconstruction" if role == "best_validation_reconstruction" else "validity")
        inherited = resume.get("inherited_selections", {}).get(key, {})
        if (inherited.get("epoch") != payload["epoch"]
                or inherited.get("checkpoint_sha256") != file_sha256(checkpoint)
                or inherited.get("path") != str(training / f"inherited_best_{key}.pt")
                or file_sha256(_inside(Path(inherited["path"]), run_root)) != file_sha256(checkpoint)):
            raise ValueError("Inherited continuation winner lacks its exact copy receipt")


def _candidate(request: dict, *, run_root: Path, data: dict) -> dict:
    required = {"id", "mode", "variant", "role", "selection_path",
                "selection_sha256", "checkpoint_sha256"}
    if set(request) != required:
        raise ValueError(f"Candidate fields differ from the contract: {sorted(set(request) ^ required)}")
    if request["mode"] not in ("ae", "vae") or request["role"] not in SELECTION_ROLES:
        raise ValueError("Candidate mode or validation-only role is unsupported")
    if not request["id"] or not request["variant"]:
        raise ValueError("Candidate id and decoder variant are required")
    selection = _inside(Path(request["selection_path"]), run_root)
    if selection.name != "selection_before_test.json":
        raise ValueError("A candidate must use the sealed pre-test selection file")
    if file_sha256(selection) != request["selection_sha256"]:
        raise ValueError("Selection-file hash differs from the predeclared plan")
    parent = json.loads((selection.parent / "run.json").read_text())
    if (parent.get("name") not in PARENT_NAMES
            or not parent["name"].startswith(f'{request["mode"]}-')
            or parent.get("status") != "completed"):
        raise ValueError("Selection is not from a completed matching training workflow")
    sealed = json.loads(selection.read_text())
    if sealed.get("test_was_used_for_selection") is not False:
        raise ValueError("Checkpoint selection must explicitly exclude held-out test results")
    matches = [entry for entry in sealed.get("checkpoints", [])
               if request["role"] in entry.get("roles", [])]
    if len(matches) != 1:
        raise ValueError("Validation role does not identify exactly one checkpoint")
    entry = matches[0]
    checkpoint = _inside(Path(entry["path"]), run_root)
    digest = file_sha256(checkpoint)
    if digest != request["checkpoint_sha256"] or digest != entry["sha256"]:
        raise ValueError("Checkpoint hash differs from the plan or sealed selection")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload.get("format_version") != 1 or payload.get("epoch") != entry["epoch"]
            or payload.get("train_config", {}).get("mode") != request["mode"]
            or payload.get("model_config", {}).get("decoder_variant") != request["variant"]
            or payload.get("dataset_sha256") != data["dataset_sha256"]
            or payload.get("continuation", {}).get("split_sha256") != data["split_sha256"]
            or payload.get("graph_spec") != data["spec"]):
        raise ValueError("Checkpoint mode, epoch, variant, data or schema differs")
    _verify_lineage(parent, checkpoint, payload, request["role"], run_root=run_root,
                    mode=request["mode"], variant=request["variant"])
    return {"id": request["id"], "mode": request["mode"],
            "variant": request["variant"], "selection_role": request["role"],
            "selection_path": str(selection),
            "selection_sha256": request["selection_sha256"],
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest,
            "checkpoint_epoch": entry["epoch"],
            "training_source_archive_sha256": payload.get("source_archive_sha256"),
            "training_algorithm_sha256": payload["continuation"]["algorithm"]["sha256"],
            "latent_dimension": payload["model_config"]["latent"],
            "parent_run_id": parent["run_id"], "parent_run_name": parent["name"]}


def seal_panel(plan_path: Path, data_manifest_path: Path, run_root: Path) -> Path:
    """Validate exact artifact identities and write one immutable study panel.

    Only a validation cohort is allocated. Its indices refer to the ordered
    ``GraphDataset(..., 'val')`` split under the recorded dataset and split
    hashes. No test summaries or external scoring outputs are opened here.
    """
    plan = json.loads(plan_path.read_text())
    if set(plan) != {"format_version", "primary_candidate_id", "candidates", "cohort",
                     "selection_rationale", "historical_test_exposure"}:
        raise ValueError("Candidate-panel plan has unknown or missing fields")
    if plan["format_version"] != 1 or not plan["candidates"]:
        raise ValueError("Unsupported or empty candidate-panel plan")
    if (not isinstance(plan["selection_rationale"], str)
            or not plan["selection_rationale"].strip()
            or not isinstance(plan["historical_test_exposure"], bool)):
        raise ValueError("Panel rationale and historical test exposure are required")
    cohort = plan["cohort"]
    if set(cohort) != {"split", "seed", "molecules"} or cohort["split"] != "val":
        raise ValueError("Candidate selection may allocate only a validation cohort")
    data = json.loads(data_manifest_path.read_text())
    if GraphSpec.from_dict(data["spec"]).fingerprint() != data["spec_fingerprint"]:
        raise ValueError("Data manifest graph schema fingerprint differs")
    count, seed = cohort["molecules"], cohort["seed"]
    if (not isinstance(count, int) or isinstance(count, bool) or count < 1
            or count > data["counts"]["val"]
            or not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError("Invalid validation cohort count or seed")
    candidates = [_candidate(item, run_root=run_root, data=data)
                  for item in plan["candidates"]]
    ids = [candidate["id"] for candidate in candidates]
    if len(ids) != len(set(ids)) or ids.count(plan["primary_candidate_id"]) != 1:
        raise ValueError("Candidate ids must be unique and identify one primary")
    indices = random.Random(seed).sample(range(data["counts"]["val"]), count)
    panel = {"format_version": 1, "primary_candidate_id": plan["primary_candidate_id"],
             "candidates": candidates,
             "selection_rationale": plan["selection_rationale"],
             "historical_test_exposure": plan["historical_test_exposure"],
             "graph_spec_sha256": data["spec_fingerprint"],
             "dataset_sha256": data["dataset_sha256"],
             "split_sha256": data["split_sha256"],
             "data_manifest_path": str(data_manifest_path.resolve()),
             "data_manifest_sha256": file_sha256(data_manifest_path),
             "cohort": {**cohort, "indices": indices},
             "selection_policy": "Parent checkpoint roles sealed before their tests; panel membership explicitly declared after historical result exposure",
             "score_policy": "No Boltz or chemistry-prior call during panel sealing"}
    options = {"plan_path": str(plan_path.resolve()), "plan_sha256": file_sha256(plan_path),
               "data_manifest_path": str(data_manifest_path.resolve()),
               "data_manifest_sha256": file_sha256(data_manifest_path),
               "primary_candidate_id": plan["primary_candidate_id"],
               "candidate_count": len(candidates), "validation_molecules": count}
    with ExperimentRun(run_root, "decoder-candidate-panel", options, Path.cwd()) as run:
        write_json(run.path / "panel.json", panel)
        run.log({"stage": "panel_sealed", "candidate_ids": ids,
                 "panel_sha256": file_sha256(run.path / "panel.json")})
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    print(seal_panel(args.plan, args.data_manifest, args.run_root))


if __name__ == "__main__":
    main()
