"""Run fixed-z diagnostics on an exactly identified full-workflow checkpoint.

This is for an LSF job depending on one completed parent full-training job.
The parent is selected by exact scheduler ID, source archive, mode and decoder
variant, never by directory recency. The checkpoint must have been sealed before
test evaluation and selected using validation reconstruction only.
"""

import argparse
import json
from pathlib import Path

import torch

from latentmol.architecture_study import run_study
from latentmol.experiments import file_sha256


ARCHITECTURE_VARIANTS = ("residual", "pair_refine", "deep_pair_refine",
                         "global_bond", "wide_global_bond")


def selected_checkpoint(run_root: Path, *, parent_job_id: str,
                        parent_source_sha256: str, mode: str, variant: str) -> Path:
    """Resolve one validation-selected checkpoint from a completed parent job."""
    if mode not in ("ae", "vae") or variant not in ARCHITECTURE_VARIANTS:
        raise ValueError("Unknown mode or decoder variant")
    parents = []
    for directory in run_root.iterdir():
        if not directory.is_dir() or f"_{mode}-full-workflow_" not in directory.name:
            continue
        manifest = directory / "run.json"
        if not manifest.is_file():
            continue
        record = json.loads(manifest.read_text())
        if (record.get("environment", {}).get("job_id") == parent_job_id
                and record.get("source_archive_sha256") == parent_source_sha256
                and record.get("name") == f"{mode}-full-workflow"
                and record.get("config", {}).get("train", {}).get("model", {}).get("decoder_variant") == variant):
            parents.append((directory, record))
    if len(parents) != 1:
        raise RuntimeError(f"Expected one exact parent workflow, found {len(parents)}")
    directory, record = parents[0]
    if record.get("status") != "completed":
        raise RuntimeError("Parent workflow did not complete successfully")
    sealed = json.loads((directory / "selection_before_test.json").read_text())
    matches = [entry for entry in sealed["checkpoints"]
               if "best_validation_reconstruction" in entry["roles"]]
    if len(matches) != 1 or sealed.get("test_was_used_for_selection") is not False:
        raise RuntimeError("Validation-selected checkpoint is missing or selection is unsealed")
    checkpoint = Path(matches[0]["path"]).resolve()
    stages = record.get("stages", [])
    if not stages:
        raise RuntimeError("Parent has no recorded training stages")
    final_training = Path(stages[-1]["training_run"]).resolve()
    if not final_training.is_relative_to(run_root.resolve()):
        raise RuntimeError("Final training stage lies outside the declared run root")
    if checkpoint != final_training / "best.pt":
        raise RuntimeError("Selected checkpoint is not the final recorded stage's best.pt")
    if file_sha256(checkpoint) != matches[0]["sha256"]:
        raise RuntimeError("Validation-selected checkpoint hash changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stage_ids = {Path(stage["training_run"]).resolve().name for stage in stages}
    # The final-stage best.pt may carry an inherited earlier-stage winner.
    # Normalize tuples because JSON stores lr_milestones as nested lists.
    checkpoint_config = json.loads(json.dumps(payload.get("train_config")))
    if (payload.get("source_archive_sha256") != parent_source_sha256
            or checkpoint_config != record["config"]["train"]
            or payload.get("train_config", {}).get("mode") != mode
            or payload.get("model_config", {}).get("decoder_variant") != variant
            or payload.get("run_id") not in stage_ids
            or payload.get("epoch") != matches[0]["epoch"]):
        raise RuntimeError("Selected checkpoint provenance differs from parent workflow")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--parent-source-sha256", required=True)
    parser.add_argument("--mode", required=True, choices=("ae", "vae"))
    parser.add_argument("--variant", required=True, choices=ARCHITECTURE_VARIANTS)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--molecules", type=int, default=128)
    parser.add_argument("--draws-per-molecule", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    checkpoint = selected_checkpoint(args.run_root, parent_job_id=args.parent_job_id,
                                     parent_source_sha256=args.parent_source_sha256,
                                     mode=args.mode, variant=args.variant)
    print(f"VALIDATION_SELECTED_CHECKPOINT={checkpoint}", flush=True)
    run_study(checkpoint, args.data, args.run_root, split="val", seed=17,
              molecules=args.molecules, draws_per_molecule=args.draws_per_molecule,
              temperature=1.0, device=args.device)


if __name__ == "__main__":
    main()
