"""Staged AE/VAE training with validation-only continuation and sealed final tests.

The first stage provides a common compute comparison. One lower-LR stage always
follows, then a declared trend rule decides further extensions up to a fixed cap.
The training function resumes exact optimizer/RNG state between these stages.
All model work and plots must execute inside the user's HPC allocation.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from latentmol.diagnostics import continuation_decision, save_learning_curves
from latentmol.evaluate import evaluate_checkpoint
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.train import TrainConfig, train


def _read_history(path):
    """Read only explicit epoch events from a returned training directory."""
    return [row for line in (path / "metrics.jsonl").read_text().splitlines()
            if "epoch" in (row := json.loads(line))]


def _checkpoint_selection(candidates):
    """Fix roles and hashes before any test evaluation; deduplicate shared epochs."""
    unique = {}
    for role, path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        epoch = int(checkpoint["epoch"])
        key = (checkpoint["run_id"], epoch)
        if key not in unique:
            unique[key] = {"path": str(path), "sha256": file_sha256(path), "epoch": epoch, "roles": []}
        unique[key]["roles"].append(role)
        del checkpoint
    return list(unique.values())


def run_full_training(config: TrainConfig, data_path: Path, run_root: Path, *,
                      initial_epochs=60, extension_epochs=20, window=10,
                      final_limit=1000000, neighborhood_limit=256, prior_limit=1024):
    """Train, diagnose, freeze checkpoint choices, then evaluate the held-out test.

    ``config.epochs`` is the maximum cumulative horizon, not a request to repeat
    that many epochs at each phase. A stopped workflow records plateau versus
    budget exhaustion explicitly; neither outcome proves optimal training.
    """
    if initial_epochs < 1 or extension_epochs < 2 * window:
        raise ValueError("Initial stage must be positive; extension must cover both trend windows")
    if config.epochs < initial_epochs + extension_epochs:
        raise ValueError("Budget must include the common stage and a mandatory continuation check")
    if not config.train_probe_limit:
        raise ValueError("Staged convergence requires deterministic train and validation probes")
    options = {"train": asdict(config), "data": str(data_path), "initial_epochs": initial_epochs,
               "extension_epochs": extension_epochs, "window": window, "final_limit": final_limit,
               "neighborhood_limit": neighborhood_limit, "prior_limit": prior_limit}
    with ExperimentRun(run_root, f"{config.mode}-full-workflow", options, Path.cwd()) as run:
        history, stages, previous, comparison_checkpoint = [], [], None, None
        end_epoch = initial_epochs
        while True:
            run.annotate(active_phase={"target_epoch": end_epoch, "resume_from": str(previous) if previous else None},
                         test_evaluation_started=False)
            trained = train(config, data_path, run_root, resume_from=previous, stop_after_epoch=end_epoch)
            phase_history = _read_history(trained)
            history.extend(phase_history)
            if [row["epoch"] for row in history] != list(range(1, end_epoch + 1)):
                raise RuntimeError("Training phases did not produce a contiguous cumulative history")
            if comparison_checkpoint is None:
                comparison_checkpoint = trained / "last.pt"
            # Diagnostics share the process's CUDA context on exclusive GPUs.
            evaluation = evaluate_checkpoint(
                trained / "last.pt", data_path, run_root, split="val", limit=final_limit,
                seed=config.seed, device=config.device, novelty=True, chemistry_limit=final_limit,
                neighborhood_limit=neighborhood_limit, prior_limit=prior_limit, shuffle_full=True,
                batch_size=config.batch_size)
            trend = continuation_decision(history, window=window)
            if end_epoch == initial_epochs:
                decision = "mandatory_lower_learning_rate_check"
                continue_training = True
            elif end_epoch >= config.epochs:
                decision = "budget_reached_while_improving" if trend["extend"] else "budget_reached_without_recent_gain"
                continue_training = False
            else:
                decision = "extend_for_validation_gain" if trend["extend"] else "stop_after_validation_plateau"
                continue_training = trend["extend"]
            stage = {"end_epoch": end_epoch, "training_run": str(trained),
                     "validation_evaluation": str(evaluation), "decision": decision, "trend": trend}
            stages.append(stage)
            run.log({"stage": "continuation_decision", **stage})
            write_json(run.path / "history.json", history)
            write_json(run.path / "workflow_state.json", {"stages": stages, "history": history,
                                                           "last_checkpoint": str(trained / "last.pt")})
            save_learning_curves(history, run.path)
            run.annotate(stages=stages, completed_epoch=end_epoch, comparison_checkpoint=str(comparison_checkpoint))
            if not continue_training:
                break
            previous = trained / "last.pt"
            end_epoch = min(end_epoch + extension_epochs, config.epochs)

        selection = _checkpoint_selection([
            (f"common_epoch_{initial_epochs}", comparison_checkpoint),
            ("best_validation_reconstruction", trained / "best.pt"),
            ("best_validation_validity", trained / "best_validity.pt"),
            ("last", trained / "last.pt"),
        ])
        # Persist the immutable choice before the first test call. Test results
        # never affect phase continuation, checkpoint choice or LR schedules.
        write_json(run.path / "selection_before_test.json", {
            "checkpoints": selection, "final_training_decision": stages[-1]["decision"],
            "selection_policy": "common epoch, minimum full validation reconstruction, fixed-probe validity winner, last",
            "test_was_used_for_selection": False})
        run.annotate(test_evaluation_started=True, final_selection=selection)
        evaluations = []
        for selected in selection:
            checkpoint = Path(selected["path"])
            if file_sha256(checkpoint) != selected["sha256"]:
                raise RuntimeError("Selected checkpoint changed after selection was sealed")
            for split in ("val", "test"):
                evaluated = evaluate_checkpoint(
                    checkpoint, data_path, run_root, split=split, limit=final_limit, seed=config.seed,
                    device=config.device, novelty=True, chemistry_limit=final_limit,
                    neighborhood_limit=neighborhood_limit, prior_limit=prior_limit, shuffle_full=True,
                    batch_size=config.batch_size)
                evaluations.append({"roles": selected["roles"], "epoch": selected["epoch"],
                                    "split": split, "run": str(evaluated), "checkpoint_sha256": selected["sha256"]})
                run.annotate(final_evaluations=evaluations)
        summary = {"mode": config.mode, "completed_epochs": end_epoch, "planned_max_epochs": config.epochs,
                   "initial_comparison_epochs": initial_epochs, "stages": stages,
                   "checkpoints": selection, "evaluations": evaluations,
                   "needs_more_training_signal": stages[-1]["trend"]["extend"],
                   "stopping_interpretation": stages[-1]["decision"],
                   "limitation": "A plateau under this schedule is not proof of global convergence or useful chemistry."}
        write_json(run.path / "summary.json", summary)
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-epochs", type=int, default=60)
    parser.add_argument("--extension-epochs", type=int, default=20)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--final-limit", type=int, default=1000000)
    parser.add_argument("--neighborhood-limit", type=int, default=256)
    parser.add_argument("--prior-limit", type=int, default=1024)
    args = parser.parse_args()
    run_full_training(TrainConfig.from_yaml(args.config), args.data, args.run_root,
                      initial_epochs=args.initial_epochs, extension_epochs=args.extension_epochs,
                      window=args.window, final_limit=args.final_limit,
                      neighborhood_limit=args.neighborhood_limit, prior_limit=args.prior_limit)


if __name__ == "__main__":
    main()
