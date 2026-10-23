"""Continue a graph AE/VAE from an exact epoch boundary, then audit checkpoints.

This coordinator lives outside ``src/latentmol`` so an unchanged training
package can pass the checkpoint's algorithm-identity gate. It never chooses a
checkpoint using held-out test metrics: training and selection use validation,
the selection is written to disk, and only then are test evaluations run.
"""

import argparse
import json
from pathlib import Path

import torch

from latentmol.diagnostics import continuation_decision, save_learning_curves
from latentmol.evaluate import evaluate_checkpoint
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.full_training import _checkpoint_selection, _read_history
from latentmol.train import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--expected-resume-sha256", required=True)
    parser.add_argument("--expected-resume-epoch", type=int, required=True)
    parser.add_argument("--parent-history", type=Path, required=True)
    parser.add_argument("--neighborhood-limit", type=int, default=256)
    parser.add_argument("--prior-limit", type=int, default=1024)
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)
    if file_sha256(args.resume_from) != args.expected_resume_sha256:
        raise ValueError("Resume checkpoint SHA-256 differs from the declared parent")
    parent = torch.load(args.resume_from, map_location="cpu", weights_only=True)
    if parent["epoch"] != args.expected_resume_epoch or config.epochs <= parent["epoch"]:
        raise ValueError("Resume epoch or extension horizon differs from the plan")
    if parent["train_config"]["mode"] != config.mode:
        raise ValueError("Resume mode differs from the parent checkpoint")
    del parent
    parent_history = json.loads(args.parent_history.read_text())
    if [row["epoch"] for row in parent_history] != list(range(1, args.expected_resume_epoch + 1)):
        raise ValueError("Parent validation history is not complete through the resume epoch")

    options = {
        "config": str(args.config), "config_sha256": file_sha256(args.config),
        "data": str(args.data), "resume_from": str(args.resume_from),
        "resume_sha256": args.expected_resume_sha256,
        "resume_epoch": args.expected_resume_epoch,
        "parent_history": str(args.parent_history),
        "parent_history_sha256": file_sha256(args.parent_history),
        "target_epoch": config.epochs,
        "neighborhood_limit": args.neighborhood_limit,
        "prior_limit": args.prior_limit,
    }
    with ExperimentRun(args.run_root, f"{config.mode}-epoch-continuation", options, Path.cwd()) as run:
        trained = train(config, args.data, args.run_root, resume_from=args.resume_from)
        run.annotate(training_run=str(trained), test_evaluation_started=False)
        extension_history = _read_history(trained)
        expected = list(range(args.expected_resume_epoch + 1, config.epochs + 1))
        if [row["epoch"] for row in extension_history] != expected:
            raise RuntimeError("Continuation did not record each requested epoch exactly once")
        history = parent_history + extension_history
        write_json(run.path / "history.json", history)
        save_learning_curves(history, run.path)
        trend = continuation_decision(history, window=10)

        selected = _checkpoint_selection([
            ("best_validation_reconstruction", trained / "best.pt"),
            ("best_validation_validity", trained / "best_validity.pt"),
            ("last", trained / "last.pt"),
        ])
        write_json(run.path / "selection_before_test.json", {
            "checkpoints": selected,
            "selection_policy": "inherited-or-new validation reconstruction and validity winners, plus final epoch",
            "test_was_used_for_selection": False,
            "training_trend_at_target_epoch": trend,
        })
        run.annotate(final_selection=selected, test_evaluation_started=True)

        evaluations = []
        for choice in selected:
            checkpoint = Path(choice["path"])
            if file_sha256(checkpoint) != choice["sha256"]:
                raise RuntimeError("Selected checkpoint changed after validation selection was sealed")
            for split in ("val", "test"):
                evaluated = evaluate_checkpoint(
                    checkpoint, args.data, args.run_root, split=split,
                    limit=config.validation_limit, seed=config.seed, device=config.device,
                    novelty=True, chemistry_limit=config.validation_limit,
                    neighborhood_limit=args.neighborhood_limit,
                    prior_limit=args.prior_limit, shuffle_full=True,
                    batch_size=config.batch_size,
                )
                evaluations.append({"roles": choice["roles"], "epoch": choice["epoch"],
                                    "split": split, "run": str(evaluated),
                                    "checkpoint_sha256": choice["sha256"]})
                run.annotate(final_evaluations=evaluations)
        write_json(run.path / "summary.json", {
            "mode": config.mode, "parent_epoch": args.expected_resume_epoch,
            "completed_epoch": config.epochs, "training_run": str(trained),
            "parent_checkpoint_sha256": args.expected_resume_sha256,
            "trend": trend, "checkpoints": selected, "evaluations": evaluations,
            "limitation": "Extra training budget, not an equal-update AE/VAE comparison; validation trend is not proof of convergence.",
        })


if __name__ == "__main__":
    main()
