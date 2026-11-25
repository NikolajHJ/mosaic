"""Small workflow entrypoint: train, then explicitly evaluate known checkpoints.

No directory glob or 'latest run' selection is used. The returned training path
is passed directly to evaluation so simultaneous jobs cannot mix artifacts.
Evaluation shares the training process's CUDA context for exclusive-process GPUs.
"""

import argparse
from pathlib import Path

from latentmol.evaluate import evaluate_checkpoint
from latentmol.train import TrainConfig, train


def train_and_evaluate(config: TrainConfig, data_path: Path, run_root: Path,
                       smoke: bool = False) -> Path:
    """Train once and evaluate explicit checkpoints without a second CUDA process.

    Each evaluator invocation creates its own complete experiment record. Failures
    propagate after that evaluator records its traceback, so an enclosing command
    runner still records overall workflow failure. Return the exact training run
    path, never a directory inferred from modification times or names.
    """
    if smoke and config.train_limit is None:
        raise ValueError("Smoke evaluation requires an explicit train_limit")
    run_path = train(config, data_path, run_root)
    jobs = [("best.pt", "val", config.validation_limit), ("last.pt", "val", config.validation_limit)]
    if smoke:
        jobs.append(("last.pt", "train", config.train_limit))
    for checkpoint, split, limit in jobs:
        evaluate_checkpoint(run_path / checkpoint, data_path, run_root, split=split,
                            limit=limit, seed=config.seed, device=config.device)
    return run_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Also evaluate the exact memorization subset")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    train_and_evaluate(config, args.data, args.run_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
