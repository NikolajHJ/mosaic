"""Run matched hard-graph diagnostics on a previously sealed validation panel.

All numerics run in an HPC allocation. This driver verifies panel/checkpoint/data
identity, invokes the reusable fixed-z study unchanged for each candidate, and
collects chemistry profiles without making a new architecture selection. It
performs no repair, external scoring, gradients, or training.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .architecture_study import run_study
from .experiments import ExperimentRun, file_sha256, write_json
from .quality import QualityScoreConfig


def run_panel(panel_path: Path, panel_sha256: str, data: Path, run_root: Path, *,
              draws_per_molecule: int = 16, temperature: float = 1.0,
              device: str = "cpu", quality_config: QualityScoreConfig | None = None) -> Path:
    """Use the sealed cohort verbatim and expose separate argmax/categorical scores."""
    panel_path, data, run_root = map(Path, (panel_path, data, run_root))
    quality_config = quality_config or QualityScoreConfig()
    config = {"panel": str(panel_path), "panel_sha256": panel_sha256,
              "data": str(data), "draws_per_molecule": draws_per_molecule,
              "temperature": temperature, "device": device,
              "quality_weights": asdict(quality_config)}
    with ExperimentRun(run_root, "graded-quality-panel", config, Path.cwd()) as run:
        if file_sha256(panel_path) != panel_sha256:
            raise ValueError("Sealed panel hash differs")
        panel = json.loads(panel_path.read_text())
        if panel.get("format_version") != 1 or panel["cohort"]["split"] != "val":
            raise ValueError("Expected a sealed validation panel")
        if file_sha256(data / "manifest.json") != panel["data_manifest_sha256"]:
            raise ValueError("Data manifest differs from the sealed panel")
        cohort = panel["cohort"]
        if len(cohort["indices"]) != cohort["molecules"]:
            raise ValueError("Panel cohort budget mismatch")
        candidates = panel["candidates"]
        if not candidates or len({c["id"] for c in candidates}) != len(candidates):
            raise ValueError("Panel candidates must be nonempty and uniquely identified")
        # Validate every checkpoint before starting either arm; no accidental
        # partial comparison because one candidate's path was replaced.
        for candidate in candidates:
            if file_sha256(Path(candidate["checkpoint_path"])) != candidate["checkpoint_sha256"]:
                raise ValueError(f"Checkpoint hash changed: {candidate['id']}")
        completed = []
        for candidate in candidates:
            child = run_study(Path(candidate["checkpoint_path"]), data, run_root,
                              split="val", seed=cohort["seed"], molecules=cohort["molecules"],
                              draws_per_molecule=draws_per_molecule, temperature=temperature,
                              device=device, source_indices=cohort["indices"], quality_config=quality_config)
            summary = json.loads((child / "summary.json").read_text())
            completed.append({"candidate": candidate, "run": str(child),
                              "summary_sha256": file_sha256(child / "summary.json"),
                              "scores": {mode: summary["results"][mode]["graded_chemistry"]
                                         for mode in ("argmax", "categorical")}})
            run.annotate(completed_candidates=completed)
        write_json(run.path / "comparison.json", {
            "panel_sha256": panel_sha256, "cohort": cohort,
            "evaluation_budgets_matched": True, "training_budgets_matched": False,
            "historical_test_exposure": panel["historical_test_exposure"],
            "selection_rationale": panel["selection_rationale"],
            "repair_calls": 0, "native_scorer_calls": 0, "candidates": completed,
            "interpretation": "Matched validation chemistry diagnostics; no final architecture selection"})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--draws-per-molecule", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--quality-config", type=Path)
    args = parser.parse_args()
    run_panel(args.panel, args.panel_sha256, args.data, args.run_root,
              draws_per_molecule=args.draws_per_molecule, temperature=args.temperature,
              device=args.device, quality_config=QualityScoreConfig.from_dict(json.loads(args.quality_config.read_text()))
              if args.quality_config else None)


if __name__ == "__main__":
    main()
