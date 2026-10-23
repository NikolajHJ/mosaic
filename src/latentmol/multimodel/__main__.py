"""Plan and run modular frozen-model experiments; full native/HPC paths untested.

`plan` imports no native model packages. `validate` audits actual prepared assets.
`run` requires that audit; `pipeline` audits then executes exploratory trajectories.
Full runs belong on HPC; user-authorized local testing is restricted to fixtures.
"""

import argparse
import json
from pathlib import Path

import torch

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .config import Study
from .engine import Counters
from .study import load_problem, plan, require_gate, run_arm, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "validate", "run", "pipeline", "confirm", "diagnose"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", help="Default: every configured arm")
    parser.add_argument("--run-root", type=Path, default=Path("outputs/multimodel"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--output", type=Path, help="Optional plan or run-receipt JSON; never overwrite")
    parser.add_argument("--allow-experimental", action="store_true",
                        help="Acknowledge prototype integration; does not bypass derivative gates")
    args = parser.parse_args()
    if args.stage in ("confirm", "diagnose") and args.arms:
        parser.error("confirm/diagnose use every completed arm in --candidate-run; --arms does not apply")
    study = Study.load(args.config)
    arms = args.arms or list(study.arms)
    planned = (plan(study, arms) if args.stage not in ("confirm", "diagnose") else
               {"status": "hard_evaluation_planned", "config_sha256": study.fingerprint(),
                "native_evaluators": list(study.native_evaluators), "config": study.to_dict()})
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == "plan":
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, planned)
        print(json.dumps(planned, indent=2))
        return
    if args.stage in ("validate", "run", "pipeline") and not args.allow_experimental:
        parser.error("Native integrations are unvalidated; --allow-experimental acknowledges this, without bypassing gates")
    if args.stage == "run" and not args.gate_report:
        parser.error("run requires --gate-report; use pipeline to validate then run")
    if args.stage in ("confirm", "diagnose") and not args.candidate_run:
        parser.error("This stage needs --candidate-run")
    if Path(__file__).resolve().parents[3] != args.source_root.resolve():
        raise ValueError("--source-root must identify the checkout containing the executing latentmol package")
    config = {"stage": args.stage, "study": study.to_dict(), "arms": arms,
              "config_file_sha256": file_sha256(args.config),
              "candidate_run": str(args.candidate_run) if args.candidate_run else None,
              "gate_report": str(args.gate_report) if args.gate_report else None}
    with ExperimentRun(args.run_root, study.name + "-" + args.stage, config, args.source_root,
                       report_kind="molecule_optimization" if args.stage in ("run", "pipeline") else None) as run:
        run.annotate(scientific_validation="not_established", native_integration="experimental_untested_on_HPC")
        counters = Counters(lambda values: write_json(run.path / "oracle_counts.json", values))
        write_json(run.path / "plan.json", planned)
        if args.stage in ("validate", "run", "pipeline"):
            problem = load_problem(study, arms, run.record["source"]["sha256"])
            write_json(run.path / "identity.json", problem.identity)
            gate_path = run.path / "engineering_gate.json"
            if args.stage in ("validate", "pipeline"):
                validate(problem, study, arms, counters, gate_path)
            else:
                require_gate(problem, study, args.gate_report)
                write_json(gate_path, json.loads(args.gate_report.read_text()))
            if args.stage in ("run", "pipeline"):
                results = {}
                try:
                    for name in arms:
                        problem.verify()
                        results[name] = run_arm(problem, study, name, counters, run.path / "arms" / name, run.log)
                        problem.verify()
                        write_json(run.path / "comparison.json", results)
                finally:
                    problem.verify()
        elif args.stage == "confirm":
            from .confirmation import confirm
            # Record the source run's exact record before executing any external oracle.
            run.annotate(candidate_run_manifest_sha256=file_sha256(args.candidate_run / "run.json"))
            confirm(study, args.candidate_run, run.path / "native", run.log)
        else:
            from latentmol.quality import QualityScoreConfig
            from .confirmation import diagnose
            from .confirmation import generation_provenance
            run.annotate(candidate_provenance=generation_provenance(args.candidate_run, study))
            quality = QualityScoreConfig.from_dict(json.loads(args.quality_config.read_text())) if args.quality_config else QualityScoreConfig()
            run.annotate(quality_config_sha256=file_sha256(args.quality_config) if args.quality_config else None)
            diagnose(args.candidate_run, run.path / "quality.json", quality)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, {"run_dir": str(run.path.resolve()), "stage": args.stage,
                                     "scientific_validation": "not_established"})


if __name__ == "__main__":
    main()
