"""Plan and run modular frozen-model experiments; full native/HPC paths untested.

`plan` imports no native model packages. `validate` audits actual prepared assets.
`run` and `pipeline` default to a short executable-path smoke then trajectories.
`--research-profile qualified` selects the detailed native/derivative audit.
Full runs belong on HPC; user-authorized local testing is restricted to fixtures.
"""

import argparse
import json
from pathlib import Path

import torch

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.research_profile import RESEARCH_PROFILES
from .config import Study
from .engine import Counters
from .study import (GRADIENT_POLICY, LEGACY_GRADIENT_POLICY, load_problem, plan,
                    exploratory_validate, require_gate, run_arm, validate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "validate", "run", "pipeline", "confirm", "diagnose"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", help="Default: every configured arm")
    parser.add_argument("--run-root", type=Path, default=Path("outputs/multimodel"))
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--gate-report", type=Path)
    parser.add_argument("--research-profile", choices=RESEARCH_PROFILES, default="exploratory")
    parser.add_argument("--observation-interval", type=int,
                        help="Save graph/sampling snapshots every N updates (exploratory 10, qualified 1); scalars every update")
    parser.add_argument("--gradient-policy", choices=(GRADIENT_POLICY, LEGACY_GRADIENT_POLICY),
                        default=GRADIENT_POLICY,
                        help="Prospective local AD by default; legacy FD requires explicit selection")
    parser.add_argument("--candidate-run", type=Path)
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--output", type=Path, help="Optional plan or run-receipt JSON; never overwrite")
    parser.add_argument("--allow-experimental", action="store_true",
                        help="Deprecated compatibility flag; exploratory work needs no extra acknowledgment")
    args = parser.parse_args()
    observation_interval = args.observation_interval or (10 if args.research_profile == "exploratory" else 1)
    if args.observation_interval is not None and args.observation_interval < 1:
        parser.error("--observation-interval must be positive")
    if args.gradient_policy == LEGACY_GRADIENT_POLICY and args.research_profile != "qualified":
        parser.error("Legacy FD replay requires --research-profile qualified")
    if args.stage in ("confirm", "diagnose") and args.arms:
        parser.error("confirm/diagnose use every completed arm in --candidate-run; --arms does not apply")
    study = Study.load(args.config)
    arms = args.arms or list(study.arms)
    planned = (plan(study, arms) if args.stage not in ("confirm", "diagnose") else
               {"status": "hard_evaluation_planned", "config_sha256": study.fingerprint(),
                "native_evaluators": list(study.native_evaluators), "config": study.to_dict()})
    if args.stage not in ("confirm", "diagnose"):
        planned["gradient_policy"] = args.gradient_policy
        planned["research_profile"] = args.research_profile
        planned["observation_interval"] = observation_interval
        planned["native_validation"] = ("optional; exploratory executable-path smoke only"
            if args.research_profile == "exploratory" else "required for qualified profile")
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if args.stage == "plan":
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, planned)
        print(json.dumps(planned, indent=2))
        return
    if args.stage == "run" and args.research_profile == "qualified" and not args.gate_report:
        parser.error("run requires --gate-report; use pipeline to validate then run")
    if args.stage in ("confirm", "diagnose") and not args.candidate_run:
        parser.error("This stage needs --candidate-run")
    if Path(__file__).resolve().parents[3] != args.source_root.resolve():
        raise ValueError("--source-root must identify the checkout containing the executing latentmol package")
    config = {"stage": args.stage, "study": study.to_dict(), "arms": arms,
              "research_profile": args.research_profile,
              "observation_interval": observation_interval,
              "gradient_policy": args.gradient_policy,
              "config_file_sha256": file_sha256(args.config),
              "candidate_run": str(args.candidate_run) if args.candidate_run else None,
              "gate_report": str(args.gate_report) if args.gate_report else None}
    with ExperimentRun(args.run_root, study.name + "-" + args.stage, config, args.source_root,
                       report_kind="molecule_optimization" if args.stage in ("run", "pipeline") else None,
                       research_profile=args.research_profile,
                       selected_source_files=(Path(__file__), Path(__file__).with_name("study.py"),
                           Path(__file__).with_name("engine.py"), args.config)) as run:
        run.annotate(scientific_validation="not_established", native_integration="experimental_untested_on_HPC")
        counters = Counters(lambda values: write_json(run.path / "oracle_counts.json", values))
        write_json(run.path / "plan.json", planned)
        if args.stage in ("validate", "run", "pipeline"):
            problem = load_problem(study, arms, run.record["source"]["sha256"])
            write_json(run.path / "identity.json", problem.identity)
            gate_path = run.path / "engineering_gate.json"
            if args.research_profile == "exploratory":
                # A supplied detailed gate remains strict evidence. A fresh cheap
                # smoke always tests the actual loaded paths used by this pilot.
                if args.gate_report:
                    require_gate(problem, study, args.gate_report, gradient_policy=args.gradient_policy)
                exploratory_validate(problem, study, arms, counters, gate_path)
            elif args.stage in ("validate", "pipeline"):
                validate(problem, study, arms, counters, gate_path, gradient_policy=args.gradient_policy)
            else:
                require_gate(problem, study, args.gate_report, gradient_policy=args.gradient_policy)
                write_json(gate_path, json.loads(args.gate_report.read_text()))
            if args.stage in ("run", "pipeline"):
                results = {}
                try:
                    for name in arms:
                        problem.verify()
                        results[name] = run_arm(problem, study, name, counters, run.path / "arms" / name, run.log,
                                                observation_interval=observation_interval)
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
