"""Allocated CPU readiness: seal exact parent prefixes and continuation inputs."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import importlib.util
import json
import os
from pathlib import Path

from mosaic.small_molecule.gradient_resume import (
    assert_identical_contexts,
    checked_path,
    validate_parent_report,
)
from mosaic.small_molecule.io import file_sha256, load_context, write_json


PARENTS = (
    (
        "chain",
        "20261016-brd4-5000-03",
        "29736629",
        "59317f5b370079dc6da1d50ac48be77ade108e1b35eb15d662fe1146c3eb8d50",
        "c293982551840c05a0b65f23fe48d5f6eba4b32a0731e9c9a627c615e44751c4",
    ),
    (
        "uniform",
        "20261016-brd4-uniform-5000-01",
        "29736735",
        "59cf4d2431bd19d6e4f199a395951f4f74157b464317a1fb7c277514d2241091",
        "34e92550358fa047ce2116c49e3f6e9a2cb29c56fd6c613ec2a7c4af4404754e",
    ),
    (
        "noisy",
        "20261016-brd4-noisy-uniform-5000-01",
        "29736761",
        "df5826c4a474f39acc71c5873d84f2f8179a1a3f5db6093cefabf0f30620b37a",
        "710a2b737b220a8fc15c73ddcf0027a101e44dd0c96c189e777ed4b3d587bb98",
    ),
)
PILOT_SHA = "7b4b87ecea730ff9c70e6a3b0f5eec81fe4ac07245aff5026fe16399260989c4"
CANONICAL_SHA = "af58a02da9f2a9b36bf526c67df59f43841970eb34d424d85e7f5ba590c5cf12"


def authenticated(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("Input SHA-256 mismatch: " + str(path))
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Parent readiness requires allocated DTU CPU")
    release = _repository_root(__file__)
    spec = importlib.util.spec_from_file_location(
        "gradient_example", _layout_path(release / "examples/protenix_relaxed_gradients.py")
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    pilot_dir = args.root / "20261016-brd4-pilot-01/payload/study"
    pilot = authenticated(pilot_dir / "summary.json", PILOT_SHA)
    if pilot["status"] != "passed":
        raise ValueError("Pilot did not pass")
    pilot_context = load_context(pilot_dir / "context", pilot["context_files"])
    canonical_dir = args.root / "20261016-brd4-canonical-diffuse-5000-01/payload/study"
    canonical = authenticated(canonical_dir / "summary.json", CANONICAL_SHA)
    if canonical["status"] != "passed" or len(canonical["cases"][0]["states"]) != 5001:
        raise ValueError("Completed canonical study changed")
    assert_identical_contexts(
        pilot_context,
        load_context(canonical_dir / "context", canonical["context_files"]),
    )
    # Only the update/check orchestration changes. Every pre-existing model,
    # feature, projection, state, loss and context module stays byte-identical.
    baseline = authenticated(
        Path(
            "/work3/s234818/mosaic/releases/20261016-brd4-canonical-diffuse-01/source/source_manifest.json"
        ),
        "421213580f84a038cde5e47bb4048a659bfa4daee46645deccbb2e7568d7ead4",
    )
    current = json.loads((release / "source_manifest.json").read_text())
    for name, digest in baseline["files"].items():
        if (
            name.startswith("src/")
            and name != "src/mosaic/small_molecule/gradient_checks.py"
        ):
            if current["files"].get(name) != digest:
                raise ValueError("Native model/input module changed: " + name)
    plan = {
        "status": "ready",
        "readiness_job_id": os.environ["LSB_JOBID"],
        "budget": "5000 total updates per case, not 5000 additional",
        "policy": "best_primal",
        "minimum_free_gib": 20,
        "pilot_directory": str(pilot_dir),
        "pilot_summary_sha256": PILOT_SHA,
        "canonical_directory": str(canonical_dir),
        "canonical_summary_sha256": CANONICAL_SHA,
        "studies": {},
    }
    for name, run, job, summary_sha, case_sha in PARENTS:
        parent = args.root / run / "payload/study"
        summary = authenticated(parent / "summary.json", summary_sha)
        if summary["status"] != "failed" or summary["job_id"] != job:
            raise ValueError("Unexpected parent outcome/identity")
        for field in (
            "fixture_sha256",
            "hard_graph_sha256",
            "model",
            "model_digest_before",
            "context_digest_before",
            "numerical_profile",
        ):
            if summary[field] != pilot[field] or canonical[field] != pilot[field]:
                raise ValueError("Changed shared native identity: " + field)
        assert_identical_contexts(
            pilot_context, load_context(parent / "context", summary["context_files"])
        )
        example._validate_prepared_config(
            summary["config"],
            pilot["config"],
            allow_update_count_change=True,
            allow_initialization_change=True,
        )
        report_path = parent / "case-0/report.json"
        report = authenticated(report_path, case_sha)
        last = validate_parent_report(report)
        if last not in (4998, 3208, 3818) or report["requested_updates"] != 5000:
            raise ValueError("Parent endpoint or original budget changed")
        files = {"report.json": case_sha}
        case = parent / "case-0"
        for state in report["states"]:
            if state["graph"]["n_atoms"] != 6:
                raise ValueError("Atom count changed")
            files[state["snapshot"]] = file_sha256(
                checked_path(case, state["snapshot"])
            )
        for pattern in (
            "initial.npz",
            "baseline-primal.npz",
            "bond-order-null.npz",
            "X-*-plus.npz",
            "X-*-minus.npz",
            "E_upper-*-plus.npz",
            "E_upper-*-minus.npz",
        ):
            for path in case.glob(pattern):
                files[path.name] = file_sha256(path)
        if "initial.npz" not in files:
            raise ValueError("Parent fixed RNG evidence is missing")
        config = json.loads(json.dumps(summary["config"]))
        config["checks"]["acceptance_policy"] = "best_primal"
        plan["studies"][name] = {
            "parent_directory": str(parent),
            "parent_job_id": job,
            "parent_summary_sha256": summary_sha,
            "config": config,
            "cases": [{"directory": str(case), "last_step": last, "files": files}]
            + [None] * (len(example._case_specs(config)) - 1),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "plan.json", plan)
    print(
        json.dumps(
            {
                "status": "ready",
                "plan_sha256": file_sha256(args.output / "plan.json"),
                "resume_steps": {
                    name: s["cases"][0]["last_step"]
                    for name, s in plan["studies"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
