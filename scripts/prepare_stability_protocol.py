"""Freeze the source063 cohort on HPC without running models or drawing z.

Metadata-only preparation. This script is UNVALIDATED until recorded HPC gates
pass. Existing exact metadata is recovered; changed metadata is never replaced.
The controller later rechecks every file hash and the source062 prerequisites.
"""
import argparse
import json
from pathlib import Path

import yaml

from latentmol.config_identity import verify_config_identity
from latentmol.experiments import file_sha256, source_manifest, write_json

RANDOM_RUN = "20261005T225808Z_vae-random-starts-056_37735924"
COMPARISON_RUN = "20261006T133202Z_source062-step-control-comparison_767219f9"
PANEL_STARTS = {
    "gstp1-c06": "20261006T020821Z_gstp1-c06_37e69ff0",
    "ppia-c06": "20261006T023034Z_ppia-c06_dad4b049",
}
CONFIDENCE_SHA = "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1"
DECODER_SHA = "1893ddc183473fd4f27c6cd921d8b302f67ab388b660ada44bf3a2d6d5e28088"
AFFINITY_SHA = "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"


def read(path):
    return json.loads(path.read_text())


def ref(path):
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected input file: {path}")
    return {"path": str(path), "sha256": file_sha256(path)}


def save_exact(path, value):
    """Recover a previous unknown preparation outcome without overwriting it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Existing protocol/receipt differs; use a new identity: {path}")
    else:
        write_json(path, value)


def member(name, pilot, template, policies, arms):
    record = read(pilot / "run.json")
    # Failed historical experiments still supply a recorded, authenticated start;
    # they are never resumed or relabeled as successful experiments here.
    verify_config_identity(record, allow_legacy_active_set_contexts=True)
    if record["config"]["decoder"]["sha256"] != DECODER_SHA:
        raise ValueError("Historical cohort does not use the selected frozen VAE")
    if record["config"]["affinity_checkpoint"]["sha256"] != AFFINITY_SHA:
        raise ValueError("Historical cohort changes the frozen Boltz checkpoint")
    return {"id": name, "parent_record": ref(pilot / "run.json"),
            "initial_latent": ref(pilot / "start.pt"), "native_template": ref(template),
            "policies": policies, "arms": arms}


def materialize(work_root, release, output):
    """Declare all 48 cases before observing any source063 optimization outcome."""
    work_root = work_root.resolve(strict=True)
    if not str(work_root).startswith("/work3/"):
        raise ValueError("This materializer is for the DTU work3 artifacts only")
    manifest = source_manifest(release)
    if manifest["sha256"] != read(release / "release_manifest.json")["sha256"]:
        raise ValueError("Immutable source manifest differs")
    historical = yaml.safe_load((release / "configs/boltz_random_start_study.yaml").read_text())
    if historical["confidence_sha256"] != CONFIDENCE_SHA:
        raise ValueError("Historical native checkpoint identity differs")
    template = Path(historical["native_template"]["path"])
    if ref(template) != historical["native_template"]:
        raise ValueError("Historical C13 native input changed")
    policies = ["legacy", "small_fixed", "backtracking"]
    arms = ["fixed_initial_mask", "active_set"]
    starts = []
    for seed in range(10):
        directory = work_root / "runs/source056-random-start-study" / RANDOM_RUN / "seeds" / f"seed-{seed:04d}" / "pilot"
        records = list(directory.glob("*/run.json"))
        if len(records) != 1:
            raise ValueError(f"Expected one original saved prior-{seed} start, found {len(records)}")
        starts.append(member(f"prior-{seed}", records[0].parent, template,
                             policies if seed in (0, 4, 6, 8) else ["backtracking"], arms))
    starts.append(member("carbon-c13", Path(historical["reference_pilot"]), template, policies, arms))
    panel = work_root / "runs/source057-protein-panel/cases"
    for name, run_id in PANEL_STARTS.items():
        attempt = panel / name / "attempt-001"
        starts.append(member(name, attempt / "optimization" / run_id,
                             attempt / "input/carbon_chain_06.yaml", policies, ["fixed_initial_mask"]))
    comparison = work_root / "runs/source062-step-control-comparison" / COMPARISON_RUN
    rows = read(comparison / "cases.json")
    reports = [ref(Path(row["execution"]["run_paths"][0]) / "comparison.json") for row in rows]
    protocol = {"version": 1, "name": "source063-stability-campaign",
        "output_root": str(work_root / "runs/source063-stability-campaign"),
        "cache": str(work_root / "cache/boltz2"),
        "confidence_checkpoint": ref(work_root / "cache/boltz2/boltz2_conf.ckpt"),
        "step_comparison": {"run": ref(comparison / "run.json"), "cases": ref(comparison / "cases.json"), "reports": reports},
        "starts": starts, "steps": 100, "gradient_audit_interval": 50,
        "optimization_timeout_seconds": 1800, "repair_timeout_seconds": 300,
        "rescore_timeout_seconds": 3600, "allocation_seconds": 79200,
        "repair_search": {"max_edits": 3, "checks_per_depth": 256, "beam_width": 4}}
    if protocol["confidence_checkpoint"]["sha256"] != CONFIDENCE_SHA:
        raise ValueError("Native confidence checkpoint changed")
    # These are metadata contracts, not numerical model or chemistry tests.
    from latentmol.boltz.stability_study import build_config, expand_cases, load_protocol, verify_comparison
    verify_comparison(protocol["step_comparison"])
    if len(expand_cases(protocol)) != 48:
        raise ValueError("Declared campaign factor membership changed")
    for start in starts:
        build_config(protocol, start, start["policies"][0], start["arms"][0], Path(protocol["output_root"]))
    save_exact(output, protocol)
    load_protocol(output)
    if source_manifest(release)["sha256"] != manifest["sha256"]:
        raise ValueError("Source changed during metadata preparation")
    receipt = {"protocol": ref(output), "source_manifest_sha256": manifest["sha256"],
               "cases": 48, "status": "prepared_not_submitted", "numerical_validation": "not_run"}
    save_exact(output.with_suffix(".identity.json"), receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    p.add_argument("--release", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(materialize(a.work_root, a.release.resolve(strict=True), a.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
