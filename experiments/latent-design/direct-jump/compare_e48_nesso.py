"""Score the original M02 control archive and compare it with the paired latent blend."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open("x", encoding="utf8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def check_search_pair(control, blend):
    keys = ("encoded_founder", "design", "search", "perturbation_seed", "scoring_configs",
            "graph_spec", "vae_checkpoint", "checkpoint_sha256", "optimization_space",
            "confirmation_geometry_seeds")
    for key in keys:
        if control[key] != blend[key]:
            raise ValueError(f"Unpaired search setting: {key}")
    if (control.get("nesso_blend") or not blend.get("nesso_blend")
            or control["job_id"] == blend["job_id"] or control["optimization_space"] != "vae_latent"):
        raise ValueError("Expected separate Protenix control and latent-blend searches")
    limits = {"updates": 9000, "fresh_attempts": 108, "founder_fresh_attempts": 1,
              "confirmation_fresh_attempts": 8}
    for run in (control, blend):
        if any(run["budget"][k] != v for k, v in limits.items()):
            raise ValueError("Unexpected M02 search or scoring budget")


def summarize(report):
    rows = report["molecules"]
    if len({r["smiles"] for r in rows}) != len(rows) or report["founder"]["smiles"] in {r["smiles"] for r in rows}:
        raise ValueError("Novel panel contains duplicate or founder identities")
    if not (len(rows) == report["summary"]["distinct_new"] <= report["protenix_attempts_new"] <= 108):
        raise ValueError("Readout identities exceed the discovery budget")
    n = sum(r["improved_both"] is not None for r in rows)
    return dict(report["summary"], founder=report["founder"], comparable_to_founder=n,
                search_status=report["search_status"], job_id=report["search_job_id"],
                updates=report["completed_updates"], protenix_attempts_new=report["protenix_attempts_new"],
                nesso_attempts_including_founder=1+len(rows))


def compare(control, blend):
    for key in ("seed", "model_sha256", "hard_recycles", "geometry_seed", "checkpoint_sha256",
                "protenix_fresh_config_sha256"):
        if control[key] != blend[key]:
            raise ValueError(f"Incompatible hard-scoring context: {key}")
    if (control["founder"]["smiles"] != blend["founder"]["smiles"]
            or control["planned"]["updates"] != blend["planned"]["updates"]
            or control["planned"]["fresh_attempts"] != blend["planned"]["fresh_attempts"]):
        raise ValueError("Different founders or planned budgets")
    left, right = summarize(control), summarize(blend)
    if not all(0 <= arm["updates"] <= control["planned"]["updates"] for arm in (left, right)):
        raise ValueError("Recorded updates outside the planned budget")
    common_updates = left["updates"] if left["updates"] == right["updates"] else None
    differences = {}
    if common_updates is not None:
        for key in ("distinct_new", "joint_hits", "triple_hits"):
            differences[key] = right[key] - left[key] if None not in (right[key], left[key]) else None
    return dict(status="completed", common_updates=common_updates,
                planned_updates=control["planned"]["updates"], planned_discovery_scores=control["planned"]["fresh_attempts"],
                shared_novel_identities=len({r["smiles"] for r in control["molecules"]} &
                                            {r["smiles"] for r in blend["molecules"]}),
                control=left, blend=right, blend_minus_control_at_equal_updates=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "blend", "output", "plan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != args.plan_sha256:
        raise ValueError("GPU plan changed")
    plan = json.loads(plan_bytes)
    control, blend, output = (p.resolve() for p in (args.control, args.blend, args.output))
    if control == blend or any(output == r or r in output.parents for r in (control, blend)):
        raise ValueError("Comparison output must be separate from both experiments")
    control_run, blend_run = (load(r / "run.json") for r in (control, blend))
    check_search_pair(control_run, blend_run)
    config = blend_run["nesso_blend"]
    if plan["nesso_blend"] != config:
        raise ValueError("Blend settings differ from the sealed plan")
    combined = load(blend / "nesso-readout.json")
    founder = load(blend / "nesso-schema-founder.json")
    if (founder["smiles"] != control_run["encoded_founder"]["smiles"]
            or combined["founder"]["smiles"] != founder["smiles"]
            or combined["search_job_id"] != blend_run["job_id"] or combined["seed"] != config["seed"]):
        raise ValueError("Unexpected blend readout or encoded founder")
    from combined_native import NessoWorker
    from latentmol.design.protenix_nesso_readout import write_nesso_readout

    output.mkdir(parents=True)
    with NessoWorker(config["seed"], None, output / "nesso-worker", config["assets_dir"],
                     schema_founder=founder) as worker:
        identity = worker.identity
        if (identity["model_sha256"] != combined["model_sha256"]
                or identity["hard_recycles"] != combined["hard_recycles"] or identity["seed"] != config["seed"]):
            raise ValueError("Control scorer differs from the original blend's scoring protocol")
        marker = os.getenv("DBMOL_LAUNCH_READY")
        if marker:
            save(Path(marker), dict(status="models_and_inputs_ready", nonce=os.environ["DBMOL_LAUNCH_NONCE"],
                                    job_id=os.environ["LSB_JOBID"]))
        baseline = write_nesso_readout(control, worker.fresh, seed=config["seed"],
                                       model_sha256=identity["model_sha256"], output_path=output / "control-readout.json")
    report = compare(baseline, combined)
    report.update(control_path=str(control), blend_path=str(blend), evaluation_job_id=os.getenv("LSB_JOBID"),
                  scorer_audit=worker.audit)
    save(output / "comparison.json", report)


if __name__ == "__main__":
    main()
