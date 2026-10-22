"""Allocated saved-only delayed-chemistry report, with explicit pending inputs."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
import base64
import json
import os
from pathlib import Path

from render_chemistry_trajectory import saved_frames, sha


def load_study(root, config_path, color, allow_partial):
    from mosaic.small_molecule.chemistry.schedule import RepairWeightSchedule
    import numpy as np

    summary = json.loads((root / "summary.json").read_text())
    config = json.loads(config_path.read_text())
    if (
        summary["config"] != config
        or sha(config_path) != summary["config_sha256"]
        or sha(root / "configuration.json") != summary["config_sha256"]
    ):
        raise ValueError("Native/report configuration identity differs")
    for stem in ("model", "context"):
        if any(
            summary.get(stem + "_digest_" + when) != config[stem + "_digest"]
            for when in ("before", "after")
        ):
            raise ValueError("Frozen identity differs; retain failure evidence")
    case = json.loads((root / "case-0/report.json").read_text())
    if summary["status"] == "passed" and (
        not summary.get("cases")
        or any(summary["cases"][0].get(k) != v for k, v in case.items())
    ):
        raise ValueError("Native terminal summary and saved case differ")
    states, updates = case["states"], case["updates"]
    schedule = RepairWeightSchedule(**config["repair_schedule"])
    if case.get("repair_weight_schedule") != schedule.record():
        raise ValueError("Saved schedule differs")
    complete = summary["status"] == case["status"] == "passed"
    if complete and (
        len(states) != config["steps"] + 1
        or len(updates) != config["steps"]
        or case["counts"]["applied"] != config["steps"]
        or any(
            summary.get(k, {}).get("status") != "passed"
            for k in (
                "native_contact_checks",
                "coupled_chemistry_probe",
                "high_weight_chemistry_probe",
                "pilot",
                "high_weight_pilot",
            )
        )
    ):
        raise ValueError("Incomplete budget or failed native gates")
    if not complete and not allow_partial:
        raise ValueError("Partial/failed native study")
    frames = saved_frames(root / "case-0", states)
    for frame in frames:
        if frame["repair_weight"] != schedule.for_state(frame["step"]):
            raise ValueError("Saved state coefficient differs")
    for state in states:
        with np.load(root / "case-0" / state["snapshot"], allow_pickle=False) as saved:
            if float(saved["repair_weight"]) != state["repair_weight"]:
                raise ValueError("Snapshot coefficient differs")
    completed_updates = [u for u in updates if "fixed_after" in u]
    if [u["step"] for u in completed_updates] != list(range(len(completed_updates))):
        raise ValueError("Missing/reordered accepted update")
    for update in updates:
        weight = schedule.for_update(update["step"] + 1)
        if update["repair_weight"] != weight or any(
            a["repair_weight"] != weight for a in update["attempts"]
        ):
            raise ValueError("Trial coefficient changed within an update")
    return {
        "label": config["label"],
        "color": color,
        "job_id": summary["job_id"],
        "frames": frames,
        "counts": case["counts"],
        "status": summary["status"],
        "complete": complete,
        "updates": completed_updates,
        "best_observed_contact": case.get("best_observed_contact"),
        "summary_sha256": sha(root / "summary.json"),
        "config_sha256": sha(config_path),
        "schedule": schedule.record(),
        "initialization": summary["initialization"],
        "model_digest": config["model_digest"],
        "context_digest": config["context_digest"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--pending-job",
        help="Publish one completed arm; second config identifies the absent future arm",
    )
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Scientific rendering requires allocated DTU CPU")
    expected_inputs = 1 if args.pending_job else 2
    if len(args.study) != expected_inputs or len(args.config) != 2:
        raise ValueError("Both independent studies must remain represented")
    if args.pending_job and (not args.pending_job.isdecimal() or args.allow_partial):
        raise ValueError(
            "Preview needs a completed included study and native pending ID"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    studies = [
        load_study(root, config, color, args.allow_partial)
        for root, config, color in zip(args.study, args.config, ("#008c95", "#b55e38"))
    ]
    pending = []
    if args.pending_job:
        absent = json.loads(args.config[1].read_text())
        for field, source in (
            ("model_digest", "model_digest"),
            ("context_digest", "context_digest"),
            ("schedule", "repair_schedule"),
        ):
            if studies[0][field] != absent[source]:
                raise ValueError("Pending comparison identity or schedule differs")
        if args.pending_job == studies[0]["job_id"]:
            raise ValueError("Pending input duplicates the included native study")
        pending = [
            {
                "job_id": args.pending_job,
                "label": absent["label"],
                "config_sha256": sha(args.config[1]),
                "initialization": absent.get("initialization"),
                "status": "not yet included; no trajectory data loaded",
            }
        ]
    for study in studies[1:]:
        for field in ("model_digest", "context_digest", "schedule"):
            if studies[0][field] != study[field]:
                raise ValueError("Comparison identity or schedule differs")
    args.output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(4, 1, figsize=(12, 14), constrained_layout=True)
    for study in studies:
        frames = study["frames"]
        steps = [f["step"] for f in frames]
        axes[0].plot(
            steps,
            [f["contact"] for f in frames],
            color=study["color"],
            label=study["label"],
        )
        axes[1].plot(
            steps,
            [f["components"]["chemistry_loss"] for f in frames],
            color=study["color"],
            label=study["label"] + " raw repair",
        )
        axes[1].plot(
            steps,
            [f["components"]["chemistry_loss"] * f["repair_weight"] for f in frames],
            color=study["color"],
            linestyle=":",
            label=study["label"] + " weighted repair",
        )
        axes[2].plot(
            steps,
            [f["repair_weight"] for f in frames],
            color=study["color"],
            label=study["label"],
        )
        updates = study.pop("updates")
        axes[3].scatter(
            [u["step"] + 1 for u in updates],
            [u["fixed_after"] - u["fixed_before"] for u in updates],
            s=4,
            color=study["color"],
            label=study["label"],
        )
        forced = [u for u in updates if u["fallback_used"]]
        axes[3].scatter(
            [u["step"] + 1 for u in forced],
            [u["fixed_after"] - u["fixed_before"] for u in forced],
            s=10,
            marker="x",
            color="#cc2457",
            label=study["label"] + " fallback",
        )
    axes[0].set(
        title="Fixed-context Protenix contact (lower is better)", ylabel="Contact"
    )
    axes[1].set(
        title="Saved repair loss and coefficient × repair (changing targets)",
        ylabel="Repair diagnostic",
    )
    axes[2].set(
        title="Coefficient used at each saved state (next update; terminal held)",
        ylabel="Chemistry coefficient",
        yscale="log",
    )
    axes[3].set(
        title="Actual accepted minus before, same coefficient and targets within each trial set",
        xlabel="Applied graph updates",
        ylabel="Fixed-reference delta",
    )
    axes[3].axhline(0, color="#777", linewidth=0.7)
    for ax in axes:
        ax.axvline(
            studies[0]["schedule"]["hold_updates"],
            linestyle="--",
            color="#777",
            linewidth=0.8,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    figure.savefig(args.output / "loss_components.svg")
    figure.savefig(args.output / "loss_components.png", dpi=160)
    plt.close(figure)
    included_complete = all(s["complete"] for s in studies)
    complete = included_complete and not pending
    budget = studies[0]["schedule"]["total_updates"]
    data = {
        "version": "mosaic-chemistry-ramp-report-v2",
        "complete": complete,
        "included_studies_complete": included_complete,
        "pending_studies": pending,
        "title": "BRD4: delayed chemistry · 10,000 updates",
        "status_text": f"50% mix complete ({budget:,} updates) — random not yet included"
        if pending
        else f"Both {budget:,}-update studies complete"
        if complete
        else "PARTIAL / FAILED comparison; reached states only",
        "intro": (
            "Separate report for the completed 50% chain/uniform run; random will be added after terminal verification. "
            if pending
            else "Two fresh independent runs. "
        )
        + "Six occupied neutral slots · frozen BRD4/Protenix-Tiny · LR 0.03 · 10,000 requested updates each. Chemistry weight 0.001 through update 5,000, then geometric ramp to 50. Dashed line marks the ramp boundary.",
        "model_digest": studies[0]["model_digest"],
        "context_digest": studies[0]["context_digest"],
        "studies": studies,
        "historical_contact": None,
        "new_counts": [s["counts"] for s in studies],
        "best_observed_contact": [s["best_observed_contact"] for s in studies],
        "limitations": "Changing coefficients and detached repair targets make combined totals incomparable across states. Only within-update fixed-reference deltas compare one surrogate. Best contact uses the fixed contact objective. Repair zero on valid argmax graphs can remain inactive regardless of coefficient. Contact is not affinity. Raw invalid/disconnected outputs remain visible. One seed per initialization; no causal or general validity claim.",
    }
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).replace(
        "<", "\\u003c"
    )
    template = (
        (_layout_path(_repository_root(__file__) / 'scripts' / "chemistry_report_template.html"))
        .read_text(encoding="utf-8")
    )
    plot = base64.b64encode((args.output / "loss_components.svg").read_bytes()).decode(
        "ascii"
    )
    (args.output / "report.html").write_text(
        template.replace("__SAVED_TRAJECTORIES__", payload).replace(
            "__LOSS_PLOT_BASE64__", plot
        ),
        encoding="utf-8",
    )
    receipt = {
        "render_job_id": os.environ["LSB_JOBID"],
        "complete": complete,
        "included_studies_complete": included_complete,
        "pending_inputs": pending,
        "input_jobs": [s["job_id"] for s in studies],
        "states": [len(s["frames"]) for s in studies],
        "inputs": [
            {
                k: s[k]
                for k in (
                    "summary_sha256",
                    "config_sha256",
                    "schedule",
                    "initialization",
                    "counts",
                    "status",
                )
            }
            for s in studies
        ],
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(args.output.iterdir())
            if p.is_file()
        },
    }
    (args.output / "report_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
